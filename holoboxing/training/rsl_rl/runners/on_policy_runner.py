"""Competition-target training stages and checkpoint saving."""

import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from holoboxing.training.rsl_rl.algorithms.ppo import PPO, require_finite
from holoboxing.training.rsl_rl.modules.actor_critic import ActorCritic
from holoboxing.training.rsl_rl.modules.amp import AMP
from holoboxing.training.rsl_rl.modules.normalization import (
    ActorNormalizer,
    unpack_actor_observations,
)
from holoboxing.training.rsl_rl.modules.normalizer import (
    EmpiricalNormalization,
)
from holoboxing.training.rsl_rl.runners import resume as checkpoint_resume
from holoboxing.training.tasks.combat.config.sandbag import AGENTS, ROLES

CHECKPOINT_FORMAT = "holoboxing_c_joint_sandbag_public_v3"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalizers(device):
    return nn.ModuleDict(
        {
            **{f"actor_{role}": ActorNormalizer() for role in ROLES},
            **{
                f"critic_{role}": EmpiricalNormalization(
                    [225], until=100_000_000
                ).eval()
                for role in ROLES
            },
        }
    ).to(device)


@torch.no_grad()
def normalize_observations(
    modules, observations, update=False, *, update_roles=ROLES
):
    result = {}
    for role, agent in zip(ROLES, AGENTS, strict=True):
        packed = observations[agent]["task_policy"]
        actor, critic = modules[f"actor_{role}"], modules[f"critic_{role}"]
        if packed.ndim != 2 or packed.shape[-1] != actor.actor_dim:
            raise ValueError(
                "Environment observation layout differs from actor"
            )
        current, task, history = unpack_actor_observations(packed)
        raw_critic = observations[agent]["critic"]
        if update and role in update_roles:
            actor.update(current, task)
            critic.update(raw_critic)
        result[agent] = (
            actor({"self": current, "task": task, "history": history}),
            critic(raw_critic),
        )
    require_finite(result, "normalized observation")
    return result


class FrozenLowerGuard:
    def __init__(self, controllers):
        self.controllers = controllers
        self.identities = self.snapshot()

    def snapshot(self):
        from holoboxing.training.envs.onnx_lower import OnnxLower

        result = {}
        for agent, controller in self.controllers.items():
            if not isinstance(controller.module, OnnxLower):
                raise ValueError(
                    "Lower controller must use the frozen public ONNX"
                )
            result[agent] = id(controller.module.session)
        return result

    def check(self):
        if self.identities != self.snapshot():
            raise RuntimeError("Frozen lower parameters changed")


class OnPolicyRunner:
    def __init__(
        self,
        env,
        demo,
        neutral_reference,
        config,
        output,
        identity,
        *,
        policy=None,
        training_mode="paired",
        opponent_pool_capacity=8,
        opponent_refresh_updates=256,
        nfsp_eta=0.1,
        nfsp_reservoir_capacity=50_000,
        nfsp_sl_batch_size=256,
        nfsp_sl_updates=20,
    ):
        self.env, self.config = env, config
        self.device, self.output = env.device, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if env.num_envs != config.num_envs:
            raise ValueError(
                "Environment count differs from the training config"
            )
        if env.actor_dim != config.actor_dim:
            raise ValueError(
                "Environment actor width differs from training config"
            )
        seed_everything(config.seed)
        self.policy = (
            ActorCritic(neutral_reference, config)
            if policy is None
            else policy
        ).to(self.device)
        self.checkpoint_format = getattr(
            self.policy, "checkpoint_format", CHECKPOINT_FORMAT
        )
        self.normalizers = normalizers(self.device)
        if training_mode not in ("paired", "standing", "selfplay", "nfsp"):
            raise ValueError("Unknown training mode")
        if training_mode == "nfsp" and not all(
            hasattr(self.policy, name)
            for name in ("average_latent", "average_parameters")
        ):
            raise ValueError("NFSP requires a dual spherical-latent policy")
        if (training_mode == "standing") != (
            getattr(env, "standing_reference", None) is not None
        ):
            raise ValueError(
                "Standing training requires world-anchored environment targets"
            )
        self.active_roles = (
            ROLES if training_mode in ("paired", "nfsp") else ("ego",)
        )
        self.training_protocol = {"mode": training_mode}
        self.nfsp_buffers = None
        self.nfsp_optimizers = {}
        self.nfsp_optimizer_steps = dict.fromkeys(ROLES, 0)
        self.nfsp_modes = None
        self.nfsp_generator = None
        if training_mode == "nfsp":
            if not 0 < nfsp_eta < 1:
                raise ValueError("NFSP anticipatory parameter must be in (0, 1)")
            if (
                type(nfsp_reservoir_capacity) is not int
                or nfsp_reservoir_capacity < 2
                or type(nfsp_sl_batch_size) is not int
                or nfsp_sl_batch_size < 2
                or type(nfsp_sl_updates) is not int
                or nfsp_sl_updates < 1
            ):
                raise ValueError("Invalid NFSP reservoir or SL schedule")
            from holoboxing.training.nfsp import LatentReservoir

            self.training_protocol.update(
                anticipatory=nfsp_eta,
                reservoir_capacity=nfsp_reservoir_capacity,
                sl_batch_size=nfsp_sl_batch_size,
                sl_updates=nfsp_sl_updates,
            )
            self.nfsp_eta = nfsp_eta
            self.nfsp_sl_batch_size = nfsp_sl_batch_size
            self.nfsp_sl_updates = nfsp_sl_updates
            self.nfsp_buffers = {
                role: LatentReservoir(
                    nfsp_reservoir_capacity, seed=config.seed + index + 1
                )
                for index, role in enumerate(ROLES)
            }
            self.nfsp_optimizers = {
                role: torch.optim.Adam(
                    self.policy.average_parameters(role),
                    lr=config.learning_rate,
                )
                for role in ROLES
            }
            self.nfsp_generator = torch.Generator(device="cpu").manual_seed(
                (config.seed + 0x4E465350) % 2**32
            )
            self.nfsp_modes = {}
            for role in ROLES:
                self.nfsp_modes[role] = torch.zeros(
                    config.num_envs, dtype=torch.bool, device=self.device
                )
        self.opponent_pool = None
        if training_mode == "selfplay":
            from holoboxing.training.opponent_pool import OpponentPool

            self.opponent_pool = OpponentPool(
                self.policy,
                self.normalizers,
                seed=config.seed,
                capacity=opponent_pool_capacity,
                interval=opponent_refresh_updates,
            )
            self.training_protocol.update(
                pool_capacity=opponent_pool_capacity,
                refresh_updates=opponent_refresh_updates,
            )
        self.amp = AMP(demo, config, self.device)
        self.algorithm = PPO(
            self.policy,
            self.amp,
            config,
            self.device,
            active_roles=self.active_roles,
            role_masked=training_mode == "nfsp",
        )
        self.lower_guard = FrozenLowerGuard(env.controllers)
        self.identity = copy.deepcopy(identity)
        self.identity["amp_tensor_sha256"] = hashlib.sha256(
            demo.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        self.iteration, self.transitions_per_role, self.elapsed_seconds = (
            0,
            0,
            0.0,
        )
        self.observations = None
        self.pending_transition = None
        self.supervised_initialization = None
        self.invalid_transitions = 0
        self.resume_source = None

    def resume(self, path, *, allow_code_change=False):
        checkpoint_resume.resume(
            self, path, self.checkpoint_format, allow_code_change
        )

    def initialize_from_supervised(self, path):
        """Import weights into both actors for a latent policy."""
        if (
            self.iteration
            or self.algorithm.storage.step
            or self.observations is not None
        ):
            raise ValueError(
                "Supervised initialization must precede RL collection"
            )
        if self.supervised_initialization is not None:
            raise ValueError("Supervised initialization can only happen once")
        from holoboxing.training.supervised.training import (
            load_supervised,
        )
        from holoboxing.training.tasks.combat.config.sandbag import (
            sha256,
        )

        device = torch.device(self.device)
        devices = (
            [
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            ]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            actor, normalizer, payload = load_supervised(
                path,
                self.config,
                self.policy.taskpolicy_ego.neutral_reference.detach().cpu(),
            )
        if hasattr(self.policy, "policy_spec"):
            if (
                getattr(actor, "spec", {}).get("decoder_sha256")
                != self.policy.policy_spec["decoder_sha256"]
            ):
                raise ValueError("Imitation and RL frozen decoders differ")
            for role in ROLES:
                getattr(
                    self.policy, f"taskpolicy_{role}"
                ).latent_policy.load_state_dict(
                    actor.latent_policy.state_dict()
                )
                if self.nfsp_buffers is not None:
                    getattr(
                        self.policy, f"averagepolicy_{role}"
                    ).latent_policy.load_state_dict(
                        actor.latent_policy.state_dict()
                    )
                self.normalizers[f"actor_{role}"].load_state_dict(
                    normalizer.state_dict(), strict=True
                )
        else:
            self.policy.taskpolicy_ego.load_state_dict(
                actor.state_dict(), strict=True
            )
            self.normalizers["actor_ego"].load_state_dict(
                normalizer.state_dict(), strict=True
            )
        self.supervised_initialization = {
            "checkpoint": str(Path(path).resolve()),
            "sha256": sha256(path),
            "step": payload.get("step"),
            "epoch": payload.get("epoch"),
        }

    def initialize_from_prior(self, path, records):
        if (
            self.iteration
            or self.observations is not None
            or self.supervised_initialization is not None
        ):
            raise ValueError("Prior initialization requires a fresh runner")
        if not hasattr(self.policy, "policy_spec"):
            raise ValueError("Prior initialization requires a latent policy")
        from holoboxing.training.supervised.statistics import fit_statistics
        from holoboxing.training.tasks.combat.config.sandbag import sha256

        _, normalizer = fit_statistics(records)
        for role in ROLES:
            self.normalizers[f"actor_{role}"].load_state_dict(
                normalizer.state_dict(), strict=True
            )
        self.supervised_initialization = {
            "checkpoint": str(Path(path).resolve()),
            "sha256": sha256(path),
            "stage": "prior_only_no_imitation",
            "normalizer_population": records.source_selection,
        }

    def initialize_from_policy(self, path):
        """Initialize a new stage from a trained actor, not a resume."""
        if (
            self.iteration
            or self.observations is not None
            or self.supervised_initialization is not None
        ):
            raise ValueError("Policy fine-tuning requires a fresh runner")
        from holoboxing.training.rsl_rl.modules.normalization import (
            validate_statistics,
        )
        from holoboxing.training.tasks.combat.config.sandbag import sha256

        saved = torch.load(path, map_location="cpu", weights_only=False)
        allowed_formats = {self.checkpoint_format}
        if self.nfsp_buffers is not None:
            from holoboxing.training.spherical_rl import FORMAT

            allowed_formats.add(FORMAT)
        if saved.get("format") not in allowed_formats:
            raise ValueError("Policy fine-tuning format differs")
        if saved["identity"]["resources"] != self.identity["resources"]:
            raise ValueError("Policy fine-tuning resources differ")
        if saved.get("policy_spec") != getattr(
            self.policy, "policy_spec", None
        ):
            raise ValueError(
                "Policy fine-tuning frozen representation differs"
            )
        actor = {
            k.removeprefix("taskpolicy_ego."): v
            for k, v in saved["policy"].items()
            if k.startswith("taskpolicy_ego.")
        }
        average_actor = {
            k.removeprefix("averagepolicy_ego."): v
            for k, v in saved["policy"].items()
            if k.startswith("averagepolicy_ego.")
        }
        stats = {
            k.removeprefix("actor_ego."): v
            for k, v in saved["normalizers"].items()
            if k.startswith("actor_ego.")
        }
        require_finite((actor, stats), "policy stage initialization")
        checkpoint_resume.validate_module_state(
            self.policy.taskpolicy_ego, actor, "stage actor"
        )
        if self.nfsp_buffers is not None:
            checkpoint_resume.validate_module_state(
                self.policy.averagepolicy_ego,
                average_actor or actor,
                "stage average actor",
            )
        checkpoint_resume.validate_module_state(
            self.normalizers["actor_ego"], stats, "stage normalizer"
        )
        validate_statistics(stats)
        if stats["count"].item() <= 0:
            raise ValueError("Policy stage requires fitted actor statistics")
        for role in ROLES:
            getattr(self.policy, f"taskpolicy_{role}").load_state_dict(
                actor, strict=True
            )
            if self.nfsp_buffers is not None:
                getattr(
                    self.policy, f"averagepolicy_{role}"
                ).load_state_dict(average_actor or actor, strict=True)
            self.normalizers[f"actor_{role}"].load_state_dict(
                stats, strict=True
            )
        self.supervised_initialization = {
            "checkpoint": str(Path(path).resolve()),
            "sha256": sha256(path),
            "stage": "new_curriculum_stage",
            "source_iteration": saved["iteration"],
            "source_protocol": saved.get(
                "training_protocol", {"mode": "paired"}
            ),
        }

    def _resample_nfsp_modes(self, done=None):
        if self.nfsp_modes is None:
            return
        indices = (
            torch.arange(self.config.num_envs)
            if done is None
            else done.detach().cpu().nonzero().flatten()
        )
        if not len(indices):
            return
        for role in ROLES:
            sample = torch.rand(
                len(indices), generator=self.nfsp_generator
            ) < self.nfsp_eta
            if not bool(sample.any()):
                sample[
                    torch.randint(
                        len(indices), (1,), generator=self.nfsp_generator
                    )
                ] = True
            sample = sample.to(self.device)
            self.nfsp_modes[role][indices.to(self.device)] = sample

    def _reset(self):
        self.observations = self.env.reset()
        self.algorithm.reset_exploration()
        self._resample_nfsp_modes()
        raw = getattr(self.env, "env", None)
        if raw is not None and hasattr(raw, "episode_length_buf"):
            raw.episode_length_buf.random_(0, raw.max_episode_length)

    @torch.no_grad()
    def _opponent_actions(self, snapshots):
        if self.training_protocol["mode"] in ("paired", "nfsp"):
            return None
        if self.training_protocol["mode"] == "standing":
            action = torch.zeros(self.config.num_envs, 225, device=self.device)
        else:
            actor = snapshots[AGENTS[1]][0]
            action = self.policy.actor_mean("opponent", actor)
            if hasattr(self.policy, "execute"):
                action = self.policy.execute("opponent", action)
        require_finite(action, "frozen opponent action")
        return {AGENTS[1]: action}

    @torch.no_grad()
    def collect(self):
        if self.opponent_pool is not None and self.opponent_pool.prepare(
            self.iteration
        ):
            # A new opponent starts fresh physical episodes and histories.
            self.observations = None
        if self.observations is None:
            self._reset()
        algorithm, cfg = self.algorithm, self.config
        algorithm.begin_rollout()
        sums = {}
        for _ in range(cfg.num_steps_per_env):
            raw_observations = self.observations
            snapshots = normalize_observations(
                self.normalizers,
                self.observations,
                update=True,
                update_roles=self.active_roles,
            )
            actions = algorithm.act(
                snapshots,
                opponent_actions=self._opponent_actions(snapshots),
                nfsp_modes=self.nfsp_modes,
            )
            self.pending_transition = {"actions": actions}
            self.observations, task_rewards, done, infos = self.env.step(
                actions
            )
            self.pending_transition.update(
                observations=self.observations,
                task_rewards=task_rewards,
                done=done,
                infos=infos,
                reward_terms=self.env.reward_terms,
            )
            rewards = {}
            self.pending_transition["rewards"] = rewards
            self.pending_transition["style_rewards"] = {}
            invalid = infos.get("invalid_envs", torch.zeros_like(done))
            valid = ~invalid
            if self.nfsp_buffers is not None:
                from holoboxing.training.motion_prior import unit_vector

                for role, agent in zip(ROLES, AGENTS, strict=True):
                    selected = self.nfsp_modes[role] & valid
                    if bool(selected.any()):
                        action = getattr(
                            algorithm.transition, f"actions_{role}"
                        )
                        self.nfsp_buffers[role].add(
                            raw_observations[agent]["task_policy"][selected],
                            unit_vector(action[selected]),
                        )
            sums["local_recovery_count"] = sums.get(
                "local_recovery_count", 0
            ) + int(invalid.sum())

            def mean_valid(value):
                selected = value[valid]
                return (
                    selected.mean()
                    if selected.numel()
                    else value.new_zeros(())
                )

            style_by_role = {}
            for role, agent in zip(ROLES, AGENTS, strict=True):
                if role not in self.active_roles:
                    continue
                if self.nfsp_modes is not None:
                    key = f"nfsp/rl_mode_fraction/{role}"
                    sums[key] = sums.get(key, 0.0) + self.nfsp_modes[
                        role
                    ].float().mean().item()
                if bool(invalid.any()):
                    style = torch.zeros_like(task_rewards[agent])
                    style[valid] = self.amp.reward(
                        infos[f"amp_obs_{role}"][valid]
                    )
                else:
                    style = self.amp.reward(infos[f"amp_obs_{role}"])
                self.pending_transition["style_rewards"][agent] = style
                style_by_role[role] = style
            nfsp_game_reward = None
            if self.nfsp_buffers is not None:
                nfsp_game_reward = (
                    cfg.task_reward_weight
                    * (task_rewards[AGENTS[0]] - task_rewards[AGENTS[1]])
                    + cfg.style_reward_weight
                    * (style_by_role["ego"] - style_by_role["opponent"])
                )
            for role, agent in zip(ROLES, AGENTS, strict=True):
                if role not in self.active_roles:
                    continue
                style = style_by_role[role]
                if nfsp_game_reward is None:
                    reward = (
                        cfg.task_reward_weight * task_rewards[agent]
                        + cfg.style_reward_weight * style
                    )
                    timeout = infos[agent]["time_outs"]
                    terminal = self.normalizers[f"critic_{role}"](
                        infos[f"terminal_critic_{role}"]
                    )
                    reward = (
                        reward
                        + cfg.gamma
                        * algorithm.policy.value(role, terminal).squeeze(-1)
                        * timeout
                    )
                else:
                    reward = (
                        nfsp_game_reward
                        if role == "ego"
                        else -nfsp_game_reward
                    )
                rewards[agent] = reward
                values = {
                    "task_reward": mean_valid(task_rewards[agent]),
                    "style_reward": mean_valid(style),
                    "done_fraction": mean_valid(done.float()),
                    "fallen_fraction": mean_valid(
                        self.env.metrics.get(agent, {})
                        .get("fallen", torch.zeros_like(done))
                        .float()
                    ),
                }
                if nfsp_game_reward is not None:
                    values["game_reward"] = mean_valid(reward)
                values.update(
                    {
                        f"reward/{name}": mean_valid(value)
                        for name, value in self.env.reward_terms.get(
                            agent, {}
                        ).items()
                    }
                )
                for name, value in values.items():
                    key = f"{role}/{name}"
                    sums[key] = sums.get(key, 0.0) + value.item()
            algorithm.process_env_step(rewards, done, infos)
            self._resample_nfsp_modes(done)
            self.pending_transition = None
        algorithm.compute_returns(
            normalize_observations(self.normalizers, self.observations)
        )
        return {
            key: (
                value
                if key == "local_recovery_count"
                else value / cfg.num_steps_per_env
            )
            for key, value in sums.items()
        }

    def update_nfsp_averages(self):
        if self.nfsp_buffers is None:
            return {}
        from holoboxing.training.rsl_rl.modules.normalization import (
            unpack_actor_observations,
        )

        metrics = {}
        for role in ROLES:
            reservoir = self.nfsp_buffers[role]
            losses = []
            if reservoir.size >= self.nfsp_sl_batch_size:
                parameters = list(self.policy.average_parameters(role))
                optimizer = self.nfsp_optimizers[role]
                for _ in range(self.nfsp_sl_updates):
                    raw, target = reservoir.sample(
                        self.nfsp_sl_batch_size, device=self.device
                    )
                    current, task, history = unpack_actor_observations(raw)
                    observation = self.normalizers[f"actor_{role}"](
                        {"self": current, "task": task, "history": history}
                    )
                    prediction = self.policy.average_latent(role, observation)
                    loss = (prediction - target).square().mean()
                    require_finite(loss, f"{role} NFSP supervised loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    require_finite(
                        [parameter.grad for parameter in parameters],
                        f"{role} NFSP supervised gradients",
                    )
                    torch.nn.utils.clip_grad_norm_(
                        parameters,
                        self.config.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    self.nfsp_optimizer_steps[role] += 1
                    require_finite(
                        (parameters, optimizer.state),
                        f"{role} NFSP supervised optimizer",
                    )
                    losses.append(loss.detach())
            metrics[f"nfsp/average_loss/{role}"] = (
                torch.stack(losses).mean().item() if losses else None
            )
            metrics[f"nfsp/average_updates/{role}"] = len(losses)
            metrics[f"nfsp/reservoir_size/{role}"] = reservoir.size
            metrics[f"nfsp/reservoir_seen/{role}"] = reservoir.seen
            metrics[f"nfsp/sl_optimizer_steps/{role}"] = (
                self.nfsp_optimizer_steps[role]
            )
        return metrics

    def learn(
        self,
        updates,
        save_interval=100,
        extra_save_interval=None,
        *,
        time_budget=None,
    ):
        if type(updates) is not int or updates <= 0 or save_interval <= 0:
            raise ValueError("Positive update and save budgets required")
        if extra_save_interval is not None and (
            type(extra_save_interval) is not int or extra_save_interval <= 0
        ):
            raise ValueError("Extra save interval must be a positive integer")
        from holoboxing.training.checkpoint_io import training_result

        first_iteration = self.iteration
        for _ in range(updates):
            if time_budget is not None and time_budget.should_stop():
                break
            started = time.monotonic()
            try:
                metrics = self.collect()
                metrics.update(self.algorithm.update())
                metrics.update(self.update_nfsp_averages())
                self.lower_guard.check()
            except Exception as error:
                # A diagnostic snapshot is not a resumable successful checkpoint.
                torch.save(
                    {
                        "error": repr(error),
                        "iteration": self.iteration,
                        "parity": self.algorithm.last_parity,
                        "policy": self.policy.state_dict(),
                        "storage": vars(self.algorithm.storage),
                        "optimizer_steps": self.algorithm.optimizer_steps,
                        "failed_rollout_step": self.algorithm.storage.step,
                        "pending_transition": self.pending_transition,
                        "normalizers": self.normalizers.state_dict(),
                        "discriminator": self.amp.discriminator.state_dict(),
                        "amp_normalizer": self.amp.normalizer.state_dict(),
                        "environment": (
                            self.env.diagnostic_state()
                            if hasattr(self.env, "diagnostic_state")
                            else None
                        ),
                    },
                    self.output / "failure.pt",
                )
                raise
            self.iteration += 1
            self.invalid_transitions += int(metrics["local_recovery_count"])
            self.transitions_per_role += (
                self.config.num_envs * self.config.num_steps_per_env
            )
            self.elapsed_seconds += time.monotonic() - started
            if time_budget is not None:
                time_budget.update_finished(time.monotonic() - started)
            record = {
                "iteration": self.iteration,
                "transitions_per_role": self.transitions_per_role,
                "total_role_transitions": len(
                    getattr(self, "active_roles", ROLES)
                )
                * self.transitions_per_role,
                "elapsed_seconds": self.elapsed_seconds,
                "learning_rates": self.algorithm.learning_rates,
                "optimizer_steps": self.algorithm.optimizer_steps,
                "disc_optimizer_steps": self.amp.optimizer_steps,
                "probability_check": self.algorithm.last_parity,
                "actor_cuda_graphs": self.policy.actor_cuda_graph_status(),
                **metrics,
            }
            with (self.output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
            if (
                self.iteration % save_interval == 0
                or (
                    extra_save_interval is not None
                    and self.iteration % extra_save_interval == 0
                )
                or (time_budget is not None and time_budget.checkpoint_due())
            ):
                self.save(self.output / f"model_{self.iteration}.pt")
                if time_budget is not None:
                    time_budget.checkpoint_saved()
        checkpoint = self.output / f"model_{self.iteration}.pt"
        self.save(checkpoint)
        if time_budget is not None:
            time_budget.checkpoint_saved()
        return training_result(
            self.output,
            checkpoint,
            budget=time_budget,
            progress={
                "requested_updates": updates,
                "completed_updates": self.iteration - first_iteration,
                "total_updates": self.iteration,
            },
        )

    def checkpoint_state(self, *, include_payload=True):
        result = {
            "format": self.checkpoint_format,
            "training_protocol": self.training_protocol,
            "config": self.config.to_dict(),
            "identity": self.identity,
            "supervised_initialization": self.supervised_initialization,
            "policy": self.policy.state_dict(),
            "normalizers": self.normalizers.state_dict(),
            "learning_rates": self.algorithm.learning_rates,
            "optimizer_steps": self.algorithm.optimizer_steps,
            "discriminator": self.amp.discriminator.state_dict(),
            "disc_optimizer_steps": self.amp.optimizer_steps,
            "amp_normalizer": self.amp.normalizer.state_dict(),
            "iteration": self.iteration,
            "transitions_per_role": self.transitions_per_role,
            "elapsed_seconds": self.elapsed_seconds,
            **checkpoint_resume.learning_state(
                self, include_nfsp_payload=include_payload
            ),
        }
        if self.opponent_pool is not None:
            result["opponent_pool"] = self.opponent_pool.state_dict()
        if hasattr(self.policy, "policy_spec"):
            self.policy.check_frozen()
            result["policy_spec"] = self.policy.policy_spec
        return result

    def save(self, path):
        if self.algorithm.storage.step:
            raise ValueError("Save only at completed update boundaries")
        self.lower_guard.check()
        state = self.checkpoint_state()
        checkpoint_resume.validate_learning_state(self, state)
        from holoboxing.training.checkpoint_io import save_checkpoint

        save_checkpoint(path, state)
