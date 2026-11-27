"""Competition-target training stages and checkpoint saving."""

import copy
import hashlib
import json
import math
import random
import time
from contextlib import contextmanager, nullcontext
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


def load_opponent_anchor(path):
    """Read a training checkpoint's ego actor into the eval opponent slot.

    Why an anchor at all: the eval opponent is the pool's current snapshot,
    which changes every refresh, so ``eval/exchange`` against it measures how
    far the learner is ahead of an opponent that moves with it. Over 150
    updates that number covered 0.75 to 1.97 for snapshots 13 to 113 updates
    old while the learner's points against any *single* opponent rose 13% --
    the ratio's spread is the pool's, not the learner's. A pinned checkpoint
    turns ``eval/*`` into a ruler: same physics seed, same spawn, same weights
    on the other side, so successive points differ only by the learner.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(state, dict)
        or "policy" not in state
        or "normalizers" not in state
    ):
        raise ValueError(
            "Eval anchor must be a training checkpoint with policy and "
            "normalizers"
        )
    actor = {
        name.removeprefix("taskpolicy_ego."): value
        for name, value in state["policy"].items()
        if name.startswith("taskpolicy_ego.")
    }
    normalizer = {
        name.removeprefix("actor_ego."): value
        for name, value in state["normalizers"].items()
        if name.startswith("actor_ego.")
    }
    if not actor or not normalizer:
        raise ValueError("Eval anchor has no ego actor or ego statistics")
    return {
        "actor": actor,
        "normalizer": normalizer,
        "iteration": int(state.get("iteration", -1)),
        "path": str(Path(path).resolve()),
    }


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
        opponent_frozen=None,
        eval_env=None,
        eval_interval=0,
        eval_steps=480,
        eval_anchor=None,
        std_max=0.0,
        std_fixed=0.0,
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
        # Probe knob, deliberately not a config field: the resume check is
        # strict about key sets, so a new entry in ``TrainConfig`` would make
        # every checkpoint saved before it unresumable. Zero means off.
        limit = float(std_max)
        if not (0.0 <= limit < float("inf")):
            raise ValueError("std_max must be a nonnegative finite float")
        self.std_max = limit
        self.policy.std_max = limit
        fixed = float(std_fixed)
        if not (0.0 <= fixed < float("inf")):
            raise ValueError("std_fixed must be a nonnegative finite float")
        # The probe (std_max) proved the noise was the binding constraint: at
        # 0.238 the same weights scored 0.29 against a snapshot and at 0.08
        # they scored 1.00. Freezing is the part that makes the result
        # survive the flag: a clamp has zero gradient at the boundary, so the
        # entropy bonus keeps paying for a sigma the sampler cannot use, and
        # the parameter still travels home in the checkpoint at 0.238.
        self.std_fixed = fixed
        if fixed:
            self.policy.std_max = max(limit, fixed)
        self.normalizers = normalizers(self.device)
        self.eval_anchor = eval_anchor
        if eval_anchor is not None:
            # Fail at startup, not at the first eval point: a mismatched
            # anchor is a configuration error, and discovering it 25 updates
            # in costs a whole eval cycle for nothing.
            checkpoint_resume.validate_module_state(
                self.policy.taskpolicy_opponent,
                eval_anchor["actor"],
                "eval anchor actor",
            )
            checkpoint_resume.validate_module_state(
                self.normalizers["actor_opponent"],
                eval_anchor["normalizer"],
                "eval anchor statistics",
            )
        if training_mode not in ("paired", "standing", "selfplay"):
            raise ValueError("Unknown training mode")
        if (training_mode == "standing") != (
            getattr(env, "standing_reference", None) is not None
        ):
            raise ValueError(
                "Standing training requires world-anchored environment targets"
            )
        self.active_roles = ROLES if training_mode == "paired" else ("ego",)
        self.training_protocol = {"mode": training_mode}
        self.opponent_pool = None
        if training_mode == "selfplay":
            from holoboxing.training.opponent_pool import OpponentPool

            self.opponent_pool = OpponentPool(
                self.policy,
                self.normalizers,
                seed=config.seed,
                capacity=opponent_pool_capacity,
                interval=opponent_refresh_updates,
                frozen=opponent_frozen,
            )
            if opponent_frozen:
                # Deliberately omits the pool recipe: with a fixed opponent
                # there is no capacity or interval to hold constant, and
                # recording them would refuse every legitimate switch of the
                # frozen set. What is pinned is recorded next to the output.
                self.training_protocol["opponent"] = "frozen"
            else:
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
        if eval_env is not None and (
            type(eval_interval) is not int
            or eval_interval < 1
            or type(eval_steps) is not int
            or eval_steps < 1
        ):
            raise ValueError("Deterministic eval needs a positive cadence")
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.eval_steps = eval_steps
        self.eval_disabled = False
        self.eval_calls = 0
        self.eval_error = None
        # One readout on the first update, before the cadence can line up.
        # A cadence that never fires (or a readout that throws on its very
        # first call) is otherwise invisible for a whole run: ``eval_env.json``
        # still says available, and metrics.jsonl simply has no eval/* columns.
        self._eval_warmup = eval_env is not None
        self._eval_rng = None

    def resume(self, path, *, allow_code_change=False):
        checkpoint_resume.resume(
            self, path, self.checkpoint_format, allow_code_change
        )
        # The checkpoint carries its own sigma, so a pinned run has to write
        # the pin back after the weights land.
        self.pin_std()

    def pin_std(self):
        """Hold sigma at ``std_fixed`` instead of hoping entropy lets it fall.

        A ceiling alone is not a fix. The clamp has zero gradient at the
        boundary, so ``std_*`` never learns that it is being clipped; the
        entropy bonus (0.0004 a nat) is a standing bid for a larger sigma; and
        the parameter is saved at its unclamped value, so a later run without
        the flag restores the noise that made 100 updates look stationary. At
        0.238 per latent axis the noise norm is 1.35 against a unit signal --
        the learner fights at ~50 degrees off its own greedy direction, which
        is why the same weights read 0.29 exchange under sampling and 1.18
        under ``actor_mean``.

        Two parts, both needed. The value is written in (so checkpoints
        describe the law they were trained under); a backward hook then zeroes
        its gradient, so no optimizer step can move it again. The parameter
        keeps ``requires_grad`` and stays in the optimizer group with real
        Adam moments -- the strict resume check counts parameters and demands a
        moment for each, and freezing the tensor instead would break every
        later resume. Dividing by ``physical_scale`` keeps the general
        joint-action actor correct; it is ones for the 32-D latent.
        """
        if not self.std_fixed:
            return
        for role in self.active_roles:
            parameter = getattr(self.policy, f"std_{role}", None)
            if parameter is None:
                continue
            scale = getattr(
                self.policy, f"taskpolicy_{role}"
            ).physical_scale.detach()
            with torch.no_grad():
                parameter.copy_(
                    (
                        torch.full_like(parameter, self.std_fixed) / scale
                    ).log()
                )
            if not getattr(parameter, "_std_pinned", False):
                parameter.register_hook(lambda grad: torch.zeros_like(grad))
                parameter._std_pinned = True

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
        if saved.get("format") != self.checkpoint_format:
            raise ValueError("Policy fine-tuning format differs")
        if saved["identity"]["resources"] != self.identity["resources"]:
            raise ValueError("Policy fine-tuning resources differ")
        if saved.get("policy_spec") != getattr(
            self.policy, "policy_spec", None
        ):
            raise ValueError(
                "Policy fine-tuning frozen representation differs"
            )
        self.policy.validate_action_mapping(saved["policy"])
        actor = {
            k.removeprefix("taskpolicy_ego."): v
            for k, v in saved["policy"].items()
            if k.startswith("taskpolicy_ego.")
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

    def _reset(self):
        self.observations = self.env.reset()
        self.algorithm.reset_exploration()
        raw = getattr(self.env, "env", None)
        if raw is not None and hasattr(raw, "episode_length_buf"):
            raw.episode_length_buf.random_(0, raw.max_episode_length)

    @torch.no_grad()
    def _opponent_actions(self, snapshots):
        if self.training_protocol["mode"] == "paired":
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
        # An evaluation rebinds Warp's process-global device; training must own
        # it again before a single physics step runs here.
        select = getattr(self.env, "select_warp", None)
        if select is not None:
            select()
        if self.opponent_pool is not None:
            # Swapping the frozen weights must not wipe the world. A global
            # reset here truncated every episode at the refresh interval --
            # sixteen updates, 7.68 s of a 60 s round -- so the closing 87% of
            # a round, including the three-second KO confirmation, never
            # entered the training distribution, and each block spent its
            # first two updates walking in from the 1.98 m spawn. The opponent
            # changes; the fight continues from wherever it was.
            self.opponent_pool.prepare(self.iteration)
        if self.observations is None:
            self._reset()
        # Annealed reward weights are keyed off this, so a resumed run inherits
        # the same schedule point instead of restarting the walk-in shaping.
        self.env.shaping_iteration = self.iteration
        algorithm, cfg = self.algorithm, self.config
        algorithm.begin_rollout()
        sums = {}
        for _ in range(cfg.num_steps_per_env):
            snapshots = normalize_observations(
                self.normalizers,
                self.observations,
                update=True,
                update_roles=self.active_roles,
            )
            actions = algorithm.act(
                snapshots, opponent_actions=self._opponent_actions(snapshots)
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
            sums["local_recovery_count"] = sums.get(
                "local_recovery_count", 0
            ) + int(invalid.sum())

            def mean_valid(value):
                # Logging must never be able to kill a run. A metric that
                # arrives as [worlds, k] (per hand, per term, ...) is folded to
                # [worlds] instead of raising on the boolean mask, and a
                # metric whose batch axis does not even match is reported as
                # zero rather than taking the whole training down.
                if value.dim() > 1:
                    value = value.reshape(value.shape[0], -1).mean(-1)
                if value.shape[0] != valid.shape[0]:
                    return value.new_zeros(())
                selected = value[valid]
                return (
                    selected.mean()
                    if selected.numel()
                    else value.new_zeros(())
                )

            def max_valid(value):
                # The mean of a per-step impulse over 2048 worlds hides the one
                # world that actually punched: a single 5 N*s event is 0.002 of
                # the mean. The max is the only number that says whether the
                # policy has ever cleared the referee's threshold.
                if value.dim() > 1:
                    value = value.reshape(value.shape[0], -1).mean(-1)
                if value.shape[0] != valid.shape[0]:
                    return value.new_zeros(())
                selected = value[valid]
                return selected.max() if selected.numel() else value.new_zeros(())

            def min_valid(value):
                # Same masking, opposite extreme: the closest one glove in
                # 2048 worlds reached, i.e. whether the target is reachable at
                # all rather than merely approached.
                if value.dim() > 1:
                    value = value.reshape(value.shape[0], -1).mean(-1)
                if value.shape[0] != valid.shape[0]:
                    return value.new_zeros(())
                selected = value[valid]
                return selected.min() if selected.numel() else value.new_zeros(())

            for role, agent in zip(ROLES, AGENTS, strict=True):
                if role not in self.active_roles:
                    continue
                if bool(invalid.any()):
                    style = torch.zeros_like(task_rewards[agent])
                    style[valid] = self.amp.reward(
                        infos[f"amp_obs_{role}"][valid]
                    )
                else:
                    style = self.amp.reward(infos[f"amp_obs_{role}"])
                self.pending_transition["style_rewards"][agent] = style
                reward = (
                    cfg.task_reward_weight * task_rewards[agent]
                    + cfg.style_reward_weight * style
                )
                timeout = infos[agent]["time_outs"]
                # Bootstrap actual executed terminal state, never reset observation.
                terminal = self.normalizers[f"critic_{role}"](
                    infos[f"terminal_critic_{role}"]
                )
                reward = (
                    reward
                    + cfg.gamma
                    * algorithm.policy.value(role, terminal).squeeze(-1)
                    * timeout
                )
                metrics = self.env.metrics.get(agent, {})
                empty = torch.zeros_like(done, dtype=torch.float32)
                rewards[agent] = reward
                values = {
                    "task_reward": mean_valid(task_rewards[agent]),
                    "style_reward": mean_valid(style),
                    "done_fraction": mean_valid(done.float()),
                    "fallen_fraction": mean_valid(
                        metrics.get("fallen", empty).float()
                    ),
                    # Official score kept per episode, so a run that trades
                    # points for contact or falls shows up immediately.
                    "score": mean_valid(metrics.get("score", empty)),
                    "conceded": mean_valid(metrics.get("conceded", empty)),
                    # Boxing range diagnosis: gap in metres, best hit in N*s,
                    # fastest fist in m/s. 5 N*s is one point, so hit_max_ns
                    # above 1.0 means the arm has started to load at all.
                    "gap_m": mean_valid(metrics.get("gap", empty)),
                    "hit_ns": mean_valid(metrics.get("impulse", empty)),
                    "hit_max_ns": max_valid(metrics.get("impulse", empty)),
                    "punch_mps": mean_valid(metrics.get("punch_speed", empty)),
                    "punch_max_mps": max_valid(
                        metrics.get("punch_speed", empty)
                    ),
                    # Glove-to-torso distance and contact rate. gap_m says the
                    # chests are 0.81 m apart; reach_m says whether the fist
                    # has ever crossed the 0.28 m of air in front of the glove
                    # that separates a jab from a shadowbox.
                    "reach_m": mean_valid(metrics.get("reach", empty)),
                    "reach_min_m": min_valid(metrics.get("reach", empty)),
                    "contact_frac": mean_valid(metrics.get("contact", empty)),
                    # Points per 60 s round. Three thousand control steps at
                    # 20 ms, and the referee pays one point per 5 N*s, so this
                    # is the only column that says "continuous scoring" without
                    # needing to reason about where in the episode we sampled.
                    "pts_round": 3000.0
                    * mean_valid(metrics.get("point_rate", empty)),
                    # Same construction for the other side, and the two
                    # columns only mean something together. ``conceded`` is
                    # the opponent's cumulative score, which resets inside the
                    # episode; 1754-2003 reported 1.28 there while the
                    # ``taken_point`` term was charging for 15.1 points a
                    # round. Read the pair, never either one alone.
                    "conceded_round": 3000.0
                    * mean_valid(metrics.get("conceded_rate", empty)),
                    # Which snapshot is on the other side. Both 1754-2003 and
                    # 1851-1942 produced a round that scored 2x and then
                    # collapsed, and in both the opponent's points per round
                    # fell by half in the same window -- consistent with the
                    # pool's retained initial opponent being drawn, never
                    # confirmed. This column settles it: a spike whose
                    # ``opponent_iteration`` is far behind the current one is
                    # a weak opponent, not progress.
                    # Tensor, not a plain float: every value in this dict is
                    # accumulated through ``value.item()`` below, and a Python
                    # float has no .item().
                    "opponent_iteration": torch.tensor(
                        float(
                            self.opponent_pool.selected_iteration
                            if self.opponent_pool is not None
                            else -1
                        )
                    ),
                    # Mean impulse of the punches that actually touched, in
                    # N*s. The unconditional mean is useless here -- 95% of
                    # world-steps are air, so it reads 0.05 no matter how hard
                    # the hitting hand is -- and the conditional mean is the
                    # one number that says whether a landed punch is worth 1
                    # N*s (a graze) or 5 (a point). Accumulated as a ratio so
                    # the per-step division below cancels out.
                    "impact_num": (
                        metrics.get("impulse", empty)
                        * (metrics.get("impulse", empty) > 0.5).float()
                    ).sum(),
                    "impact_den": (metrics.get("impulse", empty) > 0.5)
                    .float()
                    .sum(),
                    # Same conditional mean, split by hand, plus the number of
                    # points each hand actually scored. ``impact_ns`` is the
                    # max over the pair, so it reports the good hand and hides
                    # a dormant one; ``score_l``/``score_r`` are how many
                    # points per round each glove earns, and their ratio is the
                    # direct answer to "is this fighter one-handed".
                    "impact_l_num": (
                        metrics.get("impulse_l", empty)
                        * (metrics.get("impulse_l", empty) > 0.5).float()
                    ).sum(),
                    "impact_l_den": (metrics.get("impulse_l", empty) > 0.5)
                    .float()
                    .sum(),
                    "impact_r_num": (
                        metrics.get("impulse_r", empty)
                        * (metrics.get("impulse_r", empty) > 0.5).float()
                    ).sum(),
                    "impact_r_den": (metrics.get("impulse_r", empty) > 0.5)
                    .float()
                    .sum(),
                    "score_l": 3000.0
                    * mean_valid(metrics.get("scored_l", empty)),
                    "score_r": 3000.0
                    * mean_valid(metrics.get("scored_r", empty)),
                    # How hard the punches that actually scored were. The
                    # conditional mean above averages a handful of 5-9 N*s
                    # events together with a hundred 0.5-2 N*s taps, so it
                    # sits near 1.8 no matter how the scoring punches behave:
                    # it says "a typical contact is a tap", not "the point is
                    # marginal". This one is conditioned on the gate firing,
                    # so 5.1 means every point is squeezed in at the threshold
                    # (and a little more arm is worth a lot) while 7+ means the
                    # gate is already punched through and the bottleneck is
                    # frequency or position, not force.
                    "point_impulse_num": (
                        metrics.get("impulse", empty)
                        * (metrics.get("point_rate", empty) > 0).float()
                    ).sum(),
                    "point_impulse_den": (
                        metrics.get("point_rate", empty) > 0
                    )
                    .float()
                    .sum(),
                }
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
                    # Everything here is normally a tensor, but a scalar
                    # metric (a pool label, a flag) may arrive as a float.
                    # Coerce instead of assuming .item(), so adding such a
                    # column later cannot abort training at step 0.
                    sums[key] = sums.get(key, 0.0) + (
                        value.item() if torch.is_tensor(value) else float(value)
                    )
            algorithm.process_env_step(rewards, done, infos)
            self.pending_transition = None
        algorithm.compute_returns(
            normalize_observations(self.normalizers, self.observations)
        )
        scaled = {
            key: (
                value
                if key == "local_recovery_count"
                else value / cfg.num_steps_per_env
            )
            for key, value in sums.items()
        }
        for role in self.active_roles:
            numerator = scaled.pop(f"{role}/impact_num", 0.0)
            denominator = scaled.pop(f"{role}/impact_den", 0.0)
            scaled[f"{role}/impact_ns"] = (
                numerator / denominator if denominator > 1e-9 else 0.0
            )
            # Per-hand conditional means. A one-handed fighter shows ~equal
            # values here (the idle glove's few contacts are still contacts)
            # but a lopsided ``score_l``/``score_r``, which is the sharper
            # signal: the referee pays 5 N*s, and only one hand ever clears it.
            for side in ("l", "r"):
                hand_num = scaled.pop(f"{role}/impact_{side}_num", 0.0)
                hand_den = scaled.pop(f"{role}/impact_{side}_den", 0.0)
                scaled[f"{role}/impact_{side}_ns"] = (
                    hand_num / hand_den if hand_den > 1e-9 else 0.0
                )
            # Same conditioning, but only for the steps the referee paid on.
            point_num = scaled.pop(f"{role}/point_impulse_num", 0.0)
            point_den = scaled.pop(f"{role}/point_impulse_den", 0.0)
            scaled[f"{role}/point_impulse_ns"] = (
                point_num / point_den if point_den > 1e-9 else 0.0
            )
            # Fraction of world-steps with a real contact (>0.5 N*s). Paired
            # with pts_round it gives the hit rate directly: one point per
            # how many touches.
            scaled[f"{role}/touch_frac"] = denominator / max(
                1, self.env.num_envs
            )
            # The two columns the reward table is priced on, in one place.
            # ``exchange`` is points scored per point conceded; the table pays
            # 190*0.02 = 3.8 for a point and charges 140*0.02 = 2.8 for one
            # conceded, so break-even sits at 1:1.357, i.e. ``exchange`` =
            # 0.737. Above it the fighter profits from trading, below it every
            # extra contact is a loss. Keep this in step with the
            # ``taken_point`` weight in ``envs/mujoco_rewards.py``: the -120
            # table read 0.63 here, the -80 table 0.421.
            points_for = scaled.get(f"{role}/pts_round", 0.0)
            points_against = scaled.get(f"{role}/conceded_round", 0.0)
            scaled[f"{role}/net_round"] = points_for - points_against
            scaled[f"{role}/exchange"] = (
                points_for / points_against if points_against > 1e-9 else 0.0
            )
        return scaled

    @contextmanager
    def _anchor_scope(self, anchor):
        """Put the pinned opponent in place for one readout, then put it back.

        Restoring on the way out is not tidiness: ``evaluate`` runs inside a
        handler that disables eval on any failure, so an exception here must
        not leave the anchor installed -- the next ``collect`` would then train
        against it until the pool's next refresh, and nothing in the log would
        say so.
        """
        actor, stats = (
            self.policy.taskpolicy_opponent,
            self.normalizers["actor_opponent"],
        )
        saved_actor = {
            name: value.detach().clone()
            for name, value in actor.state_dict().items()
        }
        saved_stats = {
            name: value.detach().clone()
            for name, value in stats.state_dict().items()
        }
        actor.load_state_dict(anchor["actor"], strict=True)
        stats.load_state_dict(anchor["normalizer"], strict=True)
        try:
            yield
        finally:
            actor.load_state_dict(saved_actor, strict=True)
            stats.load_state_dict(saved_stats, strict=True)

    @torch.no_grad()
    def evaluate(self):
        """Greedy readout on a separate simulator, logged as ``eval/*``.

        The training columns are not a skill estimate. ``collect`` drives the
        learner through ``distribution().sample()`` while ``_opponent_actions``
        drives the frozen opponent through ``actor_mean``. The learner's noise
        is not small: ``std_ego`` is a 32-D parameter on a unit-norm latent,
        measured at 0.238 per dimension in the v8 checkpoint (0.191-0.282,
        grown 2.4x from its 0.1 init), and the rho=0.9 conditional law scales
        each step by 0.436, so the executed action still carries ~0.10 of
        fresh noise on every one of 32 latent axes -- several tens of degrees
        off the greedy direction. Every exchange ratio in ``metrics.jsonl`` is
        therefore measured with the learner fighting blindfolded against a
        clean opponent, which biases the number down by an unknown and
        drifting amount -- exactly the asymmetry that made a snapshot 39
        updates old look like a 4.5:1 loss.

        This runs the same fight with both sides greedy (``actor_mean`` of the
        base policy, exploration path bypassed entirely) on ``eval_env``, which
        owns its own physics, referees and episodes. It cannot truncate or
        perturb a training episode, which is why it is a second simulator and
        not a pause inside ``collect``. That simulator may sit on another GPU
        (``eval_env.json`` records which); the policy stays where the
        optimizers are, so observations and actions cross the bus once per
        step, which is ~1.1 MB against 4 physics substeps.

        The eval generator is rewound to the state captured on the first call,
        so successive evaluations see identical spawn poses, identical shove
        schedule and identical observation noise: a paired comparison, not two
        independent samples. Worlds still diverge once a fighter falls (the
        reset draws fresh noise), so ``eval/resets`` is logged next to the
        result.
        """
        env, started = self.eval_env, time.monotonic()
        worlds = env.num_envs
        # Fixed yardstick. The pool keeps its own opponent installed in the
        # opponent slots, so it has to be put back afterwards -- otherwise the
        # next collect() would train against the anchor until the pool's next
        # refresh, which is a silent change of training condition.
        scope = (
            self._anchor_scope(self.eval_anchor)
            if self.eval_anchor is not None
            else nullcontext()
        )
        # The optimizers, the normalizers and the actor all live together.
        trainer_device = next(self.policy.parameters()).device

        def on_trainer(snapshot):
            """Evaluator device -> actor device; a no-op on a shared GPU.

            Blocking on purpose: a cross-GPU peer copy is enqueued on one
            device's stream and consumed on the other's, and this loop is a
            pause in training, so 480 synchronous 1 MB transfers cost less
            than reasoning about that ordering.
            """
            if snapshot[AGENTS[0]]["task_policy"].device == trainer_device:
                return snapshot
            return {
                agent: {
                    name: value.to(trainer_device)
                    for name, value in parts.items()
                }
                for agent, parts in snapshot.items()
            }

        with env.warp_scope(), scope:
            if self._eval_rng is None:
                self._eval_rng = (
                    env.generator.get_state().cpu().clone(),
                    env.actor_noise.generator.get_state().cpu().clone(),
                )
            env.generator.set_state(self._eval_rng[0])
            env.actor_noise.generator.set_state(self._eval_rng[1])
            # Annealed weights and the reward table read this; the eval ignores
            # the rewards, but ``terms`` must still build a valid table.
            env.shaping_iteration = self.iteration
            observations = env.reset()
            totals = {
                "valid": 0.0,
                "resets": 0.0,
                "points": 0.0,
                "conceded": 0.0,
                "left": 0.0,
                "right": 0.0,
                "impulse": 0.0,
                "impulse_n": 0.0,
            }
            for _ in range(self.eval_steps):
                snapshots = normalize_observations(
                    self.normalizers, on_trainer(observations)
                )
                actions = {}
                for role, agent in zip(ROLES, AGENTS, strict=True):
                    value = self.policy.actor_mean(role, snapshots[agent][0])
                    if hasattr(self.policy, "execute"):
                        value = self.policy.execute(role, value)
                    actions[agent] = value.to(env.device)
                observations, _, done, infos = env.step(actions)
                valid = ~infos.get(
                    "invalid_envs",
                    torch.zeros(worlds, dtype=torch.bool, device=env.device),
                )
                metrics = env.metrics[AGENTS[0]]
                totals["valid"] += float(valid.sum())
                totals["resets"] += float(done.sum())
                for key, name in (
                    ("points", "point_rate"),
                    ("conceded", "conceded_rate"),
                    ("left", "scored_l"),
                    ("right", "scored_r"),
                ):
                    totals[key] += float(metrics[name][valid].sum())
                impulse = metrics["impulse"][valid]
                hit = impulse > 0.5
                totals["impulse"] += float((impulse * hit).sum())
                totals["impulse_n"] += float(hit.sum())
        count = max(totals["valid"], 1.0)
        points = 3000.0 * totals["points"] / count
        conceded = 3000.0 * totals["conceded"] / count
        touches = totals["impulse_n"] / count
        result = {
            "eval/points_round": points,
            "eval/conceded_round": conceded,
            "eval/exchange": points / conceded if conceded > 1e-9 else 0.0,
            # Conditional mean over touches only, same construction as
            # ``ego/impact_ns``: the unconditioned mean is 95% air.
            "eval/impact_ns": (
                totals["impulse"] / totals["impulse_n"]
                if totals["impulse_n"] > 1e-9
                else 0.0
            ),
            "eval/touch_frac": touches,
            "eval/conv": (
                touches * 3000.0 / points if points > 1e-9 else 0.0
            ),
            "eval/score_l": 3000.0 * totals["left"] / count,
            "eval/score_r": 3000.0 * totals["right"] / count,
            "eval/opponent_iteration": float(
                self.eval_anchor["iteration"]
                if self.eval_anchor is not None
                else (
                    self.opponent_pool.selected_iteration
                    if self.opponent_pool is not None
                    else -1
                )
            ),
            "eval/anchor": 1.0 if self.eval_anchor is not None else 0.0,
            "eval/resets": totals["resets"],
            "eval/steps": float(self.eval_steps),
            "eval/seconds": time.monotonic() - started,
        }
        return result

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
                # A guard, not the mechanism: the backward hook installed by
                # pin_std already zeroes sigma's gradient, so this only
                # re-asserts the value (and would catch a parameter that was
                # rebuilt behind our back).
                self.pin_std()
                # Deterministic readout. Deliberately inside its own handler:
                # an eval failure (OOM, a bad world, a shape surprise) must log
                # one line and switch itself off, never abort a run that is
                # otherwise healthy -- the same rule the metric collection
                # follows.
                if self.eval_env is not None and not self.eval_disabled:
                    due = (self.iteration + 1) % self.eval_interval == 0
                    if due or self._eval_warmup:
                        try:
                            metrics.update(self.evaluate())
                            self.eval_calls += 1
                        except Exception as error:  # noqa: BLE001
                            self.eval_disabled = True
                            self.eval_error = repr(error)
                            payload = {
                                "iteration": self.iteration,
                                "eval_disabled": repr(error),
                                "advice": (
                                    "eval/* is now off for the rest of this run; "
                                    "the training columns below are unaffected"
                                ),
                            }
                            (self.output / "eval_error.json").write_text(
                                json.dumps(payload, indent=2), encoding="utf-8"
                            )
                            print(json.dumps(payload), flush=True)
                        finally:
                            self._eval_warmup = False
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
            # Sigma itself, not just its entropy. The two are easy to confuse:
            # ``ego/entropy`` is summed over 32 latent axes and then shrunk by
            # the rho conditional (x0.436 at rho=0.9), so reading a sigma off
            # it silently assumes a dimension count and a stream mode. These
            # columns are read from the same place the sampler reads them, so
            # ``std_mean`` is the executed value and ``std_step`` is the
            # per-step innovation after the rho shrink.
            sigma = None
            if hasattr(self.policy, "action_sigma"):
                with torch.no_grad():
                    sigma = self.policy.action_sigma("ego").detach().reshape(-1)
            if sigma is not None:
                metrics["policy/std_mean"] = float(sigma.mean())
                metrics["policy/std_hi"] = float(sigma.max())
                metrics["policy/std_step"] = float(
                    sigma.mean()
                    * math.sqrt(1.0 - self.config.exploration_rho**2)
                )
            metrics["policy/std_limit"] = self.std_max
            metrics["policy/std_fixed"] = self.std_fixed
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
        if self.eval_env is not None and self.eval_calls == 0:
            # Every run that ends here produced a metrics.jsonl with no eval/*
            # columns, which reads exactly like "eval was never configured".
            payload = {
                "iteration": self.iteration,
                "eval_never_ran": True,
                "eval_interval": self.eval_interval,
                "reason": self.eval_error
                or "cadence never matched an update in this run",
            }
            (self.output / "eval_error.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            print(json.dumps(payload), flush=True)
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

    def checkpoint_state(self):
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
            **checkpoint_resume.learning_state(self),
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
