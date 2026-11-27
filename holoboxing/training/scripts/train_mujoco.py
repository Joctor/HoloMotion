"""Launch the existing PPO algorithm on competition XML physics."""

import json
from dataclasses import replace

from holoboxing.data.contract import CompetitionAssets
from holoboxing.training.envs.mujoco_env import MujocoSandbagEnv
from holoboxing.training.resources import (
    load_neutral_reference,
    resolve_resources,
)
from holoboxing.training.rsl_rl.modules.amp import load_amp_motions
from holoboxing.training.rsl_rl.runners.on_policy_runner import (
    OnPolicyRunner,
    load_opponent_anchor,
    seed_everything,
)
from holoboxing.training.scripts.common import prepare, write_json


def eval_device_for(args):
    """Where the greedy evaluator's second simulator lives.

    It must not share a GPU with the trainer unless there is only one. Two
    resident simulator worlds double the persistent buffers, and mujoco_warp
    allocates its per-step collision scratch on the *active* device -- the
    ``epa_pr`` table sized by ``naccdmax`` is 1.1 GB at 2048 worlds -- so the
    trainer used to die with an out-of-memory that looked like its own
    footprint. A second GPU is preferred automatically; when there is only
    one, the evaluator shares it and ``eval_env.json`` records that, so a
    failure there is explainable instead of mysterious.
    """
    requested = str(getattr(args, "eval_device", "") or "")
    if requested:
        return requested
    import torch

    training = str(args.device)
    if not training.startswith("cuda"):
        return training
    if torch.cuda.device_count() < 2:
        return training
    index = torch.device(training).index or 0
    return "cuda:0" if index else "cuda:1"



def train(args, config, *, time_budget=None):
    seed_everything(config.seed)
    identity = prepare(args)
    assets = CompetitionAssets.load(args.assets)
    _, paths, _ = resolve_resources(args.assets, args.lower)
    reference = load_neutral_reference(
        paths["lower_onnx"], paths["arena_scene"]
    )
    policy, prior_records = None, None
    import torch

    from holoboxing.training.motion_prior import FORMAT as PRIOR_FORMAT
    from holoboxing.training.spherical_actor import (
        FORMAT as IMITATION_FORMAT,
        actor_from_prior,
        actor_from_state,
    )
    from holoboxing.training.spherical_rl import (
        FORMAT as LATENT_RL_FORMAT,
        SphericalActorCritic,
    )
    from holoboxing.training.supervised.training import load_supervised

    initialization = (
        args.resume
        or args.supervised_checkpoint
        or args.prior_checkpoint
        or args.policy_checkpoint
    )
    saved = torch.load(initialization, map_location="cpu", weights_only=False)
    if saved.get("format") in (
        PRIOR_FORMAT,
        IMITATION_FORMAT,
        LATENT_RL_FORMAT,
    ):
        if not args.config:
            config = replace(config, horizon_rho=0, init_noise_std=0.1)
        if saved["format"] == PRIOR_FORMAT:
            from holoboxing.training.supervised.data import Records

            prior_records = Records(args.amp_motions / "manifest.json")
            validation = Records(
                args.amp_motions / "manifest.json",
                split="validation",
                allow_empty=True,
            )
            actor = actor_from_prior(
                initialization, config, prior_records, validation
            )
        elif saved["format"] == IMITATION_FORMAT:
            actor, _, _ = load_supervised(initialization, config, reference)
        else:
            actor_state = {
                k.removeprefix("taskpolicy_ego."): v
                for k, v in saved["policy"].items()
                if k.startswith("taskpolicy_ego.")
            }
            actor = actor_from_state(actor_state, saved["policy_spec"], config)
        policy = SphericalActorCritic(actor, config)
        identity["frozen_representation"] = policy.policy_spec
    write_json(args.output / "config.json", config.to_dict())
    standing = None
    if args.training_mode == "standing":
        from holoboxing.training.standing_reference import (
            WorldStandingReference,
        )

        standing = WorldStandingReference(
            reference, config.num_envs, args.device
        )
    env = MujocoSandbagEnv(
        assets,
        paths,
        config.num_envs,
        args.device,
        args.output / ".runtime",
        config.seed,
        standing_reference=standing,
        balance_exponent=float(getattr(args, "balance_exponent", 2.0) or 2.0),
    )
    # A second, smaller simulator for the greedy readout. It owns its physics
    # buffers, referees and episodes, so evaluating can never truncate or
    # perturb a training episode -- which is the whole reason it is not a pause
    # inside collect(). It is placed on another GPU when there is one, because
    # sharing a device with the trainer is what exhausts it (see
    # ``eval_device_for``). Built lazily and tolerated: if it does not fit in
    # GPU memory the run still trains, it just keeps the biased training
    # columns.
    eval_env = None
    if getattr(args, "eval_interval", 0) and args.training_mode == "selfplay":
        eval_device = eval_device_for(args)
        try:
            eval_env = MujocoSandbagEnv(
                assets,
                paths,
                args.eval_worlds,
                eval_device,
                args.output / ".runtime_eval",
                config.seed + 1,
                balance_exponent=float(
                    getattr(args, "balance_exponent", 2.0) or 2.0
                ),
            )
            write_json(
                args.output / "eval_env.json",
                {
                    "available": True,
                    "device": str(eval_env.device),
                    "worlds": args.eval_worlds,
                    "training_device": str(env.device),
                    "shared_device": str(eval_env.device) == str(env.device),
                },
            )
            # One line, because "which GPU did the evaluator land on" is the
            # first question a memory failure asks and the log is the only
            # thing that survives the run.
            print(
                json.dumps(
                    {
                        "eval_device": str(eval_env.device),
                        "training_device": str(env.device),
                        "eval_worlds": args.eval_worlds,
                    }
                ),
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            write_json(
                args.output / "eval_env.json",
                {
                    "available": False,
                    "device": eval_device,
                    "reason": repr(error),
                },
            )
            eval_env = None
        finally:
            # Building the evaluator rebound Warp's global device; the trainer
            # has to own it again before collect() steps.
            env.select_warp()
    # The ruler. Without it every eval point fights whatever the pool happens
    # to hold, so ``eval/exchange`` mixes the learner's progress with the
    # pool's own: over v14 it ranged 0.75-1.97 across snapshots 13 to 113
    # updates old while points against any single opponent rose 13%. Loaded
    # before the runner so a mismatched file fails in the first second.
    anchor = None
    if getattr(args, "eval_anchor", None):
        anchor = load_opponent_anchor(args.eval_anchor)
        write_json(
            args.output / "eval_anchor.json",
            {
                "path": anchor["path"],
                "iteration": anchor["iteration"],
                "actor_tensors": len(anchor["actor"]),
                "normalizer_tensors": len(anchor["normalizer"]),
            },
        )
        print(
            json.dumps(
                {
                    "eval_anchor": anchor["path"],
                    "eval_anchor_iteration": anchor["iteration"],
                }
            ),
            flush=True,
        )
    try:
        demo, motions = load_amp_motions(
            args.amp_motions,
            env.nominal,
            torch.zeros_like(env.nominal),
            device=args.device,
            resource_manifest_sha256=identity["resources"]["assets"],
        )
        identity["motions"] = motions
        write_json(
            args.output / "environment.json",
            {
                "physics": "mujoco-warp",
                "dt": 0.005,
                "control_dt": 0.02,
                "ordered_evaluation_physics": env.physics.ordered,
                "contact_parameters": "competition XML, no material randomization",
                "asset_identity": assets.identity(),
            },
        )
        runner = OnPolicyRunner(
            env,
            demo,
            reference,
            config,
            args.output,
            identity,
            policy=policy,
            training_mode=args.training_mode,
            opponent_pool_capacity=args.opponent_pool_capacity,
            opponent_refresh_updates=args.opponent_refresh_updates,
            eval_env=eval_env,
            eval_interval=getattr(args, "eval_interval", 0),
            eval_steps=getattr(args, "eval_steps", 480),
            std_max=float(getattr(args, "std_max", 0.0) or 0.0),
            std_fixed=float(getattr(args, "std_fixed", 0.0) or 0.0),
            eval_anchor=anchor,
        )
        if args.resume:
            runner.resume(
                args.resume, allow_code_change=args.allow_resume_code_change
            )
            write_json(args.output / "resume.json", runner.resume_source)
        elif args.supervised_checkpoint:
            runner.initialize_from_supervised(args.supervised_checkpoint)
        elif args.policy_checkpoint:
            runner.initialize_from_policy(args.policy_checkpoint)
        else:
            runner.initialize_from_prior(args.prior_checkpoint, prior_records)
        # Idempotent, and covers the paths above that do not go through
        # resume(): a pin must hold from the first rollout, not from the
        # second update.
        runner.pin_std()
        runner.policy.enable_actor_cuda_graphs()
        result = runner.learn(
            args.updates,
            save_interval=1024,
            extra_save_interval=5000,
            time_budget=time_budget,
        )
        write_json(
            args.output / "complete.json",
            {
                "updates_completed": runner.iteration,
                "training_result": result,
                "transitions_per_role": runner.transitions_per_role,
                "boxing_quality_verified": False,
            },
        )
    finally:
        if eval_env is not None:
            eval_env.close()
        env.close()
