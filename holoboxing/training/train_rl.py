"""Train or resume the boxing planner on competition physics."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from holoboxing.training import cli as _common
from holoboxing.training.tasks.combat.config.sandbag import TrainConfig
from holoboxing.training.time_budget import (
    add_budget_arguments,
    budget_from_arguments,
)


def main(argv=None):
    started_at = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_controller_arguments(parser)
    add_budget_arguments(parser)
    parser.add_argument(
        "--amp-motions",
        type=_common.existing_path,
        required=True,
        help="prepared directory from holoboxing.data prepare containing 50 Hz AMP references",
    )
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument(
        "--supervised-checkpoint",
        type=_common.existing_path,
        help="epoch_*.pt produced by holoboxing.training.train_sl",
    )
    start.add_argument("--resume", type=_common.existing_path)
    start.add_argument(
        "--policy-checkpoint",
        type=_common.existing_path,
        help="RL actor initialization for a new stage; resets learner state",
    )
    start.add_argument(
        "--prior-checkpoint",
        type=_common.existing_path,
        help="start latent RL without imitation",
    )
    parser.add_argument(
        "--config",
        type=_common.existing_path,
        help="experiment-specific TrainConfig JSON",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--training-mode",
        choices=("paired", "standing", "selfplay"),
        default="paired",
        help="paired learning, standing targets, or historical opponents",
    )
    parser.add_argument("--opponent-pool-capacity", type=int, default=8)
    parser.add_argument("--opponent-refresh-updates", type=int, default=256)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=25,
        help=(
            "run a greedy, fixed-opponent readout every N updates and log it "
            "as eval/* (0 disables). Training metrics sample the learner's "
            "exploration noise against a clean opponent, so their exchange "
            "ratio is a biased lower bound; this one is not."
        ),
    )
    parser.add_argument(
        "--eval-worlds",
        type=int,
        default=256,
        help="worlds in the dedicated eval simulator (separate GPU state)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=480,
        help=(
            "control steps per readout. 256 worlds x 480 steps is about 123k "
            "world-steps, roughly 2.5 training updates of physics, for a "
            "single-point exchange sd near 0.08"
        ),
    )
    parser.add_argument(
        "--eval-device",
        default="",
        help=(
            "device for the eval simulator's own physics buffers (empty = a "
            "second GPU when one exists, else --device). Keeping it off the "
            "training GPU matters: mujoco_warp allocates a ~1.1 GB per-step "
            "collision table at 2048 worlds, and two resident simulators on "
            "one device make that allocation fail."
        ),
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=None,
        help=(
            "override the entropy bonus (checkpoint default 4e-4). It is "
            "summed over 32 latent axes and is ~1.5x the policy surrogate in "
            "magnitude, so it sets sigma rather than the advantage does; "
            "resuming with a different value is allowed and logged."
        ),
    )
    parser.add_argument(
        "--std-max",
        type=float,
        default=0.0,
        help=(
            "probe ceiling on the sampling sigma (0 = off). The latent is a "
            "unit vector, so 0.24 is a ~50 degree departure from the greedy "
            "direction; 0.08 is a ~25 degree one. Frozen at the boundary, so "
            "read it as a probe, not a fix."
        ),
    )
    parser.add_argument(
        "--std-fixed",
        type=float,
        default=0.0,
        help=(
            "pin the sampling sigma at this value for the whole run (0 = "
            "off). Unlike --std-max this writes the value into the parameter "
            "every update, so the entropy bonus cannot walk it back, the "
            "checkpoint stores the law it was trained under, and a later run "
            "without the flag does not silently restore the noise. Implies a "
            "ceiling of the same value."
        ),
    )
    parser.add_argument(
        "--desired-kl",
        type=float,
        default=None,
        help=(
            "override the KL target the adaptive lr holds. The target lives "
            "in distribution space, so it buys less movement as sigma shrinks: "
            "holding KL fixed, the per-update mean displacement scales with "
            "sigma. Pinning sigma from 0.238 to 0.08 therefore needs about "
            "0.09 to keep the old action-space step (0.01 x (0.238/0.08)^2); "
            "at the old 0.01 the lr settles near 8e-5 and learning is ~3x "
            "slower."
        ),
    )
    parser.add_argument("--allow-resume-code-change", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new empty training output directory",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=40000,
        help="Additional PPO iterations to run (default: 40000)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device after CUDA_VISIBLE_DEVICES remapping",
    )
    parser.add_argument("--num-envs", type=int)
    args = parser.parse_args(argv)
    if not (
        args.supervised_checkpoint
        or args.resume
        or args.prior_checkpoint
        or args.policy_checkpoint
    ).is_file():
        parser.error("Initialization must be a checkpoint file")
    if args.allow_resume_code_change and not args.resume:
        parser.error("--allow-resume-code-change requires --resume")
    if args.updates <= 0:
        parser.error("--updates must be positive")
    if args.opponent_pool_capacity < 2 or args.opponent_refresh_updates < 1:
        parser.error(
            "Opponent pool needs >=2 slots and a positive refresh interval"
        )
    if args.eval_interval < 0 or args.eval_worlds < 1 or args.eval_steps < 1:
        parser.error("Eval needs a nonnegative interval and positive sizes")
    if args.eval_device and not (
        str(args.eval_device).startswith("cuda:")
        and str(args.eval_device)[5:].isdigit()
    ):
        # A bare "cuda" would resolve through torch.cuda.current_device() and
        # silently land back on the training GPU, which is the failure this
        # flag exists to avoid.
        parser.error("--eval-device must include an index, e.g. cuda:1")
    if args.entropy_coef is not None and args.entropy_coef < 0:
        parser.error("--entropy-coef must be nonnegative")
    if args.std_max < 0:
        parser.error("--std-max must be nonnegative (0 disables it)")
    if args.std_fixed < 0:
        parser.error("--std-fixed must be nonnegative (0 disables it)")
    if args.desired_kl is not None and not 0 < args.desired_kl < 10:
        parser.error("--desired-kl must lie in (0, 10)")
    try:
        budget = budget_from_arguments(args, started_at=started_at)
    except ValueError as error:
        parser.error(str(error))
    config = (
        TrainConfig(**json.loads(args.config.read_text()))
        if args.config
        else TrainConfig()
    )
    changes = {
        k: v
        for k, v in {
            "num_envs": args.num_envs,
            "seed": args.seed,
            "entropy_coef": args.entropy_coef,
            "desired_kl": args.desired_kl,
        }.items()
        if v is not None
    }
    config = replace(config, **changes)
    args.num_envs, args.seed, args.action_dim = (
        config.num_envs,
        config.seed,
        config.action_dim,
    )
    from holoboxing.training.scripts.train_mujoco import train

    with budget.handle_signals():
        train(args, config, time_budget=budget)


if __name__ == "__main__":
    main()
