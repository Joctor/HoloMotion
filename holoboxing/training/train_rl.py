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
        "--opponent-frozen",
        type=_common.existing_path,
        action="append",
        metavar="CHECKPOINT",
        default=None,
        help=(
            "pin the training opponent to a checkpoint; repeat for several. "
            "Turns off snapshotting and refresh, so the opponent stays at the "
            "level it was saved at. Needed because in symmetric self-play a "
            "reprice of the scoring terms is cancelled at the equilibrium: "
            "taken_point -80 -> -140 moved the round task term from +75.9 to "
            "+1.17 while points scored and conceded both rose (42 -> 46 and "
            "56 -> 62). Against a fixed attacker the same price is a real "
            "gradient. Requires --training-mode selfplay."
        ),
    )
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
            "single-point exchange sd near 0.08. Note that 480 steps is 16% of "
            "a 60 s round, so eval/win_rate at that horizon is a 9.6 s "
            "scoreline (eval/horizon says which); 3000 makes it the round "
            "result at about 7x the cost."
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
        "--eval-anchor",
        type=Path,
        default=None,
        help=(
            "training checkpoint whose ego actor becomes the evaluator's "
            "opponent, for the whole run. The pool's opponent changes every "
            "refresh, so eval/exchange against it is not comparable across "
            "points; a pinned opponent makes it a ruler. Pass the resume "
            "checkpoint to measure progress against the starting policy."
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
        "--balance-exponent",
        type=float,
        default=2.0,
        help=(
            "shape of the fall-margin band in balance_margin_reward, not its "
            "price (default 2.0 = the historical square). Weights are -3.0 "
            "over a 0.45 -> 0.30 m span, i.e. -180 per round at the fall "
            "line: a square charges 20 per round for sitting 5 cm lower "
            "(0.40 m) and 135 at 0.32 m, a quartic charges 2.2 and 102. Use "
            "4.0 to let the fighter crouch without letting it fall: the top "
            "of the band empties and the bottom keeps 76% of its price."
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
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help=(
            "discount factor (checkpoint default 0.995). At 50 Hz the 0.995 "
            "horizon is 200 steps = 4 s, so the final point of a 60 s round "
            "is weighted 4.9e-7 and 'win the round' is not in the objective "
            "at all. 0.9999 weights it 0.74 (0.9995 -> 0.22). The value "
            "scales with min(1/(1-gamma), 3000): ~200 steps at 0.995 and "
            "~2600 at 0.9999, so the fixed point moves from ~3.4 to ~60-80. "
            "That jump is why this flag is not usable without "
            "--critic-warmup."
        ),
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=None,
        help=(
            "GAE trace decay (checkpoint default 0.95). With gamma 0.995 the "
            "advantage has decayed to 0.4 s, so a terminal payoff at the last "
            "step can only reach ~20 steps back; the cross-round credit comes "
            "from the critic's bootstrap, not from this knob. 0.98 is the "
            "usual partner of gamma >= 0.999."
        ),
    )
    parser.add_argument(
        "--match-bonus",
        type=float,
        default=0.0,
        help=(
            "terminal payoff W*tanh(net/8) in round units (0 = off). Applied "
            "once, on the step a round ends, so W=80 is worth 0.8*80 = 64 in "
            "the return (the runner weights task 0.8, the same factor the "
            "scoreboard carries) against a round task income of ~99. The "
            "derivative at net=0 is W/8, i.e. the marginal value of one "
            "point: 3.8 is what the table already pays for one, so W below "
            "~30 changes nothing. Does nothing measurable at gamma 0.995 -- "
            "it must be paired with --gamma."
        ),
    )
    parser.add_argument(
        "--critic-warmup",
        type=int,
        default=0,
        help=(
            "fit only the critic for the first N updates of this run (0 = "
            "off). Changing gamma rescales the value target, so the resumed "
            "critic is wrong by the ratio of horizons and every advantage in "
            "the first pass is biased rather than noisy. While the flag is "
            "set the actor loss, the clip term and the entropy bonus are all "
            "dropped; watch ego/value until it stops falling."
        ),
    )
    parser.add_argument(
        "--critic-warmup-lr",
        type=float,
        default=20.0,
        help=(
            "learning-rate multiplier and gradient ceiling for the critic "
            "during --critic-warmup (default 20x). Adam moves ~lr per "
            "coordinate whatever the residual is, so at the normal 1.5e-4 the "
            "critic cannot travel from a value of 3 to a value of 50 inside "
            "any warm-up worth running. Safe because the actor carries no "
            "gradient while the warm-up is on; restored exactly afterwards "
            "and never written into a checkpoint."
        ),
    )
    parser.add_argument(
        "--shape-scale",
        type=float,
        default=1.0,
        help=(
            "multiplier on the 17 dense shaping weights, leaving official/"
            "taken/ko/match at full price (default 1.0). By update 3060 the "
            "shaping net was +50.7/round against a scoring net of +48.6, i.e. "
            "the hints were worth as much as the objective; 0.1 asks whether "
            "the policy is being carried by them. Scaling the scoreboard too "
            "would change the question, so it is exempt."
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
    if args.opponent_frozen:
        if args.training_mode != "selfplay":
            parser.error(
                "--opponent-frozen requires --training-mode selfplay"
            )
        if len(set(args.opponent_frozen)) != len(args.opponent_frozen):
            parser.error("--opponent-frozen repeats a checkpoint")
        if args.opponent_pool_capacity != 8 or (
            args.opponent_refresh_updates != 256
        ):
            print(
                "[args] --opponent-pool-capacity and "
                "--opponent-refresh-updates are ignored while the opponent "
                "is frozen",
                flush=True,
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
    if not 1.0 <= args.balance_exponent <= 8.0:
        parser.error(
            "--balance-exponent must lie in [1, 8]; below 1 the fall band is "
            "concave and charges more at 0.44 m than at 0.31 m"
        )
    if args.desired_kl is not None and not 0 < args.desired_kl < 10:
        parser.error("--desired-kl must lie in (0, 10)")
    if args.gamma is not None and not 0 < args.gamma <= 1:
        parser.error("--gamma must lie in (0, 1]")
    if args.gae_lambda is not None and not 0 <= args.gae_lambda <= 1:
        parser.error("--gae-lambda must lie in [0, 1]")
    if args.match_bonus < 0:
        parser.error("--match-bonus must be nonnegative (0 disables it)")
    if args.critic_warmup < 0:
        parser.error("--critic-warmup must be nonnegative")
    if not 1 <= args.critic_warmup_lr <= 1000:
        parser.error("--critic-warmup-lr must lie in [1, 1000]")
    if not 0 < args.shape_scale <= 1:
        parser.error(
            "--shape-scale must lie in (0, 1]; above 1 would raise the price "
            "of the hints, which is a different experiment"
        )
    if args.match_bonus > 0:
        # The penalty for not saying this is a run that looks like a failure:
        # the terminal term is weighted (1-gamma) like everything else, so at
        # 0.995 its contribution to the return is ~1e-6 of its face value.
        gamma = args.gamma if args.gamma is not None else 0.995
        if gamma < 0.999:
            parser.error(
                "--match-bonus does nothing at gamma < 0.999: the last step "
                "of a 3000-step round is discounted to ~1e-6. Pass --gamma "
                "0.9999 (and --critic-warmup)."
            )
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
            "gamma": args.gamma,
            "lam": args.gae_lambda,
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
