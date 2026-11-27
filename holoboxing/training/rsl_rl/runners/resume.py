"""Learning-state restart checks; physical episodes are deliberately reset."""

import copy
import random
from pathlib import Path

import numpy as np
import torch

from holoboxing.training.rsl_rl.algorithms.ppo import require_finite
from holoboxing.training.rsl_rl.modules.normalization import (
    validate_statistics,
)
from holoboxing.training.tasks.combat.config.sandbag import ROLES

RESTART_MODE = "reset_physics_and_episode_histories"

# Configuration keys that are allowed to differ on resume because nothing
# archived depends on them. All three are scalars that enter the loss or the
# reward mix; none of them touches a saved tensor, an optimizer state, a
# normalizer statistic, the AMP replay or a random stream, so changing one
# mid-run is a retune and not a format change.
#
# They need the exemption because they are the two knobs actually read off the
# training curve -- the entropy bonus and the task/style mix -- and the only
# other way to change one is ``--policy-checkpoint``, which discards the
# optimizer, the replay buffer and every random stream. Paying that price for
# a scalar is the wrong trade; silently paying it, because the operator gave up
# and restarted, is worse.
MUTABLE_CONFIG_KEYS = frozenset(
    {
        "entropy_coef",
        "desired_kl",
        "task_reward_weight",
        "style_reward_weight",
    }
)


def validate_module_state(module, saved, label):
    """Reject incompatible tensors before PyTorch can copy or cast any of them."""
    expected = module.state_dict()
    if not isinstance(saved, dict):
        raise ValueError(f"Invalid {label} state")
    missing = expected.keys() - saved.keys()
    # Preserve the existing four-buffer legacy normalizer migration.
    for key in expected:
        if not key.endswith("_ex2"):
            continue
        prefix = key[: -len("_ex2")]
        legacy = {prefix + n for n in ("_mean", "_var", "_std", "count")}
        added = {
            prefix + n
            for n in (
                "_ex2",
                "_last_sync_mean",
                "_last_sync_var",
                "_last_sync_count",
            )
        }
        if legacy <= saved.keys() and not (added & saved.keys()):
            missing -= added
    if missing or saved.keys() - expected.keys():
        raise ValueError(f"Incomplete or unexpected {label} keys")
    for name, value in saved.items():
        template = expected[name]
        if (
            not torch.is_tensor(value)
            or value.shape != template.shape
            or value.dtype != template.dtype
            or value.layout != template.layout
        ):
            raise ValueError(f"Incompatible {label} tensor: {name}")
    for key in expected.keys() - saved.keys():
        if key.endswith("_ex2"):
            prefix = key[: -len("_ex2")]
            ex2 = saved[prefix + "_var"] + saved[prefix + "_mean"].square()
            if not torch.isfinite(ex2).all():
                raise ValueError(f"Nonfinite migrated {label} statistics")


def check_identity(saved, current, allow_code_change=False):
    code_keys = {"code", "shared_code"}
    if {k: v for k, v in saved.items() if k not in code_keys} != {
        k: v for k, v in current.items() if k not in code_keys
    }:
        raise ValueError("Resume resources, demonstrations or runtime differ")
    changes = {
        key: {"saved": saved.get(key), "current": current.get(key)}
        for key in code_keys
        if saved.get(key) != current.get(key)
    }
    if changes and not allow_code_change:
        raise ValueError(
            "Resume code differs; review the changes before explicitly using "
            "--allow-resume-code-change"
        )
    return changes


def capture_rng(env):
    device = torch.device(env.device)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device)
        if device.type == "cuda"
        else None,
        "environment": env.generator.get_state(),
        "observation_noise": env.actor_noise.rng_state(),
    }


def validate_rng(rng, device):
    device = torch.device(device)
    if (rng["cuda"] is None) != (device.type != "cuda"):
        raise ValueError("Resume must use the same CPU/CUDA device type")
    # Validate without advancing any live random stream.
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    torch.Generator().set_state(rng["torch"])
    for key in ("environment", "observation_noise"):
        torch.Generator(device=device).set_state(rng[key])
    if device.type == "cuda":
        torch.Generator(device=device).set_state(rng["cuda"])


def restore_rng(rng, env):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.device(env.device).type == "cuda":
        torch.cuda.set_rng_state(rng["cuda"], env.device)
    env.generator.set_state(rng["environment"])
    env.actor_noise.generator.set_state(rng["observation_noise"])


def validate_adam(saved, optimizer, steps):
    groups, current = saved["param_groups"], optimizer.param_groups
    if len(groups) != len(current):
        raise ValueError("Incomplete optimizer parameter groups")
    ids = []
    for group, live in zip(groups, current, strict=True):
        if len(group["params"]) != len(live["params"]):
            raise ValueError("Optimizer parameter count differs")
        for key in live.keys() - {"params", "lr"}:
            if group.get(key) != live[key]:
                raise ValueError(f"Optimizer setting differs: {key}")
        for index, parameter in zip(
            group["params"], live["params"], strict=True
        ):
            ids.append(index)
            if steps:
                moment = saved["state"].get(index, {})
                if set(moment) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("Missing Adam moments")
                if (
                    moment["step"].numel() != 1
                    or moment["step"].item() != steps
                ):
                    raise ValueError("Adam step differs from learner progress")
                for key in ("exp_avg", "exp_avg_sq"):
                    if (
                        moment[key].shape != parameter.shape
                        or moment[key].dtype != parameter.dtype
                    ):
                        raise ValueError("Adam moment layout differs")
                if (moment["exp_avg_sq"] < 0).any():
                    raise ValueError("Invalid Adam squared moment")
    if len(set(ids)) != len(ids) or set(saved["state"]) != (
        set(ids) if steps else set()
    ):
        raise ValueError("Unexpected optimizer state ownership")


def learning_state(runner):
    replay = runner.amp.replay
    return {
        "resume_version": 1,
        "restart_mode": RESTART_MODE,
        "optimizers": {
            role: optimizer.state_dict()
            for role, optimizer in runner.algorithm.optimizers.items()
        },
        "disc_optimizer": runner.amp.optimizer.state_dict(),
        "amp_replay": {
            "capacity": replay.max_length,
            "size": replay.current_size,
            "index": replay.current_idx,
            "buffer": replay.buffer,
        },
        "rng": capture_rng(runner.env),
        "invalid_transitions": runner.invalid_transitions,
        "resume_source": runner.resume_source,
    }


def opponent_source(protocol):
    """Which opponent schedule a protocol describes: ``pool`` or ``frozen``.

    Checkpoints written before the switch existed carry no ``opponent`` key and
    are all rotating pools, so the default has to be ``pool`` rather than
    ``None`` -- otherwise every pre-existing archive would look like a
    different source and the compatibility gate below would open for changes
    it is meant to refuse.
    """
    if not isinstance(protocol, dict):
        return "pool"
    return protocol.get("opponent", "pool")


def protocol_compatible(saved, new):
    """Whether a checkpoint may continue under a different training protocol.

    Two transitions are intended. The world-anchored warm-up (``standing``)
    hands over to self-play: warm-up carries no opponent pool -- the opponent
    is a fixed sandbag and the protocol is just ``{"mode": "standing"}`` -- so
    a strict equality check refuses the handover the curriculum is built
    around. On that handover the pool starts empty, so the first ``prepare``
    snapshots the resuming policy itself: the opponent is exactly the level the
    warm-up reached.

    The second is freezing or unfreezing the training opponent. That is the
    single variable of the price experiment: in symmetric self-play a reprice
    of ``taken_point`` is cancelled because the opponent's output rises with
    the learner's, so the only way to read the price is to hold the attacker
    still. The pool state is rebuilt from the frozen checkpoint paths, so the
    archived one is never validated against the new one, and the schedule
    difference cannot pass silently.

    Everything else stays rejected: self-play back to standing would drop the
    pool, and a capacity or interval change inside one source would change the
    opponent schedule mid-run.
    """
    if saved == new:
        return True
    if not (isinstance(saved, dict) and isinstance(new, dict)):
        return False
    if (
        set(saved) == {"mode"}
        and saved.get("mode") == "standing"
        and new.get("mode") == "selfplay"
    ):
        return True
    return (
        saved.get("mode") == new.get("mode") == "selfplay"
        and opponent_source(saved) != opponent_source(new)
    )


def validate_learning_state(runner, state):
    cfg = runner.config
    iteration = state["iteration"]
    if type(iteration) is not int or iteration < 0:
        raise ValueError("Invalid completed update count")
    transitions = iteration * cfg.num_envs * cfg.num_steps_per_env
    steps = iteration * cfg.num_learning_epochs * cfg.num_mini_batches
    active_roles = runner.algorithm.active_roles
    expected_steps = {
        role: steps if role in active_roles else 0 for role in ROLES
    }
    if (
        type(state["transitions_per_role"]) is not int
        or state["transitions_per_role"] != transitions
        or state["optimizer_steps"] != expected_steps
        or state["disc_optimizer_steps"] != steps
        or state["elapsed_seconds"] < 0
    ):
        raise ValueError("Inconsistent completed training progress")
    protocol = state.get("training_protocol", {"mode": "paired"})
    if not protocol_compatible(protocol, runner.training_protocol):
        raise ValueError("Resume training mode or opponent schedule differs")
    # A source switch (rotating pool <-> frozen set) rebuilds the pool from the
    # frozen checkpoint paths, so the archived pool describes a different
    # opponent. Validating it against the new one would refuse exactly the
    # experiment the flag exists for.
    pool_state = (
        state.get("opponent_pool")
        if runner.opponent_pool is not None
        and opponent_source(protocol)
        == opponent_source(runner.training_protocol)
        else None
    )
    if pool_state is not None:
        runner.opponent_pool.validate(pool_state)
        selected = pool_state["selected"]
        if selected is not None:
            snapshot = pool_state["snapshots"][selected]
            for prefix, tensors in (
                ("taskpolicy_opponent.", snapshot["actor"]),
                ("actor_opponent.", snapshot["normalizer"]),
            ):
                saved = (
                    state["policy"]
                    if prefix.startswith("taskpolicy")
                    else state["normalizers"]
                )
                if any(
                    not torch.equal(saved[prefix + key].cpu(), value.cpu())
                    for key, value in tensors.items()
                ):
                    raise ValueError(
                        "Installed opponent differs from its frozen snapshot"
                    )
    invalid = state["invalid_transitions"]
    if type(invalid) is not int or not 0 <= invalid <= transitions:
        raise ValueError("Invalid dropped-transition count")
    for name in ("optimizers", "learning_rates"):
        if set(state[name]) != set(ROLES):
            raise ValueError(f"Missing paired {name}")
    require_finite(state, "resume checkpoint")
    for module, key in (
        (runner.policy, "policy"),
        (runner.normalizers, "normalizers"),
        (runner.amp.discriminator, "discriminator"),
        (runner.amp.normalizer, "amp_normalizer"),
    ):
        validate_module_state(module, state[key], key)
    for role in ROLES:
        actor = getattr(runner.policy, f"taskpolicy_{role}")
        for name in ("neutral_reference", "physical_scale"):
            expected = getattr(actor, name).detach().cpu()
            actual = (
                state["policy"][f"taskpolicy_{role}.{name}"].detach().cpu()
            )
            matches = (
                torch.allclose(expected, actual, atol=1e-6, rtol=0)
                if name == "neutral_reference"
                else torch.equal(expected, actual)
            )
            if not matches:
                raise ValueError(
                    f"Checkpoint action mapping differs: {role}/{name}"
                )
    validate_statistics(state["normalizers"])
    if hasattr(runner.policy, "policy_spec"):
        if state.get("policy_spec") != runner.policy.policy_spec:
            raise ValueError("Resume frozen-prior specification differs")
        runner.policy.validate_action_mapping(state["policy"])
    validate_statistics(state["amp_normalizer"])
    provenance = state["supervised_initialization"]
    if (
        not isinstance(provenance, dict)
        or not isinstance(provenance.get("checkpoint"), str)
        or not provenance["checkpoint"]
        or not isinstance(provenance.get("sha256"), str)
        or len(provenance["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in provenance["sha256"])
    ):
        raise ValueError("Missing supervised initialization provenance")
    if state["normalizers"]["actor_ego.count"].item() <= 0:
        raise ValueError("Attacker normalizer must be fitted")
    for role in ROLES:
        lr = state["learning_rates"][role]
        if not cfg.min_learning_rate <= lr <= cfg.max_learning_rate:
            raise ValueError("Invalid saved learning rate")
        if any(
            g["lr"] != lr for g in state["optimizers"][role]["param_groups"]
        ):
            raise ValueError("Optimizer and scheduler learning rates differ")
        validate_adam(
            state["optimizers"][role],
            runner.algorithm.optimizers[role],
            expected_steps[role],
        )
    validate_adam(state["disc_optimizer"], runner.amp.optimizer, steps)
    if any(
        g["lr"] != cfg.disc_learning_rate
        for g in state["disc_optimizer"]["param_groups"]
    ):
        raise ValueError("AMP optimizer learning rate differs")
    replay = state["amp_replay"]
    capacity = cfg.amp_replay_buffer_size
    inserted = len(active_roles) * (transitions - invalid)
    if (
        replay["capacity"] != capacity
        or type(replay["size"]) is not int
        or type(replay["index"]) is not int
        or replay["size"] != min(inserted, capacity)
        or replay["index"] != inserted % capacity
    ):
        raise ValueError("AMP replay differs from completed rollout progress")
    buffer = replay["buffer"]
    if inserted == 0:
        if buffer is not None:
            raise ValueError("Unexpected empty AMP replay data")
    elif (
        not torch.is_tensor(buffer)
        or buffer.shape != (capacity, 122)
        or buffer.dtype != torch.float32
    ):
        raise ValueError("Invalid AMP replay tensor")
    validate_rng(state["rng"], runner.device)


def configuration_difference(saved, current, ignore=()):
    """Name every configuration key that differs, saved value first.

    A single opaque refusal leaves the operator diffing two JSON files by
    hand, and the usual cause is one overridden flag (``--num-envs``) buried
    in forty identical entries. Rollout width cannot be remapped after the
    fact -- transition counts, AMP replay occupancy and the saved ``config``
    are all keyed to it -- so the check stays strict, but it says which key.

    ``ignore`` is for the keys in :data:`MUTABLE_CONFIG_KEYS`, whose values are
    not baked into any archived tensor and may therefore be retuned on resume.
    """
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return "saved configuration is not a mapping"
    missing = sorted(current.keys() - saved.keys())
    unexpected = sorted(saved.keys() - current.keys())
    changed = [
        f"{key}: checkpoint={saved[key]!r} current={current[key]!r}"
        for key in sorted(saved.keys() & current.keys())
        if saved[key] != current[key] and key not in ignore
    ]
    if missing:
        changed.append("absent from checkpoint: " + ", ".join(missing))
    if unexpected:
        changed.append("unknown to this build: " + ", ".join(unexpected))
    return "; ".join(changed)


def resume(runner, path, checkpoint_format, allow_code_change=False):
    if (
        runner.iteration
        or runner.observations is not None
        or runner.algorithm.storage.step
        or runner.supervised_initialization is not None
        or runner.resume_source is not None
    ):
        raise ValueError("Resume requires a fresh runner")
    state = torch.load(path, map_location="cpu", weights_only=False)
    # A warm-up checkpoint has no opponent pool at all (the opponent is a
    # world-anchored sandbag), so it cannot satisfy the self-play key set. The
    # handover seeds a fresh pool from the resuming weights instead.
    warmup = (
        isinstance(state, dict)
        and isinstance(state.get("training_protocol"), dict)
        and state["training_protocol"].get("mode") == "standing"
    )
    required = set(runner.checkpoint_state()) - {"training_protocol"}
    if warmup:
        required.discard("opponent_pool")
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError(
            "Incomplete resume checkpoint; legacy weights-only checkpoints "
            "cannot restore optimizer, replay or random history"
        )
    if state["format"] != checkpoint_format:
        raise ValueError(
            f"Resume checkpoint format differs: {state['format']!r} vs "
            f"{checkpoint_format!r}"
        )
    if state["resume_version"] != 1 or state["restart_mode"] != RESTART_MODE:
        raise ValueError(
            "Resume checkpoint layout differs: "
            f"version={state['resume_version']!r} "
            f"restart={state['restart_mode']!r}"
        )
    current_config = runner.config.to_dict()
    difference = configuration_difference(
        state["config"], current_config, ignore=MUTABLE_CONFIG_KEYS
    )
    if difference:
        raise ValueError(f"Resume configuration differs ({difference})")
    retuned = [
        f"{key}: {state['config'][key]!r} -> {current_config[key]!r}"
        for key in sorted(MUTABLE_CONFIG_KEYS & state["config"].keys())
        if state["config"][key] != current_config.get(key)
    ]
    if retuned:
        # Not a warning: these keys can only be changed here, so the run should
        # say out loud which price list it is about to train against. Silence
        # would make two metrics files incomparable for no visible reason.
        print("[resume] retuned hyperparameters: " + ", ".join(retuned), flush=True)
    changes = check_identity(
        state["identity"], runner.identity, allow_code_change
    )
    validate_learning_state(runner, state)
    runner.policy.load_state_dict(state["policy"], strict=True)
    runner.normalizers.load_state_dict(state["normalizers"], strict=True)
    runner.amp.discriminator.load_state_dict(
        state["discriminator"], strict=True
    )
    runner.amp.normalizer.load_state_dict(state["amp_normalizer"], strict=True)
    for role in ROLES:
        runner.algorithm.optimizers[role].load_state_dict(
            state["optimizers"][role]
        )
    runner.amp.optimizer.load_state_dict(state["disc_optimizer"])
    runner.algorithm.learning_rates = copy.deepcopy(state["learning_rates"])
    runner.algorithm.optimizer_steps = copy.deepcopy(state["optimizer_steps"])
    runner.amp.optimizer_steps = state["disc_optimizer_steps"]
    replay = state["amp_replay"]
    runner.amp.replay.buffer = (
        None
        if replay["buffer"] is None
        else replay["buffer"].to(runner.device)
    )
    runner.amp.replay.current_size = replay["size"]
    runner.amp.replay.current_idx = replay["index"]
    # Same condition as the validation above: a frozen set is installed from
    # its own paths at construction time, so an archive from the other source
    # must not overwrite it.
    same_source = opponent_source(
        state.get("training_protocol", {"mode": "paired"})
    ) == opponent_source(runner.training_protocol)
    if runner.opponent_pool is not None and same_source:
        pool_state = state.get("opponent_pool")
        if pool_state is not None:
            runner.opponent_pool.load_state_dict(pool_state)
    for key in (
        "iteration",
        "transitions_per_role",
        "elapsed_seconds",
        "invalid_transitions",
        "supervised_initialization",
    ):
        setattr(runner, key, copy.deepcopy(state[key]))
    restore_rng(state["rng"], runner.env)
    runner.algorithm.reset_exploration()
    runner.resume_source = {
        "checkpoint": str(Path(path).resolve()),
        "iteration": state["iteration"],
        "code_changes": changes,
    }
