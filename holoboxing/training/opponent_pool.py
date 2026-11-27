"""Bounded frozen self-play snapshots; collection never trains the opponent."""

import copy

import torch

from holoboxing.training.motion_prior import state_sha256


def cpu_state(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


class OpponentPool:
    def __init__(
        self,
        policy,
        normalizers,
        *,
        seed,
        capacity=8,
        interval=256,
        frozen=None,
    ):
        self.policy, self.normalizers = policy, normalizers
        self.generator = torch.Generator().manual_seed(seed)
        self.snapshots = []
        self.selected = None
        self.next_refresh = 0
        if frozen is not None:
            self._freeze(frozen, interval)
            return
        if (
            type(capacity) is not int
            or capacity < 2
            or type(interval) is not int
            or interval < 1
        ):
            raise ValueError(
                "Self-play needs a positive interval and at least two slots"
            )
        self.frozen = False
        self.frozen_paths = ()
        self.capacity, self.interval = capacity, interval

    def _freeze(self, frozen, interval):
        """Pin the training opponent to pre-run checkpoints, never refreshed.

        Why: in symmetric self-play the opponent improves in lockstep with the
        learner, so any linear reprice of the scoring terms is cancelled at the
        equilibrium. Raising ``taken_point`` from -80 to -140 moved the round
        task term from +75.9 to +1.17 while points scored and conceded *both*
        rose (42 -> 46 and 56 -> 62): the price worked, the trade simply grew
        to absorb it. A frozen opponent breaks the lockstep, so the same price
        becomes a gradient against an attacker whose output is constant.
        """
        if not isinstance(frozen, (list, tuple)) or not frozen:
            raise ValueError("Frozen opponents must be a nonempty sequence")
        if type(interval) is not int or interval < 1:
            raise ValueError("Frozen opponents need a positive interval")
        self.frozen = True
        self.interval = interval
        self.frozen_paths = tuple(str(item["path"]) for item in frozen)
        if len(set(self.frozen_paths)) != len(self.frozen_paths):
            raise ValueError("Frozen opponent list repeats a checkpoint")
        iterations = []
        for item in frozen:
            snapshot = self._adopt(item)
            iterations.append(snapshot["iteration"])
            self.snapshots.append(snapshot)
        if iterations != sorted(set(iterations)):
            raise ValueError("Frozen opponents must ascend by iteration")
        self.capacity = len(self.snapshots)
        self.selected = 0
        # Keeps ``iteration + interval`` finite for any plausible run, so the
        # rotation branch never fires. Expressing "never" as a boundary rather
        # than a flag keeps ``prepare`` a single comparison.
        self.next_refresh = 1 << 62
        self._install()

    def _adopt(self, item):
        """Validate one loaded checkpoint and turn it into a pool snapshot."""
        from holoboxing.training.rsl_rl.runners.resume import (
            validate_module_state,
        )

        actor = item["actor"]
        normalizer = item["normalizer"]
        validate_module_state(
            self.policy.taskpolicy_opponent, actor, "frozen opponent actor"
        )
        validate_module_state(
            self.normalizers["actor_opponent"],
            normalizer,
            "frozen opponent statistics",
        )
        actor = {k: v.detach().cpu().clone() for k, v in actor.items()}
        normalizer = {
            k: v.detach().cpu().clone() for k, v in normalizer.items()
        }
        return {
            "iteration": int(item["iteration"]),
            "actor": actor,
            "normalizer": normalizer,
            "sha256": state_sha256(
                {**actor, **{"norm." + k: v for k, v in normalizer.items()}}
            ),
        }

    def _snapshot(self, iteration):
        actor = cpu_state(self.policy.taskpolicy_ego)
        normalizer = cpu_state(self.normalizers["actor_ego"])
        return {
            "iteration": iteration,
            "actor": actor,
            "normalizer": normalizer,
            "sha256": state_sha256(
                {**actor, **{"norm." + k: v for k, v in normalizer.items()}}
            ),
        }

    def _install(self):
        snapshot = self.snapshots[self.selected]
        self.policy.taskpolicy_opponent.load_state_dict(
            snapshot["actor"], strict=True
        )
        self.normalizers["actor_opponent"].load_state_dict(
            snapshot["normalizer"], strict=True
        )

    @property
    def selected_iteration(self):
        """Iteration label of the snapshot currently installed, -1 if none.

        Logged next to the score so a spike can be attributed: a round that
        suddenly scores 2x almost always means this number jumped backwards to
        the retained initial opponent, not that the fighter improved.
        """
        if self.selected is None:
            return -1
        return int(self.snapshots[self.selected]["iteration"])

    def prepare(self, iteration):
        """Change at update boundaries; the caller keeps the fight running.

        The swap is deliberately not an episode boundary: the runner continues
        the live episodes so a 60 s round is sampled end to end. Only the
        opponent's weights and observation statistics change, which the ego
        cannot observe -- it has to read the current pose, not the opponent's
        identity.
        """
        if self.frozen:
            # Nothing to rotate: the whole point is that the other side does
            # not move while the price list is being tested.
            return False
        if self.selected is not None and iteration < self.next_refresh:
            return False
        if not self.snapshots or self.snapshots[-1]["iteration"] != iteration:
            self.snapshots.append(self._snapshot(iteration))
            if len(self.snapshots) > self.capacity:
                # Pure FIFO. Retaining the initial opponent was meant to stop
                # the fighter forgetting how to beat its earliest self, but
                # measured, that slot held iteration 474 against a current
                # 1883 -- 1409 updates behind -- and every uniform draw that
                # hit it (12.5% of refreshes) produced a 5x exchange spike:
                # opponent points per round fell 20.2 -> 6.6 and contact
                # doubled within one refresh. The signal it injects ("touch
                # a lot, he cannot answer") costs more than the forgetting
                # it prevents, so the window now slides.
                del self.snapshots[0]
        self.selected = int(
            torch.randint(len(self.snapshots), (), generator=self.generator)
        )
        self.next_refresh = iteration + self.interval
        self._install()
        return True

    def state_dict(self):
        return copy.deepcopy(
            {
                "frozen": self.frozen,
                "frozen_paths": self.frozen_paths,
                "capacity": self.capacity,
                "interval": self.interval,
                "snapshots": self.snapshots,
                "selected": self.selected,
                "next_refresh": self.next_refresh,
                "rng": self.generator.get_state(),
            }
        )

    def validate(self, state):
        from holoboxing.training.rsl_rl.algorithms.ppo import require_finite
        from holoboxing.training.rsl_rl.modules.normalization import (
            validate_statistics,
        )
        from holoboxing.training.rsl_rl.runners.resume import (
            validate_module_state,
        )

        require_finite(state, "self-play pool")
        # A frozen pool has no schedule to be compatible with, so the run must
        # not silently accept a self-play archive (or the reverse): the two
        # describe different opponents and the exchange numbers are not
        # comparable.
        if bool(state.get("frozen", False)) != self.frozen:
            raise ValueError(
                "Opponent pool mode differs: the checkpoint "
                f"{'froze' if state.get('frozen') else 'rotated'} its "
                f"opponent and this run "
                f"{'freezes' if self.frozen else 'rotates'} it"
            )
        if self.frozen:
            # Content identity, not file paths: a checkpoint copied to another
            # directory is the same opponent, and refusing it would make the
            # archives unusable after a workspace move.
            archived = [item["sha256"] for item in state["snapshots"]]
            installed = [item["sha256"] for item in self.snapshots]
            if archived != installed:
                raise ValueError("Frozen opponent checkpoints differ")
        elif (
            state["capacity"] != self.capacity
            or state["interval"] != self.interval
        ):
            raise ValueError("Self-play pool recipe differs")
        snapshots = state["snapshots"]
        selected = state["selected"]
        if not isinstance(snapshots, list) or len(snapshots) > self.capacity:
            raise ValueError("Invalid self-play pool size")
        if selected is None:
            if snapshots or state["next_refresh"] != 0:
                raise ValueError("Invalid uninitialized self-play pool")
        elif type(selected) is not int or not 0 <= selected < len(snapshots):
            raise ValueError("Invalid selected self-play opponent")
        iterations = []
        for snapshot in snapshots:
            validate_module_state(
                self.policy.taskpolicy_opponent,
                snapshot["actor"],
                "pool actor",
            )
            validate_module_state(
                self.normalizers["actor_opponent"],
                snapshot["normalizer"],
                "pool statistics",
            )
            validate_statistics(snapshot["normalizer"])
            digest = state_sha256(
                {
                    **snapshot["actor"],
                    **{
                        "norm." + k: v
                        for k, v in snapshot["normalizer"].items()
                    },
                }
            )
            if snapshot["sha256"] != digest:
                raise ValueError("Self-play snapshot hash differs")
            iteration = snapshot["iteration"]
            if type(iteration) is not int or iteration < 0:
                raise ValueError("Invalid self-play snapshot iteration")
            iterations.append(iteration)
        if iterations != sorted(set(iterations)):
            raise ValueError("Self-play snapshot order differs")
        if type(state["next_refresh"]) is not int or state["next_refresh"] < 0:
            raise ValueError("Invalid self-play refresh boundary")
        torch.Generator().set_state(state["rng"])

    def load_state_dict(self, state):
        self.validate(state)
        if self.frozen:
            # Snapshot content is fixed by the checkpoint paths, which
            # ``validate`` just proved identical; only the draw can move, and
            # with one entry it does not.
            self.selected = (
                0 if state["selected"] is None else state["selected"]
            )
            self._install()
            return
        self.snapshots = copy.deepcopy(state["snapshots"])
        self.selected, self.next_refresh = (
            state["selected"],
            state["next_refresh"],
        )
        self.generator.set_state(state["rng"])
        if self.selected is not None:
            self._install()
