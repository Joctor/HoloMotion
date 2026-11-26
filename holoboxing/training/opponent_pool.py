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
    def __init__(self, policy, normalizers, *, seed, capacity=8, interval=256):
        if (
            type(capacity) is not int
            or capacity < 2
            or type(interval) is not int
            or interval < 1
        ):
            raise ValueError(
                "Self-play needs a positive interval and at least two slots"
            )
        self.policy, self.normalizers = policy, normalizers
        self.capacity, self.interval = capacity, interval
        self.generator = torch.Generator().manual_seed(seed)
        self.snapshots = []
        self.selected = None
        self.next_refresh = 0

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
        if (
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
        self.snapshots = copy.deepcopy(state["snapshots"])
        self.selected, self.next_refresh = (
            state["selected"],
            state["next_refresh"],
        )
        self.generator.set_state(state["rng"])
        if self.selected is not None:
            self._install()
