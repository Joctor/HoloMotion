"""Rule-exact score keeping for competition-aligned boxing training.

The official referee awards one point per 5 N*s of measured normal impulse on a
legal hand within a rolling 20 ms window, then blocks that hand for 200 ms. A
policy trained on a looser proxy (any contact above 1 N) learns to *hold* its
fist against the torso, which is only reachable by leaning forward -- the main
source of self-inflicted falls.

``CompetitionPhysics.paired_history`` already stores, per physics substep, the
wrist-to-torso force for each (defender, hand) slot. The control period equals
the 20 ms window, so the four freshest substeps *are* that window and the
impulse can be reconstructed without touching any physics kernel.

The accumulated vector is the resultant of all wrist/torso contacts in the slot
rather than the sum of per-contact normal magnitudes, so highly oblique grazes
are scored slightly lower than the referee would. The gate, window, threshold
and cooldown are identical, which is what matters for behaviour.
"""

from __future__ import annotations

import torch

from holoboxing.evaluation.runtime.impulse_hit_rules import (
    COOLDOWN_SECONDS,
    PHYSICS_DT,
    THRESHOLD_NS,
    WINDOW_SECONDS,
)
from holoboxing.training.tasks.combat.config.sandbag import AGENTS, DT

SUBSTEPS = int(round(WINDOW_SECONDS / PHYSICS_DT))
COOLDOWN_STEPS = max(1, int(round(COOLDOWN_SECONDS / DT)))
ROLES = len(AGENTS)
HANDS = 2


class RuleScorer:
    """Batched referee: points, per-step scoring mask and 20 ms impulse window."""

    def __init__(self, env):
        if SUBSTEPS != int(round(DT / PHYSICS_DT)):
            raise ValueError(
                "Control period must equal the official impulse window"
            )
        self.num_envs = env.num_envs
        self.device = env.device
        self.physics = env.physics
        worlds = self.num_envs
        # Steps remaining before a hand may score again (official 200 ms).
        self.cooldown = torch.zeros(
            (worlds, ROLES, HANDS), dtype=torch.long, device=self.device
        )
        # Cumulative points per role, used for logging and post-round comparison.
        self.points = torch.zeros((ROLES, worlds), device=self.device)
        # Points awarded on the current control step.
        self.step_points = torch.zeros((ROLES, worlds), device=self.device)
        # Which (role, hand) pair just scored, for the retraction reward.
        self.scored = torch.zeros(
            (worlds, ROLES, HANDS), dtype=torch.bool, device=self.device
        )
        # Last 20 ms impulse in N*s, exposed as dense punch-quality shaping.
        self.impulse = torch.zeros((worlds, ROLES, HANDS), device=self.device)

    @torch.no_grad()
    def step(self):
        """Advance one control step of scoring; call after the physics substeps."""
        history = self.physics.paired_history
        if history.shape[1] < SUBSTEPS:
            raise ValueError(
                "Impulse history is shorter than the official window"
            )
        # |force| per substep, summed over the window, integrated at 5 ms.
        # The history is indexed by the *defender* whose torso is touched, so
        # flip that axis to index points by the fighter who threw the hand.
        impulse = (
            history[:, :SUBSTEPS].norm(dim=-1).sum(dim=1) * PHYSICS_DT
        ).flip(1)
        self.impulse.copy_(impulse)
        self.cooldown.sub_(1).clamp_(min=0)
        scored = (impulse >= THRESHOLD_NS) & (self.cooldown == 0)
        self.scored.copy_(scored)
        self.cooldown[scored] = COOLDOWN_STEPS
        self.step_points.copy_(
            scored.sum(dim=-1).transpose(0, 1).to(torch.float32)
        )
        self.points.add_(self.step_points)
        return self.points

    @torch.no_grad()
    def reset(self, ids):
        if not len(ids):
            return
        self.cooldown[ids] = 0
        self.points[:, ids] = 0
        self.step_points[:, ids] = 0
        self.scored[ids] = False
        self.impulse[ids] = 0
