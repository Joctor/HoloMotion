"""Joint five-point task/prior/decoder, independently trained for both roles."""

import time
from contextlib import contextmanager

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from holoboxing.training.rsl_rl.modules.exploration import (
    HorizonNormal,
    conditional_distribution,
    unpack_context,
)
from holoboxing.training.rsl_rl.modules.normalization import (
    unpack_actor_observations,
)
from holoboxing.training.tasks.combat.config.sandbag import (
    ACTION_DIM,
    ACTOR_DIM,
    CRITIC_DIM,
    ROLES,
)


def mlp(width, output, hidden_dims):
    layers = []
    for hidden in hidden_dims:
        layers.extend((nn.Linear(width, hidden), nn.ELU()))
        width = hidden
    layers.append(nn.Linear(width, output))
    return nn.Sequential(*layers)


def _chunked_mean(actor, observations, rows):
    if (
        observations.ndim != 2
        or observations.shape[1] != actor.actor_dim
        or not len(observations)
    ):
        raise ValueError("Expected nonempty actor snapshots of the configured width")
    return torch.cat(
        [actor(chunk.contiguous()) for chunk in observations.split(rows)],
        dim=0,
    )


class _ActorMeanPair(nn.Module):
    """Capture a whole minibatch, not a repeatedly overwritten chunk buffer."""

    def __init__(self, actor, config):
        super().__init__()
        self.actor, self.config = actor, config

    def forward(self, observations):
        if self.config.exploration_rho:
            current, previous, _, _ = observations.split(
                (self.config.actor_dim, self.config.actor_dim, self.config.action_dim, 1), dim=-1
            )
            snapshots = (current, previous)
        else:
            snapshots = (observations,)
        return tuple(
            _chunked_mean(self.actor, obs, self.config.actor_forward_rows)
            for obs in snapshots
        )


class JointActor(nn.Module):
    def __init__(self, neutral_reference, config):
        super().__init__()
        neutral = torch.as_tensor(neutral_reference, dtype=torch.float32)
        if neutral.shape != (config.action_dim,) or not torch.isfinite(neutral).all():
            raise ValueError("Expected a finite nominal-pose action reference")
        self.actor_dim = config.actor_dim
        self.task_policy = mlp(123, 32, config.hidden_dims)
        self.prior = mlp(1080, 32, config.hidden_dims)
        self.decoder = mlp(108 + 32, config.action_dim, config.hidden_dims)
        # Small random final weights keep gradients flowing into all branches.
        with torch.no_grad():
            self.decoder[-1].weight.mul_(config.decoder_init_gain)
            self.decoder[-1].bias.mul_(config.decoder_init_gain)
        self.register_buffer("neutral_reference", neutral.detach().clone())
        self.register_buffer(
            "physical_scale",
            torch.tensor(
                [config.position_scale_m] * 3 + [config.rotation_scale] * 6
            ).repeat(25),
        )

    def forward(self, observations):
        current, task, history = unpack_actor_observations(observations)
        latent = F.normalize(
            self.task_policy(task) + self.prior(history), dim=-1, eps=1e-8
        )
        residual = self.decoder(torch.cat((current, latent), dim=-1))
        return self.neutral_reference + self.physical_scale * residual


class ActorCritic(nn.Module):
    """PPO likelihood is over action references, not frozen-decoder latents."""

    is_recurrent = False
    actor_dim = ACTOR_DIM
    action_dim = ACTION_DIM

    def __init__(self, neutral_reference, config):
        super().__init__()
        self.config = config
        self.actor_dim = config.actor_dim
        self.action_dim = config.action_dim
        self.actor_storage_dim = (
            2 * self.actor_dim + self.action_dim + 1 if config.exploration_rho else self.actor_dim
        )
        for role in ROLES:
            setattr(
                self,
                f"taskpolicy_{role}",
                JointActor(neutral_reference, config),
            )
            setattr(self, f"critic_{role}", mlp(CRITIC_DIM, 1, config.hidden_dims))
            setattr(
                self,
                f"std_{role}",
                nn.Parameter(
                    torch.full((self.action_dim,), config.init_noise_std).log()
                ),
            )
        self.distributions = {}
        self.base_distributions = {}
        # Plain runtime containers: no duplicate parameters or checkpoint keys.
        self._actor_graphs = {}
        self._actor_graphs_enabled = False
        self._graph_batch_roles = None
        self._graph_capture_counts = dict.fromkeys(ROLES, 0)
        self._graph_capture_seconds = 0.0

    def enable_actor_cuda_graphs(self):
        if self._graph_batch_roles is not None:
            raise RuntimeError("Cannot enable graphs during a PPO minibatch")
        if any(
            p.device.type != "cuda" or p.dtype != torch.float32
            for p in self.parameters()
        ):
            raise ValueError("Actor CUDA graphs require a CUDA float32 policy")
        self._actor_graphs_enabled = True

    def actor_cuda_graph_status(self):
        return {
            "enabled": self._actor_graphs_enabled,
            "captured_roles": sorted(self._actor_graphs),
            "capture_counts": dict(self._graph_capture_counts),
            "capture_seconds": self._graph_capture_seconds,
        }

    @contextmanager
    def actor_graph_batch(self):
        """One call per role, with backward completed before leaving this scope."""
        if self._graph_batch_roles is not None:
            raise RuntimeError("Nested actor graph minibatch")
        self._graph_batch_roles = set()
        try:
            yield
        finally:
            self._graph_batch_roles = None

    def _training_actor_means(self, role, observations):
        if role in self._graph_batch_roles:
            raise RuntimeError("Actor graph reused before minibatch backward")
        rows = (
            self.config.num_envs
            * self.config.num_steps_per_env
            // self.config.num_mini_batches
        )
        actor = getattr(self, f"taskpolicy_{role}")
        parameter = next(actor.parameters())
        if (
            observations.shape != (rows, self.actor_storage_dim)
            or observations.device != parameter.device
            or observations.dtype != torch.float32
            or observations.requires_grad
            or parameter.device.type != "cuda"
            or parameter.dtype != torch.float32
            or torch.is_autocast_enabled()
            or not actor.training
        ):
            raise ValueError("Actor graph input differs from its training contract")
        if role not in self._actor_graphs:
            # Cached eager laws can retain AccumulateGrad nodes from the default
            # stream, which cannot participate in a new side-stream capture.
            self.distributions.pop(role, None)
            self.base_distributions.pop(role, None)
            started = time.monotonic()
            module = _ActorMeanPair(actor, self.config)
            self._actor_graphs[role] = torch.cuda.make_graphed_callables(
                module, (observations.detach().clone(),), num_warmup_iters=3
            )
            self._graph_capture_counts[role] += 1
            self._graph_capture_seconds += time.monotonic() - started
        self._graph_batch_roles.add(role)
        return self._actor_graphs[role](observations)

    def _apply(self, fn, recurse=True):
        if self._graph_batch_roles is not None:
            raise RuntimeError("Cannot move a policy during a PPO minibatch")
        self.distributions.clear()
        self.base_distributions.clear()
        self._actor_graphs.clear()
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self._graph_batch_roles is not None:
            raise RuntimeError("Cannot load a policy during a PPO minibatch")
        self.distributions.clear()
        self.base_distributions.clear()
        self._actor_graphs.clear()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def validate_action_mapping(self, state_dict):
        expected = self.state_dict()
        for role in ROLES:
            for buffer in ("neutral_reference", "physical_scale"):
                name = f"taskpolicy_{role}.{buffer}"
                if not torch.equal(state_dict[name].to(expected[name]), expected[name]):
                    raise ValueError(
                        f"Checkpoint action execution mapping differs: {name}"
                    )

    def actor_mean(self, role, observations):
        return _chunked_mean(
            getattr(self, f"taskpolicy_{role}"),
            observations,
            self.config.actor_forward_rows,
        )

    def action_sigma(self, role):
        """The sigma the sampler actually multiplies, after every scaling.

        Three things sit between the ``std_*`` parameter and the noise on an
        action: ``exp``, the actor's ``physical_scale``, and the optional
        ceiling ``std_max``. A log that reads ``std_*.exp()`` therefore reports
        a number the rollout never used. That is not hypothetical: on the first
        ``--std-max 0.08`` probe the column read 0.2382 while the sampler was
        running at 0.08, and only the entropy column proved which was real
        (-61.98 = 32 x [0.5 ln(2 pi e) + ln(0.08 x 0.436)]). Everything that
        needs sigma -- the distribution and every metric -- goes through here.

        The 32-D latent is a unit vector, so sigma is measured in units of the
        signal: at 0.238 per dimension the noise norm is 1.35 against a signal
        norm of 1, i.e. roughly a 50 degree departure from the greedy direction
        every step.
        """
        std = getattr(self, f"std_{role}").exp().clamp(1e-4, 10)
        std = std * getattr(self, f"taskpolicy_{role}").physical_scale
        limit = getattr(self, "std_max", 0.0)
        return std.clamp(max=limit) if limit else std

    def distribution(self, role, observations):
        rho = self.config.exploration_rho
        if rho:
            current, previous, value, mode = unpack_context(
                observations, self.actor_dim, self.action_dim
            )
        else:
            current = observations
        graphed = (
            self._actor_graphs_enabled
            and self._graph_batch_roles is not None
            and self.training
            and torch.is_grad_enabled()
        )
        means = self._training_actor_means(role, observations) if graphed else None
        mean = means[0] if graphed else self.actor_mean(role, current)
        std = self.action_sigma(role)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError(f"Nonfinite {role} action distribution")
        base = Normal(mean, std.expand_as(mean))
        self.base_distributions[role] = base
        conditional = (
            conditional_distribution(
                base,
                means[1] if graphed else self.actor_mean(role, previous),
                value,
                mode,
                rho,
            )
            if rho
            else base
        )
        law = (
            HorizonNormal(conditional, self.config.horizon_rho)
            if self.config.horizon_rho
            else conditional
        )
        self.distributions[role] = law
        return law

    def value(self, role, observations):
        return getattr(self, f"critic_{role}")(observations)

    def role_parameters(self, role):
        return [
            *getattr(self, f"taskpolicy_{role}").parameters(),
            *getattr(self, f"critic_{role}").parameters(),
            getattr(self, f"std_{role}"),
        ]

    def role_parameter_parts(self, role):
        """The role's optimiser parameters split by the loss that moves them.

        ``role_parameters`` is one flat list on purpose: the optimiser counts
        it as a unit and the strict resume check demands a moment for every
        entry. A critic-only warm-up breaks the assumption hiding in that
        unit. With the policy loss out of the objective the actor half gets no
        gradient, so ``Adam.step`` does not advance for it, while the critic
        half and the shared step counter do. The moments then disagree with
        the learner progress by exactly the length of the warm-up, and only a
        partition of the list can say which tensors are allowed to lag.

        Sigma belongs to the actor half either way: the entropy bonus is its
        only gradient, and that term is dropped with the surrogate.
        """
        return (
            [
                *getattr(self, f"taskpolicy_{role}").parameters(),
                getattr(self, f"std_{role}"),
            ],
            list(getattr(self, f"critic_{role}").parameters()),
        )
