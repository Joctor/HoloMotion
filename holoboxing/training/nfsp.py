"""Replay support for latent-space Neural Fictitious Self-Play."""

import torch


class LatentReservoir:
    """Uniform fixed-capacity reservoir of raw observations and unit latents."""

    def __init__(
        self,
        capacity,
        *,
        observation_dim=1097,
        latent_dim=32,
        seed=0,
    ):
        if type(capacity) is not int or capacity < 2:
            raise ValueError("NFSP reservoir capacity must be at least two")
        if (observation_dim, latent_dim) != (1097, 32):
            raise ValueError("Expected the boxing observation and latent widths")
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.observations = torch.empty(capacity, observation_dim)
        self.latents = torch.empty(capacity, latent_dim)
        self.size = 0
        self.seen = 0
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def add(self, observations, latents):
        observations = torch.as_tensor(observations).detach().to(
            device="cpu", dtype=torch.float32
        )
        latents = torch.as_tensor(latents).detach().to(
            device="cpu", dtype=torch.float32
        )
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError("Expected packed raw NFSP observations")
        if latents.shape != (len(observations), self.latent_dim):
            raise ValueError("NFSP observations and latents must align")
        if not torch.isfinite(observations).all() or not torch.isfinite(
            latents
        ).all():
            raise ValueError("NFSP reservoir samples must be finite")
        if len(observations) == 0:
            return
        lengths = torch.linalg.vector_norm(latents, dim=-1)
        if not torch.allclose(lengths, torch.ones_like(lengths), atol=1e-4, rtol=0):
            raise ValueError("NFSP targets must be unit-sphere latents")

        old_seen = self.seen
        filled = min(self.capacity - self.size, len(observations))
        if filled:
            end = self.size + filled
            self.observations[self.size : end].copy_(observations[:filled])
            self.latents[self.size : end].copy_(latents[:filled])
            self.size = end
            self.seen += filled

        remaining = len(observations) - filled
        if remaining:
            candidates = observations[filled:]
            candidate_latents = latents[filled:]
            upper = torch.arange(
                old_seen + filled + 1,
                old_seen + len(observations) + 1,
                dtype=torch.float64,
            )
            chosen = torch.floor(
                torch.rand(
                    remaining,
                    dtype=torch.float64,
                    generator=self.generator,
                )
                * upper
            ).to(torch.long)
            accepted = chosen < self.capacity
            slots = chosen[accepted]
            if slots.numel():
                source = accepted.nonzero().flatten()
                order = torch.argsort(slots, stable=True)
                slots, source = slots[order], source[order]
                last = torch.ones_like(slots, dtype=torch.bool)
                last[:-1] = slots[:-1] != slots[1:]
                slots, source = slots[last], source[last]
                self.observations[slots] = candidates[source]
                self.latents[slots] = candidate_latents[source]
            self.seen += remaining

    def sample(self, batch_size, *, device):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("NFSP sample batch size must be positive")
        if self.size == 0:
            raise ValueError("Cannot sample an empty NFSP reservoir")
        indices = torch.randint(
            self.size, (batch_size,), generator=self.generator
        )
        return (
            self.observations.index_select(0, indices).to(device),
            self.latents.index_select(0, indices).to(device),
        )

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "latent_dim": self.latent_dim,
            "size": self.size,
            "seen": self.seen,
            "observations": self.observations[: self.size].clone(),
            "latents": self.latents[: self.size].clone(),
            "rng": self.generator.get_state(),
        }

    def validate(self, state):
        if set(state) != {
            "capacity",
            "observation_dim",
            "latent_dim",
            "size",
            "seen",
            "observations",
            "latents",
            "rng",
        }:
            raise ValueError("Invalid NFSP reservoir checkpoint fields")
        size, seen = state["size"], state["seen"]
        if (
            state["capacity"] != self.capacity
            or state["observation_dim"] != self.observation_dim
            or state["latent_dim"] != self.latent_dim
            or type(size) is not int
            or not 0 <= size <= self.capacity
            or type(seen) is not int
            or seen < size
            or (size < self.capacity and seen != size)
        ):
            raise ValueError("NFSP reservoir recipe or size differs")
        observations, latents = state["observations"], state["latents"]
        if (
            observations.shape != (size, self.observation_dim)
            or latents.shape != (size, self.latent_dim)
            or observations.dtype != torch.float32
            or latents.dtype != torch.float32
            or not torch.isfinite(observations).all()
            or not torch.isfinite(latents).all()
        ):
            raise ValueError("Invalid saved NFSP reservoir samples")
        if size and not torch.allclose(
            torch.linalg.vector_norm(latents, dim=-1),
            torch.ones(size),
            atol=1e-4,
            rtol=0,
        ):
            raise ValueError("Saved NFSP targets are not unit latents")
        torch.Generator(device="cpu").set_state(state["rng"])

    def load_state_dict(self, state):
        self.validate(state)
        self.size, self.seen = state["size"], state["seen"]
        self.observations[: self.size].copy_(state["observations"])
        self.latents[: self.size].copy_(state["latents"])
        self.generator.set_state(state["rng"])
