# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause
"""Paired feedforward PPO rollouts and AMP transitions."""

import torch

from holoboxing.training.tasks.combat.config.sandbag import ROLES


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.clear()

        def clear(self):
            self.valid = None
            self.actor_mask = None   # 新增
            for role in ROLES:
                for name in (
                    "observations",
                    "privileged_observations",
                    "actions",
                    "rewards",
                    "dones",
                    "values",
                    "actions_log_prob",
                    "action_mean",
                    "action_sigma",
                    "amp_obs",
                ):
                    setattr(self, f"{name}_{role}", None)

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        device="cpu",
        *,
        active_roles=ROLES,
    ):
        self.actor_mask = {
            role: torch.ones(
                num_transitions_per_env, num_envs, dtype=torch.bool, device=device
            )
            for role in active_roles
        }
        self.device = device
        self.active_roles = tuple(active_roles)
        if self.active_roles not in (ROLES, ("ego",)):
            raise ValueError("Expected paired learning or one ego learner")
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.step = 0
        self.has_invalid = False
        self.valid = torch.ones(
            num_transitions_per_env, num_envs, dtype=torch.bool, device=device
        )
        for role in self.active_roles:
            for name, shape in (
                ("observations", obs_shape),
                ("privileged_observations", privileged_obs_shape),
                ("actions", actions_shape),
                ("mu", actions_shape),
                ("sigma", actions_shape),
                ("rewards", [1]),
                ("dones", [1]),
                ("values", [1]),
                ("actions_log_prob", [1]),
                ("returns", [1]),
                ("advantages", [1]),
            ):
                value = torch.zeros(
                    num_transitions_per_env,
                    num_envs,
                    *shape,
                    device=device,
                    dtype=torch.uint8 if name == "dones" else torch.float32,
                )
                setattr(self, f"{name}_{role}", value)
            setattr(self, f"amp_obs_{role}", None)

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError(
                "Rollout buffer overflow; update before collecting again"
            )
        if transition.valid is None:
            self.valid[self.step].fill_(True)
        else:
            self.valid[self.step].copy_(transition.valid)
            self.has_invalid |= not bool(transition.valid.all())

        for role in self.active_roles:
            if (
                transition.actor_mask is not None
                and role in transition.actor_mask
            ):
                mask = transition.actor_mask[role]
                if mask.dtype != torch.bool or mask.shape != (self.num_envs,):
                    raise ValueError(
                        f"actor_mask[{role}] must be bool [num_envs]"
                    )
                self.actor_mask[role][self.step].copy_(mask)
            else:
                self.actor_mask[role][self.step].fill_(True)

        for role in self.active_roles:
            for name in (
                "observations",
                "privileged_observations",
                "actions",
                "rewards",
                "dones",
                "values",
                "actions_log_prob",
                "mu",
                "sigma",
            ):
                source = {"mu": "action_mean", "sigma": "action_sigma"}.get(
                    name, name
                )
                value = getattr(transition, f"{source}_{role}")
                if name in ("rewards", "dones", "actions_log_prob"):
                    value = value.view(-1, 1)
                getattr(self, f"{name}_{role}")[self.step].copy_(value)
            amp = getattr(transition, f"amp_obs_{role}")
            buffer = getattr(self, f"amp_obs_{role}")
            if buffer is None:
                buffer = torch.zeros(
                    self.num_transitions_per_env,
                    self.num_envs,
                    *amp.shape[1:],
                    device=self.device,
                )
                setattr(self, f"amp_obs_{role}", buffer)
            buffer[self.step].copy_(amp)
        self.step += 1

    def clear(self):
        self.step = 0
        self.has_invalid = False

    def compute_returns(self, role, last_values, gamma, lam):
        values = getattr(self, f"values_{role}")
        rewards = getattr(self, f"rewards_{role}")
        dones = getattr(self, f"dones_{role}")
        returns = getattr(self, f"returns_{role}")
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = (
                last_values
                if step == self.num_transitions_per_env - 1
                else values[step + 1]
            )
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = (
                rewards[step]
                + next_is_not_terminal * gamma * next_values
                - values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # GAE 对所有 valid 样本传播；critic 需要这些 return
            active = self.valid[step, :, None]
            advantage = torch.where(active, advantage, 0.0)
            returns[step] = advantage + values[step]
        advantages = returns - values
        # actor 的 advantage 只在 valid & actor_mask 上归一化
        actor_mask = self.actor_mask[role]
        actor_selected = advantages[self.valid & actor_mask]
        if len(actor_selected) < 2:
            raise FloatingPointError("Not enough valid transitions for PPO")
        advantages = (
            advantages - actor_selected.mean()
        ) / (actor_selected.std() + 1e-8)
        # 非 actor 样本的 advantage 置 0，不参与 actor loss
        advantages[~(self.valid & actor_mask)] = 0
        setattr(self, f"advantages_{role}", advantages)

    def valid_flattened(self, name):
        values = getattr(self, name).flatten(0, 1)
        return values[self.valid.flatten()] if self.has_invalid else values

    def mini_batch_generator(self, num_mini_batches, num_epochs):
        has_filtered = self.has_invalid or any(
            not bool(self.actor_mask[role].all())
            for role in self.active_roles
        )
        actor_selections = {}
        critic_selections = {}
        for role in self.active_roles:
            actor_mask = self.actor_mask[role]
            selected = self.valid & actor_mask
            if has_filtered:
                actor_indices = selected.flatten().nonzero().flatten()
                critic_indices = self.valid.flatten().nonzero().flatten()
            else:
                actor_indices = None
                critic_indices = None

            actor_size = (
                len(actor_indices)
                if actor_indices is not None
                else self.num_envs * self.num_transitions_per_env
            )
            critic_size = (
                len(critic_indices)
                if critic_indices is not None
                else self.num_envs * self.num_transitions_per_env
            )
            actor_mini = actor_size // num_mini_batches
            critic_mini = critic_size // num_mini_batches
            if actor_mini < 1 or critic_mini < 1:
                raise FloatingPointError(
                    f"Not enough transitions for PPO batches: {role}"
                )
            actor_size = num_mini_batches * actor_mini
            critic_size = num_mini_batches * critic_mini

            actor_order = torch.randperm(actor_size, device=self.device)
            critic_order = torch.randperm(critic_size, device=self.device)
            if actor_indices is not None:
                actor_order = actor_indices[actor_order]
                critic_order = critic_indices[critic_order]
            actor_selections[role] = actor_order.tensor_split(num_mini_batches)
            critic_selections[role] = critic_order.tensor_split(num_mini_batches)

        actor_columns = (
            "observations", "privileged_observations", "actions",
            "values", "advantages", "returns", "actions_log_prob",
            "mu", "sigma",
        )
        critic_columns = (
            "privileged_observations", "values", "returns",
        )
        actor_batches = {
            role: tuple(
                getattr(self, f"{name}_{role}").flatten(0, 1)
                for name in actor_columns
            )
            for role in self.active_roles
        }
        critic_batches = {
            role: tuple(
                getattr(self, f"{name}_{role}").flatten(0, 1)
                for name in critic_columns
            )
            for role in self.active_roles
        }

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                yield (
                    tuple(
                        tuple(
                            actor_batches[role][j][actor_selections[role][i]]
                            for j in range(len(actor_columns))
                        )
                        for role in self.active_roles
                    ),
                    tuple(
                        tuple(
                            critic_batches[role][j][critic_selections[role][i]]
                            for j in range(len(critic_columns))
                        )
                        for role in self.active_roles
                    ),
                )