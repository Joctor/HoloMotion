from __future__ import annotations
from typing import List

import torch
from holoboxing.training.envs.entities import SceneEntityCfg
from holoboxing.training.tasks.combat.mdp.geometry import (
    quat_apply_wxyz as quat_apply,
    quat_conjugate_wxyz,
)


def quat_apply_inverse(q, v):
    return quat_apply(quat_conjugate_wxyz(q), v)


def _contact_body_names(sensor):
    # Body ordering is fixed for the lifetime of each sensor.
    names = getattr(sensor, "_holoboxing_body_names", None)
    if names is None:
        names = tuple(sensor.body_names)
        sensor._holoboxing_body_names = names
    return names


def paired_contact_forces(env, attacker, defender, attacker_sensor, defender_sensor):
    """Use named wrist filters, never coincident unrelated net contacts."""
    sensors = env.scene.sensors
    force = sensors[f"contact_sensor_{defender}_torso"].data.force_matrix_w_history[
        :, :, 0
    ]
    if force.shape[2:] != (2, 3):
        raise ValueError("Expected paired wrist-to-torso force history")
    attack_sensor, defend_sensor = (
        sensors[attacker_sensor],
        sensors[defender_sensor],
    )
    attack = torch.zeros_like(attack_sensor.data.net_forces_w_history)
    defend = torch.zeros_like(defend_sensor.data.net_forces_w_history)
    for i, name in enumerate(("left_wrist_yaw_link", "right_wrist_yaw_link")):
        attack[:, :, _contact_body_names(attack_sensor).index(name)] = force[:, :, i]
    defend[:, :, _contact_body_names(defend_sensor).index("torso_link"), 2] = (
        force.norm(dim=-1).sum(dim=-1)
    )
    return attack, defend

def windowed_impulse(env, defender, window_steps=4, phys_dt=0.005):
    """返回 [B, 2]：对方左手/右手在最近 20 ms 内对本方躯干的总接触力积分。

    paired_history 按物理步（5 ms）滚动，取前 4 步 = 20 ms。
    注意：physics 层存的力包含摩擦分量；拳击接触中法向力主导，
    用总力模长近似法向冲量，阈值相应上调（见 terms）。
    """
    force = env.scene.sensors[f"contact_sensor_{defender}_torso"].data \
        .force_matrix_w_history                    # [B, H, 1, 2, 3]
    if force.shape[2:] != (1, 2, 3):
        raise ValueError("Expected paired wrist-to-torso force history")
    w = min(window_steps, force.shape[1])
    per_hand = force[:, :w, 0].norm(dim=-1)        # [B, W, 2]
    return per_hand.sum(dim=1) * phys_dt           # [B, 2]


def _hand_scored(env, agent, impulse, threshold, cooldown_steps):
    """impulse [B,2]，返回 [B,2] bool；原地更新该 agent 每只手的冷却时间戳。"""
    if not hasattr(env, "punch_state"):
        env.punch_state = {}
    state = env.punch_state.setdefault(agent, {})
    if "last_score_step" not in state:
        state["last_score_step"] = torch.full(
            (env.num_envs, 2), -10**9, dtype=torch.long, device=env.device
        )
    last = state["last_score_step"]
    step = env.common_step_counter
    ready = (step - last) >= cooldown_steps
    scored = (impulse >= threshold) & ready
    last[scored] = step
    return scored


def reset_punch_state(env, env_ids):
    for state in getattr(env, "punch_state", {}).values():
        for name, value in state.items():
            value[env_ids] = -10**9 if "score_step" in name else 0


def custom_action_rate_l2(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    action_manager = (
        env.action_managers[asset_cfg.name]
        if hasattr(env, "action_managers")
        else env.action_manager
    )
    return torch.sum(
        torch.square(action_manager.action - action_manager.prev_action), dim=1
    )


def stand_still_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.5,
    positive: bool = False,
) -> torch.Tensor:
    ego_agent = env.scene[asset_cfg.name]
    torso_idx = ego_agent.body_names.index("torso_link")
    lin_vel = ego_agent.data.body_vel_w[:, torso_idx][..., :3]
    speed_sq = torch.sum(torch.square(lin_vel[..., :2]), dim=-1)
    return torch.exp(-speed_sq / sigma) if positive else -speed_sq


def offensive_hit_reward_punch(
    env,
    sensor_cfg1: SceneEntityCfg,
    sensor_cfg2: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    impulse_threshold: float = 5.5,
    cooldown_steps: int = 10,
    **_ignored,
) -> torch.Tensor:
    ego = asset_cfg.name
    opponent = env.get_opponent_id(ego)
    impulse = windowed_impulse(env, defender=opponent)   # 本方手打对方躯干
    scored = _hand_scored(env, ego, impulse, impulse_threshold, cooldown_steps)
    return scored.any(dim=-1).float()

def defensive_hit_penalty_punch(
    env, sensor_cfg1, sensor_cfg2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    impulse_threshold: float = 5.5,
    cooldown_steps: int = 10,
    **_ignored,
) -> torch.Tensor:
    ego = asset_cfg.name
    opponent = env.get_opponent_id(ego)
    impulse = windowed_impulse(env, defender=ego)
    scored = _hand_scored(
        env, f"{opponent}_defensive", impulse, impulse_threshold, cooldown_steps
    )
    return scored.any(dim=-1).float()

def balance_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_threshold: float = 0.7,
    with_angular: bool = False,
    is_ego: bool = True,
) -> torch.Tensor:
    """Balance reward for staying upright and stable."""
    if is_ego:
        ego_agent = env.scene[asset_cfg.name]
    else:
        ego_agent = env.get_opponent(asset_cfg.name)

    # Torso uprightness: dot product between the z axis and gravity direction.
    torso_idx = ego_agent.body_names.index("torso_link")
    torso_quat = ego_agent.data.body_quat_w[:, torso_idx]
    up_vector = torch.tensor([0, 0, 1], dtype=torch.float, device=env.device).repeat(
        torso_quat.shape[0], 1
    )

    # Extract orientation from the quaternion.
    torso_up = quat_apply(torso_quat, up_vector)
    uprightness = torso_up[:, 2]  # z component.

    # Upright reward.
    upright_reward = torch.where(
        uprightness > upright_threshold,
        uprightness,
        torch.zeros_like(uprightness),
    )

    # Stability reward; smaller angular velocity is better.
    torso_ang_vel = ego_agent.data.body_ang_vel_w[:, torso_idx]
    if with_angular:
        angular_vel_norm = torch.norm(torso_ang_vel, dim=-1)
        stability_reward = torch.exp(-angular_vel_norm)
        return upright_reward + stability_reward
    else:
        return upright_reward


def distance_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_dist: float = 1.5,
    max_dist: float = 2.0,
    scale: float = 0.075,
) -> torch.Tensor:
    ego_agent = env.scene[asset_cfg.name]
    opponent_agent = env.get_opponent(asset_cfg.name)
    torso_index = ego_agent.body_names.index("torso_link")
    ego_agent_torso_pos = ego_agent.data.body_pos_w[:, torso_index : torso_index + 1]
    opponent_agent_torso_pos = opponent_agent.data.body_pos_w[
        :, torso_index : torso_index + 1
    ]
    error = torch.sum(
        torch.square(ego_agent_torso_pos - opponent_agent_torso_pos), dim=-1
    )
    distance = torch.sqrt(error[..., :2])
    midpoint = (min_dist + max_dist) / 2.0
    sigmoid_input = (distance - midpoint) / scale
    reward = -torch.sigmoid(sigmoid_input)
    return reward.mean(-1)

    # return torch.clamp(projected_vel, max=2.0) NOTE fast vel towards opponent is not always good


def energy_saving_penalty(
    env,
    joint_names: List[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.7,
    std: float = 0.15,
) -> torch.Tensor:
    """Energy-saving penalty that encourages efficient energy use."""
    ego_agent = env.scene[asset_cfg.name]

    joint_acceleration = ego_agent.data.joint_acc[
        :, [ego_agent.joint_names.index(name) for name in joint_names]
    ]

    opp_agent = env.get_opponent(asset_cfg.name)

    ego_torso_idx = ego_agent.body_names.index("torso_link")
    opp_torso_idx = opp_agent.body_names.index("torso_link")
    distance_vec = (
        ego_agent.data.body_pos_w[:, ego_torso_idx]
        - opp_agent.data.body_pos_w[:, opp_torso_idx]
    )
    distance_vec[:, 2] = 0.0  # ignore z axis
    distance = torch.norm(distance_vec, dim=-1)  # N
    # import pdb; pdb.set_trace()
    k_distance = 1 - torch.exp(
        -((distance - distance_threshold).clip(min=0.0)) / (std**2)
    )
    return torch.sum(torch.square(joint_acceleration), dim=-1) * k_distance


# 1208: SMPL OLYMPIC REWARDS:
def facing_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_body_name: str = "torso_link",
    sigma: float = 0.5,
) -> torch.Tensor:
    ego_agent = env.scene[asset_cfg.name]
    opponent_agent = env.get_opponent(asset_cfg.name)

    ego_root_pos = ego_agent.data.root_pos_w
    ego_root_quat = ego_agent.data.root_quat_w
    opponent_target_idx = opponent_agent.body_names.index(target_body_name)
    opponent_target_pos = opponent_agent.data.body_pos_w[:, opponent_target_idx]

    tar_dir_3d = opponent_target_pos - ego_root_pos
    tar_dir_3d[:, 2] = 0.0
    tar_dir_3d = torch.nn.functional.normalize(tar_dir_3d, dim=-1)
    tar_dir_local = quat_apply_inverse(ego_root_quat, tar_dir_3d)
    rew = torch.exp(-(1 - tar_dir_local[:, 0]) / sigma)
    return rew


def vel_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_body_name: str = "torso_link",
    sigma: float = 0.25,
    tar_speed: float = 1.0,
) -> torch.Tensor:
    ego_agent = env.scene[asset_cfg.name]
    opponent_agent = env.get_opponent(asset_cfg.name)

    ego_root_pos = ego_agent.data.root_pos_w
    ego_root_vel = ego_agent.data.root_lin_vel_w
    opponent_target_idx = opponent_agent.body_names.index(target_body_name)
    opponent_target_pos = opponent_agent.data.body_pos_w[:, opponent_target_idx]

    tar_dir_2d = opponent_target_pos[..., :2] - ego_root_pos[..., :2]
    tar_dir_2d = torch.nn.functional.normalize(tar_dir_2d, dim=-1)

    tar_dir_speed = torch.sum(ego_root_vel[..., :2] * tar_dir_2d, dim=-1)

    speed_mask = tar_dir_speed <= 0
    target_vel_err = tar_speed - tar_dir_speed
    target_vel_err = torch.clamp_min(target_vel_err, min=0.0)
    rew = torch.exp(-torch.square(target_vel_err) / sigma)
    rew[speed_mask] = 0.0
    return rew


def strike_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg_ego: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    sensor_cfg_opponent: SceneEntityCfg = SceneEntityCfg("contact_sensor_opponent"),
    target_body_weights: list[float] = [1.0],
) -> torch.Tensor:
    ego = asset_cfg.name
    opponent = env.get_opponent_id(ego)
    forces = []
    for agent in (ego, opponent):
        force = env.scene.sensors[f"contact_sensor_{agent}_torso"].data.force_matrix_w
        if force.shape[1:] != (1, 2, 3):
            raise ValueError("Expected two filtered wrist-to-torso forces")
        forces.append(force[:, 0].sum(dim=1).norm(dim=-1))
    if len(target_body_weights) != 1:
        raise ValueError("The paired strike target is the torso")
    return (forces[1] - forces[0]) * target_body_weights[0]


def hit_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hit_body_names: list[str] = [
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ],
    target_body_names: list[str] = ["torso_link"],
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    hand-target-dis
    """
    ego_agent = env.scene[asset_cfg.name]
    opponent_agent = env.get_opponent(asset_cfg.name)

    ego_body_ids = [ego_agent.body_names.index(name) for name in hit_body_names]
    target_body_ids = [
        opponent_agent.body_names.index(name) for name in target_body_names
    ]
    ego_wrist_pos = ego_agent.data.body_pos_w[:, ego_body_ids]  # [B, 2, 3]
    opponent_target_pos = opponent_agent.data.body_pos_w[
        :, target_body_ids
    ]  # [B, T, 3]

    hit_dist = ego_wrist_pos.unsqueeze(2) - opponent_target_pos.unsqueeze(
        1
    )  # [B, 2, T, 3]
    hit_dist = torch.norm(hit_dist, dim=-1).reshape(hit_dist.shape[0], -1)  # [B, 2*T]
    rew = torch.exp(-torch.mean(hit_dist, dim=-1) / sigma)
    return rew


def active_hit_shaping_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hit_body_names: list[str] = [
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ],
    target_body_names: list[str] = ["torso_link"],
    sigma: float = 1.0,
    vel_threshold: float = 0.8,
    gate_temperature: float = 10.0,
) -> torch.Tensor:
    """Distance shaping gated by *self* punching speed towards the opponent.

    Motivation:
    In self-play, if the opponent actively approaches, a pure distance-based reward can
    become "free reward" for the learner even without taking meaningful actions. This
    function prevents that by requiring the agent's wrists to move towards the opponent
    target(s) with sufficient relative speed (w.r.t. the agent torso).

    Returns:
        reward: [B] in [0, 1] approximately.
    """
    ego_agent = env.scene[asset_cfg.name]
    opponent_agent = env.get_opponent(asset_cfg.name)

    torso_idx = ego_agent.body_names.index("torso_link")
    ego_torso_vel = ego_agent.data.body_vel_w[:, torso_idx, :3]  # [B, 3]

    ego_body_ids = [ego_agent.body_names.index(name) for name in hit_body_names]
    target_body_ids = [
        opponent_agent.body_names.index(name) for name in target_body_names
    ]

    wrist_pos = ego_agent.data.body_pos_w[:, ego_body_ids, :3]  # [B, H, 3]
    wrist_vel = (
        ego_agent.data.body_vel_w[:, ego_body_ids, :3] - ego_torso_vel[:, None, :]
    )

    targets_pos = opponent_agent.data.body_pos_w[:, target_body_ids, :3]  # [B, T, 3]

    # Vector from wrist -> target
    diff = targets_pos.unsqueeze(1) - wrist_pos.unsqueeze(2)  # [B, H, T, 3]
    dist = torch.norm(diff, dim=-1)  # [B, H, T]
    min_dist = torch.min(dist, dim=-1)[0]  # [B, H]

    dir_norm = diff / (torch.norm(diff, dim=-1, keepdim=True) + 1e-6)  # [B, H, T, 3]
    # Project wrist relative velocity onto directions towards targets
    proj_speed = torch.sum(wrist_vel.unsqueeze(2) * dir_norm, dim=-1)  # [B, H, T]
    best_proj_speed = torch.max(proj_speed, dim=-1)[0]  # [B, H]

    # Smooth gate: ~0 when speed below threshold, ~1 when above.
    gate = torch.sigmoid((best_proj_speed - vel_threshold) * gate_temperature)  # [B, H]

    # Distance shaping (per hand), gated by active punching motion.
    hand_reward = torch.exp(-min_dist / sigma) * gate  # [B, H]
    return hand_reward.mean(dim=-1)


def terminate_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    minimum_height: float = 0.4,
) -> torch.Tensor:
    # Based on height:

    ego_agent = env.scene[asset_cfg.name]
    opponent_agent = env.get_opponent(asset_cfg.name)  #

    ego_fallen = ego_agent.data.root_pos_w[:, 2] < minimum_height
    opponent_fallen = opponent_agent.data.root_pos_w[:, 2] < minimum_height
    rew = opponent_fallen.float() - ego_fallen.float()
    return rew


def complete_punch_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_steps: int = 40,
    v_start_thresh: float = 0.75,
    base_reward: float = 1.0,
    retract_ratio: float = 0.3,
    dist_rescale: bool = True,
    min_dist: float = 0.5,
    max_dist: float = 1.0,
    scale: float = 0.04,
    penalty_shift: float = 1.01,
    # return_debug: bool = False,
    # ) -> torch.Tensor | tuple[torch.Tensor, dict]:
) -> torch.Tensor:
    """Sparse reward for completing a fast out-and-back punch (velocity-based).

    This version implements a speed-window definition (per hand) on the XY plane:

    - We compute XY radial velocity relative to torso:
      ``v_r_xy = dot((v_wrist_xy - v_torso_xy), r_hat_xy)``.
      Positive means moving away from torso (punching out), negative means
      moving towards torso (retracting).

    - Start: ``v_r_xy > v_start_thresh``.
    - During the same time window (<= ``max_steps`` after start), track
      ``max_pos_speed = max(v_r_xy)``.
    - Complete: first time we see retracting strong enough:
      ``(-v_r_xy) >= retract_ratio * max_pos_speed``.

    Reward is emitted once at completion step and then the hand state resets.

    Notes:
    - ``guard_radius`` and ``min_extension`` are kept for backwards
      compatibility and debugging plots, but are not used in the completion
      condition in this speed-based definition.
    """

    device = env.device
    num_envs = env.num_envs

    ego_agent = env.scene[asset_cfg.name]

    torso_idx = ego_agent.body_names.index("torso_link")
    l_wrist_idx = ego_agent.body_names.index("left_wrist_yaw_link")
    r_wrist_idx = ego_agent.body_names.index("right_wrist_yaw_link")

    torso_pos = ego_agent.data.body_pos_w[:, torso_idx, :3]
    torso_vel = ego_agent.data.body_vel_w[:, torso_idx, :3]
    if dist_rescale:
        distance_pen = (
            distance_penalty(
                env,
                asset_cfg=asset_cfg,
                min_dist=min_dist,
                max_dist=max_dist,
                scale=scale,
            )
            + penalty_shift
        )

    l_pos = ego_agent.data.body_pos_w[:, l_wrist_idx, :3]
    l_vel = ego_agent.data.body_vel_w[:, l_wrist_idx, :3]
    r_pos = ego_agent.data.body_pos_w[:, r_wrist_idx, :3]
    r_vel = ego_agent.data.body_vel_w[:, r_wrist_idx, :3]

    # Radial vectors (XY plane)
    l_vec = l_pos - torso_pos
    r_vec = r_pos - torso_pos
    l_vec_xy = l_vec[:, :2]
    r_vec_xy = r_vec[:, :2]

    # Unit radial directions (XY)
    l_dir_xy = l_vec_xy / (torch.norm(l_vec_xy, dim=-1, keepdim=True) + 1e-6)
    r_dir_xy = r_vec_xy / (torch.norm(r_vec_xy, dim=-1, keepdim=True) + 1e-6)

    # Radial velocity relative to torso on XY plane (remove body translation)
    l_rel_vel = l_vel - torso_vel
    r_rel_vel = r_vel - torso_vel
    l_rel_vel_xy = l_rel_vel[:, :2]
    r_rel_vel_xy = r_rel_vel[:, :2]
    l_radial_vel = torch.sum(l_rel_vel_xy * l_dir_xy, dim=-1)
    r_radial_vel = torch.sum(r_rel_vel_xy * r_dir_xy, dim=-1)

    if not hasattr(env, "punch_state"):
        env.punch_state = {}
    if asset_cfg.name not in env.punch_state:
        env.punch_state[asset_cfg.name] = {
            "phase": torch.zeros(num_envs, 2, device=device, dtype=torch.long),
            "steps": torch.zeros(num_envs, 2, device=device, dtype=torch.long),
            "max_pos_speed": torch.zeros(num_envs, 2, device=device),
        }
    state = env.punch_state[asset_cfg.name]
    phase, steps, max_pos_speed = (
        state["phase"],
        state["steps"],
        state["max_pos_speed"],
    )

    # Helper to process one hand (index 0: left, 1: right)
    def _process_hand(
        radial_vel: torch.Tensor, hand_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hand_phase = phase[:, hand_idx]
        hand_steps = steps[:, hand_idx]
        hand_max_pos = max_pos_speed[:, hand_idx]

        # Start: outward speed above threshold.
        start_mask = (hand_phase == 0) & (radial_vel > v_start_thresh)
        if start_mask.any():
            hand_phase[start_mask] = 1
            hand_steps[start_mask] = 0
            hand_max_pos[start_mask] = radial_vel[start_mask]

        # Update active punches.
        active_mask = hand_phase == 1
        if active_mask.any():
            hand_steps[active_mask] += 1
            # Track maximal positive speed during the window.
            hand_max_pos[active_mask] = torch.maximum(
                hand_max_pos[active_mask], radial_vel[active_mask]
            )

        # Completion: strong enough retract (negative radial speed)
        # within the same time window.
        success_mask = (
            active_mask
            & (radial_vel < 0.0)
            & ((-radial_vel) >= retract_ratio * hand_max_pos)
            & (hand_steps <= max_steps)
        )

        reward = torch.zeros_like(radial_vel)
        if success_mask.any():
            reward[success_mask] = base_reward

        # Timeouts: active too long without completion.
        timeout_mask = active_mask & (hand_steps > max_steps)

        # Reset state for finished (success or timeout) punches.
        reset_mask = success_mask | timeout_mask
        if reset_mask.any():
            hand_phase[reset_mask] = 0
            hand_steps[reset_mask] = 0
            hand_max_pos[reset_mask] = 0.0

        # Write back to global tensors.
        phase[:, hand_idx] = hand_phase
        steps[:, hand_idx] = hand_steps
        max_pos_speed[:, hand_idx] = hand_max_pos
        return reward, start_mask, success_mask

    reward_l, l_start, l_complete = _process_hand(l_radial_vel, hand_idx=0)
    reward_r, r_start, r_complete = _process_hand(r_radial_vel, hand_idx=1)

    total_reward = reward_l + reward_r

    return total_reward * distance_pen if dist_rescale else total_reward


def _asset_body_ids_for_sensor(asset, sensor, body_ids):
    names = _contact_body_names(sensor)
    selected = (
        names[body_ids]
        if isinstance(body_ids, slice)
        else [names[index] for index in body_ids]
    )
    return [asset.body_names.index(name) for name in selected]


def foot_slip_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(
            torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1),
            dim=1,
        )[0]
        > threshold
    )
    # Sensor and articulation body orders can differ after asset import.
    body_ids = _asset_body_ids_for_sensor(
        asset, contact_sensor, sensor_cfg.body_ids
    )
    foot_planar_velocity = torch.linalg.norm(
        asset.data.body_lin_vel_w[:, body_ids, :2], dim=2
    )

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def foot_lift_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 0.5,
    min_height: float = 0.02,
    max_height: float = 0.3,
) -> torch.Tensor:
    """
    Reward lifting feet off the ground to encourage proper stepping.
    This helps prevent feet from dragging/sliding on the ground.

    Args:
        contact_threshold: Force threshold below which foot is considered lifted
        min_height: Minimum height above ground to get reward (meters)
        max_height: Height at which reward saturates (meters)
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Check if foot is in contact with ground
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # For each foot, check if contact force is below threshold (foot is lifted)
    # Shape: [B, num_feet] - True if foot is in contact
    is_contact = (
        torch.max(
            torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1),
            dim=1,
        )[0]
        > contact_threshold
    )
    is_lifted = ~is_contact  # Foot is lifted when NOT in contact

    # Get foot positions (z coordinate is height above ground, assuming ground at z=0)
    body_ids = _asset_body_ids_for_sensor(
        asset, contact_sensor, sensor_cfg.body_ids
    )
    foot_positions = asset.data.body_pos_w[:, body_ids, :]
    foot_heights = foot_positions[:, :, 2]  # [B, num_feet]

    # Reward based on height when foot is lifted
    # Normalize height to [0, 1] range between min_height and max_height
    # Height below min_height gives 0 reward, height at max_height gives 1.0 reward
    height_above_min = torch.clamp(foot_heights - min_height, min=0.0)
    normalized_heights = torch.clamp(
        height_above_min / (max_height - min_height + 1e-6), min=0.0, max=1.0
    )

    # Only give reward when foot is lifted AND above min_height
    # This encourages lifting feet properly instead of dragging them
    height_reward = is_lifted.float() * normalized_heights

    # Sum rewards across all feet (both left and right ankle)
    return torch.sum(height_reward, dim=1)
