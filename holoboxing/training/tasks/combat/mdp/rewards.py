from __future__ import annotations
from typing import List

import torch
from holoboxing.evaluation.runtime.fall_rules import (
    FALL_BODY_NAMES,
    FALL_HEIGHT_M,
)
from holoboxing.evaluation.runtime.impulse_hit_rules import THRESHOLD_NS
from holoboxing.training.envs.entities import SceneEntityCfg
from holoboxing.training.tasks.combat.config.sandbag import AGENTS
from holoboxing.training.tasks.combat.mdp.geometry import (
    quat_apply_wxyz as quat_apply,
    quat_conjugate_wxyz,
)

HANDS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
STRIKE_TARGET = "torso_link"


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


# Sentinel for "this fist has not been anywhere near the opponent yet". A large
# finite number rather than inf, because the value is carried through
# torch.minimum and reset by assignment.
FAR_M = 1.0e3


def reset_punch_state(env, env_ids):
    for state in getattr(env, "punch_state", {}).values():
        for key, value in state.items():
            value[env_ids] = FAR_M if "closest" in key else 0


ARM_JOINT_PARTS = ("shoulder", "elbow", "wrist")


def custom_action_rate_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm_scale: float = 1.0,
) -> torch.Tensor:
    """Smoothness penalty, with the punching arm optionally exempted.

    A squared action delta charges the legs and the arm the same amount, but
    the two are not the same act. Legs only ever need to move smoothly: any
    large delta down there is a fall or a lurch, and it should cost. An arm
    that is being thrown cannot help producing a large delta -- going from
    guard to full extension inside 0.2 s *is* a large delta, and that is the
    whole point. Charging both at the same rate means the cheapest way to keep
    the penalty down is to never punch, which is exactly the failure this
    weight was producing: -0.0092 per step against a fully saturated
    ``punch_drive`` of 0.05, a 5:1 price on standing still.

    ``arm_scale`` discounts the shoulder/elbow/wrist joints so that the
    gesture can be paid for by the striking terms without the smoother
    outbidding it. It is not zero because a jittering arm still costs real
    tracking quality and a kinetic penalty is a reasonable regulariser.
    """
    action_manager = (
        env.action_managers[asset_cfg.name]
        if hasattr(env, "action_managers")
        else env.action_manager
    )
    delta = torch.square(action_manager.action - action_manager.prev_action)
    if arm_scale != 1.0:
        joints = env.scene[asset_cfg.name].joint_names
        arm = torch.tensor(
            [any(part in n for part in ARM_JOINT_PARTS) for n in joints],
            device=delta.device,
            dtype=torch.bool,
        )
        delta = delta * torch.where(arm, arm_scale, 1.0)
    return torch.sum(delta, dim=1)


def stand_still_reward(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty on XY-plane motion of the torso link (stand still)."""
    ego_agent = env.scene[asset_cfg.name]
    torso_idx = ego_agent.body_names.index("torso_link")
    lin_vel = ego_agent.data.body_vel_w[:, torso_idx][..., :3]
    vel_xy = lin_vel[..., :2]
    speed_sq = torch.sum(torch.square(vel_xy), dim=-1)
    # Negative reward proportional to squared speed in XY plane
    return -speed_sq


def offensive_hit_reward_punch(
    env,
    sensor_cfg1: SceneEntityCfg,
    sensor_cfg2: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
    vel_threshold: float = 1.0,
    relative_by_ego: bool = True,
) -> (
    torch.Tensor
):  # The relative-velocity threshold can be lower because base velocity is subtracted.
    """
    relative_by_ego: True use self torso velocity to compute relative velocity; False use opponent torso velocity to compute relative velocity.
    """
    ego_agent = env.scene[asset_cfg.name]
    opponent_agent = env.get_opponent(asset_cfg.name)

    net_contact_forces_ego, net_contact_forces_opponent = paired_contact_forces(
        env,
        asset_cfg.name,
        env.get_opponent_id(asset_cfg.name),
        sensor_cfg1.name,
        sensor_cfg2.name,
    )

    # --- 1. Get key link indices. ---
    # Confirm the names against your robot model; this assumes standard G1 naming.
    l_wrist_idx = ego_agent.body_names.index("left_wrist_yaw_link")
    r_wrist_idx = ego_agent.body_names.index("right_wrist_yaw_link")
    torso_idx = ego_agent.body_names.index("torso_link")

    # --- 2. Get positions and velocities. ---
    # [Batch, 3]
    ego_torso_vel = ego_agent.data.body_vel_w[:, torso_idx, :3]

    l_wrist_pos = ego_agent.data.body_pos_w[:, l_wrist_idx, :3]
    l_wrist_vel = ego_agent.data.body_vel_w[:, l_wrist_idx, :3]

    r_wrist_pos = ego_agent.data.body_pos_w[:, r_wrist_idx, :3]
    r_wrist_vel = ego_agent.data.body_vel_w[:, r_wrist_idx, :3]

    opp_torso_pos = opponent_agent.data.body_pos_w[:, torso_idx, :3]
    opp_torso_vel = opponent_agent.data.body_vel_w[:, torso_idx, :3]

    # --- 3. Compute pure punching velocity relative to the ego torso. ---
    # This filters out the robot's overall forward rushing velocity.
    # If the robot only walks forward in guard, l_wrist_vel and ego_torso_vel are nearly equal, so the result is close to 0.
    l_rel_vel = l_wrist_vel - ego_torso_vel
    r_rel_vel = r_wrist_vel - ego_torso_vel

    l_oprel_vel = opp_torso_vel - ego_torso_vel
    r_oprel_vel = opp_torso_vel - ego_torso_vel

    # --- 4. Get positions for all target body parts. ---
    # Use sensor_cfg2.body_names to get positions for all target body parts.
    # sensor_cfg2.body_names contains: ["right_hip_roll_link", "left_hip_roll_link", "torso_link", "pelvis"].
    opp_target_positions = []
    for body_name in sensor_cfg2.body_names:
        if body_name in opponent_agent.body_names:
            target_idx = opponent_agent.body_names.index(body_name)
            opp_target_positions.append(
                opponent_agent.data.body_pos_w[:, target_idx, :3]
            )

    if len(opp_target_positions) == 0:
        # Fallback: use torso if no target body parts were found.
        opp_target_positions = [opp_torso_pos]

    # Stack all target body-part positions: [num_targets, Batch, 3].
    opp_target_positions_tensor = torch.stack(
        opp_target_positions, dim=0
    )  # [num_targets, B, 3]

    # --- 5. Compute attack direction and effective speed for each target body part. ---
    # Compute direction vectors from each wrist to each target body part.
    # l_wrist_pos: [B, 3], opp_target_positions_tensor: [num_targets, B, 3]
    l_dir_vecs = opp_target_positions_tensor - l_wrist_pos.unsqueeze(
        0
    )  # [num_targets, B, 3]
    r_dir_vecs = opp_target_positions_tensor - r_wrist_pos.unsqueeze(
        0
    )  # [num_targets, B, 3]

    # Normalize direction vectors.
    l_dir_norms = l_dir_vecs / (
        torch.norm(l_dir_vecs, dim=-1, keepdim=True) + 1e-6
    )  # [num_targets, B, 3]
    r_dir_norms = r_dir_vecs / (
        torch.norm(r_dir_vecs, dim=-1, keepdim=True) + 1e-6
    )  # [num_targets, B, 3]

    # Compute effective attack speed for each target body part.
    # l_rel_vel: [B, 3], l_dir_norms: [num_targets, B, 3]
    l_effective_speeds = torch.sum(
        l_rel_vel.unsqueeze(0) * l_dir_norms, dim=-1
    )  # [num_targets, B]
    r_effective_speeds = torch.sum(
        r_rel_vel.unsqueeze(0) * r_dir_norms, dim=-1
    )  # [num_targets, B]

    l_op_effective_speeds = torch.sum(
        l_oprel_vel.unsqueeze(0) * l_dir_norms, dim=-1
    )  # [num_targets, B]
    r_op_effective_speeds = torch.sum(
        r_oprel_vel.unsqueeze(0) * r_dir_norms, dim=-1
    )  # [num_targets, B]

    # --- 6. Contact check. ---
    # Check each target body part separately for hits.
    # sensor_cfg2.body_ids should contain these body-part indices in the same order as body_names.
    is_opp_body_hit_list = []
    for body_id in sensor_cfg2.body_ids:
        is_hit = (
            torch.max(
                torch.norm(net_contact_forces_opponent[:, :, body_id], dim=-1),
                dim=1,
            )[0]
            > threshold
        )
        is_opp_body_hit_list.append(is_hit)

    # Convert the list to a tensor: [num_targets, B].
    is_opp_body_hit_tensor = torch.stack(
        is_opp_body_hit_list, dim=0
    )  # [num_targets, B]

    # Check whether own fists receive force.
    # sensor_cfg1.body_ids[0] = left_wrist, sensor_cfg1.body_ids[1] = right_wrist
    if len(sensor_cfg1.body_ids) >= 2:
        is_ego_lwrist_hit = (
            torch.max(
                torch.norm(
                    net_contact_forces_ego[:, :, sensor_cfg1.body_ids[0]],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > threshold
        )
        is_ego_rwrist_hit = (
            torch.max(
                torch.norm(
                    net_contact_forces_ego[:, :, sensor_cfg1.body_ids[1]],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > threshold
        )
    else:
        # Fallback if body_ids structure is different
        is_ego_lwrist_hit = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        is_ego_rwrist_hit = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )

    # --- 7. Perform the combined check separately for each target body part. ---
    # For each target body part, check: target hit, fist receives force, and effective speed toward the target is sufficient.
    # l_effective_speeds: [num_targets, B]
    # is_opp_body_hit_tensor: [num_targets, B]
    # is_ego_lwrist_hit: [B]
    # is_ego_rwrist_hit: [B]

    if relative_by_ego:
        # Check whether the left hand effectively hits each target body part.
        # l_effective_speeds[i]: [B] - effective speed from the left hand to target body part i.
        # is_opp_body_hit_tensor[i]: [B] - whether target body part i is hit.
        is_l_valid_hit_per_target = (
            is_opp_body_hit_tensor  # [num_targets, B] - whether each body part is hit.
            & is_ego_lwrist_hit.unsqueeze(
                0
            )  # [1, B] -> [num_targets, B] - whether the left hand receives force.
            & (
                l_effective_speeds > vel_threshold
            )  # [num_targets, B] - whether effective speed to each body part is sufficient.
        )  # [num_targets, B]

        # Check whether the right hand effectively hits each target body part.
        is_r_valid_hit_per_target = (
            is_opp_body_hit_tensor  # [num_targets, B]
            & is_ego_rwrist_hit.unsqueeze(0)  # [1, B] -> [num_targets, B]
            & (r_effective_speeds > vel_threshold)  # [num_targets, B]
        )  # [num_targets, B]
    else:
        # Use opponent-relative speed.
        is_l_valid_hit_per_target = (
            is_opp_body_hit_tensor
            & is_ego_lwrist_hit.unsqueeze(0)
            & (l_op_effective_speeds > vel_threshold)
        )  # [num_targets, B]

        is_r_valid_hit_per_target = (
            is_opp_body_hit_tensor
            & is_ego_rwrist_hit.unsqueeze(0)
            & (r_op_effective_speeds > vel_threshold)
        )  # [num_targets, B]

    # Score if any target body part is effectively hit by either hand.
    # First OR per body part across hands, then OR across all body parts.
    is_l_valid_hit = is_l_valid_hit_per_target.any(
        dim=0
    )  # [B] - whether the left hand effectively hits any body part.
    is_r_valid_hit = is_r_valid_hit_per_target.any(
        dim=0
    )  # [B] - whether the right hand effectively hits any body part.
    total_hit = (is_l_valid_hit | is_r_valid_hit).float()

    # if torch.any(total_hit > 0):
    #     idx = torch.argmax(total_hit)
    #     print(f"Hit! RelSpeed: {l_effective_speed[idx]:.2f}/{r_effective_speed[idx]:.2f}")

    return total_hit


def defensive_hit_penalty_punch(
    env,
    sensor_cfg1: SceneEntityCfg,
    sensor_cfg2: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
    vel_threshold: float = 1.0,
    relative_by_ego: bool = True,
) -> (
    torch.Tensor
):  # The relative-velocity threshold can be lower because base velocity is subtracted.
    """
    relative_by_ego: True use self torso velocity to compute relative velocity; False use opponent torso velocity to compute relative velocity.
    """
    ego_agent = env.scene[asset_cfg.name]
    opponent_agent = env.get_opponent(asset_cfg.name)

    net_contact_forces_opponent, net_contact_forces_ego = paired_contact_forces(
        env,
        env.get_opponent_id(asset_cfg.name),
        asset_cfg.name,
        sensor_cfg1.name,
        sensor_cfg2.name,
    )

    # --- 1. Get key link indices. ---
    # Confirm the names against your robot model; this assumes standard G1 naming.
    l_wrist_idx = ego_agent.body_names.index("left_wrist_yaw_link")
    r_wrist_idx = ego_agent.body_names.index("right_wrist_yaw_link")
    torso_idx = ego_agent.body_names.index("torso_link")

    # --- 2. Get positions and velocities. ---
    # [Batch, 3]
    opp_torso_vel = opponent_agent.data.body_vel_w[:, torso_idx, :3]

    l_wrist_pos = opponent_agent.data.body_pos_w[:, l_wrist_idx, :3]
    l_wrist_vel = opponent_agent.data.body_vel_w[:, l_wrist_idx, :3]

    r_wrist_pos = opponent_agent.data.body_pos_w[:, r_wrist_idx, :3]
    r_wrist_vel = opponent_agent.data.body_vel_w[:, r_wrist_idx, :3]

    ego_torso_pos = ego_agent.data.body_pos_w[:, torso_idx, :3]

    ego_torso_vel = ego_agent.data.body_vel_w[:, torso_idx, :3]

    # --- 3. Compute pure punching velocity relative to the opponent torso. ---
    # This filters out the robot's overall forward rushing velocity.
    # If the robot only walks forward in guard, l_wrist_vel and ego_torso_vel are nearly equal, so the result is close to 0.
    l_rel_vel = l_wrist_vel - opp_torso_vel
    r_rel_vel = r_wrist_vel - opp_torso_vel

    l_oprel_vel = opp_torso_vel - ego_torso_vel
    r_oprel_vel = opp_torso_vel - ego_torso_vel

    # --- 4. Compute attack direction from fist to ego torso. ---
    l_dir_vec = ego_torso_pos - l_wrist_pos
    r_dir_vec = ego_torso_pos - r_wrist_pos

    # Normalize direction vectors.
    l_dir_norm = l_dir_vec / (torch.norm(l_dir_vec, dim=-1, keepdim=True) + 1e-6)
    r_dir_norm = r_dir_vec / (torch.norm(r_dir_vec, dim=-1, keepdim=True) + 1e-6)
    # --- 5. Compute effective attack velocity by projection. ---
    # The dot product is large only when relative velocity points toward the opponent.
    # dim: [Batch]
    l_effective_speed = torch.sum(l_rel_vel * l_dir_norm, dim=-1)
    r_effective_speed = torch.sum(r_rel_vel * r_dir_norm, dim=-1)

    l_op_effective_speed = torch.sum(l_oprel_vel * l_dir_norm, dim=-1)
    r_op_effective_speed = torch.sum(r_oprel_vel * r_dir_norm, dim=-1)

    # --- 6. Contact check. ---
    # Check whether the ego torso and opponent fists receive force.
    is_ego_torso_hit = (
        torch.max(
            torch.norm(net_contact_forces_ego[:, :, sensor_cfg2.body_ids[2]], dim=-1),
            dim=1,
        )[0]
        > threshold
    )
    is_opp_rwrist_hit = (
        torch.max(
            torch.norm(
                net_contact_forces_opponent[:, :, sensor_cfg1.body_ids[3]],
                dim=-1,
            ),
            dim=1,
        )[0]
        > threshold
    )
    is_opp_lwrist_hit = (
        torch.max(
            torch.norm(
                net_contact_forces_opponent[:, :, sensor_cfg1.body_ids[2]],
                dim=-1,
            ),
            dim=1,
        )[0]
        > threshold
    )
    # --- 7. Combined check. ---
    # Criterion: ego is hit and the hand has explosive relative velocity toward ego.
    if relative_by_ego:
        is_l_valid_hit = (
            is_ego_torso_hit & is_opp_lwrist_hit & (l_effective_speed > vel_threshold)
        )
        is_r_valid_hit = (
            is_ego_torso_hit & is_opp_rwrist_hit & (r_effective_speed > vel_threshold)
        )
    else:
        is_l_valid_hit = (
            is_ego_torso_hit
            & is_opp_lwrist_hit
            & (l_op_effective_speed > vel_threshold)
        )
        is_r_valid_hit = (
            is_ego_torso_hit
            & is_opp_rwrist_hit
            & (r_op_effective_speed > vel_threshold)
        )

    total_hit = (is_l_valid_hit | is_r_valid_hit).float()

    # if torch.any(total_hit > 0):
    #     idx = torch.argmax(total_hit)
    #     print(f"Hit! RelSpeed: {l_effective_speed[idx]:.2f}/{r_effective_speed[idx]:.2f}")

    return total_hit


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
    gate_floor: float = 0.25,
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
    # A gate with no floor makes the whole term identically zero at the start
    # of a run, which is where it is needed most: the policy has to already be
    # punching in order to earn anything for punching. With a floor the term
    # still pays for bringing the fist next to the torso at walking speed, so
    # the gesture can be discovered before it can be accelerated.
    gate = gate_floor + (1.0 - gate_floor) * gate

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
    reach: float = 0.45,
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
    opponent_agent = env.get_opponent(asset_cfg.name)
    opponent_torso = opponent_agent.data.body_pos_w[:, torso_idx, :3]
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

    # How close each fist actually got to the opponent torso. A punch is only
    # counted as completed if this closed inside ``reach`` during the throw,
    # so a hand cycling back and forth in the air cannot farm the term. It is
    # the difference between a jab and a shadowbox.
    l_target_dist = (l_pos - opponent_torso).norm(dim=-1)
    r_target_dist = (r_pos - opponent_torso).norm(dim=-1)

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
    # setdefault on each key rather than only on the agent: punch_retract may
    # have created this dict first, and a partially populated state would make
    # ``closest`` a KeyError instead of a missing term.
    state = env.punch_state.setdefault(asset_cfg.name, {})
    state.setdefault(
        "phase", torch.zeros(num_envs, 2, device=device, dtype=torch.long)
    )
    state.setdefault(
        "steps", torch.zeros(num_envs, 2, device=device, dtype=torch.long)
    )
    state.setdefault("max_pos_speed", torch.zeros(num_envs, 2, device=device))
    state.setdefault(
        "closest", torch.full((num_envs, 2), FAR_M, device=device)
    )
    phase, steps, max_pos_speed, closest = (
        state["phase"],
        state["steps"],
        state["max_pos_speed"],
        state["closest"],
    )

    # Helper to process one hand (index 0: left, 1: right)
    def _process_hand(
        radial_vel: torch.Tensor,
        hand_idx: int,
        target_dist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hand_phase = phase[:, hand_idx]
        hand_steps = steps[:, hand_idx]
        hand_max_pos = max_pos_speed[:, hand_idx]
        hand_closest = closest[:, hand_idx]

        # Start: outward speed above threshold.
        start_mask = (hand_phase == 0) & (radial_vel > v_start_thresh)
        if start_mask.any():
            hand_phase[start_mask] = 1
            hand_steps[start_mask] = 0
            hand_max_pos[start_mask] = radial_vel[start_mask]
            hand_closest[start_mask] = target_dist[start_mask]

        # Update active punches.
        active_mask = hand_phase == 1
        if active_mask.any():
            hand_steps[active_mask] += 1
            # Track maximal positive speed during the window.
            hand_max_pos[active_mask] = torch.maximum(
                hand_max_pos[active_mask], radial_vel[active_mask]
            )
            hand_closest[active_mask] = torch.minimum(
                hand_closest[active_mask], target_dist[active_mask]
            )

        # Completion: strong enough retract (negative radial speed)
        # within the same time window, and the fist has to have arrived.
        success_mask = (
            active_mask
            & (radial_vel < 0.0)
            & ((-radial_vel) >= retract_ratio * hand_max_pos)
            & (hand_steps <= max_steps)
            & (hand_closest <= reach)
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
            hand_closest[reset_mask] = FAR_M

        # Write back to global tensors.
        phase[:, hand_idx] = hand_phase
        steps[:, hand_idx] = hand_steps
        max_pos_speed[:, hand_idx] = hand_max_pos
        closest[:, hand_idx] = hand_closest
        return reward, start_mask, success_mask

    reward_l, l_start, l_complete = _process_hand(
        l_radial_vel, hand_idx=0, target_dist=l_target_dist
    )
    reward_r, r_start, r_complete = _process_hand(
        r_radial_vel, hand_idx=1, target_dist=r_target_dist
    )

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


# --- Competition-rule terms (impulse v6) ---------------------------------
# Everything below is keyed off the official RuleScorer so that training and
# the referee pay for exactly the same events.


def _role(agent: str) -> int:
    return AGENTS.index(agent)


def rule_point_reward(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Points awarded this step by the official impulse gate."""
    return env.scorer.step_points[_role(asset_cfg.name)]


def taken_point_penalty(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Points the opponent scored on this step (one point conceded each)."""
    return env.scorer.step_points[_role(env.get_opponent_id(asset_cfg.name))]


def impulse_progress_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = THRESHOLD_NS,
) -> torch.Tensor:
    """Reward for the live 20 ms impulse window, cubic below the gate.

    The sparse point reward only fires once the window clears 5 N*s, which is
    far too rare to learn from on its own when the warmup opponent barely
    moves. This term keeps a dense gradient alive and saturates at the scoring
    threshold so pressing harder than a point buys nothing extra.

    The exponent is the whole point. A *linear* reading was what produced the
    plateau at 3-4.5 N*s: at 4 N*s the term is already paying 80% of its cap,
    so the last 20% -- the only part that actually crosses the referee's gate
    and pays 1.6 -- was the cheapest 20% of the term to skip. Cubed, 4 N*s is
    worth 51% and 5 N*s is 100%, so the marginal value of the final newton
    rises instead of flattening. The low end is left to ``punch_drive``,
    which is the term that knows how to get a hand moving at all; this one
    only has to make arriving *hard* worth more than arriving *close*.
    """
    window = env.scorer.impulse_of(_role(asset_cfg.name))
    level = (window / threshold).clamp(max=1.0).max(dim=-1).values
    return level.pow(3)


def retract_after_score_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    release_distance: float = 0.28,
    max_steps: int = 20,
) -> torch.Tensor:
    """Reward pulling a scoring hand back out of the torso after a point.

    The referee blocks a hand for 200 ms after it scores and the impulse window
    is only 20 ms wide, so a fist parked on the torso still pays -- but only at
    the cooldown rate, and holding it there is only possible by leaning, which
    is what makes fighters fall. Paying for the punch-and-recover rhythm gives
    the policy a way to keep scoring without standing on its toes.
    """
    if not hasattr(env, "punch_state"):
        env.punch_state = {}
    state = env.punch_state.setdefault(asset_cfg.name, {})
    if "retract_armed" not in state:
        state["retract_armed"] = torch.zeros(
            env.num_envs, 2, dtype=torch.bool, device=env.device
        )
        state["retract_steps"] = torch.zeros(
            env.num_envs, 2, dtype=torch.long, device=env.device
        )
    armed, steps = state["retract_armed"], state["retract_steps"]

    scored = env.scorer.scored[:, _role(asset_cfg.name)]
    armed.logical_or_(scored)
    steps.add_(1)
    steps[scored] = 0

    ego_agent = env.scene[asset_cfg.name]
    opponent_agent = env.get_opponent(asset_cfg.name)
    torso_idx = ego_agent.body_names.index(STRIKE_TARGET)
    hand_ids = [ego_agent.body_names.index(name) for name in HANDS]
    target = opponent_agent.data.body_pos_w[:, torso_idx, :3].unsqueeze(1)
    distance = (ego_agent.data.body_pos_w[:, hand_ids, :3] - target).norm(
        dim=-1
    )

    cleared = armed & (distance > release_distance) & (steps >= 1)
    expired = armed & (steps > max_steps)
    reward = cleared.sum(dim=-1).to(torch.float32)
    armed.logical_and_(~(cleared | expired))
    steps[cleared | expired] = 0
    return reward


def balance_margin_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safe_height: float = 0.55,
) -> torch.Tensor:
    """Penalty on approaching the official 0.30 m fall line.

    The referee needs three continuous seconds below 0.30 m before it calls a
    KO, which with gamma=0.995 is far outside a 24-step credit horizon. A dense
    height margin is the only way the policy can learn the danger early.
    """
    agent = env.scene[asset_cfg.name]
    floor = env.physics.floor_height
    body_ids = [agent.body_names.index(name) for name in FALL_BODY_NAMES]
    height = agent.data.body_pos_w[:, body_ids, 2] - floor
    span = max(safe_height - FALL_HEIGHT_M, 1e-6)
    margin = ((safe_height - height).clamp_min(0.0) / span).square()
    return margin.max(dim=-1).values


def upright_reward(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty on torso tilt away from world up (linear, so it bites early)."""
    agent = env.scene[asset_cfg.name]
    torso_idx = agent.body_names.index(STRIKE_TARGET)
    quat = agent.data.body_quat_w[:, torso_idx]
    up = torch.zeros_like(quat[:, :3])
    up[:, 2] = 1.0
    return (1.0 - quat_apply(quat, up)[:, 2]).clamp_min(0.0)


def com_support_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    free_offset: float = 0.22,
    span: float = 0.22,
    limit: float = 1.0,
) -> torch.Tensor:
    """Penalty on the CoM running away from the ankle midpoint, past a dead zone.

    Measuring the offset from zero is not a balance criterion, it is a
    stand-still criterion. Walking and throwing both push the CoM 0.15-0.25 m
    ahead of the ankles, so with a 0.15 m scale both saturate the penalty at
    0.03/step while walking earns about 0.002/step of range reward. Net cost
    10:1, which is exactly why the warmup policy shuffled in at 0.07 m/s and
    never threw a punch.

    Falling is caught by two terms that do not look at this offset at all:
    ``balance_margin`` (body height dropping under 0.55 m) and ``upright``
    (torso tilt away from vertical). So this term only has to catch the
    over-committed lunge, hence the free zone.
    """
    agent = env.scene[asset_cfg.name]
    feet = [
        agent.body_names.index(name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    com = agent.data.root_com_w[..., :2]
    support = agent.data.body_pos_w[:, feet, :2].mean(dim=1)
    excess = ((com - support).norm(dim=-1) - free_offset).clamp_min(0.0)
    return (excess / span).square().clamp(max=limit)


def com_drop_reward(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty on downward centre of mass velocity; the precursor of a fall."""
    agent = env.scene[asset_cfg.name]
    return (-agent.data.root_lin_vel_w[:, 2]).clamp_min(0.0).square()


def fall_enter_penalty(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """One-shot penalty the first time a fighter drops below the fall line."""
    return env.fall_entered[:, _role(asset_cfg.name)].to(torch.float32)


def ko_terminal_reward(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Non-zero only on the step a knockout is confirmed for either fighter."""
    return env.ko_signal[:, _role(asset_cfg.name)]


def torso_gap(env, agent: str) -> torch.Tensor:
    """Horizontal torso-to-torso distance, the range a punch has to cross."""
    ego_agent = env.scene[agent]
    opponent_agent = env.get_opponent(agent)
    torso_idx = ego_agent.body_names.index(STRIKE_TARGET)
    return (
        ego_agent.data.body_pos_w[:, torso_idx, :2]
        - opponent_agent.data.body_pos_w[:, torso_idx, :2]
    ).norm(dim=-1)


def punch_drive_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.50,
    speed_cap: float = 2.5,
) -> torch.Tensor:
    """Pay for throwing a fist *fast* into the opponent's torso, in [0, 1].

    The referee only pays for 5 N*s inside a 20 ms window, i.e. about 250 N of
    average normal force. That is roughly a sixth of a real punch peak, so it is
    reachable -- but it is roughly 400x the impulse the current policy produces
    (about 0.013 N*s, a 0.65 N graze). Nothing in the reward ladder connected
    "hand resting on the chest" to "hand arriving at 2 m/s", and with a policy
    action std of 0.19 (about 10 mm of reference travel per step) a 0.4 m arm
    extension is a 40-sigma event: it cannot be found by exploration, only by
    following a gradient that keeps paying as the fist gets closer *and*
    faster.

    This is that gradient. It is the missing rung between the pure geometry of
    ``active_hit_shaping_reward`` (distance, speed-gated) and the impulse gate,
    and it is bounded so that oscillating the arm cannot farm it without bound:
    at full speed the whole per-step cap is one second of range shaping, and
    proximity only reaches 1.0 when the fist is on the torso.
    """
    ego_agent = env.scene[asset_cfg.name]
    opponent_agent = env.get_opponent(asset_cfg.name)
    torso_idx = ego_agent.body_names.index(STRIKE_TARGET)
    hand_ids = [ego_agent.body_names.index(name) for name in HANDS]

    # body_vel_w is [linear, angular]; a punch is judged against the *own*
    # torso so that simply walking forward earns nothing.
    own_torso_vel = ego_agent.data.body_vel_w[:, torso_idx, :3]
    wrist_pos = ego_agent.data.body_pos_w[:, hand_ids, :3]
    wrist_vel = (
        ego_agent.data.body_vel_w[:, hand_ids, :3] - own_torso_vel[:, None, :]
    )
    target = opponent_agent.data.body_pos_w[:, torso_idx, :3]
    target_vel = opponent_agent.data.body_vel_w[:, torso_idx, :3]

    diff = target[:, None, :] - wrist_pos  # [B, H, 3]
    distance = diff.norm(dim=-1)
    direction = diff / (distance.unsqueeze(-1) + 1e-6)
    closing = ((wrist_vel - target_vel[:, None, :]) * direction).sum(-1)
    proximity = torch.exp(-(distance / sigma).square())
    drive = closing.clamp(min=0.0, max=speed_cap) / speed_cap
    return (proximity * drive).amax(dim=-1)


def punch_speed(env, agent: str) -> torch.Tensor:
    """Best wrist closing speed towards the opponent torso, in m/s.

    Diagnostic only. The impulse log answers "did it hurt"; this answers "is
    the arm moving at all", and the two failure modes look identical in every
    other number. A 2 m/s fist that misses and a 0.05 m/s fist resting on the
    chest both read as zero impulse, but only one of them is worth waiting on.
    """
    ego_agent = env.scene[agent]
    opponent_agent = env.get_opponent(agent)
    torso_idx = ego_agent.body_names.index(STRIKE_TARGET)
    hand_ids = [ego_agent.body_names.index(name) for name in HANDS]
    own_torso_vel = ego_agent.data.body_vel_w[:, torso_idx, :3]
    wrist_pos = ego_agent.data.body_pos_w[:, hand_ids, :3]
    wrist_vel = (
        ego_agent.data.body_vel_w[:, hand_ids, :3] - own_torso_vel[:, None, :]
    )
    target = opponent_agent.data.body_pos_w[:, torso_idx, :3]
    target_vel = opponent_agent.data.body_vel_w[:, torso_idx, :3]
    diff = target[:, None, :] - wrist_pos
    distance = diff.norm(dim=-1)
    direction = diff / (distance.unsqueeze(-1) + 1e-6)
    closing = ((wrist_vel - target_vel[:, None, :]) * direction).sum(-1)
    return closing.amax(dim=-1)


def range_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_distance: float = 0.88,
    outer_sigma: float = 0.50,
    inner_sigma: float = 0.14,
) -> torch.Tensor:
    """Cauchy attractor on the torso gap at punching range.

    A stationary payoff at the right range is what pays for the walk in: pure
    closing-speed shaping is return-neutral, so idling and advancing score the
    same while walking in still costs balance and action-rate penalties.

    The kernel is Cauchy rather than Gaussian on purpose. The fighters spawn
    about 2.07 m apart, and a Gaussian with sigma 0.30 gives
    ``exp(-((2.07-0.88)/0.30)^2) = 3.4e-4`` there -- effectively no gradient, so
    for the first dozen updates nothing in the task reward pointed at the
    opponent and the approach was actually being driven by the AMP style term
    decaying away. Cauchy falls off polynomially: at 2.07 m it still carries a
    slope of about -0.10/m, which is a usable gradient over the whole runway.

    The kernel stays asymmetric -- inner falls off in 0.14 m against 0.50 m
    outside -- so crowding the opponent still costs immediately. That inner
    slope is the anti-leaning term: at this range a fist reaches the torso by
    extending the arm, not by pushing the chest forward.

    ``target_distance`` moved from 0.72 m to 0.88 m once the impulse telemetry
    became readable: fists already graze the opponent's torso at a mean gap of
    0.96 m, so 0.72 m is not striking range, it is clinch range. There the arm
    is folded and cannot accelerate, and the only way to keep the reward is to
    lean the chest into the opponent -- which scores penetration-recovery
    impulses and is exactly the behaviour that produces falls. Arm's length is
    where a jab can be thrown, so that is where the attractor belongs.
    """
    offset = torso_gap(env, asset_cfg.name) - target_distance
    sigma = torch.where(
        offset >= 0,
        torch.full_like(offset, outer_sigma),
        torch.full_like(offset, inner_sigma),
    )
    return 1.0 / (1.0 + (offset / sigma).square())


def approach_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    """Potential-based closing speed on the torso-to-torso distance.

    A plain "walk forward" reward pays every step, so a policy that paces back
    and forth collects it forever. Shaping on the *change* of distance is
    return-neutral for any closed loop, so retreating costs exactly what
    advancing earns and idling pays nothing.
    """
    if not hasattr(env, "punch_state"):
        env.punch_state = {}
    state = env.punch_state.setdefault(asset_cfg.name, {})
    if "approach_prev" not in state:
        state["approach_prev"] = torch.zeros(env.num_envs, device=env.device)
    previous = state["approach_prev"]

    distance = torso_gap(env, asset_cfg.name)
    # A zero sentinel means "no previous sample", e.g. the step after a reset.
    closing = torch.where(
        previous > 0,
        (previous - distance) / 0.02 * scale,
        torch.zeros_like(distance),
    )
    previous.copy_(distance)
    return closing.clamp(min=-1.0, max=1.0)

