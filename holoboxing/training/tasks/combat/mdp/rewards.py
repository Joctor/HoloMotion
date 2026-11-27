from __future__ import annotations
import math
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

# The joints whose large deltas *are* the punch rather than a fault. The arm is
# obvious; ``waist_yaw`` joins it because it is the torso rotation that throws
# the shoulder into the punch, and because of how the action space is scaled:
# every joint's action is normalised by ``action_scale`` (0.25 * effort /
# (inertia * (2*pi*10)^2)), and waist_yaw carries the largest gain in the body
# at 0.547 rad per unit against 0.439 for the shoulder. So one radian of torso
# yaw costs 3.34 in the squared term while one radian of shoulder pitch costs
# 5.20 -- the waist is the *cheapest* joint in the body to move per radian,
# until the arm discount is applied, after which the same radian costs 0.26 on
# the shoulder and 3.34 on the waist, a 12.8x mispricing between two joints
# doing the same job.
#
# waist_roll and waist_pitch are deliberately left out. Roll is the lateral
# lean axis with no boxing use -- ``energy_saving_penalty`` already singles it
# out as the one waist axis to keep taxed. Pitch is the forward press, which
# moves the centre of mass toward the opponent's side of the support polygon
# and is paid for by ``range``; at -9/round for ``fall_enter`` the table does
# not need a second, cheaper route into a lean.
PUNCH_CHAIN_PARTS = (*ARM_JOINT_PARTS, "waist_yaw")


def custom_action_rate_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    chain_scale: float = 1.0,
    chain_parts: tuple = PUNCH_CHAIN_PARTS,
) -> torch.Tensor:
    """Smoothness penalty, with the punch chain optionally exempted.

    A squared action delta charges the legs and the arm the same amount, but
    the two are not the same act. Legs only ever need to move smoothly: any
    large delta down there is a fall or a lurch, and it should cost. An arm
    that is being thrown cannot help producing a large delta -- going from
    guard to full extension inside 0.2 s *is* a large delta, and that is the
    whole point. Charging both at the same rate means the cheapest way to keep
    the penalty down is to never punch, which is exactly the failure this
    weight was producing: -0.0092 per step against a fully saturated
    ``punch_drive`` of 0.05, a 5:1 price on standing still.

    ``chain_scale`` discounts the joints named by ``chain_parts`` so that the
    gesture can be paid for by the striking terms without the smoother
    outbidding it. It is not zero because a jittering limb still costs real
    tracking quality and a kinetic penalty is a reasonable regulariser.
    ``chain_parts`` is a name-pattern list rather than a fixed constant so the
    caller decides where the chain ends -- see ``PUNCH_CHAIN_PARTS`` for why
    waist_yaw is in and waist_roll / waist_pitch are out.
    """
    action_manager = (
        env.action_managers[asset_cfg.name]
        if hasattr(env, "action_managers")
        else env.action_manager
    )
    delta = torch.square(action_manager.action - action_manager.prev_action)
    if chain_scale != 1.0:
        joints = env.scene[asset_cfg.name].joint_names
        chain = torch.tensor(
            [any(part in n for part in chain_parts) for n in joints],
            device=delta.device,
            dtype=torch.bool,
        )
        delta = delta * torch.where(chain, chain_scale, 1.0)
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
    reach_max: float = 0.45,
    reach_min: float = 0.24,
    throw_floor: float = 0.15,
    impulse_share: float = 0.75,
    impulse_soft: float = 0.5,
    speed_ref: float = 2.5,
    speed_floor: float = 0.35,
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

    The payoff is now dominated by the impulse the punch actually delivered.
    At iteration 182 the log read: 45 completed punches per round, a glove that
    reached 0.28 m from the opponent torso on most of them, and still only ~2
    points per round -- because the mean impact was 1-2 N*s against a 5 N*s
    gate. The distance ramp paid 0.16 for "the glove is touching", which is
    satisfied by resting the fist on the chest; the referee pays for force, and
    nothing in the table asked for force. The payoff is therefore

    ``throw_floor + (1 - throw_floor) * (share * impulse_level^2
                                        + (1 - share) * depth^2)``

    where ``impulse_level`` is a log ramp of the punch's peak 20 ms impulse
    between ``impulse_soft`` (0.5 N*s, a graze) and ``THRESHOLD_NS`` (5 N*s, the
    gate). Log rather than linear because the population sits at 1-2 N*s: a
    linear reading is nearly flat there, while the log ramp is steepest exactly
    in that band, so the first newton a fighter wins is worth as much as the
    last. ``impulse_share`` keeps a quarter on distance: closing is what the
    previous round bought, and a payoff blind to arrival would pay for swinging
    fast two metres away.

    The whole payoff is then multiplied by ``max_pos_speed / speed_ref``
    (clamped to ``speed_floor``). At update 257 the table paid 63 completed
    punches per round at a mean outbound speed of 0.72 m/s, because the cycle
    completes at 0.45 m/s and nothing above that threshold was worth anything.
    Impulse is mass times arrival speed, so the 5 N*s gate needs roughly 3 m/s,
    and the multiplier is the only term in the table that pays for the
    throwing speed itself rather than for its consequences.

    The payoff depends on how far the fist actually travelled. A hard gate on
    ``closest`` (the previous design, 0.45 m from the opponent's torso centre)
    turned out to be the single most profitable thing in the whole reward
    table to sit on: the arm is ~0.75 m from its own torso centre, so at a
    fighting gap of 0.81 m a fist parked 0.45 m short of the target is a
    comfortable half-extension -- 0.28 m of air in front of the glove -- and
    the log showed 2.4% of all world-steps completing a punch, about 72 of
    them per round and per world, against a mean impulse of 0.56% of the
    scoring threshold. The policy had learned to shadowbox at exactly the
    threshold and stop.

    So the reward is now a ramp in ``closest``: ``throw_floor`` for a throw
    that merely reached ``reach_max``, rising quadratically to 1.0 at
    ``reach_min``. Quadratic because the last 10 cm is the part that has to be
    bought -- force comes from arriving, and the arm only arrives if the
    shoulder and hip rotate behind it. ``throw_floor`` keeps a trickle on the
    motion itself so the behaviour cannot die out while the depth is learned.

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

    # How close each fist actually got to the opponent torso. The completion
    # condition uses it twice: as a gate (``reach_max``) and as the payoff
    # depth, so a hand cycling back and forth in the air earns nothing.
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
    # Peak 20 ms impulse seen from this hand while the punch was live. The
    # impact happens one or two steps before the retract that completes the
    # cycle, so the payoff has to be carried in the cycle state instead of read
    # off the current step, which by then is already back to zero.
    state.setdefault("impact", torch.zeros((num_envs, 2), device=device))
    phase, steps, max_pos_speed, closest, impact = (
        state["phase"],
        state["steps"],
        state["max_pos_speed"],
        state["closest"],
        state["impact"],
    )

    # Helper to process one hand (index 0: left, 1: right)
    def _process_hand(
        radial_vel: torch.Tensor,
        hand_idx: int,
        target_dist: torch.Tensor,
        hand_impulse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hand_phase = phase[:, hand_idx]
        hand_steps = steps[:, hand_idx]
        hand_max_pos = max_pos_speed[:, hand_idx]
        hand_closest = closest[:, hand_idx]
        hand_impact = impact[:, hand_idx]

        # Start: outward speed above threshold.
        start_mask = (hand_phase == 0) & (radial_vel > v_start_thresh)
        if start_mask.any():
            hand_phase[start_mask] = 1
            hand_steps[start_mask] = 0
            hand_max_pos[start_mask] = radial_vel[start_mask]
            hand_closest[start_mask] = target_dist[start_mask]
            hand_impact[start_mask] = hand_impulse[start_mask]

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
            hand_impact[active_mask] = torch.maximum(
                hand_impact[active_mask], hand_impulse[active_mask]
            )

        # Completion: strong enough retract (negative radial speed)
        # within the same time window, and the fist has to have arrived.
        success_mask = (
            active_mask
            & (radial_vel < 0.0)
            & ((-radial_vel) >= retract_ratio * hand_max_pos)
            & (hand_steps <= max_steps)
            & (hand_closest <= reach_max)
        )

        # Depth of the throw: 0 at ``reach_max`` (the old free lunch), 1 at
        # ``reach_min`` (glove on the torso). Squared so that the arm has to
        # keep working through the last stretch instead of coasting to the gate.
        depth = ((reach_max - hand_closest) / (reach_max - reach_min)).clamp(
            0.0, 1.0
        )
        # Impulse depth: how hard it landed, on the referee's own scale.
        impulse_level = (
            torch.log1p((hand_impact / impulse_soft).clamp_min(0.0))
            / math.log1p(THRESHOLD_NS / impulse_soft)
        ).clamp(0.0, 1.0)
        level = (
            impulse_share * impulse_level.square()
            + (1.0 - impulse_share) * depth.square()
        )
        # How hard the arm was actually thrown, independent of whether it
        # arrived. The cycle completes at ``v_start_thresh`` = 0.45 m/s, so a
        # 0.7 m/s push counts as a punch and used to be paid like one: the log
        # at update 257 has 63 completed punches per round, a mean outbound
        # speed of 0.72 m/s and 1.26 N*s per contact. Impulse is mass times
        # arrival speed, so a 5 N*s point needs about 3 m/s, and no version of
        # the payoff ever asked for it.
        #
        # The floor has to sit *below* the population or it is a constant, not
        # a slope. At 0.35 with ``speed_ref`` 2.5 it only starts paying above
        # 0.875 m/s, and every punch in the batch is thrown at 0.72 m/s, so the
        # multiplier was a flat 0.35 for all of them. At 0.20 the same batch
        # reads 0.29 and the slope is live from 0.5 m/s up.
        speed = (hand_max_pos / speed_ref).clamp(speed_floor, 1.0)

        reward = torch.zeros_like(radial_vel)
        if success_mask.any():
            reward[success_mask] = base_reward * speed[success_mask] * (
                throw_floor + (1.0 - throw_floor) * level[success_mask]
            )

        # Timeouts: active too long without completion.
        timeout_mask = active_mask & (hand_steps > max_steps)

        # Reset state for finished (success or timeout) punches.
        reset_mask = success_mask | timeout_mask
        if reset_mask.any():
            hand_phase[reset_mask] = 0
            hand_steps[reset_mask] = 0
            hand_max_pos[reset_mask] = 0.0
            hand_closest[reset_mask] = FAR_M
            hand_impact[reset_mask] = 0.0

        # Write back to global tensors.
        phase[:, hand_idx] = hand_phase
        steps[:, hand_idx] = hand_steps
        max_pos_speed[:, hand_idx] = hand_max_pos
        closest[:, hand_idx] = hand_closest
        impact[:, hand_idx] = hand_impact
        return reward, start_mask, success_mask

    window = env.scorer.impulse_of(_role(asset_cfg.name))
    reward_l, l_start, l_complete = _process_hand(
        l_radial_vel, hand_idx=0, target_dist=l_target_dist, hand_impulse=window[:, 0]
    )
    reward_r, r_start, r_complete = _process_hand(
        r_radial_vel, hand_idx=1, target_dist=r_target_dist, hand_impulse=window[:, 1]
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


def landing_speed_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_ns: float = 0.5,
    slow: float = 1.0,
    fast: float = 2.5,
    power: float = 1.0,
    memory_steps: int = 3,
    decay: float = 0.7,
) -> torch.Tensor:
    """Arrival speed of the hand that is actually touching, in [0, 1].

    The gap this fills is the one the last 500 updates could not cross. Every
    contact is a collision at a mean of 1.9 N*s (``impact_ns``) and only 5% of
    them clear the referee's 5 N*s gate (``point_impulse_ns`` 7.0), and that
    5% barely moved across 500 updates: 18.8 touches per point at 954-1203 and
    19.8 at 1204-1453. So the table pays 9:1 for a hard landing over a tap
    (punch cycle 1.0 vs 0.52, impulse progress 1.0 vs 0.10, plus 3.8 for the
    point) and the policy still stands in a 1:2 exchange of taps, which means
    the missing thing is not the price of force but a *signal that separates
    force from leaning*.

    Every existing dense term fails that test:

    * ``impulse_progress`` reads the same 20 ms window the referee does, and a
      glove pressed into the torso holds ~100 N there -- 1.9 N*s, exactly the
      measured mean contact -- so a lean is indistinguishable from a jab and
      gets paid 0.10 of the cap.
    * ``punch_drive`` pays speed near the torso, which includes a fist orbiting
      half a metre short of anything (the windmill that got it cut to 3.5).
    * ``punch_cycle`` pays the throw-and-recover rhythm and is speed-scaled,
      but it is an *event* reward for the motion, blind to what happened.

    This one is gated on real contact and then reads the speed of the touching
    hand, so the two failure modes pay zero: a press has speed ~0, and a
    windmill has no contact. Only a fist that arrives while it is arriving
    scores. ``punch_speed_by_hand`` also subtracts the torso velocity, so
    leaning (which drags both arms) cannot fake arrival speed.

    **The speed is read from before the collision, not during it.** A contact
    lasts ~3 physics substeps (15 ms), i.e. less than one 20 ms control step,
    while ``body_vel_w`` is sampled at the end of the step. By the time the
    impulse window shows the contact, the collision has already absorbed and
    reversed the normal velocity: a 1.8 m/s jab reads ~1.3 m/s of total
    rebound speed and a negative normal component. Reading the contact frame
    directly would clamp half the samples to zero and then scale the reward by
    the rebound -- the same mistake the desktop reward-hacking branch made with
    ``tip_impact_speed``, where the term sat at raw 0.0025 for 108 updates and
    the policy never received a "be fast on arrival" gradient.

    So each hand carries a short memory of its last approach speed: it is
    refreshed with ``max(speed, memory * decay)`` on non-contact steps and
    *frozen* while touching, so the contact step reads the last pre-impact
    value. The memory decays over ``memory_steps`` and the payment is capped at
    ``memory_steps`` consecutive touching steps, so parking a glove on the
    torso buys at most a few steps of the speed it had when it arrived, then
    nothing.

    Linear in the speed (``power`` 1.0), not squared. The first 50 updates
    after this term went in paid 0.67/round, which back-solves to an arrival
    speed of 1.46 m/s: the whole distribution sits below the 2.5 m/s reference,
    so squaring it multiplies every sample by 0.09 and the term becomes
    pocket change against the 12/round action-rate tax. A square is the right
    shape once the distribution is already near the target and only the last
    band needs the money; while the whole distribution is in the low band,
    linear at least pays every +0.1 m/s the same. ``power`` stays a parameter
    so the shape can be restored once arrival speeds cluster above 2 m/s.

    ``fast`` stays at 2.5 m/s -- the reference ``punch_cycle`` uses -- but note
    that 2.5 m/s is *not* enough to clear the gate on its own: 2.5 m/s x 1.7 kg
    x 100% transfer = 4.25 N*s, short of 5. So this term can only teach the
    arm to arrive fast; crossing the gate additionally needs the torso's mass
    behind the fist, which is the jab-alignment problem the desktop branch
    already failed to buy.

    Calibration, and it was wrong the first time. The 0.67/round of the first
    50 updates was back-solved to a 1.46 m/s arrival and the weight was set so
    that number would pay 3.7/round. What 1504-1753 actually paid was 15.9/round
    at 25, while ``touch_frac`` rose 0.029 -> 0.053, ``impact_ns`` fell 1.86 ->
    1.75 and conceded points rose 9.6 -> 15.1/round: the term was buying
    contact volume, not arrival speed. Two things were wrong and both are fixed
    here -- the memory was shared between the two robots (see the cache below),
    and 25 was set from a speed that the first run never confirmed. It is now
    12, and the check is not the income alone but the pair
    ``corr(income, impact_ns) > 0``: if the term pays more while delivered
    impulse falls, it is paying for touching. The gate pays ~20/round for the
    same punches, so 4-8/round is the whole budget this term may occupy.
    """
    touching = env.scorer.impulse_of(_role(asset_cfg.name)) > contact_ns
    speed = punch_speed_by_hand(env, asset_cfg.name).clamp(min=0.0)

    # Per agent, not per env. ``terms()`` runs for both robots in the same
    # step, so a single attribute made the second call read the first robot's
    # memory: ``where(touching_B, memory_A, ...)`` froze A's value whenever B
    # touched (so it could not decay) and handed B a speed that may have been
    # A's. That both extends the freeze window and lets the faster robot's
    # approach pay for the slower one's contact, which is exactly the "pay for
    # touching" failure the 1504-1753 run showed: income 15.9/round against a
    # 3.7 budget, corr(income, touch_frac) = +0.96 while corr(income,
    # impact_ns) = -0.21.
    cache = getattr(env, "_landing_speed_cache", None)
    if cache is None:
        cache = {}
        env._landing_speed_cache = cache
    memory, age = cache.get(asset_cfg.name, (None, None))
    if memory is None or memory.shape != speed.shape:
        memory, age = torch.zeros_like(speed), torch.zeros_like(speed)
    age = torch.where(touching, age + 1.0, torch.zeros_like(age)).clamp(max=64.0)
    memory = torch.where(
        touching, memory, torch.maximum(speed, memory * decay)
    ).detach()
    cache[asset_cfg.name] = (memory, age.detach())

    level = ((memory - slow) / (fast - slow)).clamp(min=0.0, max=1.0)
    armed = touching & (age <= float(memory_steps))
    # Sum over hands, not max. Both the score gate and this term used to be
    # blind to which glove threw: the gate is ``impulse.amax`` over the pair
    # and the payoff was ``amax`` as well, so a fighter that only ever throws
    # with the lead hand collected the same shaping money as one that threw
    # with both and paid half the action-rate tax for it. The referee cools
    # each hand separately for 200 ms, so the second hand is worth a whole
    # second scoring channel -- it has to be priced as one. A dormant glove
    # contributes zero, so this change leaves a one-handed fighter's income
    # exactly where it is and only opens the option of a second one.
    return (level.pow(power) * armed).sum(dim=-1)


def impulse_progress_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = THRESHOLD_NS,
    pedestal: float = 0.4,
) -> torch.Tensor:
    """Reward for the live 20 ms impulse window, per hand, steep at the gate.

    The sparse point reward only fires once the window clears 5 N*s, which is
    far too rare to learn from on its own when the warmup opponent barely
    moves. This term keeps a dense gradient alive and saturates at the scoring
    threshold so pressing harder than a point buys nothing extra.

    The convex body is the whole point. A *linear* reading was what produced
    the plateau at 3-4.5 N*s: at 4 N*s it already pays 80% of its cap, so the
    last 20% -- the only part that crosses the referee's gate -- was the
    cheapest 20% to skip. Cubed, 4 N*s is 51% and 5 N*s is 100%, so the
    marginal value of the final newton rises instead of flattening.

    Two changes turn that from a one-handed trap into a two-handed ladder.

    **Per hand, then summed.** The referee keeps a *separate* 200 ms cooldown
    per hand, so the second glove is a whole second scoring channel. Reducing
    over the pair with ``max`` before pricing threw that channel away: a
    fighter throwing only with the lead hand earned the same shaping money as
    one throwing with both. Now each hand is priced on its own reading and the
    two are added. A dormant hand contributes exactly zero, so this leaves a
    one-handed fighter's income where it was and only opens the second option.

    **A concave pedestal under the convex body.** Pricing each hand separately
    is not enough on its own, because the convex body starves the hand that is
    behind: its slope is ``L + 1.5 L^2``, which is 0.065 at the measured 0.3
    N*s of a dormant hand and 0.597 at the 1.9 N*s of the working one -- a
    9:1 gradient ratio *in favour of the hand that already works*. Additive
    shaping would then keep feeding the lead hand, i.e. reproduce the 703/953
    collapse with a larger bill. The pedestal ``(1+k) L / (L+k)`` is concave,
    so its slope is 1.46 at 0.3 N*s against 0.64 at 1.9 N*s: 2.3:1 the other
    way, which is what actually recruits the weak hand. The body still carries
    the gate (slope 1.57 at 5 N*s against 1.0 for a linear reading), so the
    last newton is still the most valuable one.

    ``k`` rather than ``sqrt`` on purpose: ``sqrt`` has infinite slope at zero,
    and zero is where a dormant hand spends every non-contact step, so the
    backward pass would carry inf through a term that is otherwise fine. The
    rational form is 0 at 0, 1 at 1, smooth, and its slope at zero is 2.4
    (the recruit component alone is 6.0) -- large enough to move a hand that
    has just touched, finite enough not to spike the update.
    """
    window = env.scorer.impulse_of(_role(asset_cfg.name))
    level = (window / threshold).clamp(min=0.0, max=1.0)
    body = 0.5 * level.square() + 0.5 * level.pow(3)
    knee = 0.2
    recruit = (1.0 + knee) * level / (level + knee)
    per_hand = pedestal * recruit + (1.0 - pedestal) * body
    return per_hand.sum(dim=-1)


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
    safe_height: float = 0.45,
    exponent: float = 2.0,
) -> torch.Tensor:
    """Penalty on approaching the official 0.30 m fall line.

    The referee needs three continuous seconds below 0.30 m before it calls a
    KO, which with gamma=0.995 is far outside a 24-step credit horizon. A dense
    height margin is the only way the policy can learn the danger early.

    ``safe_height`` was 0.55, which sat above the lowest scoring body even in a
    neutral stance (0.505 m measured), so it charged 5.8 of task reward per
    round for standing normally. At 0.45 the margin is zero until a body
    actually drops, and the 0.15 m band down to the fall line makes the ramp
    four times steeper: 0.40 m already costs 20 per round instead of 2.5, so
    the term now reacts to danger rather than to posture.

    ``exponent`` exists because that same steepness is a posture tax again once
    a boxing crouch is on the table. With a square, sitting 5 cm lower (0.40 m)
    costs 20 per round while the points it saves are worth 1.6 each; the
    cheapest response is to stand up. A quartic charges 2.2 there and 102 at
    0.32 m against the square's 135, i.e. it keeps most of the wall but makes
    the top of the band nine times cheaper -- it separates "crouching" from
    "falling" instead of charging both at the same rate. Default 2.0 is the old
    shape.
    """
    agent = env.scene[asset_cfg.name]
    floor = env.physics.floor_height
    body_ids = [agent.body_names.index(name) for name in FALL_BODY_NAMES]
    height = agent.data.body_pos_w[:, body_ids, 2] - floor
    span = max(safe_height - FALL_HEIGHT_M, 1e-6)
    margin = ((safe_height - height).clamp_min(0.0) / span).pow(exponent)
    return margin.max(dim=-1).values


def upright_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    free_tilt_deg: float = 20.0,
) -> torch.Tensor:
    """Penalty on torso tilt away from world up, past a free stance band.

    A boxing stance leans, and the previous version charged for the lean
    itself. At update 410 the mean torso tilt is 16.4 deg and this term is
    billing 4.9 of task reward per round for it -- on every step, whether or
    not a punch is being thrown. Measured against the rest of the table that
    is not a fall signal, it is a posture tax: task reward net of penalties
    fell from 12.0 to 4.3 per round while the raw income rose from 41 to 50,
    which is why 80% of the gradient was AMP imitation rather than boxing.

    The free band stops charging below 20 deg. A jab leans more than that and
    has to, while 60 deg still costs 0.44 per step (53 per round at weight
    -2.0), so the fall signal is untouched and only the standing tax is gone.
    """
    agent = env.scene[asset_cfg.name]
    torso_idx = agent.body_names.index(STRIKE_TARGET)
    quat = agent.data.body_quat_w[:, torso_idx]
    up = torch.zeros_like(quat[:, :3])
    up[:, 2] = 1.0
    tilt = (1.0 - quat_apply(quat, up)[:, 2]).clamp_min(0.0)
    free = 1.0 - math.cos(math.radians(free_tilt_deg))
    return (tilt - free).clamp_min(0.0)


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

    Priced per hand and summed, like ``impulse_progress_reward``: the referee
    cools each hand separately, so a guard hand that is closing on the target
    is a real second channel, and one hand contributes exactly zero when it is
    idle. The pair therefore spans [0, 2]; the weight carries that, not the
    shape.
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
    return (proximity * drive).sum(dim=-1)


def hand_reach(env, agent: str) -> torch.Tensor:
    """Closest fist to the opponent torso centre, in metres.

    The missing half of the range diagnosis. ``gap_m`` says how far the two
    chests are; this says whether the glove is anywhere near the target, and
    the two can disagree completely: a policy that steps in to 0.80 m and
    throws half-extension jabs reads as "in range" on the chest gap while the
    glove is still 0.45 m from anything it could hurt. Dividing this by the
    glove radius and the torso half-depth gives the actual air gap.
    """
    ego_agent = env.scene[agent]
    opponent_agent = env.get_opponent(agent)
    torso_idx = ego_agent.body_names.index(STRIKE_TARGET)
    hand_ids = [ego_agent.body_names.index(name) for name in HANDS]
    wrist_pos = ego_agent.data.body_pos_w[:, hand_ids, :3]
    target = opponent_agent.data.body_pos_w[:, torso_idx, :3]
    return (target[:, None, :] - wrist_pos).norm(dim=-1).amin(dim=-1)


def punch_speed_by_hand(env, agent: str) -> torch.Tensor:
    """Wrist closing speed towards the opponent torso, per hand, in m/s.

    Shape ``[worlds, hands]``. Kept per hand because the reward side needs the
    speed of the hand that is *actually touching*: the gloves are half a metre
    apart, so a max over both hands lets a right glove resting on the torso pay
    for a fast swing of the left arm.

    The own-torso velocity is subtracted so that leaning -- which drags both
    arms forward -- does not read as a punch.
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
    return closing


def punch_speed(env, agent: str) -> torch.Tensor:
    """Best wrist closing speed towards the opponent torso, in m/s.

    Diagnostic only. The impulse log answers "did it hurt"; this answers "is
    the arm moving at all", and the two failure modes look identical in every
    other number. A 2 m/s fist that misses and a 0.05 m/s fist resting on the
    chest both read as zero impulse, but only one of them is worth waiting on.
    """
    return punch_speed_by_hand(env, agent).amax(dim=-1)


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

