from __future__ import annotations
import math
from typing import List

import torch
from holoboxing.evaluation.runtime.fall_rules import FALL_HEIGHT_M
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

def s_stance_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hip_pitch_range=(0.35, 0.9),      # rad，髋屈曲
    knee_range=(0.5, 1.2),            # rad，膝屈曲
    ankle_range=(0.1, 0.5),           # rad，踝背屈
    pelvis_min_height=0.35,           # m，高于倒地阈值
    pelvis_max_height=0.55,
    sigma=0.2,
) -> torch.Tensor:
    ego = env.scene[asset_cfg.name]
    # 左右髋/膝/踝 pitch
    names = ["left_hip_pitch_joint", "right_hip_pitch_joint",
             "left_knee_joint", "right_knee_joint",
             "left_ankle_pitch_joint", "right_ankle_pitch_joint"]
    idx = [ego.joint_names.index(n) for n in names]
    q = ego.data.joint_pos[:, idx]  # [B, 6]

    def band_score(x, lo, hi):
        # 在 [lo,hi] 内为 1，外侧指数衰减
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        return torch.exp(-((x - center).abs() - half).clamp_min(0).square() / sigma**2)

    hip = band_score(q[:, 0:2], *hip_pitch_range).mean(-1)
    knee = band_score(q[:, 2:4], *knee_range).mean(-1)
    ankle = band_score(q[:, 4:6], *ankle_range).mean(-1)

    pelvis_idx = ego.body_names.index("pelvis")
    torso_idx = ego.body_names.index("torso_link")
    effective_h = torch.minimum(
    ego.data.body_pos_w[:, pelvis_idx, 2],
    ego.data.body_pos_w[:, torso_idx, 2],
    ) - env.physics.floor_height
    height_ok = torch.sigmoid((effective_h - pelvis_min_height) * 20.0) * \
            torch.sigmoid((pelvis_max_height - effective_h) * 20.0)

    return hip * knee * ankle * height_ok

def punch_tip_speed_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_body_names: tuple[str, ...] = ("torso_link",),
    sigma: float = 0.5,
    vel_threshold: float = 0.8,
    gate_temperature: float = 10.0,
) -> torch.Tensor:
    """奖励腕部相对躯干、沿腕→目标方向的末端速度。"""
    ego = env.scene[asset_cfg.name]
    opp = env.get_opponent(asset_cfg.name)
    torso_idx = ego.body_names.index("torso_link")
    hand_ids = [
        ego.body_names.index(n)
        for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")
    ]
    tgt_ids = [opp.body_names.index(n) for n in target_body_names]

    torso_vel = ego.data.body_vel_w[:, torso_idx, :3]
    wrist_vel = ego.data.body_vel_w[:, hand_ids, :3] - torso_vel[:, None, :]
    wrist_pos = ego.data.body_pos_w[:, hand_ids, :3]
    tgt_pos = opp.data.body_pos_w[:, tgt_ids, :3]

    diff = tgt_pos[:, None] - wrist_pos[:, :, None]       # [B,H,T,3]
    dist = diff.norm(dim=-1)                               # [B,H,T]
    dirn = diff / (dist.unsqueeze(-1) + 1e-6)
    proj = (wrist_vel[:, :, None] * dirn).sum(-1)          # [B,H,T]

    gate = torch.sigmoid((proj - vel_threshold) * gate_temperature)
    near = torch.exp(-dist.amin(dim=-1) / sigma)           # [B,H]
    return (gate.amax(dim=-1) * near).sum(dim=-1)

def retract_speed_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    v_start_thresh: float = 0.75,
    max_steps: int = 40,
    retract_ratio: float = 0.3,
) -> torch.Tensor:
    """复用 complete_punch_reward 的相位状态，奖励收拳阶段的负径向速度。"""
    device = env.device
    num_envs = env.num_envs
    ego = env.scene[asset_cfg.name]

    torso_idx = ego.body_names.index("torso_link")
    hand_ids = [
        ego.body_names.index(n)
        for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")
    ]

    torso_pos = ego.data.body_pos_w[:, torso_idx, :3]
    torso_vel = ego.data.body_vel_w[:, torso_idx, :3]
    hand_pos = ego.data.body_pos_w[:, hand_ids, :3]
    hand_vel = ego.data.body_vel_w[:, hand_ids, :3]

    vec = hand_pos - torso_pos[:, None, :]
    vec_xy = vec[:, :, :2]
    dir_xy = vec_xy / (vec_xy.norm(dim=-1, keepdim=True) + 1e-6)
    rel_vel_xy = (hand_vel - torso_vel[:, None, :])[:, :, :2]
    radial_vel = (rel_vel_xy * dir_xy).sum(-1)              # [B,2]

    if not hasattr(env, "punch_state"):
        env.punch_state = {}
    if asset_cfg.name not in env.punch_state:
        env.punch_state[asset_cfg.name] = {
            "phase": torch.zeros(num_envs, 2, device=device, dtype=torch.long),
            "steps": torch.zeros(num_envs, 2, device=device, dtype=torch.long),
            "max_pos_speed": torch.zeros(num_envs, 2, device=device),
        }
    state = env.punch_state[asset_cfg.name]
    phase, steps, max_pos = state["phase"], state["steps"], state["max_pos_speed"]

    reward = torch.zeros(num_envs, 2, device=device)
    for h in range(2):
        ph, st, mp = phase[:, h], steps[:, h], max_pos[:, h]
        start = (ph == 0) & (radial_vel[:, h] > v_start_thresh)
        ph[start] = 1; st[start] = 0; mp[start] = radial_vel[start, h]
        active = ph == 1
        st[active] += 1
        mp[active] = torch.maximum(mp[active], radial_vel[active, h])
        retract = active & (radial_vel[:, h] < 0)
        strong = retract & ((-radial_vel[:, h]) >= retract_ratio * mp)
        reward[:, h] = (-radial_vel[:, h]).clamp_min(0) * strong.float()
        done = active & (st > max_steps)
        reset = done
        ph[reset] = 0; st[reset] = 0; mp[reset] = 0.0
        phase[:, h], steps[:, h], max_pos[:, h] = ph, st, mp
    return reward.sum(dim=-1)

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


def _per_hand_normal_impulse(env, defender, window_steps, substep_dt):
    data = env.scene.sensors[f"contact_sensor_{defender}_torso"].data
    if not hasattr(data, "normal_force_w_history"):
        raise ValueError("Missing paired hand-to-head/torso normal-force history")
    normal_force = data.normal_force_w_history
    if (
        normal_force.ndim != 3
        or normal_force.shape[0] != env.num_envs
        or normal_force.shape[1] < window_steps
        or normal_force.shape[2] != 2
    ):
        raise ValueError("Expected [num_envs, physics_history, 2] normal forces")
    if window_steps <= 0 or substep_dt <= 0:
        raise ValueError("Impulse window and physics timestep must be positive")
    return normal_force[:, :window_steps].sum(dim=1) * substep_dt

def impulse_progress_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_impulse: float = 3.5,       # fallback
    window_steps: int = 4,
    substep_dt: float = 0.005,
    min_impulse: float = 0.5,
    power: float = 4.0,
    lead_ratio: float = 1.2,           # progress 目标比 scoring 阈值高 20%
) -> torch.Tensor:
    opponent = env.get_opponent_id(asset_cfg.name)
    impulse = _per_hand_normal_impulse(env, opponent, window_steps, substep_dt)

    threshold = getattr(env, "impulse_threshold", target_impulse)
    target = max(threshold * lead_ratio, 1.0)

    active = (impulse > min_impulse).float()
    progress = torch.clamp(impulse / target, max=1.0)
    reward = (progress ** power) * active
    return reward.sum(dim=-1)

def sustained_contact_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_steps: int = 4,
    substep_dt: float = 0.005,
    hold_impulse: float | None = None,   # N·s; None -> 50% of the curriculum threshold
    hold_ratio: float = 0.5,             # fraction of threshold when hold_impulse is None
    hold_floor: float = 0.5,             # minimum absolute value
    cooldown_steps: int = 10,
    decay: float = 0.5,
) -> torch.Tensor:
    """Penalize holding the glove against the opponent without scoring.

    A hand that keeps a *sub-threshold* normal impulse for many consecutive
    steps (arm extended, glove pressed on the opponent) is the exact
    degenerate behaviour ``impulse_progress_reward`` used to encourage.
    We make it costly instead: once a hand has maintained >= ``hold_impulse``
    for a full cooldown window without ever clearing the scoring threshold,
    it is charged a penalty that resets only after contact drops.

    ``hold_impulse`` defaults to ``hold_ratio * threshold`` so the "holding"
    band [hold_impulse, threshold) never collapses during the curriculum
    (a fixed 2.0 would make the band empty once the curriculum threshold
    drops below 2.0 early in training).

    Returns a *non-negative* penalty (use a negative weight in the config).
    """
    opponent = env.get_opponent_id(asset_cfg.name)
    impulse = _per_hand_normal_impulse(env, opponent, window_steps, substep_dt)

    threshold = getattr(env, "impulse_threshold", 5.0)
    if hold_impulse is None:
        hold_impulse = max(hold_ratio * threshold, hold_floor)

    if not hasattr(env, "score_state"):
        env.score_state = {}
    ego = asset_cfg.name
    if ego not in env.score_state:
        env.score_state[ego] = {}
    if "hold_steps" not in env.score_state[ego]:
        env.score_state[ego]["hold_steps"] = torch.zeros(
            env.num_envs, 2, device=env.device, dtype=torch.long
        )

    hold_steps = env.score_state[ego]["hold_steps"]

    # A hand is "holding" when it has a real but sub-threshold impulse.
    holding = (impulse >= hold_impulse) & (impulse < threshold)
    hold_steps = torch.where(holding, hold_steps + 1, torch.zeros_like(hold_steps))
    env.score_state[ego]["hold_steps"] = hold_steps

    penalty = (hold_steps >= cooldown_steps).float()
    return penalty.sum(dim=-1) * decay


def slow_press_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_body_names: tuple[str, ...] = ("torso_link",),
    contact_dist: float = 0.22,       # m, glove this close to target counts as "in contact"
    rel_speed_thresh: float = 0.4,    # m/s, relative speed below this = "pressing, not punching"
    gate_temperature: float = 10.0,
    cooldown_steps: int = 4,          # must persist this many frames to charge
) -> torch.Tensor:
    """Penalize pressing the glove against the opponent without striking.

    ``sustained_contact_penalty`` only catches impulses that linger in the
    [hold_impulse, threshold) band; a policy that slowly builds impulse past
    the scoring threshold skips that band entirely, so it is never charged.
    This term instead reads pure kinematics: when the glove is already in
    contact (close to the target) but its *relative* speed toward the target
    has collapsed (~0), the hand is pressing, not punching. A real punch has
    high approach speed at contact and is gated out, so fast hits are never
    penalized.

    Returns a *non-negative* penalty (use a negative weight in the config).
    """
    ego = env.scene[asset_cfg.name]
    opp = env.get_opponent(asset_cfg.name)
    hand_ids = [
        ego.body_names.index(n)
        for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")
    ]
    tgt_ids = [opp.body_names.index(n) for n in target_body_names]

    # Relative velocity: glove minus target (torso). [B, 2, 3]
    glove_vel = ego.data.body_lin_vel_w[:, hand_ids, :3]
    tgt_vel = opp.data.body_lin_vel_w[:, tgt_ids, :3]       # [B, T, 3]
    rel_vel = glove_vel[:, :, None] - tgt_vel[:, None]      # [B, 2, T, 3]
    rel_speed = rel_vel.norm(dim=-1)                        # [B, 2, T]

    glove_pos = ego.data.body_pos_w[:, hand_ids, :3]
    tgt_pos = opp.data.body_pos_w[:, tgt_ids, :3]
    dist = (tgt_pos[:, None] - glove_pos[:, :, None]).norm(dim=-1)  # [B,2,T]

    # Best (closest) target for each hand.
    best_dist = dist.amin(dim=-1)                          # [B, 2]
    best_speed = rel_speed.gather(-1, dist.argmin(dim=-1, keepdim=True)).squeeze(-1)

    # Contact gate: glove is effectively touching.
    contact = torch.sigmoid((contact_dist - best_dist) * 15.0)  # ~1 when dist < contact_dist
    # Slow gate: relative speed collapsed -> pressing.
    slow = torch.sigmoid((rel_speed_thresh - best_speed) * gate_temperature)

    pressing = contact * slow                              # [B, 2]

    # Persist for cooldown_steps before charging, so a single slow frame at
    # contact (e.g. the instant of a legitimate hit) is not punished.
    if not hasattr(env, "score_state"):
        env.score_state = {}
    if ego not in env.score_state:
        env.score_state[ego] = {}
    if "press_steps" not in env.score_state[ego]:
        env.score_state[ego]["press_steps"] = torch.zeros(
            env.num_envs, 2, device=env.device, dtype=torch.long
        )
    press_steps = env.score_state[ego]["press_steps"]
    press_steps = torch.where(
        pressing > 0.5, press_steps + 1, torch.zeros_like(press_steps)
    )
    env.score_state[ego]["press_steps"] = press_steps

    penalty = (press_steps >= cooldown_steps).float()
    return penalty.sum(dim=-1)


def kinetic_chain_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    chain: tuple[str, str, str] = (
        "left_shoulder_pitch_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
    ),
    sigma: float = 0.8,
    gate_temperature: float = 8.0,
) -> torch.Tensor:
    """Dense reward for the proximal-to-distal (shoulder -> elbow -> wrist)
    acceleration sequence that produces an effective punch impulse.

    This is the *process* signal ``impulse_progress_reward`` was meant to be:
    it reads only kinematics, never the normal impulse, so it cannot be farmed
    by extending the arm and holding the glove against the opponent.

    It rewards the tip (distal) link moving *linearly* faster than the
    proximal link: the tip's linear velocity is the sum of every upstream
    joint's angular velocity scaled by its moment arm, so a monotonic
    proximal -> distal linear-speed gradient means momentum is being
    transferred outward (whip effect). Linear speed is used rather than raw
    angular speed because the tip velocity (m·v) is what becomes the impulse.
    """
    ego = env.scene[asset_cfg.name]
    ids = [ego.body_names.index(n) for n in chain]

    # Linear velocity magnitude of each link along the chain (proximal->distal).
    lin_vel = ego.data.body_lin_vel_w[:, ids, :3]   # [B, 3, 3]
    speed = lin_vel.norm(dim=-1)                    # [B, 3]

    # Gradient: distal faster than proximal, summed across the two joints.
    grad = (speed[:, 1] - speed[:, 0]) + (speed[:, 2] - speed[:, 1])  # [B]

    # Gate on the tip actually moving fast enough to transfer momentum.
    tip_speed = speed[:, 2]
    gate = torch.sigmoid((tip_speed - 0.75) * gate_temperature)

    return gate * torch.sigmoid(grad / sigma)


def normal_alignment_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_body_names: tuple[str, ...] = ("torso_link",),
    sigma: float = 0.5,
    speed_gate: float = 1.0,
) -> torch.Tensor:
    """Dense reward for pointing the glove velocity along the contact normal.

    The match rule scores only the *normal* component of impulse, so a glove
    that slams in at an angle wastes most of its momentum (F_normal = F·cosθ).
    This term rewards the wrist moving toward the nearest target surface, i.e.
    the wrist aligning the punch with the normal. It reads velocity and
    position only, never impulse, so it does not induce pushing-and-holding.
    """
    ego = env.scene[asset_cfg.name]
    opp = env.get_opponent(asset_cfg.name)
    hand_ids = [
        ego.body_names.index(n)
        for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")
    ]
    tgt_ids = [opp.body_names.index(n) for n in target_body_names]

    wrist_vel = ego.data.body_vel_w[:, hand_ids, :3]  # [B,2,3]
    wrist_pos = ego.data.body_pos_w[:, hand_ids, :3]  # [B,2,3]
    tgt_pos = opp.data.body_pos_w[:, tgt_ids, :3]     # [B,T,3]

    diff = tgt_pos[:, None] - wrist_pos[:, :, None]   # [B,2,T,3]
    dist = diff.norm(dim=-1)                          # [B,2,T]
    normal = diff / (dist.unsqueeze(-1) + 1e-6)       # wrist -> target

    speed = wrist_vel.norm(dim=-1)                    # [B,2]
    vel_dir = wrist_vel / (speed.unsqueeze(-1) + 1e-6)

    # cos between velocity and each target normal, take best-matching target.
    cos = (vel_dir[:, :, None] * normal).sum(-1)      # [B,2,T]
    best_cos = cos.amax(dim=-1)                       # [B,2]

    # Gate: only reward a punch that is actually moving (not a held glove).
    gate = torch.sigmoid((speed - speed_gate) * 5.0)
    align = best_cos.clamp_min(0)  # [0,1]

    return (gate * align).sum(dim=-1)


def tip_impact_speed_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_body_names: tuple[str, ...] = ("torso_link",),
    near_dist: float = 0.25,       # m, glove must be this close to count as "about to hit"
    speed_gate: float = 1.5,       # m/s, minimum normal approach speed (dense-reward gate)
    fast_hit_speed: float = 0.3,   # m/s, looser threshold for the "fast hit" quality gate
    gate_temperature: float = 4.0,
) -> torch.Tensor:
    """Dense reward for the glove carrying high *approach speed along the
    contact normal* just before it reaches the opponent.

    This is the quantity that actually produces the scored normal impulse:
    the glove transfers its momentum (m·v) into a normal force over the
    contact window, so the normal component of the tip velocity immediately
    *before* contact is what the policy can control to maximise impulse.

    It is deliberately gated on proximity so that a glove standing far away
    and swinging in empty space scores nothing, and it reads velocity/position
    only (never the impulse itself), so it cannot be farmed by pressing the
    glove against the opponent and holding still.
    """
    ego = env.scene[asset_cfg.name]
    opp = env.get_opponent(asset_cfg.name)
    hand_ids = [
        ego.body_names.index(n)
        for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")
    ]
    tgt_ids = [opp.body_names.index(n) for n in target_body_names]

    wrist_vel = ego.data.body_vel_w[:, hand_ids, :3]   # [B,2,3]
    wrist_pos = ego.data.body_pos_w[:, hand_ids, :3]   # [B,2,3]
    tgt_pos = opp.data.body_pos_w[:, tgt_ids, :3]      # [B,T,3]

    diff = tgt_pos[:, None] - wrist_pos[:, :, None]    # [B,2,T,3]
    dist = diff.norm(dim=-1)                           # [B,2,T]
    normal = diff / (dist.unsqueeze(-1) + 1e-6)        # wrist -> target

    # Normal (approach) component of the tip velocity toward each target.
    v_normal = (wrist_vel[:, :, None] * normal).sum(-1)  # [B,2,T]
    best_v = v_normal.amax(dim=-1)                       # [B,2] best target

    # Proximity gate: only "about to contact" counts. Close glove -> ~1.
    best_dist = dist.amin(dim=-1)                        # [B,2]
    near = torch.sigmoid((near_dist - best_dist) * 10.0) # ~1 when dist < near_dist

    # Speed gate: approach must be fast enough to matter.
    speed_gate_val = torch.sigmoid((best_v - speed_gate) * gate_temperature)

    # ★ 缓存"快速命中"质量闸门 mask，带短时记忆：快速接近 → 接触 → 冲量
    # 达标是先后关系而非同时关系。若要求同一帧同时满足 fast_hit 和
    # impulse_ok，会漏掉真正的快速命中（接近帧冲量未达标、达标帧速度已
    # 归零）。故用 recent_fast_hit 计数保留最近几帧的"快速接近"信号，
    # 供课程判定"最近 fast_hit_frames 帧内快速接近过 且 当前冲量达标"。
    if not hasattr(env, "score_state"):
        env.score_state = {}
    if ego not in env.score_state:
        env.score_state[ego] = {}
    fast_hit_now = (best_v >= fast_hit_speed) & (best_dist <= near_dist)
    fast_hit_frames = 3  # 快速接近信号保留的帧数
    key = "recent_fast_hit"
    if key not in env.score_state[ego]:
        env.score_state[ego][key] = torch.zeros(
            env.num_envs, 2, device=env.device, dtype=torch.long
        )
    recent = env.score_state[ego][key]
    recent = torch.where(
        fast_hit_now, fast_hit_frames, (recent - 1).clamp_min(0)
    )
    env.score_state[ego][key] = recent
    env.score_state[ego]["last_fast_hit_mask"] = (recent > 0).detach()

    # Reward is the fast, close, and pointing-inward approach speed.
    return (near * speed_gate_val * best_v.clamp_min(0)).sum(dim=-1)


def scoring_impulse_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    impulse_threshold: float = 5.0,
    window_steps: int = 4,
    substep_dt: float = 0.005,
    cooldown_steps: int = 10,
) -> torch.Tensor:
    ego = asset_cfg.name
    opponent = env.get_opponent_id(ego)
    per_hand_impulse = _per_hand_normal_impulse(
        env, opponent, window_steps, substep_dt
    )

    # ★ 优先读课程动态阈值，配置里的 5.0 只作 fallback
    threshold = getattr(env, "impulse_threshold", impulse_threshold)

    if not hasattr(env, "score_state"):
        env.score_state = {}
    if ego not in env.score_state:
        env.score_state[ego] = {}
    if "cooldown" not in env.score_state[ego]:
        env.score_state[ego]["cooldown"] = torch.zeros(
            env.num_envs, 2, device=env.device, dtype=torch.long
        )
    cooldown = env.score_state[ego]["cooldown"]

    impulse_ok = per_hand_impulse >= threshold
    score = impulse_ok & (cooldown == 0)
    cooldown[score] = cooldown_steps
    cooldown.sub_(1).clamp_(min=0)

    # ★ 缓存原始 mask，供 curriculum 读取（不受 reward weight 符号影响）
    env.score_state[ego]["last_score_mask"] = score.detach()
    env.score_state[ego]["last_impulse_ok_mask"] = impulse_ok.detach()

    return score.float().sum(dim=-1)

def conceded_impulse_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    impulse_threshold: float = 5.0,
    window_steps: int = 4,
    substep_dt: float = 0.005,
    cooldown_steps: int = 10,
) -> torch.Tensor:
    """Penalize one point for each opponent hand scoring under match rules."""
    ego = asset_cfg.name
    per_hand_impulse = _per_hand_normal_impulse(
        env, ego, window_steps, substep_dt
    )

    threshold = getattr(env, "impulse_threshold", impulse_threshold)

    if not hasattr(env, "score_state"):
        env.score_state = {}
    if ego not in env.score_state:
        env.score_state[ego] = {}
    if "conceded_cooldown" not in env.score_state[ego]:
        env.score_state[ego]["conceded_cooldown"] = torch.zeros(
            env.num_envs, 2, device=env.device, dtype=torch.long
        )
    cooldown = env.score_state[ego]["conceded_cooldown"]

    conceded = (per_hand_impulse >= threshold) & (cooldown == 0)
    cooldown[conceded] = cooldown_steps
    cooldown.sub_(1).clamp_(min=0)
    return conceded.float().sum(dim=-1)

def reset_punch_state(env, env_ids):
    for state in getattr(env, "punch_state", {}).values():
        for value in state.values():
            value[env_ids] = 0
    for state in getattr(env, "score_state", {}).values():
        for value in state.values():
            value[env_ids] = 0
    for state in getattr(env, "approach_potential_state", {}).values():
        for value in state.values():
            value[env_ids] = 0


def custom_action_rate_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_indices: tuple[int, ...] | None = None,
) -> torch.Tensor:
    action_manager = (
        env.action_managers[asset_cfg.name]
        if hasattr(env, "action_managers")
        else env.action_manager
    )
    action_delta = action_manager.action - action_manager.prev_action
    if action_indices is not None:
        action_delta = action_delta[:, action_indices]
    return torch.square(action_delta).sum(dim=1)


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

def capture_point_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_half_length: float = 0.093,
    foot_half_width: float = 0.036,
    g: float = 9.81,
    max_penalty: float = 1.0,
    foot_local_center: tuple[float, float, float] = (0.039, 0.0, -0.025),
) -> torch.Tensor:
    """Non-negative capture-point excursion outside the support polygon.

    Returns 0 while the CP is inside the approximate support AABB, and the
    Euclidean distance (metres) from the CP to that AABB otherwise. Use a
    negative weight in the reward config, e.g. ``weight=-2.0``.

    The support polygon is approximated by an axis-aligned bounding box
    around the two foot centres, each expanded by the foot half-extents.
    Foot centres are the ankle-roll link origins plus the local collision
    centre offset (0.039, 0.0, -0.025) rotated by the link orientation.
    """
    device = env.device
    ego = env.scene[asset_cfg.name]

    # --- 1. CoM proxy: torso link position / velocity (world frame) ---
    torso_idx = ego.body_names.index("torso_link")
    torso_pos = ego.data.body_pos_w[:, torso_idx, :3]      # [B, 3]
    torso_vel = ego.data.body_vel_w[:, torso_idx, :3]      # [B, 3]

    com_xy = torso_pos[:, :2]
    com_vel_xy = torso_vel[:, :2]

    # --- 2. Capture point: xi = x + v / omega, omega = sqrt(g / h) ---
    h = torch.clamp(torso_pos[:, 2], min=0.2)              # avoid blow-up
    omega = torch.sqrt(torch.full_like(h, g) / h)          # [B]
    cp_xy = com_xy + com_vel_xy / omega[:, None]           # [B, 2]

    # --- 3. Foot centres in world XY (link origin + rotated local offset) ---
    foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_ids = [ego.body_names.index(n) for n in foot_names]
    foot_pos = ego.data.body_pos_w[:, foot_ids, :3]        # [B, 2, 3]
    foot_quat = ego.data.body_quat_w[:, foot_ids]          # [B, 2, 4] wxyz

    offset = torch.tensor(foot_local_center, device=device)  # [3]
    offset_w = quat_apply(
        foot_quat.reshape(-1, 4),
        offset.unsqueeze(0).expand(foot_quat.shape[0] * 2, -1),
    ).reshape(-1, 2, 3)
    foot_center = foot_pos + offset_w                      # [B, 2, 3]
    foot_xy = foot_center[:, :, :2]                        # [B, 2, 2]

    # --- 4. Axis-aligned support AABB around both feet ---
    half = torch.tensor(
        [foot_half_length, foot_half_width], device=device
    )
    min_xy = foot_xy.min(dim=1).values - half              # [B, 2]
    max_xy = foot_xy.max(dim=1).values + half              # [B, 2]

    # --- 5. Excursion of CP outside the AABB ---
    below = torch.clamp(min_xy - cp_xy, min=0.0)
    above = torch.clamp(cp_xy - max_xy, min=0.0)
    dist_out = torch.norm(below + above, dim=-1)           # [B]

    # Clamp so a single wild frame cannot dominate the return.
    return torch.clamp(dist_out, max=max_penalty)

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


def approach_potential_reward(
    env,
    asset_cfg: SceneEntityCfg,
    hit_body_names: tuple[str, ...] = (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ),
    target_body_names: tuple[str, ...] = ("torso_link",),
    sigma: float = 1.0,
    vel_threshold: float = 0.8,
    gate_temperature: float = 10.0,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Reward discounted changes in active, self-initiated approach potential."""
    if not hit_body_names or not target_body_names:
        raise ValueError("Approach shaping requires hands and target bodies")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(vel_threshold):
        raise ValueError("vel_threshold must be finite")
    if not math.isfinite(gate_temperature) or gate_temperature <= 0:
        raise ValueError("gate_temperature must be finite and positive")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and between zero and one")

    ego = env.scene[asset_cfg.name]
    opponent = env.get_opponent(asset_cfg.name)
    torso_idx = ego.body_names.index("torso_link")
    hand_ids = [ego.body_names.index(name) for name in hit_body_names]
    target_ids = [
        opponent.body_names.index(name) for name in target_body_names
    ]

    wrist_pos = ego.data.body_pos_w[:, hand_ids, :3]
    wrist_vel = (
        ego.data.body_vel_w[:, hand_ids, :3]
        - ego.data.body_vel_w[:, torso_idx, None, :3]
    )
    target_pos = opponent.data.body_pos_w[:, target_ids, :3]

    target_delta = target_pos[:, None, :, :] - wrist_pos[:, :, None, :]
    distance = torch.linalg.vector_norm(target_delta, dim=-1)
    direction = target_delta / distance.unsqueeze(-1).clamp_min(1e-6)
    closing_speed = (wrist_vel[:, :, None, :] * direction).sum(dim=-1)
    target_potential = torch.exp(-distance / sigma) * torch.sigmoid(
        (closing_speed - vel_threshold) * gate_temperature
    )
    potential = target_potential.amax(dim=-1)

    terminal = getattr(env, "reward_terminal_mask", None)
    if (
        terminal is None
        or terminal.shape != (potential.shape[0],)
        or terminal.dtype != torch.bool
        or terminal.device != potential.device
    ):
        raise RuntimeError("Missing or invalid reward_terminal_mask")

    state_by_agent = getattr(env, "approach_potential_state", None)
    if state_by_agent is None:
        env.approach_potential_state = {}
        state_by_agent = env.approach_potential_state
    state = state_by_agent.get(asset_cfg.name)
    if state is None:
        state = {
            "potential": torch.zeros_like(potential),
            "initialized": torch.zeros(
                potential.shape[0], dtype=torch.bool, device=potential.device
            ),
        }
        state_by_agent[asset_cfg.name] = state
    if (
        state["potential"].shape != potential.shape
        or state["potential"].device != potential.device
        or state["initialized"].shape != (potential.shape[0],)
        or state["initialized"].device != potential.device
    ):
        raise RuntimeError("Invalid cached approach potential state")

    next_potential = potential.detach().masked_fill(terminal[:, None], 0)
    shaped = gamma * next_potential - state["potential"]
    shaped = torch.where(
        state["initialized"][:, None], shaped, torch.zeros_like(shaped)
    )
    state["potential"].copy_(next_potential)
    state["initialized"].copy_(~terminal)
    return shaped.mean(dim=-1)

def fallen_penalty(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    ego = asset_cfg.name
    fallen = env.metrics.get(ego, {}).get(
        "fallen", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    return fallen.float()

def opponent_low_reward(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """对手低位：稠密奖励，引导压制对手，靠近 KO 条件。"""
    ego = asset_cfg.name
    opponent = env.get_opponent_id(ego)
    opp_low = env.metrics.get(opponent, {}).get(
        "fallen",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    return opp_low.float()

def terminate_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    minimum_height: float = 0.4,   # 保留形参兼容，但不再使用
) -> torch.Tensor:
    ego = asset_cfg.name
    opponent = env.get_opponent_id(ego)
    ego_fallen = env.metrics.get(ego, {}).get(
        "ko", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ).float()
    opp_fallen = env.metrics.get(opponent, {}).get(
        "ko", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ).float()
    return opp_fallen - ego_fallen

def ko_win_reward(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """对手被 KO 且自己未被 KO：赢。"""
    ego = asset_cfg.name
    opponent = env.get_opponent_id(ego)
    ego_ko = env.metrics.get(ego, {}).get("ko", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    opp_ko = env.metrics.get(opponent, {}).get("ko", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return (opp_ko & ~ego_ko).float()


def ko_loss_penalty(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """自己被 KO 且对手未被 KO：输。"""
    ego = asset_cfg.name
    opponent = env.get_opponent_id(ego)
    ego_ko = env.metrics.get(ego, {}).get("ko", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    opp_ko = env.metrics.get(opponent, {}).get("ko", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return (ego_ko & ~opp_ko).float()


def draw_penalty(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """双方同时 KO：平局不如赢。"""
    ego = asset_cfg.name
    opponent = env.get_opponent_id(ego)
    ego_ko = env.metrics.get(ego, {}).get("ko", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    opp_ko = env.metrics.get(opponent, {}).get("ko", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return (ego_ko & opp_ko).float()


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
