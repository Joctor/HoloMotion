"""Combat reward terms for the paired / NFSP boxing task.

The terms are grouped into four tiers so it is obvious at a glance which terms
produce "effective scoring" (Tier 1 + Tier 2) versus pure regularization
(Tier 3 + Tier 4):

  Tier 1  规则得分      — the match rules themselves (KO +1000, each 5 N·s hit +1)
  Tier 2  冲量/出拳引导 — dense gradients teaching *how* to hit hard and often
  Tier 3  反作弊        — penalties blocking reward hacking (pressing / holding)
  Tier 4  稳定/安全      — penalties for balance, joint limits, action smoothness

Only Tier-2 terms are ever shape-scaled (see MujocoSandbagEnv.SHAPING_TERMS);
Tier-1/3/4 stay at constant weight.
"""

from holoboxing.training.envs.entities import SceneEntityCfg
from holoboxing.training.tasks.combat.mdp import rewards
from holoboxing.training.tasks.combat.config.sandbag import (
    BODY_NAMES,
    DEFAULTS,
    JOINT_NAMES,
)

LOWER_BODY_JOINTS = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)
LOWER_BODY_ACTION_INDICES = tuple(
    JOINT_NAMES.index(name) for name in LOWER_BODY_JOINTS
)
UPPER_WAIST_ACTION_INDICES = tuple(
    index
    for index, name in enumerate(JOINT_NAMES)
    if name not in LOWER_BODY_JOINTS
)


def entity(name, bodies=None):
    return SceneEntityCfg(
        name,
        body_names=bodies,
        body_ids=[BODY_NAMES.index(n) for n in bodies]
        if bodies
        else slice(None),
    )


def joint_limits(env, asset_cfg):
    data = env.scene[asset_cfg.name].data
    q, limits = data.joint_pos, data.soft_joint_pos_limits
    return (
        (limits[..., 0] - q).clamp_min(0) + (q - limits[..., 1]).clamp_min(0)
    ).sum(-1)


def terms(agent, *, symmetric=False):
    own = entity(agent)
    other = "robot_B" if agent == "robot_A" else "robot_A"

    # 非对抗兜底（paired 非 competitive 模式）：robot_B 只练站立平衡。
    # NFSP competitive_selfplay=True 时 symmetric=True，两个 agent 都走主列表。
    if agent == "robot_B" and not symmetric:
        return [
            ("stand_still", 1.0, rewards.stand_still_reward, dict(asset_cfg=own)),
            (
                "balance",
                1.0,
                rewards.balance_reward,
                dict(asset_cfg=own, upright_threshold=0.7, with_angular=True),
            ),
        ]

    return [
        # ── Tier 1 · 规则得分（比赛规则本体，是"有效得分"的来源）──────────
        ("ko_win",           1000.0, rewards.ko_win_reward,      dict(asset_cfg=own)),
        ("ko_loss",         -1000.0, rewards.ko_loss_penalty,    dict(asset_cfg=own)),
        # 20ms 窗口单手累计法向冲量 >=5 N·s 得 1 分（200ms 冷却）；速度因子让
        # 慢速贴人得分打折、快速出拳拿满，拉开快慢收益差。
        ("scoring_impulse",   100.0, rewards.scoring_impulse_reward,
            dict(asset_cfg=own, impulse_threshold=5.0,
                 speed_bonus_scale=1.0, speed_ref=0.5, speed_floor=0.3)),
        ("conceded_impulse", -50.0, rewards.conceded_impulse_penalty,
            dict(asset_cfg=own, impulse_threshold=5.0)),

        # ── Tier 2 · 冲量/出拳引导（教"怎么打"的稠密梯度）──────────────
        # 肘伸展（刺拳 vs 抡臂）：奖励腕相对肩的径向伸展速度，门控"手臂指向对手"。
        ("kinetic_chain_l", 1.0, rewards.kinetic_chain_reward,
            dict(asset_cfg=own, chain=("left_shoulder_pitch_link",
                                       "left_elbow_link",
                                       "left_wrist_yaw_link"))),
        ("kinetic_chain_r", 1.0, rewards.kinetic_chain_reward,
            dict(asset_cfg=own, chain=("right_shoulder_pitch_link",
                                       "right_elbow_link",
                                       "right_wrist_yaw_link"))),
        # 拳面速度方向对齐接触法线（"手腕转正拳面"）。
        ("normal_alignment", 1.0, rewards.normal_alignment_reward,
            dict(asset_cfg=own)),
        # 躯干前顶（有效质量）：接触帧奖励躯干朝对手的前进速度。
        ("body_drive", 3.0, rewards.body_drive_reward, dict(asset_cfg=own)),
        # 肩膀前顶（有效质量）：奖励肩朝对手前进速度，门控手臂指向对手。
        # body_drive(躯干) + shoulder_drive(肩) + kinetic_chain(肘) = 完整动力链。
        ("shoulder_drive", 2.0, rewards.shoulder_drive_reward, dict(asset_cfg=own)),
        # 接触瞬间沿法线的撞击速度（真正产生冲量的量）；须排在 impulse_progress
        # 之前（后者读其缓存的 last_fast_hit_mask）。
        ("tip_impact_speed", 5.0, rewards.tip_impact_speed_reward,
            dict(asset_cfg=own, near_dist=0.4, speed_gate=0.8,
                 fast_hit_speed=0.5)),
        # 单拳冲量稠密进度：直接奖励规则量逼近 5 N·s（填 scoring 悬崖前的梯度）。
        # power=1.0 线性梯度，避免二次方在低冲量段把梯度压到近零。
        ("impulse_progress", 15.0, rewards.impulse_progress_reward,
            dict(asset_cfg=own, power=1.0, require_fast_hit=True)),
        # 收拳循环：让"重新累计冲量"成为可能（连续得分的前提）。
        ("complete_punch", 2.0, rewards.complete_punch_reward,
            dict(asset_cfg=own, max_steps=40, v_start_thresh=0.4,
                 base_reward=1.0, retract_ratio=0.2, retract_scale=0.5)),
        ("facing", 1.0, rewards.facing_reward,
            dict(asset_cfg=own, target_body_name="torso_link", sigma=0.5)),

        # ── Tier 3 · 反作弊（负项，堵 reward hacking）────────────────────
        # 反"伸手贴住对手累计冲量"：持续亚阈值接触要受罚。
        ("sustained_contact", -2.0, rewards.sustained_contact_penalty,
            dict(asset_cfg=own, window_steps=4, substep_dt=0.005,
                 cooldown_steps=10, decay=0.5)),
        # 反"慢速贴人"：接触已建立但相对速度趋零（纯运动学，不误伤快速命中）。
        ("slow_press", -2.0, rewards.slow_press_penalty,
            dict(asset_cfg=own, contact_dist=0.22, rel_speed_thresh=0.4,
                 cooldown_steps=4)),

        # ── Tier 4 · 稳定/安全（负项）────────────────────────────────────
        ("action_rate", -0.02, rewards.custom_action_rate_l2,
            dict(asset_cfg=own)),
        ("joint_limits", -0.5, joint_limits, dict(asset_cfg=own)),
        ("capture_point", -1.0, rewards.capture_point_penalty,
            dict(asset_cfg=own, foot_half_length=0.093, foot_half_width=0.036,
                 g=9.81, max_penalty=1.0,
                 foot_local_center=(0.039, 0.0, -0.025))),
        # 倒地过程惩罚（独立于 KO 终局）。ko_loss -1000 仍兜底"倒地=被 KO"。
        ("fallen_penalty", -12.0, rewards.fallen_penalty, dict(asset_cfg=own)),
    ]
