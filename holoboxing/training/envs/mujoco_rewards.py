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

Mode split on ``symmetric`` (== competitive_selfplay):
  * standing (symmetric=False) — "skill class": the opponent is a stationary
    sparring dummy, so the target is fixed and falling risk is low. The correct
    technique is "keep balance + drive the punch from trunk rotation (waist
    yaw, 88 N·m) + whip the arm out (shoulder → elbow → wrist) with a fast
    out-and-back swing". The arm is torque-capped at 25 N·m so a jab's power
    cannot come from arm muscle alone — trunk rotation carries the torso's
    rotational inertia into the glove. No torso *lunge* is needed (and it is
    rejected); trunk *rotation* is the correct power source.
  * NFSP (symmetric=True) — "fight class": the opponent hits back, so balance
    matters more; fallen stays strong and the kinetic-chain guides are moderate.
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

    # standing（技能课）vs NFSP（对抗课）的引导强度差异。打沙包的正确技术是
    # "站好维持平衡 + 肩膀带动手臂鞭打"，不需要躯干前倾——刺拳冲量来自手臂
    # 快速甩出（拳面接触瞬间高速），不是前倾加质量。故引导聚焦鞭打与快速
    # 摆动，standing 强化、NFSP 适中；fallen 仅防倒地。
    kc_w = 2.0 if not symmetric else 1.0       # 肘伸展（刺拳 vs 抡臂）
    whip_w = 3.0 if not symmetric else 2.0     # 鞭打放大（肩带动→末端甩出）
    hip_w = 4.0 if not symmetric else 2.0      # 转体驱动（腰 yaw 88N·m，破线的真正主力）
    ctr_w = 4.0 if not symmetric else 2.0      # 接触瞬间转体爆发（把转体能量送进拳面）
    lock_w = 5.0 if not symmetric else 3.0     # 接触瞬间锁定（有效质量）
    cp_w = 4.0 if not symmetric else 2.0       # 收拳循环（快速前后摆动）
    fallen_w = -8.0 if not symmetric else -12.0

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
        ("kinetic_chain_l", kc_w, rewards.kinetic_chain_reward,
            dict(asset_cfg=own, chain=("left_shoulder_pitch_link",
                                       "left_elbow_link",
                                       "left_wrist_yaw_link"))),
        ("kinetic_chain_r", kc_w, rewards.kinetic_chain_reward,
            dict(asset_cfg=own, chain=("right_shoulder_pitch_link",
                                       "right_elbow_link",
                                       "right_wrist_yaw_link"))),
        # 拳面速度方向对齐接触法线（"手腕转正拳面"）。
        ("normal_alignment", 1.0, rewards.normal_alignment_reward,
            dict(asset_cfg=own)),
        # 鞭打放大（"肩膀带动甩出去"）：奖励腕朝对手速度 > 肩朝对手速度，
        # 天然抑制躯干前倾、奖励从稳定站姿甩出末端；near gate 堵远处空甩。
        ("whipping", whip_w, rewards.whipping_reward,
            dict(asset_cfg=own, near_dist=0.6)),
        # 转体驱动（破线的真正主力）：手臂力矩被钉死 25N·m，单靠手臂有效质量
        # 只有 ~1kg，冲量卡 1.7N·s。腰 yaw 88N·m 是手臂的 3.5 倍，转体把躯干
        # 转动惯量（等效 ~10kg）送进出拳侧肩，冲量才有破 5N·s 的物理可能。
        # 用"两肩朝对手速度差"度量转体：平移(前倾)两肩同速差≈0 不奖励，
        # 转体一肩前送一肩后撤差>0 —— 与之前被否掉的"前倾"严格区分。
        ("hip_rotation", hip_w, rewards.hip_rotation_reward,
            dict(asset_cfg=own, near_dist=0.7)),
        # 接触瞬间转体爆发：hip_rotation 教"接近阶段会转体"，但策略可以在
        # 接触前就转完、能量散掉（iter63 已验证：hip_rotation 0.024、scoring
        # +36%，但 mean_impulse 仍 ~1.6、impact_lock 仅 0.00165，转体能量没
        # 送到拳面）。本项只在接触帧给"两肩速度差"奖励，逼策略"转着打穿
        # 接触瞬间"，让躯干转动惯量(~10kg)通过锁定的手臂传进拳面。与
        # impact_lock（锁定肘腕）配对 = 完整的"转体+锁定"后手直拳。
        ("contact_rotation", ctr_w, rewards.contact_rotation_reward,
            dict(asset_cfg=own)),
        # 接触瞬间锁定（co-contraction）：奖励接触帧内上臂(肘)与前臂(腕)以接近
        # 的速度一起朝对手打进去——整条手臂链刚体化，有效质量=上臂+前臂而非
        # 只有前臂。补"接触瞬间绷紧"这一课（RL 最难的共收缩），冲量破线的关键。
        ("impact_lock", lock_w, rewards.impact_lock_reward,
            dict(asset_cfg=own)),
        # 接触瞬间沿法线的撞击速度（真正产生冲量的量）；须排在 impulse_progress
        # 之前（后者读其缓存的 last_fast_hit_mask）。权重 5→8：破线靠"拳面
        # 接触瞬间高速"，这是比前倾更正确的路径。
        ("tip_impact_speed", 8.0, rewards.tip_impact_speed_reward,
            dict(asset_cfg=own, near_dist=0.4, speed_gate=0.8,
                 fast_hit_speed=0.5)),
        # 单拳冲量稠密进度：直接奖励规则量逼近 5 N·s（填 scoring 悬崖前的梯度）。
        # power=1.0 线性梯度，避免二次方在低冲量段把梯度压到近零。
        ("impulse_progress", 15.0, rewards.impulse_progress_reward,
            dict(asset_cfg=own, power=1.0, require_fast_hit=True)),
        # 收拳循环：让"重新累计冲量"成为可能（连续得分的前提）。
        ("complete_punch", cp_w, rewards.complete_punch_reward,
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
        ("fallen_penalty", fallen_w, rewards.fallen_penalty, dict(asset_cfg=own)),
    ]
