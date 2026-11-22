"""Original sandbag reward functions with named MuJoCo state selections."""

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


def terminate_reward(env, asset_cfg, minimum_height):
    return rewards.terminate_reward(
        env,
        asset_cfg=asset_cfg,
        minimum_height=env.physics.floor_height + minimum_height,
    )


def terms(agent, *, symmetric=False):
    own = entity(agent)
    other = "robot_B" if agent == "robot_A" else "robot_A"
    sensor = entity("contact_sensor_" + agent)
    opponent = entity("contact_sensor_" + other)
    if agent == "robot_B" and not symmetric:
        return [
            (
                "stand_still",
                1.0,
                rewards.stand_still_reward,
                dict(asset_cfg=own),
            ),
            (
                "balance",
                1.0,
                rewards.balance_reward,
                dict(asset_cfg=own, upright_threshold=0.7, with_angular=True),
            ),
        ]
    return [
        # === 终局：对齐规则 1000 分 KO，立即结束 ===
        ("ko_win",           1000.0, rewards.ko_win_reward,        dict(asset_cfg=own)),   # +20
        ("ko_loss",         -1000.0, rewards.ko_loss_penalty,      dict(asset_cfg=own)),   # -20

        # === 得分：精确对齐规则（20ms 法向冲量 ≥5 N·s，200ms 冷却）===
        ("scoring_impulse",   100.0, rewards.scoring_impulse_reward,
            dict(asset_cfg=own, impulse_threshold=5.0,
                 speed_bonus_scale=1.0, speed_ref=0.5, speed_floor=0.3)),
        ("conceded_impulse", -50.0, rewards.conceded_impulse_penalty,
            dict(asset_cfg=own, impulse_threshold=5.0)),

        # === 冲量 shaping：稠密、只读运动学、不诱导"贴人" ===
        # 肩→肘→腕依次加速的"动力链"时序（对应"肩膀带动"）
        ("kinetic_chain_l", 1.0, rewards.kinetic_chain_reward,
            dict(asset_cfg=own, chain=("left_shoulder_pitch_link",
                                       "left_elbow_link",
                                       "left_wrist_yaw_link"))),
        ("kinetic_chain_r", 1.0, rewards.kinetic_chain_reward,
            dict(asset_cfg=own, chain=("right_shoulder_pitch_link",
                                       "right_elbow_link",
                                       "right_wrist_yaw_link"))),
        # 拳面速度方向对齐接触法线（对应"手腕转正拳面"）
        ("normal_alignment", 1.0, rewards.normal_alignment_reward,
            dict(asset_cfg=own)),
        # 接触前末端沿法线的接近速度（真正产生冲量的量）
        # ★ 权重 1.0 → 5.0：它是「近身 + 快速接近」的唯一稠密正确信号，
        # 原 1.0 下每步仅 0.0008，远小于 kinetic_chain(0.031)，无法把
        # diag_near_v_mean 从 0.199 顶过 0.3 闸门。
        # ★ near_dist 0.25 → 0.4、speed_gate 0.5 → 0.3：原门太紧（ego 只在
        # 0.25m 内才收到「接近」梯度，而 diag_near_ratio 只有 0.012，形成
        # 鸡生蛋——不接近就学不到接近）。放宽后 ego 更早收到「朝对手加速」
        # 的稠密梯度，先学会接近，fast_hit 的接近速度门(0.3)不变，贴人仍过不了。
        ("tip_impact_speed", 5.0, rewards.tip_impact_speed_reward,
            dict(asset_cfg=own, near_dist=0.4, speed_gate=0.3,
                 fast_hit_speed=0.3)),
        # 收拳：让"重新累计冲量"成为可能（200ms 冷却规则下连续得分的前提）
        # ★ 权重 1.0 → 2.0：第二阶段命中频率卡在 0.069/0.10，根因是
        # complete_punch 稀疏奖励几乎不触发（raw≈0.0008），ego 打满一拳后
        # 不收拳、手臂伸直贴人，无法快速连击。提权重 + 加稠密收拳引导
        # （retract_scale=0.5，见 complete_punch_reward）补上收拳梯度。
        # ★ v_start_thresh 0.5 → 0.4、retract_ratio 0.3 → 0.2：让"出拳-收拳"
        # 循环更容易被识别到，先点亮这个信号再收紧。
        ("complete_punch", 2.0, rewards.complete_punch_reward,
            dict(asset_cfg=own, max_steps=40, v_start_thresh=0.4,
                 base_reward=1.0, retract_ratio=0.2, retract_scale=0.5)),
        ("facing", 1.0, rewards.facing_reward,
            dict(asset_cfg=own, target_body_name="torso_link", sigma=0.5)),

        # === 反"伸手贴住对手不得分"：持续亚阈值接触要受罚 ===
        # hold_impulse 默认取课程阈值的 50%，随课程自适应
        ("sustained_contact", -2.0, rewards.sustained_contact_penalty,
            dict(asset_cfg=own, window_steps=4, substep_dt=0.005,
                cooldown_steps=10, decay=0.5)),

        # === 反"慢速贴人累计冲量"：接触已建立但相对速度趋零 ===
        # 纯运动学，不读冲量；快速命中因相对速度高被 gate 挡掉，不误伤
        ("slow_press", -2.0, rewards.slow_press_penalty,
            dict(asset_cfg=own, contact_dist=0.22, rel_speed_thresh=0.4,
                 cooldown_steps=4)),

        ("action_rate", -0.02, rewards.custom_action_rate_l2,
            dict(asset_cfg=own)),

        ("joint_limits", -0.5, joint_limits, dict(asset_cfg=own)),

        ("capture_point", -1.0, rewards.capture_point_penalty,
            dict(asset_cfg=own, foot_half_length=0.093, foot_half_width=0.036,
                 g=9.81, max_penalty=1.0,
                 foot_local_center=(0.039, 0.0, -0.025))),

        # === 倒地：过程惩罚，独立于 KO 终局（不是替代关系）===
        # 连续倒地 3s = 150 帧；此处 -20×0.02 = -0.4/帧，
        # 持续倒地到 KO 的过程惩罚 ≈ 150×(-0.4) = -60，
        # 加上 ko_loss -20，被 KO 总代价 ≈ -80，量级协调、不过度恐惧倒地。
        # （-30 曾使 fallen 与 scoring 同量级、压制连续出拳，降到 -20）
        ("fallen_penalty",     -20.0, rewards.fallen_penalty, dict(asset_cfg=own)),
        

    ]