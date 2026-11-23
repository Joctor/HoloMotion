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
        # ★ 语义变更（rewards.py 内实现）：从「朝对手质心的速度梯度」改成
        # 「沿手臂轴线(肩→腕)的径向伸展速度梯度 × 朝对手方向门」。
        # 原因：旧版奖励「末端快于近端」，但抡臂(hook)天然满足此梯度（末端
        # 线速度=角速度×半径，腕半径最大），策略于是学会横向抡臂（诊断
        # diag_contact_align_mean ≈0.31，69% 切向），单拳冲量卡 1.5 N·s。
        # 径向伸展（肘从弯到直、腕沿肩→腕方向远离肩）是刺拳(jab)区别于抡臂
        # 的唯一运动学本质：刺拳径向速度大、抡臂径向速度≈0。故主梯度改奖励
        # 径向伸展，逼刺拳。
        # ★ 但纯径向伸展会允许「背对对手空打刺拳」——故把「朝对手」以
        # 「方向门」(cos(手臂轴线, 肩→对手)) 的形式加回，而非旧版的速度投影
        # 梯度（那才是被抡臂 hack 的根源）。抡臂手臂轴线垂直于对手→门≈0 挡掉；
        # 背对→cos<0→门挡死；只有「手臂指向对手 + 肘伸展」才拿分。
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
        # ★ 反钩拳（切向速度惩罚）：比赛只算法向冲量，钩拳（拳速垂直于
        # 拳→躯干方向）= 0 分。诊断 diag_contact_align_mean 连续卡 0.287，
        # 说明策略仍在抡臂——1.2 m/s 的接触速度只有 29% 沿法线、71% 切向
        # 擦过，这正是 mean_impulse 卡 1.68（得分线 5）而 peak force 已
        # 158N 的根因。normal_alignment 只奖励径向、惩罚不了切向残余，钩拳
        # 掠过躯干时照样拿对齐分。本项在拳套距对手 0.4m 内直接按切向速度
        # 计罚，把横摆压成直拳。权重 -1.0：钩拳时 v_tangent≈1、near≈1，
        # raw≈-0.02/步，与 fallen(-0.0105) 同量级，足以推动策略改直拳。
        ("tangential_swing", -1.0, rewards.tangential_swing_penalty,
            dict(asset_cfg=own, near_dist=0.4)),
        # 接触瞬间沿法线的「撞击速度」（真正产生冲量的量）
        # ★ 语义变更（rewards.py 内实现）：reward 从「0.4m 内接近速度」改成
        # 「接触瞬间（法向力>0）法向速度」主导，接近引导弱化到 0.2。原因：
        # 诊断锁定「接近快、接触慢」hack——接近峰值法向速度 6.7 m/s，接触
        # 瞬间法向速度仅 0.36 m/s（差 19 倍），策略抡臂到 0.4m 附近刷接近
        # 奖励、真正撞击时却收力，单拳冲量卡 1.5 N·s（得分线 5）。旧版奖励
        # 只看「接近速度」、不看「接触瞬间」，给了这个 hack 空间。
        # ★ 权重保持 5.0：改后 reward 只在接触帧给分（接触瞬间速度 0.36 比
        # 接近均值 0.057 高，虽帧率仅 ~2%，函数均值仍高于旧版，无需再加权）。
        # ★ speed_gate 0.8 现在只作用于弱化的接近引导项（0.2 权重）；
        # fast_hit_speed 0.5 仍用于课程质量闸门（接触前快速接近判定）。
        ("tip_impact_speed", 5.0, rewards.tip_impact_speed_reward,
            dict(asset_cfg=own, near_dist=0.4, speed_gate=0.8,
                 fast_hit_speed=0.5)),
        # ★ 单拳冲量稠密进度：直接奖励规则量（20ms 法向冲量逼近 5 N·s），
        # 填 scoring 悬崖（<5 零分、≥5 一分）之前"1.5→5"中间缺失的梯度。
        # 这是从"运动学代理"（速度/角度/对齐/伸展，代理总有洞：抡臂、空打、
        # 接近快接触慢、贴人）转向"规则量本身"（冲量无法 hack）的关键一步。
        # require_fast_hit=True 读 tip_impact 缓存的 last_fast_hit_mask 做门，
        # 堵死"贴人累计冲量"；本项必须排在 tip_impact_speed 之后（顺序保证
        # mask 已缓存）。
        # ★ weight 5.0 → 15.0：iter 32 里 raw 仅 0.0005，太弱、对抗不了
        # fallen 恐惧（-0.0206），推不动"正面撞击"。提到 15 后，接触帧
        # （冲量 2.5 N·s，progress 0.42）reward ≈ 15×0.17×0.02 ≈ 0.05/帧，
        # 一次重拳 3 帧 ≈ 0.15，与降后的 fallen(-0.24/倒地帧) 同量级，终于
        # 能给"冲量往上爬"足够梯度，引导策略撞破 scoring 悬崖（+100 才是主力）。
        # ★ power 2.0 → 1.0：iter 32 实测 mean_impulse 1.68、target=5×1.2=6.0，
        # progress=0.28，power=2 时 reward=0.078、梯度 d/dI=2×0.28/6=0.09；
        # 二次方把梯度在"1.5→3 N·s"这段（策略当前卡住的位置）压到近零，
        # 策略根本看不见爬到 5 的路径。线性 power=1 给恒定梯度 1/6≈0.17，
        # raw 也放大 3.6×，把冲量往上推的梯度还给策略。
        ("impulse_progress", 15.0, rewards.impulse_progress_reward,
            dict(asset_cfg=own, power=1.0, require_fast_hit=True)),
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
        # ★ -20 → -12：单拳冲量卡 1.5 N·s（得分线 5）的根因被诊断为
        # fallen_penalty 过重。证据：fallen raw -0.0206/步 是 scoring +0.0044
        # 的 4.7 倍，"怕倒"动机远强于"得分"，策略于是学会"高速接近 + 接触
        # 瞬间转向切向擦过"（diag_best_v_max 6.7 m/s 但接触瞬间法向仅 0.34，
        # 差 20 倍），既拿接近速度奖励、又躲开正面撞击的摔倒风险，有效质量
        # 只有手臂 1.65 kg、冲量永远 1.5。降到 -12 削弱"怕倒"，释放"身体前倾
        # 正面撞击"的探索（那才是能打出 5 N·s 的动作），同时 ko_loss -1000
        # 仍在兜底"倒地=被 KO"的终极惩罚。连续倒地 3s 惩罚 ≈ 150×(-0.24)
        # = -36，仍明显，不至于让策略随意倒地。
        ("fallen_penalty",     -12.0, rewards.fallen_penalty, dict(asset_cfg=own)),
        

    ]