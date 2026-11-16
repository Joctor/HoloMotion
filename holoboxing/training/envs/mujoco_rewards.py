"""Original sandbag reward functions with named MuJoCo state selections."""

from holoboxing.training.envs.entities import SceneEntityCfg
from holoboxing.training.tasks.combat.mdp import rewards
from holoboxing.training.tasks.combat.config.sandbag import (
    BODY_NAMES,
    JOINT_NAMES,
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
        ("draw",               -5.0, rewards.draw_penalty,         dict(asset_cfg=own)),   # 平局略罚

        # === 得分：精确对齐规则（20ms 法向冲量 ≥5 N·s，200ms 冷却）===
        ("scoring_impulse",   100.0, rewards.scoring_impulse_reward,  
         dict(asset_cfg=own, impulse_threshold=5.0)),
        ("conceded_impulse", -100.0, rewards.conceded_impulse_penalty, 
         dict(asset_cfg=own, impulse_threshold=5.0)),

        # === 冲量进度：稠密引导“接近当前得分门槛”（新增）===
        ("impulse_progress",    5.0, rewards.impulse_progress_reward,
         dict(asset_cfg=own, target_impulse=3.5, power=4.0,
              min_impulse=0.5, lead_ratio=1.2)),

        # === 倒地：规则 0 分，只能轻推，必须 < KO 惩罚 ===
        ("fallen_penalty",     -2.0, rewards.fallen_penalty,       
         dict(asset_cfg=own)),   # -0.04/步，3s=-6

        # === 平衡：只在出界时罚，不奖励站桩 ===
        ("capture_point", -2.0, rewards.capture_point_penalty,
         dict(asset_cfg=own, foot_half_length=0.08, foot_half_width=0.05, g=9.81)),

        # === 稳定性正则（提高，之前太小看不见）===
        ("action_rate_l2",     -0.05, rewards.custom_action_rate_l2, dict(asset_cfg=own)),
        ("foot_slip_penalty",  -0.05, rewards.foot_slip_penalty,     
         dict(asset_cfg=own,
            sensor_cfg=entity(sensor.name, ["left_ankle_roll_link", "right_ankle_roll_link"]),
            threshold=0.1)),
        ("joint_limit",        -2.0, joint_limits,                  dict(asset_cfg=own)),

        # === 探索 shaping：大幅降权，只辅助不主导 ===
        ("facing_reward",       0.1, rewards.facing_reward,         
         dict(asset_cfg=own, target_body_name="torso_link", sigma=0.5)),
        ("active_hit_shaping",  0.1, rewards.active_hit_shaping_reward, 
         dict(asset_cfg=own,
            hit_body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
            target_body_names=["torso_link"],
            sigma=1.0, vel_threshold=0.8, gate_temperature=10.0)),
        # vel_reward 删除；strike_reward 删除；offensive_hit 删除（scoring 已覆盖）；
        # complete_punch 删除；energy 删除。
    ]