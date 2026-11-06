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


def terminate_reward(env, asset_cfg):
    ego = asset_cfg.name
    opponent = env.get_opponent_id(ego)
    return (
        env.metrics[opponent]["fallen"].float()
        - env.metrics[ego]["fallen"].float()
    )


def terms(env, agent):
    own = entity(agent)
    other = "robot_B" if agent == "robot_A" else "robot_A"
    sensor = entity("contact_sensor_" + agent)
    opponent = entity("contact_sensor_" + other)
    training_mode = getattr(env, "training_mode", None)
    if agent == "robot_B" and training_mode == "standing":
        return [
            ("stand_still", 1.0, rewards.stand_still_reward, dict(asset_cfg=own)),
            ("balance", 1.0, rewards.balance_reward,
             dict(asset_cfg=own, upright_threshold=0.7, with_angular=True)),
        ]
    return [
        # ---- 密集塑形 ----
        ("balance", 1.0, rewards.balance_reward,
         dict(asset_cfg=own, upright_threshold=0.7, with_angular=True)),
        ("low_height_penalty", -1.0, rewards.low_height_penalty,
         dict(asset_cfg=own, safe_height=0.5)),
        ("facing_reward", 1.2, rewards.facing_reward,
         dict(asset_cfg=own, target_body_name="torso_link", sigma=0.5)),
        ("vel_reward", 1.0, rewards.vel_reward,
         dict(asset_cfg=own, target_body_name="torso_link", sigma=0.25, tar_speed=0.7)),
        ("hit_reward", 0.5, rewards.hit_reward,
         dict(asset_cfg=own,
              hit_body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
              target_body_names=["torso_link"], sigma=1.0)),
        ("active_hit_shaping_reward", 3.0, rewards.active_hit_shaping_reward,
         dict(asset_cfg=own,
              hit_body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
              target_body_names=["torso_link"],
              sigma=1.0, vel_threshold=0.8, gate_temperature=10.0)),
        ("force_shaping_reward", 0.5, rewards.force_shaping_reward,
         dict(asset_cfg=own, scale=1.0)),
        ("impulse_shaping_reward", 1.0, rewards.impulse_shaping_reward,
         dict(asset_cfg=own, window_frames=4, physics_dt=0.005, scale=5.0)),
        ("strike_reward", 0.1, rewards.strike_reward,
         dict(asset_cfg=own,
              sensor_cfg_ego=entity(sensor.name, ["torso_link"]),
              sensor_cfg_opponent=entity(opponent.name, ["torso_link"]),
              target_body_weights=[1.0])),
        ("complete_punch_reward", 0.05, rewards.complete_punch_reward,
         dict(asset_cfg=own, max_steps=40, v_start_thresh=0.75, retract_ratio=0.3,
              dist_rescale=True, max_dist=1.0, min_dist=0.5, scale=0.05, penalty_shift=1.0)),
        # ---- 正则 ----
        ("action_rate_l2", -0.01, rewards.custom_action_rate_l2, dict(asset_cfg=own)),
        ("foot_slip_penalty", -0.05, rewards.foot_slip_penalty,
         dict(asset_cfg=own,
              sensor_cfg=entity(sensor.name, ["left_ankle_roll_link", "right_ankle_roll_link"]),
              threshold=0.1)),
        ("joint_limit", -2.0, joint_limits, dict(asset_cfg=own)),
        ("energy_saving_penalty", -1e-6, rewards.energy_saving_penalty,
         dict(asset_cfg=own, distance_threshold=0.4, std=0.5,
              joint_names=[n for n in JOINT_NAMES
                           if any(p in n for p in ("waist", "shoulder", "elbow", "wrist"))])),
        # ---- 得分：只读裁判结果，参数已删除 ----
        ("offensive_hit_reward_punch", 50.0, rewards.offensive_hit_reward_punch,
         dict(asset_cfg=own)),
        ("defensive_hit_penalty_punch", -8.0, rewards.defensive_hit_penalty_punch,
         dict(asset_cfg=own)),
        ("terminate_reward", 2.0, terminate_reward, dict(asset_cfg=own)),
    ]