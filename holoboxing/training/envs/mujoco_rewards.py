"""Original sandbag reward functions with named MuJoCo state selections."""

from holoboxing.training.envs.entities import SceneEntityCfg
from holoboxing.training.tasks.combat.mdp import rewards
from holoboxing.training.tasks.combat.config.sandbag import (
    BODY_NAMES,
    JOINT_NAMES,
)
from holoboxing.evaluation.runtime.fall_rules import FALL_HEIGHT_M
from holoboxing.evaluation.runtime.fall_rules import FALL_BODY_NAMES

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
    ego = env.scene[asset_cfg.name]
    opponent = env.get_opponent(asset_cfg.name)
    ids = [ego.body_names.index(n) for n in FALL_BODY_NAMES]
    threshold = env.physics.floor_height + minimum_height
    ego_fallen = (ego.data.body_pos_w[:, ids, 2] < threshold).any(-1)
    opp_fallen = (opponent.data.body_pos_w[:, ids, 2] < threshold).any(-1)
    return opp_fallen.float() - ego_fallen.float()


def terms(agent, stage="full"):
    own = entity(agent)
    other = "robot_B" if agent == "robot_A" else "robot_A"
    sensor = entity("contact_sensor_" + agent)
    opponent = entity("contact_sensor_" + other)

    if agent == "robot_B":
        return [
            ("stand_still", 0.4, rewards.stand_still_reward,
             dict(asset_cfg=own, positive=True)),
            ("balance", 1.0, rewards.balance_reward,
             dict(asset_cfg=own, upright_threshold=0.7, with_angular=True)),
        ]

    base = [
        ("action_rate_l2", -0.05, rewards.custom_action_rate_l2, dict(asset_cfg=own)),
        ("foot_slip_penalty", -0.05, rewards.foot_slip_penalty,
         dict(asset_cfg=own,
              sensor_cfg=entity(sensor.name,
                                ["left_ankle_roll_link", "right_ankle_roll_link"]),
              threshold=0.1)),
        ("joint_limit", -2.0, joint_limits, dict(asset_cfg=own)),
        ("energy_saving_penalty", -1e-5, rewards.energy_saving_penalty,
         dict(asset_cfg=own, distance_threshold=0.75, std=0.5,
              joint_names=[n for n in JOINT_NAMES
                           if any(p in n for p in
                                  ("waist", "shoulder", "elbow", "wrist"))])),
        ("terminate_reward", 0.3, terminate_reward,
         dict(asset_cfg=own, minimum_height=FALL_HEIGHT_M)),
    ]

    if stage == "warmup":
        return base + [
            ("stand_still", 0.4, rewards.stand_still_reward,
             dict(asset_cfg=own, positive=True)),
            ("balance", 1.2, rewards.balance_reward,
             dict(asset_cfg=own, upright_threshold=0.78, with_angular=True)),
            ("facing_reward", 1.0, rewards.facing_reward,
             dict(asset_cfg=own, target_body_name="torso_link", sigma=0.5)),
            ("distance_penalty", 0.5, rewards.distance_penalty,
             dict(asset_cfg=own, min_dist=1.0, max_dist=1.8, scale=0.08)),
        ]

    hit_terms = [
        ("active_hit_shaping_reward", 1.0 if stage == "single_punch" else 1.2,
         rewards.active_hit_shaping_reward,
         dict(asset_cfg=own,
              hit_body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
              target_body_names=["torso_link"],
              sigma=1.0, vel_threshold=0.8, gate_temperature=10.0)),
        ("offensive_hit_reward_punch", 15.0 if stage == "single_punch" else 50.0,
         rewards.offensive_hit_reward_punch,
         dict(asset_cfg=own,
              sensor_cfg1=entity(sensor.name,
                                 ["left_wrist_yaw_link", "right_wrist_yaw_link"]),
              sensor_cfg2=entity(opponent.name, ["torso_link"]),
              impulse_threshold=5.5, cooldown_steps=10)),
        ("complete_punch_reward", 0.03, rewards.complete_punch_reward,
         dict(asset_cfg=own, max_steps=40, v_start_thresh=0.75,
              retract_ratio=0.3, dist_rescale=True,
              max_dist=1.0, min_dist=0.5, scale=0.05, penalty_shift=1.0)),
    ]

    posture = [
        ("balance", 0.8, rewards.balance_reward,
         dict(asset_cfg=own, upright_threshold=0.72, with_angular=True)),
        ("facing_reward", 0.6, rewards.facing_reward,
         dict(asset_cfg=own, target_body_name="torso_link", sigma=0.5)),
        ("distance_penalty", 0.3, rewards.distance_penalty,
         dict(asset_cfg=own, min_dist=0.8, max_dist=1.6, scale=0.08)),
    ]

    if stage == "single_punch":
        return base + posture + hit_terms

    # stage == "full"
    return base + posture + hit_terms + [
        ("defensive_hit_penalty_punch", -8.0, rewards.defensive_hit_penalty_punch,
         dict(asset_cfg=own,
              sensor_cfg1=entity(opponent.name,
                                 ["left_ankle_roll_link", "right_ankle_roll_link",
                                  "left_wrist_yaw_link", "right_wrist_yaw_link"]),
              sensor_cfg2=entity(sensor.name,
                                 ["right_hip_roll_link", "left_hip_roll_link",
                                  "torso_link"]),
              impulse_threshold=5.5, cooldown_steps=10)),
    ]