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


def terms(agent):
    own = entity(agent)
    sensor = entity("contact_sensor_" + agent)
    if agent == "robot_B":
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
    # Weights are per second of reward: the environment multiplies every term by
    # the 0.02 s control period, so an impulse-style weight of 50 pays 1.0 per
    # scoring event and a weight of 3.0 pays at most 3.0 per second.
    return [
        # --- Official score: the only events the referee also pays for ---
        (
            "official_point",
            50.0,
            rewards.rule_point_reward,
            dict(asset_cfg=own),
        ),
        (
            "taken_point",
            -75.0,
            rewards.taken_point_penalty,
            dict(asset_cfg=own),
        ),
        (
            "impulse_progress",
            2.0,
            rewards.impulse_progress_reward,
            dict(asset_cfg=own),
        ),
        (
            "punch_retract",
            25.0,
            rewards.retract_after_score_reward,
            dict(asset_cfg=own, release_distance=0.28, max_steps=20),
        ),
        (
            "ko_terminal",
            3000.0,
            rewards.ko_terminal_reward,
            dict(asset_cfg=own),
        ),
        # --- Striking geometry ---
        (
            "guard_shape",
            1.2,
            rewards.active_hit_shaping_reward,
            dict(
                asset_cfg=own,
                hit_body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
                target_body_names=["torso_link"],
                sigma=0.35,
                vel_threshold=0.6,
                gate_temperature=10.0,
            ),
        ),
        (
            "closing_distance",
            0.3,
            rewards.approach_reward,
            dict(asset_cfg=own, scale=1.0),
        ),
        # --- Fall prevention (official line is 0.30 m held for 3 s) ---
        (
            "balance_margin",
            -3.0,
            rewards.balance_margin_reward,
            dict(asset_cfg=own, safe_height=0.55),
        ),
        (
            "upright",
            -2.0,
            rewards.upright_reward,
            dict(asset_cfg=own),
        ),
        (
            "com_support",
            -1.5,
            rewards.com_support_reward,
            dict(asset_cfg=own, scale=0.15, limit=1.0),
        ),
        (
            "com_drop",
            -0.3,
            rewards.com_drop_reward,
            dict(asset_cfg=own),
        ),
        (
            "fall_enter",
            -400.0,
            rewards.fall_enter_penalty,
            dict(asset_cfg=own),
        ),
        # --- Regularisers ---
        (
            "action_rate_l2",
            -0.2,
            rewards.custom_action_rate_l2,
            dict(asset_cfg=own),
        ),
        (
            "foot_slip_penalty",
            -0.05,
            rewards.foot_slip_penalty,
            dict(
                asset_cfg=own,
                sensor_cfg=entity(
                    sensor.name,
                    ["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
                threshold=0.1,
            ),
        ),
        ("joint_limit", -2.0, joint_limits, dict(asset_cfg=own)),
        (
            "energy_saving_penalty",
            -1e-5,
            rewards.energy_saving_penalty,
            dict(
                asset_cfg=own,
                distance_threshold=0.75,
                std=0.5,
                joint_names=[
                    n
                    for n in JOINT_NAMES
                    if any(
                        part in n
                        for part in ("waist", "shoulder", "elbow", "wrist")
                    )
                ],
            ),
        ),
    ]
