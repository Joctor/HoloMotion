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


# Approach shaping has to retire or it outbids the scoreboard.
#
# Per-step weights are multiplied by the 0.02 s control period and paid on all
# 3000 steps of a round, so a weight of 2.0 on ``range`` is worth 120 reward for
# parking 0.88 m from the opponent for a full minute. A clean round of boxing
# lands maybe one point per second, and a point is worth 1.6, so parking was
# worth about twice as much as actually fighting. The walk in still has to be
# paid for first -- it is the only thing that works at 2 m -- so the attractor
# keeps full strength for the first 15 updates and is handed over to the
# scoring terms by update 45. It is now a floor, not a home: at 0.25 it is
# worth 15 per round, which eight clean punches out-earn.
RANGE_HANDOVER = (15, 45)
RANGE_WEIGHT = (1.8, 0.25)


def annealed(iteration, bounds, values):
    """Linear hand-over between two weights, clamped outside ``bounds``."""
    first, last = bounds
    if last <= first:
        return values[-1]
    t = min(max((iteration - first) / (last - first), 0.0), 1.0)
    return values[0] + (values[1] - values[0]) * t


def terms(agent, iteration=0):
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
    # the 0.02 s control period, so an event-style weight of 80 pays 1.6 per
    # scoring event and a bounded weight of 6.0 pays at most 6.0 per second.
    return [
        # --- Official score: the only events the referee also pays for ---
        (
            "official_point",
            80.0,
            rewards.rule_point_reward,
            dict(asset_cfg=own),
        ),
        (
            "taken_point",
            -60.0,
            rewards.taken_point_penalty,
            dict(asset_cfg=own),
        ),
        (
            "impulse_progress",
            8.0,
            rewards.impulse_progress_reward,
            dict(asset_cfg=own),
        ),
        (
            "punch_cycle",
            8.0,
            rewards.complete_punch_reward,
            dict(
                asset_cfg=own,
                v_start_thresh=0.45,
                base_reward=1.0,
                retract_ratio=0.3,
                max_steps=20,
                min_dist=0.45,
                max_dist=1.15,
                scale=0.22,
                penalty_shift=1.3,
                reach=0.45,
            ),
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
            "punch_drive",
            6.0,
            rewards.punch_drive_reward,
            dict(asset_cfg=own, sigma=0.55, speed_cap=3.0),
        ),
        (
            "guard_shape",
            1.2,
            rewards.active_hit_shaping_reward,
            dict(
                asset_cfg=own,
                hit_body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
                target_body_names=["torso_link"],
                sigma=0.40,
                vel_threshold=0.6,
                gate_temperature=10.0,
                gate_floor=0.25,
            ),
        ),
        (
            "range",
            annealed(iteration, RANGE_HANDOVER, RANGE_WEIGHT),
            rewards.range_reward,
            dict(
                asset_cfg=own,
                target_distance=0.88,
                outer_sigma=0.50,
                inner_sigma=0.14,
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
            dict(asset_cfg=own, free_offset=0.22, span=0.22, limit=1.0),
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
        # arm_scale 0.1: the smoother must not be able to outbid a punch. At
        # 1.0 this term read -0.0092 per step while a fully saturated
        # punch_drive paid 0.05, so every arm gesture was a net loss and the
        # cheapest policy was the one that never threw a hand.
        (
            "action_rate_l2",
            -0.03,
            rewards.custom_action_rate_l2,
            dict(asset_cfg=own, arm_scale=0.1),
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
        # Energy is charged on acceleration, so it is a second, much steeper
        # arm brake: a punch peaks around 500 rad/s^2 against a walking 45, a
        # 100x jump in a squared term. It is now restricted to waist_roll --
        # the one waist axis that has no use in boxing. waist_yaw is the
        # rotation that throws the shoulder into a punch and waist_pitch is
        # the forward press that reaches across the last 0.1 m of a 0.92 m
        # gap; both were being taxed at about -1.3e-3 per step, the second
        # largest brake in the whole table after the action rate. Tipping is
        # already caught twice over by ``upright`` and ``com_support``, so
        # nothing here needs to guard against it.
        (
            "energy_saving_penalty",
            -1e-6,
            rewards.energy_saving_penalty,
            dict(
                asset_cfg=own,
                distance_threshold=0.75,
                std=0.5,
                joint_names=[
                    n for n in JOINT_NAMES if "waist_roll" in n
                ],
            ),
        ),
    ]
