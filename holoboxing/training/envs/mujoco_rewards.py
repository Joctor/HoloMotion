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
        # 100 rather than 80: the point has to outbid a full round of hovering
        # credit. At 80 a scored point was 4.7 of 61.4 positive reward per
        # round, so the policy could collect 92% of its income without ever
        # clearing the 5 N*s gate. At 100 the price of one point is 2.0 and a
        # clean ten-point round is worth +20, which is in the same league as
        # the whole approach-shaping budget.
        (
            "official_point",
            100.0,
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
            18.0,
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
                reach_max=0.45,
                reach_min=0.24,
                throw_floor=0.15,
                impulse_share=0.75,
                impulse_soft=0.5,
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
        # Sigma is the price of distance, and it was set too wide: at 0.55 the
        # proximity kernel still pays 51% for a fist thrown from 0.45 m, which
        # is precisely the half-extension comfort pose that produced 72
        # punches and no contact. At 0.40 the same 0.45 m throw pays 28% while
        # a 0.24 m one pays 70%, so the term now prefers arriving over
        # windmilling nearby, and the far field weakens from 0.16 to 0.03 at
        # 0.75 m where nothing can be hit anyway.
        #
        # Second reading, at iteration 182: this term alone paid 29.6 of the
        # 61.4 positive reward per round -- 48% of the whole table. Worse, it
        # could not tell "arrive" from "loiter": at sigma 0.40 and cap 3.0,
        # loitering 0.50 m out at 3 m/s paid 1.26 per second and arriving on
        # the chest at 1 m/s paid 1.22, a ratio of one. Backing off half a
        # metre and swinging faster cost the policy nothing, and over the last
        # 24 updates it bought +0.4 m/s of peak fist speed and paid for it with
        # contact (4.9% -> 4.5%), punch count (6.4 -> 5.9 per round), distance
        # (0.77 -> 0.83 m) and falls (0.31% -> 0.78% of steps).
        #
        # Sigma 0.32 and cap 6.0 make the same two poses pay 0.26 and 1.40, a
        # factor of 5.4: the far pose falls because the kernel is narrower, and
        # 3 m/s stops being a free maximum so a 6 m/s arrival is the only way
        # to buy the whole cap. The weight stays at 6.0 -- the level was never
        # the problem, the geometry was.
        (
            "punch_drive",
            6.0,
            rewards.punch_drive_reward,
            dict(asset_cfg=own, sigma=0.32, speed_cap=6.0),
        ),
        (
            "guard_shape",
            1.2,
            rewards.active_hit_shaping_reward,
            dict(
                asset_cfg=own,
                hit_body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
                target_body_names=["torso_link"],
                sigma=0.32,
                vel_threshold=0.6,
                gate_temperature=10.0,
                gate_floor=0.25,
            ),
        ),
        # Retargeted from 0.88 to 0.80: the measured hitting distance is 0.81,
        # and a parked aim point *outside* where the punch has to be thrown is
        # a small but permanent headwind against closing the last centimetres.
        # At 0.80 the attractor sits at the actual striking range, still pulls
        # in from 0.95, and no longer pushes away at 0.81.
        (
            "range",
            annealed(iteration, RANGE_HANDOVER, RANGE_WEIGHT),
            rewards.range_reward,
            dict(
                asset_cfg=own,
                target_distance=0.80,
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
            # One-shot, so it only fires on a real fall -- and it fires on the
            # step the robot goes below 0.30 m, which in this curriculum it
            # never gets up from. At -400 the whole event cost 8 reward against
            # a round that earns about 31, i.e. falling was a 1.6% expense and
            # the policy bought points with it: 0.061 falls per round (one in
            # sixteen) at update 158, each one a self-inflicted KO because the
            # opponent is a statue and cannot have caused it. -1200 makes the
            # event cost 24, roughly a full round of income, which is what it
            # costs in a match.
            -1200.0,
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
