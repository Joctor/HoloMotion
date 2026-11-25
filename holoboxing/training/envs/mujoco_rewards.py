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
# scoring terms by update 45. It is now a floor, not a home: at 0.25 it was
# still worth 7.5 per round at update 257, which twelve soft punches match.
RANGE_HANDOVER = (15, 45)
# 0.25 -> 0.14 at the floor: the annealed range term was still paying 7.5/round
# at update 257, the largest single line of proximity income, and it is a pure
# "stand at the right distance" rate that says nothing about what the hands do
# there. The closing behaviour it bought is already in the checkpoint (gap
# 0.83 -> 0.74 m), so the money is better spent on the punch.
RANGE_WEIGHT = (1.8, 0.14)


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
        # 190 rather than 100: at update 257 the measured income split was
        # 17.5/round on the three *proximity rates* (range + guard + drive),
        # 9.1 on completed punches and 4.3 on points -- i.e. 52% of the money
        # paid for standing in the right place at 0.72 m/s and only 13% for
        # clearing the referee's gate. The impulse reading fell monotonically
        # from 1.50 to 1.24 N*s over those 43 updates while contact rose, which
        # is what a policy looks like when proximity is a rate and force is an
        # event: staying near pays every second, hitting hard pays once. 190
        # puts a point at 3.8 and a ten-point round at +38, five times the
        # whole proximity budget, so the cheapest way to the ceiling is now to
        # score rather than to loiter.
        (
            "official_point",
            190.0,
            rewards.rule_point_reward,
            dict(asset_cfg=own),
        ),
        # -120: 2.4/point against 3.8 paid for a point scored, i.e. the table
        # breaks even when the exchange runs at 1:1.58. Both neighbouring
        # prices were tried on the real log and both failed, in opposite
        # directions:
        #
        #   -60  (1.2/pt, break-even 1:3.17) paid a 3.17:1 rate for a fight
        #        the policy was losing 1.6:1, so trading punches was
        #        profitable and task read +9.2/round while the referee had it
        #        behind on points.
        #   -190 (3.8/pt, break-even 1:1.00) assumed a 1:1 fight that this
        #        policy never reached. Over updates 704-953 it settled at
        #        1:1.55 and every punch was booked as a loss, so it did the
        #        only rational thing: volume fell 11.8 -> 5.2 per round,
        #        score fell 8.4 -> 3.9, conceded fell just 12.4 -> 9.7
        #        (elasticity 0.38 -- being hit tracks standing in range, not
        #        throwing), and task went from -24.7 to -49.0.
        #
        # At -120 the passive in-range state (range + guard + closing, about
        # 9.5/round with no punches) is worth less than an attacking state at
        # the measured 1.55 trade, which is the ordering the rulebook wants:
        # concede one only if you land one and a half. Do not read this as a
        # return to -60's free lunch; it still forbids the 1.6:1 exchange
        # that -60 was paying for.
        (
            "taken_point",
            -120.0,
            rewards.taken_point_penalty,
            dict(asset_cfg=own),
        ),
        (
            "impulse_progress",
            8.0,
            rewards.impulse_progress_reward,
            dict(asset_cfg=own),
        ),
        # 30 rather than 18. The payoff now carries a speed multiplier, and
        # the measured outbound speed at update 257 was 0.72 m/s, i.e. 0.29 of
        # ``speed_ref``. Keeping the old weight would have cut the punch income
        # from 9.1 to about 3/round for identical behaviour; 30 restores it to
        # roughly the same level while leaving the *gradient* on speed, so the
        # table is paying the same for the same behaviour and much more for a
        # faster one: 58 pushes at 0.72 m/s earn 5.5, 40 throws at 1.5 m/s earn
        # 9.4, 25 at 2.5 m/s earn 14.
        (
            "punch_cycle",
            30.0,
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
                # Outbound speed at which a throw earns the whole payoff, and
                # the fraction a slow push still keeps. 2.5 m/s is about the
                # speed at which a 1.7 kg arm carries 5 N*s into a 20 ms
                # window. The floor drops from 0.35 to 0.20 because at 0.35 it
                # binds until 0.875 m/s -- and the whole population sits at
                # 0.72, which is *below* the plateau, so the multiplier was
                # paying a constant and contributing no gradient at all where
                # it mattered. At 0.20 the slope is live from 0.5 m/s onward.
                speed_ref=2.5,
                speed_floor=0.20,
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
        # to buy the whole cap.
        #
        # Third reading, at update 257: the geometry is now right but the
        # *level* is not. With sigma 0.32 the term still paid 4.4/round as a
        # pure rate, and together with guard_shape and range the three
        # proximity rates came to 17.5 of 33.6 gross -- 52% of the table paid
        # for being in the right place, against 9.1 for actually throwing and
        # 4.3 for scoring. A rate for proximity and an event for force is
        # precisely the price list that makes a fighter stand at 0.74 m and
        # push at 0.72 m/s, which is what the log shows. Cut to 3.5: the rate
        # drops to 2.6/round and the same money moves onto the punch.
        (
            "punch_drive",
            3.5,
            rewards.punch_drive_reward,
            dict(asset_cfg=own, sigma=0.32, speed_cap=6.0),
        ),
        # 1.2 -> 0.7 for the same reason: 5.6/round of the gross was being paid
        # for keeping a fist near the torso at 0.6 m/s, i.e. for holding the
        # shadowboxing pose the previous 250 updates converged on. This is a
        # defensive shaping and it will be worth its old price again once there
        # is an opponent that punches back; today it is buy-in for the wrong
        # equilibrium.
        (
            "guard_shape",
            0.7,
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
        # Both terms used to charge a standing tax rather than a danger
        # signal: at update 410 the lowest scoring body sits at 0.505 m and
        # the torso leans 16.4 deg, so a neutral boxing stance was billed
        # 5.8 (balance_margin) + 4.9 (upright) per round. Task reward net of
        # penalties therefore fell from 12.0 to 4.3 per round while gross
        # income rose from 41 to 50 -- the policy kept earning points and
        # kept paying them back for standing the way a boxer stands, which is
        # why 80% of the gradient had become AMP imitation. Both bands now
        # start where danger starts (0.45 m, 20 deg) and are steeper inside.
        (
            "balance_margin",
            -3.0,
            rewards.balance_margin_reward,
            dict(asset_cfg=own, safe_height=0.45),
        ),
        (
            "upright",
            -2.0,
            rewards.upright_reward,
            dict(asset_cfg=own, free_tilt_deg=20.0),
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
        # chain_scale 0.05: the smoother must not be able to outbid a punch. At
        # 1.0 this term read -0.0092 per step while a fully saturated
        # punch_drive paid 0.05, so every arm gesture was a net loss and the
        # cheapest policy was the one that never threw a hand. At 0.1 the tax
        # still scaled with punch *rate*: over updates 669-704 the round
        # carried -11.0 in this term alone, i.e. the policy was paying an
        # 11/round toll on the very behaviour the table wants most, and
        # frequency is the one channel that is not saturated (point_impulse
        # already sits 49% above the gate). chain_scale multiplies the squared
        # delta, so 0.05 halves the *arm* share and leaves torso and legs
        # untouched -- the anti-jitter function where it matters (a lurch is
        # a fall) is unchanged. The new level shows up directly in the
        # action_rate_l2 line of the reward breakdown, so the arm share is
        # read off the log rather than assumed here.
        #
        # The discounted set is PUNCH_CHAIN_PARTS, i.e. arms plus waist_yaw.
        # waist_yaw is the torso rotation that throws the shoulder through the
        # punch, and it carries the largest action gain in the body (0.547
        # rad/unit vs 0.439 for the shoulder), so before this change one
        # radian of torso yaw cost 12.8x one radian of shoulder pitch once the
        # arm discount applied. waist_roll (the lean axis, no boxing use) and
        # waist_pitch (forward press, moves the CoM) stay at full rate.
        (
            "action_rate_l2",
            -0.03,
            rewards.custom_action_rate_l2,
            dict(asset_cfg=own, chain_scale=0.05),
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
