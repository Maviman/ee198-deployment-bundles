"""Stage H environment rules that need no simulator. Pure torch, batched, CPU-testable.

rc16/hive/hive_env.py cannot be imported without Isaac, so every rule of it that can be stated on tensors alone
lives here and the env only calls it. Pinned by tests/test_rc16_hive_rules.py. Each function exists because of a
defect the 2026-11-17 review of hive_env.py found; the docstring says which.

Terms:
    tick            one 10 Hz control step, 0.1 s (TICK_S).
    decision        one commander decision = one env step = DECISION_TICKS ticks = 0.5 s.
    gamma           the PPO discount per decision (rc16/train/configs/stage_h_ppo.yaml discount_factor, 0.99).
    seen            the two-camera pipeline published a pose for the car on this tick.
    unseen ticks    consecutive ticks, up to now, on which a car was not seen (0 = seen on this tick).
"""

from __future__ import annotations

import torch

from rc16.hive import geometry as G

TICK_S = 0.1
DECISION_TICKS = 5
TRY_THROTTLE = 0.3          # |throttle| that counts as "trying" (plan.md, box-stop definition)
N_ARCS_PER_DIRECTION = 9    # boxstop.free_travel columns 0..8 forward, 9..17 reverse


# ---- the Orin's track -------------------------------------------------------------------------------------------

def track_update(pose_xy, pose_yaw, v_est, unseen, xy, yaw, seen, tick_s: float = TICK_S):
    """One camera tick of the pose track. Shapes (..., 2) / (...,); `unseen` long, `seen` bool.

    Signed forward speed (m/s) = displacement since the pose last SEEN, projected on the new heading, divided by
    the time since that pose: (unseen + 1) * tick_s. This is the deployed estimator (AI_Deployment
    pc_controller/loops.py SignedSpeed: displacement / timestamp difference). A car that is not seen keeps its
    pose and its speed estimate.

    Defect fixed: the env divided by ONE tick however long the car had been unseen, so a car re-seen after k
    missed ticks read (k + 1) times its speed for one tick (1.5 m/s -> 3.0 m/s after one dropped frame).

    Returns (pose_xy, pose_yaw, v_est, unseen), all new tensors.
    """
    d = xy - pose_xy
    dt = (unseen + 1).to(xy.dtype) * tick_s
    v = (d[..., 0] * torch.cos(yaw) + d[..., 1] * torch.sin(yaw)) / dt
    v_new = torch.where(seen, v, v_est)
    xy_new = torch.where(seen[..., None], xy, pose_xy)
    yaw_new = torch.where(seen, yaw, pose_yaw)
    unseen_new = torch.where(seen, torch.zeros_like(unseen), unseen + 1)
    return xy_new, yaw_new, v_new, unseen_new


def stale_speed(v, unseen, tick_s: float = TICK_S, brake: float = G.BRAKE_PLAN):
    """The speed (m/s, signed) the distance limiter may still assume for a car not seen for `unseen` ticks:
    the last estimate less what the planned brake (BRAKE_PLAN, 4.0 m/s^2) takes off in the time unseen, never
    past zero. A seen car (unseen = 0) keeps its estimate.

    Defect fixed: an unseen car kept its last speed estimate for ever, so the limiter's "too fast for the room:
    brake" stayed true on every tick and its brake (throttle against the remembered motion) became a blind
    full-throttle drive the other way for as long as the car was unseen. With this, the remembered speed lasts
    exactly as long as a brake from it takes (0.8 m/s: 2 ticks), then the limiter treats the car as standing.
    """
    mag = (v.abs() - brake * tick_s * unseen.to(v.dtype)).clamp(min=0.0)
    return torch.sign(v) * mag


# ---- "trying" ---------------------------------------------------------------------------------------------------

def trying_tick(sent_thr, wanted_thr, intervened, static, thr_min: float = TRY_THROTTLE):
    """(N,) bool: did the evader try to drive on this tick?

    True when any of: the throttle it SENT is >= thr_min in size; the throttle it WANTED before its own tape
    speed plan and limiter is >= thr_min in size (SmartEvader.wanted); its limiter changed the throttle; it is a
    static (parked on purpose, curriculum) evader.

    Defect fixed: only the sent throttle was read. SmartEvader caps its own throttle by the room to the tape
    before the limiter sees it, so within about 0.38 m of the tape (centre to tape, along the nose or the tail)
    every push toward the tape was 0.23 to 0.29, under 0.3: a box against the tape read "not trying" on 42 % of
    ticks and could never be held for 10.
    """
    return (sent_thr.abs() >= thr_min) | (wanted_thr.abs() >= thr_min) | intervened.bool() | static.bool()


# ---- reward: the order of the terminal costs ----------------------------------------------------------------------

def continue_cost_bound(time_cost: float, gamma: float, decision_ticks: int = DECISION_TICKS) -> float:
    """The most the time cost alone can take from the value of playing on: time_cost per tick, decision_ticks
    ticks per decision, discounted for ever (PPO bootstraps through the 20 s timeout): time_cost *
    decision_ticks / (1 - gamma). 0.01, 0.99, 5 -> 5.0."""
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"gamma must be in [0, 1), got {gamma}")
    return time_cost * decision_ticks / (1.0 - gamma)


def check_reward_order(*, win_bonus: float, time_cost: float, escape_cost: float, flip_cost: float,
                       pushed_cost: float, pawn_outside_cost: float, pawn_outside_cap_ticks: int, gamma: float,
                       decision_ticks: int = DECISION_TICKS) -> dict:
    """Raise ValueError unless no losing end of an episode is worth more than playing on.

    Required (costs are positive numbers, subtracted):
        win_bonus > 0
        escape_cost > floor + one pawn excursion      floor = continue_cost_bound, one pawn excursion =
                                                      pawn_outside_cost * pawn_outside_cap_ticks
        flip_cost >= escape_cost
        pushed_cost > flip_cost
        pushed_cost > floor + three pawn excursions   (the foul the pawns can cause themselves)
    so that  win > time out (play on) > escape >= flip > pushed.

    Defect fixed: with escape -2 and flip 0 against a time cost worth -5 for ever, a commander that was not
    winning was paid for any play choice that raised the chance of a flip or of an evader exit.
    Returns {"floor", "pawn_excursion"} (reward units).
    """
    floor = continue_cost_bound(time_cost, gamma, decision_ticks)
    one = pawn_outside_cost * pawn_outside_cap_ticks
    bad = []
    if not win_bonus > 0.0:
        bad.append(f"win_bonus {win_bonus} must be > 0")
    if not escape_cost > floor + one:
        bad.append(f"escape_cost {escape_cost} must be > {floor + one:.3f} (play-on bound {floor:.3f} + one pawn "
                   f"excursion {one:.3f})")
    if not flip_cost >= escape_cost:
        bad.append(f"flip_cost {flip_cost} must be >= escape_cost {escape_cost}")
    if not pushed_cost > flip_cost:
        bad.append(f"pushed_cost {pushed_cost} must be > flip_cost {flip_cost}")
    if not pushed_cost > floor + 3.0 * one:
        bad.append(f"pushed_cost {pushed_cost} must be > {floor + 3.0 * one:.3f} (play-on bound + three pawn "
                   "excursions)")
    if bad:
        raise ValueError("terminal costs out of order: " + "; ".join(bad))
    return {"floor": floor, "pawn_excursion": one}


# ---- reward: shaping ----------------------------------------------------------------------------------------------

def box_potential(free, g):
    """Potential (N,) in [0, 1] of how boxed the evader is: min(share of the 9 FORWARD arcs with free travel < g,
    share of the 9 REVERSE arcs with free travel < g). free (N,18) from boxstop.free_travel; g metres (a float
    or (N,)). 1.0 exactly when Boxed(g).

    Why not blocked_frac (the share of all 18): one car across the nose blocks all 9 forward arcs and no reverse
    arc; blocked_frac calls that 0.5, this calls it 0. Ceiling for ONE car: 8/9 = 0.889 (the car alongside the
    flank that blocks 17 of 18, tests/test_rc16_hive_boxstop.py ONE_CAR_17), so a high value is still not a box.
    """
    g = torch.as_tensor(g, dtype=free.dtype, device=free.device)
    under = (free < (g[..., None] if g.dim() else g)).to(free.dtype)
    k = N_ARCS_PER_DIRECTION
    return torch.minimum(under[..., :k].mean(-1), under[..., k:].mean(-1))


def shaping_reward(phi_prev, phi_now, done, gamma: float, weight: float):
    """Potential-based shaping for one decision: weight * (gamma * phi_next - phi_prev), where phi_next is
    phi_now for an env that plays on and 0 for an env whose episode ended on this decision (win, foul, escape,
    flip OR timeout: the next state the learner sees is a fresh spawn, potential 0).

    Returns (reward (N,), phi_next (N,)); the caller stores phi_next as the next phi_prev.

    Defect fixed: the term was the raw difference of blocked_frac and was never paid back, so whatever
    potential an episode ended on was banked (up to +0.94 for one car leaning on the evader at a timeout,
    against an expected win reward of 0.16 to 1.25 per episode at the baseline win rates). In this form the sum
    over any episode is -(1 - gamma) * weight * sum(phi) <= 0, and the discounted sum from a zero-potential start
    is exactly 0 (Ng, Harada, Russell 1999), so the term can steer but cannot be collected.
    """
    phi_next = torch.where(done, torch.zeros_like(phi_now), phi_now)
    return weight * (gamma * phi_next - phi_prev), phi_next


# ---- reward: pawns beyond the tape ---------------------------------------------------------------------------------

def outside_charge(outside, rearm, charged_ticks, cap_ticks: int):
    """Which pawns are charged the pawn-outside cost on this tick.

    outside (N,3) bool      the pawn's centre is beyond the tape now (true pose)
    rearm (N,3) bool        the excursion is over: the pawn is not in recovery any more
    charged_ticks (N,3) long ticks already charged in the running excursion
    Returns (charge (N,3) bool, charged_ticks (N,3) long). A pawn is charged while outside, for at most
    cap_ticks ticks per excursion; the count starts again only after `rearm` with the pawn inside.

    Defect fixed: the charge ran for as long as the pawn was outside (2.0 per second per pawn, up to -40 for a
    pawn that gave up outside at the start), which the commander cannot influence once the pawn is in recovery,
    and which made ending the episode by a foul (-10) the cheaper way out.
    """
    charge = outside & (charged_ticks < cap_ticks)
    nxt = charged_ticks + charge.long()
    nxt = torch.where(rearm & ~outside, torch.zeros_like(nxt), nxt)
    return charge, nxt


# ---- holds --------------------------------------------------------------------------------------------------------

def hold_ledger(run, hold_before, hold_after, boxed, moved, trying, rammed, captured):
    """Book one tick of the box-stop hold. All (N,); hold_* long, the rest bool.

    run          the env's outcome is not latched yet
    hold_before  BoxStopTracker.hold before its step; hold_after: after it
    boxed        BoxStopTracker.boxed after the step; moved: evader > 0.15 m from the hold's anchor
    trying       the value given to the tracker; rammed: a ram condition is active on this tick
    captured     what the tracker's step returned

    A ram cancels the hold: the returned `hold` is 0 where rammed, and `captured` is False there.
    Returns a dict of (N,) tensors:
        hold, captured            as above
        start                     a hold began on this tick (and was not cancelled by a ram on the same tick)
        break_unblocked / break_moved / break_not_trying / break_ram
                                  a running hold ended on this tick; the first reason that applies, in that order
    Every start is later matched by exactly one of: a break, a capture, or the episode ending with the hold
    still running (the env counts that as break_episode_end). Counted only where `run`.

    Defect fixed: a hold that started on the tick of a ram was counted as started and never as ended.
    """
    zero = torch.zeros_like(hold_after)
    hold = torch.where(rammed, zero, hold_after)
    captured = captured & ~rammed
    start = run & (hold_before == 0) & (hold > 0)
    broke = run & (hold_before > 0) & (hold == 0)
    return {
        "hold": hold, "captured": captured, "start": start,
        "break_unblocked": broke & ~boxed,
        "break_moved": broke & boxed & moved,
        "break_not_trying": broke & boxed & ~moved & ~trying,
        "break_ram": broke & boxed & ~moved & trying,
    }


# ---- evaluation ---------------------------------------------------------------------------------------------------

def ratio_bootstrap_ci(num, den, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """Confidence interval [low, high] of the pooled ratio sum(num) / sum(den) when the rows are EPISODES:
    episodes are resampled with replacement (n_boot times, fixed seed), the ratio is recomputed, and the
    (1 - level) / 2 and (1 + level) / 2 quantiles are returned. num, den: (E,) per-episode counts.

    Defect fixed: evaluate_hive put a Wilson interval on pooled decisions, as if the ~20 decisions of one
    episode were independent; that interval is too narrow. [0.0, 1.0] when there is nothing to divide by.
    """
    num = torch.as_tensor(num, dtype=torch.float64).reshape(-1)
    den = torch.as_tensor(den, dtype=torch.float64).reshape(-1)
    e = num.numel()
    if e == 0 or float(den.sum()) <= 0.0:
        return [0.0, 1.0]
    gen = torch.Generator().manual_seed(int(seed))
    idx = torch.randint(0, e, (int(n_boot), e), generator=gen)
    s, m = num[idx].sum(1), den[idx].sum(1)
    r = (s[m > 0] / m[m > 0])
    if r.numel() == 0:
        return [0.0, 1.0]
    q = torch.quantile(r, torch.tensor([(1.0 - level) / 2.0, (1.0 + level) / 2.0], dtype=torch.float64))
    return [float(q[0]), float(q[1])]
