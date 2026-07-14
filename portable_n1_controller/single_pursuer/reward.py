"""Minimal, tunable reward for the N-pursuer capture task.

The step reward is exactly ``sum(build_reward_contributions(...).values())`` so the
logged per-term contributions always equal the reward (no silent/untracked terms).

Design note (reward-hacking fix, 2026-06-29)
--------------------------------------------
An earlier version added a dense positive "approach_closeness" term (reward for being
near the evader every step). Because capture *ends the episode*, the policy learned to
hover near the evader forever and farm that reward instead of finishing. The shaping is
now purely ``progress`` (per-step decrease in distance), which is potential-based and
cannot be farmed by lingering, plus a per-step ``time`` cost so reaching the evader
quickly is strictly optimal.

N-pursuer notes
---------------
``progress`` is the MEAN per-pursuer closing of distance to the evader (the env computes
it), so all pursuers are pulled in together. ``surround_progress`` is the team's actual
goal: get *all* vehicles into the capture radius (the maneuvering ring around the evader).
It is the change in the count of pursuers within the radius — a potential-based signal, so
entering the ring pays and lingering/oscillating cannot be farmed (telescopes to the final
count). ``collisions`` (pursuer-pursuer overlaps) is penalized so the fleet spreads around
the evader instead of piling up. Capture is detected in the env (mode 'all' = full surround,
or 'any').

Design note (corral-and-pin shaping gap, 2026-06-29)
-----------------------------------------------------
``capture_mode='surround'`` requires the evader be enclosed AND held ~motionless for
``capture_hold_steps`` consecutive steps (see env.py). ``surround_progress`` and
``enclosure_progress`` only pay for *entering* the contained state, then go flat -- they
gave no gradient toward actually stopping and holding, so the hold was a rare accident
rather than a learned skill (observed: capture stuck near 0 even with extra training).
``pin_progress`` fixes this: it rewards the CHANGE in the env's own consecutive-hold
counter each step. It is potential-based (breaking a streak costs exactly what was
gained), so -- like ``progress`` -- it cannot be farmed by hovering; it specifically
teaches "stop and hold" rather than "touch and go".

Design note (role_slot experiment, 2026-06-30)
-----------------------------------------------
``slot_progress`` is an alternative/additional coordination signal, adapted from the
archived Isaac hive_chase task's ``role_slot()`` idea: each pursuer is matched (via the
distance-minimizing permutation) to one of N fixed points evenly spaced around the
predicted evader position, and rewarded for the CHANGE in mean distance to its assigned
slot (potential-based, same non-farmable pattern as ``progress``). This gives an explicit
"go here" target instead of relying on emergent angular spread from ``enclosure_progress``
alone -- an experiment to see whether explicit assignment converges faster/more reliably
for the N>=3 cooperative surround than the purely emergent approach.
"""

from __future__ import annotations

REWARD_WEIGHTS: dict[str, float] = {
    # NOTE: these are gentle DEFAULTS. The curriculum overrides them PER STAGE
    # (single_pursuer/curriculum.py) so coordination shaping (surround/enclosure/
    # spacing) is OFF during the approach stage and only switched on, gently, once
    # the base approach is learned. Over-strong coordination shaping otherwise
    # competes with approaching and breaks even the easy case.
    "capture": 120.0,        # one-shot bonus when the capture condition is met
    "progress": 14.0,        # mean per-pursuer closing of distance (potential-based)
    "surround": 12.0,        # per pursuer entering the capture radius (potential-based; key for N>=2 joint capture)
    "enclosure": 3.0,        # shrinking the angular gap around the evader (potential-based; spreads them to a ring)
    "slot": 0.0,             # closing on an explicitly-assigned surround slot (potential-based; off by default, experimental)
    "pin": 14.0,             # building/maintaining the consecutive-hold streak (potential-based; teaches "stop and hold")
    "spacing": 1.5,          # soft penalty for crowding teammates (forms a ring, not a cluster)
    "ttc": 0.0,              # proactive near-miss penalty: pairs on a CLOSING course with low
                             # time-to-collision are penalized BEFORE impact (severity ramps as
                             # TTC drops below the env's threshold). Off by default; adapted from
                             # the archived Isaac hive_chase pair-safety pattern (collision polish).
    "collision": 6.0,        # penalty per overlapping pursuer pair (N > 1)
    "border": 3.0,           # penalty per pursuer-step pinned against a wall
    "time": 0.1,             # per-step cost -> finish quickly, never linger
    "action_cost": 0.01,     # small L1 action cost -> avoid thrashing controls
}

# Canonical ordered term set; the env logs exactly these under info["reward_terms"].
REWARD_TERM_KEYS: list[str] = [
    "capture",
    "progress",
    "surround_progress",
    "enclosure_progress",
    "slot_progress",
    "pin_progress",
    "spacing_penalty",
    "ttc_penalty",
    "collision_penalty",
    "border_penalty",
    "time_penalty",
    "action_cost",
]


def build_reward_contributions(
    *,
    progress: float,
    surround_progress: float,
    enclosure_progress: float,
    slot_progress: float = 0.0,
    pin_progress: float = 0.0,
    spacing: float = 0.0,
    ttc_risk: float = 0.0,
    border_contacts: float = 0.0,
    collisions: float = 0.0,
    captured: bool = False,
    action=(),
    weights: dict[str, float] = REWARD_WEIGHTS,
) -> dict[str, float]:
    w = weights
    return {
        "capture": w["capture"] if captured else 0.0,
        "progress": progress * w["progress"],
        "surround_progress": surround_progress * w["surround"],
        "enclosure_progress": enclosure_progress * w["enclosure"],
        "slot_progress": slot_progress * w.get("slot", 0.0),
        "pin_progress": pin_progress * w.get("pin", 0.0),
        "spacing_penalty": -spacing * w["spacing"],
        "ttc_penalty": -ttc_risk * w.get("ttc", 0.0),
        "collision_penalty": -collisions * w["collision"],
        "border_penalty": -border_contacts * w["border"],
        "time_penalty": -w["time"],
        "action_cost": -sum(abs(float(a)) for a in action) * w["action_cost"],
    }
