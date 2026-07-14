"""Validated longitudinal (throttle -> speed) model for the project's car dynamics.

This is the single source of truth for the bicycle-model speed update shared by
every backend in the project:

* ``single_pursuer.env.SinglePursuerEnv`` -- the *current* single-agent PPO scope;
* ``foundation_demo.triangle_env.TriangleHiveEnv`` -- the frozen 3-car MAPPO stack;
* ``step2_training.sim_task_adapter.HiveCaptureIsaacDirectAdapter`` -- the frozen
  ``isaac_direct_adapter`` backend.

It is deliberately a neutral, dependency-free module so the active single-pursuer
package can reuse it without importing the frozen 3-car stack (and vice versa).
These three backends previously each kept their own copy of this equation and had
a history of silently diverging; keeping the model here -- pinned by
``tests/test_shared_longitudinal_model.py`` and the per-backend dynamics tests --
makes that impossible.
"""

from __future__ import annotations


def next_longitudinal_speed(
    speed: float,
    throttle: float,
    *,
    max_speed: float,
    max_reverse_speed: float,
    max_accel: float,
    max_reverse_accel: float,
    max_brake: float,
    dt: float,
) -> float:
    """Advance ``speed`` by one ``dt`` step under a normalized ``throttle`` in [-1, 1].

    Forward braking (``throttle < 0`` while moving forward) *decelerates toward*
    zero via ``max(0.0, ...)`` -- it must not snap to a full stop in one step.
    Reverse braking (``throttle > 0`` while moving backward) decelerates toward
    zero via ``min(0.0, ...)``. Zero throttle holds the current speed. Otherwise
    the speed integrates the throttle against the accel limit and is clamped to
    the forward/reverse speed envelope.
    """
    if throttle < 0.0:
        if speed > 0.0:
            return max(0.0, speed + throttle * max_brake * dt)
        accel_limit = max_reverse_accel
    elif throttle > 0.0:
        if speed < 0.0:
            return min(0.0, speed + throttle * max_brake * dt)
        accel_limit = max_accel
    else:
        return speed
    new_speed = speed + throttle * accel_limit * dt
    return max(-max_reverse_speed, min(max_speed, new_speed))
