"""Raw perception input type + velocity estimation from consecutive poses.

Per ``docs/perception_interface_contract.md``, the vision side publishes only raw
pose per vehicle (x, y meters in the arena-centered frame; heading radians,
math-convention). Everything else the policy needs (velocity, bearings, wall
clearances, teammate offsets, predicted lead) is DERIVED here or in
``observation_adapter`` -- perception never computes features itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehiclePose:
    """One perception sample for one vehicle."""

    x: float
    y: float
    heading: float
    timestamp_s: float


class PoseHistory:
    """Tracks one vehicle's most recent pose and estimates its speed by finite
    difference. Heading is taken directly from the latest sample (ArUco marker
    orientation is far less noisy than differentiating position for a
    slow-turning-radius car -- see the perception contract doc's recommendation).

    KNOWN LIMITATION (found via SIL testing, 2026-07-04): raw finite-difference
    speed is corrupted by any DISCONTINUOUS position correction -- physical
    contact push-out being the big one in this project (corral-and-pin). A car
    the evader gets shoved 0.3 m in one 0.1 s frame reads as ~3 m/s even though
    its true speed is 0. This is not a bug in this code; it is what a real
    position-only ArUco system would ALSO see, since position alone cannot
    distinguish "drove itself" from "got bumped".

    Partial mitigation: pass ``max_accel``/``max_decel`` (the vehicle's own known
    physical limits -- legitimate information a real controller has) to rate-limit
    the estimate so it can only change as fast as the vehicle physically could.
    This recovers roughly half the SIL capture-rate gap for the N=3
    action-history-hardened policy under this exact failure mode (measured: 53%
    -> 73%, vs a clean-perception baseline of 100%) -- a real improvement, NOT a
    complete fix. The principled long-term fix is training the policy under this
    same realistic estimation noise (domain randomization, same playbook as the
    action-history latency fix), not further polishing the estimator in
    isolation -- see single-pursuer-n3-status memory / master-roadmap.
    """

    def __init__(self, *, max_accel: float | None = None, max_decel: float | None = None) -> None:
        self._previous: VehiclePose | None = None
        self._speed: float = 0.0
        self._max_accel = max_accel
        self._max_decel = max_decel if max_decel is not None else max_accel

    def update(self, pose: VehiclePose) -> None:
        prev = self._previous
        if prev is not None:
            dt = pose.timestamp_s - prev.timestamp_s
            if dt > 1e-6:
                dx, dy = pose.x - prev.x, pose.y - prev.y
                raw_speed = ((dx * dx + dy * dy) ** 0.5) / dt
                if self._max_accel is None:
                    self._speed = raw_speed
                else:
                    lo = max(0.0, self._speed - self._max_decel * dt)
                    hi = self._speed + self._max_accel * dt
                    self._speed = min(hi, max(lo, raw_speed))
            # dt <= 0 (duplicate/out-of-order frame): keep the last known speed
            # rather than dividing by ~zero and producing a spurious spike.
        self._previous = pose

    @property
    def pose(self) -> VehiclePose | None:
        return self._previous

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def has_pose(self) -> bool:
        return self._previous is not None
