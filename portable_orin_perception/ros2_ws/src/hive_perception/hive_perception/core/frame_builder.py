"""Turns per-frame marker detections into the controller's UDP JSON pose frame.

The wire schema is owned by portable_n1_controller/pc_controller/pose_stream.py
(``parse_pose_frame``) — a byte-identical copy lives in this bundle's
``vendored/`` and selftest.py round-trips against it:

    {"t": <capture-time-s>,
     "pursuers": [{"x": ..., "y": ..., "heading": ...}, ...],
     "evader":   {"x": ..., "y": ..., "heading": ...}}

Dropout policy (safety semantics — do not soften): a vehicle whose marker was
missed keeps its last pose for at most ``hold_max_age_s`` (default 0.25 s).
Beyond that the frame is INCOMPLETE -> build_frame() returns None -> the caller
must NOT send anything -> the controller's own 0.3 s pose-stall dead-man fires
an E-stop. Never fabricate stale poses past the cap; a silent stream IS the
E-stop signal.

Timestamps: staleness is judged against ``now_t`` (the caller's current clock),
but the frame's ``t`` field is the CAPTURE time of the newest contributing
detection — per the perception contract, ``t`` feeds velocity
finite-differencing downstream and must never be a send/wall time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml


@dataclass(frozen=True)
class TimedPose:
    x: float
    y: float
    heading: float
    t: float


@dataclass(frozen=True)
class MarkerMap:
    """Contents of config/marker_map.yaml: which ArUco id is which vehicle."""

    dictionary: str
    evader_id: int
    pursuer_ids: list[int]
    calibration_corner_ids: list[int]


def load_marker_map(path) -> MarkerMap:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return MarkerMap(
        dictionary=str(data["dictionary"]),
        evader_id=int(data["evader_id"]),
        pursuer_ids=[int(i) for i in data["pursuer_ids"]],
        calibration_corner_ids=[int(i) for i in data.get("calibration_corner_ids", [])],
    )


class FrameBuilder:
    def __init__(self, *, pursuer_ids: Sequence[int], evader_id: int,
                 hold_max_age_s: float = 0.25) -> None:
        ids = [int(i) for i in pursuer_ids] + [int(evader_id)]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate vehicle ids: {ids}")
        self.pursuer_ids = [int(i) for i in pursuer_ids]
        self.evader_id = int(evader_id)
        self.hold_max_age_s = float(hold_max_age_s)
        self._last: dict[int, TimedPose] = {}

    @property
    def tracked_ids(self) -> list[int]:
        return self.pursuer_ids + [self.evader_id]

    def update(self, detections: Mapping[int, tuple[float, float, float]], t: float) -> None:
        """Record one camera frame's detections. ``detections`` maps vehicle id
        -> (x_m, y_m, heading_rad); ids not tracked here (calibration corners,
        strangers) are ignored. ``t`` is the frame's CAPTURE time in seconds."""
        for vid in self.tracked_ids:
            if vid in detections:
                x, y, heading = detections[vid]
                self._last[vid] = TimedPose(float(x), float(y), float(heading), float(t))

    def missing_ids(self, now_t: float) -> list[int]:
        """Vehicles currently blocking a frame (never seen, or held too long)."""
        out = []
        for vid in self.tracked_ids:
            last = self._last.get(vid)
            if last is None or (now_t - last.t) > self.hold_max_age_s:
                out.append(vid)
        return out

    def build_frame(self, now_t: float) -> str | None:
        """One UDP-ready JSON frame, or None if any tracked vehicle is stale
        (in which case the caller must send NOTHING — see module docstring)."""
        poses: list[TimedPose] = []
        for vid in self.tracked_ids:
            last = self._last.get(vid)
            if last is None or (now_t - last.t) > self.hold_max_age_s:
                return None
            poses.append(last)
        capture_t = max(p.t for p in poses)
        payload = {
            "t": capture_t,
            "pursuers": [{"x": p.x, "y": p.y, "heading": p.heading} for p in poses[:-1]],
            "evader": {"x": poses[-1].x, "y": poses[-1].y, "heading": poses[-1].heading},
        }
        return json.dumps(payload)
