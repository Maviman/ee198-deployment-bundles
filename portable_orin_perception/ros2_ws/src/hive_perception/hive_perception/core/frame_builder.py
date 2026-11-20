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

Lost vehicles (``report_lost=True``, what run_vision.py uses since 2026-11-19): a vehicle held
past its cap no longer silences the stream. Its entry is sent with x, y and heading null, the
seconds since it was last seen (``lost_s``) and that last seen pose (``last``):

    {"x": null, "y": null, "heading": null, "lost_s": 0.62, "last": {"x": 1.18, "y": 0.4, "heading": 0.1}}

A pose that is being HELD (seen a few frames ago, within its hold time) carries its own
capture time as "t" inside the entry; a pose from the newest frame does not.

The controller then knows WHICH car the cameras lost and where, and can bring it back (blind
recovery) instead of ending the run. Nothing is fabricated: a lost vehicle carries no pose. A
controller that predates this cannot parse the nulls, counts the frame as bad and stalls: the
old behaviour, so a version mismatch fails safe. Silence still means stop: a dead camera or a
crashed process sends nothing, and before every vehicle has been seen once nothing is sent.
The evader may be held longer than the cars (``evader_hold_max_age_s``): it is slow or parked,
and a parked car beside it often hides its tag for a few frames.

Timestamps: staleness is judged against ``now_t`` (the caller's current clock),
but the frame's ``t`` field is the CAPTURE time of the newest contributing
detection — per the perception contract, ``t`` feeds velocity
finite-differencing downstream and must never be a send/wall time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
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
    """Contents of config/marker_map.yaml: which fiducial id is which vehicle.

    ``dictionary`` names the family (ArUco DICT_4X4_50, AprilTag
    DICT_APRILTAG_36h11, ...). Everything downstream is family-agnostic — the
    detector, the calibrator and the printable-sheet generator all take the
    family from here, so switching families is a config edit plus a reprint,
    never a code change.
    """

    dictionary: str
    evader_id: int
    pursuer_ids: list[int]
    calibration_corner_ids: list[int]
    max_correction_bits: int | None = None
    extra_print_ids: list[int] | None = None
    # Printed black-square side lengths. The localizer turns these into the
    # expected on-screen tag size, which tunes the detector and gates out
    # detections that are the wrong size to be one of our tags.
    vehicle_tag_m: float = 0.07
    corner_tag_m: float = 0.10
    # How high the vehicle tags ride above the floor (m). 0 = not corrected: a roof tag then
    # reads further from the camera than the car is (arena_frame.lift_to_height). The evader's
    # may differ (another car); None = the same as the pursuers'.
    vehicle_tag_height_m: float = 0.0
    evader_tag_height_m: float | None = None

    def tag_height_m(self, vid: int) -> float:
        if int(vid) == self.evader_id and self.evader_tag_height_m is not None:
            return float(self.evader_tag_height_m)
        return float(self.vehicle_tag_height_m)

    def with_pursuers(self, n: int | None) -> MarkerMap:
        """Track only the first ``n`` pursuer tags: CAR_INDEX 0..n-1, the cars a
        model with num_pursuers = n drives. The remaining pursuer tags stay
        printable but leave the detector's subset dictionary, exactly as if the
        YAML listed only ``n``. ``None`` keeps every pursuer the YAML lists."""
        if n is None or n == len(self.pursuer_ids):
            return self
        if not 1 <= n <= len(self.pursuer_ids):
            raise ValueError(f"{n} pursuer(s) requested, but config/marker_map.yaml lists "
                             f"{len(self.pursuer_ids)} pursuer tag(s): {self.pursuer_ids}")
        rest = self.pursuer_ids[n:]
        return replace(self, pursuer_ids=list(self.pursuer_ids[:n]),
                       extra_print_ids=sorted(set(self.extra_print_ids or []) | set(rest)))

    @property
    def print_ids(self) -> list[int]:
        """Every id the sheet generator should produce, including ids reserved
        for vehicles not yet in the arena.

        Deliberately wider than ``all_ids``: printing is slow and physical, and
        a half-printed set from a previous family is a real hazard. Reserving
        the future cars' sheets now means enabling them later is a one-line
        config change, not another print run.
        """
        extra = sorted(set(self.extra_print_ids or []) - set(self.all_ids))
        return self.all_ids + extra

    @property
    def all_ids(self) -> list[int]:
        """Every id this arena prints, in the canonical order that defines the
        subset dictionary's row indices. Vehicles first, then calibration
        corners; sorted within each group so the order is reproducible from the
        YAML alone. The detector and the calibrator MUST derive their tag set
        from this same list or their row indices will disagree."""
        vehicles = sorted(set(self.pursuer_ids + [self.evader_id]))
        corners = sorted(set(self.calibration_corner_ids) - set(vehicles))
        return vehicles + corners

    def tag_set(self):
        """The subset ``TagSet`` for this arena — see core.tag_family for why a
        subset (8 codewords instead of 587) is what makes AprilTag affordable
        on the CPU, and for the row-index/real-id trap it contains."""
        from .tag_family import TagSet, family_spec

        spec = family_spec(self.dictionary)
        ids = tuple(self.all_ids)
        if self.max_correction_bits is not None:
            correction = int(self.max_correction_bits)
        else:
            # Conservative on purpose, and capped WELL below what the codeword
            # distance would allow. Two different risks pull opposite ways:
            #   - too little correction -> a blurred tag goes undetected (a
            #     dropped frame; the dead-man already handles that safely)
            #   - too much correction   -> random image noise gets "corrected"
            #     into a valid id, i.e. a PHANTOM vehicle at a wrong pose, which
            #     the pipeline has no way to distinguish from a real detection
            #     and will happily drive a car from.
            # A missed detection is recoverable; a confident wrong pose is not.
            # 2 bits is also the value the detection-rate sweep in
            # tag_family.min_detect_px was measured at, so the numbers quoted
            # there describe this configuration rather than a different one.
            correction = min(2, TagSet(self.dictionary, ids, 0).safe_correction_bits())
        _ = spec  # validates the family name early, with a clear error
        return TagSet(dictionary=self.dictionary, real_ids=ids,
                      max_correction_bits=correction)


def load_marker_map(path) -> MarkerMap:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw_correction = data.get("max_correction_bits")
    return MarkerMap(
        dictionary=str(data["dictionary"]),
        evader_id=int(data["evader_id"]),
        pursuer_ids=[int(i) for i in data["pursuer_ids"]],
        calibration_corner_ids=[int(i) for i in data.get("calibration_corner_ids", [])],
        max_correction_bits=None if raw_correction is None else int(raw_correction),
        extra_print_ids=[int(i) for i in data.get("extra_print_ids", [])] or None,
        vehicle_tag_m=float(data.get("vehicle_tag_m", 0.07)),
        corner_tag_m=float(data.get("corner_tag_m", 0.10)),
        vehicle_tag_height_m=float(data.get("vehicle_tag_height_m", 0.0) or 0.0),
        evader_tag_height_m=(None if data.get("evader_tag_height_m") is None
                             else float(data["evader_tag_height_m"])),
    )


class FrameBuilder:
    def __init__(self, *, pursuer_ids: Sequence[int], evader_id: int,
                 hold_max_age_s: float = 0.25, evader_hold_max_age_s: float | None = None,
                 report_lost: bool = False) -> None:
        ids = [int(i) for i in pursuer_ids] + [int(evader_id)]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate vehicle ids: {ids}")
        self.pursuer_ids = [int(i) for i in pursuer_ids]
        self.evader_id = int(evader_id)
        self.hold_max_age_s = float(hold_max_age_s)
        self.evader_hold_max_age_s = float(hold_max_age_s if evader_hold_max_age_s is None
                                           else evader_hold_max_age_s)
        self.report_lost = bool(report_lost)
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

    def _hold(self, vid: int) -> float:
        return self.evader_hold_max_age_s if vid == self.evader_id else self.hold_max_age_s

    def missing_ids(self, now_t: float) -> list[int]:
        """Vehicles not seen within their hold time (never seen, or held too long). Without
        ``report_lost`` any of these blocks the frame; with it, only a never-seen one does."""
        out = []
        for vid in self.tracked_ids:
            last = self._last.get(vid)
            if last is None or (now_t - last.t) > self._hold(vid):
                out.append(vid)
        return out

    def build_frame(self, now_t: float, extra: Mapping[str, object] | None = None) -> str | None:
        """One UDP-ready JSON frame, or None (the caller must then send NOTHING, see the module
        docstring): when any tracked vehicle is stale, or with ``report_lost`` only when a
        vehicle has never been seen (a stale one is then sent as a lost entry).

        ``extra`` adds optional keys after the contract fields (the fast path
        sends ``seq`` and ``lat``, see portable_n1_controller's pose_stream).
        It can never override ``t``/``pursuers``/``evader``."""
        poses: list[TimedPose] = []
        entries: list[dict] = []
        fresh_t: list[float] = []
        for vid in self.tracked_ids:
            last = self._last.get(vid)
            if last is None:
                return None
            age = now_t - last.t
            if age > self._hold(vid):
                if not self.report_lost:
                    return None
                entries.append({"x": None, "y": None, "heading": None, "lost_s": round(age, 3),
                                "last": {"x": last.x, "y": last.y, "heading": last.heading}, "t": last.t})
            else:
                entries.append({"x": last.x, "y": last.y, "heading": last.heading})
                fresh_t.append(last.t)
            poses.append(last)
        # Capture time of the newest pose in the frame. With every vehicle lost there is none:
        # the frame is then only a heartbeat ("cameras alive, nothing seen") stamped now.
        capture_t = max(fresh_t) if fresh_t else now_t
        if self.report_lost:
            # A HELD pose (missed on the newest frames, still within its hold time) says when it
            # was really captured. Without this the controller sees an unchanged pose under a
            # newer frame time and reads the car's speed as zero just before it is lost.
            for entry, pose in zip(entries, poses):
                if entry["x"] is not None and capture_t - pose.t > 1e-4:
                    entry["t"] = pose.t
        payload = {
            "t": capture_t,
            "pursuers": entries[:-1],
            "evader": entries[-1],
        }
        if extra:
            for key, value in extra.items():
                payload.setdefault(key, value)
        return json.dumps(payload)
