"""Grayscale frame -> vehicle poses in arena meters. The ONE detection engine.

Used by run_vision.py (the deployment fast path), aruco_detector_node (the ROS 2
path) and scan_arena.py, so a detection fix lands everywhere at once instead of
drifting between copies.

Three things make it faster and safer than calling detectMarkers() on every
full frame:

1. Detector parameters tuned to the tag size we KNOW we will see. The
   homography plus the printed tag size give the on-screen tag size, so the
   adaptive threshold runs two windows matched to the tag's module size
   instead of OpenCV's three fixed ones, and contours far too small or large
   to be our tags are dropped before decoding. ~1.5x single-threaded on
   synthetic 720p frames, with no loss of detections.

2. Tracking windows (ROI). Once every vehicle has been seen, each frame
   searches only a small window around each vehicle's predicted position.
   Three tags at 720p, one thread: ~1.0 ms instead of ~20.7 ms for OpenCV's
   default full-frame detector (x86 synthetic; the ratio is what transfers).
   A window that misses its vehicle triggers a full-frame search ON THE SAME
   FRAME, so tracking can cost a frame's worth of time but never a detection.
   A full-frame pass also runs every ``full_frame_every_s`` to catch duplicate
   tags and to watch the corner tags for a bumped camera. Callers with a spare
   core (the fast path) set it to None and run that audit off the hot path
   with ``audit()`` on a second detector instead.

3. Plausibility gates. A pose outside the arena, a jump faster than a car can
   drive, a tag the wrong size, or two copies of one id is rejected, not
   passed on. A rejected pose looks like a missed frame downstream: the
   FrameBuilder holds the last good pose for at most 0.25 s, then the
   dead-man stops the cars. A dropout is recoverable. A confidently wrong pose
   would steer a car, so the gates exist to turn wrong poses into dropouts.

Row-index trap (see tag_family): every detection goes through
``TagSet.to_real_id()``. Nothing here indexes by the detector's row number.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import arena_frame, marker_math
from .tag_family import family_spec


@dataclass
class Detection:
    marker_id: int
    corners: np.ndarray      # (4, 2) float64 px, full-image coordinates
    side_px: float


@dataclass
class LocalizeResult:
    found: dict[int, tuple[float, float, float]]   # vehicle id -> accepted (x, y, heading)
    detections: list[Detection]                    # everything decoded (vehicles, corners, strangers)
    rejected: list[tuple[int, str]] = field(default_factory=list)   # (vehicle id, reason)
    mode: str = "full"                             # "roi" | "full" | "roi+full"
    detect_ms: float = 0.0


@dataclass
class _Track:
    x: float
    y: float
    t: float
    px: np.ndarray           # center, pixels
    px_vel: np.ndarray       # px / s
    side_px: float


def side_px_of(corners) -> float:
    c = marker_math.as_corner_array(corners)
    return float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))


def expected_tag_px(homography, tag_m: float, half_extent: tuple[float, float]) -> tuple[float, float]:
    """(min, max) on-screen side length of a ``tag_m`` tag anywhere in the arena.

    Projects small arena-frame squares back into pixels through the inverse
    homography at the center, edges and corners. Overhead cameras give a narrow
    range; a tilted camera gives a wide one, and both ends are used."""
    Hinv = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    hw, hh = half_extent
    sides = []
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            x0, y0 = sx * hw * 0.95, sy * hh * 0.95
            sq = np.array([[x0, y0], [x0 + tag_m, y0], [x0 + tag_m, y0 + tag_m], [x0, y0 + tag_m]])
            sides.append(side_px_of(arena_frame.project_px_to_arena(Hinv, sq)))
    return float(min(sides)), float(max(sides))


def tuned_parameters(*, vehicle_px_min: float, largest_px_max: float, image_max_dim: int,
                     total_modules: int, subpixel: bool = True):
    """DetectorParameters for a frame where our tags span a KNOWN pixel range."""
    import cv2

    p = cv2.aruco.DetectorParameters()
    # Perimeter bounds as a fraction of the image's larger side (OpenCV's unit).
    # 0.5x/2x margins absorb tilt, perspective and a car roof nearer the lens.
    p.minMarkerPerimeterRate = max(0.005, 4.0 * vehicle_px_min * 0.5 / image_max_dim)
    p.maxMarkerPerimeterRate = min(4.0, 4.0 * largest_px_max * 2.0 / image_max_dim)
    # Two threshold windows scaled to the module size, instead of OpenCV's
    # fixed 3/13/23 px. Swept on blurred synthetic frames (1 thread, 720p,
    # 300 tag sightings): one window lost 1-13 detections whatever its size;
    # these two matched the default's miss count (0 at 25 px) at 2/3 of the
    # cost, and a 3 px window only ever finds noise at these tag sizes.
    module_px = vehicle_px_min / float(total_modules)
    w1 = max(5, int(round(module_px * 1.6)) | 1)
    w2 = max(w1 + 4, int(round(module_px * 4.0)) | 1)
    p.adaptiveThreshWinSizeMin = w1
    p.adaptiveThreshWinSizeMax = w2
    p.adaptiveThreshWinSizeStep = w2 - w1
    if subpixel:
        _enable_subpixel(p, module_px)
    return p


def _enable_subpixel(p, module_px: float) -> None:
    import cv2

    # Sub-pixel corners: heading comes from a ~half-tag lever arm, so +-0.5 px
    # of corner noise is ~1-2 degrees of heading jitter at 25 px. Refinement is
    # cheap for a handful of tags. The window stays inside one module so it
    # does not smear across the tag's own bit pattern.
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = int(max(2, min(5, round(module_px * 0.5))))


class Localizer:
    def __init__(self, marker_map, homography=None, *, image_size: tuple[int, int],
                 camera_matrix=None, dist_coeffs=None,
                 arena_half_extent: tuple[float, float] | None = None,
                 roi_tracking: bool = True, gating: bool = True, tuned: bool = True,
                 subpixel: bool = True, full_frame_every_s: float | None = 1.0,
                 track_timeout_s: float = 0.3, max_speed_mps: float = 2.0,
                 jump_margin_m: float = 0.10, jump_window_s: float = 0.5,
                 bounds_margin_m: float = 0.15) -> None:
        import cv2

        self.marker_map = marker_map
        self.tag_set = marker_map.tag_set()
        self.vehicle_ids = list(marker_map.pursuer_ids) + [marker_map.evader_id]
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.homography = None if homography is None else np.asarray(homography, dtype=np.float64)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.arena_half_extent = arena_half_extent
        self.gating = bool(gating) and self.homography is not None
        self.roi_tracking = bool(roi_tracking)
        self.full_frame_every_s = None if full_frame_every_s is None else float(full_frame_every_s)
        self.track_timeout_s = float(track_timeout_s)
        self.max_speed_mps = float(max_speed_mps)
        self.jump_margin_m = float(jump_margin_m)
        self.jump_window_s = float(jump_window_s)
        self.bounds_margin_m = float(bounds_margin_m)

        dictionary = self.tag_set.build_opencv_dictionary()
        modules = family_spec(marker_map.dictionary).total_modules
        self.vehicle_px_range: tuple[float, float] | None = None
        if self.homography is not None and arena_half_extent is not None:
            self.vehicle_px_range = expected_tag_px(
                self.homography, marker_map.vehicle_tag_m, arena_half_extent)
        if tuned and self.vehicle_px_range is not None:
            corner_max = expected_tag_px(
                self.homography, marker_map.corner_tag_m, arena_half_extent)[1]
            full_params = tuned_parameters(
                vehicle_px_min=self.vehicle_px_range[0],
                largest_px_max=max(corner_max, self.vehicle_px_range[1]),
                image_max_dim=max(self.image_size), total_modules=modules,
                subpixel=subpixel)
        else:
            full_params = cv2.aruco.DetectorParameters()
            if subpixel:
                _enable_subpixel(full_params, 4.0)
        self._full_params = full_params
        self.full_detector = cv2.aruco.ArucoDetector(dictionary, full_params)
        self._dictionary = dictionary
        self._modules = modules
        self._subpixel = bool(subpixel)
        self._roi_detectors: dict[int, object] = {}

        self._tracks: dict[int, _Track] = {}
        self._last_full_t = -math.inf
        self.stats = {"frames": 0, "roi_frames": 0, "rescues": 0,
                      "rejected": {"bounds": 0, "jump": 0, "size": 0, "dup": 0}}

    # ------------------------------------------------------------------ detect
    def _run(self, detector, gray, offset=(0, 0)) -> list[Detection]:
        corners, ids, _rejected = detector.detectMarkers(gray)
        if ids is None:
            return []
        out = []
        ox, oy = offset
        for marker_corners, row in zip(corners, ids.flatten()):
            marker_id = self.tag_set.to_real_id(int(row))   # raises on a config mismatch
            c = marker_math.as_corner_array(marker_corners).copy()
            c[:, 0] += ox
            c[:, 1] += oy
            out.append(Detection(marker_id, c, side_px_of(c)))
        return out

    def _roi_detector(self, window_px: int):
        """Detector for a tracking window of (bucketed) side ``window_px``.

        OpenCV's perimeter limits are RELATIVE to the image, so a detector
        tuned for the full frame is wrong for a 100 px window: its lower bound
        shrinks to a few pixels and every speck of noise becomes a candidate
        quad (measured: a 100 px window cost MORE than a 200 px one). Each
        window size therefore gets limits that are the same in absolute pixels,
        cached per 32 px bucket."""
        det = self._roi_detectors.get(window_px)
        if det is None:
            import cv2

            side = float(np.median([tr.side_px for tr in self._tracks.values()]))
            p = tuned_parameters(vehicle_px_min=side, largest_px_max=side,
                                 image_max_dim=window_px, total_modules=self._modules,
                                 subpixel=self._subpixel)
            det = cv2.aruco.ArucoDetector(self._dictionary, p)
            self._roi_detectors[window_px] = det
        return det

    def _roi_pass(self, gray, t: float) -> list[Detection]:
        h, w = gray.shape[:2]
        out: list[Detection] = []
        for vid in self.vehicle_ids:
            tr = self._tracks[vid]
            dt = min(max(t - tr.t, 0.0), self.track_timeout_s)
            pred = tr.px + tr.px_vel * dt
            motion = float(np.linalg.norm(tr.px_vel)) * dt
            half = max(2.0 * tr.side_px, tr.side_px + 2.0 * motion + 8.0)
            bucket = int(math.ceil(2.0 * half / 32.0)) * 32
            half = bucket // 2
            x0, y0 = max(0, int(pred[0]) - half), max(0, int(pred[1]) - half)
            x1, y1 = min(w, int(pred[0]) + half), min(h, int(pred[1]) + half)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            out.extend(self._run(self._roi_detector(bucket), gray[y0:y1, x0:x1], (x0, y0)))
        return out

    def _tracks_live(self, t: float) -> bool:
        return all(vid in self._tracks and (t - self._tracks[vid].t) <= self.track_timeout_s
                   for vid in self.vehicle_ids)

    # -------------------------------------------------------------------- pose
    def _pose(self, det: Detection) -> tuple[float, float, float]:
        ref = np.vstack([marker_math.marker_center_px(det.corners),
                         marker_math.marker_top_midpoint_px(det.corners)])
        ref = marker_math.undistort_points(ref, self.camera_matrix, self.dist_coeffs)
        return arena_frame.pose_from_marker(self.homography, ref[0], ref[1])

    def _gate(self, vid: int, det: Detection, pose, t: float) -> str | None:
        """Reason to reject, or None to accept."""
        if not self.gating:
            return None
        x, y, _h = pose
        if self.arena_half_extent is not None:
            hw, hh = self.arena_half_extent
            if abs(x) > hw + self.bounds_margin_m or abs(y) > hh + self.bounds_margin_m:
                return "bounds"
        if self.vehicle_px_range is not None:
            lo, hi = self.vehicle_px_range
            if not (0.5 * lo <= det.side_px <= 1.6 * hi):
                return "size"
        tr = self._tracks.get(vid)
        if tr is not None:
            dt = t - tr.t
            if 0.0 <= dt <= self.jump_window_s:
                if math.hypot(x - tr.x, y - tr.y) > self.max_speed_mps * dt + self.jump_margin_m:
                    return "jump"
        return None

    def _pick(self, vid: int, candidates: list[Detection], t: float) -> Detection | None:
        """One detection for ``vid``. Two copies of one id is ambiguous identity,
        which is resolved only when a live track makes one copy obviously right."""
        if len(candidates) == 1:
            return candidates[0]
        tr = self._tracks.get(vid)
        if tr is None or (t - tr.t) > self.track_timeout_s:
            return None
        dists = [float(np.linalg.norm(marker_math.marker_center_px(c.corners) - tr.px))
                 for c in candidates]
        order = np.argsort(dists)
        best, second = dists[order[0]], dists[order[1]]
        if best <= 2.0 * tr.side_px and second > 4.0 * tr.side_px:
            return candidates[int(order[0])]
        return None

    # --------------------------------------------------------------- frame API
    def process(self, gray, t: float | None = None) -> LocalizeResult:
        """Detect + localize one grayscale frame captured at ``t`` (seconds)."""
        t = time.perf_counter() if t is None else float(t)
        start = time.perf_counter()
        self.stats["frames"] += 1

        mode = "full"
        use_roi = (self.roi_tracking and self.homography is not None
                   and self._tracks_live(t)
                   and (self.full_frame_every_s is None
                        or (t - self._last_full_t) < self.full_frame_every_s))
        if use_roi:
            detections = self._roi_pass(gray, t)
            mode = "roi"
            seen = {d.marker_id for d in detections}
            if not all(v in seen for v in self.vehicle_ids):
                # A window missed: rescue on this same frame, never a dropped detection.
                detections = self._run(self.full_detector, gray)
                self._last_full_t = t
                mode = "roi+full"
                self.stats["rescues"] += 1
            else:
                self.stats["roi_frames"] += 1
        else:
            detections = self._run(self.full_detector, gray)
            self._last_full_t = t

        found: dict[int, tuple[float, float, float]] = {}
        rejected: list[tuple[int, str]] = []
        if self.homography is not None:
            by_id: dict[int, list[Detection]] = {}
            for d in detections:
                if d.marker_id in self.vehicle_ids:
                    by_id.setdefault(d.marker_id, []).append(d)
            if mode == "roi":
                # Overlapping windows see the same tag twice: merge copies that
                # are the same physical tag before the duplicate-id check.
                for vid, cands in by_id.items():
                    uniq: list[Detection] = []
                    for c in cands:
                        cc = marker_math.marker_center_px(c.corners)
                        if all(np.linalg.norm(cc - marker_math.marker_center_px(u.corners)) > 0.5 * c.side_px
                               for u in uniq):
                            uniq.append(c)
                    by_id[vid] = uniq
            for vid, cands in by_id.items():
                det = self._pick(vid, cands, t)
                if det is None:
                    rejected.append((vid, "dup"))
                    continue
                pose = self._pose(det)
                reason = self._gate(vid, det, pose, t)
                if reason is not None:
                    rejected.append((vid, reason))
                    continue
                found[vid] = pose
                center = marker_math.marker_center_px(det.corners)
                prev = self._tracks.get(vid)
                vel = np.zeros(2)
                if prev is not None and 1e-4 < (t - prev.t) <= self.track_timeout_s:
                    vel = (center - prev.px) / (t - prev.t)
                self._tracks[vid] = _Track(pose[0], pose[1], t, center, vel, det.side_px)
            for _vid, reason in rejected:
                self.stats["rejected"][reason] += 1

        return LocalizeResult(found=found, detections=detections, rejected=rejected, mode=mode,
                              detect_ms=(time.perf_counter() - start) * 1000.0)

    def audit(self, gray, detector=None) -> list[Detection]:
        """Full-frame detection for diagnostics only (duplicate tags, corner
        drift): no gating, no track updates, safe to run on another thread
        with its own ``detector`` (from new_full_detector(); OpenCV detector
        objects must not be shared across threads)."""
        return self._run(detector or self.full_detector, gray)

    def new_full_detector(self):
        import cv2

        return cv2.aruco.ArucoDetector(self._dictionary, self._full_params)

    def reset_tracks(self) -> None:
        self._tracks.clear()
        self._roi_detectors.clear()
        self._last_full_t = -math.inf


def corner_drift_px(detections, reference_px: dict) -> float | None:
    """Largest distance (px) between each visible calibration corner tag and
    where it sat when the arena was calibrated. None when no corner is visible.

    A few px is detection noise. Tens of px means the camera or a corner tag
    moved since calibration, and every pose is now wrong by a similar amount.
    That is invisible downstream, so the vision service alarms on it."""
    worst = None
    for d in detections:
        ref = reference_px.get(d.marker_id)
        if ref is None:
            continue
        drift = float(np.linalg.norm(marker_math.marker_center_px(d.corners) - np.asarray(ref, dtype=float)))
        worst = drift if worst is None else max(worst, drift)
    return worst
