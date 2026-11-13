"""`arena scan`: calibrate (or verify) the arena from its camera(s), and report
everything that decides whether a driving session will be reliable.

Uses the same capture path, camera locks and detector as the live vision
service. A scan that passes describes the session that follows.

    python3 scan_arena.py                  # calibrate: writes config/arena_homography.yaml
    python3 scan_arena.py --check          # verify only: has the camera moved?
    python3 scan_arena.py --tune-exposure  # also find the shortest usable exposure
    python3 scan_arena.py --synthetic      # no camera (rehearsal / tests)
    python3 scan_arena.py --camera DEV0 --camera DEV1
                                           # two corner cameras, calibrated TOGETHER

Steps
  1. camera   the configured MJPEG mode is offered and delivers its frame rate
  2. light    brightness and clipping; with --tune-exposure, sweep manual
              exposures and keep the shortest that is bright enough (short =
              less motion blur on a moving tag), saved to camera.local.yaml
  3. tags     every configured tag: detection rate, on-screen size
  4. arena    lens distortion from the boundary tape (vision/lens.py), then
              corner tags -> homography + residuals (or, with --check, how far
              the corners have drifted from where calibration saw them)
  5. cars     each vehicle tag's pose and pixel size vs the family's floor

Several cameras (``--camera`` repeated, slot order; `arena scan` passes the
[vision] cameras of arena.local.conf): steps 1-3 run for each camera in turn, then
step 4 fits them together (vision/joint_calibration.py). A camera at a corner of
the arena sees its own corner tag and the two side ones but not the far one; the
tags both cameras see (the evader, a car, a spare tag) supply what is missing, and
the scan reports how far apart the cameras put each of them. Keep every tag still
during the scan: the cameras are captured one after the other.

Writes ~/.arena/scan/latest.png (annotated snapshot; latest.<k>.png for camera
k >= 1) and latest.json, and exits non-zero if the arena is not fit to drive in.
"""

from __future__ import annotations

import os

# Before OpenCV opens any camera: grab() gives up after 1 s instead of 10, so a
# hung webcam is detected and reopened in seconds (vision/capture.py).
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "1")

import argparse  # noqa: E402
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import vision  # noqa: E402,F401  (puts hive_perception on sys.path)
from hive_perception.core import arena_frame, marker_math  # noqa: E402
from hive_perception.core.camera_config import load_camera_config, same_device, write_local_override  # noqa: E402
from hive_perception.core.frame_builder import load_marker_map  # noqa: E402
from hive_perception.core.localizer import Localizer, expected_tag_px  # noqa: E402
from hive_perception.core.tag_family import family_spec  # noqa: E402

OK, WARN, FAIL = "ok", "warn", "FAIL"


class Report:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []

    def add(self, level: str, what: str, detail: str) -> None:
        self.items.append((level, what, detail))
        tag = {"ok": " ok ", "warn": "WARN", "FAIL": "FAIL"}[level]
        print(f"  [{tag}] {what}: {detail}", flush=True)

    @property
    def failed(self) -> bool:
        return any(level == FAIL for level, _, _ in self.items)


def grab(source, n: int, timeout: float = 3.0):
    """n newest frames as (gray, jpeg_or_None, t)."""
    import cv2

    out, seq = [], 0
    deadline = time.perf_counter() + timeout + n / 10.0
    while len(out) < n and time.perf_counter() < deadline:
        f = source.slot.get_newer(seq, timeout=1.0)
        if f is None:
            continue
        seq = f.seq
        gray = cv2.imdecode(f.jpeg, cv2.IMREAD_GRAYSCALE) if f.jpeg is not None else f.gray
        if gray is not None:
            out.append((gray, f.jpeg, f.t_capture))
    return out


CORNER_NAMES = {(-1, 1): "top-left", (1, 1): "top-right", (1, -1): "bottom-right", (-1, -1): "bottom-left"}
SIZE_WARN_FRAC = 0.15        # corner tags measuring this far off their printed size: scale is suspect
# Two cameras' positions for one tag: within AGREE_OK_M is the noise of a far, small tag;
# past AGREE_FAIL_M a car is steered by whichever camera saw it last.
AGREE_OK_M, AGREE_FAIL_M = 0.03, 0.06
LOO_OK_M, LOO_FAIL_M = 0.04, 0.08
PIN_OK_M, PIN_FAIL_M = 0.02, 0.05

# `--synthetic` with two cameras: the real rig of 2026-11-12, an X1-like wide camera
# (80 deg, strong barrel distortion) outside the tag-12 corner and a C270-like one (62 deg)
# outside the tag-10 corner, each seeing its own and the two side corners; cars parked one
# in each half of the arena.
SYNTHETIC_CORNER_CAMS = {
    0: dict(corner=(1, -1), hfov_deg=80.0, height_m=1.6, back_m=0.6, aim=-0.55, roll_deg=2.0, k1=-0.30),
    1: dict(corner=(-1, 1), hfov_deg=62.0, height_m=1.6, back_m=0.8, aim=-0.5, roll_deg=-1.5, k1=-0.05),
}
SYNTHETIC_PARKED = {0: (-0.6, 0.3, 0.4), 1: (0.5, -0.4, 2.0), 2: (0.2, 0.75, 1.0), 3: (-0.3, -0.8, -1.0)}


def _quadrants(pts: dict, y_down: bool, margin: float = 0.2) -> dict | None:
    """Quadrant (sx, sy) of each point about the group's centroid, +y up; None when two
    points share a quadrant or one sits too near an axis to call (e.g. camera at ~45 deg)."""
    c = np.mean(np.array(list(pts.values()), dtype=float), axis=0)
    off = {k: np.asarray(p, dtype=float) - c for k, p in pts.items()}
    if y_down:
        off = {k: np.array([o[0], -o[1]]) for k, o in off.items()}
    r = float(np.median([np.linalg.norm(o) for o in off.values()]))
    if r <= 0 or any(min(abs(o[0]), abs(o[1])) < margin * r for o in off.values()):
        return None
    q = {k: (1 if o[0] > 0 else -1, 1 if o[1] > 0 else -1) for k, o in off.items()}
    return q if len(set(q.values())) == len(q) else None


def match_layout(centers_px: dict, points: dict) -> tuple[dict | None, str]:
    """Which configured corner position belongs to which seen corner tag.

    A 4-point homography fits ANY assignment exactly (a 0.00 cm residual), so a tag
    taped in the wrong corner silently gives a mirrored or twisted arena frame: the
    policy then steers away from the evader. arena_test_6ft.yaml describes the layout
    "as seen from the camera, +y up", so the positions are matched to the tags by
    where the camera SEES them (image right = +x, image up = +y) instead of trusting
    the ids. Returns ({tag id: position}, message): the config unchanged and "" when it
    agrees; the positions re-assigned and a message naming the moved tags when it does
    not; None and the reason when the layout cannot be told apart.
    """
    seen_q = _quadrants(centers_px, y_down=True)
    conf_q = _quadrants(points, y_down=False)
    if seen_q is None:
        # A camera at a corner of the arena looks along the diagonal: the tags sit top,
        # left, right and bottom of the image and no quadrant can be called. Trust the ids
        # then (the config says which corner each tag is taped in), but refuse a mirrored
        # frame: a perspective view keeps the corners' cyclic order, so the tags must run
        # round the image in the same sequence as round the floor.
        ids = [k for k in centers_px if k in points]
        if len(ids) < 4:
            return None, "cannot tell which corner each tag is in, and fewer than 4 configured tags are in view"

        def around(pts: dict, y_sign: float) -> list:
            c = np.mean(np.array([pts[k] for k in ids], dtype=float), axis=0)
            ang = {k: math.atan2(y_sign * (pts[k][1] - c[1]), pts[k][0] - c[0]) for k in ids}
            return sorted(ids, key=lambda k: ang[k]), ang

        seen_order, seen_ang = around(centers_px, -1.0)      # image y is down: flip to +y up
        gaps = [(seen_ang[seen_order[(i + 1) % 4]] - seen_ang[seen_order[i]]) % (2 * math.pi) for i in range(4)]
        if min(gaps) < math.radians(40):
            return None, ("cannot tell which corner each tag is in: two tags sit in one corner of the image. "
                          "Put one corner tag in each corner of the arena")
        conf_order, _ = around(points, 1.0)
        i0 = conf_order.index(seen_order[0])
        if [conf_order[(i0 + i) % 4] for i in range(4)] != seen_order:
            return None, ("the corner tags run round the image in a different order than round "
                          "arena_test_6ft.yaml: a MIRRORED frame (two tags swapped on the floor). "
                          "Tape each tag in the corner the config gives it")
        return {k: points[k] for k in ids}, ("diagonal view (corner camera): corners assigned by tag id "
                                             "as taped per arena_test_6ft.yaml; cyclic order checked, not mirrored")
    if conf_q is None or set(conf_q) != set(seen_q):
        return None, "calibration_points do not put one tag in each corner"
    by_quadrant = {q: points[k] for k, q in conf_q.items()}
    matched = {k: by_quadrant[q] for k, q in seen_q.items()}
    moved = [k for k in sorted(seen_q) if seen_q[k] != conf_q[k]]
    if not moved:
        return matched, ""
    return matched, ("corner tags are not where arena_test_6ft.yaml puts them: "
                     + ", ".join(f"tag {k} is {CORNER_NAMES[seen_q[k]]} (config: {CORNER_NAMES[conf_q[k]]})"
                                 for k in moved)
                     + ". Calibrated with the layout the camera sees")


def adjacent_corners(points: dict) -> list[tuple[int, int]]:
    """Pairs of calibration tags joined by an arena edge (not a diagonal): the boundary
    tape runs straight between them."""
    ids = sorted(points)
    d = {(a, b): math.dist(points[a], points[b]) for i, a in enumerate(ids) for b in ids[i + 1:]}
    if not d:
        return []
    side = min(d.values())
    return [p for p, v in d.items() if v <= 1.05 * side]


@dataclass
class CamScan:
    """Everything the scan learned about one camera."""
    slot: int
    label: str                       # "" for a single camera, "cam k " with several
    cam: object
    frames: list
    w: int
    h: int
    fps: float
    seen: dict                       # id -> [corner arrays], frames it was seen ONCE in
    sides: dict                      # id -> [side px]
    n: int
    loc: object
    hpath: Path
    K: np.ndarray | None = None
    D: np.ndarray | None = None
    centers: dict = field(default_factory=dict)          # reliably seen corner id -> raw centre px
    H: np.ndarray | None = None
    virtual: dict = field(default_factory=dict)
    poses: dict = field(default_factory=dict)

    def reliable(self, mid: int) -> bool:
        return len(self.seen.get(mid, [])) >= max(3, self.n // 2)

    def usable_shared(self, mid: int) -> bool:
        """Good enough to tie the cameras together. The cameras map ONE arena: a tag next to
        one camera is small and far from the other, which may decode it in only a few frames
        of a still scene. Those few are still the right corners (a wrong decode is a wrong id,
        not a shifted tag), and the fit already weights them by their small size."""
        return len(self.seen.get(mid, [])) >= max(3, self.n // 6)


def survey(args, rep: Report, slot: int, device: str | None, label: str, mm, mm_detect, cfg_dir: Path,
           half, points, multi: bool) -> CamScan | None:
    """Steps 1-3 for one camera: open it, check mode / light (tune exposure), count tags."""
    import cv2

    cam = load_camera_config(cfg_dir / "camera.yaml", slot=slot, device=device)
    if args.synthetic:
        out_dir = Path(args.out_dir).expanduser()
        hpath = out_dir / ("synthetic_homography.yaml" if slot == 0 else f"synthetic_homography.{slot}.yaml")
    else:
        hpath = cfg_dir / ("arena_homography.yaml" if slot == 0 else f"arena_homography.{slot}.yaml")
    head = f" {slot}" if multi else ""

    # ---------------------------------------------------------------- 1. camera
    print(f"\n[1/5] camera{head}" + (f": {cam.device}" if multi and not args.synthetic else ""))
    if args.synthetic:
        from vision.capture import SyntheticSource
        from vision.synthetic import SceneRenderer, corner_camera_homography, overhead_homography
        if multi:
            kw = dict(SYNTHETIC_CORNER_CAMS[slot % len(SYNTHETIC_CORNER_CAMS)])
            k1 = kw.pop("k1")
            H_true = corner_camera_homography(cam.size, half, kw.pop("corner"), **kw)
            renderer = SceneRenderer(mm_detect, H_true, cam.size, half, seed=slot, tape=55.0, k1=k1)
            parked = {k: v for k, v in SYNTHETIC_PARKED.items() if k in mm_detect.all_ids}
            source = SyntheticSource(renderer, mm_detect, fps=cam.fps, corners=points, vehicles=parked).start()
        else:
            H_true = overhead_homography(cam.size, half, fill=0.9)
            renderer = SceneRenderer(mm, H_true, cam.size, half)
            source = SyntheticSource(renderer, mm, fps=cam.fps, corners=points).start()
    else:
        from vision import camera_controls
        from vision.capture import V4l2MjpegSource
        modes = camera_controls.list_mjpeg_modes(cam.device) if camera_controls.available() else []
        if modes:
            offered = [m for m in modes if (m[0], m[1]) == cam.size]
            best = max((m[2] for m in offered), default=0)
            if not offered:
                listing = ", ".join(f"{w}x{h}@{f:g}" for w, h, f in modes[:8])
                rep.add(FAIL, f"{label}mode", f"camera does not offer MJPEG {cam.width}x{cam.height}. Offered: "
                        f"{listing}. Set width/height in config/camera.yaml to one of these.")
            elif best + 0.5 < cam.fps:
                rep.add(WARN, f"{label}mode", f"{cam.width}x{cam.height} MJPEG tops out at {best:g} fps "
                        f"(configured {cam.fps})")
            else:
                rep.add(OK, f"{label}mode", f"MJPEG {cam.mode} offered")
        source = V4l2MjpegSource(cam, log=lambda m: print(f"        {m}"),
                                 controls_log=lambda m: None).start()   # applies the camera locks
    frames = grab(source, max(args.frames, 20))
    if not frames:
        source.stop()
        rep.add(FAIL, f"{label}capture", f"no frames from {cam.device} (in use by the vision service? `arena stop` first)")
        return None
    ts = [f[2] for f in frames]
    fps = (len(ts) - 1) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else 0.0
    h, w = frames[0][0].shape[:2]
    if (w, h) != cam.size:
        rep.add(FAIL, f"{label}resolution", f"frames are {w}x{h}, configured {cam.width}x{cam.height}")
    rep.add(OK if fps >= 0.9 * cam.fps else WARN, f"{label}frame rate",
            f"{fps:.1f} fps measured (configured {cam.fps})"
            + ("" if fps >= 0.9 * cam.fps else ". Low light with exposure priority on, or USB bandwidth"))

    # ----------------------------------------------------------------- 2. light
    print(f"\n[2/5] light{head}")
    gray0 = frames[-1][0]
    mean, clip = float(gray0.mean()), float((gray0 >= 250).mean() * 100.0)
    level = OK if 60 <= mean <= 190 and clip < 3 else WARN
    rep.add(level, f"{label}brightness", f"mean {mean:.0f}/255, {clip:.1f}% clipped"
            + ("" if level == OK else " (aim for 80-170 and <3% clipped)"))
    if args.tune_exposure and not args.synthetic:
        # Motion blur is what loses a moving car's tag: at 1.4 m/s a 7 cm tag that is 18 px
        # on screen smears 6 px in a 16 ms exposure (~0% detections on synthetic frames)
        # and 1.5 px in 4 ms (100%). But a short exposure also starves the small,
        # foreshortened tags at the far side of an oblique camera: on the real 8 ft arena
        # (2026-11-11) the far corner tag went from 83% seen at 16 ms to 30% at 12 ms and
        # 10% at 8 ms, and raising gain to compensate made it WORSE (noise: 12% at 3 ms /
        # gain 128) while slowing detection. So: the shortest exposure that keeps the mean
        # grey level at MIN_MEAN, at the camera's own gain; gain goes up only when even the
        # longest exposure is too dim. More light is the real fix, and the report says so.
        # The corner fit below uses the frames grabbed BEFORE this (auto exposure, every
        # corner at its most visible); the saved exposure is for driving.
        from vision import camera_controls
        TARGET_EXPOSURE, MIN_MEAN = 80, 110                    # 8 ms (~3 px smear); mean grey
        exposures = (30, 40, 50, 60, 80, 100, 120, 160, 200, 250, 330)   # 3 ms .. 33 ms
        g = camera_controls.gain_range(cam.device)
        gains = [None] if g is None else [g[0], min(g[2], max(g[0] * 2, 16)), min(g[2], max(g[0] * 4, 32))]
        best, supported = None, True                            # (exposure, gain, mean)
        for gi, gain in enumerate(gains):
            if gain is not None and not camera_controls.set_gain(cam.device, gain):
                break
            for exp in exposures:
                if not camera_controls.set_exposure(cam.device, exp):
                    rep.add(WARN, f"{label}exposure", "camera has no manual exposure control")
                    supported = False
                    break
                time.sleep(0.3)
                sample = grab(source, 4)
                if not sample:
                    continue
                m = float(np.mean([s[0].mean() for s in sample]))
                print(f"        gain {gain if gain is not None else '-'}: exposure {exp / 10:.0f} ms -> mean {m:.0f}")
                if m >= MIN_MEAN:
                    best = (exp, gain, m)
                    break
            if not supported or best is not None:
                break                                           # gain only rescues a dim room
        if best is not None:
            exp, gain, m = best
            raised = gain is not None and gain != gains[0]
            # the camera's own gain is written too: a value left in the file by another
            # camera must not ride along
            values = {"exposure": exp, "gain": gain} if gain is not None else {"exposure": exp}
            if device:
                values["device"] = device             # these settings belong to this camera
            path = write_local_override(cfg_dir / "camera.yaml", slot=slot, **values)
            ppm = 0.9 * min(cam.size) / (2.0 * half[0])            # ~px per metre, overhead camera
            blur_px = 1.4 * exp / 10000.0 * ppm
            rep.add(OK if exp <= TARGET_EXPOSURE else WARN, f"{label}exposure",
                    f"{exp / 10:.0f} ms" + (f" at gain {gain}" if raised else "") + f" (mean {m:.0f}); "
                    f"a car at 1.4 m/s smears ~{blur_px:.1f} px; saved to {path.name}"
                    + ("" if exp <= TARGET_EXPOSURE else
                       ". That smear loses small tags on a moving car: add light over the arena"))
        elif supported:
            if g is not None:
                camera_controls.set_gain(cam.device, g[0])
            rep.add(WARN, f"{label}exposure", "even 33 ms at 4x gain is dim: add light, or leave exposure on auto")
    source.stop()

    # ------------------------------------------------------------------ 3. tags
    print(f"\n[3/5] tags{head}")
    loc = Localizer(mm_detect, None, image_size=(w, h), roi_tracking=False, gating=False)
    seen: dict[int, list[np.ndarray]] = {}      # one entry per frame the id was seen ONCE in
    sides: dict[int, list[float]] = {}
    copies: dict[int, int] = {}                 # most copies of an id in any single frame
    for gray, _jpg, t in frames:
        per_frame: dict[int, list] = {}
        for d in loc.process(gray, t).detections:
            per_frame.setdefault(d.marker_id, []).append(d)
        for mid, ds in per_frame.items():
            copies[mid] = max(copies.get(mid, 0), len(ds))
            if len(ds) == 1:
                seen.setdefault(mid, []).append(ds[0].corners)
                sides.setdefault(mid, []).append(ds[0].side_px)
    n = len(frames)
    spares = [m for m in mm_detect.all_ids if m not in mm.all_ids]
    for mid in mm.all_ids + [m for m in spares if seen.get(m)]:
        hits = len(seen.get(mid, []))
        role = ("corner" if mid in mm.calibration_corner_ids else
                "evader" if mid == mm.evader_id else
                f"P{mm.pursuer_ids.index(mid) + 1}" if mid in mm.pursuer_ids else "spare")
        if copies.get(mid, 0) > 1:
            # Two printed copies of one id: averaging them would calibrate to a
            # point between them (and still show a 0 cm residual), and a car tag
            # with a twin is an ambiguous identity. Never usable.
            rep.add(FAIL, f"{label}tag {mid} ({role})",
                    f"{copies[mid]} copies in view: remove the spare printed tag")
            seen.pop(mid, None)
        elif hits == 0:
            # A corner camera never sees the far corner: the other camera covers it.
            missing_ok = multi or args.check
            rep.add(FAIL if role == "corner" and not missing_ok else WARN, f"{label}tag {mid} ({role})",
                    f"not seen in {n} frames" + (" (another camera can cover it)" if multi and role == "corner" else ""))
        else:
            rate = 100.0 * hits / n
            rep.add(OK if rate >= 95 else WARN, f"{label}tag {mid} ({role})",
                    f"seen {rate:.0f}% of frames, {np.median(sides[mid]):.1f} px"
                    + (" (ties the cameras together)" if role == "spare" else ""))
    return CamScan(slot, label, cam, frames, w, h, fps, seen, sides, n, loc, hpath)


def lens_step(args, rep: Report, sc: CamScan, points: dict, cfg_dir: Path) -> None:
    """Load (or measure from the boundary tape) this camera's lens distortion into sc.K/D."""
    from hive_perception.calibrate_arena import load_ros_camera_info  # noqa: E402
    from vision import lens

    info_dir = Path(args.out_dir).expanduser() if args.synthetic else cfg_dir
    own = info_dir / f"camera_info.{sc.slot}.yaml"
    have = lens.read_camera_info(own) if own.exists() else None
    device = None if args.synthetic else sc.cam.device
    mine = bool(have) and have.get("camera_name") == lens.MARK
    fits = bool(have) and (int(have.get("image_width", 0)), int(have.get("image_height", 0))) == (sc.w, sc.h) \
        and (device is None or not have.get("device") or same_device(have["device"], device))
    if have and not mine and fits:
        rep.add(OK, f"{sc.label}lens", f"{own.name} (a checkerboard calibration) used")
    elif have and mine and fits and not args.lens and not args.check:
        k1 = float(have["distortion_coefficients"]["data"][0])
        rep.add(OK, f"{sc.label}lens", f"k1 {k1:+.3f} from {own.name} (measured on the tape "
                f"{time.strftime('%Y-%m-%d', time.localtime(own.stat().st_mtime))}; --lens re-measures)")
    elif not args.check:
        corners = {c: sc.centers[c] for c in sc.centers}
        sizes = {c: float(np.median(sc.sides[c])) for c in corners}
        stack = np.stack([f[0] for f in sc.frames[:15]]).astype(np.float32)
        gray = np.median(stack, axis=0).astype(np.float32)
        edges = []
        for a, b in adjacent_corners(points):
            if a in corners and b in corners:
                rest = [corners[c] for c in corners if c not in (a, b)]
                if not rest:
                    continue
                pts = lens.tape_points(gray, corners[a], corners[b], sizes[a], sizes[b], np.mean(rest, axis=0))
                if pts is not None and len(pts) >= 15:
                    edges.append(pts)
        if not edges:
            rep.add(WARN, f"{sc.label}lens", "no boundary tape found between two corner tags: lens distortion not "
                    "measured (poses near the image edges carry a few cm of error)")
        else:
            k1, before, after, K = lens.fit_k1(edges, (sc.w, sc.h))
            at_limit = min(abs(k1 - lens.K1_RANGE[0]), abs(k1 - lens.K1_RANGE[1])) < 0.02
            if at_limit or (before >= 1.0 and after > 0.7 * before):
                rep.add(WARN, f"{sc.label}lens", f"tape edges do not straighten with one radial term "
                        f"({before:.1f} -> {after:.1f} px at k1 {k1:+.2f}): not used. Is something lying on the tape?")
            else:
                lens.write_camera_info(own, (sc.w, sc.h), K, k1, device or "synthetic",
                                       f"{len(edges)} edge(s), {before:.1f} -> {after:.2f} px off straight")
                rep.add(OK, f"{sc.label}lens", f"k1 {k1:+.3f}: {len(edges)} tape edge(s) {before:.1f} px off "
                        f"straight -> {after:.2f} px; saved to {own.name}")
    info = own if own.exists() else cfg_dir / "camera_info.yaml"
    sc.K, sc.D = load_ros_camera_info(info)


def main(argv=None) -> int:
    import cv2
    import yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the existing calibration, write nothing")
    ap.add_argument("--tune-exposure", action="store_true")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--device", default=None)
    ap.add_argument("--slot", type=int, default=0,
                    help="which camera of a multi-camera arena this is (0 = the first / only one). "
                         "Slot k >= 1 reads config/camera.k.local.yaml and writes "
                         "config/arena_homography.k.yaml")
    ap.add_argument("--camera", action="append", default=None, metavar="DEV",
                    help="scan several cameras and calibrate them together (repeat, in slot order)")
    ap.add_argument("--lens", action="store_true",
                    help="re-measure each camera's lens distortion from the boundary tape")
    ap.add_argument("--synthetic", action="store_true", help="rendered arena, no camera")
    ap.add_argument("--out-dir", default=str(Path.home() / ".arena" / "scan"))
    ap.add_argument("--residual-warn-m", type=float, default=0.02)
    ap.add_argument("--drift-warn-px", type=float, default=3.0)
    ap.add_argument("--pursuers", type=int, default=None,
                    help="how many pursuer cars to look for (the first N of marker_map.yaml); default all")
    args = ap.parse_args(argv)

    cfg_dir = ROOT / "config"
    mm_all = load_marker_map(cfg_dir / "marker_map.yaml")
    mm = mm_all.with_pursuers(args.pursuers)
    arena_cfg = yaml.safe_load((cfg_dir / "arena_test_6ft.yaml").read_text(encoding="utf-8"))
    half = (arena_cfg["arena_width_m"] / 2.0, arena_cfg["arena_height_m"] / 2.0)
    points = {int(k): (float(v[0]), float(v[1])) for k, v in arena_cfg["calibration_points"].items()}
    cams = list(enumerate(args.camera)) if args.camera else [(int(args.slot or 0), args.device)]
    multi = len(cams) > 1
    # With several cameras every printed tag ties them together, the fleet's spares too.
    mm_detect = mm_all if multi else mm
    spec = family_spec(mm.dictionary)
    rep = Report()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if multi:
        print(f"arena scan: {mm.dictionary}, {len(cams)} cameras calibrated together"
              f"{' (SYNTHETIC)' if args.synthetic else ''}, arena {2 * half[0]:.2f} x {2 * half[1]:.2f} m")
        for slot, dev in cams:
            print(f"  camera {slot}: {dev}")
    else:
        cam0 = load_camera_config(cfg_dir / "camera.yaml", slot=cams[0][0], device=cams[0][1])
        print(f"arena scan: {mm.dictionary}, camera {cams[0][0]} {cam0.mode} on "
              f"{'SYNTHETIC' if args.synthetic else cam0.device}, arena {2 * half[0]:.2f} x {2 * half[1]:.2f} m")

    scans: list[CamScan] = []
    for slot, dev in cams:
        sc = survey(args, rep, slot, dev, f"cam {slot} " if multi else "", mm, mm_detect, cfg_dir, half,
                    points, multi)
        if sc is None:
            return 1
        scans.append(sc)

    # ----------------------------------------------------------------- 4. arena
    print("\n[4/5] arena calibration" + (f" ({len(scans)} cameras together)" if multi else ""))
    corner_ids = [c for c in mm.calibration_corner_ids if c in points]
    for sc in scans:
        # Median, not mean: one bad frame cannot drag a corner.
        sc.centers = {c: np.median([marker_math.marker_center_px(x) for x in sc.seen[c]], axis=0)
                      for c in corner_ids if sc.reliable(c)}
        lens_step(args, rep, sc, points, cfg_dir)

    if args.check:
        for sc in scans:
            check_drift(args, rep, sc, points, corner_ids, half)
        if multi:
            report_agreement(rep, scans, corner_ids, mm_detect, mode="check")
    elif multi:
        calibrate_together(args, rep, scans, points, corner_ids, mm, mm_detect, half)
    else:
        calibrate_alone(args, rep, scans[0], points, corner_ids, mm, half)

    # ------------------------------------------------------------------ 5. cars
    print("\n[5/5] cars")
    floor = spec.min_detect_px()
    vehicles = list(mm.pursuer_ids) + [mm.evader_id]
    for sc in scans:
        if sc.H is None:
            continue
        lo, hi = expected_tag_px(sc.H, mm.vehicle_tag_m, half)
        verdict = OK if lo >= floor else (WARN if lo >= 0.8 * floor else FAIL)
        if multi and verdict == FAIL:
            verdict = WARN       # the far side is the other camera's to see
        rep.add(verdict, f"{sc.label}tag budget",
                f"{mm.vehicle_tag_m * 100:.0f} cm vehicle tags are {lo:.1f}-{hi:.1f} px across the arena; "
                f"{mm.dictionary} needs ~{floor:.0f} px with motion blur"
                + ("" if verdict == OK else
                   " at the far side, which the other camera covers" if multi else
                   ": raise the resolution, use bigger tags, or lower the camera"))
        for vid in vehicles:
            if not sc.seen.get(vid):
                continue
            c = sc.seen[vid][-1]
            ref = np.vstack([marker_math.marker_center_px(c), marker_math.marker_top_midpoint_px(c)])
            ref = marker_math.undistort_points(ref, sc.K, sc.D)
            x, y, hd = arena_frame.pose_from_marker(sc.H, ref[0], ref[1])
            sc.poses[vid] = (x, y, hd)
            inside = abs(x) <= half[0] and abs(y) <= half[1]
            rep.add(OK if inside else WARN, f"{sc.label}car tag {vid}",
                    f"at ({x:+.2f}, {y:+.2f}) m, heading {math.degrees(hd):+.0f} deg"
                    + ("" if inside else ": outside the arena boundary"))
    missing_cars = [v for v in vehicles if not any(sc.seen.get(v) for sc in scans)]
    if missing_cars:
        rep.add(WARN, "cars", f"vehicle tag(s) {missing_cars} not in view (fine for a calibration-only scan)")

    # ------------------------------------------------------------- 6. snapshot
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for sc in scans:
        snap = snapshot(sc, vehicles, half)
        suffix = "" if sc.slot == 0 else f".{sc.slot}"
        cv2.imwrite(str(out_dir / f"latest{suffix}.png"), snap)
        cv2.imwrite(str(out_dir / f"scan-{stamp}{suffix}.png"), snap)
    first = scans[0]
    (out_dir / "latest.json").write_text(json.dumps({
        "at": stamp, "mode": first.cam.mode, "fps": round(first.fps, 1), "check": args.check,
        "items": [{"level": a, "what": b, "detail": c} for a, b, c in rep.items],
        "poses": {str(k): [round(v, 4) for v in p] for k, p in first.poses.items()},
        "cameras": [{"slot": sc.slot, "device": sc.cam.device, "fps": round(sc.fps, 1),
                     "poses": {str(k): [round(v, 4) for v in p] for k, p in sc.poses.items()}}
                    for sc in scans],
    }, indent=2), encoding="utf-8")

    print("\nsnapshot: " + ", ".join(str(out_dir / f"latest{'' if sc.slot == 0 else f'.{sc.slot}'}.png")
                                     for sc in scans))
    if rep.failed:
        print("SCAN FAILED: fix the [FAIL] items above before driving.")
        return 1
    warns = sum(1 for level, _, _ in rep.items if level == WARN)
    print("SCAN OK" + (f" with {warns} warning(s)" if warns else "") + " -- arena is ready.")
    return 0


def check_drift(args, rep: Report, sc: CamScan, points, corner_ids, half) -> None:
    if not sc.hpath.exists():
        rep.add(FAIL, f"{sc.label}calibration", "none saved yet: run `arena scan` without --check")
        return
    cal = arena_frame.load_calibration_yaml(sc.hpath)
    sc.H = cal["homography"]
    if cal.get("image_size") and tuple(cal["image_size"]) != (sc.w, sc.h):
        rep.add(FAIL, f"{sc.label}calibration", f"made at {cal['image_size']}, camera now {sc.w}x{sc.h}: re-scan")
    if cal.get("device") and not args.synthetic and not same_device(cal["device"], sc.cam.device):
        rep.add(FAIL, f"{sc.label}calibration", f"fitted for {cal['device']}, but this is {sc.cam.device}: re-scan")
    ref = cal.get("corner_px") or {}
    if not ref:
        Hinv = np.linalg.inv(sc.H)
        ref = {c: arena_frame.project_px_to_arena(Hinv, [points[c]])[0] for c in corner_ids}
    drifts = {c: float(np.linalg.norm(sc.centers[c] - np.asarray(ref[c]))) for c in sc.centers if c in ref}
    if not drifts:
        rep.add(WARN, f"{sc.label}drift", "no corner tags visible to compare against")
    else:
        worst = max(drifts.values())
        ppm = float(np.mean(expected_tag_px(sc.H, 1.0, half)))
        rep.add(OK if worst <= args.drift_warn_px else FAIL, f"{sc.label}drift",
                f"corners moved up to {worst:.1f} px (~{100 * worst / ppm:.1f} cm) since "
                f"{cal.get('calibrated_at_utc', '?')}"
                + ("" if worst <= args.drift_warn_px else ": camera or tags moved, re-run `arena scan`"))


def _tag_views(sc: CamScan, ids, mm_detect):
    from vision.joint_calibration import TagView

    tags = {}
    for mid in ids:
        if not (sc.reliable(mid) if mid in mm_detect.calibration_corner_ids else sc.usable_shared(mid)):
            continue
        raw = np.median(np.stack(sc.seen[mid]), axis=0).reshape(4, 2)
        und = marker_math.undistort_points(raw, sc.K, sc.D)
        size = mm_detect.corner_tag_m if mid in mm_detect.calibration_corner_ids else mm_detect.vehicle_tag_m
        tags[mid] = TagView(und, float(np.median(sc.sides[mid])), size)
    return tags


def report_agreement(rep: Report, scans, corner_ids, mm_detect, mode: str, shared=None) -> None:
    """How far apart the cameras put each tag they all see (not the corners: exact by construction)."""
    from vision.joint_calibration import CameraView, agreement

    if shared is None:
        views = [CameraView(sc.slot, {}, _tag_views(sc, mm_detect.all_ids, mm_detect)) for sc in scans]
        shared = agreement({sc.slot: sc.H for sc in scans}, views, skip_ids=set(corner_ids))
    if not shared:
        rep.add(WARN if mode == "check" else FAIL, "cameras agree",
                "no tag (evader, car, spare) is in view of two cameras: nothing to compare")
        return
    worst = max(s.spread_m for s in shared)
    level = OK if worst <= AGREE_OK_M else WARN if worst <= AGREE_FAIL_M else FAIL
    rep.add(level, "cameras agree",
            "; ".join(f"tag {s.tag_id} {100 * s.spread_m:.1f} cm / {s.heading_spread_deg:.0f} deg" for s in shared)
            + ("" if level == OK else
               ": the cameras put the same tag in different places. Did a tag or a camera move during "
               "the scan? Run `arena scan --lens` to re-measure the lenses"))


def calibrate_alone(args, rep: Report, sc: CamScan, points, corner_ids, mm, half) -> None:
    missing = [c for c in corner_ids if c not in sc.centers]
    layout, why = None, ""
    if len(sc.centers) < 4:
        rep.add(FAIL, "calibration", f"need 4 corner tags, reliably seen: {sorted(sc.centers)}; "
                f"missing {missing}. Check placement, focus and light"
                + (". A camera at a corner of the arena sees only 3: list both corner cameras in "
                   "[vision] cameras (deploy/arena.local.conf) to calibrate them together"
                   if len(sc.centers) == 3 else ""))
    else:
        layout, why = match_layout(sc.centers, {c: points[c] for c in sc.centers})
        if layout is None:
            rep.add(FAIL, "layout", why + ". Put one corner tag in each corner of the arena")
    if layout is None:
        return
    if why:
        rep.add(WARN, "layout", why)
    centers = sc.centers
    px = marker_math.undistort_points(np.vstack([centers[c] for c in centers]), sc.K, sc.D)
    arena_pts = np.array([layout[c] for c in centers])
    H, residuals = arena_frame.fit_homography(px, arena_pts)
    sc.H = H
    worst = float(residuals.max())
    for c, r in zip(centers, residuals):
        print(f"        tag {c}: residual {100 * r:.2f} cm")
    rep.add(OK if worst <= args.residual_warn_m else WARN, "calibration",
            f"max residual {100 * worst:.2f} cm"
            + ("" if worst <= args.residual_warn_m else
               ": re-measure the worst corner's position in arena_test_6ft.yaml"))
    measured = scale_check(rep, sc, mm, half)
    save(args, rep, sc, residuals, measured, mm, half, layout, {})


def scale_check(rep: Report, sc: CamScan, mm, half) -> float:
    # Four points always fit with ~0 residual, whatever arena size the config claims.
    # The printed tag size is the one independent ruler in view: wrong positions
    # (a 6 ft config on an 8 ft floor) scale every measured tag by the same factor.
    sides = []
    for c in sc.centers:
        q = marker_math.undistort_points(np.median(np.stack(sc.seen[c]), axis=0).reshape(4, 2), sc.K, sc.D)
        m = arena_frame.project_px_to_arena(sc.H, q)
        sides.append(float(np.mean([np.linalg.norm(m[i] - m[(i + 1) % 4]) for i in range(4)])))
    measured = float(np.median(sides))
    ratio = measured / mm.corner_tag_m
    rep.add(OK if abs(ratio - 1.0) <= SIZE_WARN_FRAC else WARN, f"{sc.label}scale",
            f"corner tags measure {100 * measured:.1f} cm (configured {100 * mm.corner_tag_m:.0f} cm)"
            + ("" if abs(ratio - 1.0) <= SIZE_WARN_FRAC else
               f": either they are printed at that size (then set corner_tag_m in marker_map.yaml), "
               f"or the arena on the floor is not the {2 * half[0]:.2f} m one in arena_test_6ft.yaml "
               f"and every distance is off by x{1 / ratio:.2f}"))
    return measured


def save(args, rep: Report, sc: CamScan, residuals, measured, mm, half, layout, joint: dict) -> None:
    lo, hi = expected_tag_px(sc.H, mm.vehicle_tag_m, half)
    extra = {"camera_mode": sc.cam.mode, "vehicle_tag_px": [round(lo, 1), round(hi, 1)],
             "scan_frames": sc.n, "corner_tag_m_measured": round(measured, 4),
             "corner_layout": {int(c): [float(v) for v in layout[c]] for c in sc.centers}}
    if not args.synthetic:
        extra["device"] = sc.cam.device            # the service refuses it on another camera
    if joint:
        extra["joint"] = joint
    arena_frame.save_homography_yaml(
        sc.hpath, sc.H, residuals, arena_config="config/arena_test_6ft.yaml", image_size=(sc.w, sc.h),
        notes="pixel -> arena-centered meters; fitted by scan_arena.py"
              + (" together with the other camera(s)" if joint else ""),
        corner_px={c: sc.centers[c] for c in sc.centers}, extra=extra)
    rep.add(OK, f"{sc.label}saved", str(sc.hpath))


def calibrate_together(args, rep: Report, scans, points, corner_ids, mm, mm_detect, half) -> None:
    from vision.joint_calibration import CameraView, fit_joint

    views = []
    for sc in scans:
        und = {c: marker_math.undistort_points(np.asarray(sc.centers[c]).reshape(1, 2), sc.K, sc.D)[0]
               for c in sc.centers}
        views.append(CameraView(sc.slot, und, _tag_views(sc, mm_detect.all_ids, mm_detect)))
    for sc in scans:
        print(f"        camera {sc.slot}: corner tags {sorted(sc.centers)}, "
              f"tags {sorted(t for t in views[scans.index(sc)].tags)}")
    fit = fit_joint(views, points, half=half, corner_ids=set(corner_ids))
    if fit.error:
        rep.add(FAIL, "calibration", fit.error)
        return
    for sc in scans:
        cf = fit.cameras[sc.slot]
        sc.H, sc.virtual = cf.H, cf.virtual
        if cf.virtual:
            (cid, v), = cf.virtual.items()
            where = "in the frame but not read" if 0 <= v[0] < sc.w and 0 <= v[1] < sc.h else \
                f"{_offframe(v, sc.w, sc.h)} the frame"
            rep.add(OK, f"{sc.label}corners",
                    f"sees {', '.join(map(str, cf.seen))}; corner {cid} ({where}) placed at "
                    f"({v[0]:.0f}, {v[1]:.0f}) px from the tags both cameras see")
            pin = cf.uncertainty_m or 0.0
            level = OK if pin <= PIN_OK_M else WARN if pin <= PIN_FAIL_M else FAIL
            rep.add(level, f"{sc.label}fit", f"the shared tags pin this camera to +-{100 * pin:.1f} cm (1 sigma, "
                    "worst point of the arena)"
                    + ("" if level == OK else ": put a tag both cameras see in each half of the arena, "
                       "away from the diagonal between the side corners"))
        else:
            rep.add(OK, f"{sc.label}corners", f"sees all of {', '.join(map(str, cf.seen))}")
    report_agreement(rep, scans, corner_ids, mm_detect, mode="fit", shared=fit.shared)
    loo = [s for s in fit.shared if s.loo_m is not None]
    if loo:
        worst = max(s.loo_m for s in loo)
        level = OK if worst <= LOO_OK_M else WARN if worst <= LOO_FAIL_M else FAIL
        rep.add(level, "cross-check",
                "each shared tag left out of the fit in turn, the cameras still put it within "
                + ", ".join(f"{100 * s.loo_m:.1f} cm (tag {s.tag_id})" for s in loo)
                + ("" if level == OK else ": the cameras do not agree away from the tags that fixed them"))
    elif fit.free_params:
        rep.add(WARN, "cross-check", f"only {len(fit.shared)} tag(s) in view of both cameras: the fit used them "
                "all, so nothing checks it. Put one more (a spare tag 2 or 3, or a car) where both cameras see it")
    if fit.rms is not None and fit.rms > 3.0:
        rep.add(WARN, "fit residual", f"{fit.rms:.1f}x the scatter tag noise explains: a tag or camera moved "
                "during the scan, or a lens is not corrected")
    for sc in scans:
        cf = fit.cameras[sc.slot]
        und = np.vstack([views[scans.index(sc)].corners_px[c] for c in cf.seen])
        residuals = np.linalg.norm(arena_frame.project_px_to_arena(sc.H, und)
                                   - np.array([points[c] for c in cf.seen]), axis=1)
        measured = scale_check(rep, sc, mm, half)
        others = [s.slot for s in scans if s is not sc]
        joint = {"with_slots": others, "seen_corners": [int(c) for c in cf.seen],
                 "fitted_corners_px": {int(c): [round(float(v[0]), 1), round(float(v[1]), 1)]
                                       for c, v in cf.virtual.items()},
                 "shared_tags": [int(s.tag_id) for s in fit.shared],
                 "agreement_m": round(max((s.spread_m for s in fit.shared), default=0.0), 4),
                 "cross_check_m": round(max((s.loo_m for s in loo), default=0.0), 4) if loo else None,
                 "pinned_m": None if cf.uncertainty_m is None else round(cf.uncertainty_m, 4)}
        save(args, rep, sc, residuals, measured, mm, half, {c: points[c] for c in cf.seen}, joint)


def _offframe(v, w, h) -> str:
    parts = []
    if v[1] < 0:
        parts.append(f"{-v[1]:.0f} px above")
    elif v[1] >= h:
        parts.append(f"{v[1] - h:.0f} px below")
    if v[0] < 0:
        parts.append(f"{-v[0]:.0f} px left of")
    elif v[0] >= w:
        parts.append(f"{v[0] - w:.0f} px right of")
    return " and ".join(parts) or "outside"


def snapshot(sc: CamScan, vehicles, half):
    """The last frame with tags, poses and the calibrated arena outline (bent by the lens,
    so it must lie on the tape)."""
    import cv2

    frames = sc.frames
    snap = cv2.imdecode(frames[-1][1], cv2.IMREAD_COLOR) if frames[-1][1] is not None \
        else cv2.cvtColor(frames[-1][0], cv2.COLOR_GRAY2BGR)
    last = {d.marker_id: d for d in sc.loc.process(frames[-1][0], frames[-1][2]).detections}
    for mid, d in last.items():
        color = (0, 220, 0) if mid in vehicles else (0, 200, 255)
        cv2.polylines(snap, [d.corners.astype(np.int32)], True, color, 2)
        cx, cy = marker_math.marker_center_px(d.corners).astype(int)
        label = f"{mid}"
        if mid in sc.poses:
            x, y, hd = sc.poses[mid]
            label += f" ({x:+.2f},{y:+.2f})"
        cv2.putText(snap, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if sc.H is not None:
        hw, hh = half
        ring = [(-hw, hh), (hw, hh), (hw, -hh), (-hw, -hh), (-hw, hh)]
        pts = np.vstack([np.linspace(ring[i], ring[i + 1], 60) for i in range(4)])
        px = arena_frame.project_px_to_arena(np.linalg.inv(sc.H), pts)
        px = distort_px(px, sc.K, sc.D)
        cv2.polylines(snap, [px.astype(np.int32)], True, (0, 255, 255), 2)
    return snap


def distort_px(px, K, D):
    """Undistorted pixel points -> where the lens actually puts them."""
    import cv2

    if K is None or D is None or not np.any(D):
        return np.asarray(px, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    norm = np.hstack([np.asarray(px, dtype=np.float64), np.ones((len(px), 1))]) @ np.linalg.inv(K).T
    out, _ = cv2.projectPoints(norm.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.asarray(D, dtype=np.float64))
    return out.reshape(-1, 2)


if __name__ == "__main__":
    raise SystemExit(main())
