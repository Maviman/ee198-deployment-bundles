"""`arena scan`: calibrate (or verify) the arena from the overhead camera, and
report everything that decides whether a driving session will be reliable.

Uses the same capture path, camera locks and detector as the live vision
service. A scan that passes describes the session that follows.

    python3 scan_arena.py                  # calibrate: writes config/arena_homography.yaml
    python3 scan_arena.py --check          # verify only: has the camera moved?
    python3 scan_arena.py --tune-exposure  # also find the shortest usable exposure
    python3 scan_arena.py --synthetic      # no camera (rehearsal / tests)

Steps
  1. camera   the configured MJPEG mode is offered and delivers its frame rate
  2. light    brightness and clipping; with --tune-exposure, sweep manual
              exposures and keep the shortest that is bright enough (short =
              less motion blur on a moving tag), saved to camera.local.yaml
  3. tags     every configured tag: detection rate, on-screen size
  4. arena    corner tags -> homography + residuals (or, with --check, how far
              the corners have drifted from where calibration saw them)
  5. cars     each vehicle tag's pose and pixel size vs the family's floor

Writes ~/.arena/scan/latest.png (annotated snapshot) and latest.json, and
exits non-zero if the arena is not fit to drive in.
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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import vision  # noqa: E402,F401  (puts hive_perception on sys.path)
from hive_perception.core import arena_frame, marker_math  # noqa: E402
from hive_perception.core.camera_config import load_camera_config, write_local_override  # noqa: E402
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


def main(argv=None) -> int:
    import cv2
    import yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the existing calibration, write nothing")
    ap.add_argument("--tune-exposure", action="store_true")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--device", default=None)
    ap.add_argument("--synthetic", action="store_true", help="rendered arena, no camera")
    ap.add_argument("--out-dir", default=str(Path.home() / ".arena" / "scan"))
    ap.add_argument("--residual-warn-m", type=float, default=0.02)
    ap.add_argument("--drift-warn-px", type=float, default=3.0)
    ap.add_argument("--pursuers", type=int, default=None,
                    help="how many pursuer cars to look for (the first N of marker_map.yaml); default all")
    args = ap.parse_args(argv)

    cfg_dir = ROOT / "config"
    mm = load_marker_map(cfg_dir / "marker_map.yaml").with_pursuers(args.pursuers)
    cam = load_camera_config(cfg_dir / "camera.yaml").with_overrides(device=args.device)
    arena_cfg = yaml.safe_load((cfg_dir / "arena_test_6ft.yaml").read_text(encoding="utf-8"))
    half = (arena_cfg["arena_width_m"] / 2.0, arena_cfg["arena_height_m"] / 2.0)
    points = {int(k): (float(v[0]), float(v[1])) for k, v in arena_cfg["calibration_points"].items()}
    # A rehearsal must never overwrite the real arena's calibration.
    if args.synthetic:
        hpath = Path(args.out_dir).expanduser() / "synthetic_homography.yaml"
    else:
        hpath = cfg_dir / "arena_homography.yaml"
    spec = family_spec(mm.dictionary)
    rep = Report()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"arena scan: {mm.dictionary}, camera {cam.mode} on "
          f"{'SYNTHETIC' if args.synthetic else cam.device}, arena {2 * half[0]:.2f} x {2 * half[1]:.2f} m")

    # ---------------------------------------------------------------- 1. camera
    print("\n[1/5] camera")
    if args.synthetic:
        from vision.capture import SyntheticSource
        from vision.synthetic import SceneRenderer, overhead_homography
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
                rep.add(FAIL, "mode", f"camera does not offer MJPEG {cam.width}x{cam.height}. Offered: {listing}. "
                        "Set width/height in config/camera.yaml to one of these.")
            elif best + 0.5 < cam.fps:
                rep.add(WARN, "mode", f"{cam.width}x{cam.height} MJPEG tops out at {best:g} fps "
                        f"(configured {cam.fps})")
            else:
                rep.add(OK, "mode", f"MJPEG {cam.mode} offered")
        source = V4l2MjpegSource(cam, log=lambda m: print(f"        {m}"),
                                 controls_log=lambda m: None).start()   # applies the camera locks
    frames = grab(source, max(args.frames, 20))
    if not frames:
        source.stop()
        rep.add(FAIL, "capture", f"no frames from {cam.device} (in use by the vision service? `arena stop` first)")
        return 1
    ts = [f[2] for f in frames]
    fps = (len(ts) - 1) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else 0.0
    h, w = frames[0][0].shape[:2]
    if (w, h) != cam.size:
        rep.add(FAIL, "resolution", f"frames are {w}x{h}, configured {cam.width}x{cam.height}")
    rep.add(OK if fps >= 0.9 * cam.fps else WARN, "frame rate",
            f"{fps:.1f} fps measured (configured {cam.fps})"
            + ("" if fps >= 0.9 * cam.fps else ". Low light with exposure priority on, or USB bandwidth"))

    # ----------------------------------------------------------------- 2. light
    print("\n[2/5] light")
    gray0 = frames[-1][0]
    mean, clip = float(gray0.mean()), float((gray0 >= 250).mean() * 100.0)
    level = OK if 60 <= mean <= 190 and clip < 3 else WARN
    rep.add(level, "brightness", f"mean {mean:.0f}/255, {clip:.1f}% clipped"
            + ("" if level == OK else " (aim for 80-170 and <3% clipped)"))
    if args.tune_exposure and not args.synthetic:
        from vision import camera_controls
        chosen = None
        for exp in (30, 50, 80, 120, 160, 250, 330):          # 3 ms .. 33 ms
            if not camera_controls.set_exposure(cam.device, exp):
                rep.add(WARN, "exposure", "camera has no manual exposure control")
                break
            time.sleep(0.3)
            sample = grab(source, 4)
            if not sample:
                continue
            m = float(np.mean([s[0].mean() for s in sample]))
            print(f"        exposure {exp / 10:.0f} ms -> mean {m:.0f}")
            if m >= 90:
                chosen = exp
                break
        if chosen is not None:
            path = write_local_override(cfg_dir / "camera.yaml", exposure=chosen)
            rep.add(OK, "exposure", f"{chosen / 10:.0f} ms is bright enough; saved to {path.name}")
        else:
            rep.add(WARN, "exposure", "even 33 ms is dim: add light, or leave exposure on auto")
        frames = grab(source, max(args.frames, 20))
    source.stop()

    # ------------------------------------------------------------------ 3. tags
    print("\n[3/5] tags")
    loc = Localizer(mm, None, image_size=(w, h), roi_tracking=False, gating=False)
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
    for mid in mm.all_ids:
        hits = len(seen.get(mid, []))
        role = ("corner" if mid in mm.calibration_corner_ids else
                "evader" if mid == mm.evader_id else f"P{mm.pursuer_ids.index(mid) + 1}")
        if copies.get(mid, 0) > 1:
            # Two printed copies of one id: averaging them would calibrate to a
            # point between them (and still show a 0 cm residual), and a car tag
            # with a twin is an ambiguous identity. Never usable.
            rep.add(FAIL, f"tag {mid} ({role})",
                    f"{copies[mid]} copies in view: remove the spare printed tag")
            seen.pop(mid, None)
        elif hits == 0:
            rep.add(FAIL if role == "corner" and not args.check else WARN, f"tag {mid} ({role})",
                    f"not seen in {n} frames")
        else:
            rate = 100.0 * hits / n
            rep.add(OK if rate >= 95 else WARN, f"tag {mid} ({role})",
                    f"seen {rate:.0f}% of frames, {np.median(sides[mid]):.1f} px")

    # ----------------------------------------------------------------- 4. arena
    print("\n[4/5] arena calibration")
    from hive_perception.calibrate_arena import load_ros_camera_info  # noqa: E402

    K, D = load_ros_camera_info(cfg_dir / "camera_info.yaml")
    corner_ids = [c for c in mm.calibration_corner_ids if c in points]
    # Median, not mean: one bad frame cannot drag a corner.
    centers = {c: np.median([marker_math.marker_center_px(x) for x in seen[c]], axis=0)
               for c in corner_ids if len(seen.get(c, [])) >= max(3, n // 2)}
    H = None
    if args.check:
        if not hpath.exists():
            rep.add(FAIL, "calibration", "none saved yet: run `arena scan` without --check")
        else:
            cal = arena_frame.load_calibration_yaml(hpath)
            H = cal["homography"]
            if cal.get("image_size") and tuple(cal["image_size"]) != (w, h):
                rep.add(FAIL, "calibration", f"made at {cal['image_size']}, camera now {w}x{h}: re-scan")
            ref = cal.get("corner_px") or {}
            if not ref:
                Hinv = np.linalg.inv(H)
                ref = {c: arena_frame.project_px_to_arena(Hinv, [points[c]])[0] for c in corner_ids}
            drifts = {c: float(np.linalg.norm(centers[c] - np.asarray(ref[c]))) for c in centers if c in ref}
            if not drifts:
                rep.add(WARN, "drift", "no corner tags visible to compare against")
            else:
                worst = max(drifts.values())
                ppm = float(np.mean(expected_tag_px(H, 1.0, half)))
                rep.add(OK if worst <= args.drift_warn_px else FAIL, "drift",
                        f"corners moved up to {worst:.1f} px (~{100 * worst / ppm:.1f} cm) since "
                        f"{cal.get('calibrated_at_utc', '?')}"
                        + ("" if worst <= args.drift_warn_px else ": camera or tags moved, re-run `arena scan`"))
    else:
        missing = [c for c in corner_ids if c not in centers]
        if len(centers) < 4:
            rep.add(FAIL, "calibration", f"need 4 corner tags, reliably seen: {sorted(centers)}; "
                    f"missing {missing}. Check placement, focus and light")
        else:
            px = marker_math.undistort_points(np.vstack([centers[c] for c in centers]), K, D)
            arena_pts = np.array([points[c] for c in centers])
            H, residuals = arena_frame.fit_homography(px, arena_pts)
            worst = float(residuals.max())
            for c, r in zip(centers, residuals):
                print(f"        tag {c}: residual {100 * r:.2f} cm")
            rep.add(OK if worst <= args.residual_warn_m else WARN, "calibration",
                    f"max residual {100 * worst:.2f} cm"
                    + ("" if worst <= args.residual_warn_m else
                       ": re-measure the worst corner's position in arena_test_6ft.yaml"))
            lo, hi = expected_tag_px(H, mm.vehicle_tag_m, half)
            arena_frame.save_homography_yaml(
                hpath, H, residuals, arena_config="config/arena_test_6ft.yaml", image_size=(w, h),
                notes="pixel -> arena-centered meters; fitted by scan_arena.py",
                corner_px={c: centers[c] for c in centers},
                extra={"camera_mode": cam.mode, "vehicle_tag_px": [round(lo, 1), round(hi, 1)],
                       "scan_frames": n})
            rep.add(OK, "saved", str(hpath))

    # ------------------------------------------------------------------ 5. cars
    print("\n[5/5] cars")
    floor = spec.min_detect_px()
    vehicles = list(mm.pursuer_ids) + [mm.evader_id]
    poses = {}
    if H is not None:
        lo, hi = expected_tag_px(H, mm.vehicle_tag_m, half)
        verdict = OK if lo >= floor else (WARN if lo >= 0.8 * floor else FAIL)
        rep.add(verdict, "tag budget",
                f"{mm.vehicle_tag_m * 100:.0f} cm vehicle tags are {lo:.1f}-{hi:.1f} px across the arena; "
                f"{mm.dictionary} needs ~{floor:.0f} px with motion blur"
                + ("" if verdict == OK else
                   ": raise the resolution, use bigger tags, or lower the camera"))
        for vid in vehicles:
            if not seen.get(vid):
                continue
            c = seen[vid][-1]
            ref = np.vstack([marker_math.marker_center_px(c), marker_math.marker_top_midpoint_px(c)])
            ref = marker_math.undistort_points(ref, K, D)
            x, y, hd = arena_frame.pose_from_marker(H, ref[0], ref[1])
            poses[vid] = (x, y, hd)
            inside = abs(x) <= half[0] and abs(y) <= half[1]
            rep.add(OK if inside else WARN, f"car tag {vid}",
                    f"at ({x:+.2f}, {y:+.2f}) m, heading {math.degrees(hd):+.0f} deg"
                    + ("" if inside else ": outside the arena boundary"))
    missing_cars = [v for v in vehicles if not seen.get(v)]
    if missing_cars:
        rep.add(WARN, "cars", f"vehicle tag(s) {missing_cars} not in view (fine for a calibration-only scan)")

    # ------------------------------------------------------------- 6. snapshot
    snap = cv2.imdecode(frames[-1][1], cv2.IMREAD_COLOR) if frames[-1][1] is not None \
        else cv2.cvtColor(frames[-1][0], cv2.COLOR_GRAY2BGR)
    last = {d.marker_id: d for d in loc.process(frames[-1][0], frames[-1][2]).detections}
    for mid, d in last.items():
        color = (0, 220, 0) if mid in vehicles else (0, 200, 255)
        cv2.polylines(snap, [d.corners.astype(np.int32)], True, color, 2)
        cx, cy = marker_math.marker_center_px(d.corners).astype(int)
        label = f"{mid}"
        if mid in poses:
            x, y, hd = poses[mid]
            label += f" ({x:+.2f},{y:+.2f})"
        cv2.putText(snap, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if H is not None:
        Hinv = np.linalg.inv(H)
        edge = arena_frame.project_px_to_arena(
            Hinv, [[-half[0], half[1]], [half[0], half[1]], [half[0], -half[1]], [-half[0], -half[1]]])
        cv2.polylines(snap, [edge.astype(np.int32)], True, (0, 255, 255), 2)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(str(out_dir / "latest.png"), snap)
    cv2.imwrite(str(out_dir / f"scan-{stamp}.png"), snap)
    (out_dir / "latest.json").write_text(json.dumps({
        "at": stamp, "mode": cam.mode, "fps": round(fps, 1), "check": args.check,
        "items": [{"level": a, "what": b, "detail": c} for a, b, c in rep.items],
        "poses": {str(k): [round(v, 4) for v in p] for k, p in poses.items()},
    }, indent=2), encoding="utf-8")

    print(f"\nsnapshot: {out_dir / 'latest.png'}")
    if rep.failed:
        print("SCAN FAILED: fix the [FAIL] items above before driving.")
        return 1
    warns = sum(1 for level, _, _ in rep.items if level == WARN)
    print("SCAN OK" + (f" with {warns} warning(s)" if warns else "") + " -- arena is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
