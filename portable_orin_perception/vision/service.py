"""The vision service: newest camera frame -> luma decode -> localize -> pose
frame out over UDP, immediately, for every frame, from one camera or several.

Nothing waits on a timer. The ROS bridge resampled poses on a fixed 10 Hz
timer, which aged every pose by up to one camera frame interval before the
controller saw it. Here a frame's pose leaves this process ~decode+detect
after the frame arrives, and the controller picks which frames to act on (one
per 100 ms control tick, see portable_n1_controller/run_controller.py).

Several cameras (``--camera DEV`` repeated: two cameras at opposite corners of
the arena) each get their own capture thread, their own calibration
(config/arena_homography.yaml for slot 0, arena_homography.k.yaml for slot k),
their own Localizer and tracking state, and their own preview (``--preview-port``
+ k, which also serves /info.json and per-frame X-Tags so the dashboard's
tag-check panel can show them). Their detections meet in vision/fusion.py: one
pose per vehicle, weighted by how big the tag is in each view, so each camera
decides for its near half of the arena. Every camera frame produces a pose
frame, so two cameras at 30 fps give the controller ~60 pose frames a second.

Clock: pose ``t`` is the frame's capture time on CLOCK_MONOTONIC (Python's
perf_counter on Linux) plus a fixed epoch offset taken at start-up. It never jumps (safe for velocity
finite-differencing) and still reads as wall-clock seconds, so cross-machine
latency tools keep working when chrony is running.

Safety semantics are the FrameBuilder's. A vehicle unseen for more than its hold
time by EVERY camera is LOST: the frames keep going with that vehicle marked
lost (no pose, only where it was last seen), and the controller brings the car
back into view or stops it. Camera unplugged, process crash: silence, and
silence stops the cars (the controller's 0.3 s dead-man). With --no-report-lost
a lost vehicle silences the stream as it did before 2026-11-19.
"""

from __future__ import annotations

import json
import math
import signal
import socket
import threading
import time
from pathlib import Path

import numpy as np

from hive_perception.core import arena_frame
from hive_perception.core.camera_config import CameraConfig, load_camera_config, same_device
from hive_perception.core.frame_builder import FrameBuilder, load_marker_map
from hive_perception.core.localizer import Localizer, corner_drift_px
from hive_perception.core.tag_family import family_spec

from . import BUNDLE_ROOT
from .fusion import PoseFuser
from .telemetry import HostStats, TelemetrySender, Window

DSCP_EF = 0xB8   # "expedited forwarding": WMM puts it in the voice queue over WiFi


def parse_targets(spec: str | None, default_port: int) -> list[tuple[str, int]]:
    """'host:port,...' -> [(ip, port)], names resolved ONCE here. sendto() with
    a hostname does a DNS/mDNS lookup on every packet, which is milliseconds
    of hidden latency (seconds, when the resolver is having a bad day)."""
    out = []
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        host, sep, port = chunk.rpartition(":")
        host, port = (host, int(port)) if sep else (chunk, default_port)
        try:
            host = socket.gethostbyname(host)
        except OSError as exc:
            raise SystemExit(f"cannot resolve pose target {host!r}: {exc}") from None
        out.append((host, port))
    return out


def load_intrinsics(path: Path, size: tuple[int, int], device: str | None = None):
    """(K, D) or (None, None). Placeholder/all-zero distortion -> (None, None)
    (undistortion would be a no-op). Real intrinsics at another resolution, or measured
    on another camera than ``device`` -> refuse: they would bend every pose."""
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        K = np.asarray(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(data["distortion_coefficients"]["data"], dtype=np.float64)
    except (OSError, KeyError, ValueError, TypeError):
        return None, None
    if not np.any(D):
        return None, None
    if device and data.get("device") and not same_device(data["device"], device):
        raise SystemExit(
            f"{path} is the lens of {data['device']}, but this slot is now {device}: the camera "
            "list changed. Run `arena scan` (it re-measures each camera's lens).")
    cal = (int(data.get("image_width", size[0])), int(data.get("image_height", size[1])))
    if cal != tuple(size):
        raise SystemExit(
            f"{path} holds real intrinsics calibrated at {cal[0]}x{cal[1]}, but the camera "
            f"runs at {size[0]}x{size[1]}. Intrinsics are in pixels and do not carry across "
            "resolutions: re-run the checkerboard calibration at the capture size.")
    return K, D


def homography_path(cfg_dir: Path, slot: int) -> Path:
    return cfg_dir / ("arena_homography.yaml" if slot == 0 else f"arena_homography.{slot}.yaml")


def intrinsics_path(cfg_dir: Path, slot: int) -> Path:
    own = cfg_dir / f"camera_info.{slot}.yaml"
    return own if own.exists() else cfg_dir / "camera_info.yaml"


def parse_cameras(args) -> list[tuple[str | None, Path | None]]:
    """``--camera DEV[=HOMOGRAPHY]`` entries in slot order, else the one legacy camera
    (``--device`` / config/camera.yaml, ``--homography``)."""
    specs = list(getattr(args, "camera", None) or [])
    if not specs:
        return [(getattr(args, "device", None), Path(args.homography) if getattr(args, "homography", None) else None)]
    out = []
    for spec in specs:
        dev, _, h = spec.partition("=")
        out.append((dev or None, Path(h) if h else None))
    return out


def camera_name(device: str, slot: int) -> str:
    """'usb-046d_C270_HD_WEBCAM_520D0960-video-index0' -> 'C270 HD WEBCAM 520D0960'."""
    name = Path(str(device)).name
    for cut in ("usb-", "-video-index0", "-video-index1"):
        name = name.replace(cut, "")
    name = name.replace("_", " ").strip()
    return name or f"camera {slot}"


class CameraSlot:
    """One camera: its config, calibration, detector state and counters."""

    def __init__(self, index: int, cam: CameraConfig, calibration: dict, localizer: Localizer,
                 vehicle_ids: list[int], half: tuple[float, float]) -> None:
        self.index = index
        self.cam = cam
        self.calibration = calibration
        self.localizer = localizer
        self.name = camera_name(cam.device, index)
        self.source = None
        self.preview = None
        self.decode_errors = 0
        self.w_decode, self.w_detect, self.w_proc, self.w_lat = Window(), Window(), Window(), Window()
        self.vis = {vid: [] for vid in vehicle_ids}
        self.frame_times: list[float] = []
        self.drift_px: float | None = None
        self.duplicates: dict[int, int] = {}
        self.audit_ms = None
        self.last_found: dict = {}
        self.last_dets: list = []
        self.last_detect_ms = 0.0
        self._audit_frame = None
        self._audit_cond = threading.Condition()
        Hinv = np.linalg.inv(calibration["homography"])
        hw, hh = half
        self.arena_px = arena_frame.project_px_to_arena(
            Hinv, [[-hw, hh], [hw, hh], [hw, -hh], [-hw, -hh]]).round(1).tolist()

    def fps(self) -> float:
        ft = self.frame_times
        return (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 2 and ft[-1] > ft[0] else 0.0

    def vis_pct(self, vid: int) -> float:
        v = self.vis[vid]
        return round(100.0 * sum(v) / len(v), 1) if v else 0.0


class VisionService:
    def __init__(self, args, log=print) -> None:
        self.args = args
        self.log = log
        cfg = BUNDLE_ROOT / "config"
        self.marker_map = load_marker_map(args.marker_map or cfg / "marker_map.yaml").with_pursuers(
            getattr(args, "pursuers", None))

        import yaml
        arena_cfg = yaml.safe_load(Path(args.arena_config or cfg / "arena_test_6ft.yaml").read_text(encoding="utf-8"))
        self.half = (float(arena_cfg["arena_width_m"]) / 2.0, float(arena_cfg["arena_height_m"]) / 2.0)
        self.synthetic = bool(args.synthetic)
        self.roles = {self.marker_map.evader_id: "EVADER"}
        self.roles.update({pid: f"P{i + 1}" for i, pid in enumerate(self.marker_map.pursuer_ids)})
        self.vehicle_ids = list(self.marker_map.pursuer_ids) + [self.marker_map.evader_id]

        self.slots: list[CameraSlot] = []
        specs = [(getattr(args, "device", None), None)] if self.synthetic else parse_cameras(args)
        for i, (dev, hpath) in enumerate(specs):
            cam = load_camera_config(args.camera_config or cfg / "camera.yaml", slot=i or None, device=dev)
            if self.synthetic:
                from .synthetic import overhead_homography
                H = overhead_homography(cam.size, self.half, fill=0.9)
                calibration = {"homography": H, "image_size": list(cam.size), "corner_px": {},
                               "max_residual_m": 0.0, "calibrated_at_utc": "synthetic"}
                K, D = None, None
            else:
                hpath = hpath or homography_path(cfg, i)
                if not hpath.exists():
                    raise SystemExit(f"{hpath} not found: camera {i} ({cam.device}) has not been scanned. "
                                     "Run `arena scan`.")
                calibration = arena_frame.load_calibration_yaml(hpath)
                cal_size = calibration.get("image_size")
                if cal_size and tuple(cal_size) != cam.size:
                    raise SystemExit(
                        f"camera {i} was calibrated at {cal_size[0]}x{cal_size[1]} but captures "
                        f"{cam.width}x{cam.height}. A homography is pixel geometry, so every pose "
                        "would be scaled wrong. Run `arena scan` at the new size.")
                if calibration.get("device") and not same_device(calibration["device"], cam.device):
                    raise SystemExit(
                        f"{hpath} was fitted for {calibration['device']}, but camera {i} is now "
                        f"{cam.device}: the camera list changed order. Run `arena scan`.")
                K, D = load_intrinsics(intrinsics_path(cfg, i), cam.size, cam.device)
            # full_frame_every_s=None: the periodic whole-frame audit (duplicate
            # tags, bumped camera) runs on the audit thread instead of stalling
            # one frame in thirty on the hot path.
            localizer = Localizer(
                self.marker_map, calibration["homography"], image_size=cam.size,
                camera_matrix=K, dist_coeffs=D, arena_half_extent=self.half,
                roi_tracking=not args.no_roi, gating=not args.no_gating,
                full_frame_every_s=None,
                bounds_margin_m=float(getattr(args, "bounds_margin_m", 0.45)))
            self.slots.append(CameraSlot(i, cam, calibration, localizer, self.vehicle_ids, self.half))
        self.cam: CameraConfig = self.slots[0].cam          # the primary camera (dashboard overlay)
        self.calibration = self.slots[0].calibration
        self.localizer = self.slots[0].localizer
        self.arena_px = self.slots[0].arena_px

        self.fuser = PoseFuser(self.vehicle_ids)
        self.builder = FrameBuilder(pursuer_ids=self.marker_map.pursuer_ids,
                                    evader_id=self.marker_map.evader_id,
                                    hold_max_age_s=args.hold_max_age_s,
                                    evader_hold_max_age_s=getattr(args, "evader_hold_max_age_s", None),
                                    report_lost=not getattr(args, "no_report_lost", True))
        self.lock = threading.Lock()            # fuser + builder + send counters
        self.targets = parse_targets(args.pose_target, 9870)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, DSCP_EF)
        except (OSError, AttributeError):
            pass
        self.telemetry = TelemetrySender(args.telemetry)
        self.host = HostStats()
        self.epoch_offset = time.time() - time.perf_counter()
        self.stop_event = threading.Event()
        self.floor_px = family_spec(self.marker_map.dictionary).min_detect_px()
        if args.preview_port:
            from .preview import PreviewServer
            for s in self.slots:
                info = {"index": s.index, "name": s.name, "device": s.cam.device, "node": s.cam.device,
                        "mode": s.cam.mode, "floor_px": self.floor_px,
                        "roles": {str(k): v for k, v in self.roles.items()}}
                s.preview = PreviewServer(args.preview_port + s.index, max_fps=args.preview_fps, info=info)

        self.sent = self.suppressed = 0
        self.seq_out = 0
        self.fused_vis = {vid: [] for vid in self.vehicle_ids}
        self.fused_share: dict[int, dict] = {}
        self.last_fused: dict = {}

    # legacy attributes some tools read
    @property
    def preview(self):
        return self.slots[0].preview

    # ------------------------------------------------------------------ source
    def _make_source(self, slot: CameraSlot):
        if self.synthetic:
            from .capture import SyntheticSource
            from .synthetic import SceneRenderer
            renderer = SceneRenderer(self.marker_map, slot.calibration["homography"], slot.cam.size, self.half)
            hw, hh = self.half
            inset = 0.115
            corners = dict(zip(self.marker_map.calibration_corner_ids,
                               [(-hw + inset, hh - inset), (hw - inset, hh - inset),
                                (hw - inset, -hh + inset), (-hw + inset, -hh + inset)]))
            return SyntheticSource(renderer, self.marker_map, fps=slot.cam.fps,
                                   world_port=self.args.world_port, corners=corners, log=self.log)
        from .capture import V4l2MjpegSource
        return V4l2MjpegSource(slot.cam, log=lambda m: self.log(f"camera {slot.index}: {m}"))

    # -------------------------------------------------------------------- loop
    def run(self) -> int:
        import cv2

        # Detection stops getting faster past 2 threads (720p, measured on the Orin:
        # 9.7 ms at 1, 6.1 at 2, 6.2 at 6), and OpenCV's default of one per core makes the
        # hot loops, the audit threads and the feeds service fight over the same six cores.
        cv2.setNumThreads(int(self.args.cv_threads) if self.args.cv_threads else 3)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda *_: self.stop_event.set())
            except ValueError:        # not the main thread (tests)
                pass
        for s in self.slots:
            s.source = self._make_source(s).start()
            if s.preview is not None:
                s.preview.start()
            threading.Thread(target=self._audit_loop, args=(s,), name=f"audit{s.index}", daemon=True).start()
        self.log(f"vision up: {len(self.slots)} camera(s): "
                 + "; ".join(f"[{s.index}] {s.name} {s.cam.mode} {'synthetic' if self.synthetic else s.cam.device}"
                             for s in self.slots)
                 + f"; tracking {[(self.roles[v], v) for v in self.vehicle_ids]} "
                 f"({self.marker_map.dictionary}), poses -> {self.targets}, "
                 f"telemetry -> {self.args.telemetry or 'off'}"
                 + (f", previews :{self.args.preview_port}" + (f"-{self.args.preview_port + len(self.slots) - 1}"
                                                              if len(self.slots) > 1 else "")
                    if self.args.preview_port else ""))
        for s in self.slots:
            if s.localizer.vehicle_px_range:
                lo, hi = s.localizer.vehicle_px_range
                self.log(f"camera {s.index}: expected vehicle tag size {lo:.1f}-{hi:.1f} px")
            tag_h = self.marker_map.vehicle_tag_height_m
            if tag_h > 0.0 and s.localizer.camera_xyz is not None:
                cx, cy, cz = s.localizer.camera_xyz
                far = max(math.hypot(cx - sx * self.half[0], cy - sy * self.half[1]) for sx in (-1, 1) for sy in (-1, 1))
                self.log(f"camera {s.index}: at ({cx:+.2f}, {cy:+.2f}), {cz:.2f} m above the floor; roof tags "
                         f"{100 * tag_h:.0f} cm up are corrected for it (by up to {100 * tag_h / cz * far:.0f} cm "
                         f"at its far corner)")
            elif tag_h > 0.0:
                self.log(f"camera {s.index}: vehicle_tag_height_m is set but this camera's position cannot be "
                         f"worked out (no measured lens matrix, or it does not fit the calibration): its poses "
                         f"are NOT corrected for the tag's height")
        workers = [threading.Thread(target=self._worker, args=(s, cv2), name=f"cam{s.index}", daemon=True)
                   for s in self.slots]
        for w in workers:
            w.start()
        next_tel, next_log = 0.0, time.perf_counter() + 5.0
        try:
            while not self.stop_event.is_set():
                now = time.perf_counter()
                if now >= next_tel:
                    next_tel = now + 1.0 / self.args.telemetry_hz
                    self._telemetry()
                if now >= next_log:
                    next_log = now + 10.0
                    self._log_summary()
                self.stop_event.wait(0.02)
        finally:
            self.stop_event.set()
            for s in self.slots:
                if s.source is not None:
                    s.source.stop()
                if s.preview is not None:
                    s.preview.stop()
            self.sock.close()
            self.log(f"vision stopped: {self.sent} frames sent, {self.suppressed} suppressed")
        return 0

    def _worker(self, slot: CameraSlot, cv2) -> None:
        last_seq, next_audit = 0, 0.0
        while not self.stop_event.is_set():
            frame = slot.source.slot.get_newer(last_seq, timeout=0.5)
            if frame is None:
                continue
            last_seq = frame.seq
            gray = self._process(slot, frame, cv2)
            now = time.perf_counter()
            if gray is not None and now >= next_audit:
                next_audit = now + 1.0
                with slot._audit_cond:
                    slot._audit_frame = gray      # decoded fresh per frame, never reused: safe to share
                    slot._audit_cond.notify()

    def _audit_loop(self, slot: CameraSlot) -> None:
        """Once a second, on its own core: whole-frame detection for things the
        tracking windows cannot see: a second copy of a car's tag anywhere in
        view, and corner tags drifting (camera bumped)."""
        detector = slot.localizer.new_full_detector()
        while not self.stop_event.is_set():
            with slot._audit_cond:
                slot._audit_cond.wait(1.5)
                gray, slot._audit_frame = slot._audit_frame, None
            if gray is None:
                continue
            t0 = time.perf_counter()
            dets = slot.localizer.audit(gray, detector)
            slot.audit_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            counts: dict[int, int] = {}
            for d in dets:
                if d.marker_id in self.vehicle_ids:
                    counts[d.marker_id] = counts.get(d.marker_id, 0) + 1
            slot.duplicates = {vid: n for vid, n in counts.items() if n > 1}
            if slot.calibration.get("corner_px"):
                drift = corner_drift_px(dets, slot.calibration["corner_px"])
                if drift is not None:
                    slot.drift_px = drift

    def _process(self, slot: CameraSlot, frame, cv2):
        t0 = time.perf_counter()
        if frame.jpeg is not None:
            gray = cv2.imdecode(frame.jpeg, cv2.IMREAD_GRAYSCALE)
            if gray is None:              # a corrupt USB transfer: skip it, never guess
                slot.decode_errors += 1
                return None
        else:
            gray = frame.gray
        t1 = time.perf_counter()
        res = slot.localizer.process(gray, frame.t_capture)
        t2 = time.perf_counter()

        sizes = {d.marker_id: d.side_px for d in res.detections if d.marker_id in res.found}
        with self.lock:
            fused = self.fuser.update(slot.index, res.found, sizes, frame.t_capture)
            for vid, (x, y, h, t_est, share) in fused.items():
                self.builder.update({vid: (x, y, h)}, t_est + self.epoch_offset)
                self.fused_share[vid] = share
            self.last_fused = {vid: f[:3] for vid, f in fused.items()}
            for vid in self.vehicle_ids:
                v = self.fused_vis[vid]
                v.append(vid in fused)
                if len(v) > 90:
                    del v[:-90]
            now_mono = time.perf_counter()
            lat = now_mono - frame.t_capture
            payload = self.builder.build_frame(now_mono + self.epoch_offset,
                                               extra={"seq": self.seq_out + 1, "lat": round(lat, 5),
                                                      "cam": slot.index,
                                                      # the taped square the cameras were calibrated on
                                                      # (half width, half height, m): where "past the
                                                      # tape" is, for the controller's dead zone
                                                      "arena": [round(self.half[0], 4), round(self.half[1], 4)]})
            if payload is not None:
                data = payload.encode("utf-8")
                for target in self.targets:
                    try:
                        self.sock.sendto(data, target)
                    except OSError:
                        pass
                self.sent += 1
                self.seq_out += 1
            else:
                self.suppressed += 1
        t3 = time.perf_counter()

        slot.w_decode.add((t1 - t0) * 1000.0)
        slot.w_detect.add((t2 - t1) * 1000.0)
        slot.w_proc.add((t3 - t0) * 1000.0)
        slot.w_lat.add((t3 - frame.t_capture) * 1000.0)
        slot.frame_times.append(frame.t_capture)
        if len(slot.frame_times) > 90:
            del slot.frame_times[:-90]
        for vid in self.vehicle_ids:
            v = slot.vis[vid]
            v.append(vid in res.found)
            if len(v) > 90:
                del v[:-90]
        slot.last_found = res.found
        slot.last_dets = [[d.marker_id, d.corners.round(1).reshape(-1).tolist()] for d in res.detections]
        slot.last_detect_ms = res.detect_ms
        if slot.preview is not None and frame.jpeg is not None:
            dets, ms = res.detections, res.detect_ms

            def tags() -> bytes:
                return json.dumps({"ms": round(ms, 1),
                                   "d": [[d.marker_id, round(d.side_px, 1)]
                                         + d.corners.round(1).reshape(-1).tolist() for d in dets]}).encode()
            slot.preview.offer(frame.jpeg, int((frame.t_capture + self.epoch_offset) * 1000), tags)
        return gray

    # --------------------------------------------------------------- reporting
    def _fps(self) -> float:
        return self.slots[0].fps()

    @staticmethod
    def _worst(stats: list[dict]) -> dict:
        """The slowest camera's window stats, so an alert on p95 catches either camera."""
        stats = [s for s in stats if s]
        return max(stats, key=lambda s: s.get("p50", 0.0)) if stats else {}

    def _telemetry(self) -> None:
        slots = self.slots
        primary = slots[0]
        vehicles = []
        with self.lock:
            fused = dict(self.last_fused)
            share = {vid: dict(s) for vid, s in self.fused_share.items()}
            spread = dict(self.fuser.spread)
            fused_vis = {vid: list(v) for vid, v in self.fused_vis.items()}
        for vid in self.vehicle_ids:
            pose = fused.get(vid)
            vis = fused_vis[vid]
            vehicles.append({
                "id": vid, "role": self.roles[vid], "seen": pose is not None,
                "x": None if pose is None else round(pose[0], 4),
                "y": None if pose is None else round(pose[1], 4),
                "h": None if pose is None else round(pose[2], 4),
                "vis": round(100.0 * sum(vis) / len(vis), 1) if vis else 0.0,
                "cams": {str(c): round(w, 2) for c, w in share.get(vid, {}).items()},
                "spread_cm": None if vid not in spread else round(100.0 * spread[vid], 1),
            })
        statuses = [s.source.status if s.source is not None else "starting" for s in slots]
        stalled = [s for s in slots if s.source is not None and time.perf_counter() - (s.frame_times[-1] if s.frame_times else 0) > 1.0]
        status = ("stalled: " + ", ".join(f"camera {s.index} {s.source.status}" for s in stalled) if stalled
                  else statuses[0] if len(set(statuses)) == 1 else "; ".join(statuses))
        frames_by_slot = []
        for s in slots:
            loc = s.localizer.stats
            n = max(loc["frames"], 1)
            frames_by_slot.append({
                "i": s.index, "name": s.name, "device": s.cam.device, "mode": s.cam.mode,
                "status": s.source.status if s.source is not None else "starting",
                "fps": round(s.fps(), 1), "frames": s.source.frames if s.source is not None else 0,
                "skipped": s.source.slot.overwritten if s.source is not None else 0,
                "reconnects": s.source.reconnects if s.source is not None else 0,
                "decode_errors": s.decode_errors,
                "ms": {"decode": s.w_decode.stats(), "detect": s.w_detect.stats(),
                       "proc": s.w_proc.stats(), "lat": s.w_lat.stats()},
                "roi_pct": round(100.0 * loc["roi_frames"] / n, 1), "rescues": loc["rescues"],
                "rejected": dict(loc["rejected"]),
                "vis": {str(vid): s.vis_pct(vid) for vid in self.vehicle_ids},
                "dets": s.last_dets, "img": list(s.cam.size), "arena_px": s.arena_px,
                "drift_px": None if s.drift_px is None else round(s.drift_px, 1),
                "duplicates": {str(k): v for k, v in s.duplicates.items()},
                "audit_ms": s.audit_ms,
                "tag_px": None if not s.localizer.vehicle_px_range
                else [round(v, 1) for v in s.localizer.vehicle_px_range],
                "calib": {"residual_m": s.calibration.get("max_residual_m"),
                          "at": s.calibration.get("calibrated_at_utc")},
                "preview_port": self.args.preview_port + s.index if self.args.preview_port else None,
            })
        loc0 = primary.localizer.stats
        self.telemetry.send({
            "src": "vision", "wall": round(time.time(), 3),
            "status": status,
            "mode": primary.cam.mode, "actual": primary.source.actual_mode if primary.source is not None else None,
            "source": "synthetic" if self.synthetic else "v4l2",
            "fps": round(min(s.fps() for s in slots), 1),
            "frames": sum(c["frames"] for c in frames_by_slot),
            "skipped": sum(c["skipped"] for c in frames_by_slot),
            "reconnects": sum(c["reconnects"] for c in frames_by_slot),
            "decode_errors": sum(c["decode_errors"] for c in frames_by_slot),
            "ms": {k: self._worst([c["ms"][k] for c in frames_by_slot]) for k in ("decode", "detect", "proc", "lat")},
            "roi_pct": round(100.0 * loc0["roi_frames"] / max(loc0["frames"], 1), 1),
            "rescues": sum(c["rescues"] for c in frames_by_slot),
            "rejected": {k: sum(c["rejected"].get(k, 0) for c in frames_by_slot) for k in loc0["rejected"]},
            "sent": self.sent, "suppressed": self.suppressed,
            "vehicles": vehicles, "dets": primary.last_dets,
            "img": list(primary.cam.size), "arena": list(self.half), "arena_px": primary.arena_px,
            "drift_px": None if not any(s.drift_px is not None for s in slots)
            else round(max(s.drift_px for s in slots if s.drift_px is not None), 1),
            "duplicates": {str(k): v for s in slots for k, v in s.duplicates.items()},
            "audit_ms": primary.audit_ms,
            "calib": {"residual_m": primary.calibration.get("max_residual_m"),
                      "at": primary.calibration.get("calibrated_at_utc")},
            "tag_px": None if not primary.localizer.vehicle_px_range
            else [round(v, 1) for v in primary.localizer.vehicle_px_range],
            "host": self.host.sample(),
            "ts_source": getattr(primary.source.slot.peek(), "ts_source", None) if primary.source is not None else None,
            "roles": {str(k): v for k, v in self.roles.items()},
            "cams": frames_by_slot,
        })

    def _log_summary(self) -> None:
        for s in self.slots:
            lat, proc = s.w_lat.stats(), s.w_proc.stats()
            vis = " ".join(f"{self.roles[v]}:{s.vis_pct(v):.0f}%" for v in self.vehicle_ids)
            tag = f"cam{s.index} " if len(self.slots) > 1 else ""
            self.log(f"{tag}{s.fps():4.1f} fps | capture->sent p50 {lat.get('p50', 0):.1f} p95 "
                     f"{lat.get('p95', 0):.1f} ms | proc p50 {proc.get('p50', 0):.1f} ms | seen {vis} | "
                     f"roi {100.0 * s.localizer.stats['roi_frames'] / max(s.localizer.stats['frames'], 1):.0f}%"
                     + (f" | CORNER DRIFT {s.drift_px:.0f}px" if s.drift_px and s.drift_px > 8 else ""))
        if len(self.slots) > 1:
            with self.lock:
                fv = " ".join(f"{self.roles[v]}:{100.0 * sum(self.fused_vis[v]) / max(len(self.fused_vis[v]), 1):.0f}%"
                              for v in self.vehicle_ids)
                apart = " ".join(f"{self.roles[v]}:{100.0 * d:.1f}cm" for v, d in self.fuser.spread.items())
            self.log(f"fused: seen {fv} | sent {self.sent} suppressed {self.suppressed}"
                     + (f" | cameras apart {apart}" if apart else ""))
        else:
            self.log(f"sent {self.sent} suppressed {self.suppressed}")
