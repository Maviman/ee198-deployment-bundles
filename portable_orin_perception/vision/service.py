"""The vision service: newest camera frame -> luma decode -> localize -> pose
frame out over UDP, immediately, for every frame.

Nothing waits on a timer. The ROS bridge resampled poses on a fixed 10 Hz
timer, which aged every pose by up to one camera frame interval before the
controller saw it. Here a frame's pose leaves this process ~decode+detect
after the frame arrives, and the controller picks which frames to act on (one
per 100 ms control tick, see portable_n1_controller/run_controller.py).

Clock: pose ``t`` is the frame's capture time on CLOCK_MONOTONIC (Python's
perf_counter on Linux) plus a fixed epoch offset taken at start-up. It never jumps (safe for velocity
finite-differencing) and still reads as wall-clock seconds, so cross-machine
latency tools keep working when chrony is running.

Safety semantics are the FrameBuilder's, unchanged: any vehicle unseen for
more than hold_max_age_s -> no frame is sent -> the controller's 0.3 s
dead-man stops the cars. Camera unplugged, process crash, bumped calibration:
each one ends in silence, and silence stops the cars.
"""

from __future__ import annotations

import signal
import socket
import threading
import time
from pathlib import Path

import numpy as np

from hive_perception.core import arena_frame
from hive_perception.core.camera_config import CameraConfig, load_camera_config
from hive_perception.core.frame_builder import FrameBuilder, load_marker_map
from hive_perception.core.localizer import Localizer, corner_drift_px

from . import BUNDLE_ROOT
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


def load_intrinsics(path: Path, size: tuple[int, int]):
    """(K, D) or (None, None). Placeholder/all-zero distortion -> (None, None)
    (undistortion would be a no-op). Real intrinsics at another resolution ->
    refuse: they are in pixels and would scale every pose."""
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        K = np.asarray(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(data["distortion_coefficients"]["data"], dtype=np.float64)
    except (OSError, KeyError, ValueError, TypeError):
        return None, None
    if not np.any(D):
        return None, None
    cal = (int(data.get("image_width", size[0])), int(data.get("image_height", size[1])))
    if cal != tuple(size):
        raise SystemExit(
            f"{path} holds real intrinsics calibrated at {cal[0]}x{cal[1]}, but the camera "
            f"runs at {size[0]}x{size[1]}. Intrinsics are in pixels and do not carry across "
            "resolutions: re-run the checkerboard calibration at the capture size.")
    return K, D


class VisionService:
    def __init__(self, args, log=print) -> None:
        self.args = args
        self.log = log
        cfg = BUNDLE_ROOT / "config"
        self.marker_map = load_marker_map(args.marker_map or cfg / "marker_map.yaml").with_pursuers(
            getattr(args, "pursuers", None))
        cam = load_camera_config(args.camera_config or cfg / "camera.yaml")
        self.cam: CameraConfig = cam.with_overrides(device=args.device)

        import yaml
        arena_cfg = yaml.safe_load(Path(args.arena_config or cfg / "arena_test_6ft.yaml").read_text(encoding="utf-8"))
        self.half = (float(arena_cfg["arena_width_m"]) / 2.0, float(arena_cfg["arena_height_m"]) / 2.0)

        self.synthetic = bool(args.synthetic)
        if self.synthetic:
            from .synthetic import overhead_homography
            H = overhead_homography(self.cam.size, self.half, fill=0.9)
            self.calibration = {"homography": H, "image_size": list(self.cam.size), "corner_px": {},
                                "max_residual_m": 0.0, "calibrated_at_utc": "synthetic"}
        else:
            hpath = Path(args.homography or cfg / "arena_homography.yaml")
            if not hpath.exists():
                raise SystemExit(f"{hpath} not found: the arena has not been scanned. Run `arena scan`.")
            self.calibration = arena_frame.load_calibration_yaml(hpath)
            cal_size = self.calibration.get("image_size")
            if cal_size and tuple(cal_size) != self.cam.size:
                raise SystemExit(
                    f"the arena was calibrated at {cal_size[0]}x{cal_size[1]} but config/camera.yaml "
                    f"captures {self.cam.width}x{self.cam.height}. A homography is pixel geometry, so "
                    "every pose would be scaled wrong. Run `arena scan` at the new size.")
        K, D = (None, None) if self.synthetic else load_intrinsics(cfg / "camera_info.yaml", self.cam.size)

        # full_frame_every_s=None: the periodic whole-frame audit (duplicate
        # tags, bumped camera) runs on the audit thread below instead of stalling
        # one frame in thirty on the hot path.
        self.localizer = Localizer(
            self.marker_map, self.calibration["homography"], image_size=self.cam.size,
            camera_matrix=K, dist_coeffs=D, arena_half_extent=self.half,
            roi_tracking=not args.no_roi, gating=not args.no_gating,
            full_frame_every_s=None)
        self._audit_frame = None
        self._audit_cond = threading.Condition()
        self.duplicates: dict[int, int] = {}
        self.audit_ms = None
        self.builder = FrameBuilder(pursuer_ids=self.marker_map.pursuer_ids,
                                    evader_id=self.marker_map.evader_id,
                                    hold_max_age_s=args.hold_max_age_s)
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
        self.preview = None
        if args.preview_port:
            from .preview import PreviewServer
            self.preview = PreviewServer(args.preview_port, max_fps=args.preview_fps)

        Hinv = np.linalg.inv(self.calibration["homography"])
        hw, hh = self.half
        self.arena_px = arena_frame.project_px_to_arena(
            Hinv, [[-hw, hh], [hw, hh], [hw, -hh], [-hw, -hh]]).round(1).tolist()
        self.roles = {self.marker_map.evader_id: "EVADER"}
        self.roles.update({pid: f"P{i + 1}" for i, pid in enumerate(self.marker_map.pursuer_ids)})
        self.vehicle_ids = list(self.marker_map.pursuer_ids) + [self.marker_map.evader_id]

        # counters and rolling windows
        self.sent = self.suppressed = self.decode_errors = 0
        self.w_decode, self.w_detect, self.w_proc, self.w_lat = Window(), Window(), Window(), Window()
        self.vis = {vid: [] for vid in self.vehicle_ids}
        self.frame_times: list[float] = []
        self.drift_px: float | None = None
        self.last_found: dict = {}
        self.last_dets: list = []

    # ------------------------------------------------------------------ source
    def _make_source(self):
        if self.synthetic:
            from .capture import SyntheticSource
            from .synthetic import SceneRenderer
            renderer = SceneRenderer(self.marker_map, self.calibration["homography"], self.cam.size, self.half)
            hw, hh = self.half
            inset = 0.115
            corners = dict(zip(self.marker_map.calibration_corner_ids,
                               [(-hw + inset, hh - inset), (hw - inset, hh - inset),
                                (hw - inset, -hh + inset), (-hw + inset, -hh + inset)]))
            return SyntheticSource(renderer, self.marker_map, fps=self.cam.fps,
                                   world_port=self.args.world_port, corners=corners, log=self.log)
        from .capture import V4l2MjpegSource
        return V4l2MjpegSource(self.cam, log=self.log)      # applies the camera locks on every open

    # -------------------------------------------------------------------- loop
    def run(self) -> int:
        import cv2

        if self.args.cv_threads:
            cv2.setNumThreads(int(self.args.cv_threads))
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda *_: self.stop_event.set())
            except ValueError:        # not the main thread (tests)
                pass
        source = self._make_source().start()
        if self.preview is not None:
            self.preview.start()
        threading.Thread(target=self._audit_loop, name="audit", daemon=True).start()
        self.log(f"vision up: {self.cam.mode} {'synthetic' if self.synthetic else self.cam.device}, "
                 f"tracking {[(self.roles[v], v) for v in self.vehicle_ids]} "
                 f"({self.marker_map.dictionary}), poses -> {self.targets}, "
                 f"telemetry -> {self.args.telemetry or 'off'}"
                 + (f", preview :{self.args.preview_port}" if self.preview else ""))
        if self.localizer.vehicle_px_range:
            lo, hi = self.localizer.vehicle_px_range
            self.log(f"expected vehicle tag size {lo:.1f}-{hi:.1f} px")

        last_seq, next_tel, next_log, next_audit = 0, 0.0, time.perf_counter() + 5.0, 0.0
        try:
            while not self.stop_event.is_set():
                frame = source.slot.get_newer(last_seq, timeout=0.5)
                now = time.perf_counter()
                if frame is None:
                    if now >= next_tel:
                        next_tel = now + 0.5
                        self._telemetry(source, stalled=True)
                    continue
                last_seq = frame.seq
                gray = self._process(frame, cv2)
                now = time.perf_counter()
                if gray is not None and now >= next_audit:
                    next_audit = now + 1.0
                    with self._audit_cond:
                        self._audit_frame = gray      # decoded fresh per frame, never reused: safe to share
                        self._audit_cond.notify()
                if now >= next_tel:
                    next_tel = now + 1.0 / self.args.telemetry_hz
                    self._telemetry(source)
                if now >= next_log:
                    next_log = now + 10.0
                    self._log_summary(source)
        finally:
            source.stop()
            if self.preview is not None:
                self.preview.stop()
            self.sock.close()
            self.log(f"vision stopped: {self.sent} frames sent, {self.suppressed} suppressed")
        return 0

    def _audit_loop(self) -> None:
        """Once a second, on its own core: whole-frame detection for things the
        tracking windows cannot see: a second copy of a car's tag anywhere in
        view, and corner tags drifting (camera bumped)."""
        detector = self.localizer.new_full_detector()
        while not self.stop_event.is_set():
            with self._audit_cond:
                self._audit_cond.wait(1.5)
                gray, self._audit_frame = self._audit_frame, None
            if gray is None:
                continue
            t0 = time.perf_counter()
            dets = self.localizer.audit(gray, detector)
            self.audit_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            counts: dict[int, int] = {}
            for d in dets:
                if d.marker_id in self.vehicle_ids:
                    counts[d.marker_id] = counts.get(d.marker_id, 0) + 1
            self.duplicates = {vid: n for vid, n in counts.items() if n > 1}
            if self.calibration.get("corner_px"):
                drift = corner_drift_px(dets, self.calibration["corner_px"])
                if drift is not None:
                    self.drift_px = drift

    def _process(self, frame, cv2):
        t0 = time.perf_counter()
        if frame.jpeg is not None:
            gray = cv2.imdecode(frame.jpeg, cv2.IMREAD_GRAYSCALE)
            if gray is None:              # a corrupt USB transfer: skip it, never guess
                self.decode_errors += 1
                return None
        else:
            gray = frame.gray
        t1 = time.perf_counter()
        res = self.localizer.process(gray, frame.t_capture)
        t2 = time.perf_counter()

        cap_t = frame.t_capture + self.epoch_offset
        self.builder.update(res.found, cap_t)
        now_mono = time.perf_counter()
        lat = now_mono - frame.t_capture
        payload = self.builder.build_frame(now_mono + self.epoch_offset,
                                           extra={"seq": frame.seq, "lat": round(lat, 5)})
        if payload is not None:
            data = payload.encode("utf-8")
            for target in self.targets:
                try:
                    self.sock.sendto(data, target)
                except OSError:
                    pass
            self.sent += 1
        else:
            self.suppressed += 1
        t3 = time.perf_counter()

        self.w_decode.add((t1 - t0) * 1000.0)
        self.w_detect.add((t2 - t1) * 1000.0)
        self.w_proc.add((t3 - t0) * 1000.0)
        self.w_lat.add((t3 - frame.t_capture) * 1000.0)
        self.frame_times.append(frame.t_capture)
        if len(self.frame_times) > 90:
            del self.frame_times[:-90]
        for vid in self.vehicle_ids:
            v = self.vis[vid]
            v.append(vid in res.found)
            if len(v) > 90:
                del v[:-90]
        self.last_found = res.found
        self.last_dets = [[d.marker_id, d.corners.round(1).reshape(-1).tolist()] for d in res.detections]
        if self.preview is not None and frame.jpeg is not None:
            self.preview.offer(frame.jpeg)
        return gray

    # --------------------------------------------------------------- reporting
    def _fps(self) -> float:
        ft = self.frame_times
        return (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 2 and ft[-1] > ft[0] else 0.0

    def _telemetry(self, source, stalled: bool = False) -> None:
        loc = self.localizer.stats
        frames = max(loc["frames"], 1)
        vehicles = []
        for vid in self.vehicle_ids:
            pose = self.last_found.get(vid)
            vis = self.vis[vid]
            vehicles.append({
                "id": vid, "role": self.roles[vid], "seen": pose is not None,
                "x": None if pose is None else round(pose[0], 4),
                "y": None if pose is None else round(pose[1], 4),
                "h": None if pose is None else round(pose[2], 4),
                "vis": round(100.0 * sum(vis) / len(vis), 1) if vis else 0.0,
            })
        self.telemetry.send({
            "src": "vision", "wall": round(time.time(), 3),
            "status": "stalled: " + source.status if stalled else source.status,
            "mode": self.cam.mode, "actual": source.actual_mode,
            "source": "synthetic" if self.synthetic else "v4l2",
            "fps": round(self._fps(), 1), "frames": source.frames,
            "skipped": source.slot.overwritten, "reconnects": source.reconnects,
            "decode_errors": self.decode_errors,
            "ms": {"decode": self.w_decode.stats(), "detect": self.w_detect.stats(),
                   "proc": self.w_proc.stats(), "lat": self.w_lat.stats()},
            "roi_pct": round(100.0 * loc["roi_frames"] / frames, 1), "rescues": loc["rescues"],
            "rejected": dict(loc["rejected"]),
            "sent": self.sent, "suppressed": self.suppressed,
            "vehicles": vehicles, "dets": self.last_dets,
            "img": list(self.cam.size), "arena": list(self.half), "arena_px": self.arena_px,
            "drift_px": None if self.drift_px is None else round(self.drift_px, 1),
            "duplicates": {str(k): v for k, v in self.duplicates.items()},
            "audit_ms": self.audit_ms,
            "calib": {"residual_m": self.calibration.get("max_residual_m"),
                      "at": self.calibration.get("calibrated_at_utc")},
            "tag_px": None if not self.localizer.vehicle_px_range
            else [round(v, 1) for v in self.localizer.vehicle_px_range],
            "host": self.host.sample(),
            "ts_source": getattr(source.slot.peek(), "ts_source", None),
            "roles": {str(k): v for k, v in self.roles.items()},
        })

    def _log_summary(self, source) -> None:
        lat, proc = self.w_lat.stats(), self.w_proc.stats()
        vis = " ".join(f"{self.roles[v]}:{(100.0 * sum(self.vis[v]) / max(len(self.vis[v]), 1)):.0f}%"
                       for v in self.vehicle_ids)
        self.log(f"{self._fps():4.1f} fps | capture->sent p50 {lat.get('p50', 0):.1f} p95 "
                 f"{lat.get('p95', 0):.1f} ms | proc p50 {proc.get('p50', 0):.1f} ms | seen {vis} | "
                 f"sent {self.sent} suppressed {self.suppressed} | "
                 f"roi {100.0 * self.localizer.stats['roi_frames'] / max(self.localizer.stats['frames'], 1):.0f}%"
                 + (f" | CORNER DRIFT {self.drift_px:.0f}px" if self.drift_px and self.drift_px > 8 else ""))
