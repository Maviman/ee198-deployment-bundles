"""Frame sources for the fast path. All of them hand over the NEWEST frame only.

Latency rule: a frame that has been superseded is never processed. The camera
thread overwrites a one-slot mailbox and the processing loop always takes
what is in it, so a slow frame costs throughput (the next frame is skipped),
never latency (no queue can build up). The old ROS pipeline could queue
frames and, when over budget, ran 0.5-0.9 s stale.

V4l2MjpegSource
    Reads the webcam's MJPEG bytes RAW (no decode in the capture path) and
    stamps each frame with the V4L2 driver timestamp. uvcvideo takes that when
    the frame's first USB packet arrives, on CLOCK_MONOTONIC, the same clock
    as Python's time.perf_counter(). The service then decodes luma only, which is
    ~1.9x cheaper at 720p than decoding colour and converting to grey. The raw
    bytes double as the live preview for free (no re-encode on the vision Orin).
    Survives a camera unplug or a hung stream: reopens when no frame has
    arrived for STALL_REOPEN_S, re-applies the camera locks (a UVC camera
    forgets them when it re-enumerates), and the cars sit in dead-man neutral
    meanwhile.

SyntheticSource
    Renders real tag images at poses from a world-state UDP feed (tools/
    sim_world.py, closed loop: the policy's commands move the simulated car)
    or from a built-in chase script. Same Frame objects, so everything
    downstream is the production code path.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    seq: int
    t_capture: float              # time.perf_counter() domain
    jpeg: np.ndarray | None       # raw MJPEG bytes (1-D uint8), when the source has them
    gray: np.ndarray | None       # decoded luma, when the source decodes itself
    ts_source: str                # "driver" | "read" | "synthetic"


class LatestFrameSlot:
    """One-slot mailbox. put() overwrites; get_newer() waits for a newer seq."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._taken = -1
        self.overwritten = 0          # frames the consumer never saw (throughput loss, not latency)

    def put(self, frame: Frame) -> None:
        with self._cond:
            if self._frame is not None and self._taken != self._frame.seq:
                self.overwritten += 1
            self._frame = frame
            self._cond.notify_all()

    def get_newer(self, after_seq: int, timeout: float) -> Frame | None:
        deadline = time.perf_counter() + timeout
        with self._cond:
            while self._frame is None or self._frame.seq <= after_seq:
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or not self._cond.wait(remaining):
                    if self._frame is None or self._frame.seq <= after_seq:
                        return None
            self._taken = self._frame.seq
            return self._frame

    def peek(self) -> Frame | None:
        return self._frame


class _SourceBase:
    def __init__(self) -> None:
        self.slot = LatestFrameSlot()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames = 0
        self.reconnects = 0
        self.read_failures = 0
        self.status = "starting"
        self.actual_mode: tuple[int, int, float] | None = None

    def start(self) -> "_SourceBase":
        self._thread = threading.Thread(target=self._run, name=type(self).__name__, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


# OpenCV's V4L2 backend waits up to OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT seconds
# (default 10) inside grab() for a frame. run_vision.py / scan_arena.py set it
# to 1 before OpenCV starts, so a hung camera is noticed in seconds.
STALL_REOPEN_S = 3.0          # no frame this long while streaming -> reopen
FIRST_FRAME_GRACE_S = 6.0     # auto-exposure warm-up after (re)opening
# An unplugged camera fails grab() instantly (ENODEV), with no select wait.
# Release the device after a few of those at once: holding the dead file
# descriptor keeps /dev/videoN reserved, and the camera would re-enumerate as
# a different node.
FAST_FAIL_S = 0.2
FAST_FAILS_TO_REOPEN = 5


class V4l2MjpegSource(_SourceBase):
    def __init__(self, cam, log=print, controls_log=None) -> None:
        super().__init__()
        self.cam = cam
        self.log = log
        self.controls_log = controls_log or log
        self.raw_mode = True
        self._cap = None

    def _open(self):
        import cv2

        dev = self.cam.device
        cap = cv2.VideoCapture(int(dev) if str(dev).isdigit() else dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam.height)
        cap.set(cv2.CAP_PROP_FPS, self.cam.fps)
        # Two driver buffers: one being filled while we hold the other. More
        # buys nothing here because this thread dequeues as fast as frames come.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)   # hand us the MJPEG bytes, undecoded
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.actual_mode = (w, h, fps)
        if (w, h) != self.cam.size:
            cap.release()
            raise RuntimeError(
                f"camera {dev} gave {w}x{h}, not the configured {self.cam.width}x{self.cam.height}. "
                "It does not offer that MJPEG mode (see `v4l2-ctl --list-formats-ext`, or "
                "`arena scan`). Pick a mode it has in config/camera.yaml and re-scan the arena.")
        return cap

    def _run(self) -> None:
        import cv2

        backoff = 0.5
        seq = 0
        while not self._stop.is_set():
            if self._cap is None:
                try:
                    self._cap = self._open()
                except RuntimeError as exc:
                    self.status = f"error: {exc}"
                    self.log(str(exc))
                    self._stop.wait(5.0)
                    continue
                if self._cap is None:
                    self.status = f"cannot open {self.cam.device}"
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 3.0)
                    continue
                # Lock exposure / focus / frame rate on EVERY open: a camera that
                # re-enumerates (or was absent at start-up) comes back on defaults.
                from . import camera_controls
                camera_controls.apply(self.cam, log=self.controls_log)
                if seq:
                    self.reconnects += 1
                    self.log(f"camera reopened ({self.reconnects} reconnects); camera locks re-applied")
                self.status = "streaming"
                backoff = 0.5
                opened_at, got_frame, t_last_ok = time.perf_counter(), False, time.perf_counter()
                fast_fails = 0
            t_ask = time.perf_counter()
            ok = self._cap.grab()
            t_read = time.perf_counter()
            if not ok:
                self.read_failures += 1
                fast = (t_read - t_ask) < FAST_FAIL_S
                fast_fails = fast_fails + 1 if fast else 0
                limit = STALL_REOPEN_S if got_frame else FIRST_FRAME_GRACE_S
                stalled = t_read - (t_last_ok if got_frame else opened_at) > limit
                if fast_fails >= FAST_FAILS_TO_REOPEN or stalled:
                    # Unplugged (instant failures) or connected but hung (no frame
                    # for `limit`): drop the handle and reopen.
                    self.status = "camera lost; reopening"
                    self.log("camera unplugged; reopening" if not stalled
                             else f"no frame for {limit:.0f} s; reopening the camera")
                    self._cap.release()
                    self._cap = None
                elif fast:
                    self._stop.wait(0.02)          # never spin a core on a failing device
                continue
            got_frame, t_last_ok, fast_fails = True, t_read, 0
            ts_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            ok, buf = self._cap.retrieve()
            if not ok or buf is None:
                self.read_failures += 1
                continue
            t_drv = ts_ms / 1000.0
            if 0.0 <= t_read - t_drv <= 0.5:
                t_cap, src = t_drv, "driver"
            else:
                t_cap, src = t_read, "read"
            seq += 1
            self.frames += 1
            if buf.ndim == 1 or buf.shape[0] == 1:
                self.slot.put(Frame(seq, t_cap, buf.reshape(-1), None, src))
            else:
                # This OpenCV build ignored CONVERT_RGB=0 and decoded for us.
                # Still correct, just slower; keep going and say so once.
                if self.raw_mode:
                    self.raw_mode = False
                    self.log("note: OpenCV decoded MJPEG itself (raw mode unsupported); "
                             "the luma-only decode saving is lost on this build")
                gray = cv2.cvtColor(buf, cv2.COLOR_BGR2GRAY) if buf.ndim == 3 else buf
                self.slot.put(Frame(seq, t_cap, None, gray, src))
        if self._cap is not None:
            self._cap.release()


class SyntheticSource(_SourceBase):
    """Renders frames at ``fps`` from world state.

    ``world_port``: listen for {"vehicles": {id: [x, y, heading]}} datagrams
    (tools/sim_world.py). Without it, a ChaseScript moves the tags.
    """

    def __init__(self, renderer, marker_map, *, fps: float, world_port: int | None = None,
                 corners: dict | None = None, jpeg_quality: int = 85, log=print) -> None:
        super().__init__()
        from .synthetic import ChaseScript

        self.renderer = renderer
        self.marker_map = marker_map
        self.fps = float(fps)
        self.corners = corners or {}
        self.jpeg_quality = int(jpeg_quality)
        self.log = log
        self.world: dict[int, tuple[float, float, float]] = {}
        self.world_port = world_port
        self.script = ChaseScript(len(marker_map.pursuer_ids), renderer.half)
        self.actual_mode = (renderer.size[0], renderer.size[1], self.fps)
        self._sock = None
        if world_port:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(("127.0.0.1", int(world_port)))
            self._sock.setblocking(False)

    def _drain_world(self) -> None:
        while True:
            try:
                raw, _ = self._sock.recvfrom(65535)
            except (BlockingIOError, ConnectionResetError, OSError):
                return
            try:
                msg = json.loads(raw.decode("utf-8"))
                self.world = {int(k): tuple(v) for k, v in msg["vehicles"].items()}
            except (ValueError, KeyError, TypeError):
                continue

    def _run(self) -> None:
        import cv2

        period = 1.0 / self.fps
        start = time.perf_counter()
        next_t = start
        seq = 0
        self.status = "streaming (synthetic)"
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_t:
                self._stop.wait(next_t - now)
                continue
            next_t += period
            if next_t < now:               # fell behind: skip, like a real camera would
                next_t = now + period
            if self._sock is not None:
                self._drain_world()
                poses = self.world
            else:
                poses = self.script.poses(now - start, self.marker_map)
            t_cap = time.perf_counter()
            gray = self.renderer.render(poses, corners=self.corners)
            ok, jpg = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            seq += 1
            self.frames += 1
            self.slot.put(Frame(seq, t_cap, jpg.reshape(-1) if ok else None,
                                None if ok else gray, "synthetic"))
        if self._sock is not None:
            self._sock.close()
