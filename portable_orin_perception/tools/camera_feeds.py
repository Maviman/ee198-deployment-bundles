"""Live feeds from every camera on the vision Orin, for aiming and calibrating.

    serve   (on the vision Orin) each camera's own MJPEG bytes over HTTP, through
            the capture path and camera locks the vision service uses
            (vision/capture.py, config/camera.yaml): no re-encode. While someone
            watches, every frame sent is searched for tags and its part carries
            them (X-Tags), which is what the dashboard draws.
            --secondary serves every camera EXCEPT the vision service's
            (config/camera.yaml's device): `arena up` runs it as the `feeds`
            service, so the dashboard shows the other corner cameras too.
    view    (on any machine with a display) one window per stream, newest frame
            only, with the stream's name, resolution, frame rate and a centre
            cross, and the tags the vision service's detector finds in it
            (t toggles the tag overlay). q or Esc closes every window.

Tag overlay: the detection engine of the vision service (Localizer, the same
subset dictionary and id mapping as config/marker_map.yaml), on the full frame
with untuned parameters and no gating, so it shows every tag the camera CAN
see, not just the ones a calibrated pipeline would accept, plus CornerTracker's
zoomed second look at each corner tag. Each outline is drawn on the frame it
was found in. Green = found in at least 95% of the last 30 frames, orange =
less: it drops out. The label gives the tag's short x long side in pixels
(a far corner tag is foreshortened: its short side is what limits detection).
The dot marks the tag's first corner (printed top-left). The bottom line gives
each id's hit rate over the last 30 searched frames.

    python3 tools/camera_feeds.py serve                  # every camera in /dev/v4l/by-id
    python3 tools/camera_feeds.py serve --device /dev/video0 --device /dev/video2
    python3 tools/camera_feeds.py view C270=http://10.42.0.1:8091/stream.mjpg \\
                                       X1=http://10.42.0.1:8092/stream.mjpg

Camera i of `ls /dev/v4l/by-id/*-video-index0` (sorted) is always on port
8091 + i, whichever cameras one server holds: /stream.mjpg, /snapshot.jpg and
/info.json (name, device, mode, tag floor, roles). Each stream part has
X-Frame-Time (unix ms at capture) and X-Tags, JSON {"ms": detect ms, "d":
[[id, side_px, x0, y0, x1, y1, x2, y2, x3, y3], ...]}. A plain `serve` holds
every camera, so it must not run beside the vision service: `arena up` stops
it first.
"""

from __future__ import annotations

import os

# Before OpenCV opens any camera: grab() gives up after 1 s instead of 10
# (same as run_vision.py), so a hung camera is reopened in seconds.
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "1")

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import urllib.request  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BY_ID = Path("/dev/v4l/by-id")


def find_cameras() -> list[str]:
    """Stable /dev/v4l/by-id paths of every capture node (a UVC camera's
    index1 node is metadata only)."""
    return sorted(str(p) for p in BY_ID.glob("*-video-index0")) if BY_ID.is_dir() else []


def select_cameras(all_cams: list[str], *, devices: list[str] | None = None,
                   exclude: str | None = None) -> list[tuple[int, str]]:
    """[(port offset, device)]. A camera's offset is its place in ``all_cams`` (the
    sorted by-id list), so its port does not depend on which others are served;
    a device outside that list takes the next free offset. ``exclude`` drops the
    camera that path names (by /dev node, so a by-id path and /dev/videoN match)."""
    real = [os.path.realpath(c) for c in all_cams]
    chosen = []
    for dev in devices or all_cams:
        r = os.path.realpath(dev)
        if exclude is not None and r == os.path.realpath(exclude):
            continue
        k = real.index(r) if r in real else len(all_cams) + len(chosen)
        chosen.append((k, dev))
    return chosen


def tag_roles(marker_map) -> dict[int, str]:
    """id -> the vision service's role names (P1.. for pursuers), corners as "corner"."""
    roles = {cid: "corner" for cid in marker_map.calibration_corner_ids}
    roles.update({pid: f"P{i + 1}" for i, pid in enumerate(marker_map.pursuer_ids)})
    roles[marker_map.evader_id] = "EVADER"
    return roles


class CornerTracker:
    """A second look at the corner tags, which do not move: once the full-frame search
    has found one, every frame also searches a small window around where it was,
    upscaled 2x.

    A corner camera sees the far corners steeply foreshortened, ~13 x 30 px at 720p:
    2.2 px per module on the short axis, under the ~3.5 the full-frame detector wants.
    Measured on both corner cameras on 2026-11-06 (150-frame clips of the static arena,
    22 detector settings): the full frame found the far tags in 84-99% of frames in
    15-17 ms; 2x windows found them in 100%, with no false positives, in 5-6 ms for three
    tags. 3x windows and contrast stretching (CLAHE) did worse; ArUco3's fast mode lost
    every far tag; raising the camera's sharpness turned the floor's texture into
    candidates (8-12x slower on the C270) without beating the windows. A tag not seen
    for FORGET_S is forgotten, so a moved camera starts over from the full frame."""

    SCALE = 2
    FORGET_S = 3.0

    def __init__(self, marker_map) -> None:
        import cv2

        self.ids = set(marker_map.calibration_corner_ids)
        self.tag_set = marker_map.tag_set()
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 2
        self.detector = cv2.aruco.ArucoDetector(self.tag_set.build_opencv_dictionary(), p)
        self.known: dict[int, tuple[float, float, float, float]] = {}   # id -> (cx, cy, long side px, t)

    def update(self, gray, dets: list, t: float) -> list:
        """This frame's full-frame ``dets``, with every known corner tag taken from its
        window instead when it is found there."""
        import cv2
        import numpy as np

        from hive_perception.core.localizer import Detection, side_px_of

        out = {d.marker_id: d for d in dets}
        h, w = gray.shape[:2]
        for tid, (cx, cy, long_px, _t) in self.known.items():
            half = int(min(max(1.5 * long_px, 32), 200))
            x0, y0 = max(0, int(cx) - half), max(0, int(cy) - half)
            crop = gray[y0:min(h, int(cy) + half), x0:min(w, int(cx) + half)]
            if crop.size == 0:
                continue
            big = cv2.resize(crop, None, fx=self.SCALE, fy=self.SCALE, interpolation=cv2.INTER_CUBIC)
            corners, ids, _ = self.detector.detectMarkers(big)
            if ids is None:
                continue
            for c, row in zip(corners, ids.flatten()):
                if self.tag_set.to_real_id(int(row)) == tid:
                    # resize maps source pixel centres x to (x + 0.5) * SCALE - 0.5
                    c = (c.reshape(4, 2).astype(np.float64) + 0.5) / self.SCALE - 0.5 + (x0, y0)
                    out[tid] = Detection(tid, c, side_px_of(c))
                    break
        for tid, d in out.items():
            if tid in self.ids:
                sides = np.linalg.norm(d.corners - np.roll(d.corners, -1, axis=0), axis=1)
                cx, cy = d.corners.mean(axis=0)
                self.known[tid] = (float(cx), float(cy), float(sides.max()), t)
        for tid in [k for k, v in self.known.items() if t - v[3] > self.FORGET_S]:
            del self.known[tid]
        return list(out.values())


class TaggedFeed:
    """One camera on one port. A frame goes out only while someone watches, at most
    ``max_fps``, and each one is searched for tags ONCE (however many viewers) before
    it is sent, so its X-Tags describe exactly that image."""

    BOUNDARY = b"arenaframe"

    def __init__(self, src, port: int, *, info: dict, marker_map, max_fps: float,
                 bind: str = "0.0.0.0") -> None:
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.src = src
        self.info = info
        self.marker_map = marker_map
        self.max_fps = float(max_fps)
        self.clients = 0
        self._cond = threading.Condition()
        self._part: tuple[bytes, bytes, int] | None = None     # (jpeg, X-Tags json, unix ms)
        self._seq = 0
        feed = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def _send(self, body: bytes, ctype: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/info"):
                    self._send(json.dumps(feed.info).encode(), "application/json")
                    return
                if self.path.startswith("/snapshot"):
                    frame = feed.src.slot.peek()
                    if frame is None or frame.jpeg is None:
                        self.send_error(503, "no frame yet")
                    else:
                        self._send(bytes(frame.jpeg), "image/jpeg")
                    return
                if not self.path.startswith("/stream"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type",
                                 f"multipart/x-mixed-replace; boundary={feed.BOUNDARY.decode()}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                feed.clients += 1
                try:
                    last = 0
                    while True:
                        part, last = feed.wait_newer(last, timeout=2.0)
                        if part is None:
                            continue
                        jpg, tags, t_ms = part
                        self.wfile.write(b"--" + feed.BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\nX-Frame-Time: {t_ms}\r\n".encode()
                                         + b"X-Tags: " + tags + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass
                finally:
                    feed.clients -= 1

        self.httpd = ThreadingHTTPServer((bind, int(port)), Handler)
        self.httpd.daemon_threads = True

    def start(self) -> "TaggedFeed":
        threading.Thread(target=self.httpd.serve_forever, name="feed-http", daemon=True).start()
        threading.Thread(target=self._run, name="feed-tags", daemon=True).start()
        return self

    def wait_newer(self, after: int, timeout: float):
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after, timeout)
            return (self._part, self._seq) if self._seq > after else (None, after)

    def _run(self) -> None:
        import json

        import cv2

        from hive_perception.core.localizer import Localizer

        loc, last, period = None, 0, 1.0 / self.max_fps
        corners = CornerTracker(self.marker_map)
        while True:
            if self.clients <= 0:
                time.sleep(0.2)
                continue
            t0 = time.perf_counter()
            frame = self.src.slot.get_newer(last, timeout=1.0)
            if frame is None:
                continue
            last = frame.seq
            if frame.jpeg is not None:
                jpg = bytes(frame.jpeg)
                gray = cv2.imdecode(frame.jpeg, cv2.IMREAD_GRAYSCALE)
            else:                                   # this OpenCV build decoded for us
                gray = frame.gray
                jpg = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            if gray is None:                        # a corrupt frame: skip it
                continue
            size = (gray.shape[1], gray.shape[0])
            if loc is None or loc.image_size != size:
                loc = Localizer(self.marker_map, image_size=size, roi_tracking=False, gating=False)
            t1 = time.perf_counter()
            dets = corners.update(gray, loc.audit(gray), t1)
            ms = (time.perf_counter() - t1) * 1000.0
            tags = json.dumps({"ms": round(ms, 1), "d": [
                [d.marker_id, round(d.side_px, 1), *[round(float(v), 1) for v in d.corners.reshape(-1)]]
                for d in dets]}, separators=(",", ":")).encode()
            t_ms = int((time.time() - (time.perf_counter() - frame.t_capture)) * 1000)
            with self._cond:
                self._part, self._seq = (jpg, tags, t_ms), self._seq + 1
                self._cond.notify_all()
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def card_name(device: str) -> str:
    try:
        out = subprocess.run(["v4l2-ctl", "-d", device, "--info"], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return Path(device).name
    for line in out.splitlines():
        if "Card type" in line:
            return line.split(":", 1)[1].strip()
    return Path(device).name


def serve(args) -> int:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "hive_perception"))
    from hive_perception.core.camera_config import load_camera_config
    from hive_perception.core.frame_builder import load_marker_map
    from hive_perception.core.tag_family import family_spec
    from vision.capture import V4l2MjpegSource

    base = load_camera_config(args.camera_config or ROOT / "config" / "camera.yaml")
    if args.mode:
        w, rest = args.mode.split("x")
        h, fps = rest.split("@")
        base = base.with_overrides(width=int(w), height=int(h), fps=int(fps))
    marker_map = load_marker_map(args.marker_map or ROOT / "config" / "marker_map.yaml")
    roles = {str(k): v for k, v in tag_roles(marker_map).items()}
    floor_px = family_spec(marker_map.dictionary).min_detect_px()
    chosen = select_cameras(find_cameras(), devices=args.device,
                            exclude=base.device if args.secondary else None)
    if not chosen:
        if args.secondary:
            print(f"no camera besides the vision service's ({base.device}): nothing to serve")
            return 0
        print("no camera found (ls /dev/v4l/by-id/ is empty): is one plugged in?")
        return 1

    feeds = []
    for k, dev in chosen:
        cam = base.with_overrides(device=dev)
        name = card_name(dev)
        tag = f"cam{k}"

        def log(msg, tag=tag):
            print(f"[{tag}] {msg}", flush=True)

        src = V4l2MjpegSource(cam, log=log).start()
        info = {"index": k, "name": name, "device": dev, "node": os.path.realpath(dev),
                "mode": cam.mode, "floor_px": floor_px, "roles": roles}
        feed = TaggedFeed(src, args.port + k, info=info, marker_map=marker_map, max_fps=args.fps).start()
        feeds.append((tag, name, cam, src, feed))
        print(f"{tag}: {name}\n      {dev} ({info['node']}), {cam.mode}\n"
              f"      http://0.0.0.0:{args.port + k}/stream.mjpg", flush=True)

    counts = {tag: src.frames for tag, _n, _c, src, _p in feeds}
    t_last = time.perf_counter()
    try:
        while True:
            time.sleep(args.stats_s)
            now = time.perf_counter()
            parts = []
            for tag, _name, _cam, src, feed in feeds:
                fps = (src.frames - counts[tag]) / (now - t_last)
                counts[tag] = src.frames
                mode = "x".join(map(str, src.actual_mode[:2])) if src.actual_mode else "?"
                parts.append(f"{tag} {fps:4.1f} fps {mode} {src.status}, "
                             f"{src.reconnects} reopen, {feed.clients} viewer(s)")
            t_last = now
            print(time.strftime("%H:%M:%S ") + " | ".join(parts), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for _tag, _name, _cam, src, feed in feeds:
            src.stop()
            feed.stop()
    return 0


class StreamReader:
    """Reads a multipart MJPEG stream; keeps only the newest decoded frame, and the
    part's X-Tags when it has them (a `serve` stream), parsed, else None."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.frame = None
        self.tags = None
        self.seq = 0
        self.t_frame = 0.0
        self.fps = 0.0
        self.error = "connecting"
        self._cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def wait_newer(self, after: int, timeout: float):
        """(frame, tags, seq) once a frame newer than ``after`` has arrived, else (None, None, after)."""
        with self._cond:
            self._cond.wait_for(lambda: self.seq > after, timeout)
            return (self.frame, self.tags, self.seq) if self.seq > after else (None, None, after)

    def _run(self) -> None:
        import json

        import cv2
        import numpy as np

        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=5) as r:
                    self.error = ""
                    n, t0 = 0, time.perf_counter()
                    while True:
                        length, tags = None, None
                        while True:                       # part headers up to the blank line
                            line = r.readline()
                            if not line:
                                raise ConnectionError("stream ended")
                            line = line.strip()
                            if not line and length is not None:
                                break
                            if line.lower().startswith(b"content-length:"):
                                length = int(line.split(b":", 1)[1])
                            elif line.lower().startswith(b"x-tags:"):
                                try:
                                    tags = json.loads(line.split(b":", 1)[1])
                                except ValueError:
                                    tags = None
                        jpg = r.read(length)
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        with self._cond:
                            self.frame, self.tags, self.t_frame = img, tags, time.perf_counter()
                            self.seq += 1
                            self._cond.notify_all()
                        n += 1
                        if self.t_frame - t0 >= 1.0:
                            self.fps, n, t0 = n / (self.t_frame - t0), 0, self.t_frame
            except Exception as exc:                      # noqa: BLE001 - any failure: retry
                self.error = f"{type(exc).__name__}: {exc}"
                time.sleep(1.0)


def put_text(img, text, org, color, scale) -> None:
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


class TagOverlay:
    """Keeps each new frame of a stream with the tags in it, so an outline never sits on
    a later image. A `serve` stream already says (X-Tags, found on the vision Orin);
    any other stream is searched here, on this thread (an OpenCV detector must not be
    shared across threads)."""

    def __init__(self, reader: StreamReader, marker_map, window: int = 30) -> None:
        self.reader = reader
        self.marker_map = marker_map
        self.roles = tag_roles(marker_map)
        self.history: deque[set[int]] = deque(maxlen=window)
        self.result = None            # (frame, detections) of the newest searched frame
        self.ms = 0.0
        self.fps = 0.0
        self.where = ""               # "vision Orin" when the stream carried the tags
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        import cv2

        from hive_perception.core.localizer import Localizer

        import numpy as np

        loc, last, n, t0 = None, 0, 0, time.perf_counter()
        corners = CornerTracker(self.marker_map)
        while True:
            img, tags, last = self.reader.wait_newer(last, timeout=1.0)
            if img is None:
                continue
            if tags is not None:
                dets = [types.SimpleNamespace(marker_id=int(d[0]), side_px=float(d[1]),
                                              corners=np.asarray(d[2:10], np.float64).reshape(4, 2))
                        for d in tags.get("d", [])]
                self.ms, self.where, done = float(tags.get("ms", 0.0)), "vision Orin", time.perf_counter()
            else:
                size = (img.shape[1], img.shape[0])
                if loc is None or loc.image_size != size:
                    loc = Localizer(self.marker_map, image_size=size, roi_tracking=False, gating=False)
                t = time.perf_counter()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = corners.update(gray, loc.audit(gray), t)
                done = time.perf_counter()
                self.ms, self.where = (done - t) * 1000.0, "here"
            self.history.append({d.marker_id for d in dets})
            self.result = (img, dets)
            n += 1
            if done - t0 >= 1.0:
                self.fps, n, t0 = n / (done - t0), 0, done

    def draw(self, img, dets, scale: float) -> None:
        import cv2
        import numpy as np

        n = len(self.history)
        rate = {i: sum(i in s for s in self.history) / n for i in set().union(*self.history)} if n else {}
        for d in dets:
            color = (0, 255, 0) if rate.get(d.marker_id, 0.0) >= 0.95 else (0, 165, 255)
            pts = np.round(d.corners).astype(np.int32)
            cv2.polylines(img, [pts], True, color, 3, cv2.LINE_AA)
            cv2.circle(img, (int(pts[0, 0]), int(pts[0, 1])), 6, (255, 0, 255), -1, cv2.LINE_AA)
            sides = np.linalg.norm(d.corners - np.roll(d.corners, -1, axis=0), axis=1)
            label = f"{d.marker_id} {self.roles.get(d.marker_id, '?')} {sides.min():.0f}x{sides.max():.0f}px"
            (tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            org = (max(5, min(int(pts[:, 0].min()), img.shape[1] - tw - 5)),
                   max(20, int(pts[:, 1].min()) - 12))
            put_text(img, label, org, color, scale)
        rates = "  ".join(f"{i}:{100 * rate[i]:.0f}%" for i in sorted(rate))
        ih = img.shape[0]
        put_text(img, f"tags, last {n} frames: {rates or 'none'}", (10, ih - int(50 * scale)),
                 (255, 255, 255), scale)
        put_text(img, f"detect {self.ms:.0f} ms ({self.where}), {self.fps:.0f} fps   green = found in >= 95% "
                      f"of frames, orange = drops out", (10, ih - int(15 * scale)), (255, 255, 255), scale)


def view(args) -> int:
    import cv2
    import numpy as np

    sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "hive_perception"))
    from hive_perception.core.frame_builder import load_marker_map

    marker_map = load_marker_map(args.marker_map or ROOT / "config" / "marker_map.yaml")
    streams = []
    for spec in args.streams:
        name, _, url = spec.partition("=") if "=" in spec.split("://")[0] else ("", "", spec)
        reader = StreamReader(url)
        streams.append((name or url, reader, TagOverlay(reader, marker_map)))
    show_tags = not args.no_tags

    w, h = args.window
    for i, (name, _r, _o) in enumerate(streams):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, w, h)
        cv2.moveWindow(name, 10 + i * (w + 20), 60)

    while True:
        now = time.perf_counter()
        for name, r, overlay in streams:
            searched = overlay.result if show_tags else None
            img = searched[0] if searched is not None else r.frame
            img = np.zeros((h, w, 3), np.uint8) if img is None else img.copy()
            ih, iw = img.shape[:2]
            scale = max(0.5, iw / 1280 * 0.8)
            stale = r.frame is None or now - r.t_frame > 2.0
            cx, cy = iw // 2, ih // 2
            cv2.line(img, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
            cv2.line(img, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)
            if searched is not None:
                overlay.draw(img, searched[1], scale)
            label = f"{name}  {iw}x{ih}  {r.fps:4.1f} fps"
            if stale:
                label += "  NO SIGNAL" + (f" ({r.error})" if r.error else "")
            put_text(img, label, (10, int(30 * scale + 6)), (0, 0, 255) if stale else (0, 255, 0), scale)
            cv2.imshow(name, img)
        key = cv2.waitKey(15) & 0xFF
        closed = any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name, _r, _o in streams)
        if key == ord("t"):
            show_tags = not show_tags
        if key in (ord("q"), 27) or closed:
            break
    cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve every camera on this machine as MJPEG")
    s.add_argument("--device", action="append", help="camera to serve (repeat); default: all in /dev/v4l/by-id")
    s.add_argument("--camera-config", default=None, help="default: config/camera.yaml (+ camera.local.yaml)")
    s.add_argument("--mode", default=None, help="override the capture mode, e.g. 1920x1080@30")
    s.add_argument("--secondary", action="store_true",
                   help="skip the vision service's camera (config/camera.yaml device); `arena up` uses this")
    s.add_argument("--marker-map", default=None, help="default: config/marker_map.yaml")
    s.add_argument("--port", type=int, default=8091, help="camera i of the by-id list is on port + i")
    s.add_argument("--fps", type=float, default=15.0,
                   help="max frames per second sent (and searched for tags) while someone watches")
    s.add_argument("--stats-s", type=float, default=10.0, help="seconds between status lines")
    v = sub.add_parser("view", help="open one window per MJPEG stream")
    v.add_argument("streams", nargs="+", help="URL or name=URL")
    v.add_argument("--window", type=int, nargs=2, default=(800, 450), metavar=("W", "H"))
    v.add_argument("--marker-map", default=None, help="default: config/marker_map.yaml")
    v.add_argument("--no-tags", action="store_true", help="start with the tag overlay off (t toggles)")
    args = ap.parse_args(argv)
    return serve(args) if args.cmd == "serve" else view(args)


if __name__ == "__main__":
    sys.exit(main())
