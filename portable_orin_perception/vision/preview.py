"""Live camera preview as MJPEG over HTTP, at zero encode cost.

The webcam already compresses every frame to JPEG, so the preview forwards
those bytes. The vision Orin never re-encodes or draws anything: detection
outlines are drawn in the browser from telemetry. It costs a memory copy and
a socket send, only while someone is watching, and is rate-limited below the
camera rate.

    GET /stream.mjpg     multipart MJPEG (what the hub's dashboard proxies)
    GET /snapshot.jpg    the latest frame
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BOUNDARY = b"arenaframe"


class PreviewServer:
    def __init__(self, port: int, *, max_fps: float = 10.0, bind: str = "0.0.0.0") -> None:
        self.max_fps = float(max_fps)
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._stored_at = 0.0
        self.clients = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):   # keep the service log about the pipeline
                pass

            def do_GET(self):
                if self.path.startswith("/snapshot"):
                    jpg = server.latest()
                    if jpg is None:
                        self.send_error(503, "no frame yet")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(jpg)
                    return
                if not self.path.startswith("/stream"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                server.clients += 1
                try:
                    last, period = 0, 1.0 / server.max_fps
                    while True:
                        t0 = time.perf_counter()
                        jpg, last = server.wait_newer(last, timeout=2.0)
                        if jpg is None:
                            continue
                        self.wfile.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                        time.sleep(max(0.0, period - (time.perf_counter() - t0)))
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass
                finally:
                    server.clients -= 1

        self.httpd = ThreadingHTTPServer((bind, int(port)), Handler)
        self.httpd.daemon_threads = True
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="preview", daemon=True)

    def start(self) -> "PreviewServer":
        self._thread.start()
        return self

    def offer(self, jpeg) -> None:
        """Called by the vision loop with each frame's raw JPEG. Copies every
        frame only while someone is watching; otherwise one per second, so
        /snapshot.jpg stays current."""
        now = time.perf_counter()
        if self.clients <= 0 and now - self._stored_at < 1.0:
            return
        with self._cond:
            self._stored_at = now
            self._jpeg = bytes(jpeg)
            self._seq += 1
            self._cond.notify_all()

    def latest(self) -> bytes | None:
        return self._jpeg

    def wait_newer(self, after: int, timeout: float) -> tuple[bytes | None, int]:
        with self._cond:
            if self._seq <= after:
                self._cond.wait(timeout)
            if self._seq <= after:
                return None, after
            return self._jpeg, self._seq

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
