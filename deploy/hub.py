"""Arena hub: collects telemetry from the vision service and the controller,
raises alerts, logs every run, and serves the live dashboard.

Runs on the control Orin (stdlib only, any python3). Nothing on the control
path depends on it: it is fed by fire-and-forget UDP, so a hung dashboard can
never slow a pose frame or a command.

    python3 deploy/hub.py --preview http://<vision-host>:8090 --control-port 9872

    GET  /                 the dashboard
    GET  /api/state        everything, as JSON (the terminal monitor polls this)
    GET  /video.mjpg       the camera, proxied from the vision Orin (one upstream
                           connection however many people are watching)
    POST /api/stop         disarm: every car gets E-stop on the next packet
                           (there is deliberately no arm endpoint; arming is
                           `arena go` from a terminal on the control Orin)
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Alert thresholds. Budget numbers come from the training-side robustness
# grid: ~0.1 s camera->command is the policy's tolerance, 0.2 s collapses it.
BUDGET_TOTAL_MS = 100.0
WARN_VISION_P95_MS = 50.0
WARN_VIS_PCT = 95.0
WARN_DRIFT_PX = 8.0
WARN_RTT_MS = 40.0
WARN_RSSI = -75
WARN_TEMP_C = 85.0


class Hub:
    def __init__(self, args) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.latest: dict[str, dict] = {}
        self.rx_at: dict[str, float] = {}
        self.series: deque = deque(maxlen=600)          # (wall, total_ms, vision_ms) ~60 s
        self.trails: dict[str, deque] = {}
        self.counters_prev: dict = {}
        self.events: deque = deque(maxlen=40)            # human-readable state changes
        self._last_state = None
        self.started = time.time()
        run_dir = Path(args.run_dir).expanduser() / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.log_fh = open(run_dir / "telemetry.jsonl", "a", encoding="utf-8", buffering=1)
        self._last_vision_log = 0.0
        # camera proxy
        self.jpeg: bytes | None = None
        self.jpeg_seq = 0
        self.jpeg_cond = threading.Condition()
        self.viewers = 0
        self._proxy_thread: threading.Thread | None = None

    # ------------------------------------------------------------- telemetry
    def ingest(self, msg: dict) -> None:
        src = msg.get("src")
        if src not in ("vision", "control"):
            return
        now = time.time()
        with self.lock:
            self.latest[src] = msg
            self.rx_at[src] = now
            if src == "vision":
                for v in msg.get("vehicles", []):
                    if v.get("x") is None:
                        continue
                    tr = self.trails.setdefault(str(v["id"]), deque(maxlen=150))
                    last = next((q for q in reversed(tr) if q is not None), None)
                    if last is not None and abs(last[0] - v["x"]) + abs(last[1] - v["y"]) > 0.3:
                        tr.append(None)          # a jump (car lifted / re-placed): break the line
                    tr.append((round(v["x"], 3), round(v["y"], 3)))
            else:
                tot = (msg.get("lat_ms") or {}).get("total") or {}
                vis = (msg.get("lat_ms") or {}).get("vision") or {}
                if tot:
                    self.series.append((round(now, 2), tot.get("p50"), vis.get("p50")))
                state = f"{msg.get('state')}/{'ARMED' if msg.get('armed') else 'disarmed'}"
                if state != self._last_state:
                    self.events.append((time.strftime("%H:%M:%S"), f"controller {state}"
                                        + (f" ({msg.get('stall_reason')})" if msg.get("state") == "STALLED" else "")))
                    self._last_state = state
        # Log controller ticks in full; vision at ~2 Hz and without the pixel dump.
        if src == "vision":
            if now - self._last_vision_log < 0.5:
                return
            self._last_vision_log = now
            msg = {k: v for k, v in msg.items() if k != "dets"}
        try:
            self.log_fh.write(json.dumps(msg, separators=(",", ":")) + "\n")
        except (OSError, ValueError):
            pass

    def udp_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.args.telemetry_port))
        while True:
            try:
                raw, _ = sock.recvfrom(65535)
                self.ingest(json.loads(raw.decode("utf-8")))
            except (ValueError, UnicodeDecodeError):
                continue
            except OSError:
                time.sleep(0.05)

    # ---------------------------------------------------------------- alerts
    def alerts(self) -> list[dict]:
        out: list[dict] = []
        now = time.time()

        def add(level, text):
            out.append({"level": level, "text": text})

        v, c = self.latest.get("vision"), self.latest.get("control")
        for name, msg in (("vision service", v), ("controller", c)):
            age = now - self.rx_at.get("vision" if name.startswith("vision") else "control", 0)
            if msg is None:
                add("warn", f"{name}: no telemetry yet")
            elif age > 2.0:
                add("error", f"{name}: silent for {age:.0f} s (crashed or stopped?)")
        if v and now - self.rx_at.get("vision", 0) <= 2.0:
            cfg_fps = float((v.get("mode") or "0@0").split("@")[-1] or 0)
            if v.get("status", "").startswith("stalled") or "lost" in v.get("status", ""):
                add("error", f"camera: {v.get('status')}")
            elif cfg_fps and v.get("fps", 0) < 0.85 * cfg_fps:
                add("warn", f"camera running {v.get('fps')} fps of {cfg_fps:g} configured")
            lat = (v.get("ms") or {}).get("lat") or {}
            if lat.get("p95", 0) > WARN_VISION_P95_MS:
                add("warn", f"perception latency p95 {lat['p95']:.0f} ms (> {WARN_VISION_P95_MS:.0f})")
            for veh in v.get("vehicles", []):
                if veh.get("vis", 100) < WARN_VIS_PCT:
                    add("warn" if veh["vis"] > 50 else "error",
                        f"{veh['role']} (tag {veh['id']}) seen in only {veh['vis']:.0f}% of frames")
            for vid, count in (v.get("duplicates") or {}).items():
                add("error", f"{count} copies of tag {vid} in view: a spare printed tag? "
                             "Identity is ambiguous whenever the car's window meets it")
            if v.get("drift_px") is not None and v["drift_px"] > WARN_DRIFT_PX:
                add("error", f"corner tags moved {v['drift_px']:.0f} px since calibration: "
                             "camera bumped? run `arena scan`")
            rej = v.get("rejected") or {}
            prev = self.counters_prev.get("rejected", {})
            new = {k: rej.get(k, 0) - prev.get(k, 0) for k in rej}
            if any(n > 0 for n in new.values()):
                self.counters_prev["rejected_recent"] = (now, new)
            recent = self.counters_prev.get("rejected_recent")
            if recent and now - recent[0] < 5.0:
                add("warn", "rejected implausible detections: "
                    + ", ".join(f"{k} {n}" for k, n in recent[1].items() if n))
            self.counters_prev["rejected"] = rej
            host = v.get("host") or {}
            if host.get("clocks_pinned") is False:
                add("warn", f"vision Orin clocks not pinned ({host.get('cpu_ghz')}/"
                            f"{host.get('cpu_max_ghz')} GHz): run `arena tune`")
            if host.get("temp_c", 0) > WARN_TEMP_C:
                add("warn", f"vision Orin at {host['temp_c']} C ({host.get('temp_zone')}): throttling risk")
            if v.get("decode_errors"):
                add("warn", f"{v['decode_errors']} corrupt camera frames (USB cable/hub?)")
        if c and now - self.rx_at.get("control", 0) <= 2.0:
            if c.get("state") == "STALLED":
                add("error", f"pose feed stalled: {c.get('stall_reason')}; cars E-stopped")
            tot = (c.get("lat_ms") or {}).get("total") or {}
            if tot.get("p95", 0) > BUDGET_TOTAL_MS:
                add("error", f"camera->command p95 {tot['p95']:.0f} ms is over the "
                             f"{BUDGET_TOTAL_MS:.0f} ms budget")
            if c.get("bad_frames"):
                add("warn", f"{c['bad_frames']} malformed pose frames skipped")
            for car in c.get("cars", []):
                if car.get("ack_age_s") is None:
                    add("error", f"car {car['idx']} ({car['addr']}) has never answered")
                elif car["ack_age_s"] > 1.0:
                    add("error", f"car {car['idx']} silent for {car['ack_age_s']:.1f} s")
                else:
                    if car.get("rtt_ms") and car["rtt_ms"] > WARN_RTT_MS:
                        add("warn", f"car {car['idx']} radio round trip {car['rtt_ms']:.0f} ms")
                    if car.get("rssi") is not None and car["rssi"] < WARN_RSSI:
                        add("warn", f"car {car['idx']} weak WiFi ({car['rssi']} dBm)")
                    if car.get("slot") is False:
                        add("error", f"car {car['idx']} finds no command at its CAR_INDEX")
        return out

    def state(self) -> dict:
        now = time.time()
        with self.lock:
            return {
                "now": now, "hub_up_s": round(now - self.started),
                "vision": self.latest.get("vision"), "control": self.latest.get("control"),
                "age": {k: round(now - t, 2) for k, t in self.rx_at.items()},
                "alerts": self.alerts(),
                "series": list(self.series)[-300:],
                "trails": {k: list(v) for k, v in self.trails.items()},
                "events": list(self.events)[-12:],
                "run_dir": str(self.run_dir),
                "preview": bool(self.args.preview),
            }

    # ---------------------------------------------------------------- control
    def disarm(self) -> dict:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        try:
            s.sendto(b'{"cmd":"disarm"}', ("127.0.0.1", self.args.control_port))
            return json.loads(s.recvfrom(1024)[0].decode("utf-8"))
        except (OSError, ValueError) as exc:
            return {"error": f"controller did not answer ({exc}); if it is not running the "
                             "cars are already in failsafe"}
        finally:
            s.close()

    # ------------------------------------------------------------ video proxy
    def _proxy(self) -> None:
        """One upstream MJPEG connection to the vision Orin, alive while anyone watches."""
        url = self.args.preview.rstrip("/") + "/stream.mjpg"
        while self.viewers > 0:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    buf = b""
                    while self.viewers > 0:
                        chunk = resp.read1(65536)   # whatever has arrived; read() would wait for 64 KB
                        if not chunk:
                            break
                        buf += chunk
                        while True:
                            start = buf.find(b"\xff\xd8")
                            end = buf.find(b"\xff\xd9", start + 2)
                            if start < 0 or end < 0:
                                if len(buf) > 4_000_000:
                                    buf = b""
                                break
                            jpg, buf = buf[start:end + 2], buf[end + 2:]
                            with self.jpeg_cond:
                                self.jpeg, self.jpeg_seq = jpg, self.jpeg_seq + 1
                                self.jpeg_cond.notify_all()
            except OSError:
                time.sleep(1.0)
        self._proxy_thread = None

    def watch(self):
        self.viewers += 1
        if self._proxy_thread is None and self.args.preview:
            self._proxy_thread = threading.Thread(target=self._proxy, daemon=True)
            self._proxy_thread.start()

    def unwatch(self):
        self.viewers -= 1


def make_handler(hub: Hub):
    page = (HERE / "dashboard.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path.startswith("/api/state"):
                self._send(200, json.dumps(hub.state()).encode("utf-8"), "application/json")
            elif self.path.startswith("/video.mjpg"):
                self._stream()
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path.startswith("/api/stop"):
                self._send(200, json.dumps(hub.disarm()).encode("utf-8"), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def _stream(self):
            if not hub.args.preview:
                self._send(404, b"no preview configured", "text/plain")
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            hub.watch()
            last = 0
            try:
                while True:
                    with hub.jpeg_cond:
                        if hub.jpeg_seq <= last:
                            hub.jpeg_cond.wait(2.0)
                        if hub.jpeg_seq <= last:
                            continue
                        jpg, last = hub.jpeg, hub.jpeg_seq
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                hub.unwatch()
                self.close_connection = True

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080, help="dashboard HTTP port")
    ap.add_argument("--telemetry-port", type=int, default=9871)
    ap.add_argument("--control-port", type=int, default=9872, help="the controller's localhost arm/disarm port")
    ap.add_argument("--preview", default="", help="vision preview base URL, e.g. http://10.42.0.1:8090")
    ap.add_argument("--run-dir", default="~/.arena/runs")
    args = ap.parse_args()
    hub = Hub(args)
    threading.Thread(target=hub.udp_loop, daemon=True).start()
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(hub))
    httpd.daemon_threads = True
    print(f"[hub] dashboard on http://0.0.0.0:{args.port}  telemetry udp :{args.telemetry_port}  "
          f"logging to {hub.run_dir}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hub.log_fh.close()


if __name__ == "__main__":
    main()
