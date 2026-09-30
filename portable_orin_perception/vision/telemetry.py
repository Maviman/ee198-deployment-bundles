"""Fire-and-forget telemetry to the arena hub, plus the host facts that explain
slowdowns: clocks not pinned, thermal throttling, CPU saturation.

Telemetry must never slow the hot path. It is one non-blocking UDP send per
report, errors swallowed, and it is sent after the pose frame goes out, never
before. If the hub is down the vision service does not notice or care.
"""

from __future__ import annotations

import glob
import json
import os
import socket
import time
from pathlib import Path


class TelemetrySender:
    def __init__(self, target: str | None) -> None:
        self.target = None
        self.sock = None
        if target:
            host, _, port = target.rpartition(":")
            try:
                host = socket.gethostbyname(host or "127.0.0.1")   # once, not per packet
            except OSError:
                return                                             # telemetry is optional
            self.target = (host, int(port))
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)

    def send(self, payload: dict) -> None:
        if self.sock is None:
            return
        try:
            self.sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), self.target)
        except OSError:
            pass            # hub down / buffer full: telemetry is optional, poses are not


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


class HostStats:
    """Cheap /proc and /sys reads, at most once per ``every_s``."""

    def __init__(self, every_s: float = 1.0) -> None:
        self.every_s = every_s
        self._last = 0.0
        self._cpu0 = (time.process_time(), time.monotonic())
        self.cache: dict = {}

    def sample(self) -> dict:
        now = time.monotonic()
        if now - self._last < self.every_s:
            return self.cache
        self._last = now
        out: dict = {}
        temps = []
        for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
            t = _read(f"{zone}/temp")
            kind = _read(f"{zone}/type") or ""
            if t and t.lstrip("-").isdigit() and "tj" not in kind.lower():
                temps.append((int(t) / 1000.0, kind))
        if temps:
            hottest = max(temps)
            out["temp_c"] = round(hottest[0], 1)
            out["temp_zone"] = hottest[1]
        cpu = "/sys/devices/system/cpu/cpu0/cpufreq"
        cur, lo, hi = (_read(f"{cpu}/scaling_cur_freq"), _read(f"{cpu}/scaling_min_freq"),
                       _read(f"{cpu}/cpuinfo_max_freq") or _read(f"{cpu}/scaling_max_freq"))
        gov = _read(f"{cpu}/scaling_governor")
        if cur and hi:
            out["cpu_ghz"] = round(int(cur) / 1e6, 2)
            out["cpu_max_ghz"] = round(int(hi) / 1e6, 2)
            # jetson_clocks pins min == max; a performance governor does too in practice.
            out["clocks_pinned"] = bool((lo and int(lo) >= 0.97 * int(hi)) or gov == "performance")
        if gov:
            out["governor"] = gov
        try:
            out["load1"] = round(os.getloadavg()[0], 2)
        except (AttributeError, OSError):
            pass
        pt, wall = time.process_time(), time.monotonic()
        dp, dw = pt - self._cpu0[0], wall - self._cpu0[1]
        self._cpu0 = (pt, wall)
        if dw > 0:
            out["proc_cpu_pct"] = round(100.0 * dp / dw, 1)
        self.cache = out
        return out


class Window:
    """Rolling samples for p50/p95/max over the last ``n`` values."""

    def __init__(self, n: int = 150) -> None:
        self.n = n
        self.values: list[float] = []

    def add(self, v: float) -> None:
        self.values.append(float(v))
        if len(self.values) > self.n:
            del self.values[: len(self.values) - self.n]

    def stats(self) -> dict:
        if not self.values:
            return {}
        s = sorted(self.values)
        pick = lambda q: s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]  # noqa: E731
        return {"p50": round(pick(0.5), 2), "p95": round(pick(0.95), 2), "max": round(s[-1], 2)}
