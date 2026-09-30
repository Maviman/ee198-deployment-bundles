"""Behavioural tests of the real-time control loop (pc_controller.realtime),
over real UDP sockets on localhost, with the real ONNX policy:

    cd portable_n1_controller && python -m pytest tests -q

Each test runs the loop in a thread, plays pose frames at it the way the
vision service does (one per camera frame, with ``seq``/``lat``), and
listens as a car would.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pc_controller.esp_link import EspLink  # noqa: E402
from pc_controller.portable_loop import PortableLoop  # noqa: E402
from pc_controller.realtime import run_realtime  # noqa: E402


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Car:
    """Records every command packet; ACKs like the firmware."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.05)
        self.port = self.sock.getsockname()[1]
        self.packets: list[tuple[float, dict]] = []
        self.stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                raw, addr = self.sock.recvfrom(4096)
            except (socket.timeout, ConnectionResetError, OSError):
                continue
            pkt = json.loads(raw)
            self.packets.append((time.perf_counter(), pkt))
            self.sock.sendto(json.dumps({"ack": pkt["seq"], "car": 0, "applied": True,
                                         "slot": True, "rssi": -40}).encode(), addr)


class Harness:
    def __init__(self, *, start_disarmed=False, max_pose_age_s=0.15) -> None:
        self.pose_port, self.ctl_port = free_port(), free_port()
        self.car = Car()
        self.loop = PortableLoop(str(ROOT / "models" / "n1_catch"))
        self.link = EspLink([("127.0.0.1", self.car.port)])
        self.logs: list[str] = []
        self.thread = threading.Thread(target=self._run, args=(start_disarmed, max_pose_age_s), daemon=True)
        self.thread.start()
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        time.sleep(0.3)

    def _run(self, start_disarmed, max_pose_age_s) -> None:
        try:
            run_realtime(self.loop, self.link, pose_port=self.pose_port, control_port=self.ctl_port,
                         start_disarmed=start_disarmed, max_pose_age_s=max_pose_age_s,
                         log=self.logs.append)
        except OSError:
            pass

    def frame(self, t: float, lat: float = 0.005, raw: bytes | None = None) -> None:
        self.seq += 1
        payload = raw or json.dumps({
            "t": t, "pursuers": [{"x": -0.5 + 0.1 * t, "y": 0.0, "heading": 0.0}],
            "evader": {"x": 0.5, "y": 0.2, "heading": 1.0}, "seq": self.seq, "lat": lat}).encode()
        self.tx.sendto(payload, ("127.0.0.1", self.pose_port))

    def stream(self, seconds: float, fps: float = 30.0, **kw) -> None:
        start = time.perf_counter()
        n = 0
        while True:
            due = start + n / fps
            now = time.perf_counter()
            if now - start >= seconds:
                return
            if now < due:
                time.sleep(due - now)
            self.frame(time.perf_counter(), **kw)
            n += 1

    def command(self, cmd: str) -> dict:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2.0)
        s.sendto(json.dumps({"cmd": cmd}).encode(), ("127.0.0.1", self.ctl_port))
        return json.loads(s.recvfrom(1024)[0])

    def close(self) -> None:
        self.car.stop.set()


@pytest.fixture
def harness():
    hs = []

    def make(**kw):
        h = Harness(**kw)
        hs.append(h)
        return h
    yield make
    for h in hs:
        h.close()


def driving(packets):
    return [p for _t, p in packets if not p["estop"]]


def test_ticks_at_10hz_from_a_30fps_stream(harness):
    h = harness()
    h.stream(2.0)
    cmds = [(t, p) for t, p in h.car.packets if not p["estop"]]
    assert 16 <= len(cmds) <= 23, f"{len(cmds)} commands in 2 s at 30 fps; expected ~20 (10 Hz)"
    gaps = [b[0] - a[0] for a, b in zip(cmds, cmds[1:])]
    assert 0.08 < sum(gaps) / len(gaps) < 0.12


def test_disarmed_sends_only_estop_until_armed(harness):
    h = harness(start_disarmed=True)
    h.stream(0.6)
    assert h.car.packets and not driving(h.car.packets), "a disarmed controller moved a car"
    assert h.command("arm")["armed"] is True
    before = len(h.car.packets)
    h.stream(0.6)
    assert driving(h.car.packets[before:]), "armed but no drive commands"
    assert h.command("disarm")["armed"] is False
    after = len(h.car.packets)
    h.stream(0.4)
    assert h.car.packets[after][1]["estop"], "disarm did not E-stop immediately"
    assert not driving(h.car.packets[after:]), "still driving after disarm"


def test_stale_poses_are_never_acted_on_and_stop_the_cars(harness):
    h = harness(max_pose_age_s=0.15)
    h.stream(0.5)
    assert driving(h.car.packets)
    time.sleep(0.03)                  # let the last fresh frame's command land first
    n = len(h.car.packets)
    h.stream(0.8, lat=0.4)            # perception latency far over budget
    later = [p for _t, p in h.car.packets[n:]]
    assert not [p for p in later if not p["estop"]], "acted on a stale pose"
    assert any(p["estop"] for p in later), "no E-stop when only stale poses arrived"
    assert any("too old" in line for line in h.logs)


def test_silence_estops_then_rearms(harness):
    h = harness()
    h.stream(0.5)
    time.sleep(0.6)
    assert h.car.packets[-1][1]["estop"], "no E-stop after the pose feed went silent"
    n = len(h.car.packets)
    h.stream(0.5)
    assert driving(h.car.packets[n:]), "did not re-arm when frames resumed"


def test_malformed_frames_are_skipped_not_fatal(harness):
    h = harness()
    for _ in range(5):
        h.frame(0.0, raw=b"{not json")
        h.frame(0.0, raw=b'{"t": 1, "pursuers": []}')
    h.stream(0.5)
    assert h.thread.is_alive()
    assert driving(h.car.packets)
