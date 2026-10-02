"""Behavioural tests of the real-time control loop (pc_controller.realtime),
over real UDP sockets on localhost, with the real ONNX policy:

    cd portable_n1_controller && python -m pytest tests -q

Each test runs the loop in a thread, plays pose frames at it the way the
vision service does (one per camera frame, with ``seq``/``lat``), and
listens as a car would.

Every test runs twice: with the car on UDP, and with the car on a UART cable
(a pseudo-terminal opened by pyserial, ``--esp serial:...``). The failsafe
behaviour -- E-stop on silence, on stale poses, on disarm; re-arm on resume --
must be identical whichever way the car is wired, so it is the same tests
rather than look-alikes. The serial runs are skipped where pyserial is absent.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from pc_controller.esp_link import EspLink  # noqa: E402
from pc_controller.portable_loop import PortableLoop  # noqa: E402
from pc_controller.realtime import run_realtime  # noqa: E402
from pc_controller.serial_link import SerialTarget  # noqa: E402
from serial_board import PtyBoard  # noqa: E402


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


class SerialCar:
    """The same recorder as Car, with the car on a UART cable: a pty, and every
    command packet arrives as a line. ``packets`` has Car's shape."""

    def __init__(self) -> None:
        self.board = PtyBoard()
        self.path = self.board.path
        self.stop = types.SimpleNamespace(set=self.board.close)

    @property
    def packets(self) -> list[tuple[float, dict]]:
        return self.board.packets


class Harness:
    def __init__(self, *, start_disarmed=False, max_pose_age_s=0.15, transport="udp") -> None:
        self.transport = transport
        self.pose_port, self.ctl_port = free_port(), free_port()
        self.car = SerialCar() if transport == "serial" else Car()
        self.loop = PortableLoop(str(ROOT / "models" / "n1_catch"))
        if transport == "serial":
            self.link = EspLink([SerialTarget(self.car.path)])
            self.link.wait_ready()          # the board proves its CAR_INDEX before anything is driven
        else:
            self.link = EspLink([("127.0.0.1", self.car.port)])
        self.logs: list[str] = []
        self.halt = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(start_disarmed, max_pose_age_s), daemon=True)
        self.thread.start()
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        time.sleep(0.3)

    def _run(self, start_disarmed, max_pose_age_s) -> None:
        try:
            run_realtime(self.loop, self.link, pose_port=self.pose_port, control_port=self.ctl_port,
                         start_disarmed=start_disarmed, max_pose_age_s=max_pose_age_s,
                         log=self.logs.append, stop=self.halt)
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
        # Stop the loop first and only then release the link: a finished test must leave no thread
        # behind (a serial endpoint would keep retrying its old /dev/pts/N, which the next test's pty
        # may be reusing), and nothing may close the link out from under a loop that is mid-tick.
        self.halt.set()
        self.thread.join(timeout=3.0)
        assert not self.thread.is_alive(), "the control loop did not stop"
        self.link.close()
        self.tx.close()
        self.car.stop.set()


@pytest.fixture(params=["udp", "serial"])
def harness(request):
    if request.param == "serial" and importlib.util.find_spec("serial") is None:
        pytest.skip("pyserial is only needed for the serial transport")
    hs = []

    def make(**kw):
        h = Harness(transport=request.param, **kw)
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
    t_disarm = time.time()
    assert h.command("disarm")["armed"] is False
    h.stream(0.4)
    # The recorder can lag the wire (a pty hands bytes over through a kernel workqueue), so a drive
    # frame sent just BEFORE the disarm may be recorded just after it. What matters is order of
    # sending: once the first E-stop that follows driving has gone out, nothing but E-stops may follow
    # it, and it must have gone out at once. Packets carry their own send time and seq for this.
    seen = [p for _t, p in h.car.packets[before:]]
    assert [p["seq"] for p in seen] == sorted(p["seq"] for p in seen), "frames reordered or repeated"
    first_stop = next(i for i, p in enumerate(seen) if p["estop"] and any(not q["estop"] for q in seen[:i]))
    assert not [p for p in seen[first_stop:] if not p["estop"]], "still driving after disarm"
    assert seen[first_stop]["t"] - t_disarm < 0.15, "disarm did not E-stop immediately"


def test_stale_poses_are_never_acted_on_and_stop_the_cars(harness):
    h = harness(max_pose_age_s=0.15)
    h.stream(0.5)
    assert driving(h.car.packets)
    n = len(h.car.packets)
    t_stale = time.time()
    h.stream(0.8, lat=0.4)            # perception latency far over budget
    time.sleep(0.1)                   # let the last frames reach the recorder
    # Judged by send time (the packet's own "t"): a fresh frame's command sent before t_stale may
    # still be on its way to the recorder, but nothing sent after it may be a drive command.
    after_stale = [p for _t, p in h.car.packets if p["t"] > t_stale + 0.02]
    assert not [p for p in after_stale if not p["estop"]], "acted on a stale pose"
    assert any(p["estop"] for _t, p in h.car.packets[n:]), "no E-stop when only stale poses arrived"
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
