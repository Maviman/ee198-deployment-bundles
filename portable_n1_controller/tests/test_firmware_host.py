"""The firmware's behaviour, with the real esp32_receiver.ino running on this PC.

    cd portable_n1_controller && python -m pytest tests/test_firmware_host.py -q

esp32/test_host/sketch_host.cpp includes the sketch itself and runs it against
thin Arduino stubs, with a clock the test controls. Frames are built by the
controller's own code (pc_controller.esp_link) and written to the sketch's UART
byte for byte, so this covers the whole contract: framing, the shared packet
handler, the 300 ms dead-man, replies. What it does not cover is the chip
(GPIO electrics, the real WiFi stack, UART timing): that still needs the car.

Needs a C++ compiler and the ArduinoJson headers PlatformIO fetches (run
`pio run` once in esp32/, or point ARDUINOJSON_DIR at ArduinoJson/src); skipped
otherwise. The sketch is built from a staged copy with placeholder WiFi
credentials, so a real wifi_credentials.h is never involved.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pc_controller.esp_link import command_packet, encode_packet  # noqa: E402
from pc_controller.serial_link import PROBE_FRAME  # noqa: E402

FW = ROOT / "esp32"
HOST = FW / "test_host"
SKETCH = FW / "esp32_receiver" / "esp32_receiver.ino"

FWD, BACK, LEFT, RIGHT = 1, 2, 42, 41           # the sketch's default button GPIOs
Event = namedtuple("Event", "ms kind text")


def find_arduinojson() -> Path | None:
    candidates = []
    if os.environ.get("ARDUINOJSON_DIR"):
        candidates.append(Path(os.environ["ARDUINOJSON_DIR"]))
    candidates += sorted(FW.glob(".pio/libdeps/*/ArduinoJson"))
    candidates.append(Path.home() / ".platformio" / "lib" / "ArduinoJson")
    for c in candidates:
        for d in (c / "src", c):
            if (d / "ArduinoJson.h").exists() and (d / "ArduinoJson").is_dir():
                return d
    return None


PLACEHOLDER_CREDENTIALS = '#define WIFI_SSID "host-test"\n#define WIFI_PASS "host-test"\n'


def stage_sketch(src_dir: Path, dest: Path) -> Path:
    """A copy of the sketch with placeholder credentials. The real wifi_credentials.h (gitignored, and
    usually defining OTA_PASSWORD, which would drag in ArduinoOTA) never takes part in a host build."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in src_dir.iterdir():
        if f.is_file() and not f.name.startswith("wifi_credentials.h"):
            shutil.copy(f, dest / f.name)
    (dest / "wifi_credentials.h").write_text(PLACEHOLDER_CREDENTIALS)
    return dest / "esp32_receiver.ino"


def build_host(sketch: Path, out: Path) -> Path:
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler on this machine")
    aj = find_arduinojson()
    if aj is None:
        pytest.skip("ArduinoJson headers not found: run `pio run` in esp32/ once, or set ARDUINOJSON_DIR")
    cmd = [cxx, "-std=c++11", "-O1", "-Wall", "-Wno-deprecated-declarations", "-Wno-unused-function",
           f"-I{HOST / 'stubs'}", f"-I{sketch.parent}", f"-I{aj}",
           f'-DSKETCH="{sketch}"', str(HOST / "sketch_host.cpp"), "-o", str(out)]
    subprocess.run(cmd, check=True, timeout=300)
    return out


@pytest.fixture(scope="module")
def host_exe(tmp_path_factory):
    work = tmp_path_factory.mktemp("fwhost")
    return build_host(stage_sketch(SKETCH.parent, work / "sketch"), work / "sketch_host")


class Firmware:
    """The sketch as a process: send it UART bytes / UDP datagrams / clock time, read back what it did."""

    def __init__(self, exe: Path, *flags: str) -> None:
        self.p = subprocess.Popen([str(exe), *flags], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self.pins: dict[int, int] = {}
        self.last_state = ""
        self.boot = self._read()

    def _read(self) -> list[Event]:
        events = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("the firmware host exited")
            line = line.rstrip("\n")
            if line == "OK":
                return events
            if line.startswith("STATE "):
                self.last_state = line
                continue
            ms, kind, text = (line.split(" ", 2) + [""])[:3]
            ev = Event(int(ms), kind, text)
            if kind == "PIN":
                gpio, level = text.split()
                self.pins[int(gpio)] = int(level)
            events.append(ev)

    def send(self, line: str) -> list[Event]:
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        return self._read()

    def rx(self, data: bytes) -> list[Event]:
        return self.send("rx " + data.hex())

    def udp(self, data: bytes, ip: str = "192.168.4.5", port: int = 5555) -> list[Event]:
        return self.send(f"udp {ip} {port} {data.hex()}")

    def run(self, ms: int, step: int = 1) -> list[Event]:
        return self.send(f"run {ms} {step}")

    def wifi(self, up: bool) -> None:
        self.send("wifi up" if up else "wifi down")

    def state(self) -> dict:
        self.send("state")
        parts = self.last_state.split()
        out = {"ms": int(parts[1])}
        for kv in parts[2:]:
            k, _, v = kv.partition("=")
            out[k] = v
        return out

    @property
    def pressed(self) -> list[int]:
        return sorted(g for g, level in self.pins.items() if level)

    def close(self) -> None:
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
        except OSError:
            pass
        self.p.wait(timeout=10)


@pytest.fixture
def fw(host_exe):
    made = []

    def make(*flags: str, mode: str | None = None) -> Firmware:
        f = Firmware(host_exe, *flags)
        made.append(f)
        if mode:                                    # None: whatever the firmware boots with
            f.rx(f"mode {mode}\n".encode())
        return f
    yield make
    for f in made:
        f.close()


def frame(seq: int, *pairs, estop: bool = False) -> bytes:
    return encode_packet(command_packet(seq, np.array(pairs, dtype=float).ravel(), estop=estop))


def acks(events):
    return [json.loads(e.text) for e in events if e.kind == "TX" and e.text.startswith('{"ack"')]


def logs(events):
    return [e.text for e in events if e.kind == "TX" and not e.text.startswith("{")]


def udp_out(events):
    out = []
    for e in events:
        if e.kind == "UDP":
            dest, _, payload = e.text.partition(" ")
            out.append((dest, payload))
    return out


# ---------------------------------------------------------------- boot ----
def test_the_car_boots_and_listens_on_the_uart_with_no_wifi_at_all(fw):
    f = fw()
    boot = logs(f.boot)
    assert any("connecting to WiFi" in line for line in boot), "the boot marker other tools count"
    assert any("CAR INDEX 0" in line for line in boot) and any(line.startswith("firmware ") for line in boot)
    assert any(line.startswith("UART:") for line in boot)
    assert not any(line.startswith("IP:") for line in boot)
    assert f.state()["udp_bound"] == "0"
    # ... and it already takes UART commands: a probe is answered
    reply = [json.loads(e.text) for e in f.rx(PROBE_FRAME) if e.kind == "TX"]
    assert len(reply) == 1
    assert reply[0]["car"] == 0 and reply[0]["fw"].endswith("-uart")
    assert reply[0]["via"] == "uart" and reply[0]["rssi"] is None
    assert reply[0]["failsafe"] is True
    assert f.pressed == [], "buttons are released from the first instant"


def test_pins_are_driven_released_at_boot_before_anything_else(fw):
    f = fw()
    first = [e for e in f.boot if e.kind == "PIN"]
    assert {int(e.text.split()[0]): e.text.split()[1] for e in first} == {FWD: "0", BACK: "0", LEFT: "0", RIGHT: "0"}


# ------------------------------------------------------------ button modes ----
def test_a_fresh_board_boots_in_binary_mode(fw):
    """The V3 car's remote does not feather (measured 2026-10-02), so a board that never saved a mode
    must come up on/off: no console step needed after a fresh flash."""
    f = fw()
    assert any("mode BINARY" in line for line in logs(f.boot)), logs(f.boot)
    assert not any("throttle limit" in line for line in logs(f.boot)), "binary has no throttle cap to show"
    info = [json.loads(e.text) for e in f.rx(PROBE_FRAME) if e.kind == "TX"][0]
    assert info["mode"] == "binary"
    # on/off at the 1/3 cut: 0.2 presses nothing, 0.34 and above hold the button solid, never pulsed
    f.rx(frame(0, (0.2, 0.2)))
    f.run(300, 10)
    assert f.pressed == []
    f.rx(frame(1, (0.34, 0.34)))
    f.run(41)                                            # a mid-stream press waits for the next slot
    for _ in range(20):                                  # 200 ms = 5 slots, inside the 300 ms dead-man window
        assert f.pressed == [FWD, LEFT], "a binary press must be solid, not pulsed"
        f.run(10)


def test_modulated_mode_is_still_there_for_a_remote_that_feathers(fw):
    f = fw(mode="mod")
    assert any("mode MODULATED" in line for line in logs(f.boot) + logs(f.rx(b"mode\n")))
    f.rx(frame(0, (0.5, 0.0)))
    seen = set()
    for _ in range(40):
        f.run(10)
        seen.add(tuple(f.pressed))
    assert () in seen and (FWD,) in seen, "0.5 throttle at the default limit must pulse, not hold"


@pytest.mark.parametrize("mode", [None, "mod"], ids=["default-binary", "modulated"])
def test_the_deadman_releases_every_button_in_both_modes(fw, mode):
    f = fw(mode=mode)
    f.rx(frame(0, (1.0, 1.0)))
    for seq in range(1, 4):                              # drive both axes hard for 300 ms
        f.run(100)
        f.rx(frame(seq, (1.0, 1.0)))
    f.run(299)                                           # 299 ms after the last frame: still inside the window
    ev = f.run(2)                                        # +301: the dead-man trips
    assert f.pressed == [], "the dead-man left a button down"
    assert any("FAILSAFE" in line for line in logs(ev))
    later = f.run(600)                                   # and nothing presses again with no frames
    assert not [e for e in later if e.kind == "PIN" and e.text.endswith(" 1")], "a button pressed after the dead-man"


@pytest.mark.parametrize("mode", [None, "mod"], ids=["default-binary", "modulated"])
def test_estop_releases_at_once_in_both_modes(fw, mode):
    f = fw(mode=mode)
    f.rx(frame(0, (1.0, -1.0)))
    f.run(120)
    ev = f.rx(frame(1, (0.0, 0.0), estop=True))
    assert f.pressed == []
    assert all(e.ms == ev[0].ms for e in ev if e.kind == "PIN"), "an E-stop releases within the same instant"
    assert not [e for e in f.run(300) if e.kind == "PIN" and e.text.endswith(" 1")]


@pytest.mark.parametrize("mode", [None, "mod"], ids=["default-binary", "modulated"])
def test_opposite_buttons_are_never_down_together_whatever_the_stream(fw, mode):
    """The sketch end to end, UART frames in and GPIO events out: a random command stream
    (reversals, deadband, out-of-range, estops, no-slot packets) must never put FWD+BACK or
    LEFT+RIGHT down at the same time."""
    import random
    rng = random.Random(1234 if mode is None else 4321)
    f = fw(mode=mode)
    levels = [0.0, 0.03, 0.2, 0.34, 0.5, 0.97, 1.0, -0.03, -0.2, -0.34, -0.5, -1.0, 1.7, -1.7]
    seq, level, presses = 0, {FWD: 0, BACK: 0, LEFT: 0, RIGHT: 0}, 0
    for _ in range(400):
        r = rng.random()
        if r < 0.08:
            pkt = frame(seq, (0.0, 0.0), estop=True)
        elif r < 0.12:
            pkt = frame(seq)                                         # no slot for this car
        else:
            pkt = frame(seq, (rng.choice(levels), rng.choice(levels)))
        seq += 1
        events = f.rx(pkt) + f.run(rng.choice([1, 5, 20, 41, 60, 100]))
        for e in events:
            if e.kind != "PIN":
                continue
            gpio, lvl = map(int, e.text.split())
            level[gpio] = lvl
            presses += lvl
            assert not (level[FWD] and level[BACK]), f"FWD and BACK down together at {e.ms}"
            assert not (level[LEFT] and level[RIGHT]), f"LEFT and RIGHT down together at {e.ms}"
    assert presses > 100, "the stream never pressed anything: the invariant was not exercised"


# --------------------------------------------------- the UART command path ----
def test_a_drive_frame_presses_and_is_acked_on_the_uart(fw):
    f = fw()
    ev = f.rx(frame(0, (0.5, 0.0)))
    assert f.pressed == [FWD]
    assert acks(ev) == [{"ack": 0, "car": 0, "applied": True, "slot": True, "rssi": None}]
    ev = f.rx(frame(1, (-0.5, 0.5)))
    assert f.pressed == [], "a reversal releases the old press at once; the new one waits for its slot"
    f.run(100)                                          # the reversal gap, then the next slots
    assert f.pressed == [BACK, LEFT]


def test_the_deadman_trips_300ms_after_the_last_frame_to_the_millisecond(fw):
    f = fw()
    t0 = f.rx(frame(0, (0.5, 0.5)))[0].ms
    assert f.pressed == [FWD, LEFT]
    f.run(300)
    assert f.pressed == [FWD, LEFT], "300 ms of silence is still within the dead-man window"
    ev = f.run(1)
    assert f.pressed == []
    released = [e for e in ev if e.kind == "PIN"]
    assert released and all(e.ms == t0 + 301 for e in released), (t0, released)
    assert "FAILSAFE: command stream stalled -> buttons released" in logs(ev)


def test_frames_keep_it_alive_and_it_never_stalls_while_they_arrive(fw):
    f = fw()
    for seq in range(30):                               # 3 s of 10 Hz frames
        f.rx(frame(seq, (0.5, 0.0)))
        ev = f.run(100)
        assert not any("FAILSAFE" in line for line in logs(ev))
        assert f.pressed == [FWD]


def test_probes_noise_and_console_lines_do_not_feed_the_deadman(fw):
    f = fw()
    t0 = f.rx(frame(0, (0.5, 0.0)))[0].ms
    keepalives = [PROBE_FRAME, b"\x00\x01garbage\n", b'{"seq": 5, "t": 17\n', b"index\n", b"\n\n",
                  b'{"ack":1,"car":0}\n']
    clock = 0
    while clock < 250:
        for junk in keepalives:
            f.rx(junk)
        f.run(50)
        clock += 50
    assert f.pressed == [FWD]
    ev = f.run(60)                                      # t0 + 310
    assert f.pressed == [], "silence plus noise must still stop the car 300 ms after the last real frame"
    assert any("FAILSAFE" in line for line in logs(ev))


def test_a_damaged_frame_is_dropped_without_an_ack_or_a_timer_refresh(fw):
    f = fw()
    f.rx(frame(0, (0.5, 0.0)))
    f.run(200)
    good = frame(1, (0.5, 0.0))
    ev = f.rx(good[:25] + b"\n")                        # cut short
    assert acks(ev) == [] and logs(ev) == []
    ev = f.rx(good.replace(b"]]}", b"]}"))              # a lost bracket: not JSON any more
    assert acks(ev) == []
    ev = f.run(101)                                     # 301 ms since the only good frame
    assert f.pressed == []


def test_estop_releases_at_once_and_logs_once_per_stop(fw):
    f = fw()
    f.rx(frame(0, (0.5, 0.5)))
    ev = f.rx(frame(1, (0.0, 0.0), estop=True))
    assert f.pressed == [] and all(e.ms == ev[0].ms for e in ev if e.kind == "PIN")
    assert acks(ev) == [{"ack": 1, "car": 0, "applied": True, "slot": True, "rssi": None}]
    assert logs(ev) == ["#1 E-STOP -> buttons released"]
    ev = f.rx(frame(2, (0.0, 0.0), estop=True))         # a disarmed controller repeats it at 10 Hz
    assert logs(ev) == [], "the E-STOP line is once per stop, not per packet"
    assert acks(ev)[0]["applied"] is True


def test_a_packet_with_no_estop_field_is_an_estop(fw):
    f = fw()
    f.rx(frame(0, (0.5, 0.5)))
    ev = f.rx(b'{"seq": 1, "cmd": [[1.0, 1.0]]}\n')
    assert f.pressed == [], "a missing estop field reads as an e-stop"
    assert acks(ev)[0]["ack"] == 1


def test_stale_and_duplicate_seq_are_acked_but_not_applied(fw):
    f = fw()
    f.rx(frame(10, (0.5, 0.0)))
    ev = f.rx(frame(8, (-0.5, 0.0)))                    # reordered
    assert acks(ev)[0]["applied"] is False and f.pressed == [FWD]
    ev = f.rx(frame(10, (-0.5, 0.0)))                   # duplicate
    assert acks(ev)[0]["applied"] is False and f.pressed == [FWD]
    ev = f.rx(frame(11, (-0.5, 0.0)))
    assert acks(ev)[0]["applied"] is True
    f.run(100)
    assert f.pressed == [BACK]


def test_a_restarted_controller_resyncs_once_the_car_has_stopped(fw):
    f = fw()
    for seq in range(5):
        f.rx(frame(100 + seq, (0.5, 0.0)))
        f.run(50)
    f.run(400)                                          # the controller died: dead-man trips
    assert f.pressed == []
    ev = f.rx(frame(0, (0.5, 0.0)))                     # its counter starts over
    assert acks(ev)[0]["applied"] is True and f.pressed == [FWD]


def test_no_slot_for_this_car_releases_it_and_looks_like_no_link(fw):
    f = fw()
    f.rx(frame(0, (0.5, 0.5)))
    ev = f.rx(frame(1))                                 # cmd: [] -- the controller drives fewer cars
    assert f.pressed == []
    assert acks(ev) == [{"ack": 1, "car": 0, "applied": True, "slot": False, "rssi": None}]
    assert any("NO COMMAND for car index 0" in line for line in logs(ev))
    ev = f.rx(frame(2))
    assert f.pressed == []


def test_each_car_takes_its_own_slot_from_the_joint_packet(fw):
    f = fw()
    f.rx(b"index 1\n")
    ev = f.rx(frame(0, (0.0, 0.0), (0.5, -0.5)))        # slot 0 is someone else's
    assert acks(ev)[0] == {"ack": 0, "car": 1, "applied": True, "slot": True, "rssi": None}
    assert f.pressed == [FWD, RIGHT]
    ev = f.rx(frame(1, (0.5, 0.5)))                     # a 1-pair packet has nothing for index 1
    assert acks(ev)[0]["slot"] is False and f.pressed == []


def test_the_largest_fleet_frame_is_accepted(fw):
    f = fw()
    f.rx(b"index 7\n")
    pairs = [(-0.9999, -0.9999)] * 7 + [(0.5, 0.0)]
    pkt = frame(0, *pairs)
    assert 200 < len(pkt) < 512
    ev = f.rx(pkt)
    assert acks(ev)[0]["slot"] is True and f.pressed == [FWD]


def test_our_own_replies_are_never_taken_for_commands(fw):
    f = fw()
    f.rx(frame(0, (0.5, 0.0)))
    echoed = [b'{"ack":5,"car":0,"applied":true,"slot":true,"rssi":null}\n',
              b'{"car":0,"fw":"2026.10-buttons-uart","mac":"AA","failsafe":false,"via":"uart"}\n',
              b'{"cfg_ok":false,"reason":"car is being driven; stop it first"}\n']
    for line in echoed * 3:
        ev = f.rx(line)
        assert ev == [], f"a reply fed back in must produce nothing (no ack, no e-stop): {ev}"
    assert f.pressed == [FWD], "and must not have stopped the car either"


def test_the_console_and_the_frames_share_the_wire(fw):
    f = fw()
    assert logs(f.rx(b"index\n")) == ["car index = 0 (drives cmd[0])"]
    f.rx(frame(0, (0.5, 0.0)))
    ev = f.rx(b"index 3\n")
    assert logs(ev) == ["REFUSED: the car is being driven. Stop the controller first."]
    assert f.pressed == [FWD], "typing at a driving car neither stops it nor changes it"
    f.rx(frame(1, (0.5, 0.0), estop=True))
    ev = f.rx(b"index 3\n")
    assert any("car index set to 3" in line for line in logs(ev))
    assert acks(f.rx(frame(2, (0.1, 0.0), (0.1, 0.0), (0.1, 0.0), (0.5, 0.0))))[0]["car"] == 3


def test_overlong_lines_are_rejected_and_the_stream_recovers(fw):
    f = fw()
    assert logs(f.rx(b"x" * 60 + b"\n")) == ["REJECTED: line too long, ignored"]
    ev = f.rx(b"{" + b"a" * 600 + b"}\n")
    assert logs(ev) == ["REJECTED: frame longer than 512 bytes, ignored"]
    ev = f.rx(b"{" + b"a" * 600 + b"}\n")
    assert logs(ev) == [], "the oversized-frame notice is rate limited"
    assert acks(f.rx(frame(0, (0.5, 0.0))))[0]["applied"] is True


def test_the_rx_buffer_holds_two_frames_of_the_largest_fleet(fw):
    f = fw()
    f.send("rxburst " + (b" " * 600).hex())             # 600 bytes into a stalled loop
    assert f.state()["dropped"] == "88", "512 bytes of room: the sketch asks for a larger ring buffer"


def test_the_cfg_command_works_over_the_uart_and_is_refused_while_driving(fw):
    f = fw()
    ev = f.rx(b'{"cfg": {"index": 2}}\n')
    info = [json.loads(e.text) for e in ev if e.kind == "TX" and e.text.startswith('{"car"')]
    assert info and info[0]["car"] == 2 and info[0]["via"] == "uart"
    f.rx(frame(0, (0.1, 0.0), (0.1, 0.0), (0.5, 0.0)))
    ev = f.rx(b'{"cfg": {"index": 0}}\n')
    assert [json.loads(e.text) for e in ev if e.kind == "TX"] == [
        {"cfg_ok": False, "reason": "car is being driven; stop it first"}]


# -------------------------------------------------- UDP, and both together ----
def test_udp_replies_are_unchanged_bytes_sent_back_to_the_sender(fw):
    f = fw("--wifi-up")
    f.run(2)                                            # WiFi is up: the network starts
    assert f.state()["udp_bound"] == "1"
    ev = f.udp(frame(9, (0.5, 0.0)), ip="192.168.4.5", port=5555)
    assert udp_out(ev) == [("192.168.4.5:5555", '{"ack":9,"car":0,"applied":true,"slot":true,"rssi":-55}')]
    assert f.pressed == [FWD]
    assert [e for e in ev if e.kind == "TX"] == [], "a UDP packet is not answered on the UART"
    reply = json.loads(udp_out(f.udp(b'{"probe": 1}'))[0][1])
    assert reply["via"] == "udp" and reply["rssi"] == -55 and reply["car"] == 0


def test_udp_and_uart_share_one_deadman_and_each_feeds_it(fw):
    f = fw("--wifi-up")
    f.run(2)
    f.udp(frame(0, (0.5, 0.0)))
    f.run(200)
    f.rx(frame(1, (0.5, 0.0)))                          # the UART refreshes what UDP started
    f.run(250)
    assert f.pressed == [FWD]
    f.udp(frame(2, (0.5, 0.0)))                         # and UDP refreshes it back
    f.run(250)
    assert f.pressed == [FWD]
    ev = f.run(60)
    assert f.pressed == [] and any("FAILSAFE" in line for line in logs(ev))


def test_udp_is_dead_without_wifi_and_starts_when_it_connects(fw):
    f = fw()
    assert f.udp(frame(0, (0.5, 0.0))) == [], "no network yet: a datagram has nobody to read it"
    f.wifi(True)
    ev = f.run(1)
    assert "IP: 192.168.4.77  listening on UDP 8888" in logs(ev)
    assert f.state()["udp_bound"] == "1"


def test_wifi_retries_only_while_stopped_and_only_until_the_first_connection(fw):
    f = fw()
    assert f.state()["wifi_begin"] == "1"
    for seq in range(450):                              # 45 s of driving over the UART
        f.rx(frame(seq, (0.5, 0.0)))
        f.run(100)
    assert f.state()["wifi_begin"] == "1", "no radio work while a button is held down"
    f.run(310)                                          # stopped
    assert f.state()["wifi_begin"] == "2", "a retry is due, and runs the moment the car is stopped"
    f.wifi(True)
    f.run(5)
    f.wifi(False)
    f.run(90_000)
    assert f.state()["wifi_begin"] == "2", "once connected, reconnecting is the WiFi stack's job, as before"


def test_boot_with_wifi_present_is_the_same_boot(fw):
    f = fw("--wifi-up")
    assert any("connecting to WiFi" in line for line in logs(f.boot))
    assert "IP: 192.168.4.77  listening on UDP 8888" in logs(f.run(1))
