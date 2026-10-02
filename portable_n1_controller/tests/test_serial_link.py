"""Tests of the UART command transport (pc_controller/serial_link.py and the
serial side of pc_controller/esp_link.py), with no hardware:

    cd portable_n1_controller && python -m pytest tests/test_serial_link.py -q

The framing and parsing tests are pure Python. The rest open real
pseudo-terminals with pyserial, the way the controller opens /dev/ttyTHS1, with
tools/mock_esp.py's car on the far end (tests/serial_board.py). They are
skipped, not failed, where pyserial is not installed: it is only needed for
``--esp serial:...``.

The behavioural tests of the real-time loop over serial (E-stop on silence,
disarm, stale poses) are tests/test_realtime.py, which runs every one of its
tests over both transports.
"""

from __future__ import annotations

import importlib.util
import json
import os
import select
import socket
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from pc_controller import serial_link  # noqa: E402
from pc_controller.esp_link import (  # noqa: E402
    EspLink, command_packet, encode_packet, parse_targets, target_label, target_uri,
)
from pc_controller.serial_link import (  # noqa: E402
    DEFAULT_BAUD, LineAssembler, SerialLinkError, SerialTarget, SerialUnavailable,
    parse_line, parse_serial_spec,
)
from serial_board import PtyBoard  # noqa: E402

needs_pyserial = pytest.mark.skipif(importlib.util.find_spec("serial") is None,
                                    reason="pyserial is only needed for the serial transport")


@pytest.fixture
def boards():
    made = []

    def make(*a, **kw):
        b = PtyBoard(*a, **kw)
        made.append(b)
        return b
    yield make
    for b in made:
        b.close()


@pytest.fixture
def links():
    made = []

    def make(*a, **kw):
        link = EspLink(*a, **kw)
        made.append(link)
        return link
    yield make
    for link in made:
        try:
            link.close()
        except OSError:
            pass


# ---------------------------------------------------------------- targets ----
def test_serial_targets_parse_with_default_and_explicit_baud():
    assert parse_serial_spec("serial:/dev/ttyUSB0") == SerialTarget("/dev/ttyUSB0", DEFAULT_BAUD)
    assert parse_serial_spec("serial:/dev/ttyTHS1@230400") == SerialTarget("/dev/ttyTHS1", 230400)
    assert parse_serial_spec("  SERIAL:/dev/ttyUSB0  ").device == "/dev/ttyUSB0"


def test_by_path_names_with_colons_survive_parsing():
    path = "/dev/serial/by-path/platform-3610000.usb-usb-0:2.1:1.0-port0"
    assert parse_serial_spec(f"serial:{path}").device == path
    assert parse_serial_spec(f"serial:{path}@921600") == SerialTarget(path, 921600)


@pytest.mark.parametrize("bad, why", [
    ("serial:", "missing device"),
    ("serial:/dev/x@abc", "bad baud"),
    ("serial:/dev/x@0", "out of range"),
    ("serial:/dev/x@99999999", "out of range"),
    ("serial:auto", "not supported"),
])
def test_bad_serial_targets_are_refused(bad, why):
    with pytest.raises(ValueError, match=why):
        parse_serial_spec(bad)


def test_fleet_lists_keep_order_and_mix_transports():
    t = parse_targets("serial:/dev/ttyTHS1,192.168.4.11:8889,serial:/dev/ttyTHS2@230400,10.0.0.5")
    assert t == [SerialTarget("/dev/ttyTHS1"), ("192.168.4.11", 8889),
                 SerialTarget("/dev/ttyTHS2", 230400), ("10.0.0.5", 8888)]
    assert [isinstance(x, SerialTarget) for x in t] == [True, False, True, False]


def test_udp_target_parsing_is_unchanged():
    assert parse_targets("127.0.0.1") == [("127.0.0.1", 8888)]
    assert parse_targets("a:1, b:2 ,") == [("a", 1), ("b", 2)]
    with pytest.raises(ValueError, match="duplicate ESP address"):
        parse_targets("a:1,a:1")
    with pytest.raises(ValueError, match="bad port"):
        parse_targets("a:x")


def test_two_slots_on_one_serial_port_are_refused():
    with pytest.raises(ValueError, match="duplicate serial device"):
        parse_targets("serial:/dev/ttyUSB0,serial:/dev/ttyUSB0@9600")


def test_a_bare_device_path_gets_a_hint_not_a_silent_udp_host():
    with pytest.raises(ValueError, match="serial:/dev/ttyUSB0"):
        parse_targets("/dev/ttyUSB0")


def test_labels_for_logs_and_the_dashboard():
    assert target_label(("10.0.0.5", 8888)) == "10.0.0.5:8888"            # unchanged for UDP
    assert target_uri(("10.0.0.5", 8888)) == "udp://10.0.0.5:8888"
    assert target_label(SerialTarget("/dev/ttyTHS1")) == "serial:/dev/ttyTHS1"
    assert target_label(SerialTarget("/dev/ttyTHS1", 230400)) == "serial:/dev/ttyTHS1@230400"
    assert target_uri(SerialTarget("/dev/ttyTHS1")) == "serial:/dev/ttyTHS1"


# ---------------------------------------------------------------- framing ----
def test_a_frame_is_the_command_packet_plus_its_newline():
    pkt = command_packet(7, np.array([0.43, -0.1]), estop=False)
    raw = encode_packet(pkt)
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    assert json.loads(raw) == pkt
    assert pkt["cmd"] == [[0.43, -0.1]]


def test_command_packets_clip_and_pair_up_like_before():
    pkt = command_packet(0, np.array([2.0, -3.0, 0.5, 0.25]))
    assert pkt["cmd"] == [[1.0, -1.0], [0.5, 0.25]]
    with pytest.raises(ValueError, match="throttle/steer pairs"):
        command_packet(0, np.array([0.1, 0.2, 0.3]))


def test_assembler_splits_lines_however_the_bytes_arrive():
    stream = b'{"ack":1}\r\nFAILSAFE: stalled\n{"ack":2}\n'
    whole = LineAssembler().feed(stream)
    assert whole == [b'{"ack":1}', b"FAILSAFE: stalled", b'{"ack":2}']
    dribble = LineAssembler()
    got = []
    for i in range(len(stream)):                  # one byte at a time
        got += dribble.feed(stream[i:i + 1])
    assert got == whole


def test_assembler_holds_a_partial_line_until_its_newline():
    a = LineAssembler()
    assert a.feed(b'{"ack":4,"car"') == []
    assert a.feed(b':0}') == []
    assert a.feed(b"\n") == [b'{"ack":4,"car":0}']


def test_assembler_drops_blank_lines_and_accepts_bare_cr():
    assert LineAssembler().feed(b"\n\r\n  \nhello\rworld\n") == [b"  ", b"hello", b"world"]


def test_assembler_resyncs_after_an_overlong_line():
    a = LineAssembler(max_line=32)
    out = a.feed(b"x" * 100 + b"\n" + b'{"ack":9}\n')       # noise with no newline for 100 bytes
    assert out == [b'{"ack":9}']
    assert a.overflows == 1
    # ... and a long line split across feeds is dropped as a whole, not just its head
    b = LineAssembler(max_line=32)
    assert b.feed(b"y" * 40) == []
    assert b.feed(b"z" * 40 + b"\nok\n") == [b"ok"]


def test_lines_are_classified_ack_info_or_log():
    assert parse_line(b'{"ack":3,"car":1,"applied":true,"slot":true,"rssi":null}')[0] == "ack"
    kind, info = parse_line(b'{"car":0,"fw":"x","mac":"m","failsafe":true}')
    assert kind == "info" and info["car"] == 0
    assert parse_line(b"FAILSAFE: command stream stalled -> buttons released") == (
        "log", "FAILSAFE: command stream stalled -> buttons released")
    assert parse_line(b'{"ack":3,"car"')[0] == "log"                # cut off: noise, not a reply
    assert parse_line(b"[1, 2]")[0] == "log"                         # JSON, but not an object
    assert parse_line(b"{garbage}")[0] == "log"
    assert parse_line(b"caf\xff\xfe\n")[0] == "log"                  # not UTF-8: still a log line


# -------------------------------------------------------- the lazy import ----
def test_a_udp_only_controller_never_imports_pyserial():
    code = ("import sys; sys.path.insert(0, '.'); import numpy as np\n"
            "from pc_controller.esp_link import EspLink, parse_targets\n"
            "parse_targets('serial:/dev/ttyTHS1,10.0.0.5')\n"       # parsing alone does not import it
            "link = EspLink('127.0.0.1:9')\n"
            "link.send_commands(np.zeros(2)); link.poll_acks(); link.wait_ready(); link.close()\n"
            "assert 'serial' not in sys.modules, 'pyserial was imported by a UDP-only run'\n")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=30)


def test_missing_pyserial_is_a_clear_start_up_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", None)               # makes `import serial` fail
    with pytest.raises(SerialUnavailable, match="pip install pyserial"):
        EspLink("serial:/dev/null")


def test_the_wrong_package_named_serial_is_diagnosed(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", types.ModuleType("serial"))   # no Serial class
    with pytest.raises(SerialUnavailable, match="not pyserial"):
        EspLink("serial:/dev/null")


# ------------------------------------------------- over a real pty (pyserial) ----
@needs_pyserial
def test_a_missing_device_says_where_the_ports_are():
    with pytest.raises(SerialLinkError, match="does not exist.*ttyTHS1"):
        EspLink("serial:/dev/definitely-not-a-port")


@needs_pyserial
def test_a_port_already_open_elsewhere_is_refused(boards, links):
    board = boards()
    links(f"serial:{board.path}")
    with pytest.raises(SerialLinkError, match="already open in another process"):
        EspLink(f"serial:{board.path}")


@needs_pyserial
def test_the_frame_on_the_wire_is_the_udp_datagram_byte_for_byte(boards, links, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1720300000.123456)    # one clock for both links
    actions = np.array([0.43, -0.1, 9.0, 0.0])                      # two cars; 9.0 clips to 1.0

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    links([("127.0.0.1", rx.getsockname()[1])]).send_commands(actions)
    datagram = rx.recv(4096)

    board = boards()
    ser = links(f"serial:{board.path}")
    ser.wait_ready()
    ser.send_commands(actions)
    assert board.wait_for(lambda: board.raw_frames())
    assert board.raw_frames()[-1] + b"\n" == datagram
    assert b'"estop": false' in datagram and datagram.endswith(b"]]}\n")


@needs_pyserial
def test_probe_verify_then_drive_and_the_round_trip_is_measured(boards, links):
    board = boards()
    link = links(f"serial:{board.path}")
    link.wait_ready()
    assert link.link_up(0) is True and link.transport(0) == "serial"
    sent = link.send_commands(np.array([0.4, -0.2]))
    deadline = time.monotonic() + 1.0
    while not link.car_stats[0]["acks"] and time.monotonic() < deadline:
        select.select(link.selectables(), [], [], 0.1)       # the real loop wakes on the port
        link.poll_acks()
    st = link.car_stats[0]
    assert st["acks"] == 1 and st["applied"] is True and st["slot"] is True
    assert st["rssi"] is None, "a UART car must not report a WiFi signal"
    assert st["rtt_ms"] is not None and st["rtt_ms"] < 60.0
    assert board.packets[-1][1]["seq"] == sent["seq"]


@needs_pyserial
def test_no_command_is_written_before_the_board_is_verified(boards, links):
    board = boards(ignore_probes=3)                          # a board still booting after the open
    link = links(f"serial:{board.path}")
    link.send_commands(np.array([0.4, -0.2]))                # before wait_ready: must not reach the car
    assert not board.wait_for(lambda: bool(board.packets), timeout_s=0.3), \
        "a command reached a board that had not proved its CAR_INDEX"
    assert link.datagrams_sent == 0
    link.wait_ready()                                        # probes repeat until the board answers
    assert len(board.probes) >= 4
    link.send_commands(np.array([0.4, -0.2]))
    assert board.wait_for(lambda: len(board.packets) == 1)


@needs_pyserial
def test_a_board_with_the_wrong_index_is_refused_and_never_commanded(boards, links):
    board = boards(index=1)
    with pytest.raises(SerialLinkError, match=r"answered as car index 1.*position 0"):
        link = EspLink(f"serial:{board.path}")
        try:
            link.wait_ready()
        finally:
            link.close()
    assert not board.packets


@needs_pyserial
def test_swapped_ports_are_caught_before_anything_moves(boards):
    car0, car1 = boards(index=0), boards(index=1)
    link = EspLink(f"serial:{car1.path},serial:{car0.path}")          # listed in the wrong order
    try:
        with pytest.raises(SerialLinkError, match="position"):
            link.wait_ready()
    finally:
        link.close()
    assert not car0.packets and not car1.packets


@needs_pyserial
def test_a_silent_wire_is_diagnosed_as_wiring(boards, monkeypatch):
    monkeypatch.setattr(serial_link, "VERIFY_TIMEOUT_S", 0.5)
    board = boards(ignore_probes=10 ** 9)
    link = EspLink(f"serial:{board.path}")
    try:
        with pytest.raises(SerialLinkError, match=r"no reply from .*nothing was received at all"):
            link.wait_ready(timeout_s=3.0)
    finally:
        link.close()


@needs_pyserial
def test_a_board_that_talks_but_ignores_probes_is_diagnosed_as_old_firmware(boards, monkeypatch):
    monkeypatch.setattr(serial_link, "VERIFY_TIMEOUT_S", 0.5)
    board = boards(ignore_probes=10 ** 9)
    link = EspLink(f"serial:{board.path}")
    try:
        board.say("connecting to WiFi")                      # its boot banner proves the RX wire works
        with pytest.raises(SerialLinkError, match="firmware is probably too old"):
            link.wait_ready(timeout_s=3.0)
    finally:
        link.close()


@needs_pyserial
def test_a_fleet_gets_one_joint_packet_and_each_board_answers_for_itself(boards, links):
    car0, car1 = boards(index=0), boards(index=1)
    link = links(f"serial:{car0.path},serial:{car1.path}")
    link.wait_ready()
    sent = link.send_commands(np.array([0.1, 0.2, 0.3, 0.4]))
    assert car0.wait_for(lambda: car0.raw_frames()) and car1.wait_for(lambda: car1.raw_frames())
    assert car0.raw_frames() == car1.raw_frames(), "every car must get the SAME joint packet"
    assert len(sent["cmd"]) == 2
    acks = link.wait_acks(sent["seq"], timeout_s=1.0)
    assert sorted(acks) == [0, 1]
    assert all(a["slot"] and a["applied"] for a in acks.values())
    assert link.seq == 1, "one seq counter for the whole fleet"


@needs_pyserial
def test_a_board_whose_index_changes_mid_run_is_dropped_on_its_next_ack(boards, links):
    board = boards(index=0)
    link = links(f"serial:{board.path}")
    link.wait_ready()
    link.send_commands(np.array([0.4, 0.0]))
    assert board.wait_for(lambda: board.packets) and link.link_up(0) is True
    board.car.index = 1                                      # re-indexed behind the controller's back
    sent = len(board.packets)
    link.send_commands(np.array([0.4, 0.0]))                 # still delivered: nobody knew yet ...
    assert board.wait_for(lambda: len(board.packets) == sent + 1)
    deadline = time.monotonic() + 1.0
    while link.link_up(0) and time.monotonic() < deadline:   # ... but its ACK says "car 1"
        link.poll_acks()
        time.sleep(0.01)
    assert link.link_up(0) is False
    assert "answered as car index 1" in link.serial_endpoints[0].last_error
    link.send_commands(np.array([0.4, 0.0]))
    assert not board.wait_for(lambda: len(board.packets) > sent + 1, timeout_s=0.3), \
        "a command was written to a board that is no longer car 0"


@needs_pyserial
def test_udp_and_serial_cars_share_one_fleet(boards, links):
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    board = boards(index=1)
    link = links([("127.0.0.1", rx.getsockname()[1]), SerialTarget(board.path)])
    link.wait_ready()
    assert [link.transport(i) for i in (0, 1)] == ["udp", "serial"]
    sent = link.send_commands(np.array([0.1, 0.2, 0.3, 0.4]))
    datagram, addr = rx.recvfrom(4096)
    rx.sendto(json.dumps({"ack": sent["seq"], "car": 0, "applied": True, "slot": True,
                          "rssi": -50}).encode(), addr)
    assert board.wait_for(lambda: board.raw_frames())
    assert board.raw_frames()[-1] + b"\n" == datagram
    acks = link.wait_acks(sent["seq"], timeout_s=1.0)
    assert sorted(acks) == [0, 1] and acks[0]["rssi"] == -50 and acks[1]["rssi"] is None


@needs_pyserial
def test_the_boards_deadman_still_stops_the_car_and_its_log_line_reaches_us(boards):
    board = boards()
    heard = []
    link = EspLink(f"serial:{board.path}", log=heard.append)
    try:
        link.wait_ready()
        for _ in range(4):
            link.send_commands(np.array([0.4, 0.0]))
            time.sleep(0.05)
        assert board.car.armed
        deadline = time.monotonic() + 1.5                   # 0.3 s of silence trips the dead-man
        while board.car.armed and time.monotonic() < deadline:
            link.poll_acks()
            time.sleep(0.01)
        assert not board.car.armed, "the board kept driving with no frames arriving"
        while not any("FAILSAFE" in m for m in heard) and time.monotonic() < deadline + 1.0:
            link.poll_acks()
            time.sleep(0.01)
        assert any("FAILSAFE: command stream stalled" in m for m in heard), heard
        assert any("FAILSAFE" in line for line in link.serial_endpoints[0].log_tail)
    finally:
        link.close()


@needs_pyserial
def test_close_ends_with_an_estop_frame_for_every_slot(boards):
    car0, car1 = boards(index=0), boards(index=1)
    link = EspLink(f"serial:{car0.path},serial:{car1.path}")
    link.wait_ready()
    link.send_commands(np.array([0.5, 0.5, -0.5, -0.5]))
    link.close()
    for board in (car0, car1):
        assert board.wait_for(lambda b=board: b.packets and b.packets[-1][1]["estop"])
        last = board.packets[-1][1]
        assert last["estop"] is True and last["cmd"] == [[0.0, 0.0], [0.0, 0.0]]


@needs_pyserial
def test_garbage_and_log_lines_between_acks_are_tolerated(boards, links):
    board = boards()
    link = links(f"serial:{board.path}")
    link.wait_ready()
    sent = link.send_commands(np.array([0.4, 0.0]))
    board.say("\x00\x01 line noise {\"ack\": ")                 # a cut-off reply, then ...
    board.say("#3 NO COMMAND for car index 0 -> released")      # ... a firmware log line
    ep = link.serial_endpoints[0]
    deadline = time.monotonic() + 2.0                            # the ack and the two lines come from
    while (link.car_stats[0]["acks"] < 1 or ep.log_lines < 2) and time.monotonic() < deadline:  # different
        link.poll_acks()                                         # threads: wait for all three, in any order
        time.sleep(0.01)
    assert link.car_stats[0]["acks"] == 1                        # the real ack still counted, once
    assert ep.log_lines == 2
    assert sent["seq"] == 0


@needs_pyserial
def test_a_pulled_cable_is_survivable_and_a_swapped_board_is_refused(boards, tmp_path, monkeypatch):
    monkeypatch.setattr(serial_link, "RETRY_S", 0.05)
    monkeypatch.setattr(serial_link, "VERIFY_TIMEOUT_S", 0.5)
    name = tmp_path / "car0"

    def point_at(board):
        tmp = tmp_path / "next"
        os.symlink(board.path, tmp)
        os.replace(tmp, name)

    first = boards()
    point_at(first)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    events = []
    link = EspLink([SerialTarget(str(name)), ("127.0.0.1", rx.getsockname()[1])], log=events.append)
    try:
        link.wait_ready()
        link.send_commands(np.array([0.1, 0.1, 0.2, 0.2]))
        rx.recvfrom(4096)

        first.close()                                       # the cable is pulled
        deadline = time.monotonic() + 2.0
        while link.link_up(0) and time.monotonic() < deadline:
            link.poll_acks()
            time.sleep(0.01)
        assert link.link_up(0) is False
        n = link.send_commands(np.array([0.1, 0.1, 0.2, 0.2]))   # must not raise ...
        assert rx.recvfrom(4096)[0].endswith(b"\n"), "... and the other car still gets its tick"
        # tick 0 reached both cars (2 frames); tick 1 reached only the UDP car
        assert n["seq"] == 1 and link.datagrams_sent == 3
        assert any("LINK DOWN" in m for m in events)

        wrong = boards(index=1)                             # a different board shows up on the path
        point_at(wrong)
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            link.poll_acks()
            link.send_commands(np.array([0.1, 0.1, 0.2, 0.2]))
            time.sleep(0.02)
        assert link.link_up(0) is False and not wrong.packets, \
            "a command reached a board that is not car 0"
        assert "position 0" in (link.serial_endpoints[0].last_error or "")

        right = boards(index=0)                             # the right board comes back
        point_at(right)
        deadline = time.monotonic() + 3.0
        while not link.link_up(0) and time.monotonic() < deadline:
            link.poll_acks()
            time.sleep(0.01)
        assert link.link_up(0) is True, link.serial_endpoints[0].last_error
        link.send_commands(np.array([0.1, 0.1, 0.2, 0.2]))
        assert right.wait_for(lambda: right.packets)
        assert any("link up" in m for m in events)
    finally:
        link.close()


# ------------------------------------------------------------ the tools ----
@needs_pyserial
def test_link_test_tool_runs_over_serial(boards):
    board = boards()
    out = subprocess.run([sys.executable, "tools/link_test.py", "--esp", f"serial:{board.path}",
                          "--count", "15", "--quiet"],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "verdict   : PASS" in out.stdout
    assert "serial:" in out.stdout and "rssi dBm" not in out.stdout


@needs_pyserial
def test_mock_esp_serial_mode_serves_a_stable_link_for_the_rehearsal(tmp_path):
    link = tmp_path / "mock-esp0"
    mock = subprocess.Popen([sys.executable, "-u", "tools/mock_esp.py", "--serial", "--link", str(link)],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 5.0
        while not link.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert link.is_symlink(), "mock_esp --serial --link made no symlink"
        ctl = EspLink(f"serial:{link}")
        try:
            ctl.wait_ready()
            ctl.send_commands(np.array([0.3, -0.3]))
            assert ctl.wait_ack(0, timeout_s=1.0)["car"] == 0
        finally:
            ctl.close()
        time.sleep(0.3)             # the mock is another process: let it read the final E-stop
    finally:
        mock.terminate()
        out = mock.communicate(timeout=5)[0]
    assert "thr=+0.30 str=-0.30" in out and "[E-STOP -> neutral]" in out, out
    assert not link.exists(), "the mock left its symlink behind"


def test_run_controller_refuses_auto_mixed_with_other_targets():
    out = subprocess.run([sys.executable, "run_controller.py", "--esp", "auto,serial:/dev/ttyTHS1"],
                         cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert out.returncode != 0 and "cannot be combined" in out.stderr


@needs_pyserial
def test_run_controller_says_what_is_wrong_with_a_missing_port():
    out = subprocess.run([sys.executable, "run_controller.py", "--esp", "serial:/dev/no-such-port",
                          "--episodes", "1"],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode != 0
    assert "serial link:" in out.stderr and "does not exist" in out.stderr
