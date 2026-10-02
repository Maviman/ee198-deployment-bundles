"""The two ends of the UART agree on framing: the bytes the controller writes
(pc_controller) go through the firmware's own framer (esp32/esp32_receiver/
uart_frame.h, compiled here with the host C++ compiler), and come out as exactly
the frames that were sent.

    cd portable_n1_controller && python -m pytest tests/test_firmware_framing.py -q

Skipped where no C++ compiler is installed (the firmware itself is built with
PlatformIO; this only needs `c++ -std=c++11`).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pc_controller.esp_link import command_packet, encode_packet  # noqa: E402
from pc_controller.serial_link import PROBE_FRAME  # noqa: E402

FW = ROOT / "esp32"
HEADER = FW / "esp32_receiver" / "uart_frame.h"
SKETCH = FW / "esp32_receiver" / "esp32_receiver.ino"


@pytest.fixture(scope="module")
def framer(tmp_path_factory):
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler on this machine")
    exe = tmp_path_factory.mktemp("fw") / "test_uart_frame"
    subprocess.run([cxx, "-std=c++11", "-Wall", "-Wextra", "-Werror", f"-I{HEADER.parent}",
                    str(FW / "test_host" / "test_uart_frame.cpp"), "-o", str(exe)],
                   check=True, timeout=120)
    return exe


def through_firmware(framer: Path, data: bytes) -> list[tuple[str, bytes]]:
    out = subprocess.run([str(framer), "--feed"], input=data, capture_output=True, check=True, timeout=30)
    events = []
    for line in out.stdout.split(b"\n"):
        if line:
            kind, _, text = line.partition(b" ")
            events.append((kind.decode(), text))
    return events


def frames_for(n_cars: int, seed: int = 0) -> list[bytes]:
    rng = np.random.default_rng(seed)
    drive = command_packet(7, rng.uniform(-1, 1, 2 * n_cars))
    stop = command_packet(8, np.zeros(2 * n_cars), estop=True)
    return [encode_packet(drive), encode_packet(stop)]


def test_the_firmwares_own_framer_checks_pass(framer):
    out = subprocess.run([str(framer)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0 and "all checks passed" in out.stdout, out.stdout


@pytest.mark.parametrize("n_cars", [1, 3, 8])
def test_every_controller_frame_comes_out_of_the_firmware_framer_whole(framer, n_cars):
    sent = frames_for(n_cars) + [PROBE_FRAME.lstrip(b"\n")]
    # the controller's stream: a newline first to resync (after open / a failed
    # write), then the frames back to back
    stream = b"\n" + b"".join(sent) * 3
    events = through_firmware(framer, stream)
    assert [k for k, _ in events] == ["F"] * (3 * len(sent))
    for (kind, text), want in zip(events, sent * 3):
        assert text + b"\n" == want
        json.loads(text)                            # and it is the JSON the controller built


def test_the_largest_fleet_fits_the_firmware_frame_buffer():
    max_line = int(re.search(r"MAX_LINE\s*=\s*(\d+)", HEADER.read_text()).group(1))
    max_cars = int(re.search(r"MAX_CARS\s*=\s*(\d+)", SKETCH.read_text()).group(1))
    worst = encode_packet(command_packet(10 ** 9, np.full(2 * max_cars, -0.9999)))
    assert len(worst) < max_line // 2, (
        f"a {max_cars}-car command packet is {len(worst)} bytes; the frame buffer is {max_line}. "
        "Keep at least 2x headroom (the sketch's RX buffer is sized for two such frames).")


def test_a_console_line_between_frames_disturbs_neither(framer):
    a, b = frames_for(1)
    events = through_firmware(framer, a + b"index 2\r\n" + b + b"mode binary\n")
    assert [k for k, _ in events] == ["F", "C", "F", "C"]
    assert events[0][1] + b"\n" == a and events[2][1] + b"\n" == b
    assert events[1][1] == b"index 2" and events[3][1] == b"mode binary"


def test_line_noise_costs_only_its_own_line(framer):
    a, b = frames_for(3)
    noise = bytes(range(1, 256)) * 6                 # 1.5 kB of everything but newline-ish bytes
    noise = noise.replace(b"\n", b"x").replace(b"\r", b"x")
    events = through_firmware(framer, a + noise + b"\n" + b)
    kinds = [k for k, _ in events]
    assert kinds[0] == "F" and kinds[-1] == "F" and events[-1][1] + b"\n" == b
    assert kinds[1:-1] in (["!C"], ["!F"]), kinds    # the noise is discarded, as one line


def test_a_frame_cut_short_is_one_bad_line_not_two_lost_frames(framer):
    a, b = frames_for(1)
    events = through_firmware(framer, a[:20] + b"\n" + b)     # a byte-loss mid-frame, then a good frame
    assert events[0] == ("F", a[:20])             # handed on as a line: the JSON parser will refuse it
    assert events[1][0] == "F" and events[1][1] + b"\n" == b
    with pytest.raises(ValueError):
        json.loads(events[0][1])
