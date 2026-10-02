"""The loop factory (pc_controller.loops): flat models still get PortableLoop, and a
role-commander manifest gets HiveLoop, which feeds the vendored runtime kinematic
state and keeps PortableLoop's interface.

The real hive runtime is not in the repo yet, so HiveLoop runs here against a
stand-in that follows the agreed interface:
    CommanderRuntime(model_dir); reset(roles=None); step(pursuers (N,4), evader (4,), dt) -> (N,2); roles
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from controller_runtime.pose_types import VehiclePose  # noqa: E402
from pc_controller import loops  # noqa: E402
from pc_controller.esp_link import EspLink  # noqa: E402
from pc_controller.portable_loop import PortableLoop  # noqa: E402
from pc_controller.realtime import run_realtime  # noqa: E402


class FakeCommander:
    """Records what it is fed; commands echo each car's signed speed as throttle."""
    instances: list = []

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self.calls: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.resets = 0
        self.roles = ["CLOSE", "PRESSURE", "CUTOFF"]
        FakeCommander.instances.append(self)

    def reset(self, roles=None) -> None:
        self.resets += 1

    def step(self, pursuers, evader, dt=0.1):
        pursuers, evader = np.asarray(pursuers), np.asarray(evader)
        assert pursuers.shape == (3, 4) and evader.shape == (4,)
        self.calls.append((pursuers.copy(), evader.copy(), dt))
        return np.column_stack([np.clip(pursuers[:, 3], -1, 1), np.full(3, 0.25)])


@pytest.fixture
def hive_model(tmp_path, monkeypatch):
    mod = types.ModuleType("fake_hive_runtime")
    mod.CommanderRuntime = FakeCommander
    monkeypatch.setitem(sys.modules, "fake_hive_runtime", mod)
    monkeypatch.setitem(loops.RUNTIMES, "fake_v0", "fake_hive_runtime")
    (tmp_path / "policy.onnx.manifest.json").write_text(json.dumps(
        {"runtime": "fake_v0", "num_pursuers": 3, "input_dim": 124, "action_dim": 6, "capture_radius_m": 0.6}))
    (tmp_path / "metrics.json").write_text("{}")
    FakeCommander.instances.clear()
    return tmp_path


def test_flat_models_still_get_portable_loop():
    loop = loops.make_loop(ROOT / "models" / "n1_catch")
    assert isinstance(loop, PortableLoop)
    assert loop.capture_radius > 0


def test_car_camera_models_are_refused(tmp_path):
    """The cars carry no cameras: a model that expects per-car camera input must not run."""
    manifest = json.loads((ROOT / "models" / "n1_catch" / "policy.onnx.manifest.json").read_text())
    manifest["use_car_cameras"] = True
    (tmp_path / "policy.onnx.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="car cameras"):
        loops.make_loop(tmp_path)


def test_unknown_runtime_is_refused(tmp_path):
    (tmp_path / "policy.onnx.manifest.json").write_text(json.dumps({"runtime": "nope_v9", "num_pursuers": 3}))
    with pytest.raises(SystemExit, match="not supported"):
        loops.make_loop(tmp_path)


def test_hive_loop_feeds_signed_speed_and_dt(hive_model):
    loop = loops.make_loop(hive_model)
    assert isinstance(loop, loops.HiveLoop)
    assert (loop.policy.num_pursuers, loop.policy.action_dim, loop.capture_radius) == (3, 6, 0.6)
    rt = FakeCommander.instances[-1]

    def frame(t, xs):   # car 0 forward, car 1 reversing (heading +x, x falling), car 2 still
        return dict(pursuer_poses=[VehiclePose(x, 0.0, 0.0, t) for x in xs],
                    evader_pose=VehiclePose(1.0, 1.0, 0.0, t))
    a0 = loop.tick(**frame(10.0, [0.0, 0.0, 0.0]))
    a1 = loop.tick(**frame(10.1, [0.05, -0.05, 0.0]))
    assert a0.shape == (6,) and a1.shape == (6,)
    speeds = rt.calls[-1][0][:, 3]
    assert speeds == pytest.approx([0.5, -0.5, 0.0])          # signed: reversing reads negative
    assert rt.calls[0][2] == pytest.approx(0.1)               # first tick: the training step
    assert rt.calls[1][2] == pytest.approx(0.1)               # then the measured interval
    assert list(a1[0::2]) == pytest.approx([0.5, -0.5, 0.0])  # flattened [t0, s0, t1, s1, t2, s2]
    assert loop.roles == ["CLOSE", "PRESSURE", "CUTOFF"]
    loop.reset()
    assert rt.resets == 1
    loop.tick(**frame(11.0, [0.3, 0.3, 0.3]))
    assert rt.calls[-1][0][:, 3] == pytest.approx([0.0, 0.0, 0.0])   # no speed carried over a reset


def test_hive_loop_refuses_the_wrong_car_count(hive_model):
    loop = loops.make_loop(hive_model)
    with pytest.raises(ValueError, match="expected 3 pursuer poses"):
        loop.tick(pursuer_poses=[VehiclePose(0, 0, 0, 1.0)], evader_pose=VehiclePose(1, 1, 0, 1.0))


def test_realtime_drives_three_cars_through_a_hive_loop(hive_model):
    """The real-time loop, real UDP, three cars: one command packet per tick
    carrying a [throttle, steer] pair for each CAR_INDEX."""
    loop = loops.make_loop(hive_model)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(0.5)
    link = EspLink([("127.0.0.1", rx.getsockname()[1])] * 3)
    port_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port_s.bind(("127.0.0.1", 0))
    pose_port = port_s.getsockname()[1]
    port_s.close()
    threading.Thread(target=lambda: run_realtime(loop, link, pose_port=pose_port, log=lambda m: None),
                     daemon=True).start()
    time.sleep(0.3)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = time.perf_counter()
    for k in range(30):          # 1 s at 30 fps
        t = time.perf_counter()
        tx.sendto(json.dumps({"t": t, "seq": k + 1, "lat": 0.005,
                              "pursuers": [{"x": -0.5 + 0.2 * i, "y": -0.5, "heading": 0.0} for i in range(3)],
                              "evader": {"x": 0.5, "y": 0.5, "heading": 1.0}}).encode(), ("127.0.0.1", pose_port))
        time.sleep(max(0.0, start + (k + 1) / 30 - time.perf_counter()))
    packets = []
    while True:
        try:
            packets.append(json.loads(rx.recvfrom(4096)[0]))
        except (socket.timeout, OSError):
            break
    driving = [p for p in packets if not p["estop"]]
    assert driving, "no command packets"
    assert all(len(p["cmd"]) == 3 and all(len(c) == 2 for c in p["cmd"]) for p in driving)
