"""The dead zone (pc_controller/deadzone.py): a car that leaves the cameras' view, or passes
the tape, comes back by itself.

The closed-loop tests drive a simple model of the MD Elite car (bicycle steering, a speed that
follows the throttle with a lag, coast at zero throttle, a hard brake on reverse throttle)
under "cameras" that see only the taped square plus a narrow band. They check the behaviour,
not centimetres: the car returns into view and is handed back to the model. The real car is
not this model; these tests cannot tell you how far it rolls.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from controller_runtime.pose_types import VehiclePose  # noqa: E402
from pc_controller import deadzone as DZ  # noqa: E402
from pc_controller.pose_stream import parse_pose_frame, parse_pose_frame_meta  # noqa: E402

HALF = 1.2192                      # the 8 ft arena
SPEC = DZ.CarSpec(half_m=HALF, top_speed_mps=2.68, max_steer_rad=math.radians(30.0))
VIEW_BAND_M = 0.10                 # how far past the tape the test's cameras still see a car
VISION_HOLD_S = 0.25               # perception holds a pose this long before calling it lost


class SimCar:
    """A stand-in for the car. Throttle t in [-1, 1]: the speed heads for t x top with a 0.35 s
    lag; against the motion it brakes at 4.8 m/s^2; |t| < 0.05 coasts at 0.8 m/s^2."""

    def __init__(self, x, y, yaw, v=0.0, *, binary_drive=False) -> None:
        self.x, self.y, self.yaw, self.v = float(x), float(y), float(yaw), float(v)
        self.binary_drive = binary_drive

    def step(self, thr: float, steer: float, dt: float = 0.1) -> None:
        if self.binary_drive:                               # firmware `drive binary`: full on at 1/3
            thr = 0.0 if abs(thr) < 1.0 / 3.0 else math.copysign(1.0, thr)
        for _ in range(10):
            h = dt / 10.0
            if abs(thr) < 0.05:
                self.v -= math.copysign(min(abs(self.v), 0.8 * h), self.v)
            elif thr * self.v < 0.0:
                self.v -= math.copysign(min(abs(self.v), 4.8 * h), self.v)
            else:
                self.v += (thr * SPEC.top_speed_mps - self.v) * h / 0.35
            self.yaw += self.v * math.tan(steer * SPEC.max_steer_rad) / SPEC.wheelbase_m * h
            self.x += self.v * math.cos(self.yaw) * h
            self.y += self.v * math.sin(self.yaw) * h

    @property
    def in_view(self) -> bool:
        return max(abs(self.x), abs(self.y)) <= HALF + VIEW_BAND_M

    @property
    def outside_m(self) -> float:
        return math.hypot(max(abs(self.x) - HALF, 0.0), max(abs(self.y) - HALF, 0.0))


def run(car: SimCar, model_cmd, *, seconds=8.0, dead: DZ.DeadZone | None = None, delay_ticks=0):
    """Drive `car` with `model_cmd(t)` through a DeadZone. Returns the per-tick log."""
    dead = dead or DZ.DeadZone(1, spec=SPEC)
    evader = VehiclePose(0.0, 0.0, 0.0, 0.0)
    last_seen_pose, last_seen_t, log, pending = None, 0.0, [], []
    msgs: list[str] = []
    for k in range(int(seconds / 0.1)):
        t = 0.1 * k
        if car.in_view:
            last_seen_pose, last_seen_t = VehiclePose(car.x, car.y, car.yaw, t), t
        age = t - last_seen_t
        held = age <= VISION_HOLD_S                         # perception holds the pose, then marks it LOST
        meta = {"t": t, "seen": [held, True], "lost_s": [0.0 if held else age, 0.0]}
        seen_l, lost_l = DZ.fresh_seen([last_seen_pose], evader, meta)
        seen = seen_l[0]
        action = np.asarray(model_cmd(t), dtype=np.float64)
        out = dead.apply(action, [last_seen_pose], evader, seen_l, lost_l, msgs.append)
        pending.append((float(out[0]), float(out[1])))
        thr, steer = pending.pop(0) if len(pending) > delay_ticks else (0.0, 0.0)
        car.step(thr, steer)
        log.append({"t": t, "x": car.x, "y": car.y, "v": car.v, "seen": seen, "in_view": car.in_view,
                    "recovering": dead.recovering[0], "blind": dead.blind[0], "gave_up": dead.gave_up[0],
                    "cmd": (thr, steer), "outside": car.outside_m})
    return log, msgs


def full_ahead(_t):
    return [1.0, 0.0]


def cruise(throttle):
    return lambda _t: [throttle, 0.0]


@pytest.mark.parametrize("throttle", [0.3, 0.6, 1.0])
@pytest.mark.parametrize("delay_ticks", [0, 1, 2])
def test_a_car_that_drives_straight_out_of_view_backs_up_its_own_track_and_is_seen_again(throttle, delay_ticks):
    # The blind recovery on its own (no edge brake): a model that holds its throttle straight
    # through the right-hand tape, from the middle of the arena (up to 2.7 m/s at the tape).
    car = SimCar(-0.6, 0.2, 0.0)
    log, msgs = run(car, cruise(throttle), seconds=9.0, delay_ticks=delay_ticks,
                    dead=DZ.DeadZone(1, spec=SPEC, guard=False))
    assert any(not r["in_view"] for r in log), "the test never left the cameras' view"
    assert any(r["blind"] for r in log), "the blind recovery never started"
    lost_at = next(i for i, r in enumerate(log) if not r["in_view"])
    back_at = next((i for i, r in enumerate(log) if i > lost_at and r["in_view"]), None)
    out = max(r["outside"] for r in log[lost_at:back_at])
    assert back_at is not None, f"never came back into view (furthest out {out:.2f} m)"
    assert not any(r["gave_up"] for r in log[lost_at:back_at]), "gave up instead of coming back"
    assert log[back_at]["t"] - log[lost_at]["t"] < DZ.GIVE_UP_S
    assert any("OUT OF VIEW" in m for m in msgs)


@pytest.mark.parametrize("throttle", [0.3, 0.6, 1.0])
@pytest.mark.parametrize("delay_ticks", [0, 1])
@pytest.mark.parametrize("yaw_deg", [0, 30, -70])
def test_with_the_edge_brake_a_car_with_room_to_stop_never_leaves_the_cameras_view(throttle, delay_ticks, yaw_deg):
    car = SimCar(-0.6, 0.2, math.radians(yaw_deg))
    log, _ = run(car, cruise(throttle), seconds=6.0, delay_ticks=delay_ticks)
    out = max(r["outside"] for r in log)
    assert out <= VIEW_BAND_M + 0.05, f"{out:.2f} m past the tape"
    assert sum(1 for r in log if not r["in_view"]) <= 3
    assert not any(r["gave_up"] for r in log)


def test_it_is_handed_back_to_the_model_only_once_it_is_well_inside_the_tape():
    car = SimCar(HALF - 0.5, 0.0, 0.0, v=1.5)
    released = {"t": None}

    def model(t):                                   # full ahead, then a parked model once it has been out
        return [1.0, 0.0] if released["t"] is None else [0.0, 0.0]

    dead = DZ.DeadZone(1, spec=SPEC)
    log, msgs = [], []
    evader = VehiclePose(0.0, 0.0, 0.0, 0.0)
    last, last_t = None, 0.0
    was_recovering = False
    for k in range(120):
        t = 0.1 * k
        if car.in_view:
            last, last_t = VehiclePose(car.x, car.y, car.yaw, t), t
        seen_l, lost_l = DZ.fresh_seen([last], evader, {"t": t, "seen": [t - last_t <= VISION_HOLD_S, True],
                                                        "lost_s": [max(t - last_t, 0.0), 0.0]})
        seen = seen_l[0]
        out = dead.apply(np.asarray(model(t)), [last], evader, seen_l, lost_l, msgs.append)
        if was_recovering and not dead.recovering[0] and released["t"] is None:
            released["t"] = t
            assert seen and HALF - max(abs(car.x), abs(car.y)) >= DZ.REJOIN_INSIDE_M - 0.02
        was_recovering = dead.recovering[0]
        car.step(float(out[0]), float(out[1]))
        log.append(car.outside_m)
    assert released["t"] is not None, "the model never got the car back"
    assert any("BACK" in m for m in msgs)


@pytest.mark.parametrize("yaw_deg", [0, 35, 80, 100, 180, -120])
def test_a_seen_car_past_the_tape_drives_back_in_from_any_heading(yaw_deg):
    # Standing 0.08 m past the right-hand tape (still in view), the model asking for nothing.
    car = SimCar(HALF + 0.08, 0.3, math.radians(yaw_deg))
    log, _ = run(car, lambda t: [0.0, 0.0], seconds=10.0)
    assert log[0]["recovering"]
    assert not log[-1]["recovering"], f"still recovering after 10 s at ({car.x:.2f}, {car.y:.2f})"
    assert HALF - max(abs(car.x), abs(car.y)) >= DZ.REJOIN_INSIDE_M - 0.05
    assert max(r["outside"] for r in log) < 0.45, "swung far outside while turning back in"
    assert max(abs(r["v"]) for r in log) <= DZ.DRIVE_IN_MPS + 0.25


def test_the_model_keeps_its_car_while_it_is_inside_and_seen():
    car = SimCar(0.0, 0.0, 0.3, v=0.5)
    dead = DZ.DeadZone(1, spec=SPEC)
    action = np.array([0.4, -0.2])
    out = dead.apply(action, [VehiclePose(car.x, car.y, car.yaw, 0.0)], VehiclePose(0.5, 0.5, 0.0, 0.0), [True, True])
    assert out is action and not dead.recovering[0]


def test_a_car_lost_while_standing_still_is_not_driven_anywhere():
    # Parked beside the evader, its tag hidden: nothing was moving, so there is nothing to replay.
    dead = DZ.DeadZone(1, spec=SPEC)
    pose, ev = VehiclePose(0.2, 0.2, 0.0, 0.0), VehiclePose(0.5, 0.2, 0.0, 0.0)
    for k in range(10):
        dead.apply(np.zeros(2), [VehiclePose(0.2, 0.2, 0.0, 0.1 * k)], ev, [True, True])
    sent = [dead.apply(np.array([0.8, 0.0]), [pose], ev, [False, True], [0.3 + 0.1 * k, 0.0]) for k in range(60)]
    assert all(abs(s[0]) < 1e-9 for s in sent), "a blind, parked car was driven"
    assert dead.gave_up[0], "unseen for GIVE_UP_S it must be flagged lost (the dashboard ends the run)"


def test_blind_for_the_give_up_time_gives_up_and_stops():
    car = SimCar(HALF - 0.3, 0.0, 0.0, v=2.0)
    dead = DZ.DeadZone(1, spec=SPEC)
    ev = VehiclePose(0.0, 0.0, 0.0, 0.0)
    for k in range(5):                                   # seen, driving out
        dead.apply(np.array([1.0, 0.0]), [VehiclePose(HALF - 0.3 + 0.02 * k, 0.0, 0.0, 0.1 * k)], ev, [True, True])
    last = VehiclePose(HALF - 0.2, 0.0, 0.0, 0.5)
    outs = [dead.apply(np.array([1.0, 0.0]), [last], ev, [False, True], [0.3 + 0.1 * k, 0.0]) for k in range(60)]
    assert dead.gave_up[0]
    assert tuple(outs[-1]) == (0.0, 0.0)
    assert any(o[0] < 0 for o in outs[:10]), "expected a brake / reverse while blind"
    assert all(o[0] <= 0.0 for o in outs), "a blind car must never be driven further the way it was going"


def test_the_evader_out_of_view_stands_every_car_still_but_a_recovering_car_keeps_coming_back():
    dead = DZ.DeadZone(2, spec=SPEC)
    inside, outside = VehiclePose(0.0, 0.0, 0.0, 0.0), VehiclePose(HALF + 0.08, 0.0, math.pi, 0.0)
    ev = VehiclePose(0.5, 0.5, 0.0, 0.0)
    out = dead.apply(np.array([0.7, 0.1, 0.7, 0.1]), [inside, outside], ev, [True, True, False], [0, 0, 1.2])
    assert dead.evader_hold
    assert tuple(out[:2]) == (0.0, 0.0)                  # the car inside waits
    assert dead.recovering[1] and out[2] != 0.7          # the car past the tape is still being recovered
    out = dead.apply(np.array([0.7, 0.1, 0.7, 0.1]), [inside, outside], ev, [True, True, True], [0, 0, 0])
    assert not dead.evader_hold and out[0] == 0.7


def test_binary_drive_gets_pulses_and_still_comes_back():
    dead = DZ.DeadZone(1, spec=SPEC, binary_drive=True, guard=False)
    car = SimCar(HALF - 1.0, 0.2, 0.0, v=1.5, binary_drive=True)
    log, _ = run(car, cruise(0.0), seconds=6.0, dead=dead)
    thr = {round(abs(r["cmd"][0]), 2) for r in log if r["recovering"] and not r["seen"]}
    assert thr <= {0.0, DZ.BINARY_PULSE}, f"blind throttles other than a pulse or nothing were sent: {thr}"
    lost_at = next(i for i, r in enumerate(log) if not r["in_view"])
    assert any(r["in_view"] for r in log[lost_at + 1:]), "never came back into view"


def test_min_throttle_raises_only_recovery_commands():
    dead = DZ.DeadZone(1, spec=SPEC, min_throttle=0.3)
    ev = VehiclePose(0.0, 0.0, 0.0, 0.0)
    out = dead.apply(np.array([0.1, 0.0]), [VehiclePose(0.0, 0.0, 0.0, 0.0)], ev, [True, True])
    assert out[0] == 0.1                                 # the model's own small throttle is untouched
    for k in range(30):                                  # past the tape, nose in: it creeps in on the floor
        out = dead.apply(np.zeros(2), [VehiclePose(HALF + 0.1, 0.0, math.pi, 0.1 * k)], ev, [True, True])
    assert dead.recovering[0] and abs(out[0]) >= 0.3


def test_a_model_without_the_cars_numbers_only_stops_an_unseen_car():
    assert DZ.CarSpec.from_manifest({"num_pursuers": 1}) is None
    dead = DZ.DeadZone(1, spec=None)
    assert dead.mode == "stop-only"
    out = dead.apply(np.array([0.9, 0.3]), [VehiclePose(0.0, 0.0, 0.0, 0.0)], VehiclePose(1, 1, 0, 0), [False, True])
    assert tuple(out) == (0.0, 0.0)


def test_car_numbers_come_from_the_md_elite_manifest():
    manifest = json.loads((ROOT / "models" / "rc16_n1_md_elite" / "policy.onnx.manifest.json").read_text())
    spec = DZ.CarSpec.from_manifest(manifest)
    assert spec is not None
    assert abs(spec.half_m - 1.2192) < 1e-3 and abs(spec.max_steer_rad - math.radians(30)) < 1e-6
    assert abs(spec.top_speed_mps - 2.68) < 0.01


# --------------------------------------------------------------------------- the wire format
def lost_frame():
    return json.dumps({"t": 10.0,
                       "pursuers": [{"x": None, "y": None, "heading": None, "lost_s": 0.62,
                                     "last": {"x": 1.18, "y": 0.4, "heading": 0.1}, "t": 9.38}],
                       "evader": {"x": 0.0, "y": 0.1, "heading": 0.2, "t": 9.9}})


def test_a_lost_vehicle_parses_to_its_last_seen_pose_and_is_flagged():
    pursuers, evader, meta = parse_pose_frame_meta(lost_frame(), expected_pursuers=1)
    assert meta["seen"] == [False, True] and meta["lost_s"] == [0.62, 0.0]
    assert (pursuers[0].x, pursuers[0].y, pursuers[0].timestamp_s) == (1.18, 0.4, 9.38)
    assert evader.timestamp_s == 9.9                     # a held pose keeps its own capture time
    assert meta["t"] == 10.0


def test_a_caller_that_needs_every_pose_refuses_a_frame_with_a_lost_vehicle():
    with pytest.raises(ValueError):
        parse_pose_frame(lost_frame(), expected_pursuers=1)


# ------------------------------------------------- models whose runtime recovers its own cars
def test_a_runtime_that_recovers_its_own_cars_is_only_supervised():
    status = {"rec": [False, False, False], "gave": [False, False, False]}
    dead = DZ.DeadZone(3, spec=SPEC, runtime_handles_unseen=True,
                       runtime_status=lambda: (status["rec"], status["gave"]))
    assert dead.mode == "runtime"
    poses = [VehiclePose(0.0, 0.3 * i, 0.0, 0.0) for i in range(3)]
    ev = VehiclePose(0.5, 0.5, 0.0, 0.0)
    action = np.array([0.5, 0.1, 0.5, 0.1, 0.5, 0.1])
    assert dead.apply(action, poses, ev, [True] * 4) is action          # all seen: untouched
    # car 1 unseen, the runtime is NOT recovering it: its own command for 1 s, then zero
    out = dead.apply(action, poses, ev, [True, False, True, True], [0, 0.3, 0, 0])
    assert out[2] == 0.5
    out = dead.apply(action, poses, ev, [True, False, True, True], [0, 1.1, 0, 0])
    assert tuple(out[2:4]) == (0.0, 0.0) and out[0] == 0.5
    # the runtime recovers it: its command goes through
    status["rec"] = [False, True, False]
    out = dead.apply(action, poses, ev, [True, False, True, True], [0, 1.2, 0, 0])
    assert out[2] == 0.5 and dead.recovering[1] and dead.blind[1]
    status["gave"] = [False, True, False]
    out = dead.apply(action, poses, ev, [True, False, True, True], [0, 3.2, 0, 0])
    assert tuple(out[2:4]) == (0.0, 0.0) and dead.gave_up[1]


def test_the_three_car_runtime_takes_seen_and_backs_an_unseen_car_out_of_the_dead_zone():
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from pc_controller.loops import make_loop

    loop = make_loop(ROOT / "models" / "rc16_hive_h2")
    assert loop.handles_unseen
    ev = lambda t: VehiclePose(-0.6, -0.6, 0.0, t)                       # noqa: E731
    others = lambda t: [VehiclePose(-0.3, 0.6, 0.0, t), VehiclePose(0.0, -0.9, 0.0, t)]   # noqa: E731
    # car 0 runs at 1.5 m/s toward the right-hand tape and is seen up to 5 cm short of it
    t, x = 0.0, HALF - 0.8
    while x < HALF - 0.05:
        loop.tick(pursuer_poses=[VehiclePose(x, 0.0, 0.0, t)] + others(t), evader_pose=ev(t), seen=[True] * 4)
        t, x = t + 0.1, x + 0.15
    last = VehiclePose(x - 0.15, 0.0, 0.0, t - 0.1)
    thr, rec, gave = [], [], []
    for _ in range(45):                                                  # then the cameras lose it
        a = loop.tick(pursuer_poses=[last] + others(t), evader_pose=ev(t), seen=[False, True, True, True])
        r, g = loop.recovery_status()
        thr.append(float(a[0])); rec.append(r[0]); gave.append(g[0])
        t += 0.1
    assert True in rec[:3], "dead-reckoned past the tape within 0.3 s: the runtime must be recovering it"
    k = rec.index(True)
    assert thr[k] < 0 and all(x <= 1e-6 for x in thr[k:]), "a blind car must brake, then back up; never forward"
    assert any(x < -0.05 for x in thr[k + 5:k + 25]), "after the brake it replays its track backwards"
    assert gave[-1] and abs(thr[-1]) < 1e-6, "blind for 3 s: stopped and flagged"
    assert not any(loop.recovery_status()[0][1:]), "the seen cars are not in recovery"


# ------------------------------------------------------------- through the real control loop
class FakeLink:
    """What run_realtime needs of a car link, recording what would be sent."""

    def __init__(self) -> None:
        self.targets, self.car_stats = [("fake", 0)], {0: {}}
        self.sent: list = []          # ("cmd", [throttle, steer]) or ("stop",)

    def label(self, i): return "fake"
    def selectables(self): return []
    def poll_acks(self): pass
    def send_commands(self, action): self.sent.append(("cmd", [float(v) for v in action]))
    def send_stop(self): self.sent.append(("stop",))


def test_the_control_loop_keeps_running_and_recovers_when_perception_reports_a_lost_car():
    pytest.importorskip("onnxruntime")
    import socket
    import threading
    import time

    from pc_controller.loops import make_loop
    from pc_controller.realtime import run_realtime

    def free_port():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    pose_port, tel_port = free_port(), free_port()
    tel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tel.bind(("127.0.0.1", tel_port))
    tel.settimeout(0.05)
    link, logs, halt = FakeLink(), [], threading.Event()
    loop = make_loop(ROOT / "models" / "rc16_n1_md_elite")
    th = threading.Thread(target=run_realtime, args=(loop, link), daemon=True,
                          kwargs=dict(pose_port=pose_port, telemetry=f"127.0.0.1:{tel_port}", log=logs.append,
                                      stop=halt))
    th.start()
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    time.sleep(0.3)

    def stream(seconds, car, evader, t_car=None):
        """30 fps for `seconds`. car / evader: fn(elapsed) -> pose entry of the frame."""
        start = time.perf_counter()
        n = 0
        while time.perf_counter() - start < seconds:
            now = time.perf_counter()
            el = now - start
            tx.sendto(json.dumps({"t": now, "pursuers": [car(el, now)], "evader": evader(el, now),
                                  "seq": n, "lat": 0.005}).encode(), ("127.0.0.1", pose_port))
            n += 1
            time.sleep(max(0.0, start + n / 30.0 - time.perf_counter()))

    def last_telemetry():
        out = None
        while True:
            try:
                out = json.loads(tel.recvfrom(65535)[0])
            except (socket.timeout, OSError):
                return out

    ev_seen = lambda el, now: {"x": -0.5, "y": 0.5, "heading": 0.0}     # noqa: E731
    try:
        # 1. the car drives at 1.2 m/s toward the right-hand tape, seen
        x0 = HALF - 1.0
        stream(0.7, lambda el, now: {"x": x0 + 1.2 * el, "y": 0.0, "heading": 0.0}, ev_seen)
        x_last, t_lost = x0 + 1.2 * 0.7, time.perf_counter()
        assert any(s[0] == "cmd" for s in link.sent), "the loop never drove"
        n_before = len(link.sent)
        # 2. perception loses it: frames keep coming, the car marked lost at its last seen pose
        stream(1.2, lambda el, now: {"x": None, "y": None, "heading": None, "lost_s": round(0.25 + el, 3),
                                     "last": {"x": x_last, "y": 0.0, "heading": 0.0}, "t": t_lost}, ev_seen)
        blind = link.sent[n_before:]
        assert not any("STALLED" in m for m in logs), "a lost car must not stall the pose feed"
        assert all(s[0] == "cmd" for s in blind), "E-stop packets while the car should be recovering"
        assert any("OUT OF VIEW" in m for m in logs)
        thr = [s[1][0] for s in blind]
        assert min(thr) < -0.1 and max(thr[2:]) <= 1e-6, f"expected a brake then a slow reverse, got {thr}"
        t = last_telemetry()
        assert t["state"] == "RUNNING" and t["seen"] == [False, True]
        assert t["deadzone"]["mode"] == "controller" and t["deadzone"]["blind"] == [True]
        # 3. seen again well inside: the model has the car back
        stream(0.5, lambda el, now: {"x": 0.3, "y": 0.0, "heading": 0.0}, ev_seen)
        assert any("BACK" in m for m in logs)
        assert last_telemetry()["deadzone"]["recovering"] == [False]
        # 4. the evader is lost: the car stands still, armed, link alive
        n_before = len(link.sent)
        stream(0.5, lambda el, now: {"x": 0.3, "y": 0.0, "heading": 0.0},
               lambda el, now: {"x": None, "y": None, "heading": None, "lost_s": round(1.0 + el, 3),
                                "last": {"x": -0.5, "y": 0.5, "heading": 0.0}, "t": now - 1.0 - el})
        held = link.sent[n_before + 1:]
        assert held and all(s == ("cmd", [0.0, 0.0]) for s in held), held[:5]
        assert any("EVADER OUT OF VIEW" in m for m in logs)
    finally:
        halt.set()
        th.join(timeout=3.0)
        tx.close()
        tel.close()


def test_an_older_flat_model_runs_as_before_and_only_stops_a_lost_car():
    pytest.importorskip("onnxruntime")
    import socket
    import threading
    import time

    from pc_controller.loops import make_loop
    from pc_controller.realtime import run_realtime

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    pose_port = probe.getsockname()[1]
    probe.close()
    link, logs, halt = FakeLink(), [], threading.Event()
    loop = make_loop(ROOT / "models" / "n1_catch")          # no car numbers in its manifest
    th = threading.Thread(target=run_realtime, args=(loop, link), daemon=True,
                          kwargs=dict(pose_port=pose_port, log=logs.append, stop=halt))
    th.start()
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    time.sleep(0.3)

    def stream(seconds, car):
        start, n = time.perf_counter(), 0
        while time.perf_counter() - start < seconds:
            now = time.perf_counter()
            tx.sendto(json.dumps({"t": now, "pursuers": [car(now)], "evader": {"x": 0.5, "y": 0.2, "heading": 1.0},
                                  "seq": n, "lat": 0.005}).encode(), ("127.0.0.1", pose_port))
            n += 1
            time.sleep(max(0.0, start + n / 30.0 - time.perf_counter()))

    try:
        stream(0.5, lambda now: {"x": -0.5, "y": 0.0, "heading": 0.0})      # the plain frames of before
        drove = [s for s in link.sent if s[0] == "cmd"]
        assert drove and any(abs(v) > 0.05 for s in drove for v in s[1]), "the model's commands must go out"
        assert any("STOPPED until it is seen again" in m for m in logs)
        n_before = len(link.sent)
        t_lost = time.perf_counter()
        stream(0.5, lambda now: {"x": None, "y": None, "heading": None, "lost_s": 0.3,
                                 "last": {"x": -0.5, "y": 0.0, "heading": 0.0}, "t": t_lost})
        after = link.sent[n_before + 1:]
        assert after and all(s == ("cmd", [0.0, 0.0]) for s in after), after[:4]
        assert not any("STALLED" in m for m in logs)
    finally:
        halt.set()
        th.join(timeout=3.0)
        tx.close()


def test_the_tape_is_taken_from_the_cameras_when_the_taped_square_is_not_the_models():
    dead = DZ.DeadZone(1, spec=SPEC)
    assert not dead.set_arena_half(HALF + 0.01)             # the same square: nothing changes
    assert dead.set_arena_half(0.9144)                      # a 6 ft square taped, an 8 ft model
    ev = VehiclePose(0.0, 0.0, 0.0, 0.0)
    dead.apply(np.zeros(2), [VehiclePose(1.0, 0.0, 0.0, 0.0)], ev, [True, True])
    assert dead.recovering[0], "1.0 m is past a 6 ft square's tape (0.914 m)"
    rt = DZ.DeadZone(3, spec=SPEC, runtime_handles_unseen=True, runtime_status=lambda: ([False] * 3, [False] * 3))
    assert not rt.set_arena_half(0.9144)                    # a runtime with its own recovery keeps its arena
