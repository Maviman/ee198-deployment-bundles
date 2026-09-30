"""Closed-loop simulated arena: a mock fleet of ESP32 cars + the training
env's car physics, feeding a synthetic overhead camera.

    controller --UDP cmd--> sim_world (mock ESPs + physics) --world state-->
    run_vision.py --synthetic --world-port (renders real tags) --poses--> controller

The loop closes through the PRODUCTION pipeline: real tag images, real
decode + detection, real pose frames, the real control loop and the real
command packets. The only stand-ins are the camera and the cars. `arena sim`
starts all of it, so the macros, the dashboard and the arm/E-stop flow can be
rehearsed with no hardware.

Each mock car follows the firmware's rules: seq ordering with resync,
estop, a 300 ms failsafe, cmd[CAR_INDEX] slot selection, the same ACK JSON,
and a side-effect-free reply to discovery probes. Physics is
SinglePursuerEnv's bicycle model at the 6 ft test-arena scale
(configs/test_arena_6ft.json), stepped at 100 Hz with a small actuator delay.

    python tools/sim_world.py --cars 1 --world-port 9880
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from single_pursuer.env import SinglePursuerEnv  # noqa: E402

FAILSAFE_S = 0.3
SEQ_RESYNC_GAP = 50


class MockCar:
    def __init__(self, index: int, port: int, bind: str) -> None:
        self.index = index
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind, port))
        self.sock.setblocking(False)
        self.port = port
        self.last_seq = -1
        self.last_packet = 0.0
        self.failsafe = True
        self.cmd = (0.0, 0.0)
        self.pending: deque = deque()     # (apply_at, cmd) for actuator delay
        self.started = time.monotonic()
        self.packets = 0

    def poll(self, now: float, delay_s: float) -> None:
        while True:
            try:
                raw, addr = self.sock.recvfrom(2048)
            except (BlockingIOError, ConnectionResetError, OSError):
                break
            try:
                pkt = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if "probe" in pkt:
                self._send(addr, {"car": self.index, "fw": "sim", "mac": f"sim:{self.index}",
                                  "failsafe": self.failsafe, "up_s": int(now - self.started),
                                  "last_seq": self.last_seq, "rssi": -40})
                continue
            self.packets += 1
            seq = int(pkt.get("seq", -1))
            estop = bool(pkt.get("estop", True))
            cmds = pkt.get("cmd") or []
            slot_ok = len(cmds) > self.index and len(cmds[self.index]) >= 2
            applied = False
            if seq > self.last_seq or self.failsafe or (self.last_seq - seq) > SEQ_RESYNC_GAP:
                self.last_seq = seq
                applied = True
                if estop:
                    self.pending.clear()
                    self.cmd, self.failsafe = (0.0, 0.0), True
                    self.last_packet = now
                elif not slot_ok:
                    self.pending.clear()
                    self.cmd, self.failsafe = (0.0, 0.0), True
                else:
                    thr, steer = float(cmds[self.index][0]), float(cmds[self.index][1])
                    self.pending.append((now + delay_s, (thr, steer)))
                    self.failsafe = False
                    self.last_packet = now
            self._send(addr, {"ack": seq, "car": self.index, "applied": applied,
                              "slot": slot_ok, "rssi": -40})
        while self.pending and self.pending[0][0] <= now:
            self.cmd = self.pending.popleft()[1]
        if not self.failsafe and now - self.last_packet > FAILSAFE_S:
            self.pending.clear()
            self.cmd, self.failsafe = (0.0, 0.0), True
            print(f"car {self.index}: FAILSAFE (command stream stalled) -> neutral", flush=True)

    def _send(self, addr, payload: dict) -> None:
        try:
            self.sock.sendto(json.dumps(payload).encode("utf-8"), addr)
        except OSError:
            pass


def load_marker_ids(path: Path, n: int) -> tuple[list[int], int]:
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        pursuers = [int(i) for i in data["pursuer_ids"]][:n]
        return pursuers, int(data["evader_id"])
    except Exception:  # noqa: BLE001 - fall back to the documented defaults
        return list(range(1, n + 1)), 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cars", type=int, default=1, help="pursuer count (must match the model)")
    ap.add_argument("--base-port", type=int, default=8888, help="car i listens on base-port + i")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--world-port", type=int, default=9880, help="where run_vision --synthetic listens")
    ap.add_argument("--config", default=str(ROOT / "configs" / "test_arena_6ft.json"))
    ap.add_argument("--marker-map", default=str(ROOT.parent / "portable_orin_perception" / "config" / "marker_map.yaml"))
    ap.add_argument("--evader-speed", type=float, default=0.35, help="fraction of evader max speed")
    ap.add_argument("--actuator-delay", type=float, default=0.05, help="command -> wheels (s)")
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = SinglePursuerEnv(config=args.config, num_pursuers=args.cars, stage="fleeing_evader",
                           evader_speed_frac=args.evader_speed, seed=args.seed)
    env.reset(seed=args.seed)
    env.dt = 1.0 / args.hz
    cars = [MockCar(i, args.base_port + i, args.bind) for i in range(args.cars)]
    pursuer_ids, evader_id = load_marker_ids(Path(args.marker_map), args.cars)
    world = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    world_addr = ("127.0.0.1", args.world_port)
    print(f"sim world: {args.cars} car(s) on UDP {[c.port for c in cars]}, arena "
          f"{2 * env.half_width:.2f} x {2 * env.half_height:.2f} m, world -> :{args.world_port}", flush=True)

    period = 1.0 / args.hz
    next_t = time.monotonic()
    episode, captures, capture_until = 1, 0, None
    last_report = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
                continue
            next_t += period
            if next_t < now:
                next_t = now + period
            for car in cars:
                car.poll(now, args.actuator_delay)
            if capture_until is None:
                for car, p in zip(cars, env.pursuers):
                    env._drive(p, car.cmd[0], car.cmd[1])
                    env._keep_inside(p)
                env._move_evader()
                env._resolve_evader_blocking()
                env._keep_inside(env.evader)
                dists = [math.hypot(p["x"] - env.evader["x"], p["y"] - env.evader["y"]) for p in env.pursuers]
                if min(dists) <= env.capture_radius:
                    captures += 1
                    capture_until = now + 2.0
                    print(f"episode {episode}: CAPTURE (min distance {min(dists):.2f} m); "
                          "resetting in 2 s", flush=True)
            elif now >= capture_until:
                env.reset()
                env.dt = period
                episode += 1
                capture_until = None
            vehicles = {str(pid): [p["x"], p["y"], p["heading"]] for pid, p in zip(pursuer_ids, env.pursuers)}
            vehicles[str(evader_id)] = [env.evader["x"], env.evader["y"], env.evader["heading"]]
            try:
                world.sendto(json.dumps({"vehicles": vehicles}).encode("utf-8"), world_addr)
            except OSError:
                pass
            if now - last_report > 10.0:
                last_report = now
                state = ", ".join(f"car{c.index} {'FAILSAFE' if c.failsafe else f'thr {c.cmd[0]:+.2f}'} "
                                  f"({c.packets} pkts)" for c in cars)
                print(f"episode {episode}, {captures} capture(s); {state}", flush=True)
    except KeyboardInterrupt:
        print(f"\n{captures} capture(s) in {episode} episode(s)")


if __name__ == "__main__":
    main()
