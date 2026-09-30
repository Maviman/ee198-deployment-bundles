"""Main entry point: run the N=1 policy on this PC and stream throttle/steer
commands to an ESP32 over UDP.

Two pose sources:

  --source sim (default)  The vendored 2D simulator plays the world (pursuer +
      evader physics), the controller sees only POSES from it -- exactly what a
      real overhead camera would provide -- and every commanded action is both
      applied to the sim and transmitted to the ESP. Zero hardware needed; with
      an ESP (or tools/mock_esp.py) attached you watch real packets flow while
      the sim episode plays out. This is the SIL harness pattern from the main
      repo with a radio bolted on.

  --source udp            Real deployment mode: consume perception pose frames
      over UDP (see pc_controller/pose_stream.py for the frame format), command
      the real car. Dead-man failsafe: if the pose feed stalls >0.3 s an E-stop
      packet is sent and the loop re-arms only when frames resume.

Examples (any Python 3.10+ with `pip install -r requirements.txt`):

  python run_controller.py --model models/n1_catch                      # sim, print only
  python run_controller.py --model models/n1_catch --esp 127.0.0.1:8888 # sim -> mock_esp
  python run_controller.py --model models/n1_catch --esp 192.168.4.10:8888 --realtime
  # fleet: addresses in CAR_INDEX order -- the first address drives cmd[0]
  python run_controller.py --model models/n3_catch --source udp \
      --esp 192.168.4.10:8888,192.168.4.11:8888,192.168.4.12:8888
  python run_controller.py --model models/n1_pin --source udp --pose-port 9870 \
      --esp 192.168.4.10:8888 --rate-limit-speed
  # what `arena up` runs on the control Orin: find the cars by broadcast, start
  # disarmed, report to the hub, take arm/disarm on localhost:9872
  python run_controller.py --model models/n1_catch --source udp --esp auto \
      --telemetry 127.0.0.1:9871 --control-port 9872 --start-disarmed
"""

from __future__ import annotations

import argparse
import signal
import time

from single_pursuer.env import DT, SinglePursuerEnv
from controller_runtime.pose_types import VehiclePose

from pc_controller.esp_link import EspLink, discover_cars, parse_targets
from pc_controller.portable_loop import PortableLoop
from pc_controller.pose_stream import UdpPoseSource
from pc_controller.realtime import run_realtime


def _world_env_from_metrics(loop: PortableLoop, seed: int) -> SinglePursuerEnv:
    """Build the sim world with the same scenario the model was evaluated on
    (same auto-fill discipline as playback.py / robustness_eval.py in the repo)."""
    m = loop.policy.metrics
    return SinglePursuerEnv(
        num_pursuers=int(m.get("num_pursuers", 1)),
        capture_mode=m.get("capture_mode", "surround"),
        use_car_cameras=bool(m.get("use_car_cameras", False)),
        evader_speed_frac=float(m.get("evader_speed_frac", 0.0)),
        capture_hold_steps=int(m.get("capture_hold_steps", 5)),
        immobile_speed_mps=float(m.get("immobile_speed_mps", 0.5)),
        require_immobility=bool(m.get("require_immobility", True)),
        seed=seed,
    )


def run_sim(loop: PortableLoop, link: EspLink | None, *, episodes: int, seed: int, realtime: bool) -> None:
    env = _world_env_from_metrics(loop, seed)
    captures = 0
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        loop.reset()
        t, steps, done, info = 0.0, 0, False, {}
        while not done:
            tick_start = time.perf_counter()
            pursuer_poses = [VehiclePose(p["x"], p["y"], p["heading"], t) for p in env.pursuers]
            evader_pose = VehiclePose(env.evader["x"], env.evader["y"], env.evader["heading"], t)
            action = loop.tick(pursuer_poses=pursuer_poses, evader_pose=evader_pose)
            if link is not None:
                link.send_commands(action)
            elif steps % 10 == 0:
                print(f"  ep{episode} step{steps:4d}  cmd throttle={action[0]:+.2f} steer={action[1]:+.2f}")
            _obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            t += DT
            steps += 1
            if realtime:
                time.sleep(max(0.0, DT - (time.perf_counter() - tick_start)))
        captured = bool(info.get("captured", False))
        captures += int(captured)
        print(f"episode {episode}: {'CAPTURE' if captured else 'no capture'} in {steps} steps")
        if link is not None:
            link.send_stop()
    print(f"\ncapture rate: {captures}/{episodes}")
    print(f"latency: {loop.latency_budget.summary()}")


def run_udp_legacy(loop: PortableLoop, link: EspLink | None, *, pose_port: int) -> None:
    """The original loop: tick on every frame received, no telemetry. Kept for
    A/B comparison (--legacy-loop). The default is pc_controller.realtime."""
    source = UdpPoseSource(pose_port, expected_pursuers=loop.policy.num_pursuers)
    print(f"listening for pose frames on UDP :{pose_port} (ctrl-C to stop) ...")
    stalled = False
    ticks = 0
    try:
        while True:
            frame = source.next_frame()
            if frame is None:
                if not stalled:
                    print("POSE FEED STALLED >0.3s -> E-stop, waiting for frames")
                    if link is not None:
                        link.send_stop()
                    loop.reset()  # stale histories would corrupt velocity estimates
                    stalled = True
                continue
            if stalled:
                print("pose feed recovered, re-armed")
                stalled = False
            pursuer_poses, evader_pose = frame
            action = loop.tick(pursuer_poses=pursuer_poses, evader_pose=evader_pose)
            if link is not None:
                link.send_commands(action)
            ticks += 1
            if ticks % 50 == 0:
                print(f"  tick {ticks}  cmd throttle={action[0]:+.2f} steer={action[1]:+.2f}  "
                      f"latency {loop.latency_budget.summary()}")
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/n1_catch", help="model dir (models/n1_catch or models/n1_pin)")
    parser.add_argument("--source", choices=["sim", "udp"], default="sim")
    parser.add_argument("--esp", default=None,
                        help="ESP32 address as ip:port, or a comma-separated list in "
                             "CAR_INDEX order for a fleet "
                             "(ip0:8888,ip1:8888,ip2:8888), or 'auto' to find the cars by "
                             "broadcast (needs exactly car indices 0..N-1 to answer). "
                             "Omit to print instead of send.")
    parser.add_argument("--pose-port", type=int, default=9870, help="UDP port for perception frames (--source udp)")
    parser.add_argument("--episodes", type=int, default=3, help="sim episodes to run (--source sim)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--realtime", action="store_true", help="pace sim at real 10 Hz (use when an ESP is attached)")
    parser.add_argument("--rate-limit-speed", action="store_true",
                        help="clamp velocity estimates to vehicle accel limits (contact-noise mitigation)")
    rt = parser.add_argument_group("real-time loop (--source udp)")
    rt.add_argument("--tick-hz", type=float, default=10.0,
                    help="control rate; the policy was trained at 10 Hz")
    rt.add_argument("--max-pose-age", type=float, default=0.15,
                    help="never act on a pose whose perception latency exceeds this (s)")
    rt.add_argument("--telemetry", default=None, help="arena hub host:port, e.g. 127.0.0.1:9871")
    rt.add_argument("--control-port", type=int, default=None,
                    help="localhost UDP port for arm/disarm (the `arena go`/`halt` commands)")
    rt.add_argument("--start-disarmed", action="store_true",
                    help="send E-stop to every car until an operator arms")
    rt.add_argument("--legacy-loop", action="store_true",
                    help="the original tick-per-frame loop, for A/B comparison")
    args = parser.parse_args()
    # The arm switch, the pose-age gate and telemetry exist only in the real-time
    # loop. Accepting them anywhere else would mean a "disarmed" start that drives
    # the cars on the first frame, and a STOP that nothing is listening for.
    if args.legacy_loop and (args.start_disarmed or args.control_port or args.telemetry):
        parser.error("--legacy-loop has no arm switch, STOP port or telemetry: drop "
                     "--start-disarmed/--control-port/--telemetry, or drop --legacy-loop")
    if args.source == "sim" and (args.start_disarmed or args.control_port):
        parser.error("--start-disarmed/--control-port apply only to --source udp "
                     "(a sim run with --esp drives the cars immediately)")

    # `arena stop` sends SIGTERM. Turn it into the same clean shutdown as
    # Ctrl-C, so the E-stop in link.close() goes out before the process exits.
    # Later TERMs (the supervisor forwards one) are ignored, so nothing can
    # interrupt that final E-stop halfway through.
    def _term(*_a):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _term)
    except (ValueError, AttributeError):
        pass

    loop = PortableLoop(args.model, rate_limit_speed=args.rate_limit_speed)
    m = loop.policy.metrics
    print(f"model: {args.model}  (N={loop.policy.num_pursuers}, obs {loop.policy.input_dim}, "
          f"capture_mode={m.get('capture_mode')}, evader_speed={m.get('evader_speed_frac')})")

    link = None
    if args.esp == "auto":
        n_pursuers = loop.policy.num_pursuers
        found = discover_cars()
        if -1 in found:
            clash = ", ".join(f"{d['ip']} (index {d['car']})" for d in found.pop(-1))
            raise SystemExit(f"two cars share a CAR_INDEX: {clash}. Give each car its own "
                             "index (`arena cars index <ip> <n>`) before driving.")
        missing = [i for i in range(n_pursuers) if i not in found]
        if missing:
            seen = ", ".join(f"index {i} @ {d['ip']}" for i, d in sorted(found.items())) or "none"
            raise SystemExit(f"--esp auto: no car answered for index {missing} (found: {seen}). "
                             "Is it powered and on this WiFi? `arena cars` lists what answers.")
        args.esp = ",".join(f"{found[i]['ip']}:8888" for i in range(n_pursuers))
        print("discovered: " + ", ".join(
            f"car {i} @ {found[i]['ip']} ({found[i].get('fw')})" for i in range(n_pursuers)))
    if args.esp:
        targets = parse_targets(args.esp)
        n_pursuers = loop.policy.num_pursuers
        # Address ORDER is the fleet order: targets[i] must be the car flashed
        # with CAR_INDEX i, because that is the car that will act on cmd[i].
        # A mismatch here means the wrong physical car obeys each command, which
        # is not something you want to discover with the wheels down.
        if len(targets) > n_pursuers:
            raise SystemExit(
                f"--esp lists {len(targets)} cars but the model drives {n_pursuers} "
                f"pursuer(s); car(s) {list(range(n_pursuers, len(targets)))} would "
                "receive no command and sit in failsafe. Use a matching model or "
                "fewer addresses.")
        if len(targets) < n_pursuers:
            print(f"WARNING: model drives {n_pursuers} pursuers but only {len(targets)} "
                  f"car(s) given; slots {list(range(len(targets), n_pursuers))} are "
                  "commanded but unaddressed (nothing listening).")
        link = EspLink(targets)
        for i, (host, port) in enumerate(targets):
            print(f"  car index {i} (cmd[{i}]) -> udp://{host}:{port}")
    else:
        print("no --esp given: dry run, commands printed only")

    try:
        if args.source == "sim":
            run_sim(loop, link, episodes=args.episodes, seed=args.seed, realtime=args.realtime or bool(args.esp))
        elif args.legacy_loop:
            run_udp_legacy(loop, link, pose_port=args.pose_port)
        else:
            run_realtime(loop, link, pose_port=args.pose_port, tick_hz=args.tick_hz,
                         max_pose_age_s=args.max_pose_age, telemetry=args.telemetry,
                         control_port=args.control_port, start_disarmed=args.start_disarmed,
                         model_name=args.model,
                         log=lambda m: print(f"[control {time.strftime('%H:%M:%S')}] {m}", flush=True))
    finally:
        if link is not None:
            link.close()


if __name__ == "__main__":
    main()
