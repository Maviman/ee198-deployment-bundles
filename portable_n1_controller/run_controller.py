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
  python run_controller.py --model models/n1_pin --source udp --pose-port 9870 \
      --esp 192.168.4.10:8888 --rate-limit-speed
"""

from __future__ import annotations

import argparse
import time

from single_pursuer.env import DT, SinglePursuerEnv
from controller_runtime.pose_types import VehiclePose

from pc_controller.esp_link import EspLink
from pc_controller.portable_loop import PortableLoop
from pc_controller.pose_stream import UdpPoseSource


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


def run_udp(loop: PortableLoop, link: EspLink | None, *, pose_port: int) -> None:
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
    parser.add_argument("--esp", default=None, help="ESP32 address as ip:port; omit to print instead of send")
    parser.add_argument("--pose-port", type=int, default=9870, help="UDP port for perception frames (--source udp)")
    parser.add_argument("--episodes", type=int, default=3, help="sim episodes to run (--source sim)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--realtime", action="store_true", help="pace sim at real 10 Hz (use when an ESP is attached)")
    parser.add_argument("--rate-limit-speed", action="store_true",
                        help="clamp velocity estimates to vehicle accel limits (contact-noise mitigation)")
    args = parser.parse_args()

    loop = PortableLoop(args.model, rate_limit_speed=args.rate_limit_speed)
    m = loop.policy.metrics
    print(f"model: {args.model}  (N={loop.policy.num_pursuers}, obs {loop.policy.input_dim}, "
          f"capture_mode={m.get('capture_mode')}, evader_speed={m.get('evader_speed_frac')})")

    link = None
    if args.esp:
        host, _, port = args.esp.partition(":")
        link = EspLink(host, int(port or 8888))
        print(f"sending commands to udp://{link.addr[0]}:{link.addr[1]}")
    else:
        print("no --esp given: dry run, commands printed only")

    try:
        if args.source == "sim":
            run_sim(loop, link, episodes=args.episodes, seed=args.seed, realtime=args.realtime or bool(args.esp))
        else:
            run_udp(loop, link, pose_port=args.pose_port)
    finally:
        if link is not None:
            link.close()


if __name__ == "__main__":
    main()
