# rc16_n1_two_camera: one pursuer, MotorDayz 1:16 toy, 8 x 8 ft arena, PWM

Runtime `rc16_n1_v1` (`pc_controller/runtimes/rc16_n1_v1.py`). One pursuer drives to the evader and catches it (centre distance <= 0.35 m for 0.3 s). Trained in Isaac Sim on the rc16 car model with the two-corner-camera pose emulator, so it needs only (x, y, heading) of both cars from the cameras.

## Evaluation (simulation, 256 episodes each, deterministic)

| run | camera | actuation | catch | pursuer out | evader out | time to catch (s) | rams/ep |
|---|---|---|---|---|---|---|---|
| n1c_emu | emulator | nominal | 1.000 ± 0.000 | 0.000 | 0.000 | 1.52 | 0.26 |
| n1c_truth | truth | nominal | 1.000 ± 0.000 | 0.000 | 0.000 | 1.60 | 0.15 |
| n1c_emu_rand | emulator | randomized | 1.000 ± 0.000 | 0.000 | 0.000 | 1.62 | 0.54 |

Evader: 30% parked, 70% scripted flee at difficulty 0.3 (duty cap 0.615, slower than the pursuer).

## Before the first real run

- Set the perception arena to 8 ft (2.4384 m), origin at the centre, +x right, +y up.
- Commands are PWM duty and a proportional steering target. Start wheels-off, then with `deploy_throttle_limit` lowered in the manifest (e.g. 0.5) for the first runs: the sim car reaches 2.66 m/s and needs ~1.2 m to stop from 2.5 m/s.
- Trained with 30% on/off steering and 50% one-step delay, so it tolerates either steering hardware and up to 0.1 s of extra command latency; beyond that, expect degradation.
- Pose stalls: the loop's dead-man stop applies (no poses -> E-stop).

ONNX verified against torch to 1.9e-06; golden replay matched the training env to 3.6e-07 over 54 ticks.
