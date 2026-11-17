# rc16_n1_md_elite: one pursuer, MotorDayz 1:16 MD Elite (4WD, servo), 8 x 8 ft arena

Runtime `rc16_n1_v1` (`pc_controller/runtimes/rc16_n1_v1.py`). One pursuer drives to the evader and catches it (centre distance <= 0.35 m for 0.3 s). Trained in Isaac Sim on the rc16 car model with the two-corner-camera pose emulator, so it needs only (x, y, heading) of both cars from the cameras.

## Evaluation (simulation, 256 episodes each, deterministic)

| run | camera | actuation | catch | pursuer out | evader out | time to catch (s) | rams/ep |
|---|---|---|---|---|---|---|---|
| n1d_emu | emulator | nominal | 1.000 ± 0.000 | 0.000 | 0.000 | 1.40 | 0.21 |
| n1d_truth | truth | nominal | 1.000 ± 0.000 | 0.000 | 0.000 | 1.49 | 0.17 |
| n1d_emu_rand | emulator | randomized | 1.000 ± 0.000 | 0.000 | 0.000 | 1.45 | 0.43 |

Evader: the same car; 30% parked, 70% scripted flee at difficulty 0.3 (slower than the pursuer).

## Before the first real run

- Set the perception arena to 8 ft (2.4384 m), origin at the centre, +x right, +y up.
- Throttle cap: the model expects a full command to reach 2.7 m/s (6 mph). On each car set the firmware `limit` (start at 0.37), time the car over 2 m at full command, and adjust `limit` until it reads that speed. The stock car does 18 mph; do not run it uncapped.
- Zero throttle coasts on this firmware. The model brakes by commanding reverse, so the car must accept reverse while rolling forward. Sim: 4.8 m/s^2 braking, 0.8 m/s^2 coasting (estimate; measure the roll-out).
- Steering: a proportional servo, 30 deg at full command. Set the end points with `servo <centre> <left> <right>` so the wheels reach that angle.
- Start wheels-off, then with `deploy_throttle_limit` lowered in the manifest (e.g. 0.5) for the first runs. Trained with up to 0.1 s of extra command latency; beyond that, expect degradation.
- Weight (0.68 kg), drag and geometry are estimates until the chassis is measured.
- Pose stalls: the loop's dead-man stop applies (no poses -> E-stop).

ONNX verified against torch to 1.9e-06; golden replay matched the training env to 5.4e-07 over 48 ticks.
