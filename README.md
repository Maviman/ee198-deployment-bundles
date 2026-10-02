# EE198 deployment bundles

RC cars chase an evader inside a marked arena, driven by a trained RL policy.
This repo is what actually runs on the hardware — no training repo, no PyTorch.

## How it works

```
  overhead camera ── USB ──► VISION ORIN: run_vision.py
                               newest frame -> tag detection -> pose frame (every frame)
                                        │ UDP :9870  (Ethernet between the Orins)
                                        ▼
                             CONTROL ORIN: run_controller.py + dashboard
                               pursuit policy (ONNX), one tick per 100 ms
                                        │ throttle / steer, UDP :8888 (WiFi)
                                        ▼
                                 ESP32 on each car
                                        │
                           ┌────────────┴────────────┐
                           ▼                         ▼
                   L298N H-bridge            steering servo
                 7.4 V drive motor
```

A camera finds each car, the policy decides where to go, the ESP32 drives the
motors. Every hop has a failsafe: if poses stop arriving, or commands stop
arriving, the car stops on its own. One Orin can also run both halves.

## The two bundles

**[`portable_orin_perception/`](portable_orin_perception/)** — runs on the
Jetson Orin. Webcam → ArUco marker detection → car positions in arena meters,
sent to the controller as UDP pose frames. Includes calibration tooling and
printable markers.

**[`portable_n1_controller/`](portable_n1_controller/)** — runs the policy.
Pose frames in, throttle/steer commands out to the car. Also holds the ESP32
firmware and the mock/link-test tools you use before touching real hardware.

Both are self-contained: clone, run the selftest, bring it up.

## Start here

Everything is driven by one CLI, `./arena` (Windows: `arena.cmd`):

```bash
./arena sim            # rehearse the whole stack on this machine, no hardware
./arena init --vision <user>@<orin> --control <user>@<orin>     # then, per session:
./arena sync && ./arena scan && ./arena up && ./arena go         # ... halt / stop
```

**[ARENA.md](ARENA.md) is the runbook**: the two-Orin split, first-time setup,
the session commands, the dashboard, and what was changed to cut latency.

| If you want to… | Read |
|---|---|
| Run the arena (two Orins, or one), day to day | [ARENA.md](ARENA.md) |
| Install on Orins that have no internet | [deploy/offline/README.md](deploy/offline/README.md) |
| Drive 1 car or 3 (`arena fleet`) | [ARENA.md § How many pursuers](ARENA.md#how-many-pursuers) |
| Bring up one Jetson by hand, step by step | [ORIN_DEPLOYMENT.md](ORIN_DEPLOYMENT.md) |
| Get the camera calibrated fast | [ORIN_QUICKSTART.md](ORIN_QUICKSTART.md) |
| Learn the ROS 2 concepts this uses | [ROS2_LEARNING.md](portable_orin_perception/ROS2_LEARNING.md) |
| Know what's planned and what's blocked | [HANDOFF.md](HANDOFF.md) |

## Before you flash the ESP32

The sketch needs WiFi credentials, and they must never be committed. Copy the
template and fill in your network:

```bash
cd portable_n1_controller/esp32/esp32_receiver
cp wifi_credentials.h.example wifi_credentials.h
```

`wifi_credentials.h` is gitignored.

## Known limits

Two things will bite you if you don't know them going in:

- **The arena is part of the model.** The policies were trained in a 14 × 10 m
  arena and the observation scaling bakes that in. A different real arena size
  means retraining, not just a config edit.
- **Tag pixels set the resolution.** A square arena fits the image's short
  side, so 7 cm car tags are ~18 px at 640×480 (below the detection floor) and
  ~25 px at the new 1280×720 default. `arena scan` measures the real number
  and says whether it is enough. [HANDOFF.md](HANDOFF.md) has the options for
  a bigger arena.
