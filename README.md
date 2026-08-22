# EE198 deployment bundles

RC cars chase an evader inside a marked arena, driven by a trained RL policy.
This repo is what actually runs on the hardware — no training repo, no PyTorch.

## How it works

```
  overhead camera
        │
        ▼
  ArUco perception  ──── pose frames ────►  pursuit policy (ONNX)
   (Jetson Orin)          UDP :9870              │
                                                 │ throttle / steer
                                                 ▼  UDP :8888
                                          ESP32 on the car
                                                 │
                                    ┌────────────┴────────────┐
                                    ▼                         ▼
                            L298N H-bridge            steering servo
                          7.4 V drive motor
```

A camera finds each car, the policy decides where to go, the ESP32 drives the
motors. Every hop has a failsafe: if poses stop arriving, or commands stop
arriving, the car stops on its own.

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

```bash
git clone git@github.com:Maviman/ee198-deployment-bundles.git
cd ee198-deployment-bundles/portable_orin_perception
python3 selftest.py        # run this before installing anything
./setup_orin.sh            # installs ROS 2 + builds
```

Then follow the bring-up ladder in whichever bundle you're working on — each
README walks from "no hardware at all" up to "real car, wheels off the ground."

| If you want to… | Read |
|---|---|
| Run both bundles on one Jetson (the demo runbook) | [ORIN_DEPLOYMENT.md](ORIN_DEPLOYMENT.md) |
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
- **Perception is at its resolution limit.** At 640×480 the 7 cm car markers
  land around 24 px — right at the ArUco detection floor. Making the arena
  bigger needs more than a wider lens. [HANDOFF.md](HANDOFF.md) has the
  measurements and the options.
