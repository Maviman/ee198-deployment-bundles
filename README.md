# EE198 deployment bundles

Canonical home for the two self-contained deployment bundles of the EE198
pursuit project (three RC pursuer cars cooperatively corral-and-pin an
evader, driven by a trained RL policy). Clone this onto a deployment machine
and each bundle brings itself up — no training repo, no PyTorch.

This is a sibling project to `AI Training` (the training factory / test
suite / sim stack); edit bundle code directly here.

| Bundle | Runs on | What it does |
|---|---|---|
| [`portable_n1_controller/`](portable_n1_controller/) | the controller PC | Runs the frozen N=1 pursuit policy (ONNX): pose frames in (UDP :9870) → throttle/steer commands out to the car's ESP32 (UDP :8888, ACK + failsafe). Includes the ESP32-S3 firmware and mock/link-test tools. Hardware-verified 2026-07-06/07. |
| [`portable_orin_perception/`](portable_orin_perception/) | Jetson Orin | ROS 2 + ArUco overhead perception: USB webcam → marker poses in arena meters → UDP pose frames for the controller. One-script setup (`setup_orin.sh`), zero-hardware `selftest.py`, calibration tooling, printable markers. |

Start with each bundle's own README (bring-up ladders, wire formats, hard
constraints). New to ROS 2: `portable_orin_perception/ROS2_LEARNING.md`.

## Quick start on the Orin

```
git clone <this-repo>
cd ee198-deployment-bundles/portable_orin_perception
python3 selftest.py        # before installing anything
./setup_orin.sh            # installs ROS 2 (Humble/Jazzy by OS) + builds
```

## Credentials

The ESP32 sketch here has PLACEHOLDER WiFi credentials (`YOUR_WIFI_SSID` /
`YOUR_WIFI_PASSWORD`) — edit before flashing, or copy
`wifi_credentials.h.example` to `wifi_credentials.h` and fill in the real
values. `wifi_credentials.h` is gitignored — real credentials never get
committed.
