# Portable N=1 Pursuit Controller

Self-contained deployment bundle for the EE198 pursuit project: a PC runs the
trained N=1 pursuit policy and streams throttle/steer commands over UDP to an
ESP32 on the car. **No PyTorch, no skrl, no GPU, no training repo needed** —
copy this folder to any machine with Python 3.10+ and it works.

```
pip install -r requirements.txt     # numpy, gymnasium, onnxruntime
python selftest.py                  # must print SELFTEST PASSED before anything else
```

## What's in here

| Path | What it is |
|---|---|
| `models/n1_catch/`, `models/n1_pin/` | The two frozen V1 single-pursuer policies as verified ONNX exports (+ manifest + training/eval metrics). |
| `single_pursuer/`, `vehicle_dynamics.py`, `configs/` | Vendored copies of the exact observation/dynamics code the policies were trained against. Used ONLY to build observation vectors — never edit these here. |
| `controller_runtime/` | Vendored pose ingest, velocity estimation, latency tracking, observation adapter. |
| `pc_controller/` | New PC-side code: ONNX inference, the 10 Hz control loop, the ESP UDP link, the perception UDP input. |
| `run_controller.py` | Main entry point (see below). |
| `selftest.py` + `reference_vectors.json` | Golden-vector health check: proves the vendored contract + models on this machine reproduce the outputs frozen at bundle creation. |
| `tools/mock_esp.py` | Fake ESP32 for testing the full PC side with zero hardware. |
| `esp32/esp32_receiver/` | Arduino sketch for the real ESP32 (WiFi UDP → ESC/servo PWM, with failsafe). |

## Which model?

- **`n1_catch` (start here):** chase-and-catch a fleeing evader (0.6× speed).
  Evaluate: 100% capture static AND fleeing. SIL (realistic position-only
  perception): 100%, exact parity. Robustness grid: 100% at the deployment
  latency spec. The confident first-hardware choice.
- **`n1_pin`:** approach-and-hold against a static evader (the "stop motion"
  spec: enclosed + ≤0.5 m/s for 5 consecutive ticks). Evaluate: 100%, zero
  collisions — but under realistic position-only perception SIL drops to ~40%
  (contact nudges read as false speed). Use `--rate-limit-speed` as a partial
  mitigation and treat this one as the stretch demo until it's retrained under
  perception noise.

## Bring-up ladder

**1. No hardware — prove the whole PC side.** Two terminals:
```
python tools/mock_esp.py
python run_controller.py --model models/n1_catch --esp 127.0.0.1:8888
```
The simulator plays the world; the controller sees only poses (like a camera
would), runs the policy, and transmits real packets. You should see captures in
the controller terminal and ~10 pkt/s with sane throttle/steer in the mock ESP.

**2. Real ESP32, wheels off the ground.** Edit WiFi credentials + pins in
`esp32/esp32_receiver/esp32_receiver.ino`, flash it, read its IP off the serial
monitor, then re-run step 1 with `--esp <esp-ip>:8888`. The sim episode now
physically twitches the real servo/ESC. Unplug the network mid-run to watch the
300 ms failsafe drop it to neutral.

**3. Real perception.** When the overhead-camera ArUco pipeline exists, have it
send pose frames (format below) and run:
```
python run_controller.py --model models/n1_catch --source udp --pose-port 9870 --esp <esp-ip>:8888
```
Dead-man stop is built in: if the pose feed stalls >0.3 s, an E-stop goes out
and histories reset until frames resume.

## Wire formats

**PC → ESP command packet** (UDP :8888, one JSON per datagram):
```json
{"seq": 42, "t": 1720300000.123, "estop": false, "cmd": [[0.43, -0.10]]}
```
`cmd` = `[throttle, steer]` per car, normalized [-1, 1], +steer = left. The ESP
ignores `seq` ≤ last applied, goes neutral on `estop` or a 300 ms stream stall.

**ESP → PC delivery confirmation** (reply to every parseable packet):
```json
{"ack": 42, "applied": true, "rssi": -55}
```
`applied` is false when the packet was dropped as stale; `rssi` is the ESP's
WiFi signal in dBm (> −70 good, < −80 expect trouble). Verify the link any time
with `python tools/link_test.py --esp <ip>:8888` — it reports per-packet
delivery, round-trip time, loss, and signal strength (safe on a live car: zero
throttle, gentle steering sweep — but first runs wheels-off anyway).

**Perception → PC pose frame** (UDP :9870, one JSON per datagram):
```json
{"t": 12.34,
 "pursuers": [{"x": 1.0, "y": -2.0, "heading": 0.5}],
 "evader":   {"x": 3.0, "y": 1.0, "heading": -1.2}}
```
Meters/radians, arena-centered, heading 0 = +x CCW+, `t` = frame CAPTURE time
(velocities are finite-differenced from consecutive frames).

## Hard constraints to respect

- **Latency budget ≤ 0.1 s camera→policy→radio, total.** Measured in training:
  0.2 s of latency collapses capture performance. The loop logs its own
  PC-side latency; budget the camera and WiFi hops too.
- **The arena is part of the model.** These policies were trained in the 14 m ×
  10 m arena defined by `configs/hive_foundation_experiment.json`, and the
  observation scaling comes from that file. Do NOT swap in a different config
  (e.g. `test_arena_6ft.json`, included for reference) and expect the same
  policy to work — a different real arena size means retraining in the main
  repo, then re-exporting here.
- **Don't edit the vendored folders.** If the contract changes in the main
  repo: re-copy the files, re-run `tools/make_reference_vectors.py`, and commit
  both together. `selftest.py` exists to catch exactly this drift.
- **Sim speeds assume the real car matches the config** (3.1 m/s top speed,
  1.6 m/s² accel, 28° steering). Before trusting closed-loop driving, do a
  simple system-ID pass on the real car and compare. Cap the drive power in
  the sketch for early runs regardless (`THROTTLE_MAX_US` for hobby-ESC
  hardware, `MAX_DUTY` for the L298N H-bridge variant — see the sketch's own
  HARDWARE MAPPING comment for which one applies to your car).
- **The policy uses reverse.** Observed in sim runs: `n1_catch` sometimes
  drives backwards (throttle −1.0) all the way to a capture — the sim treats
  reverse as symmetric with forward. The car's ESC must support smooth
  proportional reverse (crawler-style ESC, no double-tap-to-reverse lockout),
  or the real car will behave very differently from sim.

## Provenance

Models frozen 2026-07-04 as V1 deployment candidates
(`artifacts/v1_deployment_candidates/` in the main repo; full numbers in
`V1_ACCEPTANCE_REPORT.md` there). ONNX exports verified against the torch
checkpoints to ≤1e-5 at export. Torch `.pt` originals stay in the main repo —
they are not needed here.
