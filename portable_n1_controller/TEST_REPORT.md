# Test Report — Portable N=1 Pursuit Controller

**Date:** 2026-07-06/07 &nbsp;|&nbsp; **Bundle:** `portable_n1_controller/` &nbsp;|&nbsp; **Status: ALL TESTS PASSED**

> **⚠ Historical record — the car hardware has changed since this run.**
> This documents the V1 car: a hobby ESC + servo driven by the ESP32Servo
> library on GPIO 4/5. The car is now V2 — a 7.4 V brushed motor on an L298N
> H-bridge (GPIO 6/7) plus a steering servo on raw LEDC (GPIO 5), and the
> ESP32Servo dependency is gone. The comms, latency, and golden-vector results
> below still stand; **anything describing motor or servo wiring does not.**
> Kept unedited as evidence of what was actually tested on those dates. The
> "not yet tested" list at the bottom needs a fresh bench run against current
> firmware before it can be used as a checklist.

Test PC: Windows 11, Python 3.11 (conda `env_isaaclab`), onnxruntime 1.27.0.
Target hardware: Espressif ESP32-S3-DevKitC-1 (micro-USB revision), flashed via
CP2102N UART bridge on COM8.

---

## 1. Bundle selftest (`selftest.py`) — PASS

| Check | Result | Detail |
|---|---|---|
| Golden-vector parity, `n1_catch` | PASS | 5 recorded pose scenarios → observation + ONNX action match frozen references (obs tol 1e-9, action tol 1e-5) |
| Golden-vector parity, `n1_pin` | PASS | same, 5 scenarios |
| Command packet round-trip | PASS | real UDP send/receive; [-1,1] clipping verified (−1.7 → −1.0); E-stop packet correct |
| Perception frame parsing | PASS | valid frame parsed exactly; wrong pursuer count rejected |

Proves: the vendored observation contract, both ONNX models, and the wire
formats are healthy on this machine. Rerun on every new PC the folder lands on.

## 2. Policy-in-the-loop simulation (`run_controller.py --source sim`) — PASS

Full deployment pipeline: raw poses only (as a camera provides) → finite-difference
velocity estimation → observation build → ONNX inference → command out.

| Metric | Result |
|---|---|
| Capture rate (`n1_catch`, fleeing evader, 10 episodes) | **10/10 (100%)** |
| Steps to capture | 23–44 (2.3–4.4 sim seconds) |
| PC-side latency, mean / max | **0.07 ms / 0.41 ms** (budget: 100 ms) |
| Ticks within latency budget | 304/304 (100%) |

Matches the model's frozen V1 evaluation (100% capture) — the deployment path
costs nothing vs. the training-side evaluator for N=1.

## 3. Live packet streaming (sim → `tools/mock_esp.py`) — PASS

One real-time episode streamed over UDP to the mock receiver:

| Metric | Result |
|---|---|
| Packet rate | 10.2 pkt/s (target 10 Hz) |
| Packet age at receiver | < 1 ms |
| Stale/out-of-order handling | verified (dropped by seq check) |
| E-stop on episode end + shutdown | verified received |

## 4. Command exchange with delivery confirmation (`tools/link_test.py`) — PASS

Every packet ACKed by the receiver (`{"ack": seq, "applied": bool, "rssi": dBm}`),
100 packets at 10 Hz:

| Metric | Result |
|---|---|
| Delivered / confirmed | **100/100 (100.0%)** |
| Lost | 0 |
| Stale (delivered, not applied) | 0 |
| Round-trip time mean / median / p95 / max | **2.56 / 2.59 / 2.79 / 3.75 ms** |
| RTTs over 100 ms budget | 0 |
| Verdict line from tool | PASS |

Note: localhost numbers = protocol-stack floor. The same test against the real
board over WiFi is the first true hardware latency data point (pending WiFi
credentials).

## 5. Firmware build + flash (ESP32-S3-DevKitC-1) — PASS

| Step | Result |
|---|---|
| PlatformIO build (`esp32-s3-devkitc-1`, ArduinoJson + ESP32Servo) | SUCCESS — RAM 13.9%, Flash 21.1% |
| Flash via UART port (COM8, 115200) | SUCCESS, hash verified, no BOOT-button intervention |
| Boot over serial | verified: clean boot, `connecting to WiFi...` (placeholder credentials, expected) |
| ACK firmware re-flash | SUCCESS |

Root cause of the earlier "COM3 visible but upload fails": missing Silicon Labs
CP210x driver (Windows device error 28). Fixed; driver vendored into
`esp32/drivers/cp210x/` for future machines.

## 6. Over-the-air link test vs real ESP32 (2026-07-07) — PASS

Board on home WiFi (Google WiFi mesh), IP 192.168.86.199, `tools/link_test.py`,
200 packets at 10 Hz per run.

**First attempt exposed WiFi power-save:** alternating ~15/~105 ms RTTs (radio
napping between router beacons), 94% delivery, 74/200 packets over the 100 ms
budget. Fixed with `WiFi.setSleep(false)` in the firmware.

**Second attempt exposed a restart lockout:** a restarted PC controller starts
its seq counter at 0, and the board's anti-replay check silently rejected every
command (delivered 100%, applied 0%) until power-cycle. Fixed: the board resyncs
its seq tracking whenever its failsafe is engaged (boot / E-stop / stream stall
— every real restart passes through that state), plus a small backwards-gap
check; live stale/reordered packets are still rejected.

**Final results (after both fixes), two back-to-back runs:**

| Metric | Run 1 | Run 2 (simulated controller restart) |
|---|---|---|
| Delivered | 199/200 (99.5%) | **200/200 (100%)** |
| Stale (delivered, not applied) | 0 | **0** (restart resync verified) |
| RTT mean / median / p95 | 12.2 / 10.5 / 17.7 ms | 12.5 / 10.3 / 16.3 ms |
| RTTs over 100 ms budget | 1 | 1 |
| RSSI | −31 dBm | −32 dBm |
| Verdict | PASS (isolated losses only) | PASS |

Takeaway for the latency budget: WiFi command hop costs ~5–9 ms one-way typical
(half of RTT), with rare ~100+ ms outliers at ~0.5% rate — one such outlier is
one control tick late, well inside what the policy tolerates (it was trained
with 0.1 s latency robustness). Command link is cleared for driving.

## Defects found and fixed during testing

1. `mock_esp.py`: missing `import json` — crashed on first ACK (caught by test 4's 0% delivery).
2. `esp_link.py`: Windows UDP quirk — one ICMP port-unreachable bounce made
   `recvfrom` raise `ConnectionResetError`; now swallowed in `wait_ack()`.
3. `esp32_receiver.ino`: original pins GPIO 25/26 don't exist on the S3 → moved
   to GPIO 4 (throttle) / GPIO 5 (steer).
4. `esp_link.py`: numpy floats broke JSON serialization — cast to plain float.
5. `selftest.py`: wrong attribute name (`timestamp` → `timestamp_s`).

## Not yet tested (blocked on hardware/credentials)

- Motor outputs via H-bridge (L298N concept test → TB6612 for driving; bench test wheels-off first).
- OV2640 2MP car camera on the ESP32-S3: driver bring-up, still capture, MJPEG
  stream, and the command-link contention test (run `link_test.py` while
  streaming — control has priority; camera is telemetry/demo only, not in the
  v1 control loop).
- Real perception input (`--source udp`; needs teammate's ArUco pipeline).
- End-to-end latency budget with camera + WiFi in the loop (the ≤100 ms hard requirement).
