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
| `models/c37_commit3_ft_g997/` | The 3-pursuer hive role commander (manifest `"runtime": "hive_commander_v1"`), run by `pc_controller/runtimes/hive_commander_v1.py`; `golden_vectors.json` pins it in `selftest.py`. Read its README for the transfer gaps. |
| `pc_controller/loops.py` | Picks the loop a model needs from its manifest: `PortableLoop` for flat policies, `HiveLoop` for role commanders. |
| `single_pursuer/`, `vehicle_dynamics.py`, `configs/` | Vendored copies of the exact observation/dynamics code the policies were trained against. Used ONLY to build observation vectors — never edit these here. |
| `controller_runtime/` | Vendored pose ingest, velocity estimation, latency tracking, observation adapter. |
| `pc_controller/` | New PC-side code: ONNX inference, the 10 Hz control loop, the ESP UDP link, the perception UDP input. |
| `run_controller.py` | Main entry point (see below). |
| `selftest.py` + `reference_vectors.json` | Golden-vector health check: proves the vendored contract + models on this machine reproduce the outputs frozen at bundle creation. |
| `tools/mock_esp.py` | Fake ESP32 for testing the full PC side with zero hardware. |
| `esp32/esp32_receiver/` | Firmware for the ESP32 on each car (WiFi UDP → presses the FWD/BACK/LEFT/RIGHT buttons of the car's RC remote, with failsafe). `button_mod.h` turns a command into presses; `esp32/test_host/` tests it on a PC. |

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

**2. Real ESP32, wheels off the ground.** Put the WiFi credentials in
`esp32/esp32_receiver/wifi_credentials.h`, flash, read the IP off the serial
monitor, then re-run step 1 with `--esp <esp-ip>:8888`. The sim episode now
presses the real remote's buttons. Unplug the network mid-run to watch the
300 ms failsafe release every button.

**The V3 car (2026-10-02): the ESP32 presses the buttons of the car's own RC
remote.** Four GPIOs drive four NPN transistors (base resistor each; a 100 k
base-to-emitter pull-down is recommended) wired across the remote's FWD,
BACK, LEFT and RIGHT buttons; GPIO HIGH = pressed, and the ESP32 and remote
grounds are common. The sketch header is the authority; in summary:

| Button | Default GPIO | Notes |
|---|---|---|
| FWD / BACK | 1 / 2 | throttle; opposite buttons are never pressed together |
| LEFT / RIGHT | 42 / 41 | steer; a direction change always passes through a released slot |

The buttons are on/off, so the firmware turns `[throttle, steer]` into
presses in one of two modes (serial `mode`, saved on the board):

- **mod** (default): each button is held for a fraction of the time equal to
  the command (sigma-delta over `slot` ms slots, default 40), like feathering a
  button. This only works if the remote registers presses that short: tune
  `slot` on the real remote, wheels off. `limit` caps throttle (default 0.6).
- **binary**: pressed while |command| ≥ 1/3, released otherwise.

Which wire is which button is only knowable by looking. Over serial: `test`
presses FWD, BACK, LEFT, RIGHT in turn (wheels off); `pins <fwd> <back>
<left> <right>` fixes the order without reflashing. The setter refuses pins
that misbehave at boot or belong to the console, USB, flash or PSRAM.
`arena sim --buttons mod|binary` previews button control in the simulator.

**3. Real perception.** When the overhead-camera ArUco pipeline exists, have it
send pose frames (format below) and run:
```
python run_controller.py --model models/n1_catch --source udp --pose-port 9870 --esp <esp-ip>:8888
```
Dead-man stop is built in: if the pose feed stalls >0.3 s, an E-stop goes out
and histories reset until frames resume.

The `--source udp` loop (`pc_controller/realtime.py`) is frame-synchronous:
perception sends every camera frame, and the controller acts on the one nearest
each 100 ms tick the moment it arrives. It never acts on a pose whose
perception latency (`lat` in the frame) exceeds `--max-pose-age` (0.15 s).
`--esp auto` finds the cars by broadcast; `--start-disarmed` +
`--control-port` give the `arena go` / `arena halt` arm switch; `--telemetry`
feeds the dashboard. `--legacy-loop` keeps the old tick-per-frame loop for A/B
comparison. Tests: `python -m pytest tests -q`. `tools/sim_world.py` is a
simulated arena plus mock fleet (what `arena sim` runs).

## Wire formats

**PC → ESP command packet** (UDP :8888, one JSON per datagram):
```json
{"seq": 42, "t": 1720300000.123, "estop": false, "cmd": [[0.43, -0.10]]}
```
`cmd` = `[throttle, steer]` per car, normalized [-1, 1], +steer = left. The ESP
ignores `seq` ≤ last applied, goes neutral on `estop` or a 300 ms stream stall.

**Discovery / configuration** (current firmware): `{"probe": 1}` gets
`{"car", "fw", "mac", "failsafe", "up_s", "last_seq", "rssi", "ota", "act", "mode", "slot_ms", "thr_limit"}`
back with no effect on the buttons; `{"cfg": {"index": n}}` sets CAR_INDEX, but
only while the car is stopped. OTA updates (`arena flash --ota`) need
`OTA_PASSWORD` in `wifi_credentials.h` and are only serviced while stopped.

**ESP → PC delivery confirmation** (reply to every parseable packet):
```json
{"ack": 42, "car": 0, "applied": true, "slot": true, "rssi": -55}
```
`car` is the reporting board's CAR_INDEX (see below); `applied` is false when
the packet was dropped as stale; `slot` is false when the packet carried no
command at this car's index; `rssi` is the ESP's WiFi signal in dBm (> −70 good,
< −80 expect trouble). Verify the link any time with
`python tools/link_test.py --esp <ip>:8888` — it reports per-packet delivery,
round-trip time, loss, and signal strength (safe on a live car: zero throttle,
gentle steering sweep — but first runs wheels-off anyway).

## Multiple cars: CAR_INDEX

The policy is **centralised** — one forward pass emits `2N` values, the joint
action for the whole fleet. So the controller unicasts the **same** packet to
every car, and each ESP picks its own pair out of `cmd[]` using the CAR_INDEX
stored on that board:

| CAR_INDEX | drives | = policy slot | = marker_map |
|---|---|---|---|
| 0 | `cmd[0]` | pursuer 0 | `pursuer_ids[0]` |
| 1 | `cmd[1]` | pursuer 1 | `pursuer_ids[1]` |
| … | | | |

**Setup, per car.** Flash every board with the same sketch, then give each one a
different index over the serial monitor (persists in NVS, survives reflash):

```
index        → prints the current index
index 1      → sets this car to index 1 and saves
```

The index is printed loudly at boot and echoed in every ACK, because two cars
sharing an index is otherwise invisible: both obey the same command while one
slot goes undriven.

**Running the fleet.** Addresses go in CAR_INDEX order — the first address is
the car flashed as index 0:

```bash
python run_controller.py --model <N-pursuer model> --source udp \
    --esp 192.168.4.10:8888,192.168.4.11:8888,192.168.4.12:8888
```

Start-up refuses to run if you list more cars than the model drives, and warns
if you list fewer. `tools/link_test.py` takes the same list and reports **per
car**, so "which radio is worst" is answerable; an ACK from an index outside the
fleet is flagged explicitly.

**No hardware?** Run one mock per car on different ports:

```bash
python tools/mock_esp.py --index 0 --port 8888
python tools/mock_esp.py --index 1 --port 8889
python run_controller.py --model <2-pursuer model> --esp 127.0.0.1:8888,127.0.0.1:8889
```

**Safety note, new at N > 1.** The 300 ms failsafe is *per car*. One car can drop
to neutral while the others keep driving a formation that no longer exists — and
the policy has no concept of a stalled teammate, since a failsafed car still
reports a valid (stationary) pose. Whether a single-car failsafe should escalate
to a fleet-wide stop is an open design decision, not something the current code
does. A car that receives no command at its own index goes neutral **and** lets
the normal failsafe engage, so "no command" and "no link" look identical rather
than being two subtly different states.

Unicast, not broadcast: 802.11 broadcast frames get no MAC-layer retries and go
out at the lowest basic rate, which would trade reliability for airtime the link
does not need. Three cars at 10 Hz is ~5.5 kB/s and ~6% airtime.

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
  simple system-ID pass on the real car and compare. Cap the drive power for
  early runs regardless: the firmware's throttle `limit` ships at 0.6, and is
  the first thing to re-tune wheels-off.
- **Every policy here was trained on a proportional car.** The V3 car's
  remote is on/off, so what the policy asks for is approximated by button
  presses (modulated or binary). In `arena sim`, `n1_catch` caught about as
  often with buttons as without (one 60 s run per mode), but the sim's chase
  is easy and its steering instant. Measure on the real remote; a retrain on
  the real car's actuation is the real fix.
- **The policy uses reverse.** Observed in sim runs: `n1_catch` sometimes
  drives backwards (throttle −1.0) all the way to a capture. The firmware
  presses BACK directly; what to verify on the real car is that its remote
  reverses on a plain press (some RC cars brake first, then reverse on a
  second press) and that forward and reverse are roughly symmetric in speed.

## Provenance

Models frozen 2026-07-04 as V1 deployment candidates
(`artifacts/v1_deployment_candidates/` in the main repo; full numbers in
`V1_ACCEPTANCE_REPORT.md` there). ONNX exports verified against the torch
checkpoints to ≤1e-5 at export. Torch `.pt` originals stay in the main repo —
they are not needed here.
