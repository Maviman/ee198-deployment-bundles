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
| `models/c37_commit3_ft_g997_arena183/`, `models/c37_commit3_ft_g997/` | The 3-pursuer hive role commander (manifest `"runtime": "hive_commander_v1"`), scaled to the 1.83 m arena and as trained (6 m), run by `pc_controller/runtimes/hive_commander_v1.py`; `golden_vectors.json` pins each in `selftest.py`. Read their READMEs for the transfer gaps. |
| `pc_controller/loops.py` | Picks the loop a model needs from its manifest: `PortableLoop` for flat policies, `HiveLoop` for role commanders. |
| `single_pursuer/`, `vehicle_dynamics.py`, `configs/` | Vendored copies of the exact observation/dynamics code the policies were trained against. Used ONLY to build observation vectors — never edit these here. |
| `controller_runtime/` | Vendored pose ingest, velocity estimation, latency tracking, observation adapter. |
| `pc_controller/` | New PC-side code: ONNX inference, the 10 Hz control loop, the ESP command link (UDP, or a UART cable: `serial_link.py`), the perception UDP input. |
| `run_controller.py` | Main entry point (see below). |
| `selftest.py` + `reference_vectors.json` | Golden-vector health check: proves the vendored contract + models on this machine reproduce the outputs frozen at bundle creation. |
| `tools/mock_esp.py` | Fake ESP32 for testing the full PC side with zero hardware (UDP, or `--serial` for a car on a UART cable). |
| `esp32/esp32_receiver/` | Firmware for the ESP32 on each car (WiFi UDP **or a UART cable** → presses the FWD/BACK/LEFT/RIGHT buttons of the car's RC remote, with failsafe). `button_mod.h` turns a command into presses and `uart_frame.h` splits the UART into command frames; `esp32/test_host/` tests both on a PC. |

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
The same rehearsal over the UART transport (needs pyserial; the mock makes a
pseudo-terminal and the controller opens it like a real port):
```
python tools/mock_esp.py --serial --link /tmp/mock-esp0
python run_controller.py --model models/n1_catch --esp serial:/tmp/mock-esp0
```

**2. Real ESP32, wheels off the ground.** Put the WiFi credentials in
`esp32/esp32_receiver/wifi_credentials.h`, flash, read the IP off the serial
monitor, then re-run step 1 with `--esp <esp-ip>:8888`. The sim episode now
presses the real remote's buttons. Unplug the network mid-run to watch the
300 ms failsafe release every button. With the car on a UART cable instead of
WiFi, flash the same firmware and use `--esp serial:<device>` (see below);
`python tools/link_test.py --esp serial:<device>` is the cable version of the
link check.

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

- **binary** (default): pressed while |command| ≥ 1/3, released otherwise.
  This is the mode that matches the V3 car: measured 2026-10-02 (wheels on
  blocks, the user watching), its remote is effectively on/off per axis. There
  is no drive-power cap in this mode: any throttle ≥ 1/3 is the car's full
  speed, and `limit` does not apply.
- **mod**: each button is held for a fraction of the time equal to the command
  (sigma-delta over `slot` ms slots, default 40), like feathering a button;
  `limit` caps throttle (default 0.6). On the V3 car it does not work: 50% duty
  at 40, 80 and 150 ms slots spun the wheels at the same speed as a solid hold.
  It stays for a remote that does feather.

The mode is saved on the board and survives a reflash, so a board that ever
stored one keeps it; the default applies only to a board that never did. Type
`mode` on the console to see which one you have before trusting either.

Which wire is which button is only knowable by looking. Over serial: `test`
presses FWD, BACK, LEFT, RIGHT in turn (wheels off); `pins <fwd> <back>
<left> <right>` fixes the order without reflashing. The setter refuses pins
that misbehave at boot or belong to the console, USB, flash or PSRAM.
`arena sim --buttons binary|mod` previews button control in the simulator; `binary` is the one that matches the real car.

**3. Real perception.** When the overhead-camera ArUco pipeline exists, have it
send pose frames (format below) and run:
```
python run_controller.py --model models/n1_catch --source udp --pose-port 9870 --esp <esp-ip>:8888
```
Dead-man stop is built in: if the pose feed stalls >0.3 s, an E-stop goes out
and histories reset until frames resume.

**Hold zone (`--hold-radius M`, env `ARENA_HOLD_RADIUS=M`, or the number in `~/.arena/hold_radius`, which a plain `arena up` keeps).** The policy only
*reports* a capture radius; nothing in the controller ends a run when a car gets
there, so a car drives on and can overshoot the evader. With a hold zone, a car
within `M` metres of the evader is commanded `[0, 0]` (every button released)
while staying armed, so this is **not** an E-stop and **not** a capture: the link
and the firmware's dead-man keep being fed, poses keep being tracked and the
policy is still asked on every tick. The car drives again once the evader is
`M + 0.1` m away (the 0.1 m is hysteresis against pose noise), so it follows an
evader that moves off. Each car is judged on its own distance. The log says
`car 0 HOLD ...` and `car 0 RESUME ...` once per change, and the telemetry
carries `hold_radius` and `hold`. Size `M` for the run-on: an on/off car keeps
rolling after the buttons release (about 0.4 s measured on the V3 car, around
0.5 m at its 1.4 m/s), so the car stops `M` minus that from the evader. The
policy was trained to end an episode at its capture radius (0.9 m for `n1_catch`),
so a smaller `M` lets it drive in a region it never saw; it is only overridden
once it gets inside `M`. A model with an action history (k > 0) still records
its own un-held output there. Not available with `--legacy-loop` or `--source sim`.

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
if you list fewer. A car on a UART cable goes in the same list as `serial:<device>`. `tools/link_test.py` takes the same list and reports **per
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

## UART transport (the car on a cable)

The ESP32 that presses the remote's buttons can be wired to the control Orin
instead of joining WiFi: no radio hop, no airtime to share. The commands do not
change; only the pipe does.

```bash
python run_controller.py --model models/n1_catch --source udp --esp serial:/dev/ttyTHS1
python run_controller.py --model models/n1_catch --source udp \
    --esp serial:/dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_0001-if00-port0@115200
```

`serial:<device>[@baud]` goes anywhere an `ip:port` can, including in a fleet
list, where position is still the CAR_INDEX (UDP and serial cars can be mixed;
`--esp auto` is UDP discovery and cannot be combined with it). `arena up --esp
serial:/dev/ttyTHS1` passes it straight through. pyserial is imported only when
a port is opened; `pip install -r requirements.txt` provides it. The controller
user needs access to the device (`dialout` group).

**Wiring.** Either a USB cable from the Orin to the DevKitC's port labelled
"UART" (the CP2102N bridge: `/dev/ttyUSB0`, best addressed by its stable
`/dev/serial/by-id/...` name), or a header UART wired to the ESP32's pins
(`/dev/ttyTHS1` is UART1 on the Orin's 40-pin header, pins 8 TXD / 10 RXD on the
dev kit; 3.3 V logic on both sides). Cross TX and RX (Orin TX to ESP32 RX =
GPIO44, Orin RX to ESP32 TX = GPIO43) and join the grounds. Both ways are the
board's UART0, the serial console's UART, at 115200 baud. The bridge and the
header pins are the same two nets, so drive them from one source, not both.
*(The USB route is the one in use on the bench Orin: `/dev/ttyUSB0`, by its
stable `/dev/serial/by-id/usb-Silicon_Labs_CP2102N_...` path. The header-UART
route has not been tried with an ESP32.)*

**Wire format: the same bytes, one per line.** A command packet is the JSON
object of the UDP section above, built by the same code, and the bytes written
to the port are the bytes a datagram would carry, including the trailing
newline the datagram already had. That newline is the frame boundary; after any
damage both ends resynchronise on the next one. The board answers on the same
wire, one JSON line each: the ACK (`"rssi": null`, there is no radio) and the
reply to `{"probe": 1}`; its ordinary log lines (`FAILSAFE: ...`, `#12 E-STOP
...`) share the stream and are shown in the controller's log as `car N
(serial:...): ...`. There is no checksum: a damaged frame nearly always stops
being JSON and is dropped, and a dropped frame costs one tick, as a lost
datagram does. A damaged number that still parses is clamped to [-1, 1] by the
firmware, for at most one 100 ms tick; in the default binary mode there is no
power cap, so it can turn a non-press into a full press for that tick.

**Who is on the other end.** `/dev/ttyUSB0` and `/dev/ttyUSB1` swap places
across reboots, and swapped ports mean the wrong physical car obeys each
command. So the controller does not write a command to a port until the board
on it has answered a probe with the CAR_INDEX its `--esp` position promises,
and refuses to start otherwise, naming the port and what to fix (wrong index,
wiring, firmware without UART support: the firmware string ends in `-uart`).
The check repeats after every reconnect.

**Failsafe: unchanged, and still the board's.** The firmware reads frames from
the UART and UDP through one handler and one 300 ms dead-man: a command that
parsed refreshes it, nothing else does. Frames stop (controller killed, cable
pulled, port wedged) and every button is released after 300 ms, exactly as for
UDP. Released buttons are not a stopped car: measured on blocks (2026-10-02, motor
sound against pin timestamps), the V3 car keeps going for about 0.35-0.45 s after
a release (roughly 0.1-0.2 s of dead time, then spin-down), so a stall costs the
0.3 s dead-man plus that run-on; probably less on the floor. The controller adds nothing that could keep a car moving: a failed write or
read marks that car's link down (logged, other cars keep their ticks), and it
is reopened and re-verified every second until it works. Shutdown writes an
E-stop and waits for it to leave the wire before closing.

**The console shares the wire.** A line starting with `{` is a command frame;
anything else is a console command (`index`, `mode`, `slot`, `limit`, `pins`,
`test`). The controller holds the port exclusively, so stop it before using the
console. Because replies and logs share the wire, `tools/drivetrain_test.py
--serial` (a second reader) cannot run on the same port as a controller.

**Opening the port can reset the board.** USB-UART adapters assert DTR/RTS on
open, which on ESP32 dev boards drive EN and IO0. Measured on the DevKitC-1 on
`/dev/ttyUSB0` (2026-10-02): opening the port with pyserial's DTR and RTS forced
low reset the board every time, while opening it with pyserial's defaults did not
(two trials). The kernel raises both lines together, which that circuit ignores;
clearing DTR first leaves RTS alone asserted, which holds EN low. The controller
therefore does not touch those lines. If a different board still resets,
nothing is lost: every button is released while it boots (the 100 k
pull-downs), nothing is sent until the probe is answered, and the probe is
repeated for up to 4 s. A header UART wired to the board's TX/RX pins has no
line connected to EN or IO0, so it cannot reset it. **Mind what is running on
the board**: a reset restarts whatever it was doing, and firmware without UART
command support (anything not flashed from this tree) never answers the probe,
so the controller exits with a message after 4 s.

**WiFi is optional on the car.** The firmware connects to WiFi in the
background, so a car with no access point in range still boots and takes UART
commands from its first loop; UDP, discovery and OTA start when WiFi first
connects.

Tools: `tools/link_test.py` and `tools/motor_test.py` take `--esp serial:...`.
Tests: `tests/test_serial_link.py` (framing, verification, reconnect, fleets,
over real pseudo-terminals), `tests/test_realtime.py` (every real-time
behaviour, over both transports), `tests/test_firmware_framing.py` (the
controller's bytes through the firmware's own framer),
`tests/test_firmware_host.py` (the real `esp32_receiver.ino` compiled on a PC
against Arduino stubs and fed the controller's bytes: dead-man timing to the
millisecond, one timer for both transports, replies; needs a C++ compiler and
the ArduinoJson headers one `pio run` fetches) and the C++ checks in
`esp32/test_host/`.

**Status.** Everything above runs without hardware, including the firmware's
logic and the real-time loop over a pseudo-terminal, and the sketch builds for
the ESP32-S3 (`pio run`). On the bench (2026-10-02, USB, car on blocks) the four
button wires were checked with a scratch sketch and opening the port was
measured not to reset the board, but THIS firmware and transport have not yet
run together on hardware. The first wheels-off session with them should confirm
the probe is answered, the `test` console command presses the right buttons, and
the round trip in `link_test.py` (~10 ms of it is the command's time on the wire
at 115200 baud); then tune `slot` for the car (the shortest press its remote registers is
unmeasured and the 40 ms default is unproven; `limit` only matters in `mod` mode).

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
  simple system-ID pass on the real car and compare. The default binary mode
  has no firmware drive-power cap (`limit` exists only in `mod` mode, which the V3
  car ignores): a command ≥ 1/3 is the car's full speed. For early runs cap the
  risk instead: car on blocks, then a clear floor with the E-stop in reach
  (`arena halt`), remembering it runs on for ~0.4 s after any stop.
- **Every policy here was trained on a proportional car.** The V3 car's
  remote is on/off, and measured on the car (2026-10-02) modulating the presses
  does not give partial speed, so what the policy asks for is approximated by
  binary presses at the 1/3 cut. The car also lags its buttons: about 120 ms
  from a press to motor sound, and roughly 0.3-0.5 s to half level after a
  release (free-spinning on blocks; probably shorter on the floor). In `arena
  sim`, `n1_catch` caught about as often with buttons as without (one 60 s run
  per mode, modulated and binary), but the sim's chase is easy, its steering
  instant and it has none of that lag. Measure on the real car; a retrain on the
  real car's actuation is the real fix.
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
