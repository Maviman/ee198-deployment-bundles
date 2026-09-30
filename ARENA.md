# Arena runbook: two Orins, one camera, one command per step

This is how the system runs now. One overhead camera watches the arena. One
Orin turns camera frames into car poses, the other turns poses into commands
for the cars, and a single CLI (`./arena`) drives every step from any
terminal.

```
                     VISION ORIN (camera plugged in; runs nothing else)
  webcam ── USB ──►  run_vision.py
  MJPEG 1280x720@30    newest frame only ─► luma-only JPEG decode ─► tag detection
                       (tracking windows) ─► plausibility gates ─► pose frame, sent at once
                                  │
                                  │ UDP :9870 pose frames (every frame)
                                  │ UDP :9871 telemetry        ┌── Ethernet cable ──┐
                                  ▼                                (recommended)
                     CONTROL ORIN (on the cars' WiFi; the operator box)
                       run_controller.py    one tick per 100 ms, acting on the
                                            frame that arrives nearest each tick
                       hub.py               telemetry, alerts, run logs, dashboard :8080
                                  │
                                  │ UDP :8888 commands (DSCP EF = WiFi voice queue)
                                  ▼
                     ESP32 on each car ── L298N (drive) + servo (steer)
```

## Why the work is split this way

| | vision Orin | control Orin |
|---|---|---|
| runs | `run_vision.py` only | controller, hub/dashboard, operator shell |
| hard real-time? | yes: every frame, 30 fps | yes, but tiny (0.2 ms inference per tick) |
| CPU | ~1 core at 720p30 (decode + detect) | a few % |
| network | camera USB + wire to control | WiFi to cars + wire to vision |

Detection is the only heavy, latency-critical job, so it gets a machine to
itself. Nothing the operator does (dashboard, SSH, logs, a browser) can steal
CPU from it and add jitter to a pose. The controller is tiny but owns the
radio, so it sits on the box with the cars' WiFi. The dashboard lives there too,
and the camera preview it shows costs the vision Orin nothing: the camera's own
JPEG bytes are forwarded, never re-encoded, and the outlines are drawn in the
browser.

**One Orin also works** (`arena init --single`). Same software; poses go over
loopback. You lose the isolation, not the function.

## What changed, and why it is faster

| where | before | now |
|---|---|---|
| frame → pose | usb_cam decodes RGB → ROS DDS copies a 0.9–2.7 MB image to another process → RGB→BGR→gray | one process: raw MJPEG bytes → **luma-only decode** (1.9× cheaper at 720p) |
| detection | OpenCV defaults over the full frame every frame | detector **tuned to the known tag size**, then **tracking windows** around each car. 720p, 3 tags, one thread: **1.41 ms vs 19.9 ms** (14×) with zero missed detections, synthetic frames |
| pose → controller | resampled on a fixed **10 Hz timer** (poses aged up to a frame interval before sending) | sent **the instant each frame is processed** |
| controller tick | whenever a timer-resampled frame arrived | **frame-synchronous**: acts on the frame nearest each 100 ms tick the moment it lands; drains any backlog; never acts on a pose older than 150 ms |
| capture | 640×480 @ 15 | **1280×720 @ 30** (a multiple of the 10 Hz tick) |
| overload | queue grows → 0.5–0.9 s stale poses → permanent dead-man | newest-frame-only everywhere: overload **drops frames**, latency stays bounded |
| wrong poses | passed straight through | **gated**: outside the arena, faster than a car can drive, wrong tag size, or a duplicate id is rejected and becomes a (safe) dropout |
| camera | auto everything | exposure-priority **off** (auto-exposure can't silently halve the fps), autofocus **off**, anti-flicker on, optional short manual exposure (`--tune-exposure`) |
| radio | best-effort | commands marked DSCP EF (WiFi voice queue); control Orin WiFi power-save **off** (`arena tune`) |

**The pixel budget was wrong in the old docs.** A square arena fits the image's
*short* side, so at 640×480 the 7 cm car tags are **~17–18 px**, not 24 px:
below the detection floor for both tag families. In rendered tests, ArUco missed
1% and AprilTag 36h11 missed **39–63%** at 640×480; at 1280×720 (tags ~25 px)
both missed nothing. That is the main reason the default is now 720p, and a
likely cause of dead-man stops seen in testing. `arena scan` measures the
real number from your calibration and prints a verdict.

### Measured numbers (this session, dev PC, closed-loop simulation)

The whole pipeline running on one x86 PC, with a *synthetic* camera (the
renderer itself costs ~15 ms of the "camera" stage):

| stage | p50 |
|---|---|
| decode (720p, luma only) | 2.1 ms |
| detect + localize (tracking windows) | 1.0 ms |
| controller: frame in → command out | 0.3 ms (policy 0.18 ms) |
| camera → command, end to end | ~20 ms, of which ~15 ms is the synthetic renderer |
| tick rate | 10.0 Hz from a 30 fps stream |

**Orin numbers are not measured yet.** The ratios above transfer; the absolute
values will be ~3–5× slower per stage on the Orin's ARM cores (less with
`arena tune`). The dashboard and `arena monitor` show the real per-stage
figures live. Record them the first session.

## First-time setup

1. **Network.** Both Orins on your LAN/WiFi for SSH. The control Orin must be
   on the cars' WiFi. Strongly recommended: a direct Ethernet cable between the
   Orins with static addresses, so poses never share airtime with the cars:
   ```bash
   # vision Orin (check the interface name with: ip link)
   sudo nmcli con add type ethernet ifname eth0 con-name arena-link ipv4.method manual ipv4.addresses 10.42.0.1/24
   # control Orin
   sudo nmcli con add type ethernet ifname eth0 con-name arena-link ipv4.method manual ipv4.addresses 10.42.0.2/24
   ```
2. **Tell `arena` which Orin is which** (writes `deploy/arena.local.conf`, gitignored):
   ```bash
   ./arena init --vision jordan@orin-vision.local --control jordan@orin-control.local \
                --link-ip 10.42.0.2 --vision-link-ip 10.42.0.1
   ```
   `arena` uses key-based SSH to each Orin from wherever you run it:
   `ssh-copy-id jordan@<host>` once per machine. To run `arena` on the control
   Orin itself, give it a key to the vision Orin the same way.
3. **Install** (asks for the sudo password on each Orin): `./arena setup`
4. **Tune** (after every reboot: neither setting persists): `./arena tune`
   pins the CPU clocks (~1.45× faster detection) and turns WiFi power-save
   off (power-save adds 100 ms+ naps to the radio).
5. **Cars.** Flash each car once over USB, then set its index if you have
   more than one:
   ```bash
   ./arena flash --usb               # the port labelled UART; one car at a time
   ./arena cars                      # lists every car that answers, with its CAR_INDEX
   ./arena cars --index 192.168.1.41 1
   ```
   Add `#define OTA_PASSWORD "..."` to `wifi_credentials.h` before that first
   USB flash, and every later firmware update is `./arena flash --ota` (all cars,
   over WiFi, only while they are stopped).

## Every session

```bash
./arena sync           # both Orins onto this commit + selftests (push first)
./arena scan           # calibrate from the corner tags; prints the tag-pixel verdict
./arena up             # vision + controller + dashboard; cars DISARMED
                       #   -> open the dashboard URL it prints; push the cars by hand,
                       #      check poses track, check every car shows "ok"
./arena go             # ARM: the policy drives
./arena halt           # E-stop now (services keep running; `go` re-arms)
./arena stop           # E-stop + shut everything down
```

`./arena scan --check` re-verifies a saved calibration in seconds. It measures
how far the corner tags have moved and fails if the camera was bumped. The
vision service also watches the corner tags once a second while running, and
the dashboard raises **"corner tags moved"** if they drift.

`./arena scan --tune-exposure` finds the shortest exposure that is bright
enough for the room and saves it (`config/camera.local.yaml`). Shorter exposure
means less motion blur on a moving tag.

## Watching a run

- **Browser:** `http://<control-orin>:8080`. Live camera with detection
  outlines, top-down arena with trails, camera→command latency broken into
  stages against the 100 ms budget, per-car throttle/steer/radio round trip/
  signal, vision health, alerts, and a **STOP** button (anyone watching can
  stop the cars; arming is terminal-only).
- **Terminal:** `./arena monitor`. Same data, works over SSH.
- **Logs:** `./arena logs vision|controller|hub [-f]`. Every run's telemetry
  is also saved on the control Orin in `~/.arena/runs/<timestamp>/telemetry.jsonl`.
- **Health check:** `./arena status`. Code version on each Orin vs this
  checkout, services, clocks, temperature, camera, calibration, WiFi
  power-save, and current alerts.

Alerts you may see, and what to do:

| alert | meaning | fix |
|---|---|---|
| camera running N fps of 30 | light too low with exposure priority, USB bandwidth, or CPU | `scan --tune-exposure`; camera on its own USB port; `arena tune` |
| P1 seen in only N% of frames | tag too small, blurred, glare, or partly off-arena | `arena scan` tag verdict; bigger tag / lower camera / less glare |
| corner tags moved N px | camera bumped since calibration: **every pose is off** | `arena scan` |
| rejected implausible detections | gates caught a phantom or a teleport (a car lifted and put down) | occasional is normal; constant = check for a spare printed tag in view |
| camera→command p95 over budget | the pipeline is overloaded | `arena tune`; check `status` for thermal throttling |
| car N silent / radio round trip high / weak WiFi | the car is off, out of range, or browning out | battery, antenna placement, `link_test.py` |
| clocks not pinned | the board is running ~30% slow | `arena tune` (after every reboot) |

## Safety chain (unchanged in spirit, tightened in places)

Every hop fails to *stopped*:

1. A car tag unseen or rejected for > 0.25 s → the vision service sends
   **nothing** (a held pose is never guessed past that).
2. No usable pose for 0.3 s → the controller E-stops every car. "Usable" now
   also means **fresh**: a pose that took > 150 ms to arrive is never acted on.
3. No command for 0.3 s → each ESP32 goes neutral on its own.
4. **Disarmed is the default.** `arena up` starts with every car receiving
   E-stop packets (which still get ACKed, so the dashboard shows radio health
   before anything moves). Only `arena go` arms, and only from a shell on the
   control Orin: the arm port listens on localhost. `arena halt`, the dashboard
   STOP, Ctrl-C and `arena stop` all E-stop on the next packet.
5. Crashes: the vision service and controller run under a supervisor that
   restarts them in about a second (cars sit in failsafe meanwhile). After 5
   crashes in 60 s it gives up and says why (`arena logs`).
6. Firmware updates can only start on a car that is stopped.

## Rehearsal with no hardware

```bash
./arena sim            # opens the dashboard; Ctrl-C to stop
```

Runs everything on this machine through the production code paths. A
simulated arena (the training env's car physics at the 6 ft scale, with mock
ESP32s that obey the firmware's rules) renders real tag images for a synthetic
camera. The real vision service detects them, the real controller drives the
simulated car, and the real hub shows it all. Use it to learn the commands,
demo the dashboard, and check a change before touching the cars. (Episode
resets teleport the cars, so expect a few "jump" rejections; that is the gate
working.)

## Reference

| file | what |
|---|---|
| `arena`, `arena.cmd`, `deploy/arena.py` | the CLI (bash/Git Bash, Windows cmd/PowerShell) |
| `deploy/arena.conf` / `arena.local.conf` | topology defaults / your site (from `arena init`) |
| `deploy/hub.py`, `deploy/dashboard.html` | telemetry hub + dashboard (stdlib only) |
| `deploy/supervise.sh` | restart-on-crash wrapper used by `arena up` |
| `portable_orin_perception/run_vision.py` | the vision service (`--synthetic` for no camera) |
| `portable_orin_perception/scan_arena.py` | `arena scan` |
| `portable_orin_perception/config/camera.yaml` | THE capture mode, read by every consumer |
| `.../hive_perception/core/localizer.py` | the one detection engine (fast path, ROS node, scan) |
| `portable_n1_controller/pc_controller/realtime.py` | the frame-synchronous control loop |
| `portable_n1_controller/tools/sim_world.py` | simulated arena + mock fleet (`arena sim`) |

Ports: 9870 poses, 9871 telemetry, 9872 arm/disarm (localhost only), 8090
camera preview (vision Orin), 8080 dashboard (control Orin), 8888 cars, 3232
OTA. All configurable in `deploy/arena.conf`.

The ROS 2 pipeline (`run_perception.sh`) still works and now shares the same
detection engine and camera config. Use it for rviz, rosbag and learning ROS.
The arena commands use the fast path.

## Still open

- **Real Orin numbers.** Everything above was measured on a PC. First
  session: `arena tune`, `arena up`, and note the dashboard's per-stage
  latencies and the vision process CPU%.
- **The arena-scale retrain** (unchanged): the frozen `n1_*` policies assume
  the 14 × 10 m training arena. The dashboard's dashed ring (the policy's
  0.90 m capture radius, drawn inside a 1.83 m arena) makes that visible.
- **Camera mode.** 1280×720 MJPEG at 30 fps is assumed. `arena scan` lists
  what the camera actually offers if it cannot do that.
- **cuAprilTags / hardware decode** remain possible later; with tracking
  windows the CPU cost they would remove is now ~2 ms per frame.
