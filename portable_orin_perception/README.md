# Portable Orin Perception (ROS 2 + ArUco)

Self-contained perception bundle for the EE198 pursuit project: a Jetson Orin
watches the arena through an overhead USB webcam, localizes every ArUco-tagged
car, and streams pose frames over UDP to the **unchanged** pursuit controller
(`portable_n1_controller/run_controller.py --source udp`). Copy this folder to
the Orin (git clone / scp / USB stick) and it brings itself up.

```
python3 selftest.py       # ROS not needed yet — must print SELFTEST PASSED
./setup_orin.sh           # one-time: installs ROS 2 (Humble/Jazzy by OS) + builds
./calibrate_arena.sh --device /dev/video0     # after placing corner markers
./run_perception.sh <controller-pc-ip>        # camera -> poses -> UDP :9870
```

New to ROS 2? Read `ROS2_LEARNING.md` alongside — every concept this bundle
uses is mapped to the exact file where it appears.

## Two ways to run: the fast path and the ROS pipeline

**`run_vision.py` (the deployment path, what `arena up` runs):** one process,
camera → luma-only JPEG decode → `core/localizer.py` (detector tuned to the
known tag size, tracking windows, plausibility gates) → a pose frame sent the
instant each frame is processed. No DDS image hop, no 10 Hz resampling timer,
newest-frame-only so overload drops frames instead of aging them. Capture
mode from `config/camera.yaml`; calibration from `arena scan`
(`scan_arena.py`). See [ARENA.md](../ARENA.md).

**`run_perception.sh` (ROS 2):** the same `Localizer` inside
`aruco_detector_node`, the same `camera.yaml` defaults. Use it for rviz,
rosbag, and learning ROS. Everything below about the ROS nodes still applies.

Tests for the fast path: `python -m pytest tests -q`.

## What's in here

| Path | What it is |
|---|---|
| `ros2_ws/src/hive_perception/hive_perception/core/` | ALL the math (homography, heading, staleness policy, wire JSON) as plain Python — no rclpy, unit-tested here (`selftest.py`) and in the main repo (`tests/test_orin_perception_core.py`). |
| `ros2_ws/src/hive_perception/hive_perception/*_node.py` | The ROS 2 nodes: `aruco_detector` (images → `/hive/vehicle_poses`), `pose_bridge` (poses → UDP JSON at 10 Hz), and `gst_camera` (optional hardware-JPEG camera, see below). Thin wrappers over `core/`. |
| `ros2_ws/src/hive_perception/launch/perception.launch.py` | Brings up camera (`usb_cam`) + detector + bridge together. |
| `config/` | `marker_map.yaml` (which id is which car), `arena_test_6ft.yaml` (calibration marker positions), `camera_info.yaml` (intrinsics — placeholder until calibrated), `arena_homography.yaml` (written by calibration, gitignored). |
| `markers/` | Printable tag sheets (PDF at exact physical size + PNG), regenerated from `marker_map.yaml` — currently ArUco `DICT_4X4_50`. Vehicles = 7 cm ids 0–3, calibration corners = 10 cm ids 10–13. |
| `vendored/` | Byte-identical copies of the controller's pose parser (`pc_controller/pose_stream.py` + its pose type) — the schema OWNER is `portable_n1_controller`; never edit these here. |
| `tools/` | `generate_tags.py` (regenerate sheets + print the pixel budget), `probe_orin_gpu.py` (measure what GPU paths this Jetson actually offers), `pose_frame_monitor.py` (watch the live pose stream from the PC without the controller). |
| `selftest.py` | Zero-hardware health check: the configured tag family detects through the subset dictionary, conventions hold, and our frames parse with the controller's own vendored parser. |

## The wire format (owned by portable_n1_controller — do not change here)

One JSON object per UDP datagram to `<controller-ip>:9870`:

```json
{"t": 12.34,
 "pursuers": [{"x": 1.0, "y": -2.0, "heading": 0.5}],
 "evader":   {"x": 3.0, "y": 1.0, "heading": -1.2}}
```

Meters/radians, **arena-centered** (origin at arena center, +x right, +y up),
heading 0 = +x and counter-clockwise positive, `t` = frame **capture** time
(the controller finite-differences velocities from it). Slot order everywhere:
pursuers in `marker_map.yaml` order, evader last.

## Dead-man semantics (do not soften)

If any tracked marker has been missing for more than 0.25 s, the bridge sends
**nothing at all**. The controller's own 0.3 s pose-stall dead-man then fires
an E-stop on the cars. Silence IS the E-stop signal — never "help" by sending
a guessed or stale pose past the hold cap.

## Bring-up ladder

**1. No hardware (any machine).** `pip install -r requirements.txt`, then
`python3 selftest.py` → SELFTEST PASSED.

**2. Orin, no camera.** `./setup_orin.sh`, then prove ROS works:
`ros2 run demo_nodes_cpp talker` in one terminal, `ros2 run demo_nodes_py
listener` in another (both after `source /opt/ros/<distro>/setup.bash`).

**3. Markers + camera.** Run `python3 tools/generate_tags.py` (it prints the
pixel budget — heed a `TOO SMALL` verdict before printing). Print
`markers/*.pdf` at 100% scale, ruler-check one,
tape vehicle markers on roofs (canonical TOP edge = the edge marked on the
sheet = toward the car's NOSE). Mount the webcam overhead (~2 m, whole arena
in frame). `ros2 run usb_cam usb_cam_node_exe` + `ros2 run rqt_image_view
rqt_image_view` to verify the picture. Then the one-time intrinsics
calibration (CALIBRATION.md step 1).

**4. Arena calibration.** Place corner markers 10–13, measure their centers,
edit `config/arena_test_6ft.yaml` to the measured coordinates, run
`./calibrate_arena.sh --device /dev/video0`. Residuals ≤ 2 cm.

**5. Poses flowing.** `./run_perception.sh <pc-ip>`; on the PC run
`python tools/pose_frame_monitor.py` (from this bundle) — hand-push a tagged
car and watch x/y/heading track reality at ~10 Hz. Sanity: `ros2 topic echo
/hive/vehicle_poses`, or rviz2 on the same topic (frame `arena`).

**Debug overlay (optional, any rung once the camera is up).** Launch with
`debug:=true` (e.g. `./run_perception.sh 127.0.0.1 debug:=true`) to publish an
annotated frame — every detected marker outlined with its id, role
(`P1`/`P2`/.../`EVADER`/`CORNER`), and (for tracked slots) live x/y/heading —
on `/hive/debug_image`. View it with `ros2 run rqt_image_view rqt_image_view`
(topic `/hive/debug_image`) over SSH `-X`/VNC, or a monitor on the Orin. Off
by default — costs an image copy + draw calls per frame, skip it once
everything is confirmed working.

**6. Controller in the loop, no car.** On the PC:
`python tools/mock_esp.py` (from portable_n1_controller) +
`python run_controller.py --model models/n1_catch --source udp --esp
127.0.0.1:8888`. Cover the camera lens: commands must go silent / E-stop
within ~0.3 s. This rung + rung 7 are the acceptance tests.
(You can dry-run this whole rung with zero hardware first —
`python tools/fake_perception.py --duration 6 --stall 1.2 --resume 3`
stands in for the Orin, using the same FrameBuilder as the real bridge, and
exercises the E-stop + re-arm path. Verified green 2026-07-14.)

**7. Latency + clock sync.** `chronyc tracking` on the Orin (offset a few ms),
then read `capture->arrival` off pose_frame_monitor: target ≤ 50 ms so the
whole camera→policy→radio loop stays inside its hard 100 ms budget.

**8. One real car** (wheels off first), per portable_n1_controller's README.

## Hard constraints to respect

- **Latency budget ≤ 0.1 s camera→policy→radio, total.** 0.2 s collapses
  capture performance (measured in training). Perception's share: keep
  camera→pose under ~50 ms and never buffer frames.
- **`t` is capture time, never send time.** Velocity estimation depends on it.
- **The homography is per-mount.** Any camera move/zoom/refocus invalidates
  `config/arena_homography.yaml` — re-run calibration (it takes 2 minutes).
- **Marker mounting = heading convention.** Canonical top edge toward the
  nose; sideways mounting reads as a 90° heading error the policy will act on.
- **Don't edit `vendored/`.** If the frame schema ever changes in
  portable_n1_controller, re-copy the files; `selftest.py` and the main repo's
  `tests/test_orin_perception_core.py` both exist to catch drift.
- **Policies are arena-specific.** Live capture demos at the 6 ft test arena
  need the main repo's arena-scale retraining first; perception itself is
  arena-agnostic once calibrated.
- **Resolution is capped by marker pixels, not by the lens.** At 640×480 the
  7 cm vehicle markers land around 24 px — right at the ArUco detection floor.
  Raising resolution to see more arena runs straight into the CPU ceiling
  below. See [HANDOFF.md](../HANDOFF.md).

## Performance: read this before optimizing

The camera runs at **640×480 @ 15 fps** for a reason, recorded in
`perception.launch.py`: 1280×720 pegged `aruco_detector` at ~150% CPU with
frames queueing 0.5–0.9 s stale, permanently tripping the 0.25 s dead-man even
with clean detections.

**The GPU cannot fix this, and it is worth knowing that before you try.**
Measured on the Orin Nano, 2026-08-21:

| Stage | 640×480 | 1280×720 | GPU path? |
|---|---|---|---|
| MJPEG decode (`mjpeg2rgb`) | 2.8 ms | 7.5 ms | yes — NVJPEG/VPI |
| `cvtColor` BGR→GRAY | 0.1 ms | 0.3 ms | yes, but ~1% of the frame |
| `detectMarkers` | 12.2 ms | 34.0 ms | **no — CPU-only** |

OpenCV's ArUco module has no CUDA implementation in any build, and it is ~80%
of the frame. Neither installed OpenCV (pip 4.10.0, distro 4.6.0) has CUDA at
all, and `onnxruntime` here is CPU-only. Policy inference is 0.050 ms/call —
too small for GPU offload to be anything but overhead.

What actually helps, roughly in order of cost:

- `jetson_clocks` — the board sits in `MAXN_SUPER` but with the `schedutil`
  governor at 1.19 of 1.73 GHz, and identical work varied ~60% run to run.
  Does not persist across reboot.
- Tuning `DetectorParameters` against the known marker size (measured
  1.16–1.48× on synthetic frames).
- Avoiding the CPU MJPEG decode.
- NVIDIA's cuAprilTag (`isaac_ros_apriltag`) is the one genuine GPU detector.

Numbers are from synthetic frames — treat them as relative weights. Real arena
clutter makes `detectMarkers` worse, not better.

**One correction to the paragraph above.** The 1280×720 experiment changed
resolution *and* frame rate together — the rejected config was 720p at **30**
fps. At 30 fps the budget is 33.3 ms/frame against ~41.8 ms of work, i.e. 125%
loaded, which is what produced the ~150% CPU and the ever-growing staleness. At
**15 fps** the same work is ~63% of one core, and with clocks pinned ~43%. The
controller only consumes 10 Hz. Re-measure before treating 720p as unreachable
— `tools/probe_orin_gpu.py` does exactly this.

## Tag family: ArUco today, AprilTag tag36h11 ready to switch on

`config/marker_map.yaml` currently says **`DICT_4X4_50`** — the tags physically
taped to the cars. The AprilTag tag36h11 migration is **complete in code and one
line away**, left opt-in because flipping it invalidates every printed sheet:
the detector stops seeing the old tags entirely, every pose frame is suppressed,
and the cars sit in dead-man neutral until new sheets are printed and the arena
is re-calibrated. Safe, but total — not something a `git pull` should do to a
working arena. The switch procedure is in `marker_map.yaml` itself.

The destination is tag36h11 because of CUDA: cuAprilTags is the only real GPU
fiducial detector, and no ArUco equivalent exists in any OpenCV build.

Three facts make the migration cheap, all verified in `test_core.py`:

1. **OpenCV ships the genuine 36h11 codebook**, so the CPU detector reads
   exactly the same printed sheets cuAprilTags will. Moving to the GPU later
   is a detector swap, not a reprint.
2. **Detection runs against a subset dictionary** holding only the 6–8 ids this
   arena prints, built bit-identically from the real codebook
   (`core/tag_family.py`). This matters enormously — the full 587-codeword book
   costs ~37 ms/frame at 720p because every candidate quad is matched against
   all 587 codes with 5-bit error correction. The subset costs 4.65 ms, i.e.
   **cheaper than the ArUco baseline it replaces** (5.99 ms):

   | detector | 1280×720 |
   |---|---|
   | ArUco DICT_4X4_50 | 5.99 ms |
   | AprilTag 36h11, full 587 codes | 37.34 ms |
   | AprilTag 36h11, 8-code subset | **4.65 ms** |

3. **Switching family is a config edit**, not a code change: set `dictionary:`
   in `config/marker_map.yaml`, re-run `tools/generate_tags.py`, reprint,
   re-run arena calibration. Reverting works the same way. The subset trick
   helps the current ArUco config too — 50 codes down to 6, measured
   5.38 → 4.79 ms at 720p — so it is a small speedup even before the switch.

### The trap the subset creates

A subset dictionary **renumbers ids**: `detectMarkers()` reports the ROW INDEX,
not the printed tag id. Print ids `[0,1,10,11,12,13]` and detection returns
`[0..5]`. Translating wrongly swaps vehicle identity — the evader driven as a
pursuer. `TagSet.to_real_id()` is the only sanctioned translation, it raises on
an out-of-range row rather than returning something plausible, and both the
detector and the calibrator derive their tag set from the same
`marker_map.tag_set()` so their indices cannot drift apart.

### The cost, and why it forces the resolution question

36h11 is 8 modules across against ArUco 4×4's 6, so at equal tag size and
resolution it needs ~1.33× more pixels. Measured detection rate against
on-screen tag size (rotated synthetic tags):

| tag px | ArUco 4×4 | 36h11 sharp | 36h11 + motion blur |
|---|---|---|---|
| 18 | 100% | 88% | 29% |
| 20 | 100% | 100% | 79% |
| 24 | 100% | 100% | 96% |
| 28 | 100% | 100% | 100% |

Blurred is the real case for a moving car, so **28 px is the number to design
against**. A 7 cm tag in the square 1.83 m arena, which must fit the image's
short side, is ~17-18 px at 640×480 and ~25 px at 1280×720 (the width-based
24 / 49 px quoted here before overstated it). `arena scan` measures it from the
real calibration. `tools/generate_tags.py` prints a verdict before every print
run, and it says **TOO SMALL at 640×480**.

So the family switch and the resolution increase are one decision, not two.
That is affordable precisely because the subset dictionary and the hardware
decode below pay for it.

## GPU paths, and how to establish them

Run `python3 tools/probe_orin_gpu.py` on the Orin first — it inventories CUDA,
VPI, NVJPEG, GStreamer elements and Isaac ROS, then benchmarks the decode and
detection paths. Run it **twice, before and after `sudo jetson_clocks`**; any
number from an unpinned board understates it by ~1.45×.

**Hardware JPEG decode (works today, no Isaac ROS).** `gst_camera` replaces
`usb_cam` and decodes MJPEG on the Orin's NVJPG engine via GStreamer
`nvjpegdec`, freeing the 2.8 ms (640×480) / 7.5 ms (720p) that CPU decode
costs. Same topics and message types, so nothing downstream changes:

```bash
./run_perception.sh 127.0.0.1 camera_backend:=gst image_width:=1280 image_height:=720
```

It falls back to CPU `jpegdec` if `nvjpegdec` is missing and says which it
used. Caveat: `appsink` returns host buffers, so this removes the decode cost,
not the host↔device copy.

**cuAprilTags (needs Isaac ROS).** `isaac_ros_apriltag` moves detection itself
onto the GPU. The probe reports whether it is installed and whether this
board's L4T/Ubuntu is a supported combination — check that before planning
around it, and do not assume Jazzy is supported. Because the printed sheets are
already genuine tag36h11, adopting it changes the detector node only.

## Provenance

Created 2026-07-13 in the main AI Training repo (source of truth). Wire format
and dead-man behavior verified against `portable_n1_controller` (hardware-
tested 2026-07-06/07). Vendored parser copied byte-identical the same day.
