# Handoff — planned camera work

Written 2026-08-21. Two pieces of camera work are coming. Neither is started.
This file records the constraints that are already known — several of them
measured, and several of them the kind that quietly invalidate an obvious-looking
design — so whoever picks these up doesn't rediscover them the hard way.

Current state of the repo at time of writing: the V2 drivetrain rework just
landed (7.4 V brushed motor on the L298N, steering servo direct off the ESP32).
See `portable_n1_controller/esp32/esp32_receiver/esp32_receiver.ino`. That work
matters to item 1 below, because the ESP32's 5 V rail and its control loop are
now both spoken for.

---

## Item 1 — Forward-facing camera on the car

**Goal:** a camera on the car looking forward, observing what's in front of it
and relaying that back to the Jetson.

**Intended part:** OV2640 2 MP — the Arducam module at
`amazon.com/dp/B012UXNDOY` ("Arducam Module 2 Megapixels for Arduino Mega2560").

This is already on the books: `portable_n1_controller/TEST_REPORT.md` lists it
under "Not yet tested," including the command-link contention test.

### Check the interface before ordering anything else

The linked Arducam Mini is an **SPI + I2C** module (SPI for image data out of an
onboard FIFO, I2C/SCCB for sensor configuration). It is *not* the DVP-parallel
OV2640 used on ESP32-CAM boards, and the two are not interchangeable in
software or wiring.

This matters more than it looks:

- The ESP32-S3 has a dedicated **LCD_CAM** peripheral that ingests DVP parallel
  camera data at high rate with DMA. The Arducam SPI module **cannot use it** —
  data comes over ordinary SPI instead, so throughput is bounded by SPI clock
  and FIFO reads rather than by a purpose-built camera bus. Expect low
  single-digit fps at full 2 MP JPEG, not video.
- Driver path is Arducam's own Arduino library (`ArduCAM`), not
  `esp32-camera`/`esp_camera_*`. Most ESP32-camera tutorials online assume the
  DVP module and will not apply.
- If the requirement turns out to be *streaming* rather than *occasional
  stills*, price out a DVP-interface OV2640/OV5640 module on the S3's LCD_CAM
  before committing to the SPI part. Decide "stills or stream" first — it
  selects the hardware.

### Hard constraints this must not break

1. **The control loop has priority, absolutely.** `esp32_receiver.ino`'s
   `loop()` is a tight poll of `udp.parsePacket()`, and the car goes to neutral
   if no command packet is applied for 300 ms (`FAILSAFE_TIMEOUT_MS`). A
   blocking SPI read of a 2 MP JPEG will stall that poll and trip the failsafe.
   A camera that makes the car randomly stop is worse than no camera.
   Options, roughly in increasing order of effort:
   - stills on demand only, captured while the car is commanded to neutral;
   - a FreeRTOS task on the S3's second core, with the control loop pinned to
     the other core and no shared blocking resource between them;
   - a second ESP32 dedicated to the camera, sharing nothing but the battery.
     Cleanest isolation, and the option to take if the schedule is tight.
2. **WiFi bandwidth is shared with the control link.** Command RTT is currently
   ~5–9 ms one-way with ~0.5% of packets at 100+ ms
   (`TEST_REPORT.md`), inside a ≤ 0.1 s end-to-end budget that the policy was
   trained against. Image data on the same radio will contend with that.
   `tools/link_test.py` run *while streaming* is the acceptance test, and it
   already exists — use it.
3. **The ESP32's 5 V rail is already carrying the steering servo,** which can
   pull 0.5–1 A when slewing or stalled and is the current prime suspect for
   any brown-out. Adding a camera adds load to a rail that is already the
   weak point. Budget it, and consider powering the camera separately.
4. **Payload.** The V2 chassis is a toy-grade 1:16 platform (~1.5 lb) already
   carrying custom electronics. Weight is not free.

### Open questions to settle before writing code

- What does the Jetson actually *do* with the forward view? Recording for later
  analysis, a human-facing demo stream, or an input to control? Only the last
  one imposes a latency requirement — and the frozen policies take pose vectors
  only, so a forward camera **cannot** feed the current control loop without a
  retrain. Assume telemetry/demo until someone says otherwise.
- Stills or stream? (Selects the hardware — see above.)
- What frame rate is actually needed? This is the single biggest driver of
  every constraint above.

---

## Item 2 — Overhead camera FOV + resolution tooling

> **STATUS 2026-09-29: two corrections and one item done.** See ARENA.md.
>
> - **The pixel table below overstates tag size for this arena.** It assumes
>   the 1.83 m arena spans the image *width*; a square arena has to fit the
>   *short* side. Real figures: ~17–18 px at 640×480 and ~25 px at 1280×720.
>   Measured in rendered tests: at 640×480 AprilTag missed 39–63% of sightings
>   and ArUco ~1%; at 1280×720 both missed none. The default capture is now
>   1280×720 @ 30. `arena scan` reports the true tag size from the fitted
>   homography.
> - **The CPU ceiling is gone at 720p.** The fast path (run_vision.py)
>   decodes luma only and detects inside tracking windows around each car:
>   1.41 ms/frame vs 19.9 ms for OpenCV's defaults on the full frame (720p,
>   one thread, synthetic).
> - **"One config file holding the camera mode" is done:**
>   `portable_orin_perception/config/camera.yaml`, read by the fast path, the
>   ROS launch file, the calibrator, the debug tools and preflight.

> **STATUS 2026-08-22 — partly executed; read this before the section below.**
> The AprilTag tag36h11 path and the CUDA groundwork are in, with the family
> switch itself left opt-in in `config/marker_map.yaml` (flipping it invalidates
> every printed sheet, so it is not something a pull should do). What changed,
> and what it invalidates here:
>
> - **`detectMarkers` is no longer the ceiling this section assumes.** Detection
>   now runs against a *subset* dictionary of only the printed ids
>   (`core/tag_family.py`). At 1280×720: ArUco 5.99 ms, full-codebook 36h11
>   37.34 ms, **8-code subset 4.65 ms** — cheaper than the ArUco baseline. The
>   CPU argument against higher resolution is substantially weaker than written
>   below.
> - **The "1280×720 doesn't fit" finding conflated two variables.** That config
>   was 720p at **30** fps: 33.3 ms budget vs ~41.8 ms work = 125% loaded, which
>   is the ~150% CPU and the growing staleness. At 15 fps it is ~63% of a core,
>   ~43% with clocks pinned. The controller only needs 10 Hz.
> - **The pixel floor moved the wrong way and is now measured, not assumed.**
>   36h11 is 8 modules across vs ArUco's 6. Measured detection rate: 36h11 needs
>   20 px sharp, **28 px under motion blur**. A 7 cm tag is 24 px at 640×480 —
>   below the floor. `tools/generate_tags.py` prints this verdict before every
>   print run and currently reports TOO SMALL at 640×480.
> - **Tooling that now exists:** `tools/probe_orin_gpu.py` (inventory +
>   benchmarks for CUDA/VPI/NVJPEG/GStreamer/Isaac ROS on the real board),
>   `gst_camera` node (hardware NVJPG decode, no Isaac ROS needed),
>   `tools/generate_tags.py` (family-agnostic sheets + pixel budget).
> - **Still open exactly as written below:** the crop-vs-bin question for this
>   camera, the true-FOV-from-intrinsics reporting, and the five-way resolution
>   duplication (`camera_info.yaml` still says 1280×720 — though `gst_camera`
>   now *refuses* to publish intrinsics whose resolution disagrees with capture,
>   instead of silently scaling every pose).


**Goal:** the overhead camera should see more of the field, and there should be
a separate set of programs letting a user edit the camera's FOV and resolution —
expressed in normal industry units, not a normalized 0-to-1 knob.

Agreed on the units point. The rest of this section is about making sure the
tool tells the truth, because the naive version of this knob would not.

### The part that has to be said first: FOV is not a software setting

Field of view is fixed by the lens focal length and the sensor size:

```
HFOV = 2 · atan( sensor_width / (2 · focal_length) )
```

Nothing in software changes either term. A tool with a "set FOV" input that
appears to widen the view is lying to the user. **To genuinely see more of the
field you need a shorter-focal-length (wider) lens, or to mount the camera
higher.** Everything software can do falls into three narrower categories:

- **Sensor mode / resolution.** Some cameras produce lower resolutions by
  binning or scaling (FOV unchanged); others by *cropping* the sensor (FOV
  narrowed). Which one a given camera does is a per-device empirical fact that
  must be measured, never assumed. Worth checking early: if this camera crops,
  then the existing drop to 640×480 **already narrowed the field of view**,
  which cuts directly against the goal here.
- **Undistortion alpha.** `cv2.getOptimalNewCameraMatrix(alpha=...)`: alpha=0
  keeps only all-valid pixels (narrower), alpha=1 keeps every source pixel
  (wider, with black wedges at the corners). On a wide or fisheye lens this is
  a real and useful FOV knob, and it is the one place software legitimately
  buys back field of view.
- **ROI / digital crop.** Only ever narrows. Useful for CPU savings, not for
  coverage.

### Use these units

- **HFOV / VFOV / DFOV in degrees** — the standard way to spec field of view.
- **Focal length in mm**, and 35 mm-equivalent focal length for intuition.
- **Sensor format** (1/4", 1/3", …) and pixel pitch in µm.
- **Resolution as W×H** with the conventional mode names (VGA 640×480,
  720p 1280×720, 1080p 1920×1080; UXGA 1600×1200 is the OV2640's full frame).

The important trick: **the tool should not ask the user to type in an FOV.** The
checkerboard calibration already yields `fx`/`fy` in pixels, so true measured
FOV comes straight out of the intrinsics:

```
HFOV = 2 · atan( image_width  / (2 · fx) )
VFOV = 2 · atan( image_height / (2 · fy) )
```

Report that number in degrees. It is measured rather than claimed, and it is
the honest version of the feature the user asked for.

Pair it with a ground-coverage calculator, which is what actually answers "will
I see the whole arena?":

```
ground_width_covered = 2 · mount_height · tan(HFOV / 2)
```

### Coverage is currently limited by marker pixels, not by the lens

This is the finding most likely to change the plan, so check the arithmetic
before building anything.

Known values today: capture is **640×480 @ 15 fps**
(`perception.launch.py:53`), the test arena is **1.83 × 1.83 m**
(`config/arena_test_6ft.yaml`), dictionary `DICT_APRILTAG_36h11` since 2026-08-22, `DICT_4X4_50` before that
(`config/marker_map.yaml`).

Marker sizes differ by role, and **the binding number is the vehicle marker**,
since that is what has to stay detected continuously during a run
(`tools/generate_tags.py`):

- vehicle markers (ids 0–3): **7 cm** — sized to fit an 8.9 cm car roof
- calibration corners (ids 10–13): 10 cm — only needed during calibration

A marker's size in pixels is `marker_m / ground_width_m · image_width`, so for
the 7 cm vehicle markers:

| Ground width covered | Marker px @ 640 | Marker px @ 1280 |
|---|---|---|
| 1.83 m (today) | **24** | 49 |
| 4 m | 11 | 22 |
| 6 m | 7 | 15 |
| 14 m (training arena) | 3 | 6 |

A `DICT_4X4_50` marker is 6 cells across (4 data + 1 border per side), so it
needs roughly 20–25 px per side to detect at all and ~30+ to be comfortable.
Rearranged, the ceiling is:

```
max_ground_width ≈ marker_size_m · image_width / 25
                 ≈ 1.8 m at 640 px,  ≈ 3.6 m at 1280 px   (7 cm markers)
```

**The current 1.83 m arena is already sitting on that limit.** At 640×480 the
vehicle markers land around 24 px — right at the detection floor, with no
margin for motion blur, glare, or a marker tilted on a car roof. So the answer
to "can we see more field?" at the current resolution is *no, and we are
arguably already over the line* — which also makes marker dropouts a prime
suspect for any dead-man trips seen in testing.

(The px-per-cell threshold is a rule of thumb, and these are synthetic-frame
numbers. Confirm empirically — but confirm it *before* widening anything.)

Three real levers, and the tool should make the trade visible:

1. **Bigger vehicle markers.** Cheapest lever and a linear gain — but it is
   *not* just a reprint. The 7 cm size was chosen to fit an 8.9 cm car roof
   with a quiet zone, and this is a 1:16 chassis, so going bigger means a
   marker plate that overhangs the roof. Mechanically easy, but it adds
   payload to a toy-grade car and can foul the body swap. Cheap, not free.
2. **Higher resolution.** Blocked today — see the CPU ceiling below.
3. **Wider lens / higher mount.** Buys coverage, costs pixels-per-marker. Only
   helps in combination with 1 or 2.

### The CPU ceiling that blocks the obvious fix

From `perception.launch.py:41-49`, already measured on this hardware: 1280×720
pegged `aruco_detector` at **~150% CPU with frames queueing 0.5–0.9 s stale,
permanently tripping the 0.25 s dead-man** even with clean detections. 640×480
@ 15 fps was chosen specifically to fit the real-time budget.

The detector is pure-CPU on a Jetson Orin Nano with an idle GPU sitting right
there, so the obvious thought is "move it to CUDA."

**Measured 2026-08-21: that does not work, and this file previously said it
probably would. It was wrong.** OpenCV's ArUco module has no CUDA
implementation in any build — `cv2.cuda` exposes no aruco entry points at all —
and profiling on this board puts the overwhelming majority of per-frame time
inside `detectMarkers` itself:

| Stage | 640×480 | 1280×720 | GPU path? |
|---|---|---|---|
| MJPEG decode (`mjpeg2rgb`) | 2.8 ms | 7.5 ms | yes — NVJPEG/VPI hardware decoder |
| `cvtColor` BGR→GRAY | 0.1 ms | 0.3 ms | yes, but it's ~1% of the frame |
| `detectMarkers` | 12.2 ms | 34.0 ms | **no — CPU-only, ~80% of the frame** |

Supporting measurements, same date, same board:

- No CUDA-capable OpenCV is installed: the pip wheel (4.10.0, the one Python
  actually imports) and the distro build (4.6.0) both report 0 CUDA devices.
  Getting one means building OpenCV from source — which still would not
  accelerate ArUco.
- `onnxruntime` 1.27.0 exposes only `CPUExecutionProvider` and
  `AzureExecutionProvider`. No CUDA or TensorRT EP.
- Policy inference is **0.050 ms/call** — 0.05% of a 100 ms control tick. A
  CUDA kernel launch plus host↔device round trip costs about the same, so GPU
  offload of the policy is a wash at best. Settled: leave it on CPU.
- TensorRT 10.16 and VPI 4.1.3 *are* installed and currently unused.
- Board is in `MAXN_SUPER`, but the governor is `schedutil` and clocks sat at
  1.19 of 1.73 GHz during profiling. Run-to-run variance on identical work was
  ~60%, which is DVFS ramp, not measurement noise. `jetson_clocks` pins it.

The one genuine GPU path for detection is NVIDIA's **cuAprilTag**
(`isaac_ros_apriltag`), a real CUDA fiducial detector. It detects AprilTag
36h11, not ArUco 4×4 — so adopting it means reprinting every marker,
regenerating `markers/`, reworking the detector node and calibration, and
taking on the Isaac ROS dependency. It is a real option, but it is an
architecture change, not a flag.

Caveat on all of the above: these numbers come from synthetic frames
(procedural clutter plus generated markers), so treat them as *relative*
weights rather than absolute predictions. Real arena frames — carpet, cables,
glare — give the quad detector more candidates to reject, so `detectMarkers`
will get worse in the field, not better. Re-measure against real captures
before making a purchasing or architectural decision on them.

### Resolution is currently duplicated across five files

Any resolution tool has to reckon with this. The capture resolution appears in
four places that must agree, plus a fifth that currently doesn't:

| File | What it holds | Value |
|---|---|---|
| `ros2_ws/src/hive_perception/launch/perception.launch.py:53` | the actual capture | 640×480 @ 15 |
| `ros2_ws/src/hive_perception/hive_perception/calibrate_arena.py:108` | `--width`/`--height` defaults | 640×480 |
| `tools/live_arena_debug.py:41` | `--width`/`--height` defaults | 640×480 |
| `tools/snapshot_arena_debug.py:36` | `--width`/`--height` defaults | 640×480 |
| `config/camera_info.yaml` | `image_width`/`image_height` | **1280×720** |

**That last row is a live trap.** Camera intrinsics are resolution-dependent:
`fx`, `fy`, `cx`, `cy` are all in pixels and scale with the image. Right now the
mismatch is harmless only because `camera_info.yaml` is an all-placeholder file
(`fx = fy = 1000`, `cx`/`cy` exactly half of 1280×720 — a synthetic guess, and
the file says so). The moment someone runs a real checkerboard calibration, a
calibration performed at one resolution and consumed at another will scale every
pose by the ratio, silently. Fix the mismatch *before* the first real
calibration, not after.

The natural shape for the tooling, then: **one config file holding the camera
mode, read by all five consumers**, rather than a fifth `--width` flag. A tool
that adds another place to type 640 has made the problem worse.

### Suggested shape

A small set of programs, roughly:

- **`camera_modes`** — enumerate what the device actually supports
  (`v4l2-ctl --list-formats-ext`) rather than letting the user type a
  resolution the hardware will silently refuse. Report each mode's FOV
  consequence once the crop-vs-bin question has been answered for this camera.
- **`camera_report`** — given the calibrated intrinsics, print true HFOV/VFOV/
  DFOV in degrees, focal length in mm, and — given a mount height — the ground
  footprint and the resulting marker pixel size. This is the "edit the FOV"
  feature, told truthfully: it shows what the current optics actually deliver
  and what would have to change.
- **`camera_apply`** — write the chosen mode to the single shared config, and
  refuse (or loudly warn) when the selection would put marker pixels under the
  detection floor or the frame rate over the measured CPU budget.

---

## Cross-cutting

- **Two different cameras, two different jobs.** The overhead camera is
  load-bearing for control — its poses *are* the observation vector, and its
  stalls trip the dead-man. The forward camera is (currently) telemetry. Don't
  let the second one's requirements leak into the first one's budget.
- **Any camera move invalidates the homography.** Re-run
  `./calibrate_arena.sh`. This applies to changing the mount height for
  coverage, which is easy to forget precisely because it feels like a
  mechanical change rather than a software one.
- **`TEST_REPORT.md` is stale** with respect to the V2 drivetrain — it still
  describes the ESP32Servo build and the old GPIO 4/5 pin assignment. Worth a
  fresh bench run against current firmware before trusting its "not yet tested"
  list as a checklist.
