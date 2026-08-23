# Jetson tuning session — copy-paste runbook

Written 2026-08-22. Run this **on the Orin**, over SSH or at a monitor. Every
block is copy-pasteable as-is. Goal: bank the free performance currently being
left on the table, and turn "do I need an AGX?" into a measured answer.

**Fill in the results table at the bottom as you go.** A timing taken on an
unpinned board is worth nothing, which is why Step 1 comes before everything.

---

## Step 0 — session setup (run once per SSH session)

Set the repo path once; every later block uses `$BUNDLE`. Adjust the first line
if you cloned somewhere else.

```bash
export REPO=~/ee198-deployment-bundles
export BUNDLE=$REPO/portable_orin_perception
mkdir -p ~/tuning
cd "$BUNDLE" && pwd && ls
```

Pull the latest code (the Orin only ever gets code by pull):

```bash
cd "$REPO" && git pull && cd "$BUNDLE"
```

Source ROS (needed for anything `ros2 ...`). ROS's setup.bash is not
`nounset`-safe, hence the `set +u`:

```bash
UBUNTU_VER=$(. /etc/os-release && echo "$VERSION_ID")
case "$UBUNTU_VER" in 22.04) export ROS_DISTRO=humble ;; 24.04) export ROS_DISTRO=jazzy ;; esac
set +u; source /opt/ros/$ROS_DISTRO/setup.bash; source "$BUNDLE/ros2_ws/install/setup.bash"; set -u
echo "ROS_DISTRO=$ROS_DISTRO"
```

Record the environment facts (paste the output into the table at the bottom):

```bash
cat /etc/nv_tegra_release
dpkg-query --show nvidia-l4t-core 2>/dev/null
cat /proc/device-tree/model; echo
lsb_release -ds
nproc
```

---

## Step 1 — pin the clocks (the free ~1.45×)

### 1a. Baseline, BEFORE changing anything

```bash
cd "$BUNDLE"
python3 tools/probe_orin_gpu.py --label before-clocks \
  --json ~/tuning/before.json 2>&1 | tee ~/tuning/before.txt
```

Read the current clock explicitly:

```bash
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "cur GHz : $(awk '{printf "%.2f", $1/1e6}' /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)"
echo "max GHz : $(awk '{printf "%.2f", $1/1e6}' /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"
```

Expect roughly `1.19 / 1.73 GHz` with governor `schedutil` — i.e. the board
running at ~69% and every recorded timing understating it.

### 1b. Check the power mode

```bash
sudo nvpmodel -q
```

To see the modes this board actually offers (do **not** guess a mode number —
they differ per board):

```bash
grep -E '^< POWER_MODEL|^POWER_MODEL' /etc/nvpmodel.conf
```

If you are not already in the highest mode listed (`MAXN` / `MAXN_SUPER`),
switch with `sudo nvpmodel -m <ID>` using an ID from that output. A mode change
may prompt for a reboot.

### 1c. Pin

```bash
sudo jetson_clocks
sudo jetson_clocks --show
```

Confirm the clock moved:

```bash
awk '{printf "cur %.2f GHz\n", $1/1e6}' /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

### 1d. Re-measure

```bash
cd "$BUNDLE"
python3 tools/probe_orin_gpu.py --label after-clocks \
  --json ~/tuning/after_clocks.json 2>&1 | tee ~/tuning/after_clocks.txt
```

Print the delta — this is the number that matters, don't eyeball two transcripts:

```bash
python3 tools/probe_orin_gpu.py --compare ~/tuning/before.json ~/tuning/after_clocks.json \
  2>&1 | tee ~/tuning/compare_clocks.txt
```

**Expect ~1.45× faster.** That is more than an AGX Orin's clock advantage over a
*pinned* Nano — which is the whole reason this step comes first.

> `jetson_clocks` does **not** survive a reboot. If a number later looks ~45%
> worse than you remember, check this before debugging anything else.

To make it persist (only after you've decided you want the power draw always):

```bash
sudo tee /etc/systemd/system/jetson-clocks.service >/dev/null <<'EOF'
[Unit]
Description=Pin Jetson clocks
After=nvpmodel.service
[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now jetson-clocks.service
```

---

## Step 2 — camera backend A/B (CPU decode vs hardware NVJPG)

First confirm the hardware decoder exists:

```bash
gst-inspect-1.0 nvjpegdec >/dev/null 2>&1 && echo "nvjpegdec: PRESENT" || echo "nvjpegdec: MISSING"
python3 -c "import cv2; print('GStreamer:', [l.strip() for l in cv2.getBuildInformation().splitlines() if l.strip().startswith('GStreamer')])"
```

Check what your camera actually offers (do not request a mode it lacks):

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext | head -40
```

`capture_run.py` starts the pipeline, measures it, stops it, and writes a JSON
record — one command per run, no second terminal, nothing to remember to
redirect. Everything after `--` goes to `run_perception.sh`.

### 2a. Run A — usb_cam, CPU decode (current default)

```bash
cd "$BUNDLE"
python3 tools/capture_run.py --label camA --seconds 60 -- 127.0.0.1
```

### 2b. Run B — hardware NVJPG decode

```bash
cd "$BUNDLE"
python3 tools/capture_run.py --label camB-gst --seconds 60 -- 127.0.0.1 camera_backend:=gst
```

Each prints a summary and writes `~/tuning/<label>.json` + `<label>.log`. The
summary line that matters is the decoder:

- `nvjpegdec  (HARDWARE)` → the win landed.
- `jpegdec  (CPU fallback)` → nvjpegdec missing; try
  `sudo apt install nvidia-l4t-gstreamer`, or build OpenCV with GStreamer (Step 5).

### 2c. Delta

```bash
python3 tools/capture_run.py --compare ~/tuning/camA.json ~/tuning/camB-gst.json \
  2>&1 | tee ~/tuning/compare_camera.txt
```

Whole-board view while a run is going, if you want it (Ctrl-C to stop):

```bash
sudo tegrastats --interval 500
```

**Expected saving:** 2.8 ms/frame at 640×480, 7.5 ms at 1280×720, moved off the
CPU. Honest limit: `appsink` returns host buffers, so this removes the *decode*
cost, not the host↔device copy.

---

## Step 3 — settle the 720p question

The recorded claim that 1280×720 "pegs the detector" came from a run at **30
fps**. At 15 fps the arithmetic is very different:

| config | budget/frame | work | loaded |
|---|---|---|---|
| 720p @ 30 fps | 33.3 ms | ~41.8 ms | **125%** — queue grows unbounded |
| 720p @ 15 fps | 66.7 ms | ~41.8 ms | 63% |
| 720p @ 15 fps, clocks pinned | 66.7 ms | ~28.8 ms | 43% |

The controller only consumes 10 Hz, so 15 fps was always enough.

### 3a. Re-calibrate at the new resolution FIRST

A homography fitted at 640×480 is invalid pixel geometry at 1280×720. Place the
corner tags, then:

```bash
cd "$BUNDLE"
./calibrate_arena.sh --device /dev/video0 --width 1280 --height 720
```

Residuals ≤ ~0.02 m: good. Larger: re-measure the worst corner's position.

### 3b. Run at 720p15 and watch staleness, not CPU%

```bash
cd "$BUNDLE"
python3 tools/capture_run.py --label 720p15 --seconds 120 -- 127.0.0.1 \
  image_width:=1280 image_height:=720 framerate:=15
```

**Zero suppressed frames is the pass condition** — the tool prints PASS/FAIL on
that line and exits non-zero on failure. Suppression that grows over time is the
exact signature of an over-budget pipeline queueing up.

It also records the published rate; expect ~15 Hz on `/hive/vehicle_poses`.

Compare against your 640x480 baseline:

```bash
python3 tools/capture_run.py --label 640x480 --seconds 120 -- 127.0.0.1
python3 tools/capture_run.py --compare ~/tuning/640x480.json ~/tuning/720p15.json \
  2>&1 | tee ~/tuning/compare_resolution.txt
```

### 3c. Visual confirmation

```bash
cd "$BUNDLE" && ./run_perception.sh 127.0.0.1 image_width:=1280 image_height:=720 \
  framerate:=15 debug:=true
```

That opens `rqt_image_view` on `/hive/debug_image`. Or, with the pipeline
stopped and the camera free:

```bash
cd "$BUNDLE"
python3 tools/snapshot_arena_debug.py --device /dev/video0 --width 1280 --height 720
```

Writes `arena_debug_snapshot.png`. Live version (needs a display):

```bash
python3 tools/live_arena_debug.py --device /dev/video0 --width 1280 --height 720
```

### 3d. If you keep 720p, fix the intrinsics

`config/camera_info.yaml` holds **placeholder** values (`fx = fy = 1000`). Its
declared resolution finally matches capture at 720p, but the numbers are still
synthetic. Before reporting any accuracy figure:

```bash
cd "$BUNDLE"
set +u; source /opt/ros/$ROS_DISTRO/setup.bash; set -u
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 \
  image:=/image_raw camera:=/overhead
# then copy the resulting YAML over config/camera_info.yaml, keeping camera_name: overhead
```

---

## Step 4 — tag family (optional, requires printing)

**Staying on ArUco is fine.** The code is family-agnostic and verified both
ways; the subset-dictionary change even makes ArUco slightly faster.

Check which family is configured:

```bash
grep '^dictionary' "$BUNDLE/config/marker_map.yaml"
```

### To stay on ArUco

Nothing to do. If a pull switched it and you are not ready to reprint:

```bash
sed -i 's/^dictionary: .*/dictionary: DICT_4X4_50/' "$BUNDLE/config/marker_map.yaml"
cd "$BUNDLE" && python3 selftest.py
```

### To switch to AprilTag tag36h11

```bash
cd "$BUNDLE"
sed -i 's/^dictionary: .*/dictionary: DICT_APRILTAG_36h11/' config/marker_map.yaml
python3 tools/generate_tags.py          # prints the pixel budget — HEED IT
python3 selftest.py                     # must print SELFTEST PASSED
```

Then: print `markers/*.pdf` at **100% scale** (no fit-to-page), ruler-check one
black square against the size in its caption, re-tape vehicle tags with the
canonical top edge toward each car's nose, and re-run 3a.

If `generate_tags.py` prints **TOO SMALL** for your capture width, do not print
— fix the resolution first. 36h11 needs ~1.33× the pixels of ArUco 4×4:

| tag px | ArUco 4×4 | 36h11 sharp | 36h11 + motion blur |
|---|---|---|---|
| 20 | 100% | 100% | 79% |
| 24 | 100% | 100% | 96% |
| 28 | 100% | 100% | 100% |

Blur is the real case for a moving car, so design against 28 px. **The family
switch and the move to 720p are effectively one decision.** The only reason to
switch at all is cuAprilTags (Step 6) — if Step 6 says Isaac ROS is unavailable
here, stay on ArUco.

---

## Step 5 — CUDA-enabled OpenCV (highest-value item)

Confirm the problem first:

```bash
python3 -c "import cv2; print(cv2.__version__, cv2.__file__); print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

`CUDA devices: 0` means the GPU is **unreachable from Python** — not idle by
choice. Fiducial detection has no CUDA implementation anywhere, but the
colour-ID arena border work does (`cv2.cuda` has `cvtColor`, `inRange`,
morphology, contours). That pipeline costs ~16 ms/frame at 720p on CPU — about
28% of the frame, all of it GPU-able, all of it currently forced onto the CPU
because those functions are not compiled into this binary.

**Do this before any hardware purchase.** An AGX ships with the same CPU-only
wheel and runs the same CPU code ~1.3× faster.

### 5a. Swap (an 8 GB Orin will OOM on a parallel OpenCV build)

```bash
free -h
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
free -h
```

### 5b. Dependencies

```bash
sudo apt update && sudo apt install -y build-essential cmake git pkg-config \
  libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  libjpeg-dev libpng-dev libtiff-dev gfortran python3-dev python3-numpy \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

### 5c. Fetch source (contrib is required — `aruco` AND the CUDA modules live there)

```bash
export CV_VER=4.10.0
cd ~ && mkdir -p cvbuild && cd cvbuild
git clone --depth 1 -b $CV_VER https://github.com/opencv/opencv.git
git clone --depth 1 -b $CV_VER https://github.com/opencv/opencv_contrib.git
```

### 5d. Configure — `CUDA_ARCH_BIN=8.7` is the Orin's compute capability

```bash
cd ~/cvbuild/opencv && mkdir -p build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_EXTRA_MODULES_PATH="$HOME/cvbuild/opencv_contrib/modules" \
  -D WITH_CUDA=ON -D CUDA_ARCH_BIN=8.7 -D CUDA_ARCH_PTX= \
  -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D WITH_CUBLAS=ON \
  -D WITH_GSTREAMER=ON -D WITH_V4L=ON \
  -D BUILD_opencv_python3=ON -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF \
  -D BUILD_opencv_apps=OFF \
  ..
```

Before building, confirm the summary says CUDA is on and `aruco` is in the
"To be built" module list:

```bash
grep -iE "NVIDIA CUDA|cuDNN|GStreamer" CMakeCache.txt | head
```

### 5e. Build (hours — use `tmux`/`screen` so an SSH drop doesn't kill it)

```bash
tmux new -s cvbuild
make -j$(( $(nproc) - 1 )) 2>&1 | tee ~/tuning/opencv_build.log
sudo make install && sudo ldconfig
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t cvbuild`.

### 5f. The trap: the pip wheel shadows your build

The pip wheel is what Python currently imports. Remove it or the build is invisible:

```bash
pip3 uninstall -y opencv-python opencv-contrib-python opencv-python-headless
python3 -c "import cv2; print(cv2.__version__, cv2.__file__); print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

A **non-zero** device count is the goal. Then re-verify the pipeline still works:

```bash
cd "$BUNDLE" && python3 selftest.py
cd "$BUNDLE/ros2_ws" && colcon build --symlink-install
```

### 5g. Remove the swap when done (optional)

```bash
sudo swapoff /swapfile && sudo rm /swapfile
```

---

## Step 6 — Isaac ROS / cuAprilTags availability

```bash
set +u; source /opt/ros/$ROS_DISTRO/setup.bash; set -u
ros2 pkg list | grep -iE 'isaac|nitros' || echo "no Isaac ROS packages installed"
dpkg -l | grep -i isaac | head
ls /opt/nvidia 2>/dev/null
sudo find / -name '*cuapriltags*' 2>/dev/null | head
```

**If `isaac_ros_apriltag` is present:** the GPU detector path is open. Because
the printed sheets are genuine tag36h11, adopting it changes the detector node
only — not the tags, not the calibration, not anything downstream.

**If absent:** check the Isaac ROS release notes against the L4T + Ubuntu
version you recorded in Step 0 **before planning around it**. Isaac ROS ships
per ROS distro and per JetPack version; do not assume Jazzy is supported. An
unsupported combination is a real finding — record it, stay on the CPU
detector, and revisit after a JetPack upgrade.

---

## Step 7 — the hardware decision

Apply this **after** Steps 1, 5 and 6. Buy when a measurement says so:

- Perception exceeds **~70% of one core** at your target resolution and frame
  rate, *after* clocks are pinned and the colour work is on CUDA; **or**
- you commit to **3+ simultaneous cameras** — the one axis that scales badly,
  since each camera is a full decode + segment + detect pipeline contending for
  one GPU and one memory bus; **or**
- you move the policy to **image input**. A 128×128 CNN is 2.3–9 ms per camera
  per tick, ×3 cars — the one workload that genuinely wants Tensor cores.

**None of these is vehicle count.** Three cars cost essentially the same as one:
detection is O(pixels) not O(markers), and policy inference goes 0.0331 → 0.0365
ms from batch 1 to 3. Do not buy hardware for more cars.

**If you buy, price Orin NX 16GB before AGX Orin.** Same 260-pin SO-DIMM form
factor as the Orin Nano and supported on the Orin Nano dev kit carrier — a
module swap rather than a new machine, at roughly a third of AGX money, for
8 cores @ 2.0 GHz against your 6 @ 1.7. **Verify against your specific carrier
board before ordering.** Your board reports `MAXN_SUPER`, so you already have
the boosted Nano profile, which narrows the gap. Reserve the AGX for the case
where you have committed to multi-camera *plus* a vision policy.

---

## Results table

| measurement | before clocks | after clocks | after CUDA OpenCV |
|---|---|---|---|
| CPU cur / max GHz | / 1.73 | / 1.73 | |
| detect ms @ 640×480 | | | |
| detect ms @ 1280×720 | | | |
| `aruco_detector` %CPU | | | |
| camera node %CPU (usb_cam) | | | n/a |
| camera node %CPU (gst) | | | n/a |
| suppressed frames @ 720p15 | | | |
| `cv2.cuda` device count | 0 | 0 | |

Environment (record once, Step 0):

| fact | value |
|---|---|
| L4T / JetPack | |
| Ubuntu / ROS distro | |
| board model | |
| `nvjpegdec` present | |
| `isaac_ros_apriltag` present | |
| onnxruntime providers | |

The table is for you. **The JSON files are the source of truth** — they are what
gets compared, and what to hand back for analysis.

### What ends up in `~/tuning/`

| file | written by | holds |
|---|---|---|
| `before.json` / `after_clocks.json` | `probe_orin_gpu.py --json` | full board inventory + detection/decode benchmarks, labelled and timestamped |
| `camA.json` / `camB-gst.json` | `capture_run.py` | per-node CPU%, peak RSS, topic rates, suppressed frames, decoder used |
| `640x480.json` / `720p15.json` | `capture_run.py` | same, per resolution |
| `compare_*.txt` | the `--compare` modes | the deltas, already computed |
| `*.log` | both | raw pipeline output, for anything the parsers missed |

Both tools take `--compare A.json B.json`, so any two runs can be diffed after
the fact — you do not have to decide up front which comparison matters:

```bash
cd "$BUNDLE"
python3 tools/probe_orin_gpu.py --compare ~/tuning/before.json ~/tuning/after_clocks.json
python3 tools/capture_run.py   --compare ~/tuning/camA.json    ~/tuning/720p15.json
```

### Sanity-check the set before you finish

```bash
ls -la ~/tuning/
python3 - <<'EOF'
import json, pathlib
for f in sorted(pathlib.Path.home().joinpath("tuning").glob("*.json")):
    d = json.loads(f.read_text())
    cpu = d.get("cpu") or {}
    sup = (d.get("log") or {}).get("suppressed_frames")
    print(f"{f.name:22s} label={d.get('label','?'):14s} "
          f"nodes={len(cpu)} suppressed={sup}")
EOF
```

Any run showing `nodes=0` did not measure anything — the pipeline failed to
start. Check that run's `.log` and redo it before moving on.

### Bundle it up

```bash
tar czf ~/tuning_results.tgz -C ~ tuning
ls -lh ~/tuning_results.tgz
```

Copy it off the Orin (run this on the dev box, not the Jetson):

```bash
scp <user>@<orin-ip>:~/tuning_results.tgz .
```

That archive is self-contained: every number, the environment it was measured
in, and the raw logs behind it.

---

## Gotchas that will cost you an hour each

- **Any camera move invalidates the homography** — re-run `./calibrate_arena.sh`.
  Includes raising the mount for coverage, which feels mechanical rather than
  software and is therefore easy to forget.
- **`jetson_clocks` does not persist across reboot.** First thing to check when
  a number regresses ~45%.
- **The pip OpenCV wheel shadows a source build** (Step 5f). Symptom: you built
  with CUDA and `getCudaEnabledDeviceCount()` still returns 0.
- **Resolution lives in several files that must agree** — the launch file (now
  takes `image_width`/`image_height`), `calibrate_arena.py` defaults, the two
  debug tools, and `camera_info.yaml`. `gst_camera` **refuses** to publish
  intrinsics whose declared resolution disagrees with capture rather than
  silently scaling every pose; if it errors on startup, that is this problem
  telling you the truth.
- **The camera can only be opened once.** `live_arena_debug.py` /
  `snapshot_arena_debug.py` need the pipeline stopped, and vice versa.
- **WiFi power-save is per ESP32** — every new car needs `WiFi.setSleep(false)`.
  Without it: alternating 15/105 ms RTTs and ~6% packet loss, on that car only.
- **Every timing in this repo came from synthetic frames.** Real carpet, cables
  and glare give the quad detector more candidates to reject, so detection gets
  *worse* in the field. Re-measure against real captures before spending money
  on the strength of a synthetic number.
