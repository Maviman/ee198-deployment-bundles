# Orin quickstart: calibration + live two-terminal session

Condensed reference for this specific Orin's setup (ROS 2 Jazzy + the
`~/.venvs/n1ctl` controller venv, both installed 2026-07-24). Full detail
lives in `portable_orin_perception/CALIBRATION.md` and `ORIN_DEPLOYMENT.md` —
this is the short version once both bundles are already installed.

Marker map (`portable_orin_perception/config/marker_map.yaml`): evader = id 0,
pursuers = ids 1/2/3. An N=1 test uses one pursuer marker (id 1) + the evader
marker (id 0).

## 1. Camera intrinsics — one-time, ~10 min (skip if already done for this exact camera/focus)

Removes lens distortion; without it you get a few cm of error at the arena
edges.

```bash
source /opt/ros/jazzy/setup.bash
source ~/ee198-deployment-bundles/portable_orin_perception/ros2_ws/install/setup.bash
ros2 run usb_cam usb_cam_node_exe                     # terminal A
ros2 run camera_calibration cameracalibrator \
    --size 8x6 --square 0.025 image:=/image_raw camera:=/overhead   # terminal B
```

Print an 8x6-inner-corner, 25mm-square checkerboard, wave it through the
frame (near/far, all 4 corners, tilted) until X/Y/Size/Skew go green ->
**CALIBRATE** -> **SAVE**. Extract `ost.yaml` from
`/tmp/calibrationdata.tar.gz`, set `camera_name: overhead`, overwrite
`portable_orin_perception/config/camera_info.yaml`. Redo only if you swap
camera or refocus.

## 2. Arena homography — ~2 min, redo every time the camera moves

1. Place markers `10,11,12,13` (the 10cm sheets in `markers/`) - id 10
   top-left, clockwise (11 top-right, 12 bottom-right, 13 bottom-left), per
   `config/arena_test_6ft.yaml`.
2. Tape-measure each marker's **center**, arena-centered meters (origin =
   arena center, +x right, +y up). Edit `config/arena_test_6ft.yaml` ->
   `calibration_points` to the real numbers (defaults assume a clean
   0.80m-from-center square - measure anyway).
3. Run:
   ```bash
   cd ~/ee198-deployment-bundles/portable_orin_perception
   ./calibrate_arena.sh --device /dev/video0
   ```
   (writes `config/arena_homography.yaml`)
4. Check residuals printed per marker: **<= 0.02 m** = good. One marker way
   off -> re-measure that one; all bad -> camera moved mid-capture or a yaml
   entry is swapped.
5. Spot-check: put a marker at a known spot, start the pipeline (section 3
   below), `ros2 topic echo /hive/vehicle_poses` - position within ~2cm,
   heading matches how it's pointed (+x -> 0, +y -> pi/2).

## 3. Checking the camera/markers without re-running calibration every time

The pipeline itself publishes no debug image (`aruco_detector_node` only
emits pose numbers), so use these instead of guessing from bare eyes:

**Raw live feed, no overlay** (just the picture — `aruco_detector_node`
publishes no debug image, so this is plain `/image_raw`, no marker boxes, no
arena outline; run each in its own foreground terminal, `Ctrl-C` both when
done):
```bash
source /opt/ros/jazzy/setup.bash
ros2 run usb_cam usb_cam_node_exe                     # terminal 1

source /opt/ros/jazzy/setup.bash
ros2 run rqt_image_view rqt_image_view /image_raw     # terminal 2
```
Requires a physical display on the Orin (`echo $DISPLAY`) — this box has one
(GDM/Xorg session on `:1`). Only one process can hold `/dev/video0` at a
time — don't run this alongside the tools below.

**Live feed WITH detection + arena boundary overlay** (what you actually want
while adjusting the camera/markers — continuously updates, no ROS needed,
`q`/Esc to quit):
```bash
cd ~/ee198-deployment-bundles/portable_orin_perception
python3 tools/live_arena_debug.py --device /dev/video0
```
Draws green boxes + ids on every detected marker and the calibrated arena
rectangle in yellow (via the fitted homography), refreshed every frame — it
should track your tape/floor boundary closely, live, as you nudge things.

**One-shot snapshot version** of the same overlay (writes a PNG instead of a
window — useful for saving/sharing a specific check):
```bash
python3 tools/snapshot_arena_debug.py --out arena_check.png
```

## 4. The two-terminal live session

**Terminal A - perception** (this same Orin, N=1 -> one pursuer marker):
```bash
cd ~/ee198-deployment-bundles/portable_orin_perception
./run_perception.sh 127.0.0.1 expected_pursuers:=1
```

**Terminal B - the controller**:
```bash
cd ~/ee198-deployment-bundles/portable_n1_controller
~/.venvs/n1ctl/bin/python run_controller.py --model models/n1_catch \
    --source udp --pose-port 9870 --esp <esp-ip>:8888
```

Replace `<esp-ip>` with the car's ESP32 IP off its serial monitor (needs real
WiFi creds flashed first - placeholder creds in the sketch won't connect). If
the car isn't up yet, swap in `tools/mock_esp.py` from the controller bundle
instead of a real `--esp` target to watch commands with zero hardware.

**Acceptance checks, in order:** hand-push the tagged car and watch pose
track reality (`portable_orin_perception/tools/pose_frame_monitor.py`); cover
the lens -> E-stop within ~0.3s, uncover -> re-arm; keep capture->arrival
latency <= 50ms (same tool).
