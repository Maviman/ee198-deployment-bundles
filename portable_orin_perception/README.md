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

## What's in here

| Path | What it is |
|---|---|
| `ros2_ws/src/hive_perception/hive_perception/core/` | ALL the math (homography, heading, staleness policy, wire JSON) as plain Python — no rclpy, unit-tested here (`selftest.py`) and in the main repo (`tests/test_orin_perception_core.py`). |
| `ros2_ws/src/hive_perception/hive_perception/*_node.py` | The two ROS 2 nodes: `aruco_detector` (images → `/hive/vehicle_poses`) and `pose_bridge` (poses → UDP JSON at 10 Hz). Thin wrappers over `core/`. |
| `ros2_ws/src/hive_perception/launch/perception.launch.py` | Brings up camera (`usb_cam`) + detector + bridge together. |
| `config/` | `marker_map.yaml` (which id is which car), `arena_test_6ft.yaml` (calibration marker positions), `camera_info.yaml` (intrinsics — placeholder until calibrated), `arena_homography.yaml` (written by calibration, gitignored). |
| `markers/` | Printable marker sheets (PDF at exact physical size + PNG). Vehicles = 7 cm ids 0–3, calibration corners = 10 cm ids 10–13. |
| `vendored/` | Byte-identical copies of the controller's pose parser (`pc_controller/pose_stream.py` + its pose type) — the schema OWNER is `portable_n1_controller`; never edit these here. |
| `tools/` | `generate_aruco_markers.py` (regenerate sheets), `pose_frame_monitor.py` (run on the PC to watch the live pose stream without the controller). |
| `selftest.py` | Zero-hardware health check: OpenCV ArUco works, conventions hold, and our frames parse with the controller's own vendored parser. |

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

**3. Markers + camera.** Print `markers/*.pdf` at 100% scale, ruler-check one,
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

**6. Controller in the loop, no car.** On the PC:
`python tools/mock_esp.py` (from portable_n1_controller) +
`python run_controller.py --model models/n1_catch --source udp --esp
127.0.0.1:8888`. Cover the camera lens: commands must go silent / E-stop
within ~0.3 s. This rung + rung 7 are the acceptance tests.

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

## Provenance

Created 2026-07-13 in the main AI Training repo (source of truth). Wire format
and dead-man behavior verified against `portable_n1_controller` (hardware-
tested 2026-07-06/07). Vendored parser copied byte-identical the same day.
