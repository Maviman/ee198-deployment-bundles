# Arena calibration checklist

Two independent calibrations. Intrinsics (step 1) is per-CAMERA: once, ever,
unless you change camera or focus. The homography (step 2) is per-MOUNT:
re-run every time the camera moves, even slightly.

## 1. Camera intrinsics (one-time, ~10 minutes)

Removes lens distortion. A homography maps a flat plane to a flat plane
perfectly ONLY through an ideal pinhole lens — real webcam distortion bends
straight lines, worst at the image edges (= your arena corners), so skipping
this leaves a few cm of position error exactly where pinning happens.

1. Print a checkerboard (the classic 8x6 inner-corner, 25 mm squares —
   docs.ros.org camera calibration tutorial has one) and tape it to something
   rigid.
2. Camera running (`ros2 run usb_cam usb_cam_node_exe`), then:
   ```
   ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 \
       image:=/image_raw camera:=/overhead
   ```
3. Wave the board through the view — near/far, all four corners, tilted —
   until the X/Y/Size/Skew bars go green, press CALIBRATE, then SAVE.
4. Extract `ost.yaml` from the saved tarball (`/tmp/calibrationdata.tar.gz`),
   set `camera_name: overhead`, and overwrite `config/camera_info.yaml`.
5. Restart the camera node; `ros2 topic echo /camera_info --once` should now
   show your real matrix, and the detector picks it up automatically.

Until you do this, the placeholder file has zero distortion (undistortion
no-ops). Fine for first bring-up, not for numbers you report.

## 2. Arena homography (2 minutes, repeat per camera move)

Maps pixels to arena-centered meters. Four markers at known floor positions
define the fit.

1. **Place** markers 10, 11, 12, 13 (the 10 cm sheets) per the diagram in
   `config/arena_test_6ft.yaml` — id 10 top-left, going clockwise. Flat,
   unwrinkled, fully visible.
2. **Measure** each marker's CENTER position in arena coordinates (origin =
   arena center, +x right, +y up, meters) with a tape measure. The yaml's
   defaults assume an 80 cm square — measure anyway; where they ACTUALLY sit
   is what matters, and 1 cm of tape-measure error = 1 cm of pose error
   forever.
3. **Edit** `config/arena_test_6ft.yaml` → `calibration_points` to the
   measured numbers.
4. **Run** `./calibrate_arena.sh --device /dev/video0`
   (or `--image <snapshot.png>` from a saved frame).
5. **Read the residuals.** ≤ 0.02 m per marker: done —
   `config/arena_homography.yaml` is written. One marker much worse than the
   rest: its measurement is wrong, re-measure THAT one. All bad: camera moved
   mid-capture, marker misdetection (check lighting/glare), or two yaml
   entries swapped.
6. **Spot-check:** put any marker at a spot you've measured (center is easy),
   start the pipeline, `ros2 topic echo /hive/vehicle_poses` — position
   within ~2 cm and heading matching how you pointed it (+x = 0, +y = π/2
   ≈ 1.57, CCW positive).

## Parallax (the ~5 cm roof-height detail)

Vehicle markers ride on car roofs, ~5 cm above the floor plane the homography
was fitted on. Geometry: with the camera ~2 m up, a roof marker at the arena's
edge projects ~2–3 cm outward of its true floor position; zero at image
center. Two options:

- **Ignore it** (default): the policy was validated robust to 5 cm pose noise.
  `arena_test_6ft.yaml` keeps `marker_height_above_floor_m: 0.0`.
- **Cancel it** (recommended before measuring accuracy for the report): put
  the four calibration markers on boxes/jigs at the SAME height as the car
  roofs when you calibrate — the fitted plane then IS the roof plane and the
  error vanishes. Record the jig height in `marker_height_above_floor_m` so
  the config documents itself.

## When to redo what

| Event | Intrinsics | Homography |
|---|---|---|
| Camera bumped / remounted / zoomed / refocused | no | **yes** |
| Different camera or lens | **yes** | **yes** |
| Arena moved/resized (new corner positions) | no | **yes** (after re-measuring yaml) |
| Lighting changed | no | no (but re-check detection reliability) |
| New marker printout | no | only if corner markers changed size/position |
