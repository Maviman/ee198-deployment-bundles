"""Pure-Python perception core: NO rclpy imports allowed anywhere in here.

Everything contract-bearing (pixel->arena homography math, heading extraction,
dropout/staleness policy, the UDP pose-frame JSON) lives in this package so it
can be unit-tested on any machine (the main repo's Windows pytest suite and
this bundle's selftest.py both import it directly). The ROS 2 nodes one level
up are thin wrappers that only move data between topics and these functions.

Modules:
  marker_math   -- ArUco corner arrays -> center + top-edge midpoint (pixel
                   space) + lens undistortion.
  arena_frame   -- homography fit/apply, pixel -> arena-centered meters,
                   heading in the training convention (0 = +x, CCW+),
                   homography YAML save/load.
  frame_builder -- vehicle id -> slot mapping, the hold/staleness dropout
                   policy, and serialization of the exact UDP JSON frame the
                   controller's parse_pose_frame() consumes.
"""
