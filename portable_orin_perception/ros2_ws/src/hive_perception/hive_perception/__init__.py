"""hive_perception: overhead ArUco perception for the EE198 pursuit project.

ROS 2 side of the portable Orin bundle. Data flow (see the bundle README):

    usb_cam -> /image_raw + /camera_info
            -> aruco_detector_node  (cv2.aruco + core/: px -> arena meters)
            -> /hive/vehicle_poses  (geometry_msgs/PoseArray, slot-ordered)
            -> pose_bridge_node     (core/frame_builder: staleness policy +
                                     UDP JSON to the controller, 10 Hz)

All math lives in ``hive_perception.core`` (rclpy-free, unit-tested on any
machine); the node modules here only move data between ROS and those functions.
"""
