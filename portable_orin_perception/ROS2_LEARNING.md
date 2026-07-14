# ROS 2, learned through this bundle

The "how does ROS 2 actually work" companion. Each concept below is something
this pipeline genuinely uses, mapped to the exact file where you can see it —
read the concept, then open the file. Work through it in order alongside the
README's bring-up ladder. (Official tutorials, if you want the long form:
docs.ros.org → "Beginner: CLI tools" and "Beginner: Client libraries".)

## 0. What ROS 2 is (and why we only use it for perception)

ROS 2 is not an OS — it's a message bus + process manager + tooling ecosystem.
Programs ("nodes") publish typed messages on named "topics"; other nodes
subscribe. The win for us: camera drivers, image viewers, recorders, and 3D
visualizers already exist and interoperate, so the only code we write is our
actual logic. We deliberately STOP at the pose boundary: the controller and
ESP32 keep their hardware-verified UDP protocol, and one small node bridges
ROS topics → that protocol. Best of both worlds.

## 1. Distros, sourcing, environments — `setup_orin.sh`

A "distro" is a yearly ROS release tied to an Ubuntu version (22.04 → Humble,
24.04 → Jazzy; the script auto-detects). Installing gives you
`/opt/ros/<distro>/` — nothing is on your PATH until you
`source /opt/ros/<distro>/setup.bash`. That line is why every terminal that
touches ROS starts with sourcing; forget it and `ros2: command not found`.
Our run scripts source it for you — but interactive terminals are on you
(add it to `~/.bashrc` on the Orin if you like).

## 2. Nodes and topics — see it live before reading code

```
ros2 run demo_nodes_cpp talker      # terminal A: publishes on /chatter
ros2 run demo_nodes_py listener     # terminal B: subscribes
ros2 topic list                     # what's flowing right now
ros2 topic echo /chatter            # print any topic's messages
ros2 topic hz /hive/vehicle_poses   # measure a topic's actual rate
ros2 node list / ros2 node info /aruco_detector
```
Different languages, different processes — same bus. Our graph is three nodes:
`overhead_camera` (usb_cam) → `/image_raw` → `aruco_detector` →
`/hive/vehicle_poses` → `pose_bridge` → (leaves ROS via UDP).

## 3. Messages: typed, versioned structs — `aruco_detector_node.py`

Topics carry typed messages (`sensor_msgs/Image`, `geometry_msgs/PoseArray`).
`ros2 interface show geometry_msgs/msg/PoseArray` prints the schema. We chose
the standard `PoseArray` over a custom message on purpose: zero extra build
machinery, rviz2 renders it natively, and our "slot order + NaN = missing"
convention covers what a custom type would have added. Note the quaternion:
ROS has no "yaw" field, so heading h becomes z=sin(h/2), w=cos(h/2) — you'll
find that exact pair in both node files.

## 4. Writing a node in Python (rclpy) — both `*_node.py` files

The pattern to internalize, visible in ~30 lines in `pose_bridge_node.py`:

```python
class PoseBridgeNode(Node):
    def __init__(self):
        super().__init__("pose_bridge")
        self.declare_parameter("controller_ip", "127.0.0.1")   # 5. params
        self.create_subscription(PoseArray, topic, self.on_poses, 10)
        self.create_timer(0.1, self.on_tick)                   # 6. timers
rclpy.spin(node)   # hand the thread to ROS; your callbacks get called
```
Everything is callbacks after `spin()`. Notice what the nodes DON'T contain:
math. They unpack messages, call `core/` functions, pack results. That split
(testable core, thin node) is the single best habit to copy into any future
ROS work — our core runs on Windows in the main repo's pytest suite without
ROS installed at all.

## 5. Parameters — node configs without code edits

`declare_parameter` + launch-time values (see the `parameters=[{...}]` blocks
in `perception.launch.py`). Inspect/poke live:
`ros2 param list /pose_bridge`, `ros2 param get /pose_bridge controller_ip`.
This is why one bundle serves any arena/PC: ids, paths, IPs, rates are all
parameters, not constants.

## 6. Timers vs subscriptions — `pose_bridge_node.py`

The bridge does NOT forward every camera frame. The camera runs at ~30 Hz;
the contract says "≥10 Hz with the freshest pose". So the subscription only
updates state (`FrameBuilder`), and a separate 10 Hz timer sends the newest
complete frame. Decoupling input rate from output rate with a timer is a
standard ROS pattern — and here it also keeps the controller's UDP socket
from accumulating a stale-frame backlog.

## 7. QoS (the `10` you keep seeing)

Every pub/sub takes a QoS profile; the bare `10` = "keep last 10, reliable".
Sensor streams often use "best effort" (drop late frames rather than block).
Ours is fine as-is at these sizes — just know the knob exists: mismatched QoS
between publisher and subscriber is the classic "why is my topic silent"
gotcha (`ros2 topic info -v <topic>` shows both sides).

## 8. Launch files — `launch/perception.launch.py`

`ros2 run` starts one node; a launch file starts the whole graph with
arguments, parameters, and remappings in one place — it's Python, read it
top to bottom. `ros2 launch hive_perception perception.launch.py --show-args`
lists the knobs (`controller_ip`, `video_device`, ...). Remappings
(`("image_raw", "/image_raw")`) rename topics at startup — that's how you
rewire nodes without touching code.

## 9. Workspaces and colcon — `ros2_ws/`

Your code lives in a "workspace": `ros2_ws/src/<package>/` with a
`package.xml` (deps, for `rosdep`) and `setup.py` (entry points → `ros2 run`
names). `colcon build --symlink-install` compiles/installs into
`ros2_ws/install/`, and sourcing `install/setup.bash` OVERLAYS your packages
on top of the system ROS. Two-layer rule of thumb: system ROS = apt's
problem, your workspace = colcon's. `--symlink-install` means Python edits
take effect on node restart, no rebuild.

## 10. The camera stack — `usb_cam`, `camera_info`, cv_bridge

We didn't write a camera driver: the `usb_cam` package turns V4L2 into
`/image_raw` + `/camera_info`. `camera_info` carries the intrinsics
(`config/camera_info.yaml`, written once by the checkerboard tool — see
CALIBRATION.md); our detector subscribes to it and undistorts marker corners.
`cv_bridge` converts `sensor_msgs/Image` ↔ numpy arrays (one line in
`aruco_detector_node.py`). View any image topic with `rqt_image_view`.

## 11. Visualization and recording — rviz2, rosbag2

```
rviz2                                   # add PoseArray display, topic
                                        # /hive/vehicle_poses, fixed frame "arena"
ros2 bag record /image_raw /hive/vehicle_poses   # record a session
ros2 bag play <folder>                  # replay: the detector can't tell
```
Rosbag replay is how you debug detection offline and how you generate
repeatable figures for the report — record one hand-pushed lap of the arena
and you can rerun the pipeline against it forever.

## 12. Time — why `t` comes from the image header

`msg.header.stamp` is the frame's capture time; `self.get_clock().now()` is
"now". The bridge stamps outgoing frames with CAPTURE time (velocity
estimation depends on it) but judges staleness against "now". Across
machines this only works if clocks agree — hence chrony in setup and the
`chronyc tracking` check before trusting any latency number.

## What we deliberately did NOT use (yet)

Services/actions (request-reply patterns — our data only flows one way), TF2
(the transform tree — one static homography does the job), custom messages
(PoseArray sufficed), lifecycle nodes, and micro-ROS on the ESP32 (the UDP
link is hardware-verified; replacing it buys purity, not capability). Each is
a natural "week 2" topic once the pipeline is alive — and you'll recognize
the need the moment it appears.
