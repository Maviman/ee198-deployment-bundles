"""ROS 2 node: USB camera -> hardware JPEG decode (NVJPG engine) -> /image_raw.

A drop-in alternative to usb_cam's `mjpeg2rgb`, which decodes MJPEG on the CPU.
On this board that decode costs 2.8 ms/frame at 640x480 and 7.5 ms at 1280x720
(HANDOFF.md) — time taken from the same cores the detector needs. The Orin has
a dedicated NVJPG hardware decoder sitting idle; GStreamer's `nvjpegdec`
reaches it, so the decode leaves the CPU entirely.

This node deliberately does NOT depend on Isaac ROS. It is the half of the GPU
migration that works today on a plain JetPack install, and it composes with
cuAprilTags later rather than competing with it.

Honest scope note: `appsink` hands frames back as ordinary host buffers, so
this removes the *decode* cost, not the host<->device copy. A fully
GPU-resident path (decode -> detect with no host round trip) needs Isaac ROS
NITROS; until then the win here is bounded by the decode numbers above, which
is why `tools/probe_orin_gpu.py` measures rather than assumes it.

Parameters:
    video_device     /dev/video0
    image_width      640
    image_height     480
    framerate        15
    decoder          auto | nvjpegdec | jpegdec
                     auto probes nvjpegdec once and falls back to the CPU
                     element if it is missing or the pipeline will not start.
    camera_info_url  file:// path to the ROS camera_info YAML
    frame_id         camera

Publishes sensor_msgs/Image (bgr8) on image_raw and sensor_msgs/CameraInfo on
camera_info, matching what usb_cam publishes so the launch file's remappings
and the detector node need no changes.
"""

from __future__ import annotations

import sys
import threading

import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def build_pipeline(device: str, width: int, height: int, fps: int, decoder: str) -> str:
    """GStreamer pipeline string for MJPEG capture + decode to BGR.

    io-mode=2 requests MMAP buffers from V4L2 (avoids a userspace copy per
    frame). `drop=true max-buffers=1` on the sink is a correctness requirement,
    not a tuning knob: a queue that backs up turns into pose staleness, which
    trips the 0.25 s dead-man. Better to drop a frame than to deliver an old one.
    """
    if decoder == "nvjpegdec":
        # nvjpegdec can emit NVMM buffers; nvvidconv brings them back to system
        # memory where appsink/cv_bridge can read them.
        convert = "nvvidconv ! video/x-raw, format=BGRx ! videoconvert"
    else:
        convert = "videoconvert"
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
        f"{decoder} ! {convert} ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def open_capture(device: str, width: int, height: int, fps: int, decoder: str, logger):
    """Open the first working pipeline; returns (cap, decoder_used) or (None, None)."""
    import cv2

    candidates = [decoder] if decoder != "auto" else ["nvjpegdec", "jpegdec"]
    for element in candidates:
        pipeline = build_pipeline(device, width, height, fps, element)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ok, _frame = cap.read()  # opening can succeed and reading still fail
            if ok:
                logger.info(f"decoding with {element}"
                            + ("  (HARDWARE NVJPG)" if element == "nvjpegdec"
                               else "  (CPU — nvjpegdec unavailable)"))
                return cap, element
            cap.release()
        logger.warning(f"pipeline with {element} did not produce frames")
    return None, None


def load_camera_info(url: str, width: int, height: int, frame_id: str) -> CameraInfo:
    """Read the ROS camera_info YAML into a CameraInfo message.

    Loudly refuses to silently rescale: intrinsics are in pixels, so a file
    calibrated at one resolution is simply wrong at another. HANDOFF.md flags
    this exact mismatch (config/camera_info.yaml says 1280x720 while capture
    ran at 640x480) as a live trap.
    """
    msg = CameraInfo()
    msg.width, msg.height = int(width), int(height)
    msg.header.frame_id = frame_id
    path = url[len("file://"):] if url.startswith("file://") else url
    if not path:
        return msg
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except OSError:
        return msg
    msg.distortion_model = str(data.get("distortion_model", "plumb_bob"))
    msg.k = [float(v) for v in data["camera_matrix"]["data"]]
    msg.d = [float(v) for v in data["distortion_coefficients"]["data"]]
    msg.r = [float(v) for v in data["rectification_matrix"]["data"]]
    msg.p = [float(v) for v in data["projection_matrix"]["data"]]
    cal_w, cal_h = int(data.get("image_width", width)), int(data.get("image_height", height))
    if (cal_w, cal_h) != (int(width), int(height)):
        raise ValueError(
            f"{path} was calibrated at {cal_w}x{cal_h} but capture is {width}x{height}. "
            "Intrinsics are in pixels and do not carry across resolutions — re-run the "
            "checkerboard calibration at the capture resolution, or fix image_width/"
            "image_height in that file. Refusing to publish scaled-wrong intrinsics.")
    return msg


class GstCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("gst_camera")
        self.declare_parameter("video_device", "/dev/video0")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("framerate", 15)
        self.declare_parameter("decoder", "auto")
        self.declare_parameter("camera_info_url", "")
        self.declare_parameter("frame_id", "camera")

        self.width = int(self.get_parameter("image_width").value)
        self.height = int(self.get_parameter("image_height").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.camera_info = load_camera_info(
            str(self.get_parameter("camera_info_url").value),
            self.width, self.height, self.frame_id)

        self.cap, self.decoder_used = open_capture(
            str(self.get_parameter("video_device").value),
            self.width, self.height, int(self.get_parameter("framerate").value),
            str(self.get_parameter("decoder").value), self.get_logger())
        if self.cap is None:
            raise RuntimeError(
                "no working GStreamer capture pipeline. Check that OpenCV was built "
                "with GStreamer (tools/probe_orin_gpu.py reports this), that the camera "
                "offers MJPEG at this resolution (v4l2-ctl --list-formats-ext), and "
                "that nothing else holds the device.")

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, "image_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "camera_info", 10)
        self.frames = 0
        self.running = True
        # Blocking cap.read() belongs off the executor thread, or it stalls
        # every timer and subscription in this process.
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.get_logger().info(
            f"publishing {self.width}x{self.height} bgr8 on image_raw "
            f"via {self.decoder_used}")

    def _capture_loop(self) -> None:
        while self.running and rclpy.ok():
            ok, frame = self.cap.read()
            if not ok:
                self.get_logger().warning("camera read failed", throttle_duration_sec=2.0)
                continue
            # Stamp as close to capture as we can get through appsink. The
            # controller finite-differences velocity from this stamp, so it must
            # never become a publish time further down the pipeline.
            stamp = self.get_clock().now().to_msg()
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            self.image_pub.publish(msg)
            self.camera_info.header.stamp = stamp
            self.info_pub.publish(self.camera_info)
            self.frames += 1

    def destroy_node(self) -> bool:
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = GstCameraNode()
    except RuntimeError as exc:
        print(f"gst_camera: {exc}", file=sys.stderr)
        rclpy.shutdown()
        raise SystemExit(1)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
