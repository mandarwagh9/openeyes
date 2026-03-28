import sys
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

ros2_python_paths = [
    '/opt/ros/humble/local/lib/python3.10/dist-packages',
    '/opt/ros/humble/lib/python3/dist-packages',
    '/usr/lib/python3/dist-packages',
]
for p in ros2_python_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, Twist
from std_msgs.msg import Header, String

try:
    from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesis
    VISION_MSGS_AVAILABLE = False  # Force JSON mode due to compatibility issues
except (ImportError, Exception) as e:
    VISION_MSGS_AVAILABLE = False

import numpy as np
from typing import Optional, List, Any, Dict, Callable
import threading
import queue


VALID_COMMANDS = ["forward", "backward", "stop", "left", "right", "follow"]


SENSOR_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE
)

COMMAND_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE
)

STATE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)


try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False


class ImageConverter:
    """Utility class for converting between ROS Image and OpenCV."""

    def __init__(self):
        if CV_BRIDGE_AVAILABLE:
            self.bridge = CvBridge()
        else:
            self.bridge = None

    def ros_to_cv2(self, ros_image: Image, encoding: str = 'bgr8') -> Optional[np.ndarray]:
        """Convert ROS Image to OpenCV format."""
        if not self.bridge:
            self.get_logger().warn("cv_bridge not available")
            return None

        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding=encoding)
            return cv_image
        except Exception as e:
            return None

    def cv2_to_ros(self, cv_image: np.ndarray, encoding: str = 'bgr8') -> Optional[Image]:
        """Convert OpenCV image to ROS Image."""
        if not self.bridge:
            return None

        try:
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
            return ros_image
        except Exception as e:
            return None


class VisionPublisher(Node):
    """ROS2 node for publishing all vision results with command handling.
    
    Uses std_msgs/String with JSON payloads for maximum compatibility.
    """

    def __init__(
        self,
        detections_topic: str = "/vision/detections",
        depth_topic: str = "/vision/depth",
        faces_topic: str = "/vision/faces",
        gestures_topic: str = "/vision/gestures",
        poses_topic: str = "/vision/poses",
        cmd_topic: str = "/vision/cmd",
        status_topic: str = "/vision/status",
        frame_id: str = "camera_link",
        confidence_threshold: float = 0.5,
        max_depth_range: float = 5.0,
    ):
        super().__init__("openeyes_vision")

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be between 0 and 1, got {confidence_threshold}")
        if max_depth_range <= 0:
            raise ValueError(f"max_depth_range must be positive, got {max_depth_range}")

        self.detections_topic = detections_topic
        self.depth_topic = depth_topic
        self.faces_topic = faces_topic
        self.gestures_topic = gestures_topic
        self.poses_topic = poses_topic
        self.cmd_topic = cmd_topic
        self.status_topic = status_topic
        self.frame_id = frame_id
        self.confidence_threshold = confidence_threshold
        self.max_depth_range = max_depth_range
        self.vision_msgs_available = VISION_MSGS_AVAILABLE

        if self.vision_msgs_available:
            self.detection_pub = self.create_publisher(
                Detection2DArray, self.detections_topic, SENSOR_QOS
            )
            self.faces_pub = self.create_publisher(
                Detection2DArray, self.faces_topic, SENSOR_QOS
            )
            self.poses_pub = self.create_publisher(
                Detection2DArray, self.poses_topic, SENSOR_QOS
            )
        else:
            self.detection_pub = self.create_publisher(
                String, self.detections_topic, SENSOR_QOS
            )
            self.faces_pub = self.create_publisher(
                String, self.faces_topic, SENSOR_QOS
            )
            self.poses_pub = self.create_publisher(
                String, self.poses_topic, SENSOR_QOS
            )

        self.depth_pub = self.create_publisher(Image, self.depth_topic, SENSOR_QOS)
        self.gestures_pub = self.create_publisher(
            String, self.gestures_topic, SENSOR_QOS
        )
        self.status_pub = self.create_publisher(String, self.status_topic, STATE_QOS)

        self.cmd_sub = self.create_subscription(
            String, self.cmd_topic, self._cmd_callback, COMMAND_QOS
        )

        self.image_converter = ImageConverter() if CV_BRIDGE_AVAILABLE else None

        self._detection_count = 0
        self._last_status_time = self.get_clock().now()
        self._current_cmd = "stop"
        self._cmd_callback_fn: Optional[Callable[[str], None]] = None

        mode = "vision_msgs" if self.vision_msgs_available else "JSON (std_msgs)"
        self.get_logger().info(
            f"OpenEyes Vision Publisher initialized ({mode}):\n"
            f"  - detections: {self.detections_topic}\n"
            f"  - depth: {self.depth_topic}\n"
            f"  - faces: {self.faces_topic}\n"
            f"  - gestures: {self.gestures_topic}\n"
            f"  - poses: {self.poses_topic}\n"
            f"  - cmd: {self.cmd_topic}\n"
            f"  - status: {self.status_topic}"
        )

    def set_cmd_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for command processing."""
        self._cmd_callback_fn = callback

    def _cmd_callback(self, msg: String) -> None:
        """Handle incoming command messages."""
        cmd = msg.data.strip().lower()
        if cmd in VALID_COMMANDS:
            self._current_cmd = cmd
            self.get_logger().info(f">>> CMD RECEIVED: {cmd.upper()}")
            if self._cmd_callback_fn:
                self._cmd_callback_fn(cmd)
        else:
            self.get_logger().warn(f"Invalid command received: {cmd}")

    def get_current_command(self) -> str:
        """Get the current command."""
        return self._current_cmd

    def publish_detections(
        self,
        detections: List[Any],
        frame_shape: tuple = (480, 640),
        header: Optional[Header] = None
    ) -> None:
        """Publish object detection results."""
        if self.vision_msgs_available:
            self._publish_detections_vision_msg(detections, frame_shape, header)
        else:
            self._publish_detections_json(detections, frame_shape)
        self._detection_count += 1

    def _publish_detections_vision_msg(
        self,
        detections: List[Any],
        frame_shape: tuple,
        header: Optional[Header]
    ) -> None:
        """Publish using vision_msgs format."""
        msg = Detection2DArray()
        msg.header = header if header else self._create_header()
        msg.detections = []

        for det in detections:
            confidence = getattr(det, "confidence", 0.0)
            if isinstance(det, dict):
                confidence = det.get("confidence", 0.0)
            if confidence < self.confidence_threshold:
                continue

            detection = Detection2D()

            if hasattr(det, "bbox"):
                bbox = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
            elif isinstance(det, dict):
                bbox = det.get("bbox", [0, 0, 0, 0])
            else:
                bbox = [0, 0, 0, 0]

            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                detection.bbox.center.position.x = float(cx)
                detection.bbox.center.position.y = float(cy)
                detection.bbox.size_x = float(w)
                detection.bbox.size_y = float(h)

            hypothesis = ObjectHypothesis()
            if hasattr(det, "class_name"):
                hypothesis.class_id = det.class_name
            elif isinstance(det, dict):
                hypothesis.class_id = str(det.get("class_name", det.get("class_id", "unknown")))
            else:
                hypothesis.class_id = "unknown"
            hypothesis.score = float(confidence)
            detection.results.append(hypothesis)
            msg.detections.append(detection)

        self.detection_pub.publish(msg)

    def _publish_detections_json(
        self,
        detections: List[Any],
        frame_shape: tuple
    ) -> None:
        """Publish using JSON string format (fallback)."""
        detection_list = []
        for det in detections:
            confidence = getattr(det, "confidence", 0.0)
            if isinstance(det, dict):
                confidence = det.get("confidence", 0.0)
            if confidence < self.confidence_threshold:
                continue

            if hasattr(det, "bbox"):
                bbox = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
            elif isinstance(det, dict):
                bbox = det.get("bbox", [0, 0, 0, 0])
            else:
                bbox = [0, 0, 0, 0]

            class_name = getattr(det, "class_name", "unknown")
            if isinstance(det, dict):
                class_name = det.get("class_name", det.get("class_id", "unknown"))

            detection_list.append({
                "class_name": str(class_name),
                "confidence": float(confidence),
                "bbox": {
                    "x1": float(bbox[0]) if len(bbox) > 0 else 0.0,
                    "y1": float(bbox[1]) if len(bbox) > 1 else 0.0,
                    "x2": float(bbox[2]) if len(bbox) > 2 else 0.0,
                    "y2": float(bbox[3]) if len(bbox) > 3 else 0.0,
                }
            })

        msg = String()
        msg.data = json.dumps({
            "type": "detections",
            "frame_shape": frame_shape,
            "detections": detection_list
        })
        self.detection_pub.publish(msg)

    def publish_depth(self, depth_data: Any, frame_shape: tuple = (480, 640)) -> None:
        """Publish depth map as normalized 32FC1 image (0-1 meters)."""
        if not self.depth_pub:
            return

        depth_array = None
        if hasattr(depth_data, "depth_map") and depth_data.depth_map is not None:
            depth_array = depth_data.depth_map
        elif isinstance(depth_data, np.ndarray):
            depth_array = depth_data

        if depth_array is None:
            return

        if depth_array.dtype != np.float32:
            depth_array = depth_array.astype(np.float32)

        if self.max_depth_range > 0:
            depth_array = np.clip(depth_array / self.max_depth_range, 0.0, 1.0)

        if self.image_converter and self.image_converter.bridge:
            try:
                ros_image = self.image_converter.bridge.cv2_to_imgmsg(
                    depth_array, encoding="32FC1"
                )
                ros_image.header = self._create_header()
                self.depth_pub.publish(ros_image)
            except Exception:
                pass

    def publish_faces(self, faces: List[Any], frame_shape: tuple = (480, 640)) -> None:
        """Publish face detections."""
        if self.vision_msgs_available:
            self._publish_faces_vision_msg(faces, frame_shape)
        else:
            self._publish_faces_json(faces, frame_shape)

    def _publish_faces_vision_msg(self, faces: List[Any], frame_shape: tuple) -> None:
        """Publish faces using vision_msgs format."""
        msg = Detection2DArray()
        msg.header = self._create_header()
        msg.detections = []

        for face in faces:
            detection = Detection2D()

            if hasattr(face, "bbox"):
                bbox = [face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2]
                confidence = getattr(face, "confidence", 0.9)
            elif isinstance(face, dict):
                bbox = face.get("bbox", [0, 0, 0, 0])
                confidence = face.get("confidence", 0.9)
            else:
                continue

            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                detection.bbox.center.position.x = float(cx)
                detection.bbox.center.position.y = float(cy)
                detection.bbox.size_x = float(w)
                detection.bbox.size_y = float(h)

            hypothesis = ObjectHypothesis()
            hypothesis.class_id = "face"
            hypothesis.score = float(confidence)
            detection.results.append(hypothesis)
            msg.detections.append(detection)

        self.faces_pub.publish(msg)

    def _publish_faces_json(self, faces: List[Any], frame_shape: tuple) -> None:
        """Publish faces using JSON format."""
        face_list = []
        for face in faces:
            if hasattr(face, "bbox"):
                bbox = [face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2]
                confidence = getattr(face, "confidence", 0.9)
            elif isinstance(face, dict):
                bbox = face.get("bbox", [0, 0, 0, 0])
                confidence = face.get("confidence", 0.9)
            else:
                continue

            face_list.append({
                "confidence": float(confidence),
                "bbox": {
                    "x1": float(bbox[0]) if len(bbox) > 0 else 0.0,
                    "y1": float(bbox[1]) if len(bbox) > 1 else 0.0,
                    "x2": float(bbox[2]) if len(bbox) > 2 else 0.0,
                    "y2": float(bbox[3]) if len(bbox) > 3 else 0.0,
                }
            })

        msg = String()
        msg.data = json.dumps({
            "type": "faces",
            "frame_shape": frame_shape,
            "faces": face_list
        })
        self.faces_pub.publish(msg)

    def publish_gestures(self, gestures: List[Any]) -> None:
        """Publish gesture recognition results as JSON string."""
        if not gestures:
            return

        msg = String()
        gesture_list = []
        for gesture in gestures:
            if hasattr(gesture, "gesture_type"):
                gesture_list.append({
                    "gesture": gesture.gesture_type,
                    "hand": getattr(gesture, "handedness", "unknown"),
                    "confidence": getattr(gesture, "confidence", 0.0)
                })
            elif isinstance(gesture, dict):
                gesture_list.append(gesture)

        msg.data = json.dumps(gesture_list)
        self.gestures_pub.publish(msg)

    def publish_poses(self, pose_data: Any, frame_shape: tuple = (480, 640)) -> None:
        """Publish body pose estimation results."""
        if not hasattr(pose_data, "detected") or not pose_data.detected:
            return

        if self.vision_msgs_available:
            self._publish_poses_vision_msg(pose_data, frame_shape)
        else:
            self._publish_poses_json(pose_data, frame_shape)

    def _publish_poses_vision_msg(self, pose_data: Any, frame_shape: tuple) -> None:
        """Publish poses using vision_msgs format."""
        msg = Detection2DArray()
        msg.header = self._create_header()
        msg.detections = []

        detection = Detection2D()

        if hasattr(pose_data, "bbox") and pose_data.bbox:
            bbox = [pose_data.bbox.x1, pose_data.bbox.y1, pose_data.bbox.x2, pose_data.bbox.y2]
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                detection.bbox.center.position.x = float(cx)
                detection.bbox.center.position.y = float(cy)
                detection.bbox.size_x = float(w)
                detection.bbox.size_y = float(h)

        pose_info = {}
        if hasattr(pose_data, "keypoints"):
            kp_dict = {}
            for name, kp in pose_data.keypoints.items():
                kp_dict[name] = {"x": float(kp.x), "y": float(kp.y), "confidence": float(kp.confidence)}
            pose_info["keypoints"] = kp_dict

        if hasattr(pose_data, "landmarks"):
            landmarks = []
            for lm in pose_data.landmarks:
                landmarks.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})
            pose_info["landmarks"] = landmarks

        hypothesis = ObjectHypothesis()
        hypothesis.class_id = "person_pose"
        hypothesis.score = 0.95
        hypothesis.id = json.dumps(pose_info)
        detection.results.append(hypothesis)
        msg.detections.append(detection)

        self.poses_pub.publish(msg)

    def _publish_poses_json(self, pose_data: Any, frame_shape: tuple) -> None:
        """Publish poses using JSON format."""
        pose_info = {"detected": True}

        if hasattr(pose_data, "bbox") and pose_data.bbox:
            bbox = [pose_data.bbox.x1, pose_data.bbox.y1, pose_data.bbox.x2, pose_data.bbox.y2]
            pose_info["bbox"] = {
                "x1": float(bbox[0]) if len(bbox) > 0 else 0.0,
                "y1": float(bbox[1]) if len(bbox) > 1 else 0.0,
                "x2": float(bbox[2]) if len(bbox) > 2 else 0.0,
                "y2": float(bbox[3]) if len(bbox) > 3 else 0.0,
            }

        if hasattr(pose_data, "keypoints"):
            kp_dict = {}
            for name, kp in pose_data.keypoints.items():
                kp_dict[name] = {"x": float(kp.x), "y": float(kp.y), "confidence": float(kp.confidence)}
            pose_info["keypoints"] = kp_dict

        if hasattr(pose_data, "landmarks"):
            landmarks = []
            for lm in pose_data.landmarks:
                landmarks.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})
            pose_info["landmarks"] = landmarks

        msg = String()
        msg.data = json.dumps({
            "type": "poses",
            "frame_shape": frame_shape,
            "pose": pose_info
        })
        self.poses_pub.publish(msg)

    def publish_status(self, fps: float, num_objects: int, num_faces: int = 0, num_gestures: int = 0) -> None:
        """Publish vision system status."""
        current_time = self.get_clock().now()
        time_diff = (current_time - self._last_status_time).nanoseconds / 1e9

        if time_diff >= 1.0:
            msg = String()
            msg.data = f"FPS: {fps:.1f} | Objects: {num_objects} | Faces: {num_faces} | Gestures: {num_gestures} | Cmd: {self._current_cmd}"
            self.status_pub.publish(msg)
            self._last_status_time = current_time
            self._detection_count = 0

    def _create_header(self, stamp: Optional[Any] = None) -> Header:
        """Create ROS2 header with timestamp."""
        header = Header()
        if stamp:
            header.stamp = stamp
        else:
            header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        return header


class VisionControlNode(Node):
    """ROS2 node for vision-based robot control."""

    def __init__(self):
        super().__init__("vision_control")

        self.declare_parameter("target_class", "person")
        self.declare_parameter("follow_distance", 1.5)
        self.declare_parameter("max_linear_speed", 0.3)
        self.declare_parameter("max_angular_speed", 0.8)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("detection_topic", "/vision/detections")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.target_class = self.get_parameter("target_class").value
        self.follow_distance = self.get_parameter("follow_distance").value
        self.max_linear_speed = self.get_parameter("max_linear_speed").value
        self.max_angular_speed = self.get_parameter("max_angular_speed").value
        self.image_width = self.get_parameter("image_width").value
        self.image_height = self.get_parameter("image_height").value
        self.detection_topic = self.get_parameter("detection_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            COMMAND_QOS
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            self.detection_topic,
            self.detection_callback,
            SENSOR_QOS
        )

        self.latest_detection: Optional[Detection2DArray] = None
        self._enabled = True

        self.get_logger().info(f"Vision control node initialized: {self.cmd_vel_topic}")

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable control."""
        self._enabled = enabled
        if not enabled:
            self._publish_stop()

    def detection_callback(self, msg: Detection2DArray) -> None:
        """Process detection results and generate control commands."""
        if not self._enabled:
            return

        self.latest_detection = msg

        target = self._find_target(msg)

        if target is None:
            self._publish_stop()
            return

        cmd = self._calculate_control(target)
        self.cmd_vel_pub.publish(cmd)

    def _find_target(self, msg: Detection2DArray) -> Optional[BoundingBox2D]:
        """Find target detection matching target class."""
        for detection in msg.detections:
            for result in detection.results:
                if result.class_id == self.target_class:
                    return detection.bbox
        return None

    def _calculate_control(self, bbox: BoundingBox2D) -> Twist:
        """Calculate velocity command based on detection."""
        cmd = Twist()

        cx = bbox.center.position.x
        cy = bbox.center.position.y

        nx = (cx - self.image_width / 2) / (self.image_width / 2)

        cmd.angular.z = -nx * self.max_angular_speed

        bbox_area = bbox.size_x * bbox.size_y
        normalized_area = bbox_area / (self.image_width * self.image_height)

        if normalized_area < 0.01:
            cmd.linear.x = self.max_linear_speed * 0.5
        elif normalized_area > 0.1:
            cmd.linear.x = -self.max_linear_speed * 0.3
        else:
            cmd.linear.x = 0.0

        return cmd

    def _publish_stop(self) -> None:
        """Publish stop command."""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)


class VisionSubscriberNode(Node):
    """ROS2 node for subscribing to camera images and processing with vision models."""

    def __init__(self):
        super().__init__("vision_subscriber")

        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("processing_queue_size", 1)

        self.camera_topic = self.get_parameter("camera_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.queue_size = self.get_parameter("processing_queue_size").value

        self.image_converter = ImageConverter() if CV_BRIDGE_AVAILABLE else None

        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_size)

        self.camera_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self._image_callback,
            SENSOR_QOS
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            STATE_QOS
        )

        self._latest_camera_info: Optional[CameraInfo] = None
        self._running = True

        self.get_logger().info(f"Vision subscriber initialized: {self.camera_topic}")

    def _image_callback(self, msg: Image) -> None:
        """Callback for incoming images."""
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass

        if not self._frame_queue.full():
            self._frame_queue.put(msg)

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """Callback for camera info."""
        self._latest_camera_info = msg

    def get_latest_frame(self, timeout: float = 0.1) -> Optional[tuple]:
        """Get the latest frame from the queue."""
        try:
            msg = self._frame_queue.get(timeout=timeout)
            if self.image_converter:
                cv_image = self.image_converter.ros_to_cv2(msg)
                return cv_image, msg.header
            return None, msg.header
        except queue.Empty:
            return None, None

    def get_camera_info(self) -> Optional[CameraInfo]:
        """Get latest camera info."""
        return self._latest_camera_info


class VisionWrapperNode(Node):
    """Wrapper node for integrating OpenEyes with ROS2."""

    def __init__(self):
        super().__init__("openeyes_wrapper")

        self.declare_parameter("model_path", "models/yolo11n.onnx")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("enable_control", False)
        self.declare_parameter("enable_subscriber", False)

        self.model_path = self.get_parameter("model_path").value
        self.device = self.get_parameter("device").value
        self.confidence_threshold = self.get_parameter("confidence_threshold").value
        self.enable_control = self.get_parameter("enable_control").value
        self.enable_subscriber = self.get_parameter("enable_subscriber").value

        callback_group = ReentrantCallbackGroup()

        self.vision_pub = VisionPublisher()
        self.vision_pub.get_logger()

        if self.enable_control:
            self.control_node = VisionControlNode()
            self.get_logger().info("Vision control enabled")

        if self.enable_subscriber:
            self.subscriber = VisionSubscriberNode()
            self.get_logger().info("Vision subscriber enabled")

        self._publishers = []
        self._subscriptions = []

        self.get_logger().info("OpenEyes ROS2 wrapper initialized")

    def publish_results(self, result, frame: Optional[Any] = None) -> None:
        """Publish vision results to ROS2 topics."""
        detections = []
        for obj in result.objects:
            detections.append({
                "bbox": [obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3]],
                "class_name": obj.class_name,
                "confidence": obj.confidence
            })

        frame_shape = (480, 640, 3)
        self.vision_pub.publish_detections(detections, frame_shape)

    def add_publisher(self, publisher) -> None:
        """Add a publisher to the wrapper."""
        self._publishers.append(publisher)

    def add_subscription(self, subscription) -> None:
        """Add a subscription to the wrapper."""
        self._subscriptions.append(subscription)


class VisionPipeline:
    """Utility class to create a complete vision pipeline with multiple nodes."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.executor: Optional[MultiThreadedExecutor] = None

    def add_vision_publisher(
        self,
        output_topic: str = "/vision/detections",
        camera_topic: str = "/camera/image_raw"
    ) -> VisionPublisher:
        """Add a vision publisher node."""
        node = VisionPublisher(output_topic=output_topic, camera_topic=camera_topic)
        self.nodes['publisher'] = node
        return node

    def add_vision_control(
        self,
        target_class: str = "person"
    ) -> VisionControlNode:
        """Add a vision control node."""
        node = VisionControlNode()
        node.target_class = target_class
        self.nodes['control'] = node
        return node

    def add_vision_subscriber(
        self,
        camera_topic: str = "/camera/image_raw"
    ) -> VisionSubscriberNode:
        """Add a vision subscriber node."""
        node = VisionSubscriberNode()
        node.camera_topic = camera_topic
        self.nodes['subscriber'] = node
        return node

    def create_executor(self, num_threads: int = 4) -> MultiThreadedExecutor:
        """Create a multi-threaded executor for parallel node processing."""
        self.executor = MultiThreadedExecutor(num_threads=num_threads)
        for name, node in self.nodes.items():
            self.executor.add_node(node)
        return self.executor

    def spin(self) -> None:
        """Run the vision pipeline."""
        if not self.executor:
            self.create_executor()

        try:
            self.executor.spin()
        except KeyboardInterrupt:
            pass

    def shutdown(self) -> None:
        """Shutdown the pipeline."""
        for node in self.nodes.values():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def create_vision_pipeline(
    output_topic: str = "/vision/detections",
    camera_topic: str = "/camera/image_raw",
    enable_control: bool = False,
    target_class: str = "person"
) -> VisionPipeline:
    """Factory function to create a complete vision pipeline."""
    pipeline = VisionPipeline()

    pipeline.add_vision_publisher(output_topic=output_topic, camera_topic=camera_topic)

    if enable_control:
        pipeline.add_vision_control(target_class=target_class)

    return pipeline


def main(args=None):
    rclpy.init(args=args)

    pipeline = VisionPipeline()
    pipeline.add_vision_publisher()
    pipeline.add_vision_control(target_class="person")

    executor = pipeline.create_executor(num_threads=4)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
