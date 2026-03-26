import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, BoundingBox2D, ObjectHypothesis
from geometry_msgs.msg import Pose, PoseStamped, Twist
from std_msgs.msg import Header, String
import numpy as np
from typing import Optional, List, Any


class VisionPublisher(Node):
    """ROS2 node for publishing vision results."""

    def __init__(
        self,
        output_topic: str = "/vision/detections",
        camera_topic: str = "/camera/image_raw"
    ):
        super().__init__("vision_publisher")

        self.declare_parameter("output_topic", output_topic)
        self.declare_parameter("camera_topic", camera_topic)
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("confidence_threshold", 0.5)

        self.output_topic = self.get_parameter("output_topic").value
        self.camera_topic = self.get_parameter("camera_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.confidence_threshold = self.get_parameter("confidence_threshold").value

        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            self.output_topic,
            sensor_qos
        )

        self.image_pub = self.create_publisher(
            Image,
            "/vision/debug/image",
            sensor_qos
        )

        self.status_pub = self.create_publisher(
            String,
            "/vision/status",
            QoSProfile(depth=1)
        )

        self.get_logger().info(f"Vision publisher initialized: {self.output_topic}")

    def publish_detections(
        self,
        detections: List[Any],
        frame_shape: tuple
    ) -> None:
        """Publish detection results as ROS2 messages."""
        msg = Detection2DArray()
        msg.header = self._create_header()
        msg.detections = []

        height, width = frame_shape[:2]

        for det in detections:
            detection = Detection2D()

            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            detection.bbox.center.position.x = cx
            detection.bbox.center.position.y = cy
            detection.bbox.size_x = w
            detection.bbox.size_y = h

            hypothesis = ObjectHypothesis()
            hypothesis.class_id = det.get("class_name", "unknown")
            hypothesis.score = det.get("confidence", 0.0)
            detection.results.append(hypothesis)

            msg.detections.append(detection)

        self.detection_pub.publish(msg)

    def publish_status(self, fps: float, num_objects: int) -> None:
        """Publish vision system status."""
        msg = String()
        msg.data = f"FPS: {fps:.1f} | Objects: {num_objects}"
        self.status_pub.publish(msg)

    def _create_header(self) -> Header:
        """Create ROS2 header with timestamp."""
        header = Header()
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

        self.target_class = self.get_parameter("target_class").value
        self.follow_distance = self.get_parameter("follow_distance").value
        self.max_linear_speed = self.get_parameter("max_linear_speed").value
        self.max_angular_speed = self.get_parameter("max_angular_speed").value
        self.image_width = self.get_parameter("image_width").value
        self.image_height = self.get_parameter("image_height").value

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            "/vision/detections",
            self.detection_callback,
            QoSProfile(depth=5)
        )

        self.latest_detection: Optional[Detection2DArray] = None

        self.get_logger().info("Vision control node initialized")

    def detection_callback(self, msg: Detection2DArray) -> None:
        """Process detection results and generate control commands."""
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


class VisionWrapperNode(Node):
    """Wrapper node for integrating OpenEyes with ROS2."""

    def __init__(self):
        super().__init__("openeyes_wrapper")

        self.declare_parameter("model_path", "models/yolo11n.onnx")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("confidence_threshold", 0.5)

        self.model_path = self.get_parameter("model_path").value
        self.device = self.get_parameter("device").value
        self.confidence_threshold = self.get_parameter("confidence_threshold").value

        self.vision_pub = VisionPublisher()
        self.control_node = VisionControlNode()

        self.get_logger().info("OpenEyes ROS2 wrapper initialized")

    def publish_results(self, result) -> None:
        """Publish vision results to ROS2 topics."""
        detections = []
        for obj in result.objects:
            detections.append({
                "bbox": obj.bbox,
                "class_name": obj.class_name,
                "confidence": obj.confidence
            })

        self.vision_pub.publish_detections(
            detections,
            result.objects[0].bbox if result.objects else (480, 640, 3)
        )


def main(args=None):
    rclpy.init(args=args)

    try:
        rclpy.spin(VisionPublisher())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
