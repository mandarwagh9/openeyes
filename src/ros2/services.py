import sys
import os

ros2_paths = [
    '/opt/ros/humble/lib/python3/dist-packages',
    '/opt/ros/humble/local/lib/python3.10/dist-packages',
]
for p in ros2_paths:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

try:
    from std_srvs.srv import SetBool
    STD_SRVS_AVAILABLE = True
except ImportError:
    SetBool = None
    STD_SRVS_AVAILABLE = False

from vision_msgs.srv import DetectObject2D, DetectObjects2D
from sensor_msgs.msg import Image
from std_msgs.msg import String

from typing import Optional, List, Any, Dict
import numpy as np


class VisionService(Node):
    """ROS2 service node for vision model control."""

    def __init__(self):
        super().__init__("vision_service")

        self.declare_parameter("model_path", "models/yolo11n.onnx")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("device", "cuda")

        self.model_path = self.get_parameter("model_path").value
        self.confidence_threshold = self.get_parameter("confidence_threshold").value
        self.device = self.get_parameter("device").value

        self._models_enabled: Dict[str, bool] = {
            "detector": True,
            "depth": False,
            "face": True,
            "gesture": True,
            "pose": True
        }

        self._current_model: str = "yolo11n"

        self._create_services()

        self.get_logger().info("Vision service initialized")

    def _create_services(self) -> None:
        """Create all ROS2 services."""
        self.enable_model_srv = self.create_service(
            SetBool,
            "/vision/enable_model",
            self._handle_enable_model
        )

        self.set_confidence_srv = self.create_service(
            DetectObject2D,
            "/vision/set_confidence",
            self._handle_set_confidence
        )

        self.set_model_srv = self.create_service(
            String,
            "/vision/set_model",
            self._handle_set_model
        )

        self.get_status_srv = self.create_service(
            String,
            "/vision/get_status",
            self._handle_get_status
        )

        self.get_logger().info("Created services: /vision/enable_model, /vision/set_confidence, /vision/set_model, /vision/get_status")

    def _handle_enable_model(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        """Handle enable/disable model request."""
        model_name = request.data
        response.success = True

        if model_name in self._models_enabled:
            self._models_enabled[model_name] = True
            response.message = f"Model {model_name} enabled"
        else:
            response.success = False
            response.message = f"Unknown model: {model_name}"

        self.get_logger().info(response.message)
        return response

    def _handle_set_confidence(self, request: DetectObject2D.Request, response: DetectObject2D.Response) -> DetectObject2D.Response:
        """Handle set confidence threshold request."""
        response.detections.header.frame_id = "camera_link"
        response.detections.header.stamp = self.get_clock().now().to_msg()

        response.success = True
        self.confidence_threshold = request.hypothesis.id

        self.get_logger().info(f"Confidence threshold set to {self.confidence_threshold}")
        return response

    def _handle_set_model(self, request: String, response: String) -> String:
        """Handle set model request."""
        model_name = request.data
        response.data = ""

        valid_models = ["yolo11n", "yolo11s", "yolov10n", "yolov10s", "yolov8n"]

        if model_name in valid_models:
            self._current_model = model_name
            self.model_path = f"models/{model_name}.onnx"
            response.data = f"Model changed to {model_name}"
            self.get_logger().info(response.data)
        else:
            response.data = f"Invalid model: {model_name}. Valid models: {valid_models}"
            self.get_logger().warn(response.data)

        return response

    def _handle_get_status(self, request: String, response: String) -> String:
        """Handle get status request."""
        status_lines = [
            f"Current Model: {self._current_model}",
            f"Model Path: {self.model_path}",
            f"Confidence: {self.confidence_threshold}",
            f"Device: {self.device}",
            "Enabled Models:"
        ]

        for model, enabled in self._models_enabled.items():
            status_lines.append(f"  - {model}: {'enabled' if enabled else 'disabled'}")

        response.data = "\n".join(status_lines)
        return response

    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a model is enabled."""
        return self._models_enabled.get(model_name, False)

    def get_confidence(self) -> float:
        """Get current confidence threshold."""
        return self.confidence_threshold

    def set_confidence(self, value: float) -> None:
        """Set confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, value))


class VisionModelController(Node):
    """ROS2 service for controlling vision models at runtime."""

    def __init__(self):
        super().__init__("vision_model_controller")

        self.declare_parameter("default_model", "yolo11n")
        self.declare_parameter("default_confidence", 0.5)
        self.declare_parameter("default_iou", 0.45)

        self.default_model = self.get_parameter("default_model").value
        self.default_confidence = self.get_parameter("default_confidence").value
        self.default_iou = self.get_parameter("default_iou").value

        self._active_model = self.default_model
        self._confidence = self.default_confidence
        self._iou = self.default_iou
        self._frame_skip = 1

        self._create_services()
        self.get_logger().info("Model controller initialized")

    def _create_services(self) -> None:
        self._create_model_service()
        self._create_confidence_service()
        self._create_skip_service()
        self._create_status_service()

    def _create_model_service(self) -> None:
        def handle_set_model(req, res):
            valid = ["yolo11n", "yolo11s", "yolov10n", "yolov10s", "yolov8n", "yolov8s"]
            if req.model_name in valid:
                self._active_model = req.model_name
                res.success = True
                res.message = f"Switched to {req.model_name}"
            else:
                res.success = False
                res.message = f"Invalid model: {req.model_name}"
            return res

        from vision_msgs.srv import DetectObject2D
        self.model_srv = self.create_service(
            DetectObject2D,
            "/vision/control/set_model",
            handle_set_model
        )

    def _create_confidence_service(self) -> None:
        def handle_set_confidence(req, res):
            conf = float(req.hypothesis.id)
            if 0.0 <= conf <= 1.0:
                self._confidence = conf
                res.success = True
                res.message = f"Confidence set to {conf}"
            else:
                res.success = False
                res.message = "Confidence must be between 0.0 and 1.0"
            return res

        from vision_msgs.srv import DetectObject2D
        self.confidence_srv = self.create_service(
            DetectObject2D,
            "/vision/control/set_confidence",
            handle_set_confidence
        )

    def _create_skip_service(self) -> None:
        def handle_set_skip(req, res):
            skip = int(req.hypothesis.id)
            if skip >= 1:
                self._frame_skip = skip
                res.success = True
                res.message = f"Frame skip set to {skip}"
            else:
                res.success = False
                res.message = "Frame skip must be >= 1"
            return res

        from vision_msgs.srv import DetectObject2D
        self.skip_srv = self.create_service(
            DetectObject2D,
            "/vision/control/set_frame_skip",
            handle_set_skip
        )

    def _create_status_service(self) -> None:
        def handle_get_status(req, res):
            res.data = (
                f"Model: {self._active_model}\n"
                f"Confidence: {self._confidence}\n"
                f"IoU: {self._iou}\n"
                f"Frame Skip: {self._frame_skip}"
            )
            return res

        from std_msgs.msg import String
        self.status_srv = self.create_service(
            String,
            "/vision/control/get_status",
            handle_get_status
        )

    def get_active_model(self) -> str:
        return self._active_model

    def get_confidence(self) -> float:
        return self._confidence

    def get_iou(self) -> float:
        return self._iou

    def get_frame_skip(self) -> int:
        return self._frame_skip


def main(args=None):
    rclpy.init(args=args)

    try:
        rclpy.spin(VisionService())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
