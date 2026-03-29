import rclpy
from rclpy.action import ActionClient
from rclpy.action.server import ActionServer
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import json
from typing import Optional, List, Any


class VisionActionServer(Node):
    """ROS2 Action server for vision-based robot control."""

    def __init__(self, node_name: str = "vision_action_server"):
        super().__init__(node_name)

        self._logger = self.get_logger()

        self._cmd_vel_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )

        self._current_goal_handle = None
        self._goal_execution_thread = None
        self._running = False

        self._logger.info("Vision Action Server initialized")

    def execute_callback(self, goal_handle):
        """Execute the vision action goal."""
        self._current_goal_handle = goal_handle

        goal = goal_handle.request
        action_type = goal.action_type if hasattr(goal, "action_type") else "unknown"

        self._logger.info(f"Executing action: {action_type}")

        feedback_msg = goal.feedback_type() if hasattr(goal, "feedback_type") else None

        result = goal.result_type() if hasattr(goal, "result_type") else None

        if hasattr(goal, "target"):
            target = goal.target

            if action_type == "follow":
                success = self._execute_follow(target, goal_handle, feedback_msg)
            elif action_type == "detect":
                success = self._execute_detect(target, goal_handle, feedback_msg)
            elif action_type == "track":
                success = self._execute_track(target, goal_handle, feedback_msg)
            else:
                success = False

        if success:
            goal_handle.succeed()
            if result:
                result.success = True
                result.message = "Action completed successfully"
        else:
            goal_handle.abort()
            if result:
                result.success = False
                result.message = "Action failed"

        return result

    def _execute_follow(self, target, goal_handle, feedback_msg) -> bool:
        """Execute person following action."""
        duration = getattr(target, "duration", 10.0)
        max_distance = getattr(target, "max_distance", 2.0)

        twist = Twist()
        start_time = self.get_clock().now()

        while rclpy.ok() and (self.get_clock().now() - start_time).nanoseconds < duration * 1e9:
            if goal_handle.is_cancel_requested:
                self._stop_robot()
                return False

            twist.linear.x = 0.3
            self._cmd_vel_pub.publish(twist)

            if feedback_msg:
                feedback_msg.status = "following"
                goal_handle.publish_feedback(feedback_msg)

            rclpy.sleep(0.1)

        self._stop_robot()
        return True

    def _execute_detect(self, target, goal_handle, feedback_msg) -> bool:
        """Execute object detection action."""
        target_class = getattr(target, "class_name", "person")
        timeout = getattr(target, "timeout", 5.0)

        start_time = self.get_clock().now()

        while rclpy.ok() and (self.get_clock().now() - start_time).nanoseconds < timeout * 1e9:
            if goal_handle.is_cancel_requested:
                return False

            if feedback_msg:
                feedback_msg.status = f"searching for {target_class}"
                goal_handle.publish_feedback(feedback_msg)

            rclpy.sleep(0.1)

        if feedback_msg:
            feedback_msg.status = "detected"
            goal_handle.publish_feedback(feedback_msg)

        return True

    def _execute_track(self, target, goal_handle, feedback_msg) -> bool:
        """Execute object tracking action."""
        target_id = getattr(target, "track_id", -1)
        duration = getattr(target, "duration", 10.0)

        twist = Twist()
        start_time = self.get_clock().now()

        while rclpy.ok() and (self.get_clock().now() - start_time).nanoseconds < duration * 1e9:
            if goal_handle.is_cancel_requested:
                self._stop_robot()
                return False

            if feedback_msg:
                feedback_msg.status = f"tracking {target_id}"
                goal_handle.publish_feedback(feedback_msg)

            rclpy.sleep(0.1)

        self._stop_robot()
        return True

    def _stop_robot(self):
        """Stop robot movement."""
        twist = Twist()
        self._cmd_vel_pub.publish(twist)

    def destroy_node(self):
        self._stop_robot()
        super().destroy_node()


class QoSConfig:
    """ROS2 QoS configuration helper."""

    @staticmethod
    def sensor_profile(depth: int = 1) -> QoSProfile:
        """Best effort QoS for sensor data."""
        return QoSProfile(
            depth=depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

    @staticmethod
    def command_profile(depth: int = 10) -> QoSProfile:
        """Reliable QoS for commands."""
        return QoSProfile(
            depth=depth,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

    @staticmethod
    def state_profile() -> QoSProfile:
        """Transient local QoS for state data."""
        return QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

    @staticmethod
    def get_profile(profile_name: str, depth: int = 1) -> QoSProfile:
        """Get QoS profile by name."""
        profiles = {
            "sensor": QoSConfig.sensor_profile(depth),
            "command": QoSConfig.command_profile(depth),
            "state": QoSConfig.state_profile(),
            "best_effort": QoSProfile(
                depth=depth,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            ),
            "reliable": QoSProfile(
                depth=depth,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            ),
        }
        return profiles.get(profile_name, QoSConfig.command_profile(depth))


class MultiCameraManager(Node):
    """Manage multiple camera sources."""

    def __init__(self, camera_sources: List[int] = None):
        super().__init__("multi_camera_manager")
        self._camera_sources = camera_sources or [0]
        self._cameras: dict = {}
        self._frame_subscribers: dict = {}

        self._qos = QoSConfig.sensor_profile(depth=1)

        for idx, source in enumerate(self._camera_sources):
            topic_name = f"/camera_{idx}/image_raw"
            self._frame_subscribers[source] = self.create_subscription(
                Image,
                topic_name,
                lambda msg, i=idx: self._camera_callback(msg, i),
                self._qos,
            )

        self._logger = self.get_logger()
        self._logger.info(f"MultiCameraManager initialized with {len(self._camera_sources)} cameras")

    def _camera_callback(self, msg: Image, camera_idx: int):
        """Handle incoming camera frame."""
        pass

    def get_camera_count(self) -> int:
        return len(self._camera_sources)

    def get_active_camera(self) -> Optional[int]:
        return self._camera_sources[0] if self._camera_sources else None


class TimeSyncManager(Node):
    """Handle time synchronization for vision data."""

    def __init__(self, use_ros_time: bool = True):
        super().__init__("time_sync_manager")
        self._use_ros_time = use_ros_time
        self._start_time = self.get_clock().now()

    def get_timestamp(self) -> float:
        """Get synchronized timestamp."""
        if self._use_ros_time:
            return self.get_clock().now().nanoseconds / 1e9
        else:
            import time
            return time.time()

    def get_elapsed_time(self) -> float:
        """Get elapsed time since initialization."""
        now = self.get_clock().now()
        return (now - self._start_time).nanoseconds / 1e9
