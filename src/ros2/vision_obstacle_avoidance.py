import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, Odometry
from geometry_msgs.msg import Pose, Point, Twist
from nav_msgs.msg import Odometry as NavOdometry
from std_msgs.msg import String, Header
from typing import Optional, List, Tuple
import numpy as np


OBSTACLE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE
)


class VisionObstacleAvoidance(Node):
    """Vision-based obstacle avoidance using OpenEyes detections.
    
    Subscribes to OpenEyes detections and creates virtual obstacles
    for Nav2 to avoid. Can also override robot velocity when obstacle
    is too close.
    """
    
    def __init__(
        self,
        detection_topic: str = "/vision/detections",
        cmd_vel_in_topic: str = "/nav2_cmd_vel",
        cmd_vel_out_topic: str = "/cmd_vel",
        obstacle_frame: str = "base_link",
        min_obstacle_distance: float = 0.5,
        slowdown_distance: float = 1.0,
        stop_distance: float = 0.3,
        obstacle_classes: Optional[List[str]] = None,
    ):
        super().__init__("vision_obstacle_avoidance")
        
        self._detection_topic = detection_topic
        self._obstacle_frame = obstacle_frame
        self._min_obstacle_distance = min_obstacle_distance
        self._slowdown_distance = slowdown_distance
        self._stop_distance = stop_distance
        
        if obstacle_classes is None:
            self._obstacle_classes = ["person", "chair", "table", "couch", "bed", 
                                      "bottle", "cup", "keyboard", "mouse", "cell phone"]
        else:
            self._obstacle_classes = obstacle_classes
        
        self._use_velocity_override = True
        
        self._detection_sub = self.create_subscription(
            String,
            detection_topic,
            self._detection_callback,
            OBSTACLE_QOS
        )
        
        self._cmd_vel_in_sub = self.create_subscription(
            Twist,
            cmd_vel_in_topic,
            self._cmd_vel_callback,
            OBSTACLE_QOS
        )
        
        self._cmd_vel_pub = self.create_publisher(
            Twist,
            cmd_vel_out_topic,
            OBSTACLE_QOS
        )
        
        self._obstacle_pub = self.create_publisher(
            String,
            "/vision/obstacles",
            OBSTACLE_QOS
        )
        
        self._last_cmd_vel = Twist()
        self._current_detections = []
        self._robot_position = (0.0, 0.0, 0.0)
        self._frame_width = 640
        self._frame_height = 480
        
        self.get_logger().info(
            f"VisionObstacleAvoidance initialized:\n"
            f"  - detection topic: {detection_topic}\n"
            f"  - cmd_vel in: {cmd_vel_in_topic}\n"
            f"  - cmd_vel out: {cmd_vel_out_topic}\n"
            f"  - stop distance: {stop_distance}m\n"
            f"  - slowdown distance: {slowdown_distance}m\n"
            f"  - obstacle classes: {self._obstacle_classes}"
        )
    
    def _detection_callback(self, msg: String) -> None:
        """Process OpenEyes detections for obstacles."""
        try:
            import json
            data = json.loads(msg.data)
            
            if data.get("type") == "detections":
                self._current_detections = data.get("detections", [])
                self._frame_width = data.get("frame_shape", [480, 640])[1]
                self._frame_height = data.get("frame_shape", [480, 640])[0]
                
        except Exception as e:
            self.get_logger().debug(f"Detection parse error: {e}")
    
    def _cmd_vel_callback(self, msg: Twist) -> None:
        """Process incoming velocity commands with obstacle avoidance."""
        self._last_cmd_vel = msg
        
        if not self._use_velocity_override:
            self._cmd_vel_pub.publish(msg)
            return
        
        obstacle_dist = self._check_obstacle_ahead()
        
        if obstacle_dist < 0:
            self._cmd_vel_pub.publish(msg)
            return
        
        if obstacle_dist < self._stop_distance:
            stop_cmd = Twist()
            self._cmd_vel_pub.publish(stop_cmd)
            self.get_logger().info(f"STOPPED - obstacle at {obstacle_dist:.2f}m")
            return
        
        if obstacle_dist < self._slowdown_distance:
            scale = (obstacle_dist - self._stop_distance) / (self._slowdown_distance - self._stop_distance)
            scale = max(0.0, min(1.0, scale))
            
            modified_cmd = Twist()
            modified_cmd.linear.x = msg.linear.x * scale
            modified_cmd.linear.y = msg.linear.y * scale
            modified_cmd.linear.z = msg.linear.z * scale
            modified_cmd.angular.x = msg.angular.x * scale
            modified_cmd.angular.y = msg.angular.y * scale
            modified_cmd.angular.z = msg.angular.z * scale
            
            self._cmd_vel_pub.publish(modified_cmd)
            self.get_logger().debug(f"SLOWED - obstacle at {obstacle_dist:.2f}m (scale={scale:.2f})")
            return
        
        self._cmd_vel_pub.publish(msg)
    
    def _check_obstacle_ahead(self) -> float:
        """Check if there's an obstacle directly ahead.
        
        Returns:
            Distance to nearest obstacle, or -1 if clear
        """
        if not self._current_detections:
            return -1.0
        
        min_distance = float('inf')
        
        for det in self._current_detections:
            class_name = det.get("class_name", "").lower()
            
            if class_name not in self._obstacle_classes:
                continue
            
            bbox = det.get("bbox", {})
            
            cx = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2
            cy = (bbox.get("y1", 0) + bbox.get("y2", 0)) / 2
            
            nx = (cx - self._frame_width / 2) / (self._frame_width / 2)
            ny = (cy - self._frame_height / 2) / (self._frame_height / 2)
            
            if abs(nx) > 0.3:
                continue
            
            h = bbox.get("y2", 0) - bbox.get("y1", 0)
            
            distance = self._estimate_distance(h, class_name)
            
            if distance < min_distance:
                min_distance = distance
        
        if min_distance == float('inf'):
            return -1.0
        
        return min_distance
    
    def _estimate_distance(self, bbox_height: float, class_name: str) -> float:
        """Estimate distance from bounding box height.
        
        Args:
            bbox_height: Height of bounding box in pixels
            class_name: Object class name
            
        Returns:
            Estimated distance in meters
        """
        typical_heights = {
            "person": 1.7,
            "chair": 0.9,
            "table": 0.75,
            "couch": 0.8,
            "bed": 0.5,
            "bottle": 0.25,
            "cup": 0.15,
            "keyboard": 0.04,
            "mouse": 0.04,
            "cell phone": 0.15,
        }
        
        typical_height = typical_heights.get(class_name, 0.5)
        
        focal_length = 500.0
        distance = (typical_height * focal_length) / max(bbox_height, 1)
        
        return max(0.1, min(distance, 10.0))
    
    def publish_obstacles(self) -> None:
        """Publish obstacle positions for visualization."""
        if not self._current_detections:
            return
        
        obstacle_list = []
        
        for det in self._current_detections:
            class_name = det.get("class_name", "").lower()
            
            if class_name not in self._obstacle_classes:
                continue
            
            bbox = det.get("bbox", {})
            h = bbox.get("y2", 0) - bbox.get("y1", 0)
            distance = self._estimate_distance(h, class_name)
            
            obstacle_list.append({
                "class": class_name,
                "distance": distance,
                "confidence": det.get("confidence", 0.0),
            })
        
        if obstacle_list:
            import json
            msg = String()
            msg.data = json.dumps(obstacle_list)
            self._obstacle_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = VisionObstacleAvoidance()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
