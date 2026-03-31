import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose, NavigateThroughPose
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import String
from typing import Optional, List, Tuple
import math


NAV_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE
)


class NavigationGoal(Node):
    """ROS2 node for sending navigation goals to Nav2.
    
    Provides waypoint navigation, goal cancellation, and status monitoring.
    """
    
    def __init__(
        self,
        action_name: str = "/navigate_to_pose",
        frame_id: str = "map",
        timeout_sec: float = 30.0,
    ):
        super().__init__("navigation_goal")
        
        self._action_name = action_name
        self._frame_id = frame_id
        self._timeout_sec = timeout_sec
        self._action_client: Optional[ActionClient] = None
        self._goal_handle = None
        self._current_goal_id: Optional[int] = None
        
        self._goal_sub = self.create_subscription(
            String,
            "/navigation/goal",
            self._goal_callback,
            NAV_QOS
        )
        
        self._cancel_sub = self.create_subscription(
            String,
            "/navigation/cancel",
            self._cancel_callback,
            NAV_QOS
        )
        
        self._status_pub = self.create_publisher(
            String,
            "/navigation/status",
            NAV_QOS
        )
        
        self.get_logger().info(f"NavigationGoal initialized: {action_name}")
    
    def initialize(self) -> bool:
        """Initialize the action client."""
        try:
            self._action_client = ActionClient(
                self,
                NavigateToPose,
                self._action_name
            )
            self._action_client.wait_for_server(timeout_sec=5.0)
            self.get_logger().info(f"Connected to Nav2 action server: {self._action_name}")
            return True
        except Exception as e:
            self.get_logger().warning(f"Nav2 action server not available: {e}")
            return False
    
    def _goal_callback(self, msg: String) -> None:
        """Handle incoming navigation goal."""
        try:
            parts = msg.data.strip().split()
            if len(parts) < 3:
                self.get_logger().error(f"Invalid goal format: {msg.data}")
                return
            
            x = float(parts[0])
            y = float(parts[1])
            yaw = float(parts[2]) if len(parts) > 2 else 0.0
            
            self.send_goal(x, y, yaw)
            
        except ValueError as e:
            self.get_logger().error(f"Failed to parse goal: {e}")
    
    def _cancel_callback(self, msg: String) -> None:
        """Handle goal cancellation."""
        self.cancel_goal()
    
    def send_goal(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        position_z: float = 0.0,
    ) -> Optional[int]:
        """Send a navigation goal to Nav2.
        
        Args:
            x: Target X position in meters
            y: Target Y position in meters
            yaw: Target orientation in radians
            position_z: Target Z position (usually 0)
            
        Returns:
            Goal ID if successful, None otherwise
        """
        if not self._action_client:
            if not self.initialize():
                return None
        
        pose = self._create_pose(x, y, position_z, yaw)
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = ""
        
        self._current_goal_id = self._generate_goal_id()
        
        self.get_logger().info(
            f"Sending goal: x={x}, y={y}, yaw={yaw} (id={self._current_goal_id})"
        )
        
        try:
            send_goal_future = self._action_client.send_goal_async(
                goal_msg,
                feedback_callback=self._feedback_callback
            )
            send_goal_future.add_done_callback(self._goal_response_callback)
            return self._current_goal_id
        except Exception as e:
            self.get_logger().error(f"Failed to send goal: {e}")
            return None
    
    def send_waypoints(self, waypoints: List[Tuple[float, float, float]]) -> Optional[int]:
        """Send multiple waypoints to Nav2.
        
        Args:
            waypoints: List of (x, y, yaw) tuples
            
        Returns:
            Goal ID if successful, None otherwise
        """
        if not self._action_client:
            if not self.initialize():
                return None
        
        action_client = ActionClient(self, NavigateThroughPose, "/navigate_through_poses")
        
        try:
            action_client.wait_for_server(timeout_sec=5.0)
        except Exception as e:
            self.get_logger().warning(f"NavigateThroughPose not available: {e}")
            return self.send_goal(waypoints[0][0], waypoints[0][1], waypoints[0][2])
        
        goal_msg = NavigateThroughPose.Goal()
        goal_msg.poses = [self._create_pose(x, y, 0.0, yaw) for x, y, yaw in waypoints]
        
        self.get_logger().info(f"Sending {len(waypoints)} waypoints")
        
        try:
            send_goal_future = action_client.send_goal_async(
                goal_msg,
                feedback_callback=self._feedback_callback
            )
            send_goal_future.add_done_callback(self._goal_response_callback)
            return self._current_goal_id
        except Exception as e:
            self.get_logger().error(f"Failed to send waypoints: {e}")
            return None
    
    def cancel_goal(self) -> None:
        """Cancel the current navigation goal."""
        if self._goal_handle:
            try:
                cancel_future = self._goal_handle.cancel_goal_async()
                self.get_logger().info("Goal cancellation requested")
            except Exception as e:
                self.get_logger().error(f"Failed to cancel goal: {e}")
    
    def _create_pose(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float
    ) -> PoseStamped:
        """Create a PoseStamped message."""
        pose = PoseStamped()
        pose.header.frame_id = self._frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        
        pose.pose.position = Point(x=x, y=y, z=z)
        
        quat = self._euler_to_quaternion(0.0, 0.0, yaw)
        pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        
        return pose
    
    def _euler_to_quaternion(
        self,
        roll: float,
        pitch: float,
        yaw: float
    ) -> Tuple[float, float, float, float]:
        """Convert Euler angles to quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        
        return (qx, qy, qz, qw)
    
    def _generate_goal_id(self) -> int:
        """Generate unique goal ID."""
        import time
        return int(time.time() * 1000)
    
    def _goal_response_callback(self, future) -> None:
        """Handle goal response."""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            self._publish_status("rejected")
            return
        
        self.get_logger().info("Goal accepted")
        self._goal_handle = goal_handle
        self._publish_status("accepted")
        
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)
    
    def _goal_result_callback(self, future) -> None:
        """Handle goal result."""
        try:
            result = future.result().result
            self.get_logger().info(f"Goal completed: {result}")
            self._publish_status("completed")
        except Exception as e:
            self.get_logger().info(f"Goal finished with error: {e}")
            self._publish_status("failed")
    
    def _feedback_callback(self, feedback_msg) -> None:
        """Handle feedback messages."""
        feedback = feedback_msg.feedback
        pose = feedback.current_pose.pose
        self.get_logger().debug(
            f"Navigation progress: x={pose.position.x:.2f}, y={pose.position.y:.2f}"
        )
    
    def _publish_status(self, status: str) -> None:
        """Publish navigation status."""
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = NavigationGoal()
    
    if not node.initialize():
        node.get_logger().error("Failed to initialize navigation")
        return
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
