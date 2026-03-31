import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Header
from typing import Optional, Callable
import numpy as np

try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False


SENSOR_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE
)


class DepthToLaserScan(Node):
    """Convert depth image to laser scan for Nav2 integration.
    
    This node subscribes to depth images and publishes LaserScan messages
    that can be used by the ROS2 Navigation2 stack for obstacle avoidance.
    """
    
    def __init__(
        self,
        depth_topic: str = "/vision/depth",
        scan_topic: str = "/scan",
        range_min: float = 0.1,
        range_max: float = 10.0,
        angle_min: float = -np.pi / 2,
        angle_max: float = np.pi / 2,
        angle_increment: float = np.pi / 180,
        scan_time: float = 0.1,
        frame_id: str = "camera_link",
    ):
        super().__init__("depth_to_laserscan")
        
        self._range_min = range_min
        self._range_max = range_max
        self._angle_min = angle_min
        self._angle_max = angle_max
        self._angle_increment = angle_increment
        self._scan_time = scan_time
        self._frame_id = frame_id
        
        self._num_ranges = int((angle_max - angle_min) / angle_increment) + 1
        
        self._depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self._depth_callback,
            SENSOR_QOS
        )
        
        self._scan_pub = self.create_publisher(
            LaserScan,
            scan_topic,
            SENSOR_QOS
        )
        
        self._bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None
        
        self._last_depth: Optional[np.ndarray] = None
        self._image_width: int = 640
        self._image_height: int = 480
        
        self.get_logger().info(
            f"DepthToLaserScan initialized:\n"
            f"  - depth topic: {depth_topic}\n"
            f"  - scan topic: {scan_topic}\n"
            f"  - range: [{range_min}, {range_max}]m\n"
            f"  - angles: [{np.degrees(angle_min):.1f}, {np.degrees(angle_max):.1f}] deg\n"
            f"  - resolution: {np.degrees(angle_increment):.2f} deg"
        )
    
    def _depth_callback(self, msg: Image) -> None:
        """Process depth image and publish laser scan."""
        try:
            if self._bridge:
                depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            else:
                self.get_logger().warn("cv_bridge not available, cannot convert")
                return
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
            return
        
        self._image_width = depth.shape[1]
        self._image_height = depth.shape[0]
        
        scan = self._convert_depth_to_scan(depth, msg.header)
        self._scan_pub.publish(scan)
    
    def _convert_depth_to_scan(
        self,
        depth: np.ndarray,
        header: Optional[Header] = None
    ) -> LaserScan:
        """Convert depth image to laser scan."""
        scan = LaserScan()
        
        if header:
            scan.header = header
        else:
            scan.header = Header()
            scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = self._frame_id
        
        scan.angle_min = self._angle_min
        scan.angle_max = self._angle_max
        scan.angle_increment = self._angle_increment
        scan.time_increment = 0.0
        scan.scan_time = self._scan_time
        scan.range_min = self._range_min
        scan.ranges = self._range_max * np.ones(self._num_ranges, dtype=np.float32)
        
        center_x = self._image_width // 2
        center_y = self._image_height // 2
        
        ranges = []
        for i in range(self._num_ranges):
            angle = self._angle_min + i * self._angle_increment
            ray_x = int(center_x + center_x * np.tan(angle))
            
            if 0 <= ray_x < self._image_width:
                depth_col = depth[:, ray_x]
                valid_depths = depth_col[
                    (depth_col > self._range_min) & 
                    (depth_col < self._range_max) &
                    (depth_col != np.inf) &
                    (depth_col == depth_col)
                ]
                
                if len(valid_depths) > 0:
                    min_depth = np.min(valid_depths)
                    ranges.append(min_depth)
                else:
                    ranges.append(self._range_max)
            else:
                ranges.append(self._range_max)
        
        scan.ranges = ranges
        
        return scan
    
    def set_range_limits(self, min_range: float, max_range: float) -> None:
        """Set range limits for laser scan."""
        self._range_min = min_range
        self._range_max = max_range
    
    def set_angle_limits(self, angle_min: float, angle_max: float) -> None:
        """Set angle limits for laser scan."""
        self._angle_min = angle_min
        self._angle_max = angle_max
        self._num_ranges = int((angle_max - angle_min) / self._angle_increment) + 1


def main(args=None):
    rclpy.init(args=args)
    
    node = DepthToLaserScan()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
