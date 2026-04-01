"""LIDAR integration module for OpenEyes.

Provides LIDAR point cloud processing and obstacle detection for
enhanced navigation and safety in robot vision systems.

Requirements:
    - rospkg
    - sensor_msgs
    - geometry_msgs

Usage:
    # Start LIDAR processing
    python -m src.ros2.lidar_processing --topic /scan

    # Start with obstacle detection
    python -m src.ros2.lidar_processing --topic /scan --detect-obstacles
"""

from dataclasses import dataclass
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from geometry_msgs.msg import Pose, Point, Twist, TransformStamped
from std_msgs.msg import Header, String
from nav_msgs.msg import Odometry as NavOdometry
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import time


OBSTACLE_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE
)


@dataclass
class LidarObstacle:
    """Represents a detected obstacle from LIDAR."""
    x: float
    y: float
    distance: float
    angle: float
    intensity: float
    obstacle_type: str


class LidarProcessor(Node):
    """Process LIDAR data for obstacle detection and mapping.
    
    Features:
    - Real-time obstacle detection
    - Point cloud generation from LaserScan
    - Ground/obstacle segmentation
    - Multiple LIDAR support
    """
    
    def __init__(
        self,
        scan_topic: str = "/scan",
        pointcloud_topic: str = "/lidar/points",
        obstacle_topic: str = "/lidar/obstacles",
        frame_id: str = "lidar_link",
        min_distance: float = 0.1,
        max_distance: float = 10.0,
        obstacle_height: float = 0.5,
        cluster_tolerance: float = 0.2,
    ):
        super().__init__("lidar_processor")
        
        self._scan_topic = scan_topic
        self._pointcloud_topic = pointcloud_topic
        self._obstacle_topic = obstacle_topic
        self._frame_id = frame_id
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._obstacle_height = obstacle_height
        self._cluster_tolerance = cluster_tolerance
        
        self._last_scan: Optional[LaserScan] = None
        self._obstacles: List[LidarObstacle] = []
        self._obstacle_count = 0
        
        self._scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self._scan_callback,
            OBSTACLE_QOS
        )
        
        self._pointcloud_pub = self.create_publisher(
            PointCloud2,
            pointcloud_topic,
            OBSTACLE_QOS
        )
        
        self._obstacle_pub = self.create_publisher(
            String,
            obstacle_topic,
            OBSTACLE_QOS
        )
        
        self.get_logger().info(
            f"LidarProcessor initialized:\n"
            f"  - scan topic: {scan_topic}\n"
            f"  - pointcloud topic: {pointcloud_topic}\n"
            f"  - obstacle topic: {obstacle_topic}\n"
            f"  - range: [{min_distance}, {max_distance}]m"
        )
    
    def _scan_callback(self, msg: LaserScan) -> None:
        """Process incoming LaserScan data."""
        self._last_scan = msg
        
        obstacles = self._detect_obstacles(msg)
        self._obstacles = obstacles
        self._obstacle_count += 1
        
        if self._pointcloud_pub.get_subscription_count() > 0:
            pc = self._laserscan_to_pointcloud(msg)
            self._pointcloud_pub.publish(pc)
        
        if self._obstacle_pub.get_subscription_count() > 0:
            obstacle_msg = self._create_obstacle_message(obstacles, msg.header)
            self._obstacle_pub.publish(obstacle_msg)
    
    def _detect_obstacles(self, scan: LaserScan) -> List[LidarObstacle]:
        """Detect obstacles from LaserScan data."""
        obstacles: List[LidarObstacle] = []
        
        ranges = np.array(scan.ranges)
        angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
        
        if len(angles) > len(ranges):
            angles = angles[:len(ranges)]
        
        valid_mask = (
            (ranges > self._min_distance) &
            (ranges < self._max_distance) &
            (np.isfinite(ranges))
        )
        
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_ranges) == 0:
            return obstacles
        
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        clusters = self._cluster_points(x, y)
        
        for cluster in clusters:
            cx = np.mean(cluster[0])
            cy = np.mean(cluster[1])
            distance = np.sqrt(cx**2 + cy**2)
            angle = np.arctan2(cy, cx)
            
            obstacle_type = self._classify_obstacle(distance, len(cluster))
            
            obstacles.append(LidarObstacle(
                x=float(cx),
                y=float(cy),
                distance=float(distance),
                angle=float(angle),
                intensity=float(np.mean(cluster[2])) if len(cluster) > 2 else 1.0,
                obstacle_type=obstacle_type
            ))
        
        return obstacles
    
    def _cluster_points(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Cluster points using distance-based grouping."""
        if len(x) == 0:
            return []
        
        points = np.column_stack([x, y])
        clusters: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        visited = set()
        
        for i in range(len(points)):
            if i in visited:
                continue
            
            cluster_indices = [i]
            queue = [i]
            visited.add(i)
            
            while queue:
                current_idx = queue.pop(0)
                current_point = points[current_idx]
                
                for j in range(len(points)):
                    if j in visited:
                        continue
                    
                    dist = np.linalg.norm(points[j] - current_point)
                    if dist < self._cluster_tolerance:
                        visited.add(j)
                        cluster_indices.append(j)
                        queue.append(j)
            
            if len(cluster_indices) >= 3:
                cluster_x = x[cluster_indices]
                cluster_y = y[cluster_indices]
                cluster_intensity = np.ones(len(cluster_indices))
                clusters.append((cluster_x, cluster_y, cluster_intensity))
        
        return clusters
    
    def _classify_obstacle(self, distance: float, cluster_size: int) -> str:
        """Classify obstacle type based on characteristics."""
        if cluster_size < 5:
            return "small"
        elif cluster_size < 15:
            if distance < 2.0:
                return "person"
            else:
                return "furniture"
        else:
            return "wall"
    
    def _laserscan_to_pointcloud(self, scan: LaserScan) -> PointCloud2:
        """Convert LaserScan to PointCloud2."""
        ranges = np.array(scan.ranges)
        angles = np.arange(scan.angle_min, scan.angle_min + len(ranges) * scan.angle_increment, scan.angle_increment)
        
        if len(angles) > len(ranges):
            angles = angles[:len(ranges)]
        
        valid_mask = np.isfinite(ranges) & (ranges > self._min_distance)
        
        x = ranges[valid_mask] * np.cos(angles[valid_mask])
        y = ranges[valid_mask] * np.sin(angles[valid_mask])
        z = np.zeros_like(x)
        
        points = np.column_stack([x, y, z]).astype(np.float32)
        
        pc = PointCloud2()
        pc.header = scan.header
        pc.header.frame_id = self._frame_id
        pc.width = len(points)
        pc.height = 1
        pc.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc.is_bigendian = False
        pc.point_step = 12
        pc.row_step = pc.point_step * pc.width
        pc.data = points.tobytes()
        pc.is_dense = True
        
        return pc
    
    def _create_obstacle_message(
        self,
        obstacles: List[LidarObstacle],
        header: Header
    ) -> String:
        """Create JSON obstacle message."""
        import json
        
        obstacle_list = []
        for obs in obstacles:
            obstacle_list.append({
                "x": round(obs.x, 3),
                "y": round(obs.y, 3),
                "distance": round(obs.distance, 3),
                "angle": round(np.degrees(obs.angle), 1),
                "type": obs.obstacle_type
            })
        
        msg = String()
        msg.data = json.dumps({
            "timestamp": time.time(),
            "obstacle_count": len(obstacles),
            "obstacles": obstacle_list
        })
        
        return msg
    
    def get_obstacles(self) -> List[LidarObstacle]:
        """Get current list of detected obstacles."""
        return self._obstacles
    
    def get_obstacle_count(self) -> int:
        """Get total number of obstacles detected since start."""
        return self._obstacle_count


def main(args=None):
    rclpy.init(args=args)
    
    node = LidarProcessor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()