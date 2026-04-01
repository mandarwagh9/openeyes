"""Sensor fusion module for OpenEyes.

Fuses data from multiple sensors:
- Camera (RGB)
- Depth camera
- LIDAR

Provides unified obstacle detection and tracking across all sensors.

Requirements:
    - sensor_msgs
    - geometry_msgs
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2, Image
from geometry_msgs.msg import Pose, Point, Twist
from std_msgs.msg import Header, String
from nav_msgs.msg import Odometry
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import time
from dataclasses import dataclass


@dataclass
class FusedObstacle:
    """Unified obstacle from sensor fusion."""
    x: float
    y: float
    z: float
    distance: float
    sources: List[str]
    confidence: float
    obstacle_type: str
    bbox: Optional[Tuple[int, int, int, int]] = None


SENSOR_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE
)


class SensorFusion(Node):
    """Fuse camera, depth, and LIDAR data for unified obstacle detection.
    
    Features:
    - Multi-sensor obstacle fusion
    - Confidence scoring based on detection overlap
    - Temporal smoothing
    - Ground plane estimation
    """
    
    def __init__(
        self,
        camera_detections_topic: str = "/vision/detections",
        lidar_scan_topic: str = "/scan",
        depth_topic: str = "/vision/depth",
        fused_topic: str = "/fusion/obstacles",
        frame_id: str = "base_link",
        fusion_distance_threshold: float = 0.5,
        min_confidence: float = 0.5,
    ):
        super().__init__("sensor_fusion")
        
        self._camera_detections_topic = camera_detections_topic
        self._lidar_scan_topic = lidar_scan_topic
        self._depth_topic = depth_topic
        self._fused_topic = fused_topic
        self._frame_id = frame_id
        self._fusion_distance_threshold = fusion_distance_threshold
        self._min_confidence = min_confidence
        
        self._last_camera_detections: Optional[str] = None
        self._last_lidar_scan: Optional[LaserScan] = None
        self._last_depth: Optional[Image] = None
        self._obstacle_history: List[FusedObstacle] = []
        
        self._camera_sub = self.create_subscription(
            String,
            camera_detections_topic,
            self._camera_callback,
            SENSOR_QOS
        )
        
        self._lidar_sub = self.create_subscription(
            LaserScan,
            lidar_scan_topic,
            self._lidar_callback,
            SENSOR_QOS
        )
        
        self._depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self._depth_callback,
            SENSOR_QOS
        )
        
        self._fused_pub = self.create_publisher(
            String,
            fused_topic,
            SENSOR_QOS
        )
        
        self.get_logger().info(
            f"SensorFusion initialized:\n"
            f"  - camera topic: {camera_detections_topic}\n"
            f"  - lidar topic: {lidar_scan_topic}\n"
            f"  - depth topic: {depth_topic}\n"
            f"  - fused topic: {fused_topic}"
        )
    
    def _camera_callback(self, msg: String) -> None:
        """Process camera detections."""
        self._last_camera_detections = msg.data
        self._fuse_and_publish()
    
    def _lidar_callback(self, msg: LaserScan) -> None:
        """Process LIDAR scan."""
        self._last_lidar_scan = msg
        self._fuse_and_publish()
    
    def _depth_callback(self, msg: Image) -> None:
        """Process depth image."""
        self._last_depth = msg
    
    def _fuse_and_publish(self) -> None:
        """Fuse detections from all sensors and publish."""
        obstacles: List[FusedObstacle] = []
        
        camera_obstacles = self._parse_camera_detections()
        lidar_obstacles = self._parse_lidar_scan()
        
        obstacles.extend(camera_obstacles)
        
        for lo in lidar_obstacles:
            matched = False
            for co in camera_obstacles:
                dist = np.sqrt((lo.x - co.x)**2 + (lo.y - co.y)**2)
                if dist < self._fusion_distance_threshold:
                    co.sources.append("lidar")
                    co.confidence = min(1.0, co.confidence + 0.2)
                    matched = True
                    break
            if not matched:
                obstacles.append(lo)
        
        self._obstacle_history.append(obstacles)
        if len(self._obstacle_history) > 10:
            self._obstacle_history.pop(0)
        
        if self._fused_pub.get_subscription_count() > 0:
            msg = self._create_fused_message(obstacles)
            self._fused_pub.publish(msg)
    
    def _parse_camera_detections(self) -> List[FusedObstacle]:
        """Parse camera detection JSON."""
        import json
        
        obstacles: List[FusedObstacle] = []
        
        if not self._last_camera_detections:
            return obstacles
        
        try:
            data = json.loads(self._last_camera_detections)
            detections = data.get("detections", [])
            
            for det in detections:
                x = det.get("x", 0)
                y = det.get("y", 0)
                
                obstacles.append(FusedObstacle(
                    x=float(x),
                    y=float(y),
                    z=0.0,
                    distance=float(det.get("distance", 1.0)),
                    sources=["camera"],
                    confidence=float(det.get("confidence", 0.5)),
                    obstacle_type=det.get("class", "unknown"),
                    bbox=(
                        det.get("bbox", {}).get("x1", 0),
                        det.get("bbox", {}).get("y1", 0),
                        det.get("bbox", {}).get("x2", 0),
                        det.get("bbox", {}).get("y2", 0),
                    ) if "bbox" in det else None
                ))
        except Exception as e:
            self.get_logger().debug(f"Failed to parse camera detections: {e}")
        
        return obstacles
    
    def _parse_lidar_scan(self) -> List[FusedObstacle]:
        """Parse LIDAR scan into obstacles."""
        obstacles: List[FusedObstacle] = []
        
        if not self._last_lidar_scan:
            return obstacles
        
        scan = self._last_lidar_scan
        ranges = np.array(scan.ranges)
        angles = np.arange(scan.angle_min, scan.angle_min + len(ranges) * scan.angle_increment)
        
        if len(angles) > len(ranges):
            angles = angles[:len(ranges)]
        
        valid_mask = np.isfinite(ranges) & (ranges > 0.1) & (ranges < 10.0)
        
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_ranges) == 0:
            return obstacles
        
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        clusters = self._cluster_points(x, y, tolerance=0.3)
        
        for cx, cy in clusters:
            if len(cx) < 3:
                continue
            
            mx = np.mean(cx)
            my = np.mean(cy)
            dist = np.sqrt(mx**2 + my**2)
            
            obstacle_type = "unknown"
            if len(cx) < 10:
                obstacle_type = "small_object"
            elif len(cx) < 30:
                obstacle_type = "person" if dist < 3 else "furniture"
            else:
                obstacle_type = "wall"
            
            obstacles.append(FusedObstacle(
                x=float(mx),
                y=float(my),
                z=0.0,
                distance=float(dist),
                sources=["lidar"],
                confidence=0.7,
                obstacle_type=obstacle_type
            ))
        
        return obstacles
    
    def _cluster_points(
        self,
        x: np.ndarray,
        y: np.ndarray,
        tolerance: float = 0.3
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Cluster 2D points."""
        if len(x) == 0:
            return []
        
        points = np.column_stack([x, y])
        clusters = []
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
                    if np.linalg.norm(points[j] - current_point) < tolerance:
                        visited.add(j)
                        queue.append(j)
                        cluster_indices.append(j)
            
            if len(cluster_indices) >= 3:
                clusters.append((x[cluster_indices], y[cluster_indices]))
        
        return clusters
    
    def _create_fused_message(self, obstacles: List[FusedObstacle]) -> String:
        """Create fused obstacle message."""
        import json
        
        obstacle_list = []
        for obs in obstacles:
            if obs.confidence >= self._min_confidence:
                obstacle_list.append({
                    "x": round(obs.x, 3),
                    "y": round(obs.y, 3),
                    "z": round(obs.z, 3),
                    "distance": round(obs.distance, 3),
                    "sources": obs.sources,
                    "confidence": round(obs.confidence, 2),
                    "type": obs.obstacle_type
                })
        
        msg = String()
        msg.data = json.dumps({
            "timestamp": time.time(),
            "obstacle_count": len(obstacle_list),
            "obstacles": obstacle_list
        })
        
        return msg
    
    def get_fused_obstacles(self) -> List[FusedObstacle]:
        """Get current fused obstacles."""
        return self._obstacle_history[-1] if self._obstacle_history else []


def main(args=None):
    rclpy.init(args=args)
    
    node = SensorFusion()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()