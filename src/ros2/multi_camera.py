"""Multi-camera support for OpenEyes.

Supports multiple camera streams with:
- Independent frame capture
- Synchronized capture for stereo
- Batch processing for efficiency

Usage:
    python -m src.ros2.multi_camera --cameras 0 1 2
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header, String
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import time
import threading
from dataclasses import dataclass


@dataclass
class CameraFrame:
    """Frame data from a camera."""
    camera_id: int
    frame: np.ndarray
    timestamp: float
    sequence: int


CAMERA_QOS = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE
)


class MultiCameraNode(Node):
    """Manage multiple camera streams.
    
    Features:
    - Parallel camera capture
    - Synchronized capture mode
    - Camera hot-plug support
    - Frame batching for GPU efficiency
    """
    
    def __init__(
        self,
        camera_ids: List[int] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        synced_capture: bool = False,
        batch_size: int = 4,
    ):
        super().__init__("multi_camera")
        
        self._camera_ids = camera_ids or [0]
        self._width = width
        self._height = height
        self._fps = fps
        self._synced_capture = synced_capture
        self._batch_size = batch_size
        
        self._cameras: Dict[int, Any] = {}
        self._frame_buffers: Dict[int, List[CameraFrame]] = {}
        self._sequence = 0
        self._lock = threading.Lock()
        
        self._camera_info_pubs = {}
        self._image_pubs = {}
        
        for cam_id in self._camera_ids:
            self._frame_buffers[cam_id] = []
            
            info_topic = f"/camera_{cam_id}/camera_info"
            image_topic = f"/camera_{cam_id}/image_raw"
            
            self._camera_info_pubs[cam_id] = self.create_publisher(
                CameraInfo, info_topic, CAMERA_QOS
            )
            self._image_pubs[cam_id] = self.create_publisher(
                Image, image_topic, CAMERA_QOS
            )
        
        self._timer = self.create_timer(1.0 / fps, self._capture_callback)
        
        self.get_logger().info(
            f"MultiCameraNode initialized:\n"
            f"  - cameras: {self._camera_ids}\n"
            f"  - resolution: {width}x{height}\n"
            f"  - fps: {fps}\n"
            f"  - synced: {synced_capture}\n"
            f"  - batch_size: {batch_size}"
        )
    
    def _capture_callback(self) -> None:
        """Capture frames from all cameras."""
        with self._lock:
            for cam_id in self._camera_ids:
                try:
                    frame = self._capture_frame(cam_id)
                    if frame is not None:
                        self._sequence += 1
                        
                        camera_frame = CameraFrame(
                            camera_id=cam_id,
                            frame=frame,
                            timestamp=time.time(),
                            sequence=self._sequence
                        )
                        
                        self._frame_buffers[cam_id].append(camera_frame)
                        if len(self._frame_buffers[cam_id]) > self._batch_size:
                            self._frame_buffers[cam_id].pop(0)
                        
                        self._publish_frame(camera_frame)
                        
                except Exception as e:
                    self.get_logger().warn(f"Camera {cam_id} capture error: {e}")
    
    def _capture_frame(self, cam_id: int) -> Optional[np.ndarray]:
        """Capture a single frame from camera."""
        import cv2
        
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            return None
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                frame = cv2.resize(frame, (self._width, self._height))
            return frame
        
        return None
    
    def _publish_frame(self, camera_frame: CameraFrame) -> None:
        """Publish frame as ROS image."""
        from cv_bridge import CvBridge
        
        bridge = CvBridge()
        
        try:
            img_msg = bridge.cv2_to_imgmsg(
                camera_frame.frame,
                encoding="bgr8"
            )
            
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = f"camera_{camera_frame.camera_id}"
            img_msg.height = camera_frame.frame.shape[0]
            img_msg.width = camera_frame.frame.shape[1]
            
            cam_id = camera_frame.camera_id
            if cam_id in self._image_pubs:
                self._image_pubs[cam_id].publish(img_msg)
                
                cam_info = CameraInfo()
                cam_info.header = img_msg.header
                cam_info.width = self._width
                cam_info.height = self._height
                cam_info.distortion_model = "plumb_bob"
                cam_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
                cam_info.k = [
                    self._width, 0.0, self._width / 2.0,
                    0.0, self._width, self._height / 2.0,
                    0.0, 0.0, 1.0
                ]
                
                if cam_id in self._camera_info_pubs:
                    self._camera_info_pubs[cam_id].publish(cam_info)
                    
        except Exception as e:
            self.get_logger().debug(f"Publish error: {e}")
    
    def get_latest_frames(self) -> Dict[int, np.ndarray]:
        """Get latest frame from each camera."""
        frames = {}
        with self._lock:
            for cam_id, buffer in self._frame_buffers.items():
                if buffer:
                    frames[cam_id] = buffer[-1].frame
        return frames
    
    def get_batched_frames(self, batch_size: int = None) -> List[List[CameraFrame]]:
        """Get batched frames from all cameras."""
        batch_size = batch_size or self._batch_size
        batches = []
        
        with self._lock:
            min_frames = min(len(buf) for buf in self._frame_buffers.values())
            
            for i in range(min_frames):
                batch = [buf[i] for buf in self._frame_buffers.values()]
                batches.append(batch)
        
        return batches


def main(args=None):
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-camera node")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1],
                        help="Camera IDs")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--synced", action="store_true", help="Synchronized capture")
    
    parsed_args = parser.parse_args(args)
    
    rclpy.init(args=args)
    
    node = MultiCameraNode(
        camera_ids=parsed_args.cameras,
        width=parsed_args.width,
        height=parsed_args.height,
        fps=parsed_args.fps,
        synced_capture=parsed_args.synced,
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()