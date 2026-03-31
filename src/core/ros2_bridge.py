"""ROS2 Bridge Protocol for decoupling ROS2 from business logic."""

from typing import Protocol, Optional, Any
from abc import abstractmethod


class ROS2Bridge(Protocol):
    """Protocol for ROS2 publishing - allows easy swapping of implementations."""
    
    @abstractmethod
    def publish_detections(self, detections: list[Any], frame_shape: tuple) -> None:
        """Publish object detections."""
        ...
    
    @abstractmethod
    def publish_depth(self, depth_data: Any, frame_shape: tuple) -> None:
        """Publish depth data."""
        ...
    
    @abstractmethod
    def publish_faces(self, faces: list[Any], frame_shape: tuple) -> None:
        """Publish face detections."""
        ...
    
    @abstractmethod
    def publish_gestures(self, gestures: list[Any]) -> None:
        """Publish gesture recognitions."""
        ...
    
    @abstractmethod
    def publish_poses(self, pose: Any, frame_shape: tuple) -> None:
        """Publish pose landmarks."""
        ...
    
    @abstractmethod
    def publish_status(self, fps: float, objects: int, faces: int, gestures: int) -> None:
        """Publish system status."""
        ...
    
    @abstractmethod
    def set_cmd_callback(self, callback: Any) -> None:
        """Set command callback."""
        ...


class NoOpROS2Bridge:
    """No-op implementation when ROS2 is not available."""
    
    def __init__(self):
        self._connected = False
    
    def publish_detections(self, detections: list[Any], frame_shape: tuple) -> None:
        pass
    
    def publish_depth(self, depth_data: Any, frame_shape: tuple) -> None:
        pass
    
    def publish_faces(self, faces: list[Any], frame_shape: tuple) -> None:
        pass
    
    def publish_gestures(self, gestures: list[Any]) -> None:
        pass
    
    def publish_poses(self, pose: Any, frame_shape: tuple) -> None:
        pass
    
    def publish_status(self, fps: float, objects: int, faces: int, gestures: int) -> None:
        pass
    
    def set_cmd_callback(self, callback: Any) -> None:
        pass


class UDPSenderBridge:
    """Bridge for UDP output."""
    
    def __init__(self, host: str, port: int):
        from src.output.udp_sender import UDPSender
        from src.output.json_formatter import format_vision_result
        
        self._sender = UDPSender(host=host, port=port)
        self._sender.open()
        self._formatter = format_vision_result
    
    def send(self, result: Any) -> None:
        """Send vision result via UDP."""
        json_output = self._formatter(result)
        self._sender.send(json_output)
    
    def close(self) -> None:
        """Close UDP sender."""
        self._sender.close()