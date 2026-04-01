"""
Camera Handler for OpenEyes - Jetson Orin Nano Optimized

Supports:
- CSI Camera (IMX219, IMX477) via nvarguscamerasrc
- USB Webcam via V4L2
- Auto-detection of Jetson platform
"""

import os
import subprocess
import time
from typing import Optional

import cv2
import numpy as np

from src.camera.types import CameraInterface
from src.exceptions import CameraError
from src.utils.logger import get_logger


class CameraHandler:
    """Camera handler optimized for Jetson Orin Nano with CSI camera support."""

    def __init__(
        self,
        source: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        if source < 0:
            raise ValueError(f"Camera source must be non-negative, got {source}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Camera width and height must be positive, got {width}x{height}")
        if fps <= 0:
            raise ValueError(f"Camera FPS must be positive, got {fps}")
        
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._logger = get_logger(__name__)
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3
        self._is_jetson = self._detect_jetson()

    @staticmethod
    def _detect_jetson() -> bool:
        """Detect if running on NVIDIA Jetson platform."""
        if os.environ.get("OPENEYES_TEST_MODE") == "true":
            return False
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
                return "jetson" in model or "tegra" in model
        except Exception:
            return False

    def _check_csi_available(self) -> bool:
        """Check if CSI camera device exists."""
        device_path = f"/dev/video{self._source}"
        exists = os.path.exists(device_path)
        if exists:
            self._logger.debug(f"CSI camera device found at {device_path}")
        else:
            self._logger.debug(f"CSI camera device not found at {device_path}")
        return exists

    def open(self) -> None:
        """Open the camera with appropriate backend for the platform."""
        self._logger.info(f"Opening camera (source={self._source}, {self._width}x{self._height}@{self._fps})")
        self._logger.info(f"Jetson platform detected: {self._is_jetson}")
        self._logger.info(f"CSI camera device exists: {self._check_csi_available()}")

        if self._is_jetson and self._check_csi_available():
            if self._try_jetson_csi():
                self._logger.info("Using Jetson CSI camera (nvarguscamerasrc)")
                return

        if self._try_usb_camera():
            self._logger.info("Using USB camera")
            return

        self._logger.error("All camera methods failed")
        raise CameraError("Failed to open camera. No available camera found.")

    def _get_jetson_pipeline(self, width: int, height: int, fps: int) -> str:
        """Build GStreamer pipeline for Jetson CSI camera."""
        return (
            f"nvarguscamerasrc sensor-id={self._source} ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},format=NV12,framerate={fps}/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw,format=BGRx ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "queue ! "
            "appsink drop=True"
        )

    def _try_jetson_csi(self) -> bool:
        """Try to open CSI camera using nvarguscamerasrc."""
        self._logger.info("Trying Jetson CSI camera...")

        resolutions = [
            (1920, 1080),
            (1280, 720),
            (self._width, self._height),
            (640, 480),
        ]

        for width, height in resolutions:
            pipeline = self._get_jetson_pipeline(width, height, self._fps)
            self._logger.debug(f"Trying CSI pipeline: {width}x{height}")

            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            if self._cap is not None and self._cap.isOpened():
                time.sleep(0.1)
                
                for _ in range(3):
                    ret, frame = self._cap.read()
                    if ret and frame is not None:
                        self._width, self._height = frame.shape[1], frame.shape[0]
                        self._logger.info(f"CSI camera opened: {self._width}x{self._height}")
                        return True
                self._cap.release()

            self._cap = None
            self._logger.debug(f"Failed to open at {width}x{height}")

        self._logger.info("Trying default CSI pipeline (1920x1080)...")
        pipeline = self._get_jetson_pipeline(1920, 1080, 30)
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if self._cap is not None and self._cap.isOpened():
            time.sleep(0.2)
            for _ in range(5):
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    self._height, self._width = frame.shape[:2]
                    self._logger.info(f"CSI camera opened (default): {self._width}x{self._height}")
                    return True
            self._cap.release()

        self._cap = None
        return False

    def _try_usb_camera(self) -> bool:
        """Try to open USB camera."""
        self._logger.info("Trying USB camera...")

        # Try with V4L2 first
        self._cap = cv2.VideoCapture(self._source, cv2.CAP_V4L2)

        if self._cap is not None and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._height, self._width = frame.shape[:2]
                return True
            self._cap.release()

        self._cap = None

        # Try with DirectShow (Windows) or standard V4L2
        self._cap = cv2.VideoCapture(self._source)
        if self._cap is not None and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._height, self._width = frame.shape[:2]
                return True
            self._cap.release()

        self._cap = None
        return False

    def read(self) -> Optional[np.ndarray]:
        """Read a frame from the camera."""
        if not self.is_opened:
            if not self._attempt_reconnect():
                return None

        if self._cap is None:
            return None

        ret, frame = self._cap.read()

        if not ret or frame is None:
            self._logger.warning("Failed to read frame from camera")
            if self._attempt_reconnect():
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    return None
            else:
                return None

        self._frame_count += 1
        if self._frame_count % 300 == 0:
            self._logger.debug(f"Frame count: {self._frame_count}")
        return frame

    def _attempt_reconnect(self) -> bool:
        """Attempt to reconnect to the camera."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self._logger.error("Max reconnection attempts reached")
            return False

        self._reconnect_attempts += 1
        self._logger.info(f"Attempting to reconnect (attempt {self._reconnect_attempts})")

        if self._cap is not None:
            self._cap.release()
            self._cap = None
            time.sleep(0.5)

        # Try restarting the Argus daemon on Jetson
        if self._is_jetson:
            try:
                subprocess.run(
                    ["sudo", "systemctl", "restart", "nvargus-daemon"],
                    capture_output=True,
                    timeout=5,
                )
                time.sleep(1)
            except Exception as e:
                self._logger.warning(f"Failed to restart Argus daemon: {e}")

        self.open()

        if self.is_opened:
            self._reconnect_attempts = 0
            self._logger.info("Camera reconnected successfully")
            return True

        return False

    def release(self) -> None:
        """Release the camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._logger.info("Camera released")

    @property
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_count(self) -> int:
        """Get total frames captured."""
        return self._frame_count

    @property
    def width(self) -> int:
        """Get frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Get frame height."""
        return self._height

    @property
    def fps(self) -> int:
        """Get target FPS."""
        return self._fps
