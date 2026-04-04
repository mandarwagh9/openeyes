"""
Video Source for OpenEyes - Process pre-recorded video files.

Drop-in replacement for CameraHandler when using --video flag.
"""

import os
from typing import Optional

import cv2
import numpy as np

from src.exceptions import CameraError
from src.utils.logger import get_logger


class VideoSource:
    """Video file source for processing pre-recorded footage."""

    def __init__(
        self,
        path: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        if not os.path.exists(path):
            raise CameraError(f"Video file not found: {path}")

        self._path = path
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._total_frames = 0
        self._logger = get_logger(__name__)

    def open(self) -> None:
        """Open the video file."""
        self._logger.info(f"Opening video: {self._path}")
        self._cap = cv2.VideoCapture(self._path)

        if not self._cap.isOpened():
            raise CameraError(f"Failed to open video file: {self._path}")

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = self._cap.get(cv2.CAP_PROP_FPS)
        src_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._logger.info(
            f"Video opened: {src_w}x{src_h} @ {src_fps:.1f} FPS, "
            f"{self._total_frames} frames"
        )

    def read(self) -> Optional[np.ndarray]:
        """Read the next frame from the video."""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        self._frame_count += 1
        return frame

    def release(self) -> None:
        """Release the video resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._logger.info("Video source released")

    @property
    def is_opened(self) -> bool:
        """Check if video is opened."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_count(self) -> int:
        """Get frames captured so far."""
        return self._frame_count

    @property
    def total_frames(self) -> int:
        """Get total frames in the video."""
        return self._total_frames

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

    @property
    def progress(self) -> float:
        """Get playback progress (0.0 to 1.0)."""
        if self._total_frames == 0:
            return 0.0
        return self._frame_count / self._total_frames
