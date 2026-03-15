import time
from typing import Optional

import cv2
import numpy as np

from src.camera.types import CameraInterface
from src.exceptions import CameraError
from src.utils.logger import get_logger


class CameraHandler:
    def __init__(
        self,
        source: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._logger = get_logger(__name__)
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        if not self.is_opened:
            raise CameraError(f"Failed to open camera source: {self._source}")

        self._logger.info(
            f"Camera opened: {self._width}x{self._height} @ {self._fps} FPS"
        )

    def read(self) -> Optional[np.ndarray]:
        if not self.is_opened:
            if not self._attempt_reconnect():
                return None

        if self._cap is None:
            return None

        ret, frame = self._cap.read()

        if not ret:
            self._logger.warning("Failed to read frame from camera")
            if self._attempt_reconnect():
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                return None

        self._frame_count += 1
        return frame

    def _attempt_reconnect(self) -> bool:
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self._logger.error("Max reconnection attempts reached")
            return False

        self._reconnect_attempts += 1
        self._logger.info(f"Attempting to reconnect (attempt {self._reconnect_attempts})")

        if self._cap is not None:
            self._cap.release()
            time.sleep(0.5)

        self._cap = cv2.VideoCapture(self._source)

        if self.is_opened:
            self._reconnect_attempts = 0
            self._logger.info("Camera reconnected successfully")
            return True

        return False

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._logger.info("Camera released")

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        if self._cap is not None:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return self._width

    @property
    def height(self) -> int:
        if self._cap is not None:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self._height
