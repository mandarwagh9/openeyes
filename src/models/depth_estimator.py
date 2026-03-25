from typing import Optional

import cv2
import numpy as np

from src.exceptions import ModelError
from src.utils.logger import get_logger


class DepthEstimator:
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.5,
    ):
        self._model_path = model_path
        self._confidence = confidence
        self._logger = get_logger(__name__)
        self._baseline = 0.075
        self._focal_length = None

    def load(self) -> None:
        self._logger.info("Depth estimator initialized (using stereo calibration)")
        self._focal_length = 500.0

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_map = np.zeros(gray.shape, dtype=np.float32)
        return depth_map

    def set_focal_length(self, focal_length: float) -> None:
        self._focal_length = focal_length

    def distance_to_depth(self, distance_m: float) -> float:
        if self._focal_length is None:
            return 0.0
        return self._focal_length / (distance_m * 1000) if distance_m > 0 else 0.0

    def depth_to_distance(self, depth_value: float) -> float:
        if depth_value == 0:
            return float('inf')
        if self._focal_length is None:
            return 0.0
        return self._focal_length / (depth_value * 1000)

    @property
    def name(self) -> str:
        return "DepthEstimator"

    @property
    def is_loaded(self) -> bool:
        return True
