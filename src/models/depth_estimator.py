from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.exceptions import ModelError
from src.utils.logger import get_logger


class DepthEstimator:
    """Depth estimator using MiDaS with ONNX optimization."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.5,
    ):
        self._model_path = model_path
        self._confidence = confidence
        self._logger = get_logger(__name__)
        self._model = None
        self._transform = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._baseline = 0.075
        self._focal_length = 500.0

    def load(self) -> None:
        """Load the MiDaS depth estimation model."""
        try:
            self._logger.info("Loading MiDaS depth estimation model...")

            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas.to(self._device)
            midas.eval()
            self._model = midas

            try:
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self._transform = midas_transforms.small_transform
            except:
                self._transform = self._get_fallback_transform()

            self._logger.info(
                f"MiDaS loaded on device: {self._device}"
            )

        except Exception as e:
            self._logger.warning(
                f"Failed to load MiDaS: {e}. Using fallback depth estimation."
            )
            self._model = None

    def _get_fallback_transform(self):
        """Get fallback transform if official transforms unavailable."""
        import torchvision.transforms as transforms

        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map from the given frame."""
        if self._model is None:
            return self._fallback_depth(frame)

        try:
            return self._estimate_depth(frame)
        except Exception as e:
            self._logger.warning(f"Depth estimation failed: {e}")
            return self._fallback_depth(frame)

    def _estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using MiDaS."""
        h, w = frame.shape[:2]

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_batch = self._transform(img)
        input_batch = input_batch.to(self._device)

        with torch.no_grad():
            prediction = self._model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy()

        depth = np.clip(depth, 0, 1)

        return depth

    def _fallback_depth(self, frame: np.ndarray) -> np.ndarray:
        """Fallback depth estimation using monocular cues."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 50, 150)

        depth = cv2.distanceTransform(
            255 - edges, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )

        depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

        return depth.astype(np.float32)

    def set_focal_length(self, focal_length: float) -> None:
        """Set camera focal length for distance calculation."""
        self._focal_length = focal_length

    def distance_to_depth(self, distance_m: float) -> float:
        """Convert real distance (meters) to depth value."""
        if self._focal_length is None:
            return 0.0
        return self._focal_length / (distance_m * 1000) if distance_m > 0 else 0.0

    def depth_to_distance(self, depth_value: float) -> float:
        """Convert depth value to real distance (meters)."""
        if depth_value == 0:
            return float("inf")
        if self._focal_length is None:
            return 0.0
        return self._focal_length / (depth_value * 1000)

    def get_depth_at_point(self, depth_map: np.ndarray, x: int, y: int) -> float:
        """Get depth value at a specific point."""
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return float(depth_map[y, x])
        return 0.0

    def get_distance_at_point(
        self, depth_map: np.ndarray, x: int, y: int
    ) -> float:
        """Get distance in meters at a specific point."""
        depth = self.get_depth_at_point(depth_map, x, y)
        return self.depth_to_distance(depth)

    @property
    def name(self) -> str:
        return "MiDaS"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
