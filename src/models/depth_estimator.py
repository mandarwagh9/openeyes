"""Unified depth estimation module.

Supports multiple depth estimation backends:
- MiDaS (legacy, lightweight)
- Depth Anything V3 (SOTA, recommended)

Usage:
    # Default: Depth Anything V3
    estimator = DepthEstimator(model="da3-small")
    
    # Legacy MiDaS
    estimator = DepthEstimator(model="midas-small")
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.exceptions import ModelError
from src.utils.logger import get_logger


class DepthEstimator:
    """Unified depth estimator supporting MiDaS and Depth Anything V2."""

    SUPPORTED_MODELS = {
        "da3-small": {
            "name": "Depth Anything V2 Small",
            "params": 25,
            "recommended_fps": 15,
        },
        "da3-base": {
            "name": "Depth Anything V2 Base",
            "params": 98,
            "recommended_fps": 8,
        },
        "midas-small": {
            "name": "MiDaS Small",
            "params": 5,
            "recommended_fps": 20,
        },
    }

    def __init__(
        self,
        model: str = "da3-small",
        model_path: Optional[str] = None,
        confidence: float = 0.5,
    ):
        self._model_name = model
        self._model_path = model_path
        self._confidence = confidence
        self._logger = get_logger(__name__)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._baseline = 0.075
        self._focal_length = 500.0

        self._model = None
        self._transform = None
        self._processor = None
        self._is_loaded = False

        if model not in self.SUPPORTED_MODELS:
            self._logger.warning(
                f"Unknown depth model: {model}. "
                f"Falling back to da3-small. "
                f"Available: {list(self.SUPPORTED_MODELS.keys())}"
            )
            self._model_name = "da3-small"

    def load(self) -> None:
        """Load the depth estimation model."""
        if self._model_name.startswith("da3"):
            self._load_depth_anything_v3()
        else:
            self._load_midas()

    def _load_depth_anything_v3(self) -> None:
        """Load Depth Anything V2 model (V3 not available on HuggingFace)."""
        self._logger.info(f"Loading Depth Anything V2 ({self._model_name})...")

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            hf_ids = {
                "da3-small": "depth-anything/Depth-Anything-V2-Small-hf",
                "da3-base": "depth-anything/Depth-Anything-V2-Base-hf",
            }
            hf_id = hf_ids.get(self._model_name, hf_ids["da3-small"])

            self._processor = AutoImageProcessor.from_pretrained(hf_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(
                hf_id,
                torch_dtype="auto",
            )

            if self._device.type == "cuda":
                self._model = self._model.to("cuda")
            self._model.eval()

            self._is_loaded = True
            info = self.SUPPORTED_MODELS[self._model_name]
            self._logger.info(
                f"Depth Anything V2 loaded: "
                f"{info['params']}M params on {self._device}"
            )

        except ImportError:
            self._logger.warning(
                "transformers not installed. Falling back to MiDaS. "
                "Install with: pip install transformers"
            )
            self._model_name = "midas-small"
            self._load_midas()

        except Exception as e:
            self._logger.warning(f"Failed to load Depth Anything V3: {e}")
            self._is_loaded = False

    def _load_midas(self) -> None:
        """Load MiDaS depth estimation model."""
        try:
            self._logger.info("Loading MiDaS depth estimation model...")

            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas.to(self._device)
            midas.eval()
            self._model = midas

            try:
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self._transform = midas_transforms.small_transform
            except Exception:
                self._transform = self._get_fallback_transform()

            self._is_loaded = True
            self._logger.info(f"MiDaS loaded on device: {self._device}")

        except Exception as e:
            self._logger.warning(f"Failed to load MiDaS: {e}")
            self._model = None
            self._is_loaded = False

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
        if not self._is_loaded or self._model is None:
            return self._fallback_depth(frame)

        try:
            if self._model_name.startswith("da3"):
                return self._estimate_depth_da3(frame)
            else:
                return self._estimate_depth_midas(frame)
        except Exception as e:
            self._logger.warning(f"Depth estimation failed: {e}")
            return self._fallback_depth(frame)

    def _estimate_depth_da3(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using Depth Anything V3."""
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        inputs = self._processor(images=pil_image, return_tensors="pt")

        if self._device.type == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth

        depth = predicted_depth.squeeze().cpu().numpy()

        target_h, target_w = frame.shape[:2]
        depth_resized = cv2.resize(
            depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        depth_normalized = (depth_resized - depth_resized.min()) / (
            depth_resized.max() - depth_resized.min() + 1e-8
        )

        return depth_normalized.astype(np.float32)

    def _estimate_depth_midas(self, frame: np.ndarray) -> np.ndarray:
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
        return self.SUPPORTED_MODELS.get(self._model_name, {}).get(
            "name", self._model_name
        )

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self._model is not None
