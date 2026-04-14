from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from src.exceptions import ModelError
from src.utils.logger import get_logger


class SelfieSegmentation:
    def __init__(
        self,
        model_selection: int = 0,
    ):
        self._model_selection = model_selection
        self._logger = get_logger(__name__)
        self._segmenter = None
        self._debug = False

    def load(self) -> None:
        try:
            self._mp_segmentation = mp.solutions.selfie_segmentation
            self._segmenter = self._mp_segmentation.SelfieSegmentation(
                model_selection=self._model_selection
            )
            self._logger.info(
                f"Selfie segmentation loaded (model_selection={self._model_selection})"
            )
        except Exception as e:
            raise ModelError(f"Failed to load selfie segmentation: {e}")

    def segment(self, frame: np.ndarray) -> np.ndarray:
        if self._segmenter is None:
            raise ModelError("Model not loaded. Call load() first.")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._segmenter.process(frame_rgb)

        if results.segmentation_mask is None:
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        mask = cv2.resize(
            results.segmentation_mask,
            (frame.shape[1], frame.shape[0])
        )
        mask = (mask * 255).astype(np.uint8)
        return mask

    def segment_with_threshold(
        self, frame: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        mask = self.segment(frame)
        _, binary = cv2.threshold(mask, int(threshold * 255), 255, cv2.THRESH_BINARY)
        return binary

    def get_foreground_mask(
        self, frame: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        return self.segment_with_threshold(frame, threshold)

    def get_background_mask(
        self, frame: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        foreground = self.segment_with_threshold(frame, threshold)
        background = cv2.bitwise_not(foreground)
        return background

    def apply_mask(
        self, frame: np.ndarray, background: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        mask = self.segment_with_threshold(frame, threshold)

        if background.shape != frame.shape:
            background = cv2.resize(
                background,
                (frame.shape[1], frame.shape[0])
            )

        if len(background.shape) == 2:
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        foreground = cv2.bitwise_and(frame, frame, mask=mask)
        bg_mask = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(background, background, mask=bg_mask)

        result = cv2.add(foreground, bg)
        return result

    def draw_segmentation(
        self, frame: np.ndarray, mask: np.ndarray, alpha: float = 0.3
    ) -> np.ndarray:
        if not self._debug:
            return frame

        colored_mask = np.zeros_like(frame)
        colored_mask[:, :, 1] = mask

        result = cv2.addWeighted(frame, 1, colored_mask, alpha, 0)
        return result

    @property
    def name(self) -> str:
        return "MediaPipeSelfieSegmentation"

    @property
    def is_loaded(self) -> bool:
        return self._segmenter is not None