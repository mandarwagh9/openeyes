from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from src.camera.types import BoundingBox, Detection
from src.exceptions import ModelError
from src.utils.logger import get_logger


class ObjectDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        self._model_path = model_path
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._model: Optional[YOLO] = None
        self._logger = get_logger(__name__)

    def load(self) -> None:
        model_file = Path(self._model_path)

        if not model_file.exists():
            self._logger.warning(
                f"Model file not found: {self._model_path}. "
                "YOLOv8 will download on first use."
            )

        try:
            self._model = YOLO(self._model_path)
            self._logger.info(f"Object detector loaded: {self._model_path}")
        except Exception as e:
            raise ModelError(f"Failed to load model: {e}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._model is None:
            raise ModelError("Model not loaded. Call load() first.")

        results = self._model(
            frame,
            conf=self._confidence,
            iou=self._iou_threshold,
            verbose=False,
        )

        detections = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]

                    detection = Detection(
                        class_name=class_name,
                        bbox=BoundingBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                        ),
                        confidence=conf,
                    )
                    detections.append(detection)

        return detections

    @property
    def name(self) -> str:
        return "YOLOv8"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
