from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from src.camera.types import BoundingBox, FaceDetection
from src.exceptions import ModelError
from src.utils.logger import get_logger


class FaceDetector:
    def __init__(
        self,
        model_selection: int = 0,
        min_confidence: float = 0.3,
    ):
        self._model_selection = model_selection
        self._min_confidence = min_confidence
        self._logger = get_logger(__name__)
        self._face_mesh = None
        self._mp_drawing = None
        self._mp_face_mesh = None
        self._debug = False

    def load(self) -> None:
        try:
            self._mp_face_mesh = mp.solutions.face_mesh
            self._mp_drawing = mp.solutions.drawing_utils
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=3,
                refine_landmarks=False,
                min_detection_confidence=self._min_confidence,
                min_tracking_confidence=0.5,
            )
            self._logger.info("Face detector loaded successfully")
        except Exception as e:
            raise ModelError(f"Failed to load face detector: {e}")

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        if self._face_mesh is None:
            raise ModelError("Model not loaded. Call load() first.")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(frame_rgb)

        faces = []

        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                x_coords = [lm.x * w for lm in face_landmarks.landmark]
                y_coords = [lm.y * h for lm in face_landmarks.landmark]

                x1 = max(0, min(x_coords))
                y1 = max(0, min(y_coords))
                x2 = min(w, max(x_coords))
                y2 = min(h, max(y_coords))

                face_confidence = 0.5 + (0.4 * (1.0 - idx * 0.1))

                face = FaceDetection(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=face_confidence,
                )
                faces.append(face)

        return faces

    def draw_faces(self, frame: np.ndarray, faces: List[FaceDetection]) -> np.ndarray:
        for face in faces:
            bbox = face.bbox
            cv2.rectangle(
                frame,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Face {face.confidence:.2f}",
                (int(bbox.x1), int(bbox.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
        return frame

    @property
    def name(self) -> str:
        return "MediaPipeFace"

    @property
    def is_loaded(self) -> bool:
        return self._face_mesh is not None
