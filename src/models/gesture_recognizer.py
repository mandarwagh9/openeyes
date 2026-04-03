from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from src.camera.types import Gesture as GestureType
from src.exceptions import ModelError
from src.utils.logger import get_logger


class GestureRecognizer:
    def __init__(
        self,
        min_confidence: float = 0.3,
    ):
        self._min_confidence = min_confidence
        self._logger = get_logger(__name__)
        self._hands = None
        self._mp_hands = None
        self._mp_drawing = None
        self._debug = False

    def load(self) -> None:
        try:
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                model_complexity=0,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._logger.info("Gesture recognizer loaded successfully")
        except Exception as e:
            raise ModelError(f"Failed to load gesture recognizer: {e}")

    def recognize(self, frame: np.ndarray) -> List[GestureType]:
        if self._hands is None:
            raise ModelError("Model not loaded. Call load() first.")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        gestures = []

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = "right"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label
                    handedness = handedness.lower()

                gesture_type = self._classify_gesture(hand_landmarks.landmark)

                gesture_confidence = 0.5 + (0.3 * (1.0 - idx * 0.1))

                gesture = GestureType(
                    gesture_type=gesture_type,
                    handedness=handedness,
                    confidence=gesture_confidence,
                )
                gestures.append(gesture)

        return gestures

    def _classify_gesture(self, landmarks) -> str:
        fingertips = [
            landmarks[8].y,   # index
            landmarks[12].y,  # middle
            landmarks[16].y,  # ring
            landmarks[20].y,  # pinky
        ]
        finger_pips = [
            landmarks[6].y,   # index
            landmarks[10].y,  # middle
            landmarks[14].y,  # ring
            landmarks[18].y,  # pinky
        ]

        fingers_up = sum(1 for tip, pip in zip(fingertips, finger_pips) if tip < pip)

        thumb_tip = landmarks[4]
        index_base = landmarks[5]
        thumb_up = thumb_tip.y < index_base.y

        if self._debug:
            self._logger.debug(f"Gesture: {fingers_up} fingers up, thumb_up={thumb_up}")

        if fingers_up == 0 and not thumb_up:
            return "fist"
        elif fingers_up == 0 and thumb_up:
            return "thumbs_down"
        elif fingers_up == 1:
            return "point"
        elif fingers_up == 2:
            return "peace"
        elif fingers_up == 3:
            return "three"
        elif fingers_up == 4 and not thumb_up:
            return "open_palm"
        elif fingers_up == 4 and thumb_up:
            return "thumbs_up"
        else:
            return f"fingers_{fingers_up}"

    def draw_gestures(self, frame: np.ndarray, gestures: List[GestureType]) -> np.ndarray:
        for gesture in gestures:
            label = f"{gesture.gesture_type} ({gesture.handedness})"
            cv2.putText(
                frame,
                label,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        return frame

    @property
    def name(self) -> str:
        return "MediaPipeHands"

    @property
    def is_loaded(self) -> bool:
        return self._hands is not None
