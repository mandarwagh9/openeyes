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
                max_num_hands=2,
                min_detection_confidence=self._min_confidence,
                min_tracking_confidence=0.5,
            )
            self._logger.info("Gesture recognizer loaded successfully")
        except Exception as e:
            raise ModelError(f"Failed to load gesture recognizer: {e}")

    def recognize(self, frame: np.ndarray) -> List[GestureType]:
        if self._hands is None:
            raise ModelError("Model not loaded. Call load() first.")

        if self._debug:
            self._logger.info(f"[GESTURE] Processing frame shape: {frame.shape}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        gestures = []

        if results.multi_hand_landmarks:
            if self._debug:
                self._logger.info(f"[GESTURE] Found {len(results.multi_hand_landmarks)} hand(s)")
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
        elif self._debug:
            self._logger.debug("No hands detected in frame")

        return gestures

    def _classify_gesture(self, landmarks) -> str:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        thumb_ip = landmarks[3]
        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]

        fingers = []
        fingers.append(thumb_tip.x > thumb_ip.x)
        fingers.append(index_tip.y < index_pip.y)
        fingers.append(middle_tip.y < middle_pip.y)
        fingers.append(ring_tip.y < ring_pip.y)
        fingers.append(pinky_tip.y < pinky_pip.y)

        fingers_count = sum(fingers)

        if fingers_count == 5:
            return "open_palm"
        elif fingers_count == 0:
            return "fist"
        elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            return "peace"
        elif fingers[1] and not fingers[2]:
            return "point"
        elif fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            return "thumbs_up"
        elif fingers[0] and not fingers[1]:
            return "ok_sign"
        else:
            return f"unknown_{fingers_count}"

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
