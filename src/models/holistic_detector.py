from typing import List, Optional, Dict, Any

import cv2
import mediapipe as mp
import numpy as np

from src.camera.types import BoundingBox, FaceDetection, GestureDetection, PoseDetection
from src.exceptions import ModelError
from src.utils.logger import get_logger


class HolisticDetector:
    def __init__(
        self,
        model_complexity: int = 1,
        min_face_confidence: float = 0.3,
        min_pose_confidence: float = 0.3,
        min_hand_confidence: float = 0.1,
        enable_face: bool = True,
        enable_pose: bool = True,
        enable_hands: bool = True,
    ):
        self._model_complexity = model_complexity
        self._min_face_confidence = min_face_confidence
        self._min_pose_confidence = min_pose_confidence
        self._min_hand_confidence = min_hand_confidence
        self._enable_face = enable_face
        self._enable_pose = enable_pose
        self._enable_hands = enable_hands
        self._logger = get_logger(__name__)
        self._holistic = None
        self._mp_drawing = None
        self._debug = False

    def load(self) -> None:
        try:
            self._mp_holistic = mp.solutions.holistic
            self._mp_drawing = mp.solutions.drawing_utils
            self._holistic = self._mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=self._model_complexity,
                enable_face_landmarks=self._enable_face,
                enable_pose_landmarks=self._enable_pose,
                enable_hand_landmarks=self._enable_hands,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._logger.info(
                f"Holistic detector loaded (face={self._enable_face}, "
                f"pose={self._enable_pose}, hands={self._enable_hands})"
            )
        except Exception as e:
            raise ModelError(f"Failed to load holistic detector: {e}")

    def detect(
        self, frame: np.ndarray
    ) -> Dict[str, List[Any]]:
        if self._holistic is None:
            raise ModelError("Model not loaded. Call load() first.")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._holistic.process(frame_rgb)

        return {
            "faces": self._detect_faces(frame, results),
            "poses": self._detect_pose(frame, results),
            "hands": self._detect_hands(frame, results),
        }

    def _detect_faces(self, frame: np.ndarray, results) -> List[FaceDetection]:
        faces = []
        if not self._enable_face or not results.face_landmarks:
            return faces

        h, w = frame.shape[:2]
        for face_landmarks in results.face_landmarks:
            x_coords = [lm.x * w for lm in face_landmarks.landmark]
            y_coords = [lm.y * h for lm in face_landmarks.landmark]

            x1 = max(0, min(x_coords))
            y1 = max(0, min(y_coords))
            x2 = min(w, max(x_coords))
            y2 = min(h, max(y_coords))

            face = FaceDetection(
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=self._min_face_confidence,
            )
            faces.append(face)

        return faces

    def _detect_pose(self, frame: np.ndarray, results) -> List[PoseDetection]:
        poses = []
        if not self._enable_pose or not results.pose_landmarks:
            return poses

        h, w = frame.shape[:2]
        landmarks = results.pose_landmarks.landmark

        min_x = min(lm.x for lm in landmarks) * w
        max_x = max(lm.x for lm in landmarks) * w
        min_y = min(lm.y for lm in landmarks) * h
        max_y = max(lm.y for lm in landmarks) * h

        keypoints = {}
        for idx, name in HolisticDetector._POSE_landmarks:
            if idx < len(landmarks):
                keypoints[name] = (
                    landmarks[idx].x * w,
                    landmarks[idx].y * h,
                    landmarks[idx].visibility,
                )

        pose = PoseDetection(
            bbox=BoundingBox(x1=min_x, y1=min_y, x2=max_x, y2=max_y),
            keypoints=keypoints,
            confidence=self._min_pose_confidence,
        )
        poses.append(pose)

        return poses

    def _detect_hands(self, frame: np.ndarray, results) -> List[GestureDetection]:
        gestures = []
        if not self._enable_hands:
            return gestures

        h, w = frame.shape[:2]

        left_hand = results.left_hand_landmarks
        right_hand = results.right_hand_landmarks

        if left_hand:
            gestures.extend(self._process_hand(left_hand, "left", w, h))
        if right_hand:
            gestures.extend(self._process_hand(right_hand, "right", w, h))

        return gestures

    def _process_hand(self, hand_landmarks, hand_type: str, w: int, h: int):
        gestures = []
        gestures_list = self._classify_hand(hand_landmarks, w, h)

        for gesture_name, bbox in gestures_list:
            gesture = GestureDetection(
                gesture_name=gesture_name,
                hand_type=hand_type,
                bbox=bbox,
                confidence=self._min_hand_confidence,
            )
            gestures.append(gesture)

        return gestures

    def _classify_hand(self, hand_landmarks, w: int, h: int):
        gestures = []
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        x1 = max(0, min(x_coords) - 20)
        y1 = max(0, min(y_coords) - 20)
        x2 = min(w, max(x_coords) + 20)
        y2 = min(h, max(y_coords) + 20)

        fingers = self._count_fingers(hand_landmarks)

        gesture_name = self._identify_gesture(fingers)
        gestures.append((gesture_name, BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)))

        return gestures

    def _count_fingers(self, hand_landmarks):
        fingers = []

        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        fingers.append(thumb_tip.x > thumb_ip.x)

        for tip_id in [8, 12, 16, 20]:
            tip = hand_landmarks.landmark[tip_id]
            pip = hand_landmarks.landmark[tip_id - 2]
            fingers.append(tip.y < pip.y)

        return fingers

    def _identify_gesture(self, fingers):
        if fingers == [False, False, False, False, False]:
            return "fist"
        elif fingers == [False, True, True, False, False]:
            return "peace"
        elif fingers == [True, False, False, False, False]:
            return "point"
        elif fingers == [False, True, True, True, True]:
            return "open_palm"
        elif fingers == [True, True, False, False, False]:
            return "thumbs_up"
        elif fingers == [False, False, True, True, True]:
            return "thumbs_down"
        else:
            return "unknown"

    _POSE_landmarks = [
        (11, "left_shoulder"),
        (12, "right_shoulder"),
        (13, "left_elbow"),
        (14, "right_elbow"),
        (15, "left_wrist"),
        (16, "right_wrist"),
        (23, "left_hip"),
        (24, "right_hip"),
        (25, "left_knee"),
        (26, "right_knee"),
        (27, "left_ankle"),
        (28, "right_ankle"),
    ]

    def draw_results(
        self, frame: np.ndarray, results: Dict[str, List[Any]]
    ) -> np.ndarray:
        if self._debug:
            if results.get("faces"):
                for face in results["faces"]:
                    bbox = face.bbox
                    cv2.rectangle(
                        frame,
                        (int(bbox.x1), int(bbox.y1)),
                        (int(bbox.x2), int(bbox.y2)),
                        (255, 0, 0),
                        2,
                    )

            if results.get("poses"):
                for pose in results["poses"]:
                    keypoints = pose.keypoints
                    for name, (x, y, vis) in keypoints.items():
                        if vis > 0.5:
                            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            if results.get("hands"):
                for hand in results["hands"]:
                    bbox = hand.bbox
                    cv2.rectangle(
                        frame,
                        (int(bbox.x1), int(bbox.y1)),
                        (int(bbox.x2), int(bbox.y2)),
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        hand.gesture_name,
                        (int(bbox.x1), int(bbox.y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

        return frame

    @property
    def name(self) -> str:
        return "MediaPipeHolistic"

    @property
    def is_loaded(self) -> bool:
        return self._holistic is not None