from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.camera.types import PoseData, PoseKeypoint
from src.exceptions import ModelError
from src.utils.logger import get_logger


class PoseEstimator:
    def __init__(
        self,
        min_confidence: float = 0.5,
    ):
        self._min_confidence = min_confidence
        self._logger = get_logger(__name__)
        self._pose = None
        self._mp_pose = None
        self._mp_drawing = None

    def load(self) -> None:
        try:
            self._mp_pose = mp.solutions.pose
            self._mp_drawing = mp.solutions.drawing_utils
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=self._min_confidence,
                min_tracking_confidence=0.5,
            )
            self._logger.info("Pose estimator loaded successfully")
        except Exception as e:
            raise ModelError(f"Failed to load pose estimator: {e}")

    def estimate(self, frame: np.ndarray) -> PoseData:
        if self._pose is None:
            raise ModelError("Model not loaded. Call load() first.")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(frame_rgb)

        if not results.pose_landmarks:
            return PoseData(detected=False, keypoints=None)

        keypoints = []
        h, w = frame.shape[:2]

        for landmark in results.pose_landmarks.landmark:
            keypoint = PoseKeypoint(
                x=landmark.x * w,
                y=landmark.y * h,
                visibility=landmark.visibility,
            )
            keypoints.append(keypoint)

        return PoseData(detected=True, keypoints=keypoints)

    def draw_pose(self, frame: np.ndarray, pose_data: PoseData) -> np.ndarray:
        if not pose_data.detected or not pose_data.keypoints:
            return frame

        h, w = frame.shape[:2]
        connections = self._mp_pose.POSE_CONNECTIONS

        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(pose_data.keypoints) and end_idx < len(pose_data.keypoints):
                start = pose_data.keypoints[start_idx]
                end = pose_data.keypoints[end_idx]

                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x), int(start.y))
                    end_point = (int(end.x), int(end.y))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        for idx, keypoint in enumerate(pose_data.keypoints):
            if keypoint.visibility > 0.5:
                cv2.circle(
                    frame,
                    (int(keypoint.x), int(keypoint.y)),
                    4,
                    (0, 0, 255),
                    -1,
                )

        return frame

    @property
    def name(self) -> str:
        return "MediaPipePose"

    @property
    def is_loaded(self) -> bool:
        return self._pose is not None
