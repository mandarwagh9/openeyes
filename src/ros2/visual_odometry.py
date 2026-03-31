import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import time

from src.utils.logger import get_logger


@dataclass
class OdometryData:
    """Visual odometry data output."""
    translation: np.ndarray
    rotation: np.ndarray
    confidence: float
    timestamp: float
    frame_count: int


class VisualOdometry:
    """Compute visual odometry from camera frames using optical flow.
    
    Uses Lucas-Kanade optical flow to estimate frame-to-frame motion.
    This provides relative odometry that can be integrated for navigation.
    """

    def __init__(
        self,
        focal_length: float = 500.0,
        baseline: float = 0.1,
        max_features: int = 100,
        quality_level: float = 0.01,
        min_distance: int = 10,
    ):
        self._logger = get_logger(__name__)
        self._focal_length = focal_length
        self._baseline = baseline
        self._max_features = max_features
        self._quality_level = quality_level
        self._min_distance = min_distance

        self._prev_frame: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None
        self._frame_count = 0

        self._total_translation = np.array([0.0, 0.0, 0.0])
        self._total_rotation = np.array([0.0, 0.0, 0.0])

        self._logger.info(
            f"VisualOdometry initialized: focal={focal_length}, "
            f"baseline={baseline}, max_features={max_features}"
        )

    def process_frame(self, frame: np.ndarray) -> Optional[OdometryData]:
        """Process a new frame and compute odometry.
        
        Args:
            frame: Input BGR frame (H, W, 3)
            
        Returns:
            OdometryData with translation/rotation, or None if tracking not ready
        """
        if frame is None or frame.size == 0:
            return None

        self._frame_count += 1
        timestamp = time.time()

        gray = self._to_gray(frame)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_frame = frame
            self._prev_points = self._detect_features(gray)
            return None

        current_points, status, error = self._track_features(
            self._prev_gray, gray, self._prev_points
        )

        if current_points is None or status is None or len(current_points) < 10:
            self._prev_gray = gray
            self._prev_frame = frame
            self._prev_points = self._detect_features(gray)
            return None

        valid = status.flatten() == 1
        if np.sum(valid) < 10:
            self._prev_gray = gray
            self._prev_frame = frame
            self._prev_points = self._detect_features(gray)
            return None

        prev_pts = self._prev_points[valid]
        curr_pts = current_points[valid]

        translation, rotation = self._estimate_motion(prev_pts, curr_pts, error[valid])

        self._total_translation += translation
        self._total_rotation += rotation

        confidence = self._compute_confidence(error[valid])

        self._prev_gray = gray
        self._prev_frame = frame
        self._prev_points = self._detect_features(gray)

        return OdometryData(
            translation=translation,
            rotation=rotation,
            confidence=confidence,
            timestamp=timestamp,
            frame_count=self._frame_count,
        )

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale."""
        if len(frame.shape) == 3:
            return np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return frame

    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect good features to track."""
        import cv2

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self._max_features,
            qualityLevel=self._quality_level,
            minDistance=self._min_distance,
            blockSize=7,
        )
        return corners

    def _track_features(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_points: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Track features using optical flow."""
        import cv2

        if prev_points is None:
            return None, None, None

        current_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_points,
            None,
            winSize=(21, 21),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            minEigThreshold=0.001,
        )

        return current_points, status, error

    def _estimate_motion(
        self,
        prev_pts: np.ndarray,
        curr_pts: np.ndarray,
        errors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate translation and rotation from feature correspondences."""
        import cv2

        try:
            fx = self._focal_length
            fy = self._focal_length
            cx = prev_pts[:, 0].mean()
            cy = prev_pts[:, 1].mean()

            k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            E, mask = cv2.findEssentialMat(
                prev_pts,
                curr_pts,
                k,
                cv2.RANSAC,
                0.999,
                1.0,
            )

            if E is None:
                return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

            _, R, t, mask = cv2.recoverPose(
                E,
                prev_pts,
                curr_pts,
                k,
            )

            translation = t.flatten()
            rotation = cv2.Rodrigues(R)[0].flatten()

            return translation, rotation

        except Exception as e:
            self._logger.debug(f"Motion estimation error: {e}")
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

    def _compute_confidence(self, errors: np.ndarray) -> float:
        """Compute confidence based on tracking errors."""
        if len(errors) == 0:
            return 0.0
        mean_error = np.mean(errors)
        if mean_error < 1.0:
            return 0.95
        elif mean_error < 5.0:
            return 0.8
        elif mean_error < 10.0:
            return 0.6
        else:
            return 0.3

    def get_total_displacement(self) -> np.ndarray:
        """Get total displacement since initialization."""
        return self._total_translation.copy()

    def get_total_rotation(self) -> np.ndarray:
        """Get total rotation since initialization."""
        return self._total_rotation.copy()

    def reset(self) -> None:
        """Reset odometry accumulators."""
        self._prev_frame = None
        self._prev_gray = None
        self._prev_points = None
        self._total_translation = np.array([0.0, 0.0, 0.0])
        self._total_rotation = np.array([0.0, 0.0, 0.0])
        self._frame_count = 0
        self._logger.info("VisualOdometry reset")

    @property
    def frame_count(self) -> int:
        """Get number of frames processed."""
        return self._frame_count

    @property
    def is_ready(self) -> bool:
        """Check if odometry is ready (has processed at least one frame)."""
        return self._prev_gray is not None
