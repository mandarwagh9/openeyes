import cv2
import numpy as np
from typing import Optional, Any, Dict


class FrameSkipProcessor:
    """Universal frame skipper for any inference model."""

    def __init__(self, skip_interval: int = 2, warmup_frames: int = 5):
        self.skip_interval = skip_interval
        self.warmup_frames = warmup_frames
        self.frame_count = 0
        self.last_result: Optional[Any] = None

    def should_process(self) -> bool:
        if self.frame_count < self.warmup_frames:
            return True
        return (self.frame_count % self.skip_interval) == 0

    def process(self, frame: np.ndarray, model_fn, interpolate: bool = True) -> Optional[Any]:
        self.frame_count += 1

        if self.should_process():
            self.last_result = model_fn(frame)
            return self.last_result
        else:
            if interpolate and self.last_result is not None:
                return self.last_result
            return None

    def reset(self) -> None:
        self.frame_count = 0
        self.last_result = None


class AdaptiveFrameSkipper:
    """Adaptive frame skipper that adjusts based on motion detection."""

    def __init__(
        self,
        base_skip: int = 2,
        motion_threshold: float = 5000.0,
        min_skip: int = 1,
        max_skip: int = 4
    ):
        self.base_skip = base_skip
        self.motion_threshold = motion_threshold
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.current_skip = base_skip
        self.frame_count = 0
        self.previous_frame: Optional[np.ndarray] = None

    def compute_motion(self, frame: np.ndarray) -> float:
        h, w = frame.shape[:2]
        
        if self.previous_frame is None:
            self.previous_frame = np.empty((h, w), dtype=np.uint8)
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self.previous_frame)
            return 0.0

        gray = np.empty((h, w), dtype=np.uint8)
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=gray)
        
        diff = cv2.absdiff(self.previous_frame, gray)
        motion_score = float(cv2.countNonZero(diff))

        self.previous_frame = gray
        return motion_score

    def should_process(self, frame: np.ndarray) -> bool:
        self.frame_count += 1

        motion = self.compute_motion(frame)

        if motion < self.motion_threshold * 0.3:
            self.current_skip = min(self.base_skip * 2, self.max_skip)
        elif motion > self.motion_threshold:
            self.current_skip = self.min_skip
        else:
            self.current_skip = self.base_skip

        return (self.frame_count % self.current_skip) == 0


class MultiModelFrameScheduler:
    """Scheduler for managing frame skipping across multiple models."""

    def __init__(self, skip_intervals: Optional[Dict[str, int]] = None, turbo: bool = False):
        if turbo:
            self.skip_intervals = skip_intervals or {
                'detector': 1,
                'depth': 16,
                'face': 12,
                'gesture': 12,
                'pose': 12
            }
        else:
            self.skip_intervals = skip_intervals or {
                'detector': 1,
                'depth': 8,
                'face': 6,
                'gesture': 6,
                'pose': 6
            }
        self.frame_count = 0
        self.last_results: dict[str, Any] = {}

    def should_run(self, model_name: str) -> bool:
        skip = self.skip_intervals.get(model_name, 1)
        return (self.frame_count % skip) == 0

    def update(self, model_name: str, result: Any) -> None:
        self.last_results[model_name] = result

    def get_last(self, model_name: str) -> Optional[Any]:
        return self.last_results.get(model_name)

    def next_frame(self) -> None:
        self.frame_count += 1