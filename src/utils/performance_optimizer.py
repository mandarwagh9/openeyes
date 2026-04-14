import time
from typing import Optional, Callable, Any, List
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PipelineConfig:
    detector_interval: int = 1
    depth_interval: int = 8
    face_interval: int = 6
    gesture_interval: int = 6
    pose_interval: int = 6
    segmentation_interval: int = 12
    turbo_mode: bool = False


@dataclass
class FPSCounter:
    fps: float = 0.0
    frame_count: int = 0
    last_time: float = field(default_factory=time.time)
    window_size: int = 30
    times: List[float] = field(default_factory=list)

    def update(self) -> float:
        current_time = time.time()
        self.frame_count += 1
        self.times.append(current_time)

        if len(self.times) > self.window_size:
            self.times.pop(0)

        if len(self.times) >= 2:
            elapsed = self.times[-1] - self.times[0]
            if elapsed > 0:
                self.fps = (len(self.times) - 1) / elapsed

        return self.fps

    def get_fps(self) -> float:
        return self.fps


class PipelineOptimizer:
    def __init__(
        self,
        target_fps: float = 30.0,
        turbo: bool = False,
    ):
        self.target_fps = target_fps
        self.turbo = turbo
        self.fps_counter = FPSCounter()
        self.frame_count = 0

        if turbo:
            self._config = PipelineConfig(
                detector_interval=1,
                depth_interval=16,
                face_interval=12,
                gesture_interval=12,
                pose_interval=12,
                segmentation_interval=16,
                turbo_mode=True,
            )
        else:
            self._config = PipelineConfig(
                detector_interval=1,
                depth_interval=8,
                face_interval=6,
                gesture_interval=6,
                pose_interval=6,
                segmentation_interval=12,
                turbo_mode=False,
            )

    def get_config(self) -> PipelineConfig:
        return self._config

    def should_run_detector(self) -> bool:
        return (self.frame_count % self._config.detector_interval) == 0

    def should_run_depth(self) -> bool:
        return (self.frame_count % self._config.depth_interval) == 0

    def should_run_face(self) -> bool:
        return (self.frame_count % self._config.face_interval) == 0

    def should_run_gesture(self) -> bool:
        return (self.frame_count % self._config.gesture_interval) == 0

    def should_run_pose(self) -> bool:
        return (self.frame_count % self._config.pose_interval) == 0

    def should_run_segmentation(self) -> bool:
        return (self.frame_count % self._config.segmentation_interval) == 0

    def tick(self) -> float:
        self.frame_count += 1
        return self.fps_counter.update()

    def get_current_fps(self) -> float:
        return self.fps_counter.get_fps()

    def adapt_intervals(self, current_fps: float) -> None:
        if current_fps < self.target_fps * 0.7:
            self._config.detector_interval = min(self._config.detector_interval + 1, 4)
            self._config.depth_interval = min(self._config.depth_interval + 2, 24)
            self._config.face_interval = min(self._config.face_interval + 2, 18)
            self._config.gesture_interval = min(self._config.gesture_interval + 2, 18)
            self._config.pose_interval = min(self._config.pose_interval + 2, 18)
        elif current_fps > self.target_fps * 0.9:
            self._config.detector_interval = max(self._config.detector_interval - 1, 1)
            self._config.depth_interval = max(self._config.depth_interval - 2, 4)
            self._config.face_interval = max(self._config.face_interval - 2, 4)
            self._config.gesture_interval = max(self._config.gesture_interval - 2, 4)
            self._config.pose_interval = max(self._config.pose_interval - 2, 4)

    def get_all_models_to_run(self) -> List[str]:
        models = []
        if self.should_run_detector():
            models.append("detector")
        if self.should_run_depth():
            models.append("depth")
        if self.should_run_face():
            models.append("face")
        if self.should_run_gesture():
            models.append("gesture")
        if self.should_run_pose():
            models.append("pose")
        if self.should_run_segmentation():
            models.append("segmentation")
        return models


class ResolutionOptimizer:
    RESOLUTIONS = {
        "ultra": (640, 480),
        "high": (416, 416),
        "medium": (320, 320),
        "low": (256, 256),
    }

    def __init__(self, preset: str = "medium"):
        self.preset = preset
        self._resolution = ResolutionOptimizer.RESOLUTIONS.get(preset, (320, 320))

    def get_resolution(self) -> tuple:
        return self._resolution

    def optimize_for_fps(self, target_fps: float, current_fps: float) -> str:
        if current_fps < target_fps * 0.5:
            return "low"
        elif current_fps < target_fps * 0.75:
            return "medium"
        elif current_fps < target_fps:
            return "high"
        else:
            return "ultra"

    def set_resolution(self, width: int, height: int) -> None:
        self._resolution = (width, height)

    @property
    def width(self) -> int:
        return self._resolution[0]

    @property
    def height(self) -> int:
        return self._resolution[1]


class ParallelProcessor:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._results: dict = {}

    def run_parallel(
        self,
        frame: np.ndarray,
        model_fns: List[Callable[[np.ndarray], Any]],
    ) -> List[Any]:
        import concurrent.futures

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(fn, frame) for fn in model_fns]
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    results.append(None)

        return results


class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._times: List[float] = []
        self._start_time: float = time.time()

    def record_inference(self, duration: float) -> None:
        self._times.append(duration)
        if len(self._times) > self.window_size:
            self._times.pop(0)

    def get_average_time(self) -> float:
        if not self._times:
            return 0.0
        return sum(self._times) / len(self._times)

    def get_total_frames(self) -> int:
        return len(self._times)

    def get_uptime(self) -> float:
        return time.time() - self._start_time

    def get_fps(self) -> float:
        avg_time = self.get_average_time()
        if avg_time > 0:
            return 1.0 / avg_time
        return 0.0