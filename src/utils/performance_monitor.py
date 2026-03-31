import time
import threading
from typing import Callable, Dict, List, Optional
from collections import deque
from dataclasses import dataclass, field

from src.utils.logger import get_logger


@dataclass
class PerformanceStats:
    fps: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    detection_count: int = 0
    frame_count: int = 0
    model_times: Dict[str, float] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(
        self,
        enabled: bool = True,
        stats_interval: int = 5,
        log_performance: bool = True,
    ):
        self._enabled = enabled
        self._stats_interval = stats_interval
        self._log_performance = log_performance
        self._logger = get_logger(__name__)

        self._frame_times: deque = deque(maxlen=100)
        self._latency_times: deque = deque(maxlen=100)
        self._model_times: Dict[str, deque] = {}
        self._detection_counts: deque = deque(maxlen=100)

        self._start_time = time.time()
        self._frame_count = 0
        self._detection_count = 0

        self._lock = threading.Lock()
        self._current_model_time: Dict[str, float] = {}

        self._stats_callbacks: List[Callable] = []

    def start_model(self, model_name: str) -> None:
        """Mark start of model inference."""
        if not self._enabled:
            return
        self._current_model_time[model_name] = time.perf_counter()

    def end_model(self, model_name: str) -> None:
        """Mark end of model inference."""
        if not self._enabled or model_name not in self._current_model_time:
            return

        elapsed = (time.perf_counter() - self._current_model_time[model_name]) * 1000
        del self._current_model_time[model_name]

        with self._lock:
            if model_name not in self._model_times:
                self._model_times[model_name] = deque(maxlen=100)
            self._model_times[model_name].append(elapsed)

    def record_frame(self, detection_count: int = 0) -> None:
        """Record a processed frame."""
        if not self._enabled:
            return

        now = time.perf_counter()

        with self._lock:
            if self._frame_times:
                latency = (now - self._frame_times[-1]) * 1000
                self._latency_times.append(latency)

            self._frame_times.append(now)
            self._frame_count += 1
            self._detection_count += detection_count
            self._detection_counts.append(detection_count)

    def get_stats(self) -> PerformanceStats:
        """Get current performance statistics."""
        with self._lock:
            now = time.time()
            elapsed = now - self._start_time

            fps = self._frame_count / elapsed if elapsed > 0 else 0.0

            avg_latency = 0.0
            min_latency = 0.0
            max_latency = 0.0
            if self._latency_times:
                avg_latency = sum(self._latency_times) / len(self._latency_times)
                min_latency = min(self._latency_times)
                max_latency = max(self._latency_times)

            memory_used = 0.0
            memory_total = 0.0
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                memory_used = mem_info.rss / (1024 * 1024)
                mem = psutil.virtual_memory()
                memory_total = mem.total / (1024 * 1024)
            except ImportError:
                pass

            model_times_avg = {}
            for model_name, times in self._model_times.items():
                if times:
                    model_times_avg[model_name] = sum(times) / len(times)

            avg_detections = 0
            if self._detection_counts:
                avg_detections = sum(self._detection_counts) / len(self._detection_counts)

            return PerformanceStats(
                fps=fps,
                avg_latency_ms=avg_latency,
                min_latency_ms=min_latency,
                max_latency_ms=max_latency,
                memory_used_mb=memory_used,
                memory_total_mb=memory_total,
                detection_count=int(avg_detections),
                frame_count=self._frame_count,
                model_times=model_times_avg,
            )

    def log_stats(self) -> None:
        """Log current performance statistics."""
        if not self._enabled or not self._log_performance:
            return

        stats = self.get_stats()

        self._logger.info(
            f"Performance | FPS: {stats.fps:.1f} | "
            f"Latency: {stats.avg_latency_ms:.1f}ms (avg) | "
            f"Memory: {stats.memory_used_mb:.0f}MB"
        )

        if stats.model_times:
            model_str = " | ".join(
                f"{k}: {v:.1f}ms" for k, v in stats.model_times.items()
            )
            self._logger.info(f"  Models: {model_str}")

    def add_callback(self, callback: Callable) -> None:
        """Add a callback to be called with stats."""
        self._stats_callbacks.append(callback)

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._frame_times.clear()
            self._latency_times.clear()
            self._model_times.clear()
            self._detection_counts.clear()
            self._start_time = time.time()
            self._frame_count = 0
            self._detection_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
