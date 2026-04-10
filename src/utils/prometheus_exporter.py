"""Prometheus metrics exporter for OpenEyes."""

from typing import Dict, Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"

from src.utils.performance_monitor import PerformanceMonitor
from src.utils.health_monitor import SystemHealthMonitor


FPS = Gauge("openeyes_fps", "Current frames per second")
LATENCY = Histogram(
    "openeyes_latency_ms",
    "Frame processing latency in milliseconds",
    buckets=[5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
)
DETECTIONS = Counter("openeyes_detections_total", "Total objects detected")
FRAMES = Counter("openeyes_frames_total", "Total frames processed")
MEMORY = Gauge("openeyes_memory_mb", "Memory usage in MB")
ERRORS = Counter("openeyes_errors_total", "Total errors encountered")
MODEL_INFERENCE = Histogram(
    "openeyes_model_inference_ms",
    "Model inference time in milliseconds",
    buckets=[1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0],
)


class PrometheusExporter:
    def __init__(
        self,
        perf_monitor: Optional[PerformanceMonitor] = None,
        health_monitor: Optional[SystemHealthMonitor] = None,
    ):
        self._perf_monitor = perf_monitor
        self._health_monitor = health_monitor

    def update_from_performance(self) -> None:
        if not self._perf_monitor:
            return

        stats = self._perf_monitor.get_stats()

        FPS.set(stats.fps)
        if stats.avg_latency_ms > 0:
            LATENCY.observe(stats.avg_latency_ms)
        DETECTIONS.inc(stats.detection_count)
        FRAMES.inc(stats.frame_count)
        if stats.memory_used_mb > 0:
            MEMORY.set(stats.memory_used_mb)

        for model_name, model_time in stats.model_times.items():
            if model_time > 0:
                MODEL_INFERENCE.observe(model_time)

    def update_from_health(self) -> None:
        if not self._health_monitor:
            return

        metrics = self._health_monitor.get_current_metrics()
        if not metrics:
            return

        if metrics.error_count > 0:
            ERRORS.inc(metrics.error_count)

    def update(self) -> None:
        self.update_from_performance()
        self.update_from_health()

    def generate(self) -> bytes:
        if not PROMETHEUS_AVAILABLE:
            return b"# Prometheus not available"
        return generate_latest()

    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST


_exporter: Optional[PrometheusExporter] = None


def get_exporter(
    perf_monitor: Optional[PerformanceMonitor] = None,
    health_monitor: Optional[SystemHealthMonitor] = None,
) -> PrometheusExporter:
    global _exporter
    if _exporter is None:
        _exporter = PrometheusExporter(perf_monitor, health_monitor)
    return _exporter


def update_metrics() -> None:
    if _exporter:
        _exporter.update()


def metrics() -> bytes:
    if _exporter:
        return _exporter.generate()
    return b""