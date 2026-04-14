import pytest
from unittest.mock import Mock, patch, MagicMock
import time
import numpy as np

from src.utils.performance_optimizer import (
    PipelineOptimizer,
    PipelineConfig,
    FPSCounter,
    ResolutionOptimizer,
    ParallelProcessor,
    PerformanceMonitor,
)


class TestPipelineConfig:
    def test_default_config(self):
        config = PipelineConfig()
        assert config.detector_interval == 1
        assert config.depth_interval == 8
        assert config.face_interval == 6

    def test_turbo_config(self):
        config = PipelineConfig(turbo_mode=True)
        assert config.turbo_mode is True
        assert config.depth_interval == 8


class TestFPSCounter:
    def test_initialization(self):
        counter = FPSCounter()
        assert counter.fps == 0.0
        assert counter.frame_count == 0

    def test_update_fps(self):
        counter = FPSCounter(window_size=3)
        counter.update()
        time.sleep(0.05)
        counter.update()
        counter.update()

        assert counter.get_fps() >= 0

    def test_window_size(self):
        counter = FPSCounter(window_size=5)
        for _ in range(10):
            counter.update()
            time.sleep(0.01)

        assert len(counter.times) <= 5


class TestPipelineOptimizer:
    def test_initialization_default(self):
        optimizer = PipelineOptimizer(target_fps=30.0)
        assert optimizer.target_fps == 30.0
        assert optimizer.turbo is False

    def test_initialization_turbo(self):
        optimizer = PipelineOptimizer(target_fps=30.0, turbo=True)
        assert optimizer.turbo is True

    def test_should_run_detector_default(self):
        optimizer = PipelineOptimizer()
        assert optimizer.should_run_detector() is True

    def test_should_run_depth(self):
        optimizer = PipelineOptimizer()
        optimizer.frame_count = 8

        assert optimizer.should_run_depth() is True

    def test_should_run_face(self):
        optimizer = PipelineOptimizer()
        optimizer.frame_count = 6

        assert optimizer.should_run_face() is True

    def test_should_run_gesture(self):
        optimizer = PipelineOptimizer()
        optimizer.frame_count = 6

        assert optimizer.should_run_gesture() is True

    def test_should_run_pose(self):
        optimizer = PipelineOptimizer()
        optimizer.frame_count = 6

        assert optimizer.should_run_pose() is True

    def test_tick(self):
        optimizer = PipelineOptimizer()
        fps = optimizer.tick()

        assert optimizer.frame_count == 1
        assert isinstance(fps, float)

    def test_get_current_fps(self):
        optimizer = PipelineOptimizer()
        optimizer.tick()
        fps = optimizer.get_current_fps()

        assert isinstance(fps, float)

    def test_adapt_intervals_downscale(self):
        optimizer = PipelineOptimizer(target_fps=30.0)
        optimizer._config.detector_interval = 1

        optimizer.adapt_intervals(15.0)

        assert optimizer._config.detector_interval >= 1

    def test_adapt_intervals_upscale(self):
        optimizer = PipelineOptimizer(target_fps=30.0)
        optimizer._config.detector_interval = 3

        optimizer.adapt_intervals(35.0)

        assert optimizer._config.detector_interval <= 3

    def test_get_all_models_to_run(self):
        optimizer = PipelineOptimizer(turbo=True)
        models = optimizer.get_all_models_to_run()

        assert "detector" in models
        assert isinstance(models, list)


class TestResolutionOptimizer:
    def test_initialization_default(self):
        optimizer = ResolutionOptimizer()
        assert optimizer.preset == "medium"

    def test_initialization_custom(self):
        optimizer = ResolutionOptimizer(preset="high")
        assert optimizer.preset == "high"

    def test_get_resolution(self):
        optimizer = ResolutionOptimizer(preset="medium")
        res = optimizer.get_resolution()

        assert res == (320, 320)

    def test_resolutions_dict(self):
        resolutions = ResolutionOptimizer.RESOLUTIONS
        assert "ultra" in resolutions
        assert "high" in resolutions
        assert "medium" in resolutions
        assert "low" in resolutions

    def test_optimize_for_fps_low(self):
        optimizer = ResolutionOptimizer()
        result = optimizer.optimize_for_fps(30.0, 10.0)

        assert result == "low"

    def test_optimize_for_fps_medium(self):
        optimizer = ResolutionOptimizer()
        result = optimizer.optimize_for_fps(30.0, 20.0)

        assert result == "medium"

    def test_optimize_for_fps_high(self):
        optimizer = ResolutionOptimizer()
        result = optimizer.optimize_for_fps(30.0, 25.0)

        assert result == "high"

    def test_optimize_for_fps_ultra(self):
        optimizer = ResolutionOptimizer()
        result = optimizer.optimize_for_fps(30.0, 35.0)

        assert result == "ultra"

    def test_set_resolution(self):
        optimizer = ResolutionOptimizer()
        optimizer.set_resolution(224, 224)

        assert optimizer.width == 224
        assert optimizer.height == 224


class TestParallelProcessor:
    def test_initialization(self):
        processor = ParallelProcessor(max_workers=2)
        assert processor.max_workers == 2

    def test_run_parallel(self):
        processor = ParallelProcessor(max_workers=2)

        def dummy_fn(frame):
            return frame.shape

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = processor.run_parallel(frame, [dummy_fn, dummy_fn])

        assert len(results) == 2

    def test_run_parallel_with_failure(self):
        processor = ParallelProcessor(max_workers=2)

        def failing_fn(frame):
            raise ValueError("Error")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = processor.run_parallel(frame, [failing_fn])

        assert results[0] is None


class TestPerformanceMonitor:
    def test_initialization(self):
        monitor = PerformanceMonitor(window_size=10)
        assert monitor.window_size == 10

    def test_record_inference(self):
        monitor = PerformanceMonitor()
        monitor.record_inference(0.033)

        assert monitor.get_total_frames() == 1

    def test_get_average_time(self):
        monitor = PerformanceMonitor()
        monitor.record_inference(0.033)
        monitor.record_inference(0.034)

        avg = monitor.get_average_time()
        assert avg > 0

    def test_window_size(self):
        monitor = PerformanceMonitor(window_size=5)
        for _ in range(10):
            monitor.record_inference(0.033)

        assert monitor.get_total_frames() <= 5

    def test_get_fps(self):
        monitor = PerformanceMonitor()
        monitor.record_inference(0.033)

        fps = monitor.get_fps()
        assert fps > 0

    def test_get_uptime(self):
        monitor = PerformanceMonitor()
        time.sleep(0.01)
        uptime = monitor.get_uptime()

        assert uptime > 0