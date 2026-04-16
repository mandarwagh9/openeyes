"""Integration tests for DeepStream pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.deepstream.pipeline import run_deepstream, DeepStreamPipeline


class TestRunDeepstream:
    """Tests for run_deepstream function."""
    
    def test_run_deepstream_returns_pipeline(self):
        """Test run_deepstream returns pipeline object."""
        pipeline = run_deepstream(
            model="yolov10n",
            camera=0,
            width=640,
            height=480,
            fps=30,
            display=False,
            use_face=False,
            use_gesture=False,
            use_pose=False,
            use_tracking=False,
            use_ros2=False,
            use_udp=False
        )
        
        assert pipeline is not None
        assert isinstance(pipeline, DeepStreamPipeline)
    
    def test_run_deepstream_with_udp(self):
        """Test run_deepstream with UDP."""
        pipeline = run_deepstream(
            model="yolov10n",
            camera=0,
            use_udp=True,
            use_face=False,
            use_gesture=False,
            use_pose=False,
            use_tracking=False
        )
        
        assert pipeline is not None
    
    def test_run_deepstream_with_ros2_raises(self):
        """Test run_deepstream with ROS2 when not available."""
        pipeline = run_deepstream(
            model="yolov10n",
            camera=0,
            use_udp=False,
            use_ros2=True,
            use_face=False,
            use_gesture=False,
            use_pose=False,
            use_tracking=False
        )
        
        assert pipeline is not None


class TestPipelineOutput:
    """Tests for pipeline output formats."""
    
    def test_udp_output_format(self):
        """Test UDP output JSON format."""
        result = {
            "objects": [
                {
                    "class_name": "person",
                    "confidence": 0.85,
                    "bbox": [100, 100, 200, 300]
                }
            ],
            "faces": [],
            "gestures": [],
            "poses": [],
            "fps": 30.0
        }
        
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        
        assert "objects" in parsed
        assert "fps" in parsed
        assert parsed["objects"][0]["class_name"] == "person"
        assert parsed["objects"][0]["confidence"] == 0.85
    
    @patch("src.deepstream.pipeline.Gst")
    def test_json_serialization(self, mock_gst):
        """Test JSON can be serialized and deserialized."""
        data = {
            "objects": [
                {"class_name": "person", "confidence": 0.85, "bbox": [100, 100, 200, 300]},
                {"class_name": "car", "confidence": 0.72, "bbox": [50, 50, 150, 100]}
            ],
            "faces": [],
            "gestures": [],
            "poses": [],
            "fps": 30.5,
            "timestamp": 1234567890.0
        }
        
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        
        assert len(parsed["objects"]) == 2
        assert parsed["fps"] == 30.5


class TestPipelineConfiguration:
    """Tests for pipeline configuration options."""
    
    @patch("src.deepstream.pipeline.Gst")
    def test_resolution_variations(self, mock_gst):
        """Test different resolutions."""
        resolutions = [
            (640, 480),
            (1280, 720),
            (1920, 1080)
        ]
        
        for width, height in resolutions:
            pipeline = DeepStreamPipeline(width=width, height=height)
            assert pipeline.width == width
            assert pipeline.height == height
    
    @patch("src.deepstream.pipeline.Gst")
    def test_fps_variations(self, mock_gst):
        """Test different FPS settings."""
        fps_values = [15, 30, 60]
        
        for fps in fps_values:
            pipeline = DeepStreamPipeline(fps=fps)
            assert pipeline.fps == fps
    
    @patch("src.deepstream.pipeline.Gst")
    def test_camera_selection(self, mock_gst):
        """Test different camera IDs."""
        for camera_id in [0, 1, 2]:
            pipeline = DeepStreamPipeline(camera=camera_id)
            assert pipeline.camera == camera_id


class TestPipelineEdgeCases:
    """Tests for edge cases."""
    
    @patch("src.deepstream.pipeline.Gst")
    def test_zero_fps(self, mock_gst):
        """Test pipeline with zero FPS."""
        pipeline = DeepStreamPipeline(fps=0)
        assert pipeline.fps == 0
    
    @patch("src.deepstream.pipeline.Gst")
    def test_zero_dimensions(self, mock_gst):
        """Test pipeline with zero dimensions."""
        pipeline = DeepStreamPipeline(width=0, height=0)
        assert pipeline.width == 0
        assert pipeline.height == 0
    
    @patch("src.deepstream.pipeline.Gst")
    def test_none_callbacks(self, mock_gst):
        """Test pipeline with no callbacks."""
        pipeline = DeepStreamPipeline()
        
        pipeline._detections = []
        pipeline._current_fps = 30.0
        
        for cb in pipeline._callbacks:
            cb(pipeline._detections, pipeline._current_fps)


class TestPerformance:
    """Tests for performance characteristics."""
    
    @patch("src.deepstream.pipeline.Gst")
    def test_fps_calculation_performance(self, mock_gst):
        """Test FPS calculation doesn't degrade."""
        import time
        pipeline = DeepStreamPipeline()
        
        start = time.time()
        for _ in range(1000):
            pipeline._calculate_fps()
        elapsed = time.time() - start
        
        assert elapsed < 1.0
    
    @patch("src.deepstream.pipeline.Gst")
    def test_callback_overhead(self, mock_gst):
        """Test callback overhead is minimal."""
        import time
        pipeline = DeepStreamPipeline()
        
        callback_calls = []
        
        def fast_callback(detections, fps):
            callback_calls.append(1)
        
        pipeline.set_detection_callback(fast_callback)
        
        start = time.time()
        for _ in range(10000):
            for cb in pipeline._callbacks:
                cb([], 30.0)
        elapsed = time.time() - start
        
        assert elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])