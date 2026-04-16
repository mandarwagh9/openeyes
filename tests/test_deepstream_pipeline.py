"""Unit tests for DeepStream pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.deepstream.pipeline import DeepStreamPipeline, DetectionResult


class TestDetectionResult:
    """Tests for DetectionResult class."""
    
    def test_creation(self):
        """Test DetectionResult creation."""
        det = DetectionResult(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox_left=100.0,
            bbox_top=100.0,
            bbox_width=200.0,
            bbox_height=300.0
        )
        
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.confidence == 0.85
        assert det.bbox_left == 100.0
        assert det.bbox_top == 100.0
        assert det.bbox_width == 200.0
        assert det.bbox_height == 300.0
    
    def test_repr(self):
        """Test DetectionResult string representation."""
        det = DetectionResult(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox_left=100.0,
            bbox_top=100.0,
            bbox_width=200.0,
            bbox_height=300.0
        )
        
        repr_str = repr(det)
        assert "person" in repr_str
        assert "0.85" in repr_str


class TestDeepStreamPipeline:
    """Tests for DeepStreamPipeline class."""
    
    @patch("src.deepstream.pipeline.Gst")
    def test_initialization(self, mock_gst):
        """Test pipeline initialization."""
        pipeline = DeepStreamPipeline(
            model="yolov10n",
            camera=0,
            width=640,
            height=480,
            fps=30,
            display=True
        )
        
        assert pipeline.model == "yolov10n"
        assert pipeline.camera == 0
        assert pipeline.width == 640
        assert pipeline.height == 480
        assert pipeline.fps == 30
        assert pipeline.display is True
    
    @patch("src.deepstream.pipeline.Gst")
    def test_initialization_defaults(self, mock_gst):
        """Test pipeline with default values."""
        pipeline = DeepStreamPipeline()
        
        assert pipeline.model == "yolo11n"
        assert pipeline.camera == 0
        assert pipeline.width == 640
        assert pipeline.height == 480
        assert pipeline.fps == 30
    
    @patch("src.deepstream.pipeline.Gst")
    def test_get_config_path(self, mock_gst):
        """Test config path resolution."""
        pipeline = DeepStreamPipeline(model="yolov10n")
        config_path = pipeline._get_config_path()
        
        assert "config_yolov10n.txt" in config_path
    
    @patch("src.deepstream.pipeline.Gst")
    @patch("src.deepstream.pipeline.os.path.exists")
    def test_get_config_path_fallback(self, mock_exists, mock_gst):
        """Test config path fallback to default model."""
        mock_exists.return_value = False
        
        pipeline = DeepStreamPipeline(model="yolo11n")
        config_path = pipeline._get_config_path()
        
        assert "config_" in config_path
    
    @patch("src.deepstream.pipeline.Gst")
    def test_coco_classes(self, mock_gst):
        """Test COCO classes list."""
        pipeline = DeepStreamPipeline()
        
        assert len(pipeline.COCO_CLASSES) == 80
        assert pipeline.COCO_CLASSES[0] == "person"
        assert pipeline.COCO_CLASSES[15] == "cat"
        assert pipeline.COCO_CLASSES[67] == "cell phone"
    
    def test_detection_callback_registration(self):
        """Test callback registration."""
        pipeline = DeepStreamPipeline()
        
        callback = Mock()
        pipeline.set_detection_callback(callback)
        
        assert callback in pipeline._callbacks
    
    @patch("src.deepstream.pipeline.Gst")
    def test_callback_multiple(self, mock_gst):
        """Test multiple callbacks."""
        pipeline = DeepStreamPipeline()
        
        cb1 = Mock()
        cb2 = Mock()
        
        pipeline.set_detection_callback(cb1)
        pipeline.set_detection_callback(cb2)
        
        assert len(pipeline._callbacks) == 2
    
    @patch("src.deepstream.pipeline.Gst")
    def test_get_fps(self, mock_gst):
        """Test get_fps method."""
        pipeline = DeepStreamPipeline()
        
        fps = pipeline.get_fps()
        
        assert isinstance(fps, float)
        assert fps >= 0.0
    
    @patch("src.deepstream.pipeline.Gst")
    def test_get_last_detections(self, mock_gst):
        """Test get_last_detections method."""
        pipeline = DeepStreamPipeline()
        
        detections = pipeline.get_last_detections()
        
        assert isinstance(detections, list)


class TestDeepStreamPipelineCallbacks:
    """Tests for pipeline callback system."""
    
    @patch("src.deepstream.pipeline.Gst")
    def test_callback_invocation(self, mock_gst):
        """Test callback is invoked."""
        pipeline = DeepStreamPipeline()
        
        callback = Mock()
        pipeline.set_detection_callback(callback)
        
        test_detections = [
            DetectionResult(0, "person", 0.85, 100, 100, 200, 300)
        ]
        
        for cb in pipeline._callbacks:
            cb(test_detections, 30.0)
        
        callback.assert_called_once()
    
    @patch("src.deepstream.pipeline.Gst")
    def test_multiple_callbacks_invoked(self, mock_gst):
        """Test all callbacks are invoked."""
        pipeline = DeepStreamPipeline()
        
        cb1 = Mock()
        cb2 = Mock()
        
        pipeline.set_detection_callback(cb1)
        pipeline.set_detection_callback(cb2)
        
        test_detections = []
        
        for cb in pipeline._callbacks:
            cb(test_detections, 30.0)
        
        cb1.assert_called_once()
        cb2.assert_called_once()


class TestPipelineConfiguration:
    """Tests for pipeline configuration."""
    
    @patch("src.deepstream.pipeline.Gst")
    @patch("os.path.exists")
    def test_model_selection(self, mock_exists, mock_gst):
        """Test model selection logic."""
        mock_exists.return_value = True
        
        pipeline = DeepStreamPipeline(model="yolov10n")
        
        assert pipeline.model == "yolov10n"
    
    @patch("src.deepstream.pipeline.Gst")
    def test_display_flag(self, mock_gst):
        """Test display flag."""
        pipeline_no_display = DeepStreamPipeline(display=False)
        pipeline_display = DeepStreamPipeline(display=True)
        
        assert pipeline_no_display.display is False
        assert pipeline_display.display is True


class TestFPSCalculation:
    """Tests for FPS calculation."""
    
    def test_calculate_fps(self):
        """Test FPS calculation runs."""
        pipeline = DeepStreamPipeline()
        
        initial_count = pipeline._frame_count
        
        pipeline._calculate_fps()
        
        assert pipeline._frame_count == initial_count + 1
    
    def test_fps_timer_returns_true(self):
        """Test FPS timer returns True to continue."""
        pipeline = DeepStreamPipeline()
        pipeline._running = True
        
        result = pipeline._print_fps_timer()
        
        assert result is True
    
    def test_fps_timer_handles_stopped_pipeline(self):
        """Test FPS timer handles stopped pipeline."""
        pipeline = DeepStreamPipeline()
        pipeline._running = False
        
        result = pipeline._print_fps_timer()
        
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])