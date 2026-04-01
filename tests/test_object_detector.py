import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

from src.models.object_detector import ObjectDetector
from src.exceptions import ModelError
from src.camera.types import BoundingBox, Detection


class TestObjectDetector:
    def test_initialization(self):
        detector = ObjectDetector(
            model_path="yolov8n.pt",
            confidence=0.5,
            iou_threshold=0.45
        )
        assert detector._model_path == "yolov8n.pt"
        assert detector._confidence == 0.5
        assert detector._iou_threshold == 0.45

    @patch("ultralytics.YOLO")
    def test_load_model(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ObjectDetector()
        detector.load()

        assert detector.is_loaded is True
        mock_yolo.assert_called_once()

    @patch("ultralytics.YOLO")
    def test_load_model_with_custom_path(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ObjectDetector(model_path="custom_model.pt")
        detector.load()

        mock_yolo.assert_called_once_with("custom_model.pt")

    def test_detect_without_loading_raises_error(self):
        detector = ObjectDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(ModelError):
            detector.detect(frame)

    @patch("ultralytics.YOLO")
    def test_detect_returns_list(self, mock_yolo):
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_result.names = {0: "person"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = ObjectDetector()
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert isinstance(result, list)

    @patch("ultralytics.YOLO")
    def test_detect_with_boxes(self, mock_yolo):
        mock_box = MagicMock()
        mock_box.xyxy = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
        mock_box.conf = torch.tensor([0.95])
        mock_box.cls = torch.tensor([0])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "person"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = ObjectDetector(confidence=0.5)
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 1
        assert isinstance(detections[0], Detection)
        assert detections[0].class_name == "person"
        assert detections[0].confidence == pytest.approx(0.95, rel=0.01)

    @patch("ultralytics.YOLO")
    def test_detect_filters_by_confidence(self, mock_yolo):
        mock_box = MagicMock()
        mock_box.xyxy = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
        mock_box.conf = torch.tensor([0.3])
        mock_box.cls = torch.tensor([0])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "person"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = ObjectDetector(confidence=0.5)
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["conf"] == 0.5
        assert call_kwargs["iou"] == 0.45
        assert call_kwargs["verbose"] is False

    def test_name_property(self):
        detector = ObjectDetector()
        assert detector.name == "YOLOv10"

    @patch("ultralytics.YOLO")
    def test_is_loaded_before_load(self, mock_yolo):
        detector = ObjectDetector()
        assert detector.is_loaded is False

    @patch("ultralytics.YOLO")
    def test_is_loaded_after_load(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ObjectDetector()
        detector.load()

        assert detector.is_loaded is True
