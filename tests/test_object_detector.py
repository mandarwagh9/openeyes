import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

from src.models.object_detector import ObjectDetector, TensorRTDetector
from src.exceptions import ModelError
from src.camera.types import BoundingBox, Detection


class TestObjectDetector:
    def test_initialization(self):
        detector = ObjectDetector(
            model_path="yolo11n.pt",
            confidence=0.5,
            iou_threshold=0.45
        )
        assert detector._model_path == "yolo11n.pt"
        assert detector._confidence == 0.5
        assert detector._iou_threshold == 0.45

    def test_initialization_default(self):
        detector = ObjectDetector()
        assert detector._model_path == "yolo11n.pt"
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
        assert detector.name == "YOLO11n"

    @patch("ultralytics.YOLO")
    def test_name_property_with_tensorrt(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_model.names = {0: "person"}

        detector = ObjectDetector(model_path="yolo11n.engine")
        detector.load()
        detector._using_tensorrt = True
        assert "YOLO11n" in detector.name
        assert "TensorRT" in detector.name

    @patch("ultralytics.YOLO")
    def test_name_property_with_onnx(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_model.names = {0: "person"}

        detector = ObjectDetector(model_path="yolo11n.onnx")
        detector.load()
        detector._using_onnx = True
        assert "YOLO11n" in detector.name
        assert "ONNX" in detector.name

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

    @patch("ultralytics.YOLO")
    def test_multiple_detections(self, mock_yolo):
        mock_box1 = MagicMock()
        mock_box1.xyxy = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
        mock_box1.conf = torch.tensor([0.9])
        mock_box1.cls = torch.tensor([0])

        mock_box2 = MagicMock()
        mock_box2.xyxy = torch.tensor([[200.0, 300.0, 400.0, 500.0]])
        mock_box2.conf = torch.tensor([0.85])
        mock_box2.cls = torch.tensor([1])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box1, mock_box2]
        mock_result.names = {0: "person", 1: "car"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = ObjectDetector(confidence=0.5)
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 2
        assert detections[0].class_name == "person"
        assert detections[1].class_name == "car"

    @patch("ultralytics.YOLO")
    def test_tensorrt_engine_loading(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ObjectDetector(model_path="yolo11n.engine")
        detector.load()

        assert detector.is_loaded is True

    @patch("ultralytics.YOLO")
    def test_onnx_loading(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ObjectDetector(model_path="yolo11n.onnx")
        detector.load()

        assert detector.is_loaded is True


class TestTensorRTDetector:
    @patch("tensorrt.Logger")
    @patch("tensorrt.Builder")
    @patch("tensorrt.OnnxParser")
    def test_check_tensorrt_available(self, mock_parser, mock_builder, mock_logger):
        with patch.dict("sys.modules", {"tensorrt": MagicMock()}):
            detector = TensorRTDetector("model.onnx")
            result = detector._check_tensorrt_available()
            assert isinstance(result, bool)

    def test_check_cuda_available(self):
        detector = TensorRTDetector("model.onnx")
        result = detector._check_cuda_available()
        assert isinstance(result, bool)

    def test_preprocess(self):
        detector = TensorRTDetector("model.onnx")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector.preprocess(frame, (640, 640))

        assert result.shape == (1, 3, 640, 640)
        assert result.dtype == np.float32

    def test_preprocess_different_sizes(self):
        detector = TensorRTDetector("model.onnx")

        for size in [(320, 320), (416, 416), (640, 640)]:
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = detector.preprocess(frame, size)
            assert result.shape == (1, 3, size[0], size[1])


class TestCOCOClasses:
    def test_all_coco_classes_present(self):
        detector = ObjectDetector()
        coco_classes = detector._get_class_name(0)
        assert coco_classes == "person"

        person_idx = detector._get_class_name(0)
        assert person_idx == "person"

        bicycle_idx = detector._get_class_name(1)
        assert bicycle_idx == "bicycle"

    def test_unknown_class(self):
        detector = ObjectDetector()
        unknown = detector._get_class_name(1000)
        assert "class_1000" in unknown