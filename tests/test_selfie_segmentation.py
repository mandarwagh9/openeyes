import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.models.selfie_segmentation import SelfieSegmentation
from src.exceptions import ModelError


class TestSelfieSegmentation:
    def test_initialization_default(self):
        segmenter = SelfieSegmentation()
        assert segmenter._model_selection == 0

    def test_initialization_custom(self):
        segmenter = SelfieSegmentation(model_selection=1)
        assert segmenter._model_selection == 1

    @patch("mediapipe.solutions.selfie_segmentation.SelfieSegmentation")
    def test_load(self, mock_segmenter):
        mock_instance = MagicMock()
        mock_segmenter.return_value = mock_instance

        segmenter = SelfieSegmentation()
        segmenter.load()

        assert segmenter.is_loaded is True
        mock_segmenter.assert_called_once()

    @patch("mediapipe.solutions.selfie_segmentation.SelfieSegmentation")
    def test_load_failure(self, mock_segmenter):
        mock_segmenter.side_effect = Exception("Load error")

        segmenter = SelfieSegmentation()
        with pytest.raises(ModelError):
            segmenter.load()

    @patch("mediapipe.solutions.selfie_segmentation.SelfieSegmentation")
    def test_segment_without_loading(self, mock_segmenter):
        segmenter = SelfieSegmentation()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(ModelError):
            segmenter.segment(frame)

    @patch("mediapipe.solutions.selfie_segmentation.SelfieSegmentation")
    def test_segment_returns_mask(self, mock_segmenter):
        mock_instance = MagicMock()
        mock_segmenter.return_value = mock_instance

        mock_result = MagicMock()
        mock_result.segmentation_mask = np.ones((480, 640), dtype=np.float32)
        mock_instance.process.return_value = mock_result

        segmenter = SelfieSegmentation()
        segmenter.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = segmenter.segment(frame)

        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8

    @patch("mediapipe.solutions.selfie_segmentation.SelfieSegmentation")
    def test_segment_with_no_mask(self, mock_segmenter):
        mock_instance = MagicMock()
        mock_segmenter.return_value = mock_instance

        mock_result = MagicMock()
        mock_result.segmentation_mask = None
        mock_instance.process.return_value = mock_result

        segmenter = SelfieSegmentation()
        segmenter.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = segmenter.segment(frame)

        assert mask.shape == (480, 640)
        assert mask.sum() == 0

    @patch("mediapipe.solutions.selfie_segmentation.SelfieSegmentation")
    def test_segment_with_threshold(self, mock_segmenter):
        mock_instance = MagicMock()
        mock_segmenter.return_value = mock_instance

        mock_result = MagicMock()
        mock_result.segmentation_mask = np.ones((480, 640), dtype=np.float32)
        mock_instance.process.return_value = mock_result

        segmenter = SelfieSegmentation()
        segmenter.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = segmenter.segment_with_threshold(frame, threshold=0.5)

        assert mask.shape == (480, 640)

    @patch("mediapipe.solutions.selfie_segmentation.SelfieSegmentation")
    def test_get_foreground_mask(self, mock_segmenter):
        mock_instance = MagicMock()
        mock_segmenter.return_value = mock_instance

        mock_result = MagicMock()
        mock_result.segmentation_mask = np.ones((480, 640), dtype=np.float32)
        mock_instance.process.return_value = mock_result

        segmenter = SelfieSegmentation()
        segmenter.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = segmenter.get_foreground_mask(frame)

        assert mask.shape == (480, 640)

    @patch("mediapipe.solutions.selfie_segmentation.SelfieSegmentation")
    def test_get_background_mask(self, mock_segmenter):
        mock_instance = MagicMock()
        mock_segmenter.return_value = mock_instance

        mock_result = MagicMock()
        mock_result.segmentation_mask = np.zeros((480, 640), dtype=np.float32)
        mock_instance.process.return_value = mock_result

        segmenter = SelfieSegmentation()
        segmenter.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = segmenter.get_background_mask(frame)

        assert mask.shape == (480, 640)

    @patch("mediapipe.solutions.selfie_segmentation.SelfieSegmentation")
    def test_apply_mask(self, mock_segmenter):
        mock_instance = MagicMock()
        mock_segmenter.return_value = mock_instance

        mock_result = MagicMock()
        mock_result.segmentation_mask = np.ones((480, 640), dtype=np.float32)
        mock_instance.process.return_value = mock_result

        segmenter = SelfieSegmentation()
        segmenter.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        background = np.zeros((480, 640, 3), dtype=np.uint8)

        result = segmenter.apply_mask(frame, background)

        assert result.shape == frame.shape

    def test_name_property(self):
        segmenter = SelfieSegmentation()
        assert segmenter.name == "MediaPipeSelfieSegmentation"

    def test_is_loaded_before_load(self):
        segmenter = SelfieSegmentation()
        assert segmenter.is_loaded is False

    @patch("mediapipe.solutions.selfie_segmentation.SelfieSegmentation")
    def test_is_loaded_after_load(self, mock_segmenter):
        mock_instance = MagicMock()
        mock_segmenter.return_value = mock_instance

        segmenter = SelfieSegmentation()
        segmenter.load()

        assert segmenter.is_loaded is True