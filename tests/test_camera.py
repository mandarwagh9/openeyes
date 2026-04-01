import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

os.environ["OPENEYES_TEST_MODE"] = "true"

from src.camera.camera_handler import CameraHandler
from src.exceptions import CameraError


class TestCameraHandler:
    def test_initialization(self):
        camera = CameraHandler(source=0, width=640, height=480, fps=30)
        assert camera._source == 0
        assert camera._width == 640
        assert camera._height == 480
        assert camera._fps == 30

    @patch("src.camera.camera_handler.cv2.VideoCapture")
    def test_open_success(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        camera = CameraHandler()
        camera.open()

        assert camera.is_opened is True
        mock_video_capture.assert_called()

    @patch("src.camera.camera_handler.cv2.VideoCapture")
    def test_open_failure(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        camera = CameraHandler(source=999)
        with pytest.raises(CameraError):
            camera.open()

    @patch("src.camera.camera_handler.cv2.VideoCapture")
    def test_read_returns_frame(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        camera = CameraHandler()
        camera.open()
        frame = camera.read()

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)

    @patch("src.camera.camera_handler.CameraHandler._check_csi_available", return_value=False, create=True)
    @patch("src.camera.camera_handler.cv2.VideoCapture")
    def test_read_returns_none_on_failure(self, mock_video_capture, mock_csi):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        call_count = [0]
        def read_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return (True, np.zeros((480, 640, 3), dtype=np.uint8))
            return (False, None)
        mock_cap.read.side_effect = read_side_effect
        mock_video_capture.return_value = mock_cap

        camera = CameraHandler()
        camera._max_reconnect_attempts = 0
        camera.open()
        frame = camera.read()

        assert frame is None

    @patch("src.camera.camera_handler.cv2.VideoCapture")
    def test_read_when_not_opened(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        camera = CameraHandler()
        camera._reconnect_attempts = camera._max_reconnect_attempts
        frame = camera.read()
        assert frame is None

    @patch("src.camera.camera_handler.cv2.VideoCapture")
    def test_release(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        camera = CameraHandler()
        camera.open()
        camera.release()

        mock_cap.release.assert_called_once()
        assert camera.is_opened is False

    @patch("src.camera.camera_handler.cv2.VideoCapture")
    def test_frame_count(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        camera = CameraHandler()
        camera.open()
        camera.read()
        camera.read()

        assert camera.frame_count == 2

    @patch("src.camera.camera_handler.cv2.VideoCapture")
    def test_width_property(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640.0
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        camera = CameraHandler()
        camera.open()

        assert camera.width == 640

    @patch("src.camera.camera_handler.cv2.VideoCapture")
    def test_height_property(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 480.0
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        camera = CameraHandler()
        camera.open()

        assert camera.height == 480
