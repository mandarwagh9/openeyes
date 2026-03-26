import json
import socket
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.camera.types import (
    BoundingBox,
    Detection,
    DepthData,
    FaceDetection,
    Gesture,
    PoseData,
    PoseKeypoint,
    VisionResult,
)
from src.output.json_formatter import format_vision_result, format_objects
from src.output.udp_sender import UDPSender
from src.exceptions import OutputError


class TestJsonFormatter:
    def test_format_vision_result_empty(self):
        result = VisionResult(
            timestamp=123456.0,
            frame_id=1,
            objects=[],
            depth=DepthData(enabled=False),
            faces=[],
            gestures=[],
            pose=PoseData(detected=False),
        )

        output = format_vision_result(result)
        data = json.loads(output)

        assert data["timestamp"] == 123456.0
        assert data["frame_id"] == 1
        assert data["objects"] == []
        assert data["depth"]["enabled"] is False

    def test_format_vision_result_with_objects(self):
        result = VisionResult(
            timestamp=123456.0,
            frame_id=1,
            objects=[
                Detection(
                    class_name="person",
                    bbox=BoundingBox(x1=10, y1=20, x2=100, y2=200),
                    confidence=0.95
                )
            ],
            depth=DepthData(enabled=True, min_distance=1.5, max_distance=5.0),
            faces=[],
            gestures=[],
            pose=PoseData(detected=False),
        )

        output = format_vision_result(result)
        data = json.loads(output)

        assert len(data["objects"]) == 1
        assert data["objects"][0]["class"] == "person"
        assert data["objects"][0]["bbox"] == [10.0, 20.0, 100.0, 200.0]
        assert data["objects"][0]["confidence"] == 0.95
        assert data["depth"]["enabled"] is True
        assert data["depth"]["min_distance"] == 1.5
        assert data["depth"]["max_distance"] == 5.0

    def test_format_vision_result_with_faces(self):
        result = VisionResult(
            timestamp=123456.0,
            frame_id=1,
            objects=[],
            depth=DepthData(enabled=False),
            faces=[
                FaceDetection(
                    bbox=BoundingBox(x1=50, y1=60, x2=150, y2=160),
                    confidence=0.98
                )
            ],
            gestures=[],
            pose=PoseData(detected=False),
        )

        output = format_vision_result(result)
        data = json.loads(output)

        assert len(data["faces"]) == 1
        assert data["faces"][0]["bbox"] == [50.0, 60.0, 150.0, 160.0]
        assert data["faces"][0]["confidence"] == 0.98

    def test_format_vision_result_with_gestures(self):
        result = VisionResult(
            timestamp=123456.0,
            frame_id=1,
            objects=[],
            depth=DepthData(enabled=False),
            faces=[],
            gestures=[
                Gesture(
                    gesture_type="thumbs_up",
                    handedness="right",
                    confidence=0.92
                )
            ],
            pose=PoseData(detected=False),
        )

        output = format_vision_result(result)
        data = json.loads(output)

        assert len(data["gestures"]) == 1
        assert data["gestures"][0]["type"] == "thumbs_up"
        assert data["gestures"][0]["handedness"] == "right"
        assert data["gestures"][0]["confidence"] == 0.92

    def test_format_vision_result_with_pose(self):
        result = VisionResult(
            timestamp=123456.0,
            frame_id=1,
            objects=[],
            depth=DepthData(enabled=False),
            faces=[],
            gestures=[],
            pose=PoseData(
                detected=True,
                keypoints=[
                    PoseKeypoint(x=100.0, y=200.0, visibility=0.9)
                ]
            ),
        )

        output = format_vision_result(result)
        data = json.loads(output)

        assert data["pose"]["detected"] is True
        assert len(data["pose"]["keypoints"]) == 1
        assert data["pose"]["keypoints"][0]["x"] == 100.0

    def test_format_objects(self):
        detections = [
            Detection(
                class_name="cup",
                bbox=BoundingBox(x1=10, y1=20, x2=50, y2=60),
                confidence=0.85
            )
        ]

        output = format_objects(detections)
        data = json.loads(output)

        assert len(data["objects"]) == 1
        assert data["objects"][0]["class"] == "cup"


class TestUDPSender:
    def test_initialization(self):
        sender = UDPSender(host="127.0.0.1", port=5000)
        assert sender._host == "127.0.0.1"
        assert sender._port == 5000
        assert sender.is_opened is False

    @patch("src.output.udp_sender.socket.socket")
    def test_open(self, mock_socket_class):
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket

        sender = UDPSender()
        sender.open()

        assert sender.is_opened is True
        mock_socket_class.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)

    @patch("src.output.udp_sender.socket.socket")
    def test_send(self, mock_socket_class):
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket

        sender = UDPSender()
        sender.open()
        sender.send('{"test": "data"}')

        mock_socket.sendto.assert_called_once()

    @patch("src.output.udp_sender.socket.socket")
    def test_send_without_open_raises_error(self, mock_socket_class):
        sender = UDPSender()

        with pytest.raises(OutputError):
            sender.send('{"test": "data"}')

    @patch("src.output.udp_sender.socket.socket")
    def test_close(self, mock_socket_class):
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket

        sender = UDPSender()
        sender.open()
        sender.close()

        mock_socket.close.assert_called_once()
        assert sender.is_opened is False

    @patch("src.output.udp_sender.socket.socket")
    def test_close_when_not_open(self, mock_socket_class):
        sender = UDPSender()
        sender.close()

        mock_socket_class.return_value.close.assert_not_called()
