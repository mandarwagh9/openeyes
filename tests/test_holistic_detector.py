import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.models.holistic_detector import HolisticDetector
from src.camera.types import BoundingBox, FaceDetection, GestureDetection, PoseDetection
from src.exceptions import ModelError


class TestHolisticDetector:
    def test_initialization_default(self):
        detector = HolisticDetector()
        assert detector._model_complexity == 1
        assert detector._min_face_confidence == 0.3
        assert detector._min_pose_confidence == 0.3
        assert detector._min_hand_confidence == 0.1
        assert detector._enable_face is True
        assert detector._enable_pose is True
        assert detector._enable_hands is True

    def test_initialization_custom(self):
        detector = HolisticDetector(
            model_complexity=2,
            min_face_confidence=0.5,
            min_pose_confidence=0.5,
            min_hand_confidence=0.2,
            enable_face=False,
            enable_pose=True,
            enable_hands=False,
        )
        assert detector._model_complexity == 2
        assert detector._min_face_confidence == 0.5
        assert detector._min_pose_confidence == 0.5
        assert detector._min_hand_confidence == 0.2
        assert detector._enable_face is False
        assert detector._enable_pose is True
        assert detector._enable_hands is False

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_load(self, mock_holistic):
        mock_instance = MagicMock()
        mock_holistic.return_value = mock_instance

        detector = HolisticDetector()
        detector.load()

        assert detector.is_loaded is True
        mock_holistic.assert_called_once()

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_load_failure(self, mock_holistic):
        mock_holistic.side_effect = Exception("Load error")

        detector = HolisticDetector()
        with pytest.raises(ModelError):
            detector.load()

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_detect_without_loading(self, mock_holistic):
        detector = HolisticDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(ModelError):
            detector.detect(frame)

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_detect_returns_dict(self, mock_holistic):
        mock_instance = MagicMock()
        mock_holistic.return_value = mock_instance
        mock_instance.process.return_value = MagicMock(
            face_landmarks=None,
            pose_landmarks=None,
            left_hand_landmarks=None,
            right_hand_landmarks=None,
        )

        detector = HolisticDetector()
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)

        assert isinstance(results, dict)
        assert "faces" in results
        assert "poses" in results
        assert "hands" in results

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_detect_with_face_landmarks(self, mock_holistic):
        mock_instance = MagicMock()
        mock_holistic.return_value = mock_instance

        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5

        mock_instance.process.return_value = MagicMock(
            face_landmarks=[MagicMock(landmark=[mock_landmark] * 468)],
            pose_landmarks=None,
            left_hand_landmarks=None,
            right_hand_landmarks=None,
        )

        detector = HolisticDetector()
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)

        assert "faces" in results

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_detect_with_pose_landmarks(self, mock_holistic):
        mock_instance = MagicMock()
        mock_holistic.return_value = mock_instance

        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.visibility = 0.9

        mock_instance.process.return_value = MagicMock(
            face_landmarks=None,
            pose_landmarks=MagicMock(landmark=[mock_landmark] * 33),
            left_hand_landmarks=None,
            right_hand_landmarks=None,
        )

        detector = HolisticDetector()
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)

        assert "poses" in results
        assert len(results["poses"]) == 1

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_detect_with_hand_landmarks(self, mock_holistic):
        mock_instance = MagicMock()
        mock_holistic.return_value = mock_instance

        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5

        mock_hand_landmarks = MagicMock()
        mock_hand_landmarks.landmark = [mock_landmark] * 21

        mock_instance.process.return_value = MagicMock(
            face_landmarks=None,
            pose_landmarks=None,
            left_hand_landmarks=mock_hand_landmarks,
            right_hand_landmarks=mock_hand_landmarks,
        )

        detector = HolisticDetector()
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)

        assert "hands" in results

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_detect_disabled_face(self, mock_holistic):
        mock_instance = MagicMock()
        mock_holistic.return_value = mock_instance

        mock_result = MagicMock()
        mock_result.face_landmarks = None
        mock_result.pose_landmarks = None
        mock_result.left_hand_landmarks = None
        mock_result.right_hand_landmarks = None
        mock_instance.process.return_value = mock_result

        detector = HolisticDetector(enable_face=False)
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)

        assert results["faces"] == []

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_detect_disabled_pose(self, mock_holistic):
        mock_instance = MagicMock()
        mock_holistic.return_value = mock_instance

        mock_result = MagicMock()
        mock_result.face_landmarks = None
        mock_result.pose_landmarks = None
        mock_result.left_hand_landmarks = None
        mock_result.right_hand_landmarks = None
        mock_instance.process.return_value = mock_result

        detector = HolisticDetector(enable_pose=False)
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)

        assert results["poses"] == []

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_detect_disabled_hands(self, mock_holistic):
        mock_instance = MagicMock()
        mock_holistic.return_value = mock_instance

        mock_result = MagicMock()
        mock_result.face_landmarks = None
        mock_result.pose_landmarks = None
        mock_result.left_hand_landmarks = None
        mock_result.right_hand_landmarks = None
        mock_instance.process.return_value = mock_result

        detector = HolisticDetector(enable_hands=False)
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)

        assert results["hands"] == []

    def test_name_property(self):
        detector = HolisticDetector()
        assert detector.name == "MediaPipeHolistic"

    def test_is_loaded_before_load(self):
        detector = HolisticDetector()
        assert detector.is_loaded is False

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_is_loaded_after_load(self, mock_holistic):
        mock_instance = MagicMock()
        mock_holistic.return_value = mock_instance

        detector = HolisticDetector()
        detector.load()

        assert detector.is_loaded is True


class TestGestureClassification:
    @patch("mediapipe.solutions.holistic.Holistic")
    def test_fist_gesture(self, mock_holistic):
        detector = HolisticDetector()

        hand_landmarks = []
        for _ in range(21):
            lm = MagicMock()
            lm.x = 0.5
            lm.y = 0.5
            hand_landmarks.append(lm)

        fingers = [False, False, False, False, False]
        gesture = detector._identify_gesture(fingers)
        assert gesture == "fist"

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_peace_gesture(self, mock_holistic):
        detector = HolisticDetector()

        fingers = [False, True, True, False, False]
        gesture = detector._identify_gesture(fingers)
        assert gesture == "peace"

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_point_gesture(self, mock_holistic):
        detector = HolisticDetector()

        fingers = [True, False, False, False, False]
        gesture = detector._identify_gesture(fingers)
        assert gesture == "point"

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_open_palm_gesture(self, mock_holistic):
        detector = HolisticDetector()

        fingers = [False, True, True, True, True]
        gesture = detector._identify_gesture(fingers)
        assert gesture == "open_palm"

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_thumbs_up_gesture(self, mock_holistic):
        detector = HolisticDetector()

        fingers = [True, True, False, False, False]
        gesture = detector._identify_gesture(fingers)
        assert gesture == "thumbs_up"

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_thumbs_down_gesture(self, mock_holistic):
        detector = HolisticDetector()

        fingers = [False, False, True, True, True]
        gesture = detector._identify_gesture(fingers)
        assert gesture == "thumbs_down"

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_unknown_gesture(self, mock_holistic):
        detector = HolisticDetector()

        fingers = [True, False, True, False, True]
        gesture = detector._identify_gesture(fingers)
        assert gesture == "unknown"


class TestPoseLandmarks:
    def test_pose_landmark_names(self):
        detector = HolisticDetector()
        expected_landmarks = [
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        names = [name for _, name in HolisticDetector._POSE_landmarks]
        assert names == expected_landmarks


class TestDrawResults:
    @patch("mediapipe.solutions.holistic.Holistic")
    def test_draw_results_default(self, mock_holistic):
        mock_instance = MagicMock()
        mock_holistic.return_value = mock_instance

        detector = HolisticDetector()
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = {
            "faces": [],
            "poses": [],
            "hands": [],
        }

        result_frame = detector.draw_results(frame, results)
        assert result_frame.shape == frame.shape

    @patch("mediapipe.solutions.holistic.Holistic")
    def test_draw_results_with_debug(self, mock_holistic):
        mock_instance = MagicMock()
        mock_holistic.return_value = mock_instance

        detector = HolisticDetector()
        detector._debug = True
        detector.load()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face = FaceDetection(
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            confidence=0.8,
        )
        results = {
            "faces": [face],
            "poses": [],
            "hands": [],
        }

        result_frame = detector.draw_results(frame, results)
        assert result_frame.shape == frame.shape