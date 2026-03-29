from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from src.camera.types import BoundingBox, Detection
from src.exceptions import ModelError
from src.utils.logger import get_logger


class GraspDetector:
    """Grasping point detector for robot manipulation."""

    def __init__(
        self,
        model_path: str = "models/grasp_model.onnx",
        confidence: float = 0.5,
    ):
        self._model_path = model_path
        self._confidence = confidence
        self._model = None
        self._logger = get_logger(__name__)
        self._session = None

    def load(self) -> None:
        """Load the grasp detection model."""
        model_file = Path(self._model_path)

        if not model_file.exists():
            self._logger.warning(
                f"Grasp model not found: {self._model_path}. "
                "Falling back to detection-based grasp estimation."
            )
            return

        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            providers = []
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append(("CUDAExecutionProvider", {"device_id": 0}))
            providers.append("CPUExecutionProvider")

            self._session = ort.InferenceSession(
                self._model_path,
                sess_options,
                providers=providers,
            )

            self._logger.info(f"Grasp detector loaded: {self._model_path}")

        except Exception as e:
            self._logger.warning(f"Failed to load grasp model: {e}")
            self._logger.info("Using detection-based grasp estimation")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect grasp points in the frame."""
        if self._session is None:
            return self._estimate_grasp_from_detection(frame)

        try:
            input_name = self._session.get_inputs()[0].name
            output_name = self._session.get_outputs()[0].name

            img = self._preprocess(frame, (640, 640))

            outputs = self._session.run([output_name], {input_name: img})

            return self._postprocess(outputs[0], frame.shape)

        except Exception as e:
            self._logger.warning(f"Grasp detection failed: {e}")
            return self._estimate_grasp_from_detection(frame)

    def _preprocess(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Preprocess frame for grasp detection."""
        import cv2

        h, w = frame.shape[:2]
        target_h, target_w = target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        img = padded.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return img

    def _postprocess(self, output: np.ndarray, original_shape: Tuple) -> List[Detection]:
        """Postprocess grasp detection output."""
        detections = []

        for pred in output[0]:
            if len(pred) < 6:
                continue

            x1, y1, x2, y2, conf, angle = pred[:6]

            if conf < self._confidence:
                continue

            detection = Detection(
                class_name="grasp_point",
                bbox=BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                ),
                confidence=float(conf),
            )
            detections.append(detection)

        return detections

    def _estimate_grasp_from_detection(self, frame: np.ndarray) -> List[Detection]:
        """Estimate grasp points from detected objects."""
        from src.models.object_detector import ObjectDetector

        detector = ObjectDetector(
            model_path="models/yolo11n.onnx",
            confidence=0.3,
        )

        try:
            detector.load()
            objects = detector.detect(frame)

            grasp_points = []
            for obj in objects:
                if obj.class_name in ["bottle", "cup", "bowl", "phone", "remote"]:
                    bbox = obj.bbox
                    cx = (bbox.x1 + bbox.x2) / 2
                    cy = (bbox.y1 + bbox.y2) / 2

                    grasp_points.append(Detection(
                        class_name="grasp_point",
                        bbox=BoundingBox(
                            x1=cx - 20,
                            y1=cy - 20,
                            x2=cx + 20,
                            y2=cy + 20,
                        ),
                        confidence=obj.confidence * 0.8,
                    ))

            return grasp_points

        except Exception as e:
            self._logger.warning(f"Estimation failed: {e}")
            return []

    @property
    def is_loaded(self) -> bool:
        return self._session is not None


class FallDetector:
    """Human fall detection model for safety monitoring."""

    def __init__(
        self,
        model_path: str = "models/fall_model.onnx",
        confidence: float = 0.5,
    ):
        self._model_path = model_path
        self._confidence = confidence
        self._model = None
        self._logger = get_logger(__name__)
        self._session = None

    def load(self) -> None:
        """Load the fall detection model."""
        model_file = Path(self._model_path)

        if not model_file.exists():
            self._logger.warning(
                f"Fall model not found: {self._model_path}. "
                "Using pose-based fall detection."
            )
            return

        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            providers = []
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append(("CUDAExecutionProvider", {"device_id": 0}))
            providers.append("CPUExecutionProvider")

            self._session = ort.InferenceSession(
                self._model_path,
                sess_options,
                providers=providers,
            )

            self._logger.info(f"Fall detector loaded: {self._model_path}")

        except Exception as e:
            self._logger.warning(f"Failed to load fall model: {e}")
            self._logger.info("Using pose-based fall detection")

    def detect(self, frame: np.ndarray, pose_data: dict = None) -> dict:
        """Detect if a person has fallen."""
        if self._session is not None:
            return self._detect_from_model(frame)
        else:
            return self._detect_from_pose(pose_data, frame.shape)

    def _detect_from_model(self, frame: np.ndarray) -> dict:
        """Detect fall using ML model."""
        try:
            input_name = self._session.get_inputs()[0].name
            output_name = self._session.get_outputs()[0].name

            img = self._preprocess(frame, (640, 640))

            outputs = self._session.run([output_name], {input_name: img})

            result = outputs[0][0]

            status = "standing"
            confidence = float(result[0])

            if result[1] > self._confidence:
                status = "falling"
                confidence = float(result[1])
            elif result[2] > self._confidence:
                status = "fallen"
                confidence = float(result[2])

            return {
                "status": status,
                "confidence": confidence,
                "is_fall": status in ["falling", "fallen"],
            }

        except Exception as e:
            self._logger.warning(f"Fall detection failed: {e}")
            return {"status": "unknown", "confidence": 0.0, "is_fall": False}

    def _detect_from_pose(self, pose_data: Optional[dict], frame_shape: Tuple) -> dict:
        """Detect fall from pose keypoints."""
        if pose_data is None or not pose_data.get("detected"):
            return {"status": "unknown", "confidence": 0.0, "is_fall": False}

        try:
            keypoints = pose_data.get("keypoints", [])
            if len(keypoints) < 11:
                return {"status": "unknown", "confidence": 0.0, "is_fall": False}

            h, w = frame_shape[:2]

            nose = keypoints[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]

            if not all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
                return {"status": "unknown", "confidence": 0.0, "is_fall": False}

            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2

            body_height = abs(hip_y - shoulder_y)
            if body_height < 10:
                return {"status": "unknown", "confidence": 0.0, "is_fall": False}

            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            body_ratio = body_height / max(shoulder_width, 1)

            if body_ratio < 1.5:
                return {
                    "status": "fallen",
                    "confidence": 0.85,
                    "is_fall": True,
                }
            elif body_ratio < 2.5:
                return {
                    "status": "falling",
                    "confidence": 0.7,
                    "is_fall": True,
                }

            return {
                "status": "standing",
                "confidence": 0.9,
                "is_fall": False,
            }

        except Exception as e:
            self._logger.warning(f"Pose-based fall detection failed: {e}")
            return {"status": "unknown", "confidence": 0.0, "is_fall": False}

    def _preprocess(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Preprocess frame for fall detection."""
        import cv2

        h, w = frame.shape[:2]
        target_h, target_w = target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        img = padded.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return img

    @property
    def is_loaded(self) -> bool:
        return self._session is not None
