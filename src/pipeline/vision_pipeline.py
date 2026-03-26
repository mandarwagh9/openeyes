import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera.camera_handler import CameraHandler
from src.camera.types import (
    DepthData,
    Detection,
    FaceDetection,
    Gesture,
    PoseData,
    VisionResult,
)
from src.exceptions import CameraError, ModelError
from src.models import (
    DepthEstimator,
    FaceDetector,
    GestureRecognizer,
    ObjectDetector,
    PoseEstimator,
)
from src.output.json_formatter import format_vision_result
from src.output.udp_sender import UDPSender
from src.utils.logger import get_logger


class VisionPipeline:
    def __init__(
        self,
        camera_source: int = 0,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        yolo_path: str = "models/yolov8n.pt",
        yolo_confidence: float = 0.5,
        output_host: str = "127.0.0.1",
        output_port: int = 5000,
        enable_face: bool = True,
        enable_gesture: bool = True,
        enable_pose: bool = True,
        enable_depth: bool = False,
        debug: bool = False,
    ):
        self._logger = get_logger(__name__)

        self._camera_source = camera_source
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._camera_fps = camera_fps

        self._yolo_path = yolo_path
        self._yolo_confidence = yolo_confidence

        self._output_host = output_host
        self._output_port = output_port

        self._enable_face = enable_face
        self._enable_gesture = enable_gesture
        self._enable_pose = enable_pose
        self._enable_depth = enable_depth

        self._debug = debug

        self._camera: Optional[CameraHandler] = None
        self._object_detector: Optional[ObjectDetector] = None
        self._depth_estimator: Optional[DepthEstimator] = None
        self._face_detector: Optional[FaceDetector] = None
        self._gesture_recognizer: Optional[GestureRecognizer] = None
        self._pose_estimator: Optional[PoseEstimator] = None
        self._udp_sender: Optional[UDPSender] = None

        self._running = False
        self._frame_id = 0
        self._fps_counter = 0
        self._fps_start_time = time.time()

    def start(self) -> None:
        self._logger.info("Starting Vision Pipeline")

        self._init_camera()
        self._init_models()
        self._init_output()

        self._running = True
        self._logger.info("Vision Pipeline started successfully")

    def stop(self) -> None:
        self._running = False

        if self._camera:
            self._camera.release()

        if self._udp_sender:
            self._udp_sender.close()

        self._logger.info("Vision Pipeline stopped")

    def _init_camera(self) -> None:
        self._camera = CameraHandler(
            source=self._camera_source,
            width=self._camera_width,
            height=self._camera_height,
            fps=self._camera_fps,
        )
        try:
            self._camera.open()
            self._logger.info(f"Camera initialized: {self._camera_width}x{self._camera_height} @ {self._camera_fps} FPS")
        except CameraError as e:
            self._logger.error(f"Camera initialization failed: {e}")
            raise

    def _init_models(self) -> None:
        self._object_detector = ObjectDetector(
            model_path=self._yolo_path,
            confidence=self._yolo_confidence,
        )
        try:
            self._object_detector.load()
            self._logger.info("Object Detector loaded")
        except ModelError as e:
            self._logger.error(f"Object Detector failed: {e}")
            raise

        if self._enable_depth:
            self._depth_estimator = DepthEstimator()
            self._depth_estimator.load()
            self._logger.info("Depth Estimator loaded")

        if self._enable_face:
            self._face_detector = FaceDetector()
            self._face_detector.load()
            self._logger.info("Face Detector loaded")

        if self._enable_gesture:
            self._gesture_recognizer = GestureRecognizer()
            self._gesture_recognizer.load()
            self._logger.info("Gesture Recognizer loaded")

        if self._enable_pose:
            self._pose_estimator = PoseEstimator()
            self._pose_estimator.load()
            self._logger.info("Pose Estimator loaded")

    def _init_output(self) -> None:
        self._udp_sender = UDPSender(
            host=self._output_host,
            port=self._output_port,
        )
        self._udp_sender.open()
        self._logger.info(f"UDP output configured: {self._output_host}:{self._output_port}")

    def process_frame(self, frame: np.ndarray) -> VisionResult:
        timestamp = time.time()

        detections = []
        if self._object_detector:
            detections = self._object_detector.detect(frame)

        depth_data = DepthData(enabled=self._enable_depth)

        faces: List[FaceDetection] = []
        if self._face_detector and self._enable_face:
            faces = self._face_detector.detect(frame)

        gestures: List[Gesture] = []
        if self._gesture_recognizer and self._enable_gesture:
            gestures = self._gesture_recognizer.recognize(frame)

        pose_data = PoseData(detected=False)
        if self._pose_estimator and self._enable_pose:
            pose_data = self._pose_estimator.estimate(frame)

        result = VisionResult(
            timestamp=timestamp,
            frame_id=self._frame_id,
            objects=detections,
            depth=depth_data,
            faces=faces,
            gestures=gestures,
            pose=pose_data,
        )

        self._frame_id += 1
        return result

    def get_result(self) -> Optional[VisionResult]:
        if not self._camera or not self._running:
            return None

        frame = self._camera.read()
        if frame is None:
            return None

        return self.process_frame(frame)

    def run(self) -> None:
        self.start()

        while self._running:
            result = self.get_result()

            if result:
                json_output = format_vision_result(result)
                self._udp_sender.send(json_output)

                self._fps_counter += 1
                if time.time() - self._fps_start_time >= 1.0:
                    fps = self._fps_counter / (time.time() - self._fps_start_time)
                    self._logger.info(
                        f"FPS: {fps:.1f} | Objects: {len(result.objects)} | "
                        f"Faces: {len(result.faces)} | Gestures: {len(result.gestures)}"
                    )
                    self._fps_counter = 0
                    self._fps_start_time = time.time()

                time.sleep(0.033)

        self.stop()

    @property
    def is_running(self) -> bool:
        return self._running
