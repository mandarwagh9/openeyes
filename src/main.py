import argparse
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2

if not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':0'

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera.camera_handler import CameraHandler
from src.camera.types import DepthData, FaceDetection, Gesture, PoseData, VisionResult
from src.exceptions import CameraError, ModelError
from src.models.object_detector import ObjectDetector
from src.models.depth_estimator import DepthEstimator
from src.models.face_detector import FaceDetector
from src.models.gesture_recognizer import GestureRecognizer
from src.models.pose_estimator import PoseEstimator
from src.output.json_formatter import format_vision_result
from src.output.udp_sender import UDPSender
from src.utils.config import Config
from src.utils.logger import setup_logger


class VisionSystem:
    """Optimized vision system with parallel processing."""

    def __init__(self, config: Config):
        self._config = config
        self._logger = setup_logger(
            "openeyes",
            level=logging.DEBUG if config.debug else logging.INFO
        )
        self._camera: Optional[CameraHandler] = None
        self._detector: Optional[ObjectDetector] = None
        self._depth_estimator: Optional[DepthEstimator] = None
        self._face_detector: Optional[FaceDetector] = None
        self._gesture_recognizer: Optional[GestureRecognizer] = None
        self._pose_estimator: Optional[PoseEstimator] = None
        self._udp_sender: Optional[UDPSender] = None
        self._running = False
        self._frame_id = 0

        self._fps_counter = 0
        self._fps_start_time = time.time()
        self._last_pose = None
        self._last_faces = []
        self._last_gestures = []

        self._use_parallel = True
        self._pose_skip_frames = 1

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self._logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)

    def start(self) -> None:
        self._logger.info("Starting OpenEyes Vision System (Optimized)")
        self._logger.info(f"Parallel processing: {self._use_parallel}")
        self._logger.info(f"Pose skip frames: {self._pose_skip_frames + 1}")

        self._init_camera()
        self._init_detectors()
        self._init_output()

        self._running = True
        self._logger.info("Vision system started successfully")

        self._process_loop()

    def stop(self) -> None:
        self._running = False

        if self._camera:
            self._camera.release()

        if self._udp_sender:
            self._udp_sender.close()

        self._logger.info("Vision system stopped")

    def _init_camera(self) -> None:
        self._camera = CameraHandler(
            source=self._config.camera_source,
            width=self._config.camera_width,
            height=self._config.camera_height,
            fps=self._config.camera_fps,
        )
        try:
            self._camera.open()
        except CameraError as e:
            self._logger.error(f"Camera initialization failed: {e}")
            raise

    def _init_detectors(self) -> None:
        self._detector = ObjectDetector(
            model_path=self._config.yolo_path or "models/yolov10n.onnx",
            confidence=self._config.yolo_confidence,
            iou_threshold=self._config.yolo_iou_threshold,
        )
        try:
            self._detector.load()
            self._logger.info(f"Object Detector loaded: {self._detector.name}")
        except ModelError as e:
            self._logger.error(f"Detector initialization failed: {e}")
            raise

        try:
            self._depth_estimator = DepthEstimator()
            self._depth_estimator.load()
            if self._depth_estimator.is_loaded:
                self._logger.info("Depth Estimator loaded")
            else:
                self._logger.warning("Depth Estimator using fallback")
        except ModelError as e:
            self._logger.warning(f"Depth Estimator not available: {e}")

        try:
            self._face_detector = FaceDetector()
            self._face_detector.load()
            self._logger.info("Face Detector loaded")
        except ModelError as e:
            self._logger.warning(f"Face Detector not available: {e}")

        try:
            self._gesture_recognizer = GestureRecognizer()
            self._gesture_recognizer.load()
            self._logger.info("Gesture Recognizer loaded")
        except ModelError as e:
            self._logger.warning(f"Gesture Recognizer not available: {e}")

        try:
            self._pose_estimator = PoseEstimator()
            self._pose_estimator.load()
            self._logger.info("Pose Estimator loaded")
        except ModelError as e:
            self._logger.warning(f"Pose Estimator not available: {e}")

    def _init_output(self) -> None:
        self._udp_sender = UDPSender(
            host=self._config.output_host,
            port=self._config.output_port,
        )
        self._udp_sender.open()

    def _process_loop(self) -> None:
        frame_time = 1.0 / self._config.target_fps

        while self._running:
            loop_start = time.time()

            frame = self._camera.read()
            if frame is None:
                self._logger.warning("No frame received, skipping")
                time.sleep(0.1)
                continue

            result = self._process_frame(frame)

            json_output = format_vision_result(result)
            self._udp_sender.send(json_output)

            if self._config.debug:
                self._debug_display(frame, result)

            self._fps_counter += 1
            elapsed_total = time.time() - self._fps_start_time
            if elapsed_total >= 1.0:
                fps = self._fps_counter / elapsed_total
                self._logger.info(
                    f"FPS: {fps:.1f} | Objects: {len(result.objects)} | "
                    f"Faces: {len(result.faces)} | Gestures: {len(result.gestures)}"
                )
                self._fps_counter = 0
                self._fps_start_time = time.time()

            elapsed = time.time() - loop_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            self._frame_id += 1

    def _process_frame(self, frame) -> VisionResult:
        timestamp = time.time()

        detections = []
        if self._detector:
            detections = self._detector.detect(frame)

        depth = DepthData(enabled=False)

        if self._use_parallel:
            faces, gestures, pose = self._process_models_parallel(frame)
        else:
            faces, gestures, pose = self._process_models_sequential(frame)

        result = VisionResult(
            timestamp=timestamp,
            frame_id=self._frame_id,
            objects=detections,
            depth=depth,
            faces=faces,
            gestures=gestures,
            pose=pose,
        )

        return result

    def _process_models_parallel(
        self, frame
    ) -> tuple:
        """Process face, gesture, and pose in parallel."""
        faces = []
        gestures = []
        pose = PoseData(detected=False)

        results = {}

        def safe_face():
            try:
                return self._face_detector.detect(frame)
            except Exception as e:
                return []

        def safe_gesture():
            try:
                return self._gesture_recognizer.recognize(frame)
            except Exception as e:
                return []

        def safe_pose():
            try:
                return self._pose_estimator.estimate(frame)
            except Exception as e:
                return PoseData(detected=False)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []

            if self._face_detector:
                futures.append(("face", executor.submit(safe_face)))

            if self._gesture_recognizer:
                futures.append(("gesture", executor.submit(safe_gesture)))

            if self._pose_estimator:
                if self._frame_id % (self._pose_skip_frames + 1) == 0:
                    futures.append(("pose", executor.submit(safe_pose)))
                else:
                    pose = self._last_pose if self._last_pose else PoseData(detected=False)

            for key, future in futures:
                try:
                    result = future.result()
                    if key == "face":
                        faces = result if isinstance(result, list) else []
                        if faces:
                            self._last_faces = faces
                    elif key == "gesture":
                        gestures = result if isinstance(result, list) else []
                        if gestures:
                            self._last_gestures = gestures
                    elif key == "pose":
                        if hasattr(result, 'detected'):
                            pose = result
                            if result.detected:
                                self._last_pose = result
                except Exception as e:
                    self._logger.warning(f"Model {key} failed: {e}")

        if not faces and self._last_faces:
            faces = self._last_faces
        if not gestures and self._last_gestures:
            gestures = self._last_gestures
        if not pose.detected and self._last_pose:
            pose = self._last_pose

        return faces, gestures, pose

    def _process_models_sequential(
        self, frame
    ) -> tuple:
        """Process models sequentially (fallback)."""
        faces: list[FaceDetection] = []
        if self._face_detector:
            try:
                faces = self._face_detector.detect(frame)
            except Exception as e:
                self._logger.warning(f"Face detection failed: {e}")

        gestures: list[Gesture] = []
        if self._gesture_recognizer:
            try:
                gestures = self._gesture_recognizer.recognize(frame)
            except Exception as e:
                self._logger.warning(f"Gesture recognition failed: {e}")

        pose = PoseData(detected=False)
        if self._pose_estimator:
            if self._frame_id % (self._pose_skip_frames + 1) == 0:
                try:
                    pose = self._pose_estimator.estimate(frame)
                    self._last_pose = pose
                except Exception as e:
                    self._logger.warning(f"Pose estimation failed: {e}")
            else:
                pose = self._last_pose if self._last_pose else PoseData(detected=False)

        return faces, gestures, pose

    def _debug_display(self, frame, result: VisionResult) -> None:
        for det in result.objects:
            bbox = det.bbox
            cv2.rectangle(
                frame,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (0, 255, 0),
                2,
            )
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (int(bbox.x1), int(bbox.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        for face in result.faces:
            bbox = face.bbox
            cv2.rectangle(
                frame,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (255, 0, 0),
                2,
            )

        cv2.imshow("OpenEyes Debug", frame)
        cv2.waitKey(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEyes Vision System")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera source index",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Output host IP",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Output port",
    )
    parser.add_argument(
        "--no-face",
        action="store_true",
        help="Disable face detection",
    )
    parser.add_argument(
        "--no-gesture",
        action="store_true",
        help="Disable gesture recognition",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable pose estimation",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--pose-every",
        type=int,
        default=2,
        help="Run pose estimation every N frames",
    )

    args = parser.parse_args()

    config = Config()
    if args.camera is not None:
        config._config["camera"]["source"] = args.camera
    if args.debug:
        config._config["debug"] = True
    if args.config:
        config._config_path = args.config
    if args.width:
        config._config["camera"]["width"] = args.width
    if args.height:
        config._config["camera"]["height"] = args.height
    if args.fps:
        config._config["camera"]["fps"] = args.fps

    try:
        system = VisionSystem(config)

        if args.no_parallel:
            system._use_parallel = False
        if args.pose_every:
            system._pose_skip_frames = args.pose_every - 1

        system.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
