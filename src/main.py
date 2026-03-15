import argparse
import logging
import signal
import sys
import time
from typing import Optional

import cv2

from src.camera.camera_handler import CameraHandler
from src.camera.types import DepthData, FaceDetection, Gesture, PoseData, VisionResult
from src.exceptions import CameraError, ModelError
from src.models.object_detector import ObjectDetector
from src.output.json_formatter import format_vision_result
from src.output.udp_sender import UDPSender
from src.utils.config import Config
from src.utils.logger import setup_logger


class VisionSystem:
    def __init__(self, config: Config):
        self._config = config
        self._logger = setup_logger(
            "project0",
            level=logging.DEBUG if config.debug else logging.INFO
        )
        self._camera: Optional[CameraHandler] = None
        self._detector: Optional[ObjectDetector] = None
        self._udp_sender: Optional[UDPSender] = None
        self._running = False
        self._frame_id = 0
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self._logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)

    def start(self) -> None:
        self._logger.info("Starting PROJECT0 Vision System")

        self._init_camera()
        self._init_detector()
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

    def _init_detector(self) -> None:
        self._detector = ObjectDetector(
            model_path=self._config.yolo_path,
            confidence=self._config.yolo_confidence,
            iou_threshold=self._config.yolo_iou_threshold,
        )
        try:
            self._detector.load()
        except ModelError as e:
            self._logger.error(f"Detector initialization failed: {e}")
            raise

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

        result = VisionResult(
            timestamp=timestamp,
            frame_id=self._frame_id,
            objects=detections,
            depth=DepthData(enabled=False),
            faces=[],
            gestures=[],
            pose=PoseData(detected=False),
        )

        return result

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

        cv2.imshow("PROJECT0 Debug", frame)
        cv2.waitKey(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="PROJECT0 Vision System")
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

    args = parser.parse_args()

    config = Config()
    if args.camera is not None:
        config._config["camera"]["source"] = args.camera
    if args.debug:
        config._config["debug"] = True
    if args.config:
        config._config_path = args.config

    try:
        system = VisionSystem(config)
        system.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
