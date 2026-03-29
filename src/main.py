import argparse
import logging
import os
import signal
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import cv2

if not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':0'

sys.path.insert(0, str(Path(__file__).parent.parent))

ros2_python_paths = [
    '/opt/ros/humble/local/lib/python3.10/dist-packages',
    '/opt/ros/humble/lib/python3/dist-packages',
    '/usr/lib/python3/dist-packages',
]
for p in ros2_python_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

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
from src.utils.frame_skipper import FrameSkipProcessor, AdaptiveFrameSkipper, MultiModelFrameScheduler

ROS2_AVAILABLE = False
try:
    import rclpy
    from src.ros2.vision_node import VisionPublisher
    ROS2_AVAILABLE = True
except ImportError:
    VisionPublisher = None
    rclpy = None


class VisionSystem:
    """Optimized vision system with parallel processing."""

    def __init__(self, config: Config, use_ros2: bool = False):
        self._config = config
        self._use_ros2 = use_ros2
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
        self._ros2_pub: Optional[VisionPublisher] = None
        self._ros2_node = None
        self._running = False
        self._frame_id = 0

        self._fps_counter = 0
        self._fps_start_time = time.time()
        self._last_pose = None
        self._last_faces = []
        self._last_gestures = []
        self._last_depth = None

        self._use_parallel = True
        self._pose_skip_frames = 1

        self._use_face = True
        self._use_gesture = True
        self._use_pose = True
        self._use_depth = True

        self._frame_scheduler: Optional[MultiModelFrameScheduler] = None
        self._adaptive_skipper: Optional[AdaptiveFrameSkipper] = None

        self._ros2_queue: Optional[Queue] = None
        self._ros2_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=5)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self._logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)

    def _init_frame_scheduler(self) -> None:
        skip_intervals = {
            'detector': 1,
            'depth': 8,
            'face': 6,
            'gesture': 6,
            'pose': 6
        }
        self._frame_scheduler = MultiModelFrameScheduler(skip_intervals)
        self._adaptive_skipper = AdaptiveFrameSkipper(
            base_skip=2,
            motion_threshold=5000.0,
            min_skip=1,
            max_skip=4
        )
        self._logger.info(f"Frame scheduler initialized: {skip_intervals}")

    def start(self) -> None:
        self._logger.info("Starting OpenEyes Vision System (Optimized)")
        self._logger.info(f"Parallel processing: {self._use_parallel}")
        self._logger.info(f"Pose skip frames: {self._pose_skip_frames + 1}")

        enabled_models = []
        disabled_models = []
        if self._detector:
            enabled_models.append("detector")
        if self._use_depth and self._depth_estimator:
            enabled_models.append("depth")
        elif self._use_depth:
            enabled_models.append("depth")
        else:
            disabled_models.append("depth")
        if self._use_face and self._face_detector:
            enabled_models.append("face")
        elif self._use_face:
            enabled_models.append("face")
        else:
            disabled_models.append("face")
        if self._use_gesture and self._gesture_recognizer:
            enabled_models.append("gesture")
        elif self._use_gesture:
            enabled_models.append("gesture")
        else:
            disabled_models.append("gesture")
        if self._use_pose and self._pose_estimator:
            enabled_models.append("pose")
        elif self._use_pose:
            enabled_models.append("pose")
        else:
            disabled_models.append("pose")

        self._logger.info(f"Enabled models: {', '.join(enabled_models) if enabled_models else 'none'}")
        if disabled_models:
            self._logger.info(f"Disabled models: {', '.join(disabled_models)}")

        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
                if "jetson" in model or "tegra" in model:
                    self._logger.info("Jetson detected. Run 'sudo nvpmodel -m 0 && sudo jetson_clocks' for max performance")
        except Exception:
            pass

        self._init_frame_scheduler()
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

        if self._use_depth:
            try:
                self._depth_estimator = DepthEstimator()
                self._depth_estimator.load()
                if self._depth_estimator.is_loaded:
                    self._logger.info("Depth Estimator loaded")
                else:
                    self._logger.warning("Depth Estimator using fallback")
            except ModelError as e:
                self._logger.warning(f"Depth Estimator not available: {e}")

        if self._use_face:
            try:
                self._face_detector = FaceDetector()
                self._face_detector.load()
                if self._config.debug:
                    self._face_detector._debug = True
                self._logger.info("Face Detector loaded")
            except ModelError as e:
                self._logger.warning(f"Face Detector not available: {e}")

        if self._use_gesture:
            try:
                self._gesture_recognizer = GestureRecognizer()
                self._gesture_recognizer.load()
                if self._config.debug:
                    self._gesture_recognizer._debug = True
                self._logger.info("Gesture Recognizer loaded")
            except ModelError as e:
                self._logger.warning(f"Gesture Recognizer not available: {e}")

        if self._use_pose:
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

        if self._use_ros2 and ROS2_AVAILABLE and VisionPublisher:
            self._init_ros2()
        elif self._use_ros2 and not ROS2_AVAILABLE:
            self._logger.warning("ROS2 requested but not available. Install ros-humble-vision-msgs")

    def _init_ros2(self) -> None:
        """Initialize ROS2 publisher."""
        try:
            self._logger.info("Initializing ROS2...")
            if not rclpy.ok():
                self._logger.info("Calling rclpy.init()")
                rclpy.init()
            
            self._logger.info(f"rclpy.ok() = {rclpy.ok()}")

            self._ros2_pub = VisionPublisher(
                detections_topic="/vision/detections",
                depth_topic="/vision/depth",
                faces_topic="/vision/faces",
                gestures_topic="/vision/gestures",
                poses_topic="/vision/poses",
                cmd_topic="/vision/cmd",
                status_topic="/vision/status",
                frame_id="camera_link",
                confidence_threshold=self._config.yolo_confidence,
                max_depth_range=5.0,
            )

            def handle_cmd(cmd: str):
                self._logger.info(f">>> ROBOT COMMAND: {cmd.upper()}")

            self._ros2_pub.set_cmd_callback(handle_cmd)

            self._ros2_node = rclpy.node.Node("openeyes_main")
            
            self._ros2_executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
            self._ros2_executor.add_node(self._ros2_pub)
            
            self._ros2_queue = Queue(maxsize=2)
            
            def ros2_queue_processor():
                while self._running:
                    try:
                        result, frame_shape = self._ros2_queue.get(timeout=0.05)
                        self._publish_ros2(result, frame_shape)
                    except Empty:
                        continue
                    except Exception as e:
                        self._logger.warning(f"ROS2 queue error: {e}")
            
            self._ros2_queue_thread = threading.Thread(target=ros2_queue_processor, daemon=True)
            self._ros2_queue_thread.start()
            
            self._ros2_thread = threading.Thread(target=self._ros2_executor.spin, daemon=True)
            self._ros2_thread.start()
            
            self._logger.info("ROS2 Vision Publisher initialized")
            self._logger.info("  Subscribed to: /vision/cmd")
            self._logger.info("  Publishing: /vision/detections, /vision/depth, /vision/faces, /vision/gestures, /vision/poses, /vision/status")
            
            time.sleep(0.5)
            self._logger.info("ROS2 initialization complete")
        except Exception as e:
            self._logger.error(f"Failed to initialize ROS2: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            self._ros2_pub = None

    def _process_loop(self) -> None:
        frame_time = 1.0 / self._config.target_fps
        frame_shape = (self._config.camera_height, self._config.camera_width, 3)

        while self._running:
            loop_start = time.time()

            frame = self._camera.read()
            if frame is None:
                self._logger.warning("No frame received, skipping")
                time.sleep(0.1)
                continue

            frame_shape = frame.shape
            result = self._process_frame(frame)

            json_output = format_vision_result(result)
            self._udp_sender.send(json_output)

            if self._ros2_pub and self._ros2_queue:
                try:
                    self._ros2_queue.put_nowait((result, frame_shape))
                except:
                    pass

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
                if self._ros2_pub:
                    self._ros2_pub.publish_status(
                        fps,
                        len(result.objects),
                        len(result.faces),
                        len(result.gestures)
                    )
                self._fps_counter = 0
                self._fps_start_time = time.time()

            elapsed = time.time() - loop_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            self._frame_id += 1

    def _publish_ros2(self, result: VisionResult, frame_shape: tuple) -> None:
        """Publish vision results to ROS2 topics."""
        if not self._ros2_pub:
            return

        try:
            detections = []
            for obj in result.objects:
                detections.append({
                    "bbox": [obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2],
                    "class_name": obj.class_name,
                    "confidence": obj.confidence
                })
            self._ros2_pub.publish_detections(detections, (frame_shape[1], frame_shape[0]))
            self._logger.debug(f"Published {len(detections)} detections")

            if result.depth and result.depth.enabled:
                self._ros2_pub.publish_depth(result.depth, (frame_shape[1], frame_shape[0]))

            if result.faces:
                self._ros2_pub.publish_faces(result.faces, (frame_shape[1], frame_shape[0]))

            if result.gestures:
                self._ros2_pub.publish_gestures(result.gestures)

            if result.pose and result.pose.detected:
                self._ros2_pub.publish_poses(result.pose, (frame_shape[1], frame_shape[0]))

        except Exception as e:
            self._logger.warning(f"ROS2 publish error: {e}")

    def _process_frame(self, frame) -> VisionResult:
        timestamp = time.time()

        if self._frame_scheduler and self._adaptive_skipper:
            should_process = self._adaptive_skipper.should_process(frame)
            if not should_process:
                last_objects = self._frame_scheduler.get_last('detector')
                last_faces = self._frame_scheduler.get_last('face')
                last_gestures = self._frame_scheduler.get_last('gesture')
                last_pose = self._frame_scheduler.get_last('pose')

                last_depth_data = DepthData(
                    enabled=self._last_depth is not None,
                    depth_map=self._last_depth
                )
                return VisionResult(
                    timestamp=timestamp,
                    frame_id=self._frame_id,
                    objects=last_objects if last_objects else [],
                    depth=last_depth_data,
                    faces=last_faces if last_faces else [],
                    gestures=last_gestures if last_gestures else [],
                    pose=last_pose if last_pose else PoseData(detected=False),
                )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        futures = {}
        
        if self._detector:
            futures['detector'] = self._executor.submit(self._detector.detect, frame)
        
        if self._depth_estimator and self._depth_estimator.is_loaded:
            futures['depth'] = self._executor.submit(self._depth_estimator.estimate, frame)
        
        if self._face_detector:
            futures['face'] = self._executor.submit(self._face_detector.detect, frame_rgb)
        
        if self._gesture_recognizer:
            futures['gesture'] = self._executor.submit(self._gesture_recognizer.recognize, frame_rgb)
        
        if self._pose_estimator:
            futures['pose'] = self._executor.submit(self._pose_estimator.estimate, frame_rgb)

        detections = []
        depth_map = None
        depth_enabled = False
        faces = []
        gestures = []
        pose = PoseData(detected=False)

        for key, future in futures.items():
            try:
                result = future.result()
                if key == 'detector':
                    detections = result
                    if self._frame_scheduler:
                        self._frame_scheduler.update('detector', detections)
                elif key == 'depth':
                    depth_map = result
                    depth_enabled = True
                elif key == 'face':
                    faces = result
                elif key == 'gesture':
                    gestures = result
                elif key == 'pose':
                    pose = result
            except Exception as e:
                self._logger.warning(f"Model {key} failed: {e}")

        depth = DepthData(enabled=depth_enabled, depth_map=depth_map)

        if depth_map is not None:
            self._last_depth = depth_map

        if self._frame_scheduler:
            self._frame_scheduler.update('face', faces)
            self._frame_scheduler.update('gesture', gestures)
            self._frame_scheduler.update('pose', pose)
            self._frame_scheduler.next_frame()

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
        """Process face, gesture, and pose in parallel with frame skipping."""
        faces = []
        gestures = []
        pose = PoseData(detected=False)

        skip_face = False
        skip_gesture = False
        skip_pose = False

        if skip_face and self._last_faces:
            return self._last_faces, self._last_gestures, self._last_pose if self._last_pose else PoseData(detected=False)

        results = {}

        def safe_face():
            try:
                if self._face_detector:
                    return self._face_detector.detect(frame)
                return []
            except Exception as e:
                self._logger.warning(f"Face detection failed: {e}")
                return []

        def safe_gesture():
            try:
                if self._gesture_recognizer:
                    return self._gesture_recognizer.recognize(frame)
                return []
            except Exception as e:
                self._logger.warning(f"Gesture recognition failed: {e}")
                return []

        def safe_pose():
            try:
                if self._pose_estimator:
                    return self._pose_estimator.estimate(frame)
                return PoseData(detected=False)
            except Exception as e:
                self._logger.warning(f"Pose estimation failed: {e}")
                return PoseData(detected=False)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []

            if self._face_detector and not skip_face:
                futures.append(("face", executor.submit(safe_face)))
            elif skip_face and self._last_faces:
                results["face"] = self._last_faces

            if self._gesture_recognizer and not skip_gesture:
                futures.append(("gesture", executor.submit(safe_gesture)))
            elif skip_gesture and self._last_gestures:
                results["gesture"] = self._last_gestures

            if self._pose_estimator and not skip_pose:
                futures.append(("pose", executor.submit(safe_pose)))
            elif skip_pose and self._last_pose:
                results["pose"] = self._last_pose

            for key, future in futures:
                try:
                    result = future.result()
                    results[key] = result
                except Exception as e:
                    self._logger.warning(f"Model {key} failed: {e}")

        faces = results.get("face", [])
        gestures = results.get("gesture", [])
        pose = results.get("pose", PoseData(detected=False))

        if faces:
            self._last_faces = faces
        elif not faces and self._last_faces:
            faces = self._last_faces

        if gestures:
            self._last_gestures = gestures
        elif not gestures and self._last_gestures:
            gestures = self._last_gestures

        if hasattr(pose, 'detected') and pose.detected:
            self._last_pose = pose
        elif not (hasattr(pose, 'detected') and pose.detected) and self._last_pose:
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

        display_frame = cv2.resize(frame, (640, 360))
        cv2.imshow("OpenEyes Debug", display_frame)
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
        "--no-depth",
        action="store_true",
        help="Disable depth estimation",
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
    parser.add_argument(
        "--ros2",
        action="store_true",
        help="Enable ROS2 publishing (requires ros-humble-vision-msgs)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="OpenEyes v0.1.1",
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
        system = VisionSystem(config, use_ros2=args.ros2)

        if args.no_parallel:
            system._use_parallel = False
        if args.pose_every:
            system._pose_skip_frames = args.pose_every - 1
        if args.no_face:
            system._use_face = False
        if args.no_gesture:
            system._use_gesture = False
        if args.no_pose:
            system._use_pose = False
        if args.no_depth:
            system._use_depth = False

        system.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
