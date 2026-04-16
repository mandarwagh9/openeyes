import logging
import signal
import sys
import threading
import time
from queue import Queue, Empty
from typing import Optional, Any

import numpy as np
import cv2

from src.camera.camera_handler import CameraHandler
from src.camera.types import VisionResult
from src.output.json_formatter import format_vision_result
from src.output.udp_sender import UDPSender
from src.utils.config import Config
from src.utils.frame_skipper import FrameSkipProcessor, AdaptiveFrameSkipper, MultiModelFrameScheduler
from src.utils.tracker import ObjectTracker
from src.core.frame_processor import FrameProcessor
from src.core.initialization import init_all_components


ROS2_AVAILABLE = False
try:
    import rclpy
    from src.ros2.vision_node import VisionPublisher
    ROS2_AVAILABLE = True
except ImportError:
    VisionPublisher = None
    rclpy = None


class VisionSystem:
    """Optimized vision system with parallel processing and modular architecture."""

    _instance: Optional["VisionSystem"] = None

    def __init__(self, config: Config, use_ros2: bool = False, log_file: Optional[str] = None,
                 video_path: Optional[str] = None, output_path: Optional[str] = None):
        import logging
        import logging.handlers
        
        self._config = config
        self._use_ros2 = use_ros2
        self._video_path = video_path
        self._output_path = output_path
        self._logger = logging.getLogger("openeyes")
        self._logger.setLevel(logging.DEBUG if config.debug else logging.INFO)
        
        if log_file:
            handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=5*1024*1024, backupCount=3
            )
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            self._logger.addHandler(handler)
        else:
            logging.basicConfig(
                level=logging.DEBUG if config.debug else logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        self._camera: Any = None
        self._video_writer = None
        self._detector = None
        self._depth_estimator = None
        self._face_detector = None
        self._gesture_recognizer = None
        self._pose_estimator = None
        self._udp_sender: Optional[UDPSender] = None
        self._ros2_pub = None
        self._ros2_node = None
        self._running = False
        self._frame_id = 0

        self._use_parallel = True
        self._pose_skip_frames = 1
        self._use_face = True
        self._use_gesture = True
        self._use_pose = True
        self._use_depth = True

        self._perf_monitor = None
        self._use_tracking = config.tracking_enabled
        self._use_visual_odom = False
        self._visual_odom = None
        self._use_vla = False
        self._real_vla_model = None
        self._vla_model = None
        self._advanced_ai = None
        self._tracker: Optional[ObjectTracker] = None
        self._follow_target = config.follow_enabled

        self._world_model = None
        self._use_world_model = False
        self._wm_horizon = 10
        self._wm_samples = 100
        self._prediction_fps = 30
        self._occlusion_frames = 5
        self._safety_predict = False
        self._turbo_mode = False

        self._frame_scheduler: Optional[MultiModelFrameScheduler] = None
        self._adaptive_skipper: Optional[AdaptiveFrameSkipper] = None
        self._frame_processor: Optional[FrameProcessor] = None

        self._ros2_queue: Optional[Queue] = None
        self._ros2_thread: Optional[threading.Thread] = None
        self._ros2_executor = None

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        VisionSystem._instance = self

    def _signal_handler(self, signum, frame):
        self._logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)

    def start(self) -> None:
        self._logger.info("Starting OpenEyes Vision System (Optimized)")
        self._logger.info(f"Parallel processing: {self._use_parallel}")
        self._logger.info(f"Pose skip frames: {self._pose_skip_frames + 1}")

        self._init_components()
        self._running = True
        self._logger.info("Vision system started successfully")

        self._process_loop()

    def stop(self) -> None:
        self._running = False

        if self._camera:
            self._camera.release()

        if self._video_writer is not None:
            self._video_writer.release()
            self._logger.info(f"Output video saved")

        if self._udp_sender:
            self._udp_sender.close()

        if self._frame_processor:
            self._frame_processor.shutdown()

        self._logger.info("Vision system stopped")

    def _init_components(self) -> None:
        """Initialize all components."""
        enabled_models = []
        disabled_models = []
        
        if self._use_depth:
            enabled_models.append("depth")
        else:
            disabled_models.append("depth")
        if self._use_face:
            enabled_models.append("face")
        else:
            disabled_models.append("face")
        if self._use_gesture:
            enabled_models.append("gesture")
        else:
            disabled_models.append("gesture")
        if self._use_pose:
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

    def _init_frame_scheduler(self) -> None:
        skip_intervals = {
            'detector': 1,
            'depth': 8,
            'face': 6,
            'gesture': 6,
            'pose': 6
        }
        if getattr(self, '_turbo_mode', False):
            skip_intervals = {
                'detector': 1,
                'depth': 16,
                'face': 12,
                'gesture': 12,
                'pose': 12
            }
            self._logger.info("TURBO MODE: Aggressive frame skipping enabled")
        self._frame_scheduler = MultiModelFrameScheduler(skip_intervals, turbo=getattr(self, '_turbo_mode', False))
        self._adaptive_skipper = AdaptiveFrameSkipper(
            base_skip=2,
            motion_threshold=5000.0,
            min_skip=1,
            max_skip=4
        )
        self._logger.info(f"Frame scheduler initialized: {skip_intervals}")

    def _init_camera(self) -> None:
        if self._video_path:
            from src.camera.video_source import VideoSource
            self._camera = VideoSource(
                path=self._video_path,
                width=self._config.camera_width,
                height=self._config.camera_height,
                fps=self._config.camera_fps,
            )
            self._logger.info(f"Using video source: {self._video_path}")
        else:
            self._camera = CameraHandler(
                source=self._config.camera_source,
                width=self._config.camera_width,
                height=self._config.camera_height,
                fps=self._config.camera_fps,
            )
        try:
            self._camera.open()
        except Exception as e:
            self._logger.error(f"Camera initialization failed: {e}")
            raise

    def _init_detectors(self) -> None:
        from src.models.object_detector import ObjectDetector
        from src.models.depth_estimator import DepthEstimator
        from src.models.face_detector import FaceDetector
        from src.models.gesture_recognizer import GestureRecognizer
        from src.models.pose_estimator import PoseEstimator
        from src.exceptions import ModelError

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
                self._logger.info("Face Detector loaded")
            except ModelError as e:
                self._logger.warning(f"Face Detector not available: {e}")

        if self._use_gesture:
            try:
                self._gesture_recognizer = GestureRecognizer(
                    min_confidence=self._config.gesture_confidence
                )
                self._gesture_recognizer.load()
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
            self._logger.warning("ROS2 requested but not available")

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
        is_video = self._video_path is not None

        if self._output_path:
            import cv2
            out_w = self._camera.width
            out_h = self._camera.height
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                self._output_path, fourcc, self._config.camera_fps, (out_w, out_h)
            )
            if not self._video_writer.isOpened():
                self._logger.warning(f"Failed to open video writer for {self._output_path}")
                self._video_writer = None
            else:
                self._logger.info(f"Recording output video to {self._output_path}")

        while self._running:
            loop_start = time.time()

            frame = self._camera.read()
            if frame is None:
                if is_video:
                    self._logger.info("Video playback complete")
                    break
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
                except Exception:
                    pass

            if self._config.debug or is_video:
                self._debug_display(frame, result)

            if self._video_writer is not None:
                self._video_writer.write(frame)

            self._fps_counter += 1
            elapsed_total = time.time() - self._fps_start_time
            if elapsed_total >= 1.0:
                fps = self._fps_counter / elapsed_total
                stats = self._perf_monitor.get_stats() if self._perf_monitor else {}
                progress = ""
                if is_video and hasattr(self._camera, 'progress'):
                    progress = f" | Progress: {self._camera.progress*100:.1f}%"
                self._logger.info(
                    f"FPS: {fps:.1f} | Objects: {len(result.objects)} | "
                    f"Faces: {len(result.faces)} | Gestures: {len(result.gestures)}{progress}"
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

    _fps_counter = 0
    _fps_start_time = time.time()

    def _process_frame(self, frame) -> VisionResult:
        """Process a single frame using FrameProcessor."""
        if self._frame_processor is None:
            self._frame_processor = FrameProcessor(
                camera=self._camera,
                detector=self._detector,
                depth_estimator=self._depth_estimator,
                face_detector=self._face_detector,
                gesture_recognizer=self._gesture_recognizer,
                pose_estimator=self._pose_estimator,
                tracker=self._tracker,
                perf_monitor=self._perf_monitor,
                use_parallel=self._use_parallel,
                use_face=self._use_face,
                use_gesture=self._use_gesture,
                use_pose=self._use_pose,
                use_depth=self._use_depth,
                use_tracking=self._use_tracking,
                use_vla=self._use_vla,
                vla_model=self._vla_model,
                real_vla_model=self._real_vla_model,
                advanced_ai=self._advanced_ai,
                frame_scheduler=self._frame_scheduler,
                adaptive_skipper=self._adaptive_skipper,
                logger=self._logger,
                world_model=self._world_model,
                use_world_model=self._use_world_model,
                world_model_horizon=self._wm_horizon,
                world_model_samples=self._wm_samples,
                prediction_fps=self._prediction_fps,
                occlusion_frames=self._occlusion_frames,
                safety_predict=self._safety_predict,
            )
            self._frame_processor.set_follow_target(self._follow_target)
            self._frame_processor.set_pose_skip_frames(self._pose_skip_frames)
            self._frame_processor.frame_id = self._frame_id

        result = self._frame_processor.process_frame(frame)
        self._frame_id = self._frame_processor.frame_id + 1
        self._frame_processor.frame_id = self._frame_id
        
        return result

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

    def _debug_display(self, frame, result: VisionResult) -> None:
        """Display debug information on frame."""
        import cv2
        
        h, w = frame.shape[:2]
        box_color = (0, 255, 0)
        text_color = (0, 255, 0)

        fps = 0.0
        if hasattr(self, '_fps_counter') and hasattr(self, '_fps_start_time'):
            elapsed = time.time() - self._fps_start_time
            if elapsed > 0:
                fps = self._fps_counter / elapsed

        overlay_y = 35
        line_height = 28

        stats_lines = [
            f"FPS: {fps:.1f}",
            f"Objects: {len(result.objects)}",
            f"Faces: {len(result.faces)}",
            f"Gestures: {len(result.gestures)}",
        ]

        for line in stats_lines:
            (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (5, overlay_y - text_h - 5), (text_w + 20, overlay_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, line, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            overlay_y += line_height

        if result.depth and result.depth.enabled and result.depth.depth_map is not None:
            depth_map = result.depth.depth_map
            if depth_map is not None and depth_map.size > 0:
                dh, dw = depth_map.shape[:2]
                if dh != h or dw != w:
                    depth_map = cv2.resize(depth_map, (w, h))
                    dh, dw = h, w
                depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_normalized = depth_normalized.astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                fw = w // 4
                fh = int(dh * (fw / dw))
                depth_resized = cv2.resize(depth_colored, (fw, fh))
                depth_bg = np.zeros((fh + 30, fw, 3), dtype=np.uint8)
                depth_bg[0:fh, 0:fw] = depth_resized
                cv2.putText(depth_bg, "DEPTH", (10, fh + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                target_h = min(fh + 40, h)
                target_fh = target_h - 40
                depth_bg_resized = cv2.resize(depth_bg, (fw, target_h))
                frame[h - target_h:h, 10:10 + fw] = depth_bg_resized

        for det in result.objects:
            bbox = det.bbox
            cv2.rectangle(frame, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), box_color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (int(bbox.x1), int(bbox.y1) - text_h - 10), (int(bbox.x1) + text_w + 5, int(bbox.y1)), (0, 0, 0), -1)
            cv2.putText(frame, label, (int(bbox.x1), int(bbox.y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        for face in result.faces:
            bbox = face.bbox
            cv2.rectangle(frame, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), box_color, 2)

        for track in result.tracks:
            bbox = track.bbox
            is_predicted = getattr(track, 'is_predicted', False)
            track_color = (255, 165, 0) if is_predicted else box_color
            cv2.rectangle(frame, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), track_color, 2)
            label = f"ID:{track.track_id}"
            if is_predicted:
                label += " [PRED]"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (int(bbox.x1), int(bbox.y1) - text_h - 10), (int(bbox.x1) + text_w + 5, int(bbox.y1)), (0, 0, 0), -1)
            cv2.putText(frame, label, (int(bbox.x1), int(bbox.y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 2)

        if result.predictions:
            colors = [
                (255, 0, 255),
                (255, 255, 0),
                (0, 255, 255),
                (128, 0, 255),
                (0, 128, 255),
            ]
            for track_idx, future_bboxes in enumerate(result.predictions):
                color = colors[track_idx % len(colors)]
                for step_idx, future_bbox in enumerate(future_bboxes):
                    alpha = 1.0 - (step_idx * 0.15)
                    thickness = max(1, 3 - step_idx)
                    x1 = int(future_bbox.x1)
                    y1 = int(future_bbox.y1)
                    x2 = int(future_bbox.x2)
                    y2 = int(future_bbox.y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    label = f"+{step_idx+1}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.circle(frame, (cx, cy), 3, color, -1)
                    cv2.putText(frame, label, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if result.predictions or self._use_world_model:
            wm_y = h - 35
            wm_lines = [f"WM: PREDICTIONS={len(result.predictions)}"]
            if self._world_model and hasattr(self._world_model, 'get_info'):
                try:
                    info = self._world_model.get_info()
                    wm_lines.append(f"  LATENT={info.get('latent_dim', '?')}")
                    wm_lines.append(f"  PLAN_MS={info.get('planning_time_ms', 0):.1f}")
                except Exception:
                    pass
            for line in wm_lines:
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (5, wm_y - th - 5), (tw + 15, wm_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, line, (10, wm_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                wm_y -= 22

        gesture_y = h - 35
        for gesture in result.gestures:
            label = f"{gesture.gesture_type}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (5, gesture_y - text_h - 5), (text_w + 15, gesture_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, gesture_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            gesture_y -= 30

        if result.pose and result.pose.detected and result.pose.landmarks:
            for lm in result.pose.landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("OpenEyes Debug", display_frame)
        cv2.waitKey(1)
