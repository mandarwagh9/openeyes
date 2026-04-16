"""DeepStream Pipeline for OpenEyes Vision System.

Provides high-performance inference using NVIDIA DeepStream SDK.
Optimized for Jetson Orin Nano with CSI camera input.

Usage:
    pipeline = DeepStreamPipeline(model="yolo11n", camera=0)
    pipeline.run()
"""

import sys
import os
import time
import json
from typing import Optional, Callable, List, Dict, Any

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np

from src.utils.logger import get_logger

def _check_ros2():
    """Check if ROS2 is available and properly configured."""
    try:
        import sys
        for path in [
            "/opt/ros/humble/local/lib/python3.10/dist-packages",
            "/opt/ros/humble/lib/python3.10/site-packages",
        ]:
            if path not in sys.path:
                sys.path.insert(0, path)
        import rclpy
        return True
    except Exception:
        return False

PYDS_AVAILABLE = False
pyds = None
try:
    import pyds
    PYDS_AVAILABLE = True
except ImportError:
    pass

ROS2_AVAILABLE = _check_ros2()

Gst.init(None)

logger = get_logger(__name__)


class DetectionResult:
    """Detection result from DeepStream inference."""
    
    def __init__(self, class_id: int, class_name: str, confidence: float,
                 bbox_left: float, bbox_top: float, 
                 bbox_width: float, bbox_height: float):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox_left = bbox_left
        self.bbox_top = bbox_top
        self.bbox_width = bbox_width
        self.bbox_height = bbox_height
    
    def __repr__(self):
        return (f"DetectionResult(class={self.class_name}, "
                f"conf={self.confidence:.2f})")


class DeepStreamPipeline:
    """DeepStream pipeline for real-time inference on Jetson.
    
    Pipeline: CSI Camera → Hardware Decode → TensorRT Inference → OSD → Display
    
    Expected performance: 40-70 FPS on Jetson Orin Nano
    """
    
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(
        self,
        model: str = "yolo11n",
        camera: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        display: bool = True,
        display_fps: bool = True,
        enable_face: bool = False,
        enable_gesture: bool = False,
        enable_pose: bool = False,
        enable_depth: bool = False,
    ):
        self.model = model
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps
        self.display = display
        self.display_fps = display_fps
        self.enable_face = enable_face
        self.enable_gesture = enable_gesture
        self.enable_pose = enable_pose
        self.enable_depth = enable_depth
        
        self.pipeline = None
        self.loop = None
        self.running = False
        self._callbacks: List[Callable[[List[DetectionResult], float], None]] = []
        self._detections: List[DetectionResult] = []
        self._last_fps_time = time.time()
        self._frame_count = 0
        self._current_fps = 0.0
    
    def set_detection_callback(self, callback: Callable[[List[DetectionResult], float], None]):
        """Set callback for detection results."""
        self._callbacks.append(callback)
    
    def _get_config_path(self) -> str:
        """Get path to model config file."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Use yolov10n which has working engine
        self.model = "yolov10n"
        
        config_dir = os.path.join(base_dir, "..", "deepstream")
        config_path = os.path.join(config_dir, f"config_{self.model}.txt")
        
        config_abs = os.path.abspath(config_path)
        logger.info(f"Using config: {config_abs}")
        
        return config_abs
    
    def _get_all_configs(self) -> List[str]:
        """Get list of config paths for multi-model pipeline."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(base_dir, "..", "deepstream")
        
        configs = []
        
        # Always include YOLO detection (gie-unique-id=1)
        configs.append(("yolov10n", os.path.join(config_dir, "config_yolov10n.txt"), 1))
        
        # Optional secondary models (gie-unique-id >= 2)
        # Note: enable_face uses Python face detector, not DeepStream
        # DeepStream face needs custom output parser for landmarks
        if self.enable_gesture:
            configs.append(("gesture", os.path.join(config_dir, "config_gesture.txt"), 3))
        if self.enable_pose:
            configs.append(("pose", os.path.join(config_dir, "config_pose.txt"), 4))
        if self.enable_depth:
            configs.append(("depth", os.path.join(config_dir, "config_depth.txt"), 5))
        
        return configs
    
    def _create_pipeline_string(self) -> str:
        """Create pipeline string for multi-model DeepStream pipeline."""
        configs = self._get_all_configs()
        
        # Build pipeline with multiple nvinfer instances
        pipeline_parts = [
            f"nvarguscamerasrc sensor-id={self.camera} ! ",
            f"video/x-raw(memory:NVMM),format=NV12,width={self.width},height={self.height},framerate={self.fps}/1 ! ",
            "nvvidconv ! ",
            "video/x-raw(memory:NVMM),format=NV12 ! ",
            f"m.sink_0 nvstreammux name=m batch-size=1 width={self.width} height={self.height} live-source=1 ! ",
        ]
        
        # Add nvinfer for each model
        for i, (model_name, config_path, gie_id) in enumerate(configs):
            config_abs = os.path.abspath(config_path)
            pipeline_parts.append(f'nvinfer name=nvinfer{gie_id} config-file-path={config_abs} ! ')
        
        # OSD and display
        pipeline_parts.extend([
            "nvdsosd name=nvdsosd display-text=1 display-bbox=1 display-mask=1 ! ",
            "nvvidconv ! ",
            "video/x-raw,format=RGBA ! ",
            'textoverlay name=osdfps text="FPS: 0 | Obj: 0" valignment=top halignment=right ! ',
            "queue ! nv3dsink sync=0",
        ])
        
        # Log enabled models
        enabled = [name for name, _, _ in configs]
        logger.info(f"Enabled models: {enabled}")
        logger.info(f"Resolution: {self.width}x{self.height} @ {self.fps} FPS")
        
        return "".join(pipeline_parts)
    
    def create_pipeline(self) -> Gst.Pipeline:
        """Create the DeepStream pipeline."""
        pipeline_str = self._create_pipeline_string()
        logger.info(f"Creating DeepStream pipeline...")
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            raise
        
        if not self.pipeline:
            raise RuntimeError("Failed to create DeepStream pipeline")
        
        self._setup_bus()
        
        return self.pipeline
    
    def _setup_bus(self):
        """Setup bus message handler."""
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
    
    def _on_bus_message(self, bus, msg):
        """Handle bus messages."""
        msg_type = msg.type
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            logger.error(f"Pipeline error: {err}")
            self.stop()
            
        elif msg_type == Gst.MessageType.WARNING:
            warn, _ = msg.parse_warning()
            logger.warning(f"Pipeline warning: {warn}")
            
        elif msg_type == Gst.MessageType.EOS:
            logger.info("End of stream")
            self.stop()
            
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if msg.src == self.pipeline:
                old, new, _ = msg.parse_state_changed()
                if old == Gst.State.READY and new == Gst.State.PAUSED:
                    logger.info("Pipeline paused - starting inference")
    
    def _parse_detections(self, frame_meta) -> List[DetectionResult]:
        """Parse detection metadata from frame."""
        detections = []
        
        if not PYDS_AVAILABLE:
            return detections
        
        try:
            obj_meta = frame_meta.obj_meta_list
            while obj_meta is not None:
                try:
                    obj = pyds.NvDsObjectMeta.cast(obj_meta.data)
                    
                    class_id = int(obj.class_id)
                    class_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else f"class_{class_id}"
                    confidence = float(obj.confidence)
                    
                    detection = DetectionResult(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox_left=obj.rect_params.left,
                        bbox_top=obj.rect_params.top,
                        bbox_width=obj.rect_params.width,
                        bbox_height=obj.rect_params.height,
                    )
                    detections.append(detection)
                    
                except StopIteration:
                    break
                finally:
                    obj_meta = obj_meta.next
                    
        except Exception as e:
            logger.warning(f"Failed to parse detections: {e}")
            
        return detections
    
    def _osd_sink_pad_buffer_probe(self, pad, info):
        """Probe to extract inference results."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        if not PYDS_AVAILABLE:
            self._calculate_fps()
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        
        self._calculate_fps()
        
        try:
            frame_meta = batch_meta.frame_meta_list
            while frame_meta is not None:
                try:
                    frame = pyds.NvDsFrameMeta.cast(frame_meta.data)
                    detections = self._parse_detections(frame)
                    self._detections = detections
                    
                    for callback in self._callbacks:
                        callback(detections, self._current_fps)
                        
                except StopIteration:
                    break
                finally:
                    frame_meta = frame_meta.next
                    
        except Exception as e:
            logger.debug(f"Probe error: {e}")
            
        return Gst.PadProbeReturn.OK
    
    def _calculate_fps(self):
        """Calculate current FPS."""
        self._frame_count += 1
        current_time = time.time()
        elapsed = current_time - self._last_fps_time
        
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = current_time
    
    def _frame_probe(self, pad, info):
        """Probe to count frames for FPS calculation."""
        self._frame_count += 1
        return Gst.PadProbeReturn.OK
    
    def _osd_probe(self, pad, info):
        """Count frames for FPS calculation."""
        self._frame_count += 1
        return Gst.PadProbeReturn.OK
    
    def _update_osd_text(self):
        """Update the on-screen FPS display."""
        if self.running:
            try:
                osd = self.pipeline.get_by_name("osdfps")
                if osd:
                    det_count = len(self._detections) if self._detections else 0
                    det_names = ", ".join([d.class_name for d in self._detections[:3]]) if self._detections else ""
                    text = f"FPS: {self._current_fps:.0f} | Obj: {det_count}"
                    if det_names:
                        text += f" | {det_names}"
                    osd.set_property("text", text)
            except Exception as e:
                logger.debug(f"OSD text update: {e}")
        return True
    
    def run(self):
        """Run the pipeline."""
        self.create_pipeline()
        
        self.pipeline.set_state(Gst.State.PLAYING)
        
        time.sleep(1)
        
        osd = self.pipeline.get_by_name("nvdsosd")
        if osd:
            try:
                src_pad = osd.get_static_pad("src")
                if src_pad:
                    src_pad.add_probe(Gst.PadProbeType.BUFFER, self._osd_probe)
                    logger.info("Added OSD probe on src pad")
            except Exception as e:
                logger.warning(f"Probe error: {e}")
        
        self.running = True
        self.loop = GLib.MainLoop()
        
        GLib.timeout_add(500, self._update_osd_text)
        GLib.timeout_add(1000, self._print_fps_timer)
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def _print_fps_timer(self):
        """Print FPS and detection info every second."""
        if self.running:
            current_time = time.time()
            elapsed = current_time - self._last_fps_time
            
            if elapsed >= 1.0:
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = current_time
            
            det_count = len(self._detections) if self._detections else 0
            det_names = [d.class_name for d in self._detections[:3]] if self._detections else []
            
            logger.info(f"FPS: {self._current_fps:.1f} | Objects: {det_count} | Detected: {', '.join(det_names) if det_names else 'scanning...'}")
        return True
    
    def _osd_probe(self, pad, info):
        """Probe to get detection results for metadata extraction."""
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK
        
        self._frame_count += 1
        
        if not PYDS_AVAILABLE:
            return Gst.PadProbeReturn.OK
        
        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            if batch_meta:
                frame_meta = batch_meta.frame_meta_list
                while frame_meta:
                    try:
                        frame = pyds.NvDsFrameMeta.cast(frame_meta.data)
                        detections = self._parse_detections(frame)
                        self._detections = detections
                        
                        for callback in self._callbacks:
                            callback(detections, self._current_fps)
                            
                    except StopIteration:
                        break
                    finally:
                        frame_meta = frame_meta.next
                        
        except Exception as e:
            logger.debug(f"OSD probe error: {e}")
        
        return Gst.PadProbeReturn.OK
    
    def _on_new_sample(self, appsink):
        """Handle new sample from appsink - get frame for Python processing."""
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK
        
        try:
            # Calculate FPS
            self._frame_count += 1
            current_time = time.time()
            elapsed = current_time - self._last_fps_time
            if elapsed >= 1.0:
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = current_time
            
            # Call callbacks with empty detections (metadata extraction needs Jetson)
            # On Jetson: use osd_probe for metadata; appsink is for frame data
            for callback in self._callbacks:
                callback(self._detections, self._current_fps)
            
            # Log FPS periodically
            if int(current_time) % 3 == 0:
                logger.info(f"FPS: {self._current_fps:.1f} | Objects: {len(self._detections)}")
        
        except Exception as e:
            logger.debug(f"Appsink sample error: {e}")
        
        return Gst.FlowReturn.OK
    
    def _on_appsink_frame(self, appsink):
        """Handle new frame from appsink for Python detection."""
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK
        
        try:
            # Get buffer
            buf = sample.get_buffer()
            if not buf:
                return Gst.FlowReturn.OK
            
            # Map buffer and get data
            success, buf_map = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.OK
            
            # Get frame data as numpy array
            frame_data = np.ndarray(
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
                buffer=buf_map.data
            )
            
            # Unmap buffer
            buf.unmap(buf_map)
            
            # Update FPS
            self._calculate_fps()
            
            # Call registered callbacks for detection
            for callback in self._callbacks:
                callback(self._detections, self._current_fps)
        
        except Exception as e:
            logger.debug(f"Appsink frame error: {e}")
        
        return Gst.FlowReturn.OK
    
    def _on_appsink_sample(self, appsink):
        """Handle new sample from appsink."""
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK
        
        buf = sample.get_buffer()
        if not buf:
            return Gst.FlowReturn.OK
        
        self._calculate_fps()
        
        if not PYDS_AVAILABLE:
            return Gst.FlowReturn.OK
        
        try:
            batch_meta = pyds.NvDsBatchMeta.cast(buf)
            if batch_meta:
                frame_meta = batch_meta.frame_meta_list
                while frame_meta is not None:
                    try:
                        frame = pyds.NvDsFrameMeta.cast(frame_meta.data)
                        detections = self._parse_detections(frame)
                        self._detections = detections
                        
                        for callback in self._callbacks:
                            callback(detections, self._current_fps)
                            
                    except StopIteration:
                        break
                    finally:
                        frame_meta = frame_meta.next
                        
        except Exception as e:
            logger.debug(f"Probe error: {e}")
        
        return Gst.FlowReturn.OK
    
    def stop(self):
        """Stop the pipeline."""
        self.running = False
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            logger.info("Pipeline stopped")
        
        if self.loop:
            try:
                self.loop.quit()
            except:
                pass
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self._current_fps
    
    def get_last_detections(self) -> List[DetectionResult]:
        """Get last frame detections."""
        return self._detections


class DeepStreamMultiCameraPipeline(DeepStreamPipeline):
    """Multi-camera DeepStream pipeline."""
    
    def __init__(self, cameras: List[int], model: str = "yolo11n",
                 width: int = 640, height: int = 480, fps: int = 30):
        super().__init__(model=model, camera=0, width=width, height=height)
        self.cameras = cameras
    
    def _create_pipeline_string(self) -> str:
        """Create pipeline string for multi-camera."""
        config_path = self._get_config_path()
        
        sources = []
        for i, cam_id in enumerate(self.cameras):
            source_str = (
                f"nvarguscamerasrc sensor-id={cam_id} ! "
                f"video/x-raw(memory:NVMM),width={self.width},height={self.height},"
                f"format=NV12,framerate={self.fps}/1 ! "
                f"nvvidconv ! "
                f"video/x-raw ! "
                f"queue name=cam{i}_queue ! "
                f"muxer.sink_{i}"
            )
            sources.append(source_str)
        
        muxer_str = f"nvstreammux name=muxer batch-size={len(self.cameras)} width={self.width} height={self.height}"
        
        if self.display:
            sink = "nv3dsink sync=0"
        else:
            sink = "fakesink sync=0"
        
        pipeline = (
            sources[0] + " " +
            sources[1] + " " +
            muxer_str + " ! " +
            f"queue ! nvinfer config-file-path={config_path} ! " +
            f"queue ! nvdsosd display-text=1 ! " +
            f"queue ! nvvidconv ! {sink}"
        )
        
        return pipeline


def run_deepstream(
    model: str = "yolo11n",
    camera: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    display: bool = True,
    use_face: bool = True,
    use_gesture: bool = True,
    use_pose: bool = True,
    use_depth: bool = False,
    use_tracking: bool = True,
    use_ros2: bool = False,
    use_udp: bool = True,
    output_host: str = "127.0.0.1",
    output_port: int = 5000,
    video_path: Optional[str] = None,
    output_path: Optional[str] = None,
    debug: bool = False,
) -> DeepStreamPipeline:
    """Run DeepStream pipeline with full Python model integration.
    
    This integrates the old OpenCV pipeline models (face, gesture, pose, tracking)
    with DeepStream's hardware-accelerated YOLO inference.
    """
    # Initialize Python models (same as old pipeline)
    _detector = None
    _face_detector = None
    _gesture_recognizer = None
    _pose_estimator = None
    _tracker = None
    _udp_sender = None
    _ros2_pub = None
    
    # Use Python YOLO for detection results (we need this for metadata)
    # The DeepStream nvinfer handles display, but Python detector gives us metadata
    try:
        from src.models.object_detector import ObjectDetector
        _detector = ObjectDetector(
            model_path="models/yolov10n.onnx",
            confidence=0.25,
            iou_threshold=0.45,
        )
        _detector.load()
        logger.info("Python Object Detector loaded for metadata")
    except Exception as e:
        logger.warning(f"Object Detector not available: {e}")
    
    # Initialize face detector
    if use_face:
        try:
            from src.models.face_detector import FaceDetector
            _face_detector = FaceDetector()
            _face_detector.load()
            logger.info("Face Detector loaded")
        except Exception as e:
            logger.warning(f"Face Detector not available: {e}")
    
    # Initialize gesture recognizer
    if use_gesture:
        try:
            from src.models.gesture_recognizer import GestureRecognizer
            _gesture_recognizer = GestureRecognizer(min_confidence=0.3)
            _gesture_recognizer.load()
            logger.info("Gesture Recognizer loaded")
        except Exception as e:
            logger.warning(f"Gesture Recognizer not available: {e}")
    
    # Initialize pose estimator
    if use_pose:
        try:
            from src.models.pose_estimator import PoseEstimator
            _pose_estimator = PoseEstimator()
            _pose_estimator.load()
            logger.info("Pose Estimator loaded")
        except Exception as e:
            logger.warning(f"Pose Estimator not available: {e}")
    
    # Initialize object tracker
    if use_tracking:
        try:
            from src.utils.tracker import ObjectTracker
            _tracker = ObjectTracker()
            logger.info("Object Tracker initialized")
        except Exception as e:
            logger.warning(f"Object Tracker not available: {e}")
    
    # Initialize UDP sender
    if use_udp:
        try:
            from src.output.udp_sender import UDPSender
            _udp_sender = UDPSender(host=output_host, port=output_port)
            _udp_sender.open()
            logger.info(f"UDP sender initialized: {output_host}:{output_port}")
        except Exception as e:
            logger.warning(f"UDP sender not available: {e}")
    
    _ros2_pub = None
    if use_ros2 and ROS2_AVAILABLE:
        try:
            from src.ros2.vision_node import VisionPublisher
            import rclpy
            if not rclpy.ok():
                rclpy.init()
            _ros2_pub = VisionPublisher(
                detections_topic="/vision/detections",
                depth_topic="/vision/depth",
                faces_topic="/vision/faces",
                gestures_topic="/vision/gestures",
                poses_topic="/vision/poses",
                cmd_topic="/vision/cmd",
                status_topic="/vision/status",
                frame_id="camera_link",
                confidence_threshold=0.25,
                max_depth_range=5.0,
            )
            logger.info("ROS2 Vision Publisher initialized")
        except Exception as e:
            logger.warning(f"ROS2 not available: {e}")
    
    # Create pipeline with multi-model support
    pipeline = DeepStreamPipeline(
        model=model,
        camera=camera,
        width=width,
        height=height,
        fps=fps,
        display=display,
        enable_face=use_face,
        enable_gesture=use_gesture,
        enable_pose=use_pose,
        enable_depth=use_depth,
    )
    
    logger.info(f"Multi-model config: face={use_face}, gesture={use_gesture}, pose={use_pose}, depth={use_depth}")
    
    _frame_count = 0
    _last_udp_time = time.time()
    
    def process_frame(detections, current_fps):
        """Process detections and send to UDP."""
        nonlocal _frame_count, _last_udp_time
        _frame_count += 1
        current_time = time.time()
        
        if _udp_sender and (current_time - _last_udp_time) >= 1.0:
            _last_udp_time = current_time
            
            detections_list = []
            if detections:
                for det in detections:
                    detections_list.append({
                        "class_name": det.class_name,
                        "confidence": round(det.confidence, 2),
                        "bbox": [det.bbox_left, det.bbox_top, det.bbox_width, det.bbox_height]
                    })
            
            result = {
                "objects": detections_list,
                "faces": [],  # To add face detection, integrate FaceDetector in probe
                "gestures": [],
                "poses": [],
                "fps": round(current_fps, 1),
                "timestamp": current_time
            }
            
            try:
                _udp_sender.send(json.dumps(result))
                logger.info(f"UDP: {len(detections_list)} objects -> {output_host}:{output_port}")
            except Exception as e:
                logger.warning(f"UDP error: {e}")
            
            if _ros2_pub:
                try:
                    ros2_detections = []
                    for det in detections_list:
                        ros2_detections.append({
                            "bbox": det["bbox"],
                            "class_name": det["class_name"],
                            "confidence": det["confidence"]
                        })
                    _ros2_pub.publish_detections(ros2_detections, (width, height))
                    _ros2_pub.publish_status(current_fps, len(detections_list), 0, 0)
                except Exception as e:
                    logger.warning(f"ROS2 error: {e}")
    
    pipeline.set_detection_callback(process_frame)
    
    return pipeline


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenEyes DeepStream Pipeline")
    parser.add_argument("--model", default="yolo11n", help="Model name")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    
    args = parser.parse_args()
    
    pipeline = run_deepstream(
        model=args.model,
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        display=not args.no_display,
    )
    
    pipeline.run()
