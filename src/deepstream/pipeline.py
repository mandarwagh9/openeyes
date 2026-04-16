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
import cv2
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


class FaceResult:
    """Face detection result."""
    
    def __init__(self, x: int, y: int, w: int, h: int, landmarks: list = None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.landmarks = landmarks or []
    
    def __repr__(self):
        return f"Face(bbox=({self.x},{self.y},{self.w},{self.h}))"


class GestureResult:
    """Gesture detection result."""
    
    def __init__(self, gesture_type: str, handedness: str, confidence: float):
        self.gesture_type = gesture_type
        self.handedness = handedness
        self.confidence = confidence
    
    def __repr__(self):
        return f"Gesture({self.gesture_type}, {self.handedness}, conf={self.confidence:.2f})"


class PoseResult:
    """Pose detection result."""
    
    def __init__(self, keypoints: dict, confidence: float):
        self.keypoints = keypoints
        self.confidence = confidence
    
    def __repr__(self):
        return f"Pose(keypoints={len(self.keypoints)}, conf={self.confidence:.2f})"


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
        width: int = 1280,
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
        self._face_callbacks: List[Callable[[List[FaceResult], float], None]] = []
        self._detections: List[DetectionResult] = []
        self._faces: List[FaceResult] = []
        self._gestures: List[Any] = []
        self._poses: List[Any] = []
        self._last_fps_time = time.time()
        self._face_detector = None
        self._gesture_recognizer = None
        self._pose_estimator = None
        self._face_count = 0  # Counter for face detection interval
        
        # Load face detector if enabled
        if enable_face:
            try:
                from src.models.face_detector import FaceDetector
                self._face_detector = FaceDetector()
                self._face_detector.load()
                logger.info("Face detector loaded in DeepStream pipeline")
            except Exception as e:
                logger.warning(f"Face detector not available: {e}")
        
        # Load gesture recognizer if enabled
        if enable_gesture:
            try:
                from src.models.gesture_recognizer import GestureRecognizer
                self._gesture_recognizer = GestureRecognizer(min_confidence=0.3)
                self._gesture_recognizer.load()
                logger.info("Gesture recognizer loaded in DeepStream pipeline")
            except Exception as e:
                logger.warning(f"Gesture recognizer not available: {e}")
        
        # Load pose estimator if enabled
        if enable_pose:
            try:
                from src.models.pose_estimator import PoseEstimator
                self._pose_estimator = PoseEstimator()
                self._pose_estimator.load()
                logger.info("Pose estimator loaded in DeepStream pipeline")
            except Exception as e:
                logger.warning(f"Pose estimator not available: {e}")
        self._frame_count = 0
        self._current_fps = 0.0
    
    def set_detection_callback(self, callback: Callable[[List[DetectionResult], float], None]):
        """Set callback for detection results."""
        self._callbacks.append(callback)
    
    def set_face_callback(self, callback: Callable[[List[FaceResult], float], None]):
        """Set callback for face detection results."""
        self._face_callbacks.append(callback)
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceResult]:
        """Detect faces in frame using MediaPipe."""
        if not self._face_detector:
            return []
        
        try:
            faces = self._face_detector.detect(frame)
            return [FaceResult(
                x=int(face.bbox.x),
                y=int(face.bbox.y),
                w=int(face.bbox.width),
                h=int(face.bbox.height),
            ) for face in faces]
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
            return []
    
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
        
        # Only include YOLO detection (gie-unique-id=1)
        # Note: Face, gesture, pose use Python models via appsink, not DeepStream nvinfer
        configs.append(("yolov10n", os.path.join(config_dir, "config_yolov10n.txt"), 1))
        
        return configs
    
    def _create_pipeline_string(self) -> str:
        """Create pipeline string for multi-model DeepStream pipeline."""
        configs = self._get_all_configs()
        
        # Check if we need appsink for Python model processing
        need_appsink = (self.enable_face or self.enable_gesture or 
                       self.enable_pose or self.enable_depth)
        
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
            "nvdsosd name=nvdsosd display-text=1 display-bbox=1 ! ",
            "nvvidconv ! ",
            "video/x-raw,format=RGBA ! ",
        ])
        
        # Add OSD fps text element
        
        # Display and appsink for Python processing
        if need_appsink:
            pipeline_parts.extend([
                "tee name=t ! ",
                "queue ! nv3dsink sync=0 t. ! ",
                "queue ! appsink name=python-appsink emit-signals=true",
            ])
        else:
            pipeline_parts.append("queue ! nv3dsink sync=0")
        
        # Log enabled models
        enabled = [name for name, _, _ in configs]
        if need_appsink:
            enabled.append("python-models")
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
        
        # Setup appsink for Python models if enabled
        need_appsink = (self.enable_face or self.enable_gesture or 
                       self.enable_pose or self.enable_depth)
        
        if need_appsink and self.pipeline:
            appsink = self.pipeline.get_by_name("python-appsink")
            if appsink:
                appsink.connect("new-sample", self._on_appsink_sample)
                logger.info("Appsink connected for Python models")
        
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
        """Update terminal with current stats."""
        if self.running:
            self._print_fps_timer()
        return True
    
    def run(self):
        """Run the pipeline."""
        self.create_pipeline()
        
        self.pipeline.set_state(Gst.State.PLAYING)
        
        time.sleep(1)
        
        osd = self.pipeline.get_by_name("nvdsosd")
        if osd:
            try:
                sink_pad = osd.get_static_pad("sink")
                if sink_pad:
                    sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._osd_probe)
            except Exception as e:
                logger.debug(f"OSD sink probe error: {e}")
        
        self.running = True
        self.loop = GLib.MainLoop()
        
        GLib.timeout_add(1000, self._update_osd_text)
        
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
            face_count = len(self._faces) if self._faces else 0
            gest_count = len(self._gestures) if self._gestures else 0
            pose_count = len(self._poses) if self._poses else 0
            
            text = f"FPS: {self._current_fps:.0f} | Obj: {det_count}"
            if face_count > 0:
                text += f" | Face: {face_count}"
            if gest_count > 0:
                gest_names = ", ".join([g.gesture_type for g in self._gestures])
                text += f" | Hand: {gest_names}"
            if pose_count > 0:
                text += f" | Pose: {pose_count}"
            
            logger.info(text)
        return True
    
    def _osd_probe(self, pad, info):
        """Probe to get detection results + face detection for OSD."""
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
                        
                        # Get frame for face detection and draw boxes directly
                        if self.enable_face and self._face_detector:
                            n_frame = pyds.get_nvds_buf_surface(hash(buf), frame.batch_id)
                            if n_frame:
                                # Get frame as numpy (RGBA)
                                frame_np = np.array(n_frame, copy=False, order='C')
                                
                                # Run face detection
                                frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2RGB)
                                results = self._face_detector._face_mesh.process(frame_rgb)
                                
                                if results.multi_face_landmarks:
                                    h, w = frame_np.shape[:2]
                                    for lm in results.multi_face_landmarks:
                                        xs = [p.x * w for p in lm.landmark]
                                        ys = [p.y * h for p in lm.landmark]
                                        face_x, face_y = int(min(xs)), int(min(ys))
                                        face_w, face_h = int(max(xs) - face_x), int(max(ys) - face_y)
                                        self._faces.append(FaceResult(face_x, face_y, face_w, face_h))
                                        
                                        # Draw face box directly on frame (green)
                                        cv2.rectangle(frame_np, (face_x, face_y), 
                                                   (face_x + face_w, face_y + face_h),
                                                   (0, 255, 0, 255), 3)
                                        cv2.putText(frame_np, "Face", (face_x, face_y - 8),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0, 255), 2)
                        
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
                logger.debug(f"FPS: {self._current_fps:.1f} | Objects: {len(self._detections)}")
        
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
        """Handle new sample from appsink - extract frame for Python models."""
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK
        
        buf = sample.get_buffer()
        if not buf:
            return Gst.FlowReturn.OK
        
        self._calculate_fps()
        
        # Extract frame for Python models
        try:
            # Get caps to determine actual format and dimensions
            caps = sample.get_caps()
            if not caps:
                logger.debug("No caps from appsink")
                return Gst.FlowReturn.OK
            
            gst_struct = caps.get_structure(0)
            w = gst_struct.get_value("width") or self.width
            h = gst_struct.get_value("height") or self.height
            
            # Map buffer for reading
            success, buf_map = buf.map(Gst.MapFlags.READ)
            if not success:
                logger.debug("Failed to map buffer")
                return Gst.FlowReturn.OK
            
            try:
                # Create numpy array WITH COPY - critical step from NVIDIA sample
                # This creates a proper copy rather than viewing the buffer
                frame = np.ndarray(
                    shape=(h, w, 4),
                    dtype=np.uint8,
                    buffer=buf_map.data
                )
                frame_copy = np.array(frame, copy=True, order='C')
                
                # Convert RGBA to BGR (OpenCV default) - NVIDIA uses RGBA2BGRA
                frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
                
                # Unmap buffer before processing
                buf.unmap(buf_map)
                
                logger.debug(f"Frame processed: {frame_bgr.shape}")
                
                # Run Python models if enabled
                faces = []
                gestures = []
                poses = []
                
                # Face detection (every frame for testing)
                if self.enable_face and self._face_detector:
                    try:
                        # Convert BGR to RGB for MediaPipe
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        results = self._face_detector._face_mesh.process(frame_rgb)
                        if results.multi_face_landmarks:
                            for lm in results.multi_face_landmarks:
                                fh, fw = frame_rgb.shape[:2]
                                xs = [p.x * fw for p in lm.landmark]
                                ys = [p.y * fh for p in lm.landmark]
                                face_x, face_y = int(min(xs)), int(min(ys))
                                face_w, face_h = int(max(xs) - face_x), int(max(ys) - face_y)
                                faces.append(FaceResult(face_x, face_y, face_w, face_h))
                    except Exception as e:
                        logger.debug(f"Face detection error: {e}")
                
                # Gesture detection every frame
                if self.enable_gesture and self._gesture_recognizer:
                    try:
                        gestures = self._gesture_recognizer.recognize(frame_bgr)
                    except Exception as e:
                        logger.debug(f"Gesture detection error: {e}")
                        gestures = []
                
                # Pose detection every frame
                if self.enable_pose and self._pose_estimator:
                    try:
                        pose_result = self._pose_estimator.estimate(frame_bgr)
                        poses = [pose_result] if pose_result.detected else []
                    except Exception as e:
                        logger.debug(f"Pose detection error: {e}")
                        poses = []
                
                self._faces = faces
                self._gestures = gestures
                self._poses = poses
                self._face_count += 1
                
                # Call face callbacks
                for callback in self._face_callbacks:
                    callback(faces, self._current_fps)
                
            except Exception as e:
                logger.debug(f"Frame processing error: {e}")
                try:
                    buf.unmap(buf_map)
                except:
                    pass
                
        except Exception as e:
            logger.debug(f"Appsink frame error: {e}")
        
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
                 width: int = 1280, height: int = 480, fps: int = 30):
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
    width: int = 1280,
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
            
            # Get faces from pipeline
            faces_list = []
            if use_face and hasattr(pipeline, '_faces'):
                for face in pipeline._faces:
                    faces_list.append({
                        "bbox": [face.x, face.y, face.w, face.h]
                    })
            
            # Get gestures from pipeline
            gestures_list = []
            if use_gesture and hasattr(pipeline, '_gestures'):
                for gest in pipeline._gestures:
                    gestures_list.append({
                        "type": gest.gesture_type,
                        "hand": gest.handedness,
                        "confidence": round(gest.confidence, 2)
                    })
            
            # Get poses from pipeline
            poses_list = []
            if use_pose and hasattr(pipeline, '_poses'):
                for pose in pipeline._poses:
                    poses_list.append({
                        "keypoints": pose.keypoints,
                        "confidence": round(pose.confidence, 2)
                    })
            
            result = {
                "objects": detections_list,
                "faces": faces_list,
                "gestures": gestures_list,
                "poses": poses_list,
                "fps": round(current_fps, 1),
                "timestamp": current_time
            }
            
            try:
                _udp_sender.send(json.dumps(result))
            except Exception as e:
                logger.debug(f"UDP error: {e}")
            
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
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
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
