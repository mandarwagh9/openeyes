import logging
from typing import TYPE_CHECKING, Optional, Any

from src.camera.camera_handler import CameraHandler
from src.models.object_detector import ObjectDetector
from src.models.depth_estimator import DepthEstimator
from src.models.face_detector import FaceDetector
from src.models.gesture_recognizer import GestureRecognizer
from src.models.pose_estimator import PoseEstimator
from src.output.udp_sender import UDPSender
from src.utils.config import Config
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.tracker import ObjectTracker
from src.exceptions import CameraError, ModelError

if TYPE_CHECKING:
    from src.models.vla import VLAModel, AdvancedAI, VLACommand
    from src.models.vla_models import AnyVLAWrapper


class InitializationManager:
    """Manages initialization of all vision system components."""

    def __init__(self, config: Config, logger: logging.Logger):
        self._config = config
        self._logger = logger

    def init_camera(self) -> CameraHandler:
        """Initialize camera handler."""
        camera = CameraHandler(
            source=self._config.camera_source,
            width=self._config.camera_width,
            height=self._config.camera_height,
            fps=self._config.camera_fps,
        )
        try:
            camera.open()
        except CameraError as e:
            self._logger.error(f"Camera initialization failed: {e}")
            raise
        return camera

    def init_detector(self) -> ObjectDetector:
        """Initialize object detector."""
        detector = ObjectDetector(
            model_path=self._config.yolo_path or "models/yolov10n.onnx",
            confidence=self._config.yolo_confidence,
            iou_threshold=self._config.yolo_iou_threshold,
        )
        try:
            detector.load()
            self._logger.info(f"Object Detector loaded: {detector.name}")
        except ModelError as e:
            self._logger.error(f"Detector initialization failed: {e}")
            raise
        return detector

    def init_depth_estimator(self, enabled: bool, model: str = "da3-small") -> Optional[DepthEstimator]:
        """Initialize depth estimator if enabled."""
        if not enabled:
            return None
        try:
            estimator = DepthEstimator(model=model)
            estimator.load()
            if estimator.is_loaded:
                self._logger.info(f"Depth Estimator loaded ({estimator.name})")
            else:
                self._logger.warning("Depth Estimator using fallback")
            return estimator
        except ModelError as e:
            self._logger.warning(f"Depth Estimator not available: {e}")
            return None

    def init_face_detector(self, enabled: bool) -> Optional[FaceDetector]:
        """Initialize face detector if enabled."""
        if not enabled:
            return None
        try:
            detector = FaceDetector()
            detector.load()
            if self._config.debug:
                detector._debug = True
            self._logger.info("Face Detector loaded")
            return detector
        except ModelError as e:
            self._logger.warning(f"Face Detector not available: {e}")
            return None

    def init_gesture_recognizer(self, enabled: bool) -> Optional[GestureRecognizer]:
        """Initialize gesture recognizer if enabled."""
        if not enabled:
            return None
        try:
            recognizer = GestureRecognizer(
                min_confidence=self._config.gesture_confidence
            )
            recognizer.load()
            if self._config.debug:
                recognizer._debug = True
            self._logger.info("Gesture Recognizer loaded")
            return recognizer
        except ModelError as e:
            self._logger.warning(f"Gesture Recognizer not available: {e}")
            return None

    def init_pose_estimator(self, enabled: bool) -> Optional[PoseEstimator]:
        """Initialize pose estimator if enabled."""
        if not enabled:
            return None
        try:
            estimator = PoseEstimator()
            estimator.load()
            self._logger.info("Pose Estimator loaded")
            return estimator
        except ModelError as e:
            self._logger.warning(f"Pose Estimator not available: {e}")
            return None

    def init_tracker(self, enabled: bool) -> Optional[ObjectTracker]:
        """Initialize object tracker if enabled."""
        if not enabled:
            return None
        tracker = ObjectTracker(
            max_age=self._config.tracking_max_age,
            min_hits=self._config.tracking_min_hits,
            iou_threshold=self._config.tracking_iou_threshold,
            follow_distance_min=self._config.follow_distance_min,
            follow_distance_max=self._config.follow_distance_max,
        )
        self._logger.info(
            f"Object tracking enabled (max_age={self._config.tracking_max_age}, "
            f"min_hits={self._config.tracking_min_hits}, "
            f"follow_dist={self._config.follow_distance_min}-{self._config.follow_distance_max}m)"
        )
        return tracker

    def init_performance_monitor(self) -> PerformanceMonitor:
        """Initialize performance monitor."""
        return PerformanceMonitor(
            enabled=self._config.performance_monitoring_enabled,
            stats_interval=self._config.performance_stats_interval,
            log_performance=self._config.log_performance,
        )

    def init_udp_sender(self) -> UDPSender:
        """Initialize UDP sender."""
        sender = UDPSender(
            host=self._config.output_host,
            port=self._config.output_port,
        )
        sender.open()
        return sender

    def init_vla(
        self,
        enabled: bool,
        vla_module: Optional[Any] = None,
    ) -> tuple[Optional[Any], Optional["AdvancedAI"]]:
        """Initialize VLA models.
        
        Returns:
            Tuple of (vla_model, advanced_ai)
        """
        if not enabled or vla_module is None:
            return None, None

        vla_model = None
        advanced_ai = None

        try:
            from src.models.vla import VLAModel, AdvancedAI
            
            vla_model = VLAModel()
            vla_model.load()
            self._logger.info("VLA Model loaded")

            advanced_ai = AdvancedAI()
            advanced_ai.initialize()
            self._logger.info("Advanced AI initialized")
        except Exception as e:
            self._logger.warning(f"VLA Model not available: {e}")

        return vla_model, advanced_ai

    def init_real_vla(
        self,
        model_type: str,
        device: str = "cuda",
    ) -> Optional[Any]:
        """Initialize a real VLA model.
        
        Args:
            model_type: One of 'smolvla', 'openvla', 'octo'
            device: Device to run on ('cuda' or 'cpu')
            
        Returns:
            Loaded VLA model wrapper or None
        """
        if not model_type:
            return None

        try:
            from src.models.vla_models import create_vla_model
            
            model = create_vla_model(model_type=model_type, device=device)
            if model and model.is_loaded:
                self._logger.info(f"Real VLA model loaded: {model_type}")
                return model
            else:
                self._logger.warning(f"Failed to load {model_type}")
                return None
        except Exception as e:
            self._logger.warning(f"Real VLA not available: {e}")
            return None


def init_all_components(
    config: Config,
    logger: logging.Logger,
    use_face: bool = True,
    use_gesture: bool = True,
    use_pose: bool = True,
    use_depth: bool = True,
    use_tracking: bool = True,
    use_vla: bool = False,
    real_vla_type: str = "",
) -> dict[str, Any]:
    """Initialize all vision system components.
    
    Args:
        config: Configuration object
        logger: Logger instance
        use_face: Enable face detection
        use_gesture: Enable gesture recognition
        use_pose: Enable pose estimation
        use_depth: Enable depth estimation
        use_tracking: Enable object tracking
        use_vla: Enable VLA
        real_vla_type: Real VLA model type
        
    Returns:
        Dictionary of initialized components
    """
    from src.models.vla_models import create_vla_model

    init_mgr = InitializationManager(config, logger)

    components = {
        "camera": init_mgr.init_camera(),
        "detector": init_mgr.init_detector(),
        "depth_estimator": init_mgr.init_depth_estimator(use_depth, config.depth_model),
        "face_detector": init_mgr.init_face_detector(use_face),
        "gesture_recognizer": init_mgr.init_gesture_recognizer(use_gesture),
        "pose_estimator": init_mgr.init_pose_estimator(use_pose),
        "tracker": init_mgr.init_tracker(use_tracking),
        "perf_monitor": init_mgr.init_performance_monitor(),
        "udp_sender": init_mgr.init_udp_sender(),
    }

    vla_model, advanced_ai = init_mgr.init_vla(use_vla, None)
    components["vla_model"] = vla_model
    components["advanced_ai"] = advanced_ai

    if real_vla_type:
        components["real_vla_model"] = init_mgr.init_real_vla(real_vla_type)
    else:
        components["real_vla_model"] = None

    return components
