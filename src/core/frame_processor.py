import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Any, Protocol

from src.camera.types import (
    DepthData,
    FaceDetection,
    Gesture,
    PoseData,
    VisionResult,
    TrackData,
    BoundingBox,
)
from src.camera.camera_handler import CameraHandler
from src.models.object_detector import ObjectDetector
from src.models.depth_estimator import DepthEstimator
from src.models.face_detector import FaceDetector
from src.models.gesture_recognizer import GestureRecognizer
from src.models.pose_estimator import PoseEstimator
from src.utils.tracker import ObjectTracker
from src.utils.frame_skipper import AdaptiveFrameSkipper, MultiModelFrameScheduler
from src.exceptions import ModelError


class PerformanceMonitor(Protocol):
    """Protocol for performance monitoring."""
    def record_frame(self, detection_count: int) -> None: ...
    def get_stats(self) -> dict: ...
    def log_stats(self) -> None: ...


class FrameProcessor:
    """Handles frame processing logic for the vision system.
    
    This class is responsible for:
    - Running model inference (parallel and sequential)
    - Object tracking and person following
    - VLA processing
    - Frame skipping
    """

    def __init__(
        self,
        camera: CameraHandler,
        detector: ObjectDetector,
        depth_estimator: Optional[DepthEstimator] = None,
        face_detector: Optional[FaceDetector] = None,
        gesture_recognizer: Optional[GestureRecognizer] = None,
        pose_estimator: Optional[PoseEstimator] = None,
        tracker: Optional[ObjectTracker] = None,
        perf_monitor: Optional[PerformanceMonitor] = None,
        use_parallel: bool = True,
        use_face: bool = True,
        use_gesture: bool = True,
        use_pose: bool = True,
        use_depth: bool = True,
        use_tracking: bool = True,
        use_vla: bool = False,
        vla_model: Optional[Any] = None,
        real_vla_model: Optional[Any] = None,
        advanced_ai: Optional[Any] = None,
        frame_scheduler: Optional[MultiModelFrameScheduler] = None,
        adaptive_skipper: Optional[AdaptiveFrameSkipper] = None,
        logger: Optional[Any] = None,
        world_model: Optional[Any] = None,
        use_world_model: bool = False,
        world_model_horizon: int = 10,
        world_model_samples: int = 100,
        prediction_fps: int = 30,
        occlusion_frames: int = 5,
        safety_predict: bool = False,
    ):
        self._camera = camera
        self._detector = detector
        self._depth_estimator = depth_estimator
        self._face_detector = face_detector
        self._gesture_recognizer = gesture_recognizer
        self._pose_estimator = pose_estimator
        self._tracker = tracker
        self._perf_monitor = perf_monitor
        self._use_parallel = use_parallel
        self._use_face = use_face
        self._use_gesture = use_gesture
        self._use_pose = use_pose
        self._use_depth = use_depth
        self._use_tracking = use_tracking
        self._use_vla = use_vla
        self._vla_model = vla_model
        self._real_vla_model = real_vla_model
        self._advanced_ai = advanced_ai
        self._frame_scheduler = frame_scheduler
        self._adaptive_skipper = adaptive_skipper
        self._logger = logger

        self._world_model = world_model
        self._use_world_model = use_world_model
        self._wm_horizon = world_model_horizon
        self._wm_samples = world_model_samples
        self._prediction_fps = prediction_fps
        self._occlusion_frames = occlusion_frames
        self._safety_predict = safety_predict

        self._executor = ThreadPoolExecutor(max_workers=5)
        
        self._last_depth: Optional[np.ndarray] = None
        self._last_pose: Optional[PoseData] = None
        self._last_faces: list[FaceDetection] = []
        self._last_gestures: list[Gesture] = []
        self._frame_id: int = 0
        self._follow_target: bool = False
        self._pose_skip_frames: int = 1

        self._wm_prediction_frame: int = 0
        self._wm_prediction_interval: int = max(1, 30 // prediction_fps) if prediction_fps > 0 else 30
        self._last_predictions: dict = {}
        self._current_latent: Optional[np.ndarray] = None

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @frame_id.setter
    def frame_id(self, value: int) -> None:
        self._frame_id = value

    def set_follow_target(self, enabled: bool) -> None:
        self._follow_target = enabled

    def set_pose_skip_frames(self, skip: int) -> None:
        self._pose_skip_frames = max(1, skip)

    def process_frame(self, frame: np.ndarray) -> VisionResult:
        """Process a single frame through all enabled models.
        
        Args:
            frame: Input frame as BGR numpy array
            
        Returns:
            VisionResult containing all detections and analysis
        """
        timestamp = time.time()

        should_skip = False
        if self._frame_scheduler and self._adaptive_skipper:
            should_skip = not self._adaptive_skipper.should_process(frame)

        if should_skip:
            return self._create_cached_result(timestamp)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections, depth_map, faces, gestures, pose = self._run_inference(
            frame, frame_rgb
        )


        depth_enabled = depth_map is not None
        if depth_map is not None:
            self._last_depth = depth_map

        depth = DepthData(enabled=depth_enabled, depth_map=depth_map)

        tracks = self._process_tracking(detections, frame, depth_map)

        self._update_frame_scheduler(faces, gestures, pose)

        track_data_list = self._convert_tracks_to_data(tracks)

        predictions = self._collect_predictions_for_result(tracks)

        vla_commands = self._process_vla(
            frame, detections, faces, gestures, pose, track_data_list, depth
        )

        for cmd in vla_commands:
            if self._logger:
                self._logger.info(f"VLA Command: {cmd.action} - {cmd.reasoning}")

        result = VisionResult(
            timestamp=timestamp,
            frame_id=self._frame_id,
            objects=detections,
            depth=depth,
            faces=faces,
            gestures=gestures,
            pose=pose,
            tracks=track_data_list,
            predictions=predictions,
        )

        return result

    def _run_inference(
        self, frame: np.ndarray, frame_rgb: np.ndarray
    ) -> tuple[
        list[Any],
        Optional[np.ndarray],
        list[FaceDetection],
        list[Gesture],
        PoseData,
    ]:
        """Run all model inference (parallel or sequential)."""
        if self._use_parallel:
            return self._run_parallel(frame, frame_rgb)
        else:
            return self._run_sequential(frame, frame_rgb)

    def _run_parallel(
        self, frame: np.ndarray, frame_rgb: np.ndarray
    ) -> tuple[
        list[Any],
        Optional[np.ndarray],
        list[FaceDetection],
        list[Gesture],
        PoseData,
    ]:
        """Run models in parallel using ThreadPoolExecutor."""
        futures = {}

        if self._detector:
            futures["detector"] = self._executor.submit(self._detector.detect, frame)

        if self._depth_estimator and self._depth_estimator.is_loaded:
            futures["depth"] = self._executor.submit(
                self._depth_estimator.estimate, frame
            )

        if self._face_detector:
            futures["face"] = self._executor.submit(
                self._face_detector.detect, frame_rgb
            )

        if self._gesture_recognizer:
            futures["gesture"] = self._executor.submit(
                self._gesture_recognizer.recognize, frame_rgb
            )

        if self._pose_estimator:
            futures["pose"] = self._executor.submit(
                self._pose_estimator.estimate, frame_rgb
            )

        from concurrent.futures import Future
        
        detections: list[Any] = []
        depth_map: Optional[np.ndarray] = None
        faces: list[FaceDetection] = []
        gestures: list[Gesture] = []
        pose = PoseData(detected=False)

        if not futures:
            return detections, depth_map, faces, gestures, pose

        future_list: list[tuple[str, Future]] = list(futures.items())
        for key, future in future_list:
            try:
                result = future.result()
                if key == "detector":
                    detections = result
                    if self._frame_scheduler:
                        self._frame_scheduler.update("detector", detections)
                elif key == "depth":
                    depth_map = result
                elif key == "face":
                    faces = result
                elif key == "gesture":
                    gestures = result
                elif key == "pose":
                    pose = result
            except Exception as e:
                if self._logger:
                    self._logger.warning(f"Model {key} failed: {e}")

        return detections, depth_map, faces, gestures, pose

    def _run_sequential(
        self, frame: np.ndarray, frame_rgb: np.ndarray
    ) -> tuple[
        list[Any],
        Optional[np.ndarray],
        list[FaceDetection],
        list[Gesture],
        PoseData,
    ]:
        """Run models sequentially (fallback)."""
        detections = []
        if self._detector:
            try:
                detections = self._detector.detect(frame)
            except Exception as e:
                if self._logger:
                    self._logger.warning(f"Detection failed: {e}")

        depth_map = None
        if self._depth_estimator and self._depth_estimator.is_loaded:
            try:
                depth_map = self._depth_estimator.estimate(frame)
            except Exception as e:
                if self._logger:
                    self._logger.warning(f"Depth estimation failed: {e}")

        faces = []
        if self._face_detector:
            try:
                faces = self._face_detector.detect(frame_rgb)
            except Exception as e:
                if self._logger:
                    self._logger.warning(f"Face detection failed: {e}")

        gestures = []
        if self._gesture_recognizer:
            try:
                gestures = self._gesture_recognizer.recognize(frame_rgb)
            except Exception as e:
                if self._logger:
                    self._logger.warning(f"Gesture recognition failed: {e}")

        pose = PoseData(detected=False)
        if self._pose_estimator:
            if self._frame_id % (self._pose_skip_frames + 1) == 0:
                try:
                    pose = self._pose_estimator.estimate(frame_rgb)
                    self._last_pose = pose
                except Exception as e:
                    if self._logger:
                        self._logger.warning(f"Pose estimation failed: {e}")
            else:
                pose = self._last_pose if self._last_pose else PoseData(detected=False)

        return detections, depth_map, faces, gestures, pose

    def _create_cached_result(self, timestamp: float) -> VisionResult:
        """Create result using cached values when frame is skipped."""
        last_objects = []
        if self._frame_scheduler:
            last_objects = self._frame_scheduler.get_last("detector") or []
            last_faces = self._frame_scheduler.get_last("face") or []
            last_gestures = self._frame_scheduler.get_last("gesture") or []
            last_pose = self._frame_scheduler.get_last("pose")
        else:
            last_faces = self._last_faces
            last_gestures = self._last_gestures
            last_pose = self._last_pose

        last_depth_data = DepthData(
            enabled=self._last_depth is not None,
            depth_map=self._last_depth,
        )

        return VisionResult(
            timestamp=timestamp,
            frame_id=self._frame_id,
            objects=last_objects,
            depth=last_depth_data,
            faces=last_faces,
            gestures=last_gestures,
            pose=last_pose if last_pose else PoseData(detected=False),
            tracks=[],
        )

    def _process_tracking(
        self,
        detections: list[Any],
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
    ) -> list[Any]:
        """Process object tracking and person following."""
        if not self._tracker or not detections:
            return []

        if self._use_world_model and self._world_model and self._world_model.is_loaded:
            tracks = self._process_tracking_with_world_model(
                detections, frame, depth_map
            )
        else:
            tracks = self._tracker.update(detections, (frame.shape[1], frame.shape[0]))

        if self._follow_target and frame.shape:
            self._handle_follow_command(detections, depth_map, frame.shape)

        return tracks

    def _process_tracking_with_world_model(
        self,
        detections: list[Any],
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
    ) -> list[Any]:
        """Process tracking with world model predictions for occlusion handling."""
        self._wm_prediction_frame += 1

        should_predict = (
            self._wm_prediction_frame % self._wm_prediction_interval == 0
        )

        predictions = None
        if should_predict and self._world_model and self._world_model.is_loaded:
            predictions = self._generate_predictions(detections, frame)

        tracks = self._tracker.update_with_predictions(
            detections,
            (frame.shape[1], frame.shape[0]),
            predictions=predictions,
            max_occlusion_frames=self._occlusion_frames,
        )

        if should_predict and self._world_model.is_loaded:
            try:
                self._current_latent = self._world_model.encode(frame)
                self._world_model.record_state(self._current_latent)
            except Exception as e:
                if self._logger:
                    self._logger.debug(f"World model encode error: {e}")

        return tracks

    def _generate_predictions(
        self,
        detections: list[Any],
        frame: np.ndarray,
    ) -> Optional[dict]:
        """Generate world model predictions for tracked objects."""
        if not self._world_model or not self._world_model.is_loaded:
            return None

        if not self._tracker:
            return None

        predictions = {}
        h, w = frame.shape[:2]

        active_tracks = self._tracker._get_active_tracks() if hasattr(self._tracker, '_get_active_tracks') else []

        for track in active_tracks:
            if track.time_since_update > 0:
                bbox = (track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2)

                pred = self._world_model.predict_bbox_trajectory(
                    track_id=track.track_id,
                    class_name=track.class_name,
                    current_bbox=bbox,
                    frame_shape=(w, h),
                    horizon=self._wm_horizon,
                )

                next_pos = pred.get_next_position()
                if next_pos:
                    predictions[track.track_id] = (
                        next_pos.x1, next_pos.y1, next_pos.x2, next_pos.y2
                    )
                    self._last_predictions[track.track_id] = pred

        return predictions if predictions else None

    def _handle_follow_command(
        self,
        detections: list[Any],
        depth_map: Optional[np.ndarray],
        frame_shape: tuple,
    ) -> None:
        """Handle person following logic."""
        if not self._tracker:
            return

        gestures = self._last_gestures
        if gestures:
            gesture_positions = {}
            for det in detections:
                if det.class_name.lower() == "person":
                    cx = (det.bbox.x1 + det.bbox.x2) / 2
                    cy = (det.bbox.y1 + det.bbox.y2) / 2
                    track_id = getattr(det, "track_id", 0)
                    gesture_positions[track_id] = (gestures[0].gesture_type, (cx, cy))
                    if self._tracker.owner_track_id is None:
                        self._tracker.set_owner_from_gesture(detections, gesture_positions)

        if self._depth_estimator and depth_map is not None:
            follow_cmd = self._tracker.get_follow_command_with_depth(
                detections,
                depth_map,
                (frame_shape[1], frame_shape[0]),
            )
        else:
            self._tracker.select_follow_target(frame_shape[1], frame_shape[0])
            frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)
            follow_cmd = self._tracker.get_follow_command(frame_center)

        if follow_cmd and self._logger:
            self._logger.info(f"Follow command: {follow_cmd}")

    def _update_frame_scheduler(
        self,
        faces: list[FaceDetection],
        gestures: list[Gesture],
        pose: PoseData,
    ) -> None:
        """Update frame scheduler with new results."""
        if not self._frame_scheduler:
            return

        self._frame_scheduler.update("face", faces)
        self._frame_scheduler.update("gesture", gestures)
        self._frame_scheduler.update("pose", pose)
        self._frame_scheduler.next_frame()

    def _convert_tracks_to_data(self, tracks: list[Any]) -> list[TrackData]:
        """Convert tracker tracks to TrackData objects."""
        track_data_list = []
        for track in tracks:
            track_data_list.append(
                TrackData(
                    track_id=track.track_id,
                    class_name=track.class_name,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    centroid=track.centroid,
                    age=track.age,
                    is_predicted=getattr(track, 'is_predicted', False),
                )
            )
        return track_data_list

    def _collect_predictions_for_result(self, tracks: list[Any]) -> list[list]:
        """Collect world model predictions for visualization."""
        if not self._use_world_model or not self._world_model:
            return []

        predictions = []
        h, w = 480, 640
        if hasattr(self, '_camera') and self._camera:
            try:
                h, w = self._camera.height, self._camera.width
            except Exception:
                pass

        for track in tracks:
            if hasattr(track, 'bbox'):
                bbox = (track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2)
                try:
                    pred = self._world_model.predict_bbox_trajectory(
                        track_id=track.track_id,
                        class_name=track.class_name,
                        current_bbox=bbox,
                        frame_shape=(w, h),
                        horizon=5,
                    )
                    future_bboxes = [
                        BoundingBox(x1=p.x1, y1=p.y1, x2=p.x2, y2=p.y2)
                        for p in pred.positions
                    ]
                    predictions.append(future_bboxes)
                except Exception:
                    pass

        return predictions

    def _process_vla(
        self,
        frame: np.ndarray,
        detections: list[Any],
        faces: list[FaceDetection],
        gestures: list[Gesture],
        pose: PoseData,
        tracks: list[TrackData],
        depth: DepthData,
    ) -> list[Any]:
        """Process VLA commands."""
        if not self._use_vla or self._vla_model is None:
            return []

        try:
            vla_context = {
                "detections": detections,
                "depth": depth,
                "faces": faces,
                "gesture": gestures[0] if gestures else None,
                "pose": pose,
                "tracks": tracks,
            }

            vla_commands = self._vla_model.process(frame, detections, vla_context)

            if self._real_vla_model is not None:
                real_commands = self._process_real_vla(frame, vla_context)
                vla_commands.extend(real_commands)

            return vla_commands

        except Exception as e:
            if self._logger:
                self._logger.warning(f"VLA processing error: {e}")
            return []

    def _process_real_vla(
        self, frame: np.ndarray, vla_context: dict
    ) -> list[Any]:
        """Process real VLA model predictions."""
        if self._real_vla_model is None:
            return []

        try:
            from src.models.vla_models import VLAAction
            from src.models.vla import VLACommand

            instruction = vla_context.get("instruction", "follow the person")
            real_action = self._real_vla_model.predict_action(frame, instruction)

            if real_action is None:
                return []

            if self._logger:
                self._logger.info(
                    f"Real VLA Action: {real_action.action_type} "
                    f"(confidence: {real_action.confidence:.2f})"
                )

            cmd = VLACommand(
                action=real_action.action_type,
                target=None,
                confidence=real_action.confidence,
                reasoning=real_action.reasoning,
            )
            return [cmd]

        except Exception as e:
            if self._logger:
                self._logger.warning(f"Real VLA prediction failed: {e}")
            return []

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)
