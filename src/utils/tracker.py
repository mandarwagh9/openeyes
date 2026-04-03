from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np
import time

from src.camera.types import BoundingBox, Detection
from src.utils.logger import get_logger


@dataclass
class Track:
    track_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float
    timestamp: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    centroid: Tuple[float, float] = field(default=(0.0, 0.0))
    is_predicted: bool = False
    _max_occlusion: int = 5


class ObjectTracker:
    """Simple IoU-based object tracker for person following."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_tracks: int = 50,
        follow_distance_min: float = 1.5,
        follow_distance_max: float = 2.5,
    ):
        self._logger = get_logger(__name__)
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._max_tracks = max_tracks

        self._follow_distance_min = follow_distance_min
        self._follow_distance_max = follow_distance_max

        self._tracks: Dict[int, Track] = {}
        self._next_track_id = 1
        self._frame_count = 0

        self._tracked_person: Optional[Track] = None
        self._follow_target_id: Optional[int] = None
        self._owner_track_id: Optional[int] = None

    def _compute_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)

        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _compute_centroid(self, bbox: BoundingBox) -> Tuple[float, float]:
        """Compute centroid of bounding box."""
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        return (cx, cy)

    def update(self, detections: List[Detection], frame_shape: Tuple[int, int]) -> List[Track]:
        """Update tracks with new detections."""
        self._frame_count += 1
        timestamp = time.time()

        if not detections:
            self._age_tracks()
            return self._get_active_tracks()

        detection_bboxes = [(d.bbox, d.class_name, d.confidence) for d in detections]

        matched_tracks = set()
        matched_detections = set()

        for track_id, track in self._tracks.items():
            if track.time_since_update > self._max_age:
                continue

            best_iou = 0
            best_idx = -1

            for idx, (bbox, class_name, conf) in enumerate(detection_bboxes):
                if idx in matched_detections:
                    continue

                if track.class_name != class_name:
                    continue

                iou = self._compute_iou(track.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= self._iou_threshold:
                bbox, class_name, conf = detection_bboxes[best_idx]
                track.bbox = bbox
                track.class_name = class_name
                track.confidence = conf
                track.timestamp = timestamp
                track.age += 1
                track.hits += 1
                track.time_since_update = 0
                track.centroid = self._compute_centroid(bbox)

                matched_tracks.add(track_id)
                matched_detections.add(best_idx)

        for idx, (bbox, class_name, conf) in enumerate(detection_bboxes):
            if idx in matched_detections:
                continue

            if len(self._tracks) >= self._max_tracks:
                break

            track_id = self._next_track_id
            self._next_track_id += 1

            centroid = self._compute_centroid(bbox)

            new_track = Track(
                track_id=track_id,
                class_name=class_name,
                bbox=bbox,
                confidence=conf,
                timestamp=timestamp,
                centroid=centroid,
            )
            self._tracks[track_id] = new_track

        self._age_tracks()

        return self._get_active_tracks()

    def _age_tracks(self) -> None:
        """Age all tracks and remove old ones."""
        for track_id in list(self._tracks.keys()):
            self._tracks[track_id].time_since_update += 1

        to_remove = [
            track_id
            for track_id, track in self._tracks.items()
            if track.time_since_update > self._max_age
        ]

        for track_id in to_remove:
            if self._follow_target_id == track_id:
                self._follow_target_id = None
                self._tracked_person = None
            del self._tracks[track_id]

    def _get_active_tracks(self) -> List[Track]:
        """Get tracks that have met minimum hit requirements."""
        return [
            track
            for track in self._tracks.values()
            if track.hits >= self._min_hits
        ]

    def get_tracks_by_class(self, class_name: str) -> List[Track]:
        """Get tracks filtered by class name."""
        return [
            track
            for track in self._get_active_tracks()
            if track.class_name.lower() == class_name.lower()
        ]

    def get_person_tracks(self) -> List[Track]:
        """Get person tracks only."""
        return self.get_tracks_by_class("person")

    def select_follow_target(self, frame_width: int, frame_height: int) -> Optional[Track]:
        """Select the best person to follow (closest to center)."""
        person_tracks = self.get_person_tracks()

        if not person_tracks:
            self._follow_target_id = None
            self._tracked_person = None
            return None

        center_x = frame_width / 2
        center_y = frame_height / 2

        best_track = None
        best_distance = float("inf")

        for track in person_tracks:
            cx, cy = track.centroid
            distance = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5

            if distance < best_distance:
                best_distance = distance
                best_track = track

        if best_track:
            self._follow_target_id = best_track.track_id
            self._tracked_person = best_track

        return best_track

    def get_follow_target(self) -> Optional[Track]:
        """Get the current follow target."""
        if self._tracked_person and self._tracked_person.time_since_update < 5:
            return self._tracked_person

        self._tracked_person = None
        return None

    def get_follow_command(self, frame_center: Tuple[int, int]) -> Optional[str]:
        """Get movement command based on tracked person position and distance."""
        target = self.get_follow_target()

        if not target:
            return None

        tx, ty = target.centroid
        fx, fy = frame_center

        bbox = target.bbox
        bbox_height = bbox.y2 - bbox.y1

        ideal_height = fy * 0.5
        height_diff = bbox_height - ideal_height

        if abs(height_diff) < fy * 0.15:
            pass
        elif height_diff > 0:
            return "backward"
        else:
            return "forward"

        dx = tx - fx
        threshold_x = fx * 0.2

        if dx < -threshold_x:
            return "left"
        elif dx > threshold_x:
            return "right"

        return "stop"

    def get_follow_command_with_depth(
        self,
        detections: List[Detection],
        depth_map: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> Optional[str]:
        """Get follow command using actual depth from MiDaS.
        
        Args:
            detections: Current frame detections from YOLO
            depth_map: MiDaS depth map (normalized 0-1, where 1=closest)
            frame_shape: (width, height) of the frame
            
        Returns:
            Movement command: 'forward', 'backward', 'left', 'right', 'stop', or None
        """
        person_dets = [d for d in detections if d.class_name.lower() == "person"]
        
        if not person_dets:
            if self._owner_track_id is not None:
                self._logger.info("Owner lost - stopping")
                self._tracked_person = None
            return None
        
        target_detection = None
        
        if self._owner_track_id is not None:
            for det in person_dets:
                if hasattr(det, 'track_id') and det.track_id == self._owner_track_id:
                    target_detection = det
                    break
            if target_detection is None:
                self._logger.info("Owner not visible - waiting")
        
        if target_detection is None:
            target_detection = max(person_dets, key=lambda d: d.confidence)
            if self._owner_track_id is None:
                self._logger.debug(f"Following person with confidence {target_detection.confidence:.2f}")
        
        bbox = target_detection.bbox
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)
        
        fw, fh = frame_shape
        x1 = max(0, min(x1, fw - 1))
        x2 = max(0, min(x2, fw))
        y1 = max(0, min(y1, fh - 1))
        y2 = max(0, min(y2, fh))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        if depth_map is None:
            return None
        
        depth_region = depth_map[y1:y2, x1:x2]
        if depth_region.size == 0:
            return None
        
        avg_depth = float(np.mean(depth_region))
        
        if avg_depth < 0.01:
            return None
        
        closeness = avg_depth
        
        # print(f"[DEPTH] bbox_h={y2-y1} ({((y2-y1)/fh)*100:.0f}%), cmd={distance_cmd}")
        
        bbox_height = y2 - y1
        height_ratio = bbox_height / fh
        
        # Distance zones:
        # forward: < 60% (person small = far away)
        # stop: 60-90% (person medium = just right)
        # backward: > 95% (person large = too close)
        if height_ratio > 0.95:
            distance_cmd = "backward"
        elif height_ratio < 0.60:
            distance_cmd = "forward"
        else:
            distance_cmd = "stop"
        
        # print(f"[DEPTH] bbox_h={y2-y1} ({height_ratio*100:.0f}%), cmd={distance_cmd}")
        
        tx = (bbox.x1 + bbox.x2) / 2
        ty = (bbox.y1 + bbox.y2) / 2
        fx, fy = fw / 2, fh / 2
        
        dx = tx - fx
        threshold_x = fw * 0.15
        
        if abs(dx) < threshold_x:
            lateral_cmd = "stop"
        elif dx < -threshold_x:
            lateral_cmd = "left"
        else:
            lateral_cmd = "right"
        
        if distance_cmd == "stop":
            return lateral_cmd
        elif lateral_cmd == "stop":
            return distance_cmd
        else:
            return distance_cmd

    def set_owner_from_gesture(
        self,
        detections: List[Detection],
        gesture_track_positions: Dict[int, Tuple[float, float]],
    ) -> bool:
        """Set the owner based on gesture recognition.
        
        When a person shows 'open_palm' gesture, that person becomes the owner.
        
        Args:
            detections: Current detections
            gesture_track_positions: Dict of {track_id: (gesture_type, centroid)}
                                     If gesture_type == 'open_palm', that track becomes owner
                                     
        Returns:
            True if owner was set, False otherwise
        """
        if not detections or not gesture_track_positions:
            return False
        
        person_dets = [d for d in detections if d.class_name.lower() == "person"]
        
        for det in person_dets:
            det_track_id = getattr(det, 'track_id', None)
            if det_track_id is None:
                for track_id, (gesture_type, centroid) in gesture_track_positions.items():
                    if gesture_type == "open_palm":
                        bbox = det.bbox
                        cx = (bbox.x1 + bbox.x2) / 2
                        cy = (bbox.y1 + bbox.y2) / 2
                        dist = ((cx - centroid[0]) ** 2 + (cy - centroid[1]) ** 2) ** 0.5
                        if dist < 100:
                            self._owner_track_id = det_track_id
                            self._logger.info(f"Owner set to track_id={det_track_id} (showed open_palm)")
                            return True
        
        for track_id, (gesture_type, centroid) in gesture_track_positions.items():
            if gesture_type == "open_palm":
                for det in person_dets:
                    bbox = det.bbox
                    cx = (bbox.x1 + bbox.x2) / 2
                    cy = (bbox.y1 + bbox.y2) / 2
                    dist = ((cx - centroid[0]) ** 2 + (cy - centroid[1]) ** 2) ** 0.5
                    if dist < 100:
                        det_track_id = getattr(det, 'track_id', None)
                        if det_track_id is not None:
                            self._owner_track_id = det_track_id
                            self._logger.info(f"Owner set to track_id={det_track_id} (showed open_palm)")
                            return True
        
        return False

    def clear_owner(self) -> None:
        """Clear the owner - robot will follow any person again."""
        self._owner_track_id = None
        self._logger.info("Owner cleared - will follow any person")

    @property
    def owner_track_id(self) -> Optional[int]:
        """Get the current owner's track ID."""
        return self._owner_track_id

    def reset(self) -> None:
        """Reset all tracks."""
        self._tracks.clear()
        self._next_track_id = 1
        self._follow_target_id = None
        self._tracked_person = None
        self._frame_count = 0

    @property
    def tracked_count(self) -> int:
        """Get number of active tracks."""
        return len(self._get_active_tracks())

    @property
    def follow_target_id(self) -> Optional[int]:
        return self._follow_target_id

    def update_with_predictions(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, int],
        predictions: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        max_occlusion_frames: int = 5,
    ) -> List[Track]:
        """Update tracks with detections and world model predictions.

        When detections are missing for a track, uses predicted bounding boxes
        to maintain tracking through occlusions.

        Args:
            detections: Current frame detections
            frame_shape: (width, height) of the frame
            predictions: Dict of {track_id: (x1, y1, x2, y2)} from world model
            max_occlusion_frames: Max frames to predict through occlusion

        Returns:
            Updated list of active tracks
        """
        self._frame_count += 1
        timestamp = time.time()

        if not detections and not predictions:
            self._age_tracks()
            return self._get_active_tracks()

        detection_bboxes = [(d.bbox, d.class_name, d.confidence) for d in detections]
        matched_tracks = set()
        matched_detections = set()

        for track_id, track in list(self._tracks.items()):
            if track.time_since_update > max(track._max_occlusion or max_occlusion_frames, self._max_age):
                continue

            best_iou = 0
            best_idx = -1

            for idx, (bbox, class_name, conf) in enumerate(detection_bboxes):
                if idx in matched_detections:
                    continue
                if track.class_name != class_name:
                    continue
                iou = self._compute_iou(track.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= self._iou_threshold:
                bbox, class_name, conf = detection_bboxes[best_idx]
                track.bbox = bbox
                track.class_name = class_name
                track.confidence = conf
                track.timestamp = timestamp
                track.age += 1
                track.hits += 1
                track.time_since_update = 0
                track.centroid = self._compute_centroid(bbox)
                track.is_predicted = False
                matched_tracks.add(track_id)
                matched_detections.add(best_idx)

        if predictions:
            for track_id, pred_bbox in predictions.items():
                if track_id in matched_tracks:
                    continue
                if track_id not in self._tracks:
                    continue

                track = self._tracks[track_id]
                if track.time_since_update > max_occlusion_frames:
                    continue

                px1, py1, px2, py2 = pred_bbox
                pred_bbox_obj = BoundingBox(x1=px1, y1=py1, x2=px2, y2=py2)

                track.bbox = pred_bbox_obj
                track.timestamp = timestamp
                track.age += 1
                track.time_since_update += 1
                track.centroid = self._compute_centroid(pred_bbox_obj)
                track.is_predicted = True
                track.confidence *= 0.95

                matched_tracks.add(track_id)

        for idx, (bbox, class_name, conf) in enumerate(detection_bboxes):
            if idx in matched_detections:
                continue
            if len(self._tracks) >= self._max_tracks:
                break

            track_id = self._next_track_id
            self._next_track_id += 1
            centroid = self._compute_centroid(bbox)

            new_track = Track(
                track_id=track_id,
                class_name=class_name,
                bbox=bbox,
                confidence=conf,
                timestamp=timestamp,
                centroid=centroid,
                is_predicted=False,
            )
            new_track._max_occlusion = max_occlusion_frames
            self._tracks[track_id] = new_track

        self._age_tracks()
        return self._get_active_tracks()

    def get_predicted_tracks(self) -> List[Track]:
        """Get tracks that are currently using predicted positions."""
        return [t for t in self._get_active_tracks() if t.is_predicted]
