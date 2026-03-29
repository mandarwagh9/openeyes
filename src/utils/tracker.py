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


class ObjectTracker:
    """Simple IoU-based object tracker for person following."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_tracks: int = 50,
    ):
        self._logger = get_logger(__name__)
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._max_tracks = max_tracks

        self._tracks: Dict[int, Track] = {}
        self._next_track_id = 1
        self._frame_count = 0

        self._tracked_person: Optional[Track] = None
        self._follow_target_id: Optional[int] = None

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
        """Get movement command based on tracked person position."""
        target = self.get_follow_target()

        if not target:
            return None

        tx, ty = target.centroid
        fx, fy = frame_center

        dx = tx - fx
        dy = ty - fy

        threshold_x = fx * 0.3
        threshold_y = fy * 0.3

        if abs(dx) < threshold_x and abs(dy) < threshold_y:
            return "stop"

        if dy < -threshold_y:
            return "forward"
        elif dy > threshold_y:
            return "backward"

        if dx < -threshold_x:
            return "left"
        elif dx > threshold_x:
            return "right"

        return "stop"

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
