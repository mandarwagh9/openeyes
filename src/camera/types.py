from dataclasses import dataclass
from typing import List, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Detection:
    class_name: str
    bbox: BoundingBox
    confidence: float


@dataclass
class DepthData:
    enabled: bool
    depth_map: Optional["np.ndarray"] = None
    min_distance: Optional[float] = None
    max_distance: Optional[float] = None


@dataclass
class FaceDetection:
    bbox: BoundingBox
    confidence: float


@dataclass
class Gesture:
    gesture_type: str
    handedness: str
    confidence: float


@dataclass
class PoseKeypoint:
    x: float
    y: float
    visibility: float


@dataclass
class PoseData:
    detected: bool
    keypoints: Optional[List[PoseKeypoint]] = None
    bbox: Optional[BoundingBox] = None
    landmarks: Optional[List[PoseKeypoint]] = None


@dataclass
class VisionResult:
    timestamp: float
    frame_id: int
    objects: List[Detection]
    depth: DepthData
    faces: List[FaceDetection]
    gestures: List[Gesture]
    pose: PoseData


class CameraInterface(Protocol):
    def read(self) -> Optional["np.ndarray"]:
        ...

    def release(self) -> None:
        ...

    @property
    def is_opened(self) -> bool:
        ...
