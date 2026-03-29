from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
import time

from src.camera.types import BoundingBox, Detection
from src.utils.logger import get_logger


@dataclass
class VLACommand:
    action: str
    target: Optional[str]
    confidence: float
    reasoning: str


class VLAModel:
    """Vision-Language-Action model for intelligent robot control."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        language_model: str = "llama2",
        device: str = "cuda",
    ):
        self._model_path = model_path
        self._language_model = language_model
        self._device = device
        self._logger = get_logger(__name__)
        self._model = None
        self._is_loaded = False

    def load(self) -> None:
        """Load the VLA model."""
        try:
            self._logger.info("Initializing VLA model...")
            
            self._is_loaded = True
            self._logger.info("VLA model ready (simulation mode)")

        except Exception as e:
            self._logger.warning(f"VLA model not available: {e}")
            self._logger.info("Using rule-based fallback")
            self._is_loaded = True

    def process(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[VLACommand]:
        """Process vision input and generate actions."""
        if not self._is_loaded:
            return []

        commands = []

        person_detected = any(d.class_name == "person" for d in detections)
        closest_person = None
        min_distance = float("inf")

        for det in detections:
            if det.class_name == "person":
                bbox = det.bbox
                cx = (bbox.x1 + bbox.x2) / 2
                cy = (bbox.y1 + bbox.y2) / 2
                distance = cx * cx + cy * cy
                if distance < min_distance:
                    min_distance = distance
                    closest_person = det

        if closest_person:
            bbox = closest_person.bbox
            h = bbox.y2 - bbox.y1
            w = bbox.x2 - bbox.x1

            if h < 100:
                commands.append(VLACommand(
                    action="move_forward",
                    target="person",
                    confidence=0.9,
                    reasoning="Person is far, moving closer"
                ))
            elif h > 300:
                commands.append(VLACommand(
                    action="move_backward",
                    target="person",
                    confidence=0.9,
                    reasoning="Person is too close, backing up"
                ))

            cx = (bbox.x1 + bbox.x2) / 2
            frame_center = frame.shape[1] / 2

            if cx < frame_center - 50:
                commands.append(VLACommand(
                    action="turn_left",
                    target="person",
                    confidence=0.8,
                    reasoning="Person is to the left"
                ))
            elif cx > frame_center + 50:
                commands.append(VLACommand(
                    action="turn_right",
                    target="person",
                    confidence=0.8,
                    reasoning="Person is to the right"
                ))

        if context:
            gesture = context.get("gesture")
            if gesture:
                if "stop" in gesture.gesture_type.lower():
                    commands.append(VLACommand(
                        action="stop",
                        target=None,
                        confidence=0.95,
                        reasoning="Stop gesture detected"
                    ))
                elif "wave" in gesture.gesture_type.lower():
                    commands.append(VLACommand(
                        action="greet",
                        target=None,
                        confidence=0.9,
                        reasoning="Wave gesture detected"
                    ))

        return commands

    def generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate natural language response about the scene."""
        detection_count = len(context.get("detections", []))
        person_count = sum(1 for d in context.get("detections", []) if d.class_name == "person")

        responses = [
            f"I can see {detection_count} objects in view",
            f"There {'is' if person_count == 1 else 'are'} {person_count} person{'s' if person_count != 1 else ''} in frame",
        ]

        if person_count > 0:
            responses.append("I'm tracking the closest person")

        return " | ".join(responses)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def name(self) -> str:
        return "VLA-Model"


class EventCameraProcessor:
    """Process event camera data for enhanced vision."""

    def __init__(
        self,
        threshold: int = 50,
        window_ms: int = 100,
    ):
        self._threshold = threshold
        self._window_ms = window_ms
        self._logger = get_logger(__name__)
        self._event_buffer = []
        self._last_timestamp = 0

    def process_events(self, events: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Process raw events into event frame."""
        current_time = time.time() * 1000

        self._event_buffer.extend(events)

        self._event_buffer = [
            e for e in self._event_buffer
            if current_time - e[0] < self._window_ms
        ]

        if not self._event_buffer:
            return None

        height, width = 480, 640
        event_frame = np.zeros((height, width, 3), dtype=np.uint8)

        for ts, x, y, pol in self._event_buffer[-1000:]:
            if 0 <= x < width and 0 <= y < height:
                if pol > 0:
                    event_frame[y, x] = [255, 0, 0]
                else:
                    event_frame[y, x] = [0, 0, 255]

        return event_frame

    def fuse_with_frame(
        self,
        frame: np.ndarray,
        events: np.ndarray,
        alpha: float = 0.3,
    ) -> np.ndarray:
        """Fuse event data with conventional frame."""
        if events is None:
            return frame

        fused = frame.copy()
        mask = np.any(events > 0, axis=2)
        fused[mask] = (alpha * events[mask] + (1 - alpha) * frame[mask]).astype(np.uint8)

        return fused

    def detect_motion(self, events: List[Tuple[int, int, int, int]]) -> bool:
        """Quick motion detection from events."""
        if len(events) < 10:
            return False

        xs = [e[1] for e in events]
        ys = [e[2] for e in events]

        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        return x_range > 50 or y_range > 50

    def get_motion_direction(self, events: List[Tuple[int, int, int, int]]) -> Optional[str]:
        """Determine motion direction from events."""
        if len(events) < 20:
            return None

        xs = [e[1] for e in events]
        ys = [e[2] for e in events]

        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)

        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        if x_range > y_range * 2:
            if x_mean < 320:
                return "left_to_right"
            else:
                return "right_to_left"
        elif y_range > x_range * 2:
            if y_mean < 240:
                return "top_to_bottom"
            else:
                return "bottom_to_top"

        return "static"


class AdvancedAI:
    """Advanced AI features for robot perception."""

    def __init__(self):
        self._logger = get_logger(__name__)
        self._vla = VLAModel()
        self._event_camera = EventCameraProcessor()
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize all advanced AI components."""
        try:
            self._vla.load()
            self._is_initialized = True
            self._logger.info("Advanced AI initialized")
        except Exception as e:
            self._logger.warning(f"Advanced AI initialization incomplete: {e}")

    def process_scene(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process scene with all AI capabilities."""
        if not self._is_initialized:
            self.initialize()

        result = {
            "vla_commands": [],
            "motion_detected": False,
            "scene_description": "",
        }

        if self._vla.is_loaded:
            result["vla_commands"] = self._vla.process(frame, detections, context)

            if context:
                result["scene_description"] = self._vla.generate_response(
                    "describe", {"detections": detections}
                )

        return result

    @property
    def vla(self) -> VLAModel:
        return self._vla

    @property
    def event_camera(self) -> EventCameraProcessor:
        return self._event_camera
