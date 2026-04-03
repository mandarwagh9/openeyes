"""Data types for world model predictions and planning."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import time


@dataclass
class PredictedBBox:
    """A predicted bounding box at a future timestep."""
    step: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
        }


@dataclass
class Prediction:
    """Prediction for a single tracked object."""
    track_id: int
    class_name: str
    positions: List[PredictedBBox]
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def get_position_at_step(self, step: int) -> Optional[PredictedBBox]:
        for pos in self.positions:
            if pos.step == step:
                return pos
        return None

    def get_next_position(self) -> Optional[PredictedBBox]:
        if not self.positions:
            return None
        return self.positions[0]

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "positions": [p.to_dict() for p in self.positions],
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class Plan:
    """A sequence of actions from world model planning."""
    actions: List[str]
    expected_states: List[np.ndarray]
    confidence: float = 0.0
    horizon: int = 0
    timestamp: float = field(default_factory=time.time)

    def get_next_action(self) -> Optional[str]:
        if not self.actions:
            return None
        return self.actions[0]

    def to_dict(self) -> dict:
        return {
            "actions": self.actions,
            "confidence": self.confidence,
            "horizon": self.horizon,
            "timestamp": self.timestamp,
        }


@dataclass
class WorldModelState:
    """Internal world model state representation."""
    latent: np.ndarray
    frame_id: int
    timestamp: float = field(default_factory=time.time)
    last_action: Optional[str] = None
