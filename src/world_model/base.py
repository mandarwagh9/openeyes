"""Abstract world model interface.

All world model implementations must extend this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np

from src.world_model.types import Prediction, Plan, WorldModelState


class WorldModel(ABC):
    """Abstract world model for predictive tracking and planning.

    World models learn internal representations of environmental dynamics,
    enabling prediction of future states and planning in latent space.

    Subclasses must implement:
    - encode(): Convert frame to latent state
    - predict(): Predict next latent state given action
    - plan(): Generate action sequence to reach goal
    """

    def __init__(
        self,
        device: str = "cuda",
        precision: str = "fp16",
    ):
        self.device = device
        self.precision = precision
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights and prepare for inference.

        Must set self._is_loaded = True on success.
        """
        ...

    @abstractmethod
    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Encode a single frame to latent state.

        Args:
            frame: Input frame as BGR numpy array (H, W, 3)

        Returns:
            Latent state vector (embedding dimension depends on model)
        """
        ...

    @abstractmethod
    def predict(
        self,
        latent: np.ndarray,
        action: Optional[str] = None
    ) -> np.ndarray:
        """Predict next latent state given current state and optional action.

        Args:
            latent: Current latent state
            action: Optional action string (e.g., "forward", "left")

        Returns:
            Predicted next latent state
        """
        ...

    @abstractmethod
    def predict_trajectory(
        self,
        latent: np.ndarray,
        actions: Optional[List[str]] = None,
        horizon: int = 10
    ) -> List[np.ndarray]:
        """Predict trajectory of latent states over horizon.

        Args:
            latent: Starting latent state
            actions: Optional sequence of actions (if None, predict forward)
            horizon: Number of steps to predict

        Returns:
            List of predicted latent states for each step
        """
        ...

    @abstractmethod
    def plan(
        self,
        current_latent: np.ndarray,
        goal_latent: np.ndarray,
        horizon: int = 10,
        num_samples: int = 100
    ) -> Plan:
        """Plan action sequence to reach goal latent state.

        Uses Cross-Entropy Method (CEM) for optimization in latent space.

        Args:
            current_latent: Current latent state
            goal_latent: Target latent state to reach
            horizon: Planning horizon (number of steps)
            num_samples: Number of action sequences to sample

        Returns:
            Plan containing best action sequence and expected states
        """
        ...

    def predict_bbox_trajectory(
        self,
        track_id: int,
        class_name: str,
        current_bbox: Tuple[float, float, float, float],
        frame_shape: Tuple[int, int],
        horizon: int = 10,
        action: Optional[str] = None
    ) -> Prediction:
        """Predict bounding box trajectory for a tracked object.

        Default implementation uses simple linear extrapolation.
        Subclasses can override with learned dynamics.

        Args:
            track_id: Track ID of the object
            class_name: Object class name
            current_bbox: (x1, y1, x2, y2) of current bounding box
            frame_shape: (width, height) of the frame
            horizon: Number of future steps to predict
            action: Optional action affecting prediction

        Returns:
            Prediction object with predicted bounding boxes
        """
        from src.world_model.types import PredictedBBox

        x1, y1, x2, y2 = current_bbox
        w, h = frame_shape

        dx = (x2 - x1) * 0.02
        dy = (y2 - y1) * 0.01

        if action == "forward":
            dy *= 1.5
        elif action == "backward":
            dy *= -1.0
        elif action == "left":
            dx -= abs(dx) * 0.5
        elif action == "right":
            dx += abs(dx) * 0.5

        positions = []
        for step in range(1, horizon + 1):
            px1 = max(0, min(w - 1, x1 + dx * step))
            py1 = max(0, min(h - 1, y1 + dy * step))
            px2 = max(0, min(w - 1, x2 + dx * step))
            py2 = max(0, min(h - 1, y2 + dy * step))

            conf = max(0.0, 1.0 - step * 0.05)

            positions.append(PredictedBBox(
                step=step,
                x1=px1,
                y1=py1,
                x2=px2,
                y2=py2,
                confidence=conf,
            ))

        return Prediction(
            track_id=track_id,
            class_name=class_name,
            positions=positions,
            confidence=0.9,
        )

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def get_info(self) -> dict:
        """Return model information (name, params, latency, etc.)."""
        ...
