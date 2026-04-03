"""Predictive safety evaluator using world model predictions.

Evaluates actions for safety before execution by simulating
their outcomes in the world model's latent space.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from src.world_model.base import WorldModel
from src.world_model.types import Prediction
from src.utils.logger import get_logger


@dataclass
class SafetyResult:
    """Result of a safety evaluation."""
    is_safe: bool
    risk_level: float
    reason: str
    predicted_collisions: int = 0
    min_predicted_distance: float = float("inf")
    unsafe_actions: List[str] = None

    def __post_init__(self):
        if self.unsafe_actions is None:
            self.unsafe_actions = []

    def to_dict(self) -> dict:
        return {
            "is_safe": self.is_safe,
            "risk_level": self.risk_level,
            "reason": self.reason,
            "predicted_collisions": self.predicted_collisions,
            "min_predicted_distance": self.min_predicted_distance,
            "unsafe_actions": self.unsafe_actions,
        }


class SafetyEvaluator:
    """Evaluates action safety using world model predictions.

    Before executing an action, simulates its outcome using the
    world model and checks for potential collisions or unsafe states.
    """

    def __init__(
        self,
        world_model: WorldModel,
        min_safe_distance: float = 0.3,
        max_risk_level: float = 0.7,
        prediction_horizon: int = 10,
    ):
        self._logger = get_logger(__name__)
        self._wm = world_model
        self._min_safe_distance = min_safe_distance
        self._max_risk_level = max_risk_level
        self._horizon = prediction_horizon

        self._obstacle_latents: List[np.ndarray] = []
        self._current_latent: Optional[np.ndarray] = None

    def evaluate_action(
        self,
        current_latent: np.ndarray,
        action: str,
        obstacle_latents: Optional[List[np.ndarray]] = None,
    ) -> SafetyResult:
        """Evaluate if an action is safe to execute.

        Args:
            current_latent: Current world state latent
            action: Action to evaluate
            obstacle_latents: Latents representing nearby obstacles/objects

        Returns:
            SafetyResult with safety assessment
        """
        self._current_latent = current_latent
        if obstacle_latents:
            self._obstacle_latents = obstacle_latents

        predicted_states = self._wm.predict_trajectory(
            current_latent,
            actions=[action] * self._horizon,
            horizon=self._horizon,
        )

        collisions = 0
        min_distance = float("inf")
        unsafe_actions = []

        for state in predicted_states:
            for obs_latent in self._obstacle_latents:
                dist = np.linalg.norm(state - obs_latent)
                min_distance = min(min_distance, float(dist))

                if dist < self._min_safe_distance:
                    collisions += 1
                    if action not in unsafe_actions:
                        unsafe_actions.append(action)

        risk = min(1.0, collisions * 0.3 + max(0, 1.0 - min_distance * 2))

        if risk > self._max_risk_level:
            return SafetyResult(
                is_safe=False,
                risk_level=risk,
                reason=f"Action '{action}' predicted {collisions} collision(s), "
                       f"min_distance={min_distance:.3f}",
                predicted_collisions=collisions,
                min_predicted_distance=min_distance,
                unsafe_actions=unsafe_actions,
            )

        return SafetyResult(
            is_safe=True,
            risk_level=risk,
            reason=f"Action '{action}' is safe (risk={risk:.3f})",
            predicted_collisions=collisions,
            min_predicted_distance=min_distance,
        )

    def evaluate_action_sequence(
        self,
        current_latent: np.ndarray,
        actions: List[str],
        obstacle_latents: Optional[List[np.ndarray]] = None,
    ) -> SafetyResult:
        """Evaluate a sequence of actions for safety."""
        self._current_latent = current_latent
        if obstacle_latents:
            self._obstacle_latents = obstacle_latents

        predicted_states = self._wm.predict_trajectory(
            current_latent,
            actions=actions,
            horizon=len(actions),
        )

        collisions = 0
        min_distance = float("inf")
        unsafe_actions = []

        for i, state in enumerate(predicted_states):
            for obs_latent in self._obstacle_latents:
                dist = np.linalg.norm(state - obs_latent)
                min_distance = min(min_distance, float(dist))

                if dist < self._min_safe_distance:
                    collisions += 1
                    action = actions[i] if i < len(actions) else "unknown"
                    if action not in unsafe_actions:
                        unsafe_actions.append(action)

        risk = min(1.0, collisions * 0.2 + max(0, 1.0 - min_distance * 2))

        if risk > self._max_risk_level:
            return SafetyResult(
                is_safe=False,
                risk_level=risk,
                reason=f"Sequence has {collisions} predicted collision(s)",
                predicted_collisions=collisions,
                min_predicted_distance=min_distance,
                unsafe_actions=unsafe_actions,
            )

        return SafetyResult(
            is_safe=True,
            risk_level=risk,
            reason=f"Sequence is safe (risk={risk:.3f})",
            predicted_collisions=collisions,
            min_predicted_distance=min_distance,
        )

    def get_safe_actions(
        self,
        current_latent: np.ndarray,
        candidate_actions: Optional[List[str]] = None,
        obstacle_latents: Optional[List[np.ndarray]] = None,
    ) -> List[str]:
        """Get list of safe actions from candidates."""
        candidates = candidate_actions or ["forward", "left", "right", "stop"]
        safe = []

        for action in candidates:
            result = self.evaluate_action(
                current_latent, action, obstacle_latents
            )
            if result.is_safe:
                safe.append(action)

        if not safe:
            safe.append("stop")

        return safe

    def update_obstacle_latents(
        self,
        predictions: List[Prediction],
        current_latent: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> List[np.ndarray]:
        """Convert predictions to obstacle latents for safety evaluation.

        Args:
            predictions: Predicted object positions from world model
            current_latent: Current robot state latent
            frame_shape: (width, height) of the frame

        Returns:
            List of obstacle latent vectors
        """
        obstacles = []

        for pred in predictions:
            next_pos = pred.get_next_position()
            if next_pos is None:
                continue

            cx, cy = next_pos.centroid
            w, h = frame_shape

            obstacle_latent = current_latent.copy()
            obstacle_latent[:2] = np.array([cx / w, cy / h], dtype=np.float32)
            obstacle_latent[2:4] = np.array(
                [next_pos.width / w, next_pos.height / h],
                dtype=np.float32,
            )

            obstacles.append(obstacle_latent)

        return obstacles

    def reset(self) -> None:
        self._obstacle_latents.clear()
        self._current_latent = None
