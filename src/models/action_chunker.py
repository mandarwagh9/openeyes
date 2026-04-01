"""Action chunking and real-time control for VLA models.

Provides:
- Action chunking for continuous control
- Temporal aggregation for smooth motion
- Action smoothing and interpolation
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class ActionChunk:
    """A chunk of actions predicted by VLA."""
    actions: np.ndarray
    timestamps: List[float]
    confidences: List[float]
    instruction: str


class ActionChunker:
    """Chunk VLA actions for real-time control.
    
    Instead of predicting one action at a time, predicts chunks of
    actions that can be executed over multiple time steps.
    
    Benefits:
    - Reduces inference frequency (10-30 Hz control from 1-5 Hz inference)
    - Smooths jerky motion
    - Enables temporal planning
    """
    
    def __init__(
        self,
        chunk_size: int = 10,
        execution_rate: float = 20.0,
        smoothing: float = 0.5,
        use_temporal_aggregation: bool = True,
    ):
        self._chunk_size = chunk_size
        self._execution_rate = execution_rate
        self._smoothing = smoothing
        self._use_temporal_aggregation = use_temporal_aggregation
        
        self._action_queue: deque = deque(maxlen=100)
        self._last_actions: Optional[np.ndarray] = None
        self._chunk_history: List[ActionChunk] = []
        self._max_history = 5
        
    def add_action(self, action: np.ndarray, confidence: float = 1.0) -> None:
        """Add a single action to the queue.
        
        Args:
            action: Action vector (e.g., 7-DoF for robot arm)
            confidence: Confidence score for the action
        """
        timestamp = time.time()
        self._action_queue.append({
            "action": action.copy(),
            "confidence": confidence,
            "timestamp": timestamp
        })
        
        if self._last_actions is not None and self._smoothing > 0:
            smoothed = (1 - self._smoothing) * action + self._smoothing * self._last_actions
            self._last_actions = smoothed
        else:
            self._last_actions = action.copy()
    
    def get_next_action(self) -> Optional[np.ndarray]:
        """Get the next action to execute.
        
        Returns:
            Next action vector, or None if queue is empty
        """
        if not self._action_queue:
            return None
        
        entry = self._action_queue.popleft()
        return entry["action"]
    
    def get_action_batch(self, batch_size: int) -> List[np.ndarray]:
        """Get a batch of actions for batch processing.
        
        Args:
            batch_size: Number of actions to retrieve
            
        Returns:
            List of action vectors
        """
        actions = []
        for _ in range(min(batch_size, len(self._action_queue))):
            action = self.get_next_action()
            if action is not None:
                actions.append(action)
        return actions
    
    def has_actions(self) -> bool:
        """Check if there are actions in the queue."""
        return len(self._action_queue) > 0
    
    def queue_size(self) -> int:
        """Get current queue size."""
        return len(self._action_queue)
    
    def clear_queue(self) -> None:
        """Clear all pending actions."""
        self._action_queue.clear()
        self._last_actions = None
    
    @property
    def execution_rate(self) -> float:
        """Get execution rate in Hz."""
        return self._execution_rate
    
    @execution_rate.setter
    def execution_rate(self, rate: float) -> None:
        """Set execution rate in Hz."""
        self._execution_rate = max(1.0, min(100.0, rate))
    
    @property
    def chunk_size(self) -> int:
        """Get chunk size."""
        return self._chunk_size


class TemporalActionAggregator:
    """Aggregate actions over time for smoother control.
    
    Applies temporal smoothing and predicts future actions
    based on historical patterns.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        prediction_horizon: int = 3,
        use_motion_model: bool = True,
    ):
        self._window_size = window_size
        self._prediction_horizon = prediction_horizon
        self._use_motion_model = use_motion_model
        
        self._action_history: deque = deque(maxlen=window_size)
        self._velocity_history: deque = deque(maxlen=window_size)
        
    def add_action(self, action: np.ndarray) -> None:
        """Add action to history."""
        self._action_history.append(action.copy())
        
        if len(self._action_history) >= 2:
            velocity = self._action_history[-1] - self._action_history[-2]
            self._velocity_history.append(velocity)
    
    def get_smoothed_action(self, alpha: float = 0.7) -> Optional[np.ndarray]:
        """Get smoothed action using exponential moving average.
        
        Args:
            alpha: Smoothing factor (higher = more smoothing)
            
        Returns:
            Smoothed action vector
        """
        if not self._action_history:
            return None
        
        if len(self._action_history) == 1:
            return self._action_history[0].copy()
        
        smoothed = self._action_history[-1].copy()
        for i in range(len(self._action_history) - 2, -1, -1):
            smoothed = alpha * smoothed + (1 - alpha) * self._action_history[i]
        
        return smoothed
    
    def predict_future_actions(self) -> List[np.ndarray]:
        """Predict future actions based on motion model.
        
        Returns:
            List of predicted future actions
        """
        if not self._velocity_history:
            return []
        
        predictions = []
        avg_velocity = np.mean(list(self._velocity_history), axis=0)
        
        last_action = self._action_history[-1]
        
        for i in range(1, self._prediction_horizon + 1):
            predicted = last_action + avg_velocity * i * (1.0 / self._execution_rate)
            predictions.append(predicted)
        
        return predictions
    
    def clear_history(self) -> None:
        """Clear action and velocity history."""
        self._action_history.clear()
        self._velocity_history.clear()


class RealTimeVLAController:
    """Real-time VLA controller with action chunking.
    
    Combines:
    - ActionChunker for inference batching
    - TemporalActionAggregator for smoothing
    - Optional velocity clamping for safety
    """
    
    def __init__(
        self,
        chunk_size: int = 10,
        execution_rate: float = 20.0,
        smoothing: float = 0.5,
        velocity_limits: Optional[Dict[str, float]] = None,
        enable_prediction: bool = True,
    ):
        self._chunker = ActionChunker(
            chunk_size=chunk_size,
            execution_rate=execution_rate,
            smoothing=smoothing,
        )
        
        self._aggregator = TemporalActionAggregator(
            window_size=5,
            prediction_horizon=3,
        )
        
        self._velocity_limits = velocity_limits or {
            "linear": 0.5,
            "angular": 1.0,
            "gripper": 1.0,
        }
        
        self._enable_prediction = enable_prediction
        self._control_frequency = execution_rate
        self._last_update = time.time()
        
    def process_vla_output(
        self,
        action: np.ndarray,
        confidence: float = 1.0
    ) -> None:
        """Process VLA output into action queue.
        
        Args:
            action: Raw VLA action output
            confidence: VLA confidence score
        """
        clamped_action = self._clamp_velocity(action)
        
        self._chunker.add_action(clamped_action, confidence)
        self._aggregator.add_action(clamped_action)
    
    def _clamp_velocity(self, action: np.ndarray) -> np.ndarray:
        """Clamp action velocities for safety."""
        if len(action) < 7:
            return action
        
        clamped = action.copy()
        
        clamped[0] = np.clip(clamped[0], -self._velocity_limits["linear"], self._velocity_limits["linear"])
        clamped[1] = np.clip(clamped[1], -self._velocity_limits["linear"], self._velocity_limits["linear"])
        clamped[2] = np.clip(clamped[2], -self._velocity_limits["linear"], self._velocity_limits["linear"])
        clamped[3] = np.clip(clamped[3], -self._velocity_limits["angular"], self._velocity_limits["angular"])
        clamped[4] = np.clip(clamped[4], -self._velocity_limits["angular"], self._velocity_limits["angular"])
        clamped[5] = np.clip(clamped[5], -self._velocity_limits["angular"], self._velocity_limits["angular"])
        clamped[6] = np.clip(clamped[6], -self._velocity_limits["gripper"], self._velocity_limits["gripper"])
        
        return clamped
    
    def get_action(self) -> Optional[np.ndarray]:
        """Get next action for execution."""
        return self._chunker.get_next_action()
    
    def get_smoothed_action(self) -> Optional[np.ndarray]:
        """Get temporally smoothed action."""
        return self._aggregator.get_smoothed_action()
    
    def get_predicted_actions(self) -> List[np.ndarray]:
        """Get predicted future actions."""
        if not self._enable_prediction:
            return []
        return self._aggregator.predict_future_actions()
    
    def is_ready(self) -> bool:
        """Check if controller is ready (has actions)."""
        return self._chunker.has_actions()
    
    def reset(self) -> None:
        """Reset controller state."""
        self._chunker.clear_queue()
        self._aggregator.clear_history()
        self._last_update = time.time()
    
    @property
    def queue_size(self) -> int:
        """Get action queue size."""
        return self._chunker.queue_size()
    
    @property
    def control_frequency(self) -> float:
        """Get control frequency in Hz."""
        return self._control_frequency
    
    @control_frequency.setter
    def control_frequency(self, freq: float) -> None:
        """Set control frequency in Hz."""
        self._control_frequency = max(1.0, min(100.0, freq))
        self._chunker.execution_rate = freq


def create_action_chunker(
    chunk_size: int = 10,
    execution_rate: float = 20.0,
    smoothing: float = 0.5,
    enable_prediction: bool = True,
) -> RealTimeVLAController:
    """Factory function to create action chunker.
    
    Args:
        chunk_size: Number of actions to chunk
        execution_rate: Control frequency in Hz
        smoothing: Action smoothing factor
        enable_prediction: Enable future action prediction
        
    Returns:
        RealTimeVLAController instance
    """
    return RealTimeVLAController(
        chunk_size=chunk_size,
        execution_rate=execution_rate,
        smoothing=smoothing,
        enable_prediction=enable_prediction,
    )