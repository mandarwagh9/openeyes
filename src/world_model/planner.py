"""Cross-Entropy Method (CEM) planner for latent-space planning.

Implements model-predictive control using CEM optimization
over action sequences in the world model's latent space.
"""

from typing import List, Optional, Tuple
import numpy as np
import time

from src.world_model.types import Plan
from src.utils.logger import get_logger


ACTIONS = ["forward", "backward", "left", "right", "stop"]


class CEMPlanner:
    """Cross-Entropy Method planner for world model-based planning.

    Samples action sequences, rolls them out through the world model,
    and iteratively refines the distribution toward high-reward sequences.

    Performance target: <5ms for 100 samples, horizon=10 on Jetson Orin Nano.
    """

    def __init__(
        self,
        actions: Optional[List[str]] = None,
        num_samples: int = 100,
        num_elites: int = 10,
        horizon: int = 10,
        num_iterations: int = 3,
    ):
        self._logger = get_logger(__name__)
        self.actions = actions or ACTIONS
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.horizon = horizon
        self.num_iterations = num_iterations

        self._action_probs = np.ones(len(self.actions)) / len(self.actions)
        self._last_plan: Optional[Plan] = None
        self._last_plan_time: float = 0.0

    def plan(
        self,
        current_latent: np.ndarray,
        goal_latent: np.ndarray,
        predict_fn,
        horizon: Optional[int] = None,
        num_samples: Optional[int] = None,
    ) -> Plan:
        """Plan action sequence using CEM optimization.

        Args:
            current_latent: Current latent state (embedding vector)
            goal_latent: Target latent state to reach
            predict_fn: Function(latent, action) -> next_latent
            horizon: Override planning horizon
            num_samples: Override sample count

        Returns:
            Plan with best action sequence and expected states
        """
        start_time = time.perf_counter()

        h = horizon or self.horizon
        n = num_samples or self.num_samples

        action_probs = np.ones(len(self.actions)) / len(self.actions)
        best_plan = None
        best_score = float("inf")

        for iteration in range(self.num_iterations):
            action_sequences = self._sample_sequences(n, h, action_probs)

            scores = []
            for seq in action_sequences:
                score = self._evaluate_sequence(
                    current_latent, goal_latent, seq, predict_fn
                )
                scores.append(score)

            scores_np = np.array(scores)
            elite_indices = np.argsort(scores_np)[:self.num_elites]
            elite_sequences = [action_sequences[i] for i in elite_indices]

            action_probs = self._update_probs(elite_sequences)

            best_idx = int(np.argmin(scores_np))
            if scores_np[best_idx] < best_score:
                best_score = float(scores_np[best_idx])
                best_plan = Plan(
                    actions=list(action_sequences[best_idx]),
                    expected_states=self._rollout(
                        current_latent, action_sequences[best_idx], predict_fn
                    ),
                    confidence=max(0.0, 1.0 - best_score),
                    horizon=h,
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._last_plan = best_plan
        self._last_plan_time = elapsed_ms

        self._logger.debug(
            f"CEM planning: {elapsed_ms:.1f}ms, "
            f"score={best_score:.4f}, "
            f"actions={best_plan.actions[:3] if best_plan else 'none'}..."
        )

        return best_plan or Plan(actions=["stop"], expected_states=[], horizon=h)

    def _sample_sequences(
        self,
        num_samples: int,
        horizon: int,
        probs: np.ndarray
    ) -> List[List[str]]:
        """Sample action sequences from categorical distribution."""
        sequences = []
        for _ in range(num_samples):
            indices = np.random.choice(
                len(self.actions),
                size=horizon,
                p=probs,
            )
            seq = [self.actions[i] for i in indices]
            sequences.append(seq)
        return sequences

    def _evaluate_sequence(
        self,
        start_latent: np.ndarray,
        goal_latent: np.ndarray,
        actions: List[str],
        predict_fn,
    ) -> float:
        """Evaluate an action sequence by rolling out and scoring against goal."""
        latent = start_latent.copy()
        total_cost = 0.0

        for i, action in enumerate(actions):
            latent = predict_fn(latent, action)
            distance = np.linalg.norm(latent - goal_latent)
            total_cost += distance * (1.0 + i * 0.1)

        return float(total_cost)

    def _rollout(
        self,
        start_latent: np.ndarray,
        actions: List[str],
        predict_fn,
    ) -> List[np.ndarray]:
        """Roll out action sequence to get expected states."""
        states = []
        latent = start_latent.copy()
        for action in actions:
            latent = predict_fn(latent, action)
            states.append(latent.copy())
        return states

    def _update_probs(self, elite_sequences: List[List[str]]) -> np.ndarray:
        """Update action probabilities from elite sequences."""
        counts = np.zeros(len(self.actions))
        for seq in elite_sequences:
            for action in seq:
                idx = self.actions.index(action)
                counts[idx] += 1

        total = counts.sum()
        if total == 0:
            return np.ones(len(self.actions)) / len(self.actions)

        probs = counts / total
        smoothing = 0.1
        probs = (1 - smoothing) * probs + smoothing / len(self.actions)
        return probs / probs.sum()

    def get_last_plan(self) -> Optional[Plan]:
        return self._last_plan

    def get_last_planning_time_ms(self) -> float:
        return float(self._last_plan_time)

    def reset(self) -> None:
        self._action_probs = np.ones(len(self.actions)) / len(self.actions)
        self._last_plan = None
        self._last_plan_time = 0.0
