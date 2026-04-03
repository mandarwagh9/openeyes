import pytest
import numpy as np
from typing import Tuple

from src.world_model.types import (
    PredictedBBox,
    Prediction,
    Plan,
    WorldModelState,
)
from src.world_model.base import WorldModel
from src.world_model.planner import CEMPlanner, ACTIONS
from src.world_model.lewm import LeWorldModel
from src.world_model.safety_evaluator import SafetyEvaluator, SafetyResult


class TestPredictedBBox:
    def test_centroid(self):
        bbox = PredictedBBox(step=1, x1=10, y1=20, x2=110, y2=120)
        cx, cy = bbox.centroid
        assert cx == 60.0
        assert cy == 70.0

    def test_width_height(self):
        bbox = PredictedBBox(step=1, x1=10, y1=20, x2=110, y2=120)
        assert bbox.width == 100.0
        assert bbox.height == 100.0

    def test_to_dict(self):
        bbox = PredictedBBox(step=2, x1=0, y1=0, x2=50, y2=50, confidence=0.9)
        d = bbox.to_dict()
        assert d["step"] == 2
        assert d["x1"] == 0
        assert d["confidence"] == 0.9


class TestPrediction:
    def test_get_position_at_step(self):
        positions = [
            PredictedBBox(step=1, x1=10, y1=10, x2=50, y2=50),
            PredictedBBox(step=2, x1=20, y1=20, x2=60, y2=60),
        ]
        pred = Prediction(track_id=1, class_name="person", positions=positions)
        pos = pred.get_position_at_step(2)
        assert pos is not None
        assert pos.x1 == 20

    def test_get_position_at_step_not_found(self):
        positions = [PredictedBBox(step=1, x1=10, y1=10, x2=50, y2=50)]
        pred = Prediction(track_id=1, class_name="person", positions=positions)
        assert pred.get_position_at_step(5) is None

    def test_get_next_position(self):
        positions = [
            PredictedBBox(step=1, x1=10, y1=10, x2=50, y2=50),
            PredictedBBox(step=2, x1=20, y1=20, x2=60, y2=60),
        ]
        pred = Prediction(track_id=1, class_name="person", positions=positions)
        next_pos = pred.get_next_position()
        assert next_pos is not None
        assert next_pos.step == 1

    def test_get_next_position_empty(self):
        pred = Prediction(track_id=1, class_name="person", positions=[])
        assert pred.get_next_position() is None

    def test_to_dict(self):
        positions = [PredictedBBox(step=1, x1=0, y1=0, x2=50, y2=50)]
        pred = Prediction(track_id=1, class_name="person", positions=positions, confidence=0.85)
        d = pred.to_dict()
        assert d["track_id"] == 1
        assert d["class_name"] == "person"
        assert len(d["positions"]) == 1
        assert d["confidence"] == 0.85


class TestPlan:
    def test_get_next_action(self):
        plan = Plan(actions=["forward", "left", "stop"], expected_states=[])
        assert plan.get_next_action() == "forward"

    def test_get_next_action_empty(self):
        plan = Plan(actions=[], expected_states=[])
        assert plan.get_next_action() is None

    def test_to_dict(self):
        plan = Plan(actions=["forward", "stop"], expected_states=[], confidence=0.9, horizon=2)
        d = plan.to_dict()
        assert d["actions"] == ["forward", "stop"]
        assert d["confidence"] == 0.9
        assert d["horizon"] == 2


class TestWorldModelState:
    def test_creation(self):
        latent = np.zeros(384, dtype=np.float32)
        state = WorldModelState(latent=latent, frame_id=0)
        assert state.frame_id == 0
        assert state.last_action is None

    def test_with_action(self):
        latent = np.zeros(384, dtype=np.float32)
        state = WorldModelState(latent=latent, frame_id=1, last_action="forward")
        assert state.last_action == "forward"


class TestCEMPlanner:
    def test_init_defaults(self):
        planner = CEMPlanner()
        assert planner.num_samples == 100
        assert planner.num_elites == 10
        assert planner.horizon == 10
        assert planner.num_iterations == 3
        assert planner.actions == ACTIONS

    def test_init_custom(self):
        planner = CEMPlanner(
            actions=["a", "b"],
            num_samples=50,
            num_elites=5,
            horizon=5,
            num_iterations=2,
        )
        assert planner.num_samples == 50
        assert planner.horizon == 5
        assert planner.actions == ["a", "b"]

    def test_plan_returns_plan(self):
        planner = CEMPlanner(num_samples=20, num_elites=5, num_iterations=2)

        def dummy_predict(latent, action):
            if action == "forward":
                return latent + 0.1
            return latent

        current = np.zeros(64, dtype=np.float32)
        goal = np.ones(64, dtype=np.float32) * 5.0

        plan = planner.plan(current, goal, dummy_predict, horizon=5)

        assert isinstance(plan, Plan)
        assert len(plan.actions) == 5
        assert len(plan.expected_states) == 5

    def test_plan_favors_goal_direction(self):
        planner = CEMPlanner(num_samples=50, num_elites=10, num_iterations=5)

        def dummy_predict(latent, action):
            delta = {"forward": 1.0, "backward": -1.0, "left": 0.0, "right": 0.0, "stop": 0.0}
            return latent + delta.get(action, 0.0)

        current = np.zeros(16, dtype=np.float32)
        goal = np.ones(16, dtype=np.float32) * 10.0

        plan = planner.plan(current, goal, dummy_predict, horizon=3)

        assert isinstance(plan, Plan)
        assert plan.confidence >= 0.0

    def test_reset(self):
        planner = CEMPlanner(num_samples=20)
        current = np.zeros(32, dtype=np.float32)
        goal = np.ones(32, dtype=np.float32)
        planner.plan(current, goal, lambda l, a: l)

        planner.reset()
        assert planner.get_last_plan() is None
        assert planner.get_last_planning_time_ms() == 0.0

    def test_planning_time_recorded(self):
        planner = CEMPlanner(num_samples=20, num_iterations=2)
        current = np.zeros(32, dtype=np.float32)
        goal = np.ones(32, dtype=np.float32)
        planner.plan(current, goal, lambda l, a: l)

        assert planner.get_last_planning_time_ms() >= 0.0


class TestLeWorldModel:
    def test_init(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        assert model.latent_dim == 128
        assert model.use_dinov2 is False
        assert not model.is_loaded

    def test_load_without_dinov2(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()
        assert model.is_loaded
        assert model._encoder is None

    def test_encode_simple(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        latent = model.encode(frame)

        assert latent.shape[0] == 128
        assert latent.dtype == np.float32

    def test_predict(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        latent = np.random.randn(128).astype(np.float32)
        next_latent = model.predict(latent, action="forward")

        assert next_latent.shape == (128,)
        assert next_latent.dtype == np.float32

    def test_predict_trajectory(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        latent = np.random.randn(128).astype(np.float32)
        trajectory = model.predict_trajectory(latent, horizon=5)

        assert len(trajectory) == 5
        assert all(t.shape == (128,) for t in trajectory)

    def test_plan(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        current = np.random.randn(128).astype(np.float32)
        goal = np.random.randn(128).astype(np.float32)

        plan = model.plan(current, goal, horizon=5, num_samples=20)

        assert isinstance(plan, Plan)
        assert len(plan.actions) == 5

    def test_predict_bbox_trajectory(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        pred = model.predict_bbox_trajectory(
            track_id=1,
            class_name="person",
            current_bbox=(100, 50, 200, 300),
            frame_shape=(640, 480),
            horizon=5,
        )

        assert pred.track_id == 1
        assert pred.class_name == "person"
        assert len(pred.positions) == 5
        assert all(0 <= p.x1 < 640 for p in pred.positions)
        assert all(0 <= p.y1 < 480 for p in pred.positions)

    def test_predict_bbox_trajectory_with_action(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        for i in range(10):
            latent = np.random.randn(128).astype(np.float32) * 0.1
            model.record_state(latent, "forward")

        pred_forward = model.predict_bbox_trajectory(
            track_id=1,
            class_name="person",
            current_bbox=(100, 50, 200, 300),
            frame_shape=(640, 480),
            horizon=5,
            action="forward",
        )

        pred_left = model.predict_bbox_trajectory(
            track_id=1,
            class_name="person",
            current_bbox=(100, 50, 200, 300),
            frame_shape=(640, 480),
            horizon=5,
            action="left",
        )

        assert len(pred_forward.positions) == 5
        assert len(pred_left.positions) == 5
        assert all(p.confidence > 0 for p in pred_forward.positions)
        assert all(0 <= p.x1 < 640 for p in pred_left.positions)
        assert all(0 <= p.y1 < 480 for p in pred_left.positions)

    def test_update_transition_model(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        state_t = np.random.randn(128).astype(np.float32)
        state_t1 = np.random.randn(128).astype(np.float32)

        model.update_transition_model(state_t, state_t1, action="forward")

        assert model._transition_weights is not None
        assert model._transition_bias is not None

    def test_record_state_and_learn(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        for i in range(35):
            latent = np.random.randn(128).astype(np.float32)
            action = ACTIONS[i % len(ACTIONS)]
            model.record_state(latent, action)

        assert len(model._state_history) <= 15

    def test_get_info(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        info = model.get_info()
        assert info["name"] == "LeWorldModel"
        assert info["params"] == 15_000_000
        assert info["latent_dim"] == 128
        assert info["device"] == "cpu"
        assert info["dinov2"] is False

    def test_reset(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        latent = np.random.randn(128).astype(np.float32)
        model.record_state(latent, "forward")
        assert len(model._state_history) > 0

        model.reset()
        assert len(model._state_history) == 0

    def test_encode_without_load_raises(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="not loaded"):
            model.encode(frame)

    def test_predict_without_load_raises(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        latent = np.zeros(128, dtype=np.float32)

        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict(latent)


class TestSafetyEvaluator:
    def test_init(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        evaluator = SafetyEvaluator(
            world_model=model,
            min_safe_distance=0.5,
            max_risk_level=0.8,
            prediction_horizon=5,
        )

        assert evaluator._min_safe_distance == 0.5
        assert evaluator._max_risk_level == 0.8
        assert evaluator._horizon == 5

    def test_evaluate_action_safe(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        evaluator = SafetyEvaluator(
            world_model=model,
            min_safe_distance=0.1,
            max_risk_level=0.9,
        )

        current = np.zeros(128, dtype=np.float32)
        result = evaluator.evaluate_action(current, "stop")

        assert isinstance(result, SafetyResult)
        assert result.is_safe

    def test_evaluate_action_with_obstacles(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        evaluator = SafetyEvaluator(
            world_model=model,
            min_safe_distance=0.01,
            max_risk_level=0.95,
        )

        current = np.zeros(128, dtype=np.float32)
        obstacles = [np.ones(128, dtype=np.float32) * 10.0]

        result = evaluator.evaluate_action(current, "forward", obstacles)

        assert isinstance(result, SafetyResult)

    def test_evaluate_action_sequence(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        evaluator = SafetyEvaluator(world_model=model)

        current = np.zeros(128, dtype=np.float32)
        result = evaluator.evaluate_action_sequence(
            current, ["forward", "left", "stop"]
        )

        assert isinstance(result, SafetyResult)

    def test_get_safe_actions(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        evaluator = SafetyEvaluator(world_model=model)

        current = np.zeros(128, dtype=np.float32)
        safe = evaluator.get_safe_actions(current)

        assert isinstance(safe, list)
        assert len(safe) > 0

    def test_safety_result_to_dict(self):
        result = SafetyResult(
            is_safe=True,
            risk_level=0.3,
            reason="Test",
            predicted_collisions=0,
            min_predicted_distance=1.5,
        )

        d = result.to_dict()
        assert d["is_safe"] is True
        assert d["risk_level"] == 0.3
        assert d["reason"] == "Test"

    def test_reset(self):
        model = LeWorldModel(device="cpu", latent_dim=128, use_dinov2=False)
        model.load()

        evaluator = SafetyEvaluator(world_model=model)
        evaluator._obstacle_latents = [np.zeros(128)]
        evaluator._current_latent = np.zeros(128)

        evaluator.reset()
        assert len(evaluator._obstacle_latents) == 0
        assert evaluator._current_latent is None
