import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.utils.motor_control import MotorController, MotorCommand


class TestMotorController:
    def test_initialization_default(self):
        controller = MotorController()
        assert controller.max_linear == 0.5
        assert controller.max_angular == 1.0

    def test_initialization_custom(self):
        controller = MotorController(
            max_linear_velocity=0.3,
            max_angular_velocity=0.5
        )
        assert controller.max_linear == 0.3
        assert controller.max_angular == 0.5

    def test_forward_command(self):
        controller = MotorController()
        cmd = controller.command_to_velocity("forward")

        assert cmd.linear_x == 0.5
        assert cmd.angular_z == 0.0
        assert cmd.command == "forward"

    def test_backward_command(self):
        controller = MotorController()
        cmd = controller.command_to_velocity("backward")

        assert cmd.linear_x == -0.5
        assert cmd.angular_z == 0.0
        assert cmd.command == "backward"

    def test_left_command(self):
        controller = MotorController()
        cmd = controller.command_to_velocity("left")

        assert cmd.linear_x == 0.0
        assert cmd.angular_z == 1.0
        assert cmd.command == "left"

    def test_right_command(self):
        controller = MotorController()
        cmd = controller.command_to_velocity("right")

        assert cmd.linear_x == 0.0
        assert cmd.angular_z == -1.0
        assert cmd.command == "right"

    def test_stop_command(self):
        controller = MotorController()
        cmd = controller.command_to_velocity("stop")

        assert cmd.linear_x == 0.0
        assert cmd.angular_z == 0.0
        assert cmd.command == "stop"

    def test_follow_command(self):
        controller = MotorController()
        cmd = controller.command_to_velocity("follow")

        assert cmd.linear_x == 0.25
        assert cmd.angular_z == 0.0
        assert cmd.command == "follow"

    def test_unknown_command_defaults_to_stop(self):
        controller = MotorController()
        cmd = controller.command_to_velocity("spin")

        assert cmd.linear_x == 0.0
        assert cmd.angular_z == 0.0
        assert cmd.command == "stop"

    def test_get_twist_command(self):
        controller = MotorController()
        twist = controller.get_twist_command("forward")

        assert twist["linear"]["x"] == 0.5
        assert twist["angular"]["z"] == 0.0

    def test_apply_smoothing_with_previous(self):
        controller = MotorController()
        target = MotorCommand(0.5, 0.0, "forward")
        previous = MotorCommand(0.0, 0.0, "stop")

        smoothed = controller.apply_smoothing(target, previous, alpha=0.3)

        assert smoothed.linear_x == pytest.approx(0.15, rel=0.01)

    def test_apply_smoothing_without_previous(self):
        controller = MotorController()
        target = MotorCommand(0.5, 0.0, "forward")

        smoothed = controller.apply_smoothing(target, None, alpha=0.3)

        assert smoothed.linear_x == 0.5

    def test_clamp_velocity(self):
        controller = MotorController(max_linear_velocity=0.3)
        linear, angular = controller.clamp_velocity(1.0, 2.0)

        assert linear == 0.3
        assert angular == 1.0

    def test_negative_clamp_velocity(self):
        controller = MotorController(max_linear_velocity=0.3)
        linear, angular = controller.clamp_velocity(-1.0, -2.0)

        assert linear == -0.3
        assert angular == -1.0

    def test_current_command_property(self):
        controller = MotorController()
        controller.command_to_velocity("forward")

        assert controller.current_command == "forward"

    def test_set_limits(self):
        controller = MotorController()
        controller.set_limits(0.3, 0.5)

        cmd = controller.command_to_velocity("forward")
        assert cmd.linear_x == 0.3

    def test_to_tuple(self):
        cmd = MotorCommand(0.5, 1.0, "forward")
        assert cmd.to_tuple() == (0.5, 1.0)


class TestMotorCommand:
    def test_motor_command_creation(self):
        cmd = MotorCommand(0.5, 0.0, "forward")

        assert cmd.linear_x == 0.5
        assert cmd.angular_z == 0.0
        assert cmd.command == "forward"

    def test_motor_command_to_tuple(self):
        cmd = MotorCommand(-0.3, 0.5, "left")
        result = cmd.to_tuple()

        assert result == (-0.3, 0.5)