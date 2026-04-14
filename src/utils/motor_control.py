import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class MotorCommand:
    linear_x: float
    angular_z: float
    command: str

    def to_tuple(self) -> Tuple[float, float]:
        return (self.linear_x, self.angular_z)


class MotorController:
    def __init__(
        self,
        max_linear_velocity: float = 0.5,
        max_angular_velocity: float = 1.0,
    ):
        self._max_linear_velocity = max_linear_velocity
        self._max_angular_velocity = max_angular_velocity
        self._current_cmd = "stop"

    def command_to_velocity(self, command: str) -> MotorCommand:
        command = command.lower().strip()
        self._current_cmd = command

        if command == "forward":
            return MotorCommand(
                linear_x=self._max_linear_velocity,
                angular_z=0.0,
                command="forward"
            )
        elif command == "backward":
            return MotorCommand(
                linear_x=-self._max_linear_velocity,
                angular_z=0.0,
                command="backward"
            )
        elif command == "left":
            return MotorCommand(
                linear_x=0.0,
                angular_z=self._max_angular_velocity,
                command="left"
            )
        elif command == "right":
            return MotorCommand(
                linear_x=0.0,
                angular_z=-self._max_angular_velocity,
                command="right"
            )
        elif command == "follow":
            return MotorCommand(
                linear_x=self._max_linear_velocity * 0.5,
                angular_z=0.0,
                command="follow"
            )
        elif command == "stop":
            return MotorCommand(
                linear_x=0.0,
                angular_z=0.0,
                command="stop"
            )
        else:
            return MotorCommand(
                linear_x=0.0,
                angular_z=0.0,
                command="stop"
            )

    def get_twist_command(self, command: str) -> dict:
        cmd = self.command_to_velocity(command)
        return {
            "linear": {"x": cmd.linear_x, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": cmd.angular_z},
        }

    def apply_smoothing(
        self,
        target_cmd: MotorCommand,
        previous_cmd: Optional[MotorCommand] = None,
        alpha: float = 0.3,
    ) -> MotorCommand:
        if previous_cmd is None:
            return target_cmd

        smoothed = MotorCommand(
            linear_x=alpha * target_cmd.linear_x + (1 - alpha) * previous_cmd.linear_x,
            angular_z=alpha * target_cmd.angular_z + (1 - alpha) * previous_cmd.angular_z,
            command=target_cmd.command,
        )
        return smoothed

    def clamp_velocity(
        self,
        linear_x: float,
        angular_z: float,
    ) -> Tuple[float, float]:
        linear_x = np.clip(
            linear_x,
            -self._max_linear_velocity,
            self._max_linear_velocity
        )
        angular_z = np.clip(
            angular_z,
            -self._max_angular_velocity,
            self._max_angular_velocity
        )
        return linear_x, angular_z

    @property
    def current_command(self) -> str:
        return self._current_cmd

    @property
    def max_linear(self) -> float:
        return self._max_linear_velocity

    @property
    def max_angular(self) -> float:
        return self._max_angular_velocity

    def set_limits(
        self,
        max_linear: float,
        max_angular: float,
    ) -> None:
        self._max_linear_velocity = max_linear
        self._max_angular_velocity = max_angular