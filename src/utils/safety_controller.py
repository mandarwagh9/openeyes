"""Safety controller for OpenEyes.

Provides:
- Emergency stop functionality
- Safe speed monitoring
- Position limits
- ISO 10218 compliance helpers

Requirements:
    - ROS2 for actual robot control (optional)
"""

import time
import threading
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum
import numpy as np


class SafetyState(Enum):
    """Safety system states."""
    NORMAL = "normal"
    WARNING = "warning"
    STOPPED = "stopped"
    ESTOP = "emergency_stop"


@dataclass
class SafetyLimits:
    """Safety limits for robot motion."""
    max_linear_velocity: float = 0.5
    max_angular_velocity: float = 1.0
    max_acceleration: float = 2.0
    min_distance: float = 0.3
    max_height: float = 2.0
    min_height: float = 0.0
    workspace_radius: float = 2.0


@dataclass
class SafetyStatus:
    """Current safety status."""
    state: SafetyState
    timestamp: float
    current_velocity: float = 0.0
    distance_to_obstacle: float = 0.0
    emergency_reason: Optional[str] = None


class EmergencyStopController:
    """Emergency stop controller for robot safety.
    
    Features:
    - Software emergency stop
    - Automatic stop on sensor triggers
    - Graceful deceleration
    - State logging
    """
    
    def __init__(self):
        self._state = SafetyState.NORMAL
        self._state_lock = threading.Lock()
        
        self._estop_triggers: List[Callable[[], bool]] = []
        self._stop_callbacks: List[Callable[[str], None]] = []
        self._resume_callbacks: List[Callable[[], None]] = []
        
        self._stop_time: Optional[float] = None
        self._stop_reason: Optional[str] = None
        
        self._logger = self._get_logger()
    
    def _get_logger(self):
        try:
            from src.utils.logger import get_logger
            return get_logger(__name__)
        except:
            import logging
            return logging.getLogger(__name__)
    
    def register_estop_trigger(self, trigger: Callable[[], bool]) -> None:
        """Register an emergency stop trigger."""
        self._estop_triggers.append(trigger)
    
    def register_stop_callback(self, callback: Callable[[str], None]) -> None:
        """Register callback for emergency stops."""
        self._stop_callbacks.append(callback)
    
    def register_resume_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for resume/normal operation."""
        self._resume_callbacks.append(callback)
    
    def trigger_estop(self, reason: str = "manual") -> None:
        """Trigger emergency stop."""
        with self._state_lock:
            if self._state == SafetyState.ESTOP:
                return
            
            self._state = SafetyState.ESTOP
            self._stop_time = time.time()
            self._stop_reason = reason
            
            self._logger.critical(f"EMERGENCY STOP: {reason}")
            
            for callback in self._stop_callbacks:
                try:
                    callback(reason)
                except Exception as e:
                    self._logger.error(f"Stop callback error: {e}")
    
    def clear_estop(self) -> bool:
        """Clear emergency stop and resume operation."""
        with self._state_lock:
            if self._state != SafetyState.ESTOP:
                return True
            
            self._state = SafetyState.NORMAL
            self._stop_time = None
            self._stop_reason = None
            
            self._logger.info("Emergency stop cleared")
            
            for callback in self._resume_callbacks:
                try:
                    callback()
                except Exception as e:
                    self._logger.error(f"Resume callback error: {e}")
            
            return True
    
    def check_estop_triggers(self) -> None:
        """Check all registered triggers."""
        for trigger in self._estop_triggers:
            try:
                if trigger():
                    self.trigger_estop("sensor_trigger")
                    return
            except Exception as e:
                self._logger.error(f"Trigger check error: {e}")
    
    def get_state(self) -> SafetyState:
        """Get current safety state."""
        with self._state_lock:
            return self._state
    
    def is_stopped(self) -> bool:
        """Check if system is stopped."""
        return self._state in [SafetyState.STOPPED, SafetyState.ESTOP]
    
    def is_estop(self) -> bool:
        """Check if emergency stop is active."""
        return self._state == SafetyState.ESTOP
    
    def get_stop_duration(self) -> Optional[float]:
        """Get duration of current stop."""
        if self._stop_time is None:
            return None
        return time.time() - self._stop_time


class SafeSpeedMonitor:
    """Monitor and enforce safe speed limits.
    
    Features:
    - Velocity limiting
    - Acceleration limiting
    - Distance-based speed reduction
    - Configurable limits
    """
    
    def __init__(
        self,
        limits: Optional[SafetyLimits] = None,
    ):
        self._limits = limits or SafetyLimits()
        
        self._current_velocity = 0.0
        self._velocity_history: List[float] = []
        self._max_history = 10
        
        self._current_position: Optional[np.ndarray] = None
        self._position_history: List[np.ndarray] = []
        
        self._min_distance = float("inf")
        self._last_update = time.time()
        
        self._override_active = False
        self._override_velocity = 0.0
        
        self._logger = self._get_logger()
    
    def _get_logger(self):
        try:
            from src.utils.logger import get_logger
            return get_logger(__name__)
        except:
            import logging
            return logging.getLogger(__name__)
    
    def set_limits(self, limits: SafetyLimits) -> None:
        """Update safety limits."""
        self._limits = limits
    
    def update_velocity(self, velocity: float) -> None:
        """Update current velocity and apply limits."""
        now = time.time()
        dt = now - self._last_update
        
        if dt > 0:
            acceleration = (velocity - self._current_velocity) / dt
            
            if abs(acceleration) > self._limits.max_acceleration:
                if acceleration > 0:
                    velocity = self._current_velocity + self._limits.max_acceleration * dt
                else:
                    velocity = self._current_velocity - self._limits.max_acceleration * dt
        
        velocity = np.clip(
            velocity,
            -self._limits.max_linear_velocity,
            self._limits.max_linear_velocity
        )
        
        self._velocity_history.append(velocity)
        if len(self._velocity_history) > self._max_history:
            self._velocity_history.pop(0)
        
        self._current_velocity = velocity
        self._last_update = now
    
    def update_distance(self, distance: float) -> None:
        """Update distance to nearest obstacle."""
        self._min_distance = min(self._min_distance, distance)
    
    def get_safe_velocity(self) -> float:
        """Get velocity limited by safety constraints."""
        if self._override_active:
            return self._override_velocity
        
        velocity = self._current_velocity
        
        if self._min_distance < self._limits.min_distance:
            return 0.0
        
        if self._min_distance < self._limits.min_distance * 2:
            factor = (self._min_distance - self._limits.min_distance) / self._limits.min_distance
            velocity *= max(0, factor)
        
        return velocity
    
    def update_position(self, position: np.ndarray) -> None:
        """Update current position and check limits."""
        self._current_position = position.copy()
        
        self._position_history.append(position.copy())
        if len(self._position_history) > self._max_history:
            self._position_history.pop(0)
        
        distance_from_origin = np.linalg.norm(position[:2])
        
        if distance_from_origin > self._limits.workspace_radius:
            self._logger.warning(f"Outside workspace: {distance_from_origin:.2f}m")
        
        if len(position) > 2:
            if position[2] > self._limits.max_height:
                self._logger.warning(f"Height limit exceeded: {position[2]:.2f}m")
            if position[2] < self._limits.min_height:
                self._logger.warning(f"Below minimum height: {position[2]:.2f}m")
    
    def set_override(self, active: bool, velocity: float = 0.0) -> None:
        """Override normal speed control."""
        self._override_active = active
        self._override_velocity = velocity
    
    def reset_distance(self) -> None:
        """Reset minimum distance tracking."""
        self._min_distance = float("inf")
    
    def get_current_velocity(self) -> float:
        """Get current (unlimited) velocity."""
        return self._current_velocity
    
    def get_average_velocity(self) -> float:
        """Get average velocity over history."""
        if not self._velocity_history:
            return 0.0
        return np.mean(self._velocity_history)
    
    def get_acceleration(self) -> float:
        """Get current acceleration."""
        if len(self._velocity_history) < 2:
            return 0.0
        return self._velocity_history[-1] - self._velocity_history[-2]
    
    def get_limits(self) -> SafetyLimits:
        """Get current safety limits."""
        return self._limits


class SafetyController:
    """Combined safety controller.
    
    Combines:
    - EmergencyStopController
    - SafeSpeedMonitor
    - State management
    """
    
    def __init__(
        self,
        limits: Optional[SafetyLimits] = None,
    ):
        self._estop = EmergencyStopController()
        self._speed_monitor = SafeSpeedMonitor(limits)
        
        self._warnings: List[str] = []
        self._max_warnings = 10
        
        self._last_safety_check = 0
        self._check_interval = 0.1
        
        self._logger = self._get_logger()
    
    def _get_logger(self):
        try:
            from src.utils.logger import get_logger
            return get_logger(__name__)
        except:
            import logging
            return logging.getLogger(__name__)
    
    def check_safety(self) -> SafetyStatus:
        """Perform safety check and return status."""
        now = time.time()
        if now - self._last_safety_check < self._check_interval:
            return self._get_current_status()
        
        self._last_safety_check = now
        
        self._estop.check_estop_triggers()
        
        distance = self._speed_monitor._min_distance
        velocity = self._speed_monitor.get_safe_velocity()
        
        state = SafetyState.NORMAL
        
        if self._estop.is_estop():
            state = SafetyState.ESTOP
        elif self._speed_monitor.get_acceleration() > self._speed_monitor._limits.max_acceleration * 1.5:
            state = SafetyState.WARNING
            self._add_warning("High acceleration detected")
        elif distance < self._speed_monitor._limits.min_distance * 1.5:
            state = SafetyState.WARNING
            self._add_warning(f"Close to obstacle: {distance:.2f}m")
        
        return SafetyStatus(
            state=state,
            timestamp=now,
            current_velocity=velocity,
            distance_to_obstacle=distance,
            emergency_reason=self._estop._stop_reason,
        )
    
    def _get_current_status(self) -> SafetyStatus:
        """Get current safety status without full check."""
        distance = self._speed_monitor._min_distance
        velocity = self._speed_monitor.get_safe_velocity()
        
        state = SafetyState.NORMAL
        if self._estop.is_estop():
            state = SafetyState.ESTOP
        elif self._speed_monitor.get_acceleration() > self._speed_monitor._limits.max_acceleration * 1.5:
            state = SafetyState.WARNING
        elif distance < self._speed_monitor._limits.min_distance * 1.5:
            state = SafetyState.WARNING
        
        return SafetyStatus(
            state=state,
            timestamp=time.time(),
            current_velocity=velocity,
            distance_to_obstacle=distance,
        )
    
    def _add_warning(self, warning: str) -> None:
        """Add warning message."""
        self._warnings.append(f"{time.time():.1f}: {warning}")
        if len(self._warnings) > self._max_warnings:
            self._warnings.pop(0)
    
    def trigger_estop(self, reason: str = "manual") -> None:
        """Trigger emergency stop."""
        self._estop.trigger_estop(reason)
    
    def clear_estop(self) -> bool:
        """Clear emergency stop."""
        return self._estop.clear_estop()
    
    def update_velocity(self, velocity: float) -> float:
        """Update velocity with safety limits."""
        self._speed_monitor.update_velocity(velocity)
        return self._speed_monitor.get_safe_velocity()
    
    def update_distance(self, distance: float) -> None:
        """Update obstacle distance."""
        self._speed_monitor.update_distance(distance)
    
    def update_position(self, position: np.ndarray) -> None:
        """Update robot position."""
        self._speed_monitor.update_position(position)
    
    def get_safe_velocity(self) -> float:
        """Get safe velocity after all checks."""
        return self._speed_monitor.get_safe_velocity()
    
    def get_warnings(self) -> List[str]:
        """Get recent warnings."""
        return self._warnings.copy()
    
    def clear_warnings(self) -> None:
        """Clear warning history."""
        self._warnings.clear()
    
    def set_limits(self, limits: SafetyLimits) -> None:
        """Update safety limits."""
        self._speed_monitor.set_limits(limits)
    
    @property
    def is_safe(self) -> bool:
        """Check if system is in safe state."""
        return not self._estop.is_estop()


def create_safety_controller(
    max_linear: float = 0.5,
    max_angular: float = 1.0,
    min_distance: float = 0.3,
) -> SafetyController:
    """Factory function to create safety controller."""
    limits = SafetyLimits(
        max_linear_velocity=max_linear,
        max_angular_velocity=max_angular,
        min_distance=min_distance,
    )
    return SafetyController(limits)