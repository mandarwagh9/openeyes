"""Health monitoring and diagnostics for OpenEyes.

Provides:
- System health monitoring
- Performance metrics
- Auto-recovery mechanisms
- Watchdog for 24/7 operation
"""

import time
import threading
import psutil
import os
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class HealthMetrics:
    """System health metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    fps: float = 0.0
    inference_time_ms: float = 0.0
    frame_drop_rate: float = 0.0
    error_count: int = 0
    status: str = "healthy"


@dataclass
class ComponentHealth:
    """Health status of individual components."""
    name: str
    status: str
    last_check: float
    error_message: Optional[str] = None
    restart_count: int = 0


class SystemHealthMonitor:
    """Monitor system health and trigger auto-recovery.
    
    Features:
    - CPU/Memory monitoring
    - GPU monitoring (if available)
    - FPS and performance tracking
    - Auto-restart for failed components
    - Watchdog for hung processes
    """
    
    def __init__(
        self,
        check_interval: float = 5.0,
        max_memory_percent: float = 90.0,
        max_cpu_percent: float = 95.0,
        min_fps: float = 1.0,
        enable_auto_recovery: bool = True,
    ):
        self._check_interval = check_interval
        self._max_memory_percent = max_memory_percent
        self._max_cpu_percent = max_cpu_percent
        self._min_fps = min_fps
        self._enable_auto_recovery = enable_auto_recovery
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        self._metrics_history: deque = deque(maxlen=100)
        self._component_health: Dict[str, ComponentHealth] = {}
        
        self._error_callbacks: List[Callable[[str, str], None]] = []
        self._recovery_callbacks: List[Callable[[str], None]] = []
        
        self._last_metrics: Optional[HealthMetrics] = None
        self._start_time = time.time()
        
        self._logger = self._get_logger()
    
    def _get_logger(self):
        """Get logger instance."""
        try:
            from src.utils.logger import get_logger
            return get_logger(__name__)
        except:
            import logging
            return logging.getLogger(__name__)
    
    def register_error_callback(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for errors."""
        self._error_callbacks.append(callback)
    
    def register_recovery_callback(self, callback: Callable[[str], None]) -> None:
        """Register callback for recovery actions."""
        self._recovery_callbacks.append(callback)
    
    def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        self._logger.info("Health monitoring started")
    
    def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        self._logger.info("Health monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                metrics = self._collect_metrics()
                self._metrics_history.append(metrics)
                self._last_metrics = metrics
                
                self._check_thresholds(metrics)
                self._check_components()
                
            except Exception as e:
                self._logger.error(f"Monitor error: {e}")
            
            time.sleep(self._check_interval)
    
    def _collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        metrics = HealthMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            status="healthy"
        )
        
        try:
            import subprocess
            result = subprocess.run(
                ["tegrastats", "--interval", "1"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                output = result.stdout
                if "GPU" in output:
                    for line in output.split("\n"):
                        if "GPU" in line and "%" in line:
                            parts = line.split()
                            for i, p in enumerate(parts):
                                if "GPU" in p and i + 1 < len(parts):
                                    try:
                                        metrics.gpu_percent = float(parts[i+1].replace("%", ""))
                                    except:
                                        pass
        except:
            pass
        
        return metrics
    
    def _check_thresholds(self, metrics: HealthMetrics) -> None:
        """Check if metrics exceed thresholds."""
        issues = []
        
        if metrics.memory_percent > self._max_memory_percent:
            issues.append(f"Memory high: {metrics.memory_percent:.1f}%")
        
        if metrics.cpu_percent > self._max_cpu_percent:
            issues.append(f"CPU high: {metrics.cpu_percent:.1f}%")
        
        if metrics.fps > 0 and metrics.fps < self._min_fps:
            issues.append(f"FPS low: {metrics.fps:.1f}")
        
        if metrics.error_count > 10:
            issues.append(f"Errors: {metrics.error_count}")
        
        if issues:
            error_msg = "; ".join(issues)
            for callback in self._error_callbacks:
                try:
                    callback("system", error_msg)
                except:
                    pass
    
    def _check_components(self) -> None:
        """Check component health."""
        for name, health in self._component_health.items():
            if health.status == "failed":
                if self._enable_auto_recovery:
                    self._attempt_recovery(name)
    
    def register_component(
        self,
        name: str,
        check_func: Optional[Callable[[], bool]] = None
    ) -> None:
        """Register a component for health monitoring."""
        self._component_health[name] = ComponentHealth(
            name=name,
            status="healthy",
            last_check=time.time(),
            check_func=check_func
        )
    
    def update_component_status(
        self,
        name: str,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """Update component status."""
        if name in self._component_health:
            health = self._component_health[name]
            old_status = health.status
            health.status = status
            health.last_check = time.time()
            health.error_message = error_message
            
            if status == "failed" and old_status != "failed":
                self._logger.error(f"Component {name} failed: {error_message}")
                for callback in self._error_callbacks:
                    try:
                        callback(name, error_message or "Unknown error")
                    except:
                        pass
    
    def _attempt_recovery(self, component_name: str) -> bool:
        """Attempt to recover a failed component."""
        self._logger.warning(f"Attempting recovery for {component_name}")
        
        health = self._component_health.get(component_name)
        if not health:
            return False
        
        health.restart_count += 1
        
        if health.restart_count > 3:
            self._logger.error(f"Max restarts exceeded for {component_name}")
            return False
        
        for callback in self._recovery_callbacks:
            try:
                callback(component_name)
                self._logger.info(f"Recovery triggered for {component_name}")
                return True
            except Exception as e:
                self._logger.error(f"Recovery failed: {e}")
        
        return False
    
    def get_current_metrics(self) -> Optional[HealthMetrics]:
        """Get current health metrics."""
        return self._last_metrics
    
    def get_metrics_history(self, count: int = 10) -> List[HealthMetrics]:
        """Get recent metrics history."""
        return list(self._metrics_history)[-count:]
    
    def get_component_health(self) -> Dict[str, ComponentHealth]:
        """Get all component health status."""
        return self._component_health.copy()
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self._start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        if not self._metrics_history:
            return {"status": "no_data"}
        
        recent = list(self._metrics_history)[-10:]
        
        return {
            "status": self._get_overall_status(),
            "uptime_seconds": self.get_uptime(),
            "avg_cpu": np.mean([m.cpu_percent for m in recent]),
            "avg_memory": np.mean([m.memory_percent for m in recent]),
            "avg_fps": np.mean([m.fps for m in recent if m.fps > 0]),
            "components": {
                name: health.status 
                for name, health in self._component_health.items()
            }
        }
    
    def _get_overall_status(self) -> str:
        """Get overall system status."""
        if not self._metrics_history:
            return "unknown"
        
        recent = list(self._metrics_history)[-5:]
        
        failed_count = sum(
            1 for h in self._component_health.values() 
            if h.status == "failed"
        )
        
        if failed_count > 0:
            return "degraded"
        
        if any(m.memory_percent > 90 for m in recent):
            return "warning"
        
        return "healthy"


class WatchdogTimer:
    """Watchdog timer for detecting hung processes."""
    
    def __init__(
        self,
        timeout: float = 30.0,
        reset_interval: float = 1.0,
    ):
        self._timeout = timeout
        self._reset_interval = reset_interval
        
        self._last_reset = time.time()
        self._running = False
        self._watchdog_thread: Optional[threading.Thread] = None
        self._expired = False
        
        self._logger = self._get_logger()
    
    def _get_logger(self):
        try:
            from src.utils.logger import get_logger
            return get_logger(__name__)
        except:
            import logging
            return logging.getLogger(__name__)
    
    def start(self) -> None:
        """Start watchdog timer."""
        if self._running:
            return
        
        self._running = True
        self._last_reset = time.time()
        self._expired = False
        
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True
        )
        self._watchdog_thread.start()
        self._logger.info(f"Watchdog started (timeout: {self._timeout}s)")
    
    def stop(self) -> None:
        """Stop watchdog timer."""
        self._running = False
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2.0)
        self._logger.info("Watchdog stopped")
    
    def reset(self) -> None:
        """Reset the watchdog timer."""
        self._last_reset = time.time()
        self._expired = False
    
    def _watchdog_loop(self) -> None:
        """Watchdog monitoring loop."""
        while self._running:
            elapsed = time.time() - self._last_reset
            
            if elapsed > self._timeout and not self._expired:
                self._expired = True
                self._logger.error(f"Watchdog expired! {elapsed:.1f}s > {self._timeout}s")
            
            time.sleep(self._reset_interval)
    
    def is_expired(self) -> bool:
        """Check if watchdog has expired."""
        return self._expired
    
    @property
    def timeout(self) -> float:
        """Get timeout in seconds."""
        return self._timeout
    
    @timeout.setter
    def timeout(self, value: float) -> None:
        """Set timeout in seconds."""
        self._timeout = max(1.0, value)


def create_health_monitor(
    check_interval: float = 5.0,
    enable_auto_recovery: bool = True,
) -> SystemHealthMonitor:
    """Factory function to create health monitor."""
    return SystemHealthMonitor(
        check_interval=check_interval,
        enable_auto_recovery=enable_auto_recovery,
    )