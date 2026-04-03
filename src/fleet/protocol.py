"""Fleet management protocol - device heartbeat and deployment."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import json


@dataclass
class DeviceHeartbeat:
    """Heartbeat message from an edge device to the fleet server."""
    device_id: str
    group: str
    platform: str
    uptime_seconds: int
    fps: float
    latency_ms: float
    cpu_percent: float
    gpu_percent: float
    memory_percent: float
    temperature_c: float
    model_versions: Dict[str, str] = field(default_factory=dict)
    error_count: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "group": self.group,
            "platform": self.platform,
            "uptime_seconds": self.uptime_seconds,
            "fps": self.fps,
            "latency_ms": self.latency_ms,
            "cpu_percent": self.cpu_percent,
            "gpu_percent": self.gpu_percent,
            "memory_percent": self.memory_percent,
            "temperature_c": self.temperature_c,
            "model_versions": self.model_versions,
            "error_count": self.error_count,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeviceHeartbeat":
        return cls(
            device_id=data.get("device_id", ""),
            group=data.get("group", ""),
            platform=data.get("platform", ""),
            uptime_seconds=data.get("uptime_seconds", 0),
            fps=data.get("fps", 0.0),
            latency_ms=data.get("latency_ms", 0.0),
            cpu_percent=data.get("cpu_percent", 0.0),
            gpu_percent=data.get("gpu_percent", 0.0),
            memory_percent=data.get("memory_percent", 0.0),
            temperature_c=data.get("temperature_c", 0.0),
            model_versions=data.get("model_versions", {}),
            error_count=data.get("error_count", 0),
            timestamp=data.get("timestamp", time.time()),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "DeviceHeartbeat":
        return cls.from_dict(json.loads(json_str))


@dataclass
class ModelDeployment:
    """Model deployment instruction from fleet server to edge device."""
    model_name: str
    version: str
    checksum: str
    download_url: str
    target_devices: List[str] = field(default_factory=lambda: ["*"])
    target_groups: List[str] = field(default_factory=list)
    rollback_version: Optional[str] = None
    requires_restart: bool = False
    timestamp: float = field(default_factory=time.time)

    def is_targeted(self, device_id: str, device_group: str) -> bool:
        """Check if this deployment targets the given device."""
        if "*" in self.target_devices:
            if not self.target_groups or device_group in self.target_groups:
                return True
        if device_id in self.target_devices:
            return True
        if device_group in self.target_groups:
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "checksum": self.checksum,
            "download_url": self.download_url,
            "target_devices": self.target_devices,
            "target_groups": self.target_groups,
            "rollback_version": self.rollback_version,
            "requires_restart": self.requires_restart,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelDeployment":
        return cls(
            model_name=data.get("model_name", ""),
            version=data.get("version", ""),
            checksum=data.get("checksum", ""),
            download_url=data.get("download_url", ""),
            target_devices=data.get("target_devices", ["*"]),
            target_groups=data.get("target_groups", []),
            rollback_version=data.get("rollback_version"),
            requires_restart=data.get("requires_restart", False),
            timestamp=data.get("timestamp", time.time()),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "ModelDeployment":
        return cls.from_dict(json.loads(json_str))


class FleetClient:
    """Client running on each edge device for fleet communication.

    Handles heartbeat sending, deployment receiving, and status reporting.
    """

    def __init__(self, device_id: str, server_url: str = ""):
        self.device_id = device_id
        self.server_url = server_url
        self._heartbeat_interval = 30
        self._last_heartbeat: float = 0
        self._start_time: float = time.time()
        self._group: str = "default"
        self._platform: str = "unknown"

    def set_group(self, group: str) -> None:
        self._group = group

    def set_platform(self, platform: str) -> None:
        self._platform = platform

    def create_heartbeat(
        self,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        cpu_percent: float = 0.0,
        gpu_percent: float = 0.0,
        memory_percent: float = 0.0,
        temperature_c: float = 0.0,
        model_versions: Optional[Dict[str, str]] = None,
        error_count: int = 0,
    ) -> DeviceHeartbeat:
        """Create a heartbeat message with current device stats."""
        uptime = int(time.time() - self._start_time)
        return DeviceHeartbeat(
            device_id=self.device_id,
            group=self._group,
            platform=self._platform,
            uptime_seconds=uptime,
            fps=fps,
            latency_ms=latency_ms,
            cpu_percent=cpu_percent,
            gpu_percent=gpu_percent,
            memory_percent=memory_percent,
            temperature_c=temperature_c,
            model_versions=model_versions or {},
            error_count=error_count,
        )

    def should_send_heartbeat(self) -> bool:
        """Check if it's time to send a heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self._heartbeat_interval:
            self._last_heartbeat = now
            return True
        return False

    def set_heartbeat_interval(self, interval: int) -> None:
        self._heartbeat_interval = interval
