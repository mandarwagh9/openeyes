"""Fleet management module for multi-device deployment.

Provides device registration, heartbeat protocol, model deployment,
and telemetry collection across heterogeneous edge vision devices.
"""

from src.fleet.protocol import DeviceHeartbeat, ModelDeployment, FleetClient
from src.fleet.model_registry import ModelRegistry as FleetModelRegistry

__all__ = [
    "DeviceHeartbeat",
    "ModelDeployment",
    "FleetClient",
    "FleetModelRegistry",
]
