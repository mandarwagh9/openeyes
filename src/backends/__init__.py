"""Hardware Abstraction Layer - Backend interfaces.

Provides a unified interface for running inference across different
hardware backends: TensorRT, OpenVINO, TVM, Hailo DFC, QNN, ONNXRuntime.
"""

from src.backends.base import Backend, BackendType, BackendInfo, ModelHandle
from src.backends.registry import BackendRegistry

__all__ = [
    "Backend",
    "BackendType",
    "BackendInfo",
    "ModelHandle",
    "BackendRegistry",
]
