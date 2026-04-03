"""Abstract backend interface for all inference engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class BackendType(Enum):
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    TVM = "tvm"
    HAILO_DFC = "hailo_dfc"
    QNN = "qnn"
    ONNXRUNTIME = "onnxruntime"


@dataclass
class BackendInfo:
    name: str
    backend_type: BackendType
    version: str
    device_name: str
    available_memory_bytes: int
    supported_precisions: list
    max_batch_size: int


@dataclass
class ModelHandle:
    backend: BackendType
    model_path: str
    input_shape: tuple
    output_shapes: list
    precision: str
    _internal: object = None


class Backend(ABC):
    """Abstract base for all inference backends."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this backend is available on the current system."""
        ...

    @classmethod
    @abstractmethod
    def get_info(cls) -> BackendInfo:
        """Return hardware/backend capabilities."""
        ...

    @abstractmethod
    def load_model(
        self,
        model_path: str,
        precision: str = "fp16",
        batch_size: int = 1,
        **kwargs
    ) -> ModelHandle:
        """Load a model into this backend."""
        ...

    @abstractmethod
    def infer(
        self,
        handle: ModelHandle,
        inputs: np.ndarray
    ) -> list[np.ndarray]:
        """Run inference on the given inputs."""
        ...

    @abstractmethod
    def export_model(
        self,
        source_model_path: str,
        output_path: str,
        precision: str = "fp16",
        calibration_data: Optional[np.ndarray] = None,
        **kwargs
    ) -> str:
        """Export/convert a model to this backend's format."""
        ...

    @abstractmethod
    def benchmark(
        self,
        handle: ModelHandle,
        num_iterations: int = 100
    ) -> dict:
        """Benchmark model inference latency and throughput."""
        ...
