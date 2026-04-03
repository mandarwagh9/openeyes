"""Backend registry - discovers and ranks available backends."""

import time
from typing import Optional

from src.backends.base import Backend, BackendType, BackendInfo
from src.utils.logger import get_logger


class TensorRTBackendStub(Backend):
    """TensorRT backend stub - wraps existing TensorRT optimization logic."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tensorrt as trt
            return True
        except ImportError:
            return False

    @classmethod
    def get_info(cls) -> BackendInfo:
        import tensorrt as trt
        return BackendInfo(
            name="TensorRT",
            backend_type=BackendType.TENSORRT,
            version=trt.__version__,
            device_name="NVIDIA GPU",
            available_memory_bytes=0,
            supported_precisions=["fp32", "fp16", "int8"],
            max_batch_size=1,
        )

    def load_model(self, model_path: str, precision: str = "fp16",
                   batch_size: int = 1, **kwargs) -> "ModelHandle":
        from src.backends.base import ModelHandle
        return ModelHandle(
            backend=BackendType.TENSORRT,
            model_path=model_path,
            input_shape=(1, 3, 640, 640),
            output_shapes=[(1, 84, 8400)],
            precision=precision,
        )

    def infer(self, handle: "ModelHandle", inputs: "np.ndarray") -> list["np.ndarray"]:
        raise NotImplementedError("Use existing TensorRT detector instead")

    def export_model(self, source_model_path: str, output_path: str,
                     precision: str = "fp16", **kwargs) -> str:
        raise NotImplementedError("Use export_tensorrt_optimized.py instead")

    def benchmark(self, handle: "ModelHandle", num_iterations: int = 100) -> dict:
        raise NotImplementedError("Use trtexec for benchmarking")


class ONNXRuntimeBackendStub(Backend):
    """ONNXRuntime backend - universal fallback."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            import onnxruntime
            return True
        except ImportError:
            return False

    @classmethod
    def get_info(cls) -> BackendInfo:
        import onnxruntime
        return BackendInfo(
            name="ONNXRuntime",
            backend_type=BackendType.ONNXRUNTIME,
            version=onnxruntime.__version__,
            device_name="CPU",
            available_memory_bytes=0,
            supported_precisions=["fp32", "fp16"],
            max_batch_size=1,
        )

    def load_model(self, model_path: str, precision: str = "fp32",
                   batch_size: int = 1, **kwargs) -> "ModelHandle":
        from src.backends.base import ModelHandle
        return ModelHandle(
            backend=BackendType.ONNXRUNTIME,
            model_path=model_path,
            input_shape=(1, 3, 640, 640),
            output_shapes=[(1, 84, 8400)],
            precision=precision,
        )

    def infer(self, handle: "ModelHandle", inputs: "np.ndarray") -> list["np.ndarray"]:
        import onnxruntime
        sess = onnxruntime.InferenceSession(handle.model_path)
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: inputs})
        return outputs

    def export_model(self, source_model_path: str, output_path: str,
                     precision: str = "fp32", **kwargs) -> str:
        raise NotImplementedError("ONNX is already the source format")

    def benchmark(self, handle: "ModelHandle", num_iterations: int = 100) -> dict:
        import onnxruntime
        import numpy as np
        sess = onnxruntime.InferenceSession(handle.model_path)
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        dummy = np.random.randn(*input_shape).astype(np.float32)

        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            sess.run(None, {input_name: dummy})
            times.append(time.perf_counter() - start)

        return {
            "mean_ms": np.mean(times) * 1000,
            "p50_ms": np.percentile(times, 50) * 1000,
            "p95_ms": np.percentile(times, 95) * 1000,
            "p99_ms": np.percentile(times, 99) * 1000,
            "fps": 1.0 / (np.mean(times) + 1e-9),
        }


class BackendRegistry:
    """Discovers and ranks available backends."""

    _backends = {
        BackendType.TENSORRT: TensorRTBackendStub,
        BackendType.ONNXRUNTIME: ONNXRuntimeBackendStub,
    }

    _priority = [
        BackendType.TENSORRT,
        BackendType.OPENVINO,
        BackendType.HAILO_DFC,
        BackendType.TVM,
        BackendType.QNN,
        BackendType.ONNXRUNTIME,
    ]

    @classmethod
    def auto_select(cls, preferences: Optional[list] = None) -> Optional[Backend]:
        """Select the best available backend."""
        order = preferences or cls._priority
        for backend_type in order:
            backend_cls = cls._backends.get(backend_type)
            if backend_cls and backend_cls.is_available():
                return backend_cls()
        return None

    @classmethod
    def list_available(cls) -> list[BackendInfo]:
        """List all available backends with their info."""
        available = []
        for backend_type in cls._priority:
            backend_cls = cls._backends.get(backend_type)
            if backend_cls and backend_cls.is_available():
                try:
                    available.append(backend_cls.get_info())
                except Exception:
                    pass
        return available

    @classmethod
    def get_backend(cls, backend_type: BackendType) -> Optional[Backend]:
        """Get a specific backend by type."""
        backend_cls = cls._backends.get(backend_type)
        if backend_cls and backend_cls.is_available():
            return backend_cls()
        return None
