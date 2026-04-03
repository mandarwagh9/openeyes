import pytest
import numpy as np

from src.backends.base import BackendType, BackendInfo, ModelHandle
from src.backends.registry import BackendRegistry, ONNXRuntimeBackendStub
from src.platforms.detector import PlatformDetector, PlatformInfo


class TestBackendType:
    def test_enum_values(self):
        assert BackendType.TENSORRT.value == "tensorrt"
        assert BackendType.OPENVINO.value == "openvino"
        assert BackendType.ONNXRUNTIME.value == "onnxruntime"


class TestBackendInfo:
    def test_creation(self):
        info = BackendInfo(
            name="TensorRT",
            backend_type=BackendType.TENSORRT,
            version="10.3.0",
            device_name="Orin",
            available_memory_bytes=8 * 1024**3,
            supported_precisions=["fp32", "fp16", "int8"],
            max_batch_size=1,
        )
        assert info.name == "TensorRT"
        assert info.backend_type == BackendType.TENSORRT
        assert len(info.supported_precisions) == 3


class TestModelHandle:
    def test_creation(self):
        handle = ModelHandle(
            backend=BackendType.TENSORRT,
            model_path="model.engine",
            input_shape=(1, 3, 640, 640),
            output_shapes=[(1, 84, 8400)],
            precision="fp16",
        )
        assert handle.backend == BackendType.TENSORRT
        assert handle.model_path == "model.engine"
        assert handle.precision == "fp16"


class TestONNXRuntimeBackend:
    def test_is_available(self):
        assert ONNXRuntimeBackendStub.is_available() is True

    def test_get_info(self):
        info = ONNXRuntimeBackendStub.get_info()
        assert isinstance(info, BackendInfo)
        assert info.backend_type == BackendType.ONNXRUNTIME
        assert "ONNXRuntime" in info.name

    def test_load_model(self):
        backend = ONNXRuntimeBackendStub()
        handle = backend.load_model("models/yolo11n.onnx")
        assert handle.backend == BackendType.ONNXRUNTIME
        assert handle.model_path == "models/yolo11n.onnx"

    def test_infer(self):
        backend = ONNXRuntimeBackendStub()
        handle = backend.load_model("models/yolo11n.onnx")
        dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
        outputs = backend.infer(handle, dummy)
        assert isinstance(outputs, list)
        assert len(outputs) > 0

    def test_benchmark(self):
        backend = ONNXRuntimeBackendStub()
        handle = backend.load_model("models/yolo11n.onnx")
        result = backend.benchmark(handle, num_iterations=10)
        assert "mean_ms" in result
        assert "fps" in result
        assert result["mean_ms"] > 0
        assert result["fps"] > 0


class TestBackendRegistry:
    def test_auto_select_returns_backend(self):
        backend = BackendRegistry.auto_select()
        assert backend is not None

    def test_list_available(self):
        available = BackendRegistry.list_available()
        assert len(available) >= 1
        assert all(isinstance(info, BackendInfo) for info in available)

    def test_get_backend_onnxruntime(self):
        backend = BackendRegistry.get_backend(BackendType.ONNXRUNTIME)
        assert backend is not None
        assert isinstance(backend, ONNXRuntimeBackendStub)


class TestPlatformDetector:
    def test_detect_returns_platform_info(self):
        info = PlatformDetector.detect()
        assert isinstance(info, PlatformInfo)
        assert isinstance(info.name, str)
        assert isinstance(info.vendor, str)
        assert isinstance(info.cpu, str)
        assert isinstance(info.total_memory_gb, int)
        assert info.total_memory_gb >= 1
        assert len(info.available_backends) >= 1
        assert isinstance(info.recommended_precision, str)
        assert isinstance(info.recommended_batch_size, int)

    def test_is_jetson(self):
        result = PlatformDetector._is_jetson()
        assert isinstance(result, bool)

    def test_get_cpu_info(self):
        cpu = PlatformDetector._get_cpu_info()
        assert isinstance(cpu, str)
        assert len(cpu) > 0

    def test_get_total_memory_gb(self):
        mem = PlatformDetector._get_total_memory_gb()
        assert isinstance(mem, int)
        assert mem >= 1
