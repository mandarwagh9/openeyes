"""Platform detection - auto-detects hardware platform."""

import os
import platform
from dataclasses import dataclass
from typing import Optional

from src.utils.logger import get_logger


@dataclass
class PlatformInfo:
    name: str
    vendor: str
    cpu: str
    gpu: Optional[str]
    npu: Optional[str]
    total_memory_gb: int
    available_backends: list
    recommended_precision: str
    recommended_batch_size: int


class PlatformDetector:
    """Auto-detects the current hardware platform."""

    @classmethod
    def detect(cls) -> PlatformInfo:
        """Detect the current platform and return PlatformInfo."""
        logger = get_logger(__name__)

        is_jetson = cls._is_jetson()
        is_pi = cls._is_raspberry_pi()
        is_intel = cls._is_intel_npu()

        cpu_info = cls._get_cpu_info()
        total_mem_gb = cls._get_total_memory_gb()

        if is_jetson:
            model = cls._get_jetson_model()
            gpu = "NVIDIA Orin GPU" if "orin" in model.lower() else "NVIDIA GPU"
            return PlatformInfo(
                name=model,
                vendor="nvidia",
                cpu=cpu_info,
                gpu=gpu,
                npu="DLA" if "orin" in model.lower() else None,
                total_memory_gb=total_mem_gb,
                available_backends=["tensorrt", "onnxruntime"],
                recommended_precision="fp16",
                recommended_batch_size=1,
            )
        elif is_pi:
            return PlatformInfo(
                name="raspberry-pi",
                vendor="raspberry-pi",
                cpu=cpu_info,
                gpu="VideoCore VII" if total_mem_gb >= 4 else "VideoCore VI",
                npu="Hailo-10H" if cls._has_hailo() else None,
                total_memory_gb=total_mem_gb,
                available_backends=["onnxruntime", "hailo_dfc"] if cls._has_hailo() else ["onnxruntime"],
                recommended_precision="int8",
                recommended_batch_size=1,
            )
        elif is_intel:
            return PlatformInfo(
                name="intel-npu",
                vendor="intel",
                cpu=cpu_info,
                gpu="Intel Arc",
                npu="Intel NPU",
                total_memory_gb=total_mem_gb,
                available_backends=["openvino", "onnxruntime"],
                recommended_precision="int8",
                recommended_batch_size=1,
            )
        else:
            return PlatformInfo(
                name="generic-linux",
                vendor="generic",
                cpu=cpu_info,
                gpu=cls._get_gpu_info(),
                npu=None,
                total_memory_gb=total_mem_gb,
                available_backends=["onnxruntime"],
                recommended_precision="fp32",
                recommended_batch_size=1,
            )

    @staticmethod
    def _is_jetson() -> bool:
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
                return "jetson" in model or "tegra" in model
        except Exception:
            return False

    @staticmethod
    def _is_raspberry_pi() -> bool:
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
                return "raspberry" in model
        except Exception:
            return False

    @staticmethod
    def _is_intel_npu() -> bool:
        try:
            import subprocess
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5
            )
            return "npu" in result.stdout.lower() or "neural" in result.stdout.lower()
        except Exception:
            return False

    @staticmethod
    def _has_hailo() -> bool:
        try:
            import subprocess
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5
            )
            return "hailo" in result.stdout.lower()
        except Exception:
            return False

    @staticmethod
    def _get_jetson_model() -> str:
        try:
            with open("/proc/device-tree/model", "r") as f:
                return f.read().strip().replace("NVIDIA ", "")
        except Exception:
            return "jetson-unknown"

    @staticmethod
    def _get_cpu_info() -> str:
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":")[1].strip()
        except Exception:
            return platform.machine()
        return platform.machine()

    @staticmethod
    def _get_total_memory_gb() -> int:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return max(1, kb // (1024 * 1024))
        except Exception:
            return 4
        return 4

    @staticmethod
    def _get_gpu_info() -> Optional[str]:
        try:
            import subprocess
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split("\n"):
                if "VGA" in line or "3D" in line or "Display" in line:
                    return line.split(":")[-1].strip()
        except Exception:
            pass
        return None
