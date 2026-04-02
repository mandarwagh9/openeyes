# TECHNICAL SPECIFICATION v2.0.0

> **Project**: OpenEyes - Hardware-Agnostic Edge Vision Framework
> **Version**: v2.0.0
> **Date**: 2026-04-02
> **Status**: Draft

---

## 1. Executive Summary

OpenEyes v2.0.0 transforms the project from a Jetson Orin Nano-specific robot vision system into a hardware-agnostic edge vision framework. The v2.0 architecture introduces a Hardware Abstraction Layer (HAL), unified inference pipeline, fleet management, and multi-platform support targeting the $47.6B edge AI market.

**Key Metrics**:
- 5-minute setup from zero to production vision
- Support for 7+ hardware platforms (Jetson, Pi, Intel NPU, Hailo, Qualcomm)
- 30+ FPS on target hardware with full detection + tracking + depth pipeline
- Zero-code model export across backends (TensorRT, OpenVINO, TVM, Hailo DFC)

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ CLI Tool │  │ ROS2 Node│  │ Fleet API│  │ Industry       │  │
│  │          │  │          │  │          │  │ Templates      │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───────┬────────┘  │
│       │              │              │                │            │
├───────┼──────────────┼──────────────┼────────────────┼───────────┤
│       │      Unified Pipeline Layer  │                │            │
│       │  ┌────────────────────────────────────────┐  │            │
│       │  │          Pipeline Orchestrator          │  │            │
│       │  │  ┌─────────┐ ┌──────┐ ┌─────────────┐ │  │            │
│       │  │  │Detector │ │Depth │ │Segmentation │ │  │            │
│       │  │  │(YOLO26) │ │(DA3) │ │(SAM 3)      │ │  │            │
│       │  │  └────┬────┘ └──┬───┘ └──────┬──────┘ │  │            │
│       │  │       │         │             │        │  │            │
│       │  │  ┌────┴─────────┴─────────────┴──────┐│  │            │
│       │  │  │         Model Scheduler            ││  │            │
│       │  │  │  (frame skipping, parallel exec)   ││  │            │
│       │  │  └───────────────────────────────────┘│  │            │
│       │  └──────────────────┬────────────────────┘  │            │
│       └─────────────────────┼───────────────────────┘            │
│                             │                                     │
├─────────────────────────────┼─────────────────────────────────────┤
│          HAL Layer          │                                     │
│  ┌──────────────────────────┴──────────────────────────────────┐ │
│  │              Hardware Abstraction Interface                  │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────┐ ┌──────────┐ ┌───────┐ │ │
│  │  │TensorRT  │ │OpenVINO  │ │ TVM  │ │Hailo DFC │ │QNN    │ │ │
│  │  │Backend   │ │Backend   │ │Backend│ │Backend   │ │Backend│ │ │
│  │  └────┬─────┘ └────┬─────┘ └──┬───┘ └────┬─────┘ └───┬───┘ │ │
│  │       │              │          │           │           │     │ │
│  │  ┌────┴──────────────┴──────────┴───────────┴───────────┴──┐│ │
│  │  │              Model Registry & Export                      ││ │
│  │  │  (ONNX intermediate, auto-quantization, calibration)     ││ │
│  │  └─────────────────────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                    Platform Detection Layer                        │
│  ┌────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐ │
│  │Jetson  │ │Pi 5+HAT  │ │Intel   │ │Hailo-8 │ │Qualcomm      │ │
│  │Orin    │ │(Hailo-10H)│ │Core    │ │Standalone│ │RB5/RB6      │ │
│  │Nano/NX │ │           │ │Ultra   │ │         │ │              │ │
│  └────────┘ └──────────┘ └────────┘ └─────────┘ └──────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

### 2.2 Design Principles

1. **Hardware Agnostic**: Single API, multiple backends. Application code never touches backend-specific code.
2. **Progressive Enhancement**: Works on cheapest hardware (Pi 5 + AI HAT+ 2, $150), scales to most powerful (Jetson T5000, $3,499).
3. **Zero-Config Defaults**: Sensible defaults for every platform. `openeyes start` just works.
4. **Graceful Degradation**: If a model or backend fails, fall back to lighter alternative without crashing.
5. **Observable by Default**: Every component emits metrics, logs, and health status.

---

## 3. Hardware Abstraction Layer (HAL)

### 3.1 Backend Interface

```python
# src/backends/base.py

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
    supported_precisions: list[str]  # ["fp32", "fp16", "int8", "int4"]
    max_batch_size: int


@dataclass
class ModelHandle:
    """Opaque handle to a loaded model on any backend."""
    backend: BackendType
    model_path: str
    input_shape: tuple[int, ...]
    output_shapes: list[tuple[int, ...]]
    precision: str
    _internal: object  # backend-specific handle


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
        inputs: np.ndarray | list[np.ndarray]
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
```

### 3.2 Backend Auto-Selection

```python
# src/backends/registry.py

class BackendRegistry:
    """Discovers and ranks available backends."""

    _backends: dict[BackendType, type[Backend]] = {
        BackendType.TENSORRT: TensorRTBackend,
        BackendType.OPENVINO: OpenVINOBackend,
        BackendType.TVM: TVMBackend,
        BackendType.HAILO_DFC: HailoDFCBackend,
        BackendType.QNN: QNNBackend,
        BackendType.ONNXRUNTIME: ONNXRuntimeBackend,
    }

    @classmethod
    def auto_select(cls, preferences: list[str] | None = None) -> Backend:
        """Select the best available backend.

        Priority order (default):
        1. TensorRT (Jetson GPUs - highest performance)
        2. OpenVINO (Intel CPU/NPU)
        3. Hailo DFC (Hailo accelerators)
        4. TVM (hardware-agnostic, requires tuning)
        5. QNN (Qualcomm Hexagon NPU)
        6. ONNXRuntime (universal fallback)
        """
        ...
```

### 3.3 Backend Implementations

#### TensorRT Backend (refactored from existing)
```
src/backends/tensorrt_backend.py
- Wraps existing TensorRT optimization logic
- Supports FP16, INT8 (with calibration), FP32
- DLA offloading support
- Jetson-specific memory management
```

#### OpenVINO Backend (new)
```
src/backends/openvino_backend.py
- Intel CPU, iGPU, and NPU support
- NNCF INT8/INT4 quantization
- Model caching via OpenVINO IR format
- Supports Core Ultra Series 2/3 NPUs (48 TOPS)
```

#### TVM Backend (new)
```
src/backends/tvm_backend.py
- Hardware-agnostic auto-tuning
- Supports Ethos-U, RZ/V2L, Hailo, i.MX
- AutoTVM for kernel optimization
- Requires initial tuning run (cached thereafter)
```

#### Hailo DFC Backend (new)
```
src/backends/hailo_backend.py
- Hailo Dataflow Compiler integration
- Supports Hailo-8 (26 TOPS), Hailo-10H (40 TOPS), Hailo-15 (20 TOPS)
- HEF file format
- Raspberry Pi AI HAT+ 2 support
```

#### QNN Backend (new)
```
src/backends/qnn_backend.py
- Qualcomm Hexagon NPU support
- RB5 (15 TOPS), RB6 (30+ TOPS)
- DLC model format
- SNPE compatibility layer
```

#### ONNXRuntime Backend (fallback)
```
src/backends/onnxruntime_backend.py
- Universal fallback for any platform
- CPU and GPU (CUDA, DirectML, CoreML) execution providers
- INT8 quantization via onnxruntime.quantization
- Always available as last resort
```

### 3.4 Platform Detection

```python
# src/platforms/detector.py

@dataclass
class PlatformInfo:
    name: str                    # "jetson-orin-nano", "pi5-ai-hat", etc.
    vendor: str                  # "nvidia", "raspberry-pi", "intel", etc.
    cpu: str                     # CPU description
    gpu: str | None              # GPU description
    npu: str | None              # NPU description
    total_memory_gb: int
    available_backends: list[BackendType]
    recommended_precision: str   # "fp16", "int8", etc.
    recommended_batch_size: int


class PlatformDetector:
    """Auto-detects the current hardware platform."""

    DETECTION_ORDER = [
        JetsonPlatformDetector,      # Check for Jetson first
        RaspberryPiDetector,          # Then Pi
        IntelPlatformDetector,        # Then Intel
        HailoPlatformDetector,        # Then Hailo standalone
        QualcommPlatformDetector,     # Then Qualcomm
        GenericLinuxDetector,         # Fallback
    ]

    @classmethod
    def detect(cls) -> PlatformInfo:
        ...
```

### 3.5 Platform Profiles

```yaml
# src/platforms/profiles/jetson-orin-nano.yaml
platform: jetson-orin-nano
vendor: nvidia
backends:
  - tensorrt
  - onnxruntime
recommended:
  precision: fp16
  batch_size: 1
  depth_model: "da3-small"
  detection_model: "yolo26n"
  tracking: "bytetrack"
limits:
  max_memory_gb: 8
  max_model_params_m: 100
  thermal_throttle_temp_c: 85

# src/platforms/profiles/pi5-ai-hat.yaml
platform: pi5-ai-hat
vendor: raspberry-pi
backends:
  - hailo_dfc
  - onnxruntime
recommended:
  precision: int8
  batch_size: 1
  depth_model: "da3-small-int8"
  detection_model: "yolo26n-int8"
  tracking: "bytetrack"
limits:
  max_memory_gb: 8
  max_model_params_m: 50
  thermal_throttle_temp_c: 80
```

---

## 4. Unified Inference Pipeline

### 4.1 Pipeline Definition

```python
# src/pipeline/unified_pipeline.py

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class StageConfig:
    """Configuration for a single pipeline stage."""
    name: str
    model_name: str
    enabled: bool = True
    skip_frames: int = 1         # Run every N frames
    confidence_threshold: float = 0.5
    backend: str = "auto"        # "auto" or specific backend
    precision: str = "auto"
    input_resolution: tuple[int, int] = (640, 640)
    output_callback: Callable | None = None


@dataclass
class PipelineConfig:
    """Complete pipeline definition."""
    name: str
    stages: list[StageConfig]
    camera_source: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    max_workers: int = 4
    zero_copy: bool = True       # Zero-copy GPU memory where supported
    output_format: str = "json"  # "json", "ros2", "mqtt", "udp"


class UnifiedPipeline:
    """Declarative pipeline with automatic scheduling and execution."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._stages: dict[str, StageRunner] = {}
        self._scheduler = FrameScheduler(config.stages)
        self._memory_pool = GPUMemoryPool() if zero_copy else None

    def run_frame(self, frame: np.ndarray) -> PipelineResult:
        """Process a single frame through all active stages."""
        # 1. Check which stages should run this frame
        active_stages = self._scheduler.get_active_stages(frame_number)

        # 2. Schedule parallel execution
        futures = {}
        for stage_name in active_stages:
            stage = self._stages[stage_name]
            if stage.should_run(frame):
                futures[stage_name] = self._executor.submit(
                    stage.run, frame
                )

        # 3. Collect results
        results = {}
        for stage_name, future in futures.items():
            results[stage_name] = future.result()

        # 4. Merge into unified result
        return PipelineResult(
            frame_number=frame_number,
            timestamp=time.time(),
            stages=results,
        )
```

### 4.2 Frame Scheduler

```python
# src/pipeline/scheduler.py

class FrameScheduler:
    """Intelligent frame scheduling for multi-model pipelines.

    Strategy:
    - Detection: every frame (critical for real-time)
    - Depth: every 3rd frame (depth changes slowly)
    - Segmentation: every 5th frame (expensive, changes slowly)
    - Pose: every 2nd frame (medium cost)
    - Face/Gesture: every 3rd frame (medium cost)

    Adaptive mode: if FPS drops below target, increase skip intervals.
    """

    def __init__(self, stages: list[StageConfig], target_fps: int = 30):
        self.stages = stages
        self.target_fps = target_fps
        self._frame_count = 0
        self._current_fps = 0.0

    def get_active_stages(self, frame_number: int) -> list[str]:
        """Return list of stage names that should run this frame."""
        active = []
        for stage in self.stages:
            if not stage.enabled:
                continue
            if frame_number % stage.skip_frames == 0:
                active.append(stage.name)
        return active

    def adapt_to_performance(self, measured_fps: float):
        """Increase skip intervals if FPS drops below target."""
        if measured_fps < self.target_fps * 0.7:
            for stage in self.stages:
                stage.skip_frames = min(stage.skip_frames * 2, 10)
```

### 4.3 Pipeline Configuration (YAML)

```yaml
# pipelines/warehouse.yaml
name: warehouse-inspection
camera_source: 0
camera_width: 640
camera_height: 480
camera_fps: 30

stages:
  - name: detection
    model: yolo26n
    skip_frames: 1
    confidence: 0.5
    backend: auto

  - name: depth
    model: da3-small
    skip_frames: 3
    backend: auto

  - name: tracking
    algorithm: bytetrack
    skip_frames: 1

  - name: segmentation
    model: edgesam
    skip_frames: 5
    confidence: 0.4
    backend: auto

  - name: anomaly_detection
    model: fastsam
    skip_frames: 10
    confidence: 0.3
    backend: auto

output:
  format: ros2
  topics:
    detections: /vision/detections
    depth: /vision/depth
    tracking: /vision/tracks
    anomalies: /vision/anomalies
```

---

## 5. Model Updates

### 5.1 YOLO26 Integration

```python
# src/models/yolo26_detector.py

class YOLO26Detector(ObjectDetector):
    """YOLO26 object detector with NMS-free end-to-end predictions.

    Benchmarks on Jetson Orin Nano (FP16 TensorRT):
    - YOLO26n: ~35-40 FPS, 40.9% mAP, 2.4M params
    - YOLO26s: ~20-25 FPS, 48.6% mAP, 9.5M params
    - YOLO26m: ~10-15 FPS, 53.1% mAP, 20.4M params
    """

    SUPPORTED_VARIANTS = {
        "yolo26n": {"params": 2.4, "flops": 5.4, "map": 40.9},
        "yolo26s": {"params": 9.5, "flops": 20.7, "map": 48.6},
        "yolo26m": {"params": 20.4, "flops": 68.2, "map": 53.1},
        "yolo26l": {"params": 24.8, "flops": 86.4, "map": 55.0},
        "yolo26x": {"params": 55.7, "flops": 193.9, "map": 57.5},
    }

    def __init__(self, variant: str = "yolo26n", **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        self._backend = None  # Set by HAL
```

### 5.2 Depth Anything V3 Integration

```python
# src/models/depth_anything_v3.py

class DepthAnythingV3(DepthEstimator):
    """Depth Anything V3 - SOTA monocular depth estimation.

    ICLR 2026 Oral. Single plain transformer with depth-ray representation.
    Outperforms DA2 by 35.7% in camera pose accuracy.

    Models:
    - da3-small: ~25M params, ~15 FPS on Orin Nano
    - da3-base: ~98M params, ~8 FPS on Orin Nano
    - da3-large: ~335M params, ~3 FPS on Orin Nano (not recommended for edge)
    """

    VARIANTS = {
        "da3-small": {"params": 25, "hf_id": "depth-anything/Depth-Anything-V3-Small"},
        "da3-base": {"params": 98, "hf_id": "depth-anything/Depth-Anything-V3-Base"},
        "da3-large": {"params": 335, "hf_id": "depth-anything/Depth-Anything-V3-Large"},
    }

    def __init__(self, variant: str = "da3-small", **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        self.supports_multi_view = True  # DA3 capability
```

### 5.3 SAM 3 Integration

```python
# src/models/sam3_segmenter.py

class SAM3Segmenter:
    """SAM 3 - Segment Anything with Concepts.

    Meta, Mar 2026. 4M unique concept labels in training data.
    Doubles accuracy of prior systems on promptable concept segmentation.

    Edge variant: EdgeSAM (40x speedup, ~11ms on edge devices)
    """

    def __init__(self, variant: str = "edgesam", **kwargs):
        self.variant = variant
        # "sam3" for full model, "edgesam" for edge-optimized

    def segment_by_prompt(self, frame: np.ndarray, prompt: str) -> np.ndarray:
        """Segment objects matching text prompt."""
        ...

    def track(self, video_frames: list[np.ndarray], initial_mask: np.ndarray) -> list[np.ndarray]:
        """Track segmented objects across video frames using SAM 3 tracker."""
        ...
```

### 5.4 Model Registry Updates

```python
# Updated model registry entries

MODEL_REGISTRY = {
    # Detection
    "yolo11n": {"class": "YOLODetector", "variant": "yolo11n"},
    "yolo11s": {"class": "YOLODetector", "variant": "yolo11s"},
    "yolo12n": {"class": "YOLODetector", "variant": "yolo12n"},
    "yolo26n": {"class": "YOLO26Detector", "variant": "yolo26n"},  # NEW
    "yolo26s": {"class": "YOLO26Detector", "variant": "yolo26s"},  # NEW
    "rtmdet_nano": {"class": "RTMDetDetector"},

    # Depth
    "midas-small": {"class": "DepthEstimator", "variant": "small"},
    "da3-small": {"class": "DepthAnythingV3", "variant": "da3-small"},  # NEW
    "da3-base": {"class": "DepthAnythingV3", "variant": "da3-base"},    # NEW

    # Segmentation
    "sam3": {"class": "SAM3Segmenter", "variant": "sam3"},       # NEW
    "edgesam": {"class": "SAM3Segmenter", "variant": "edgesam"}, # NEW
    "fastsam": {"class": "FastSAMSegmenter"},

    # Tracking
    "bytetrack": {"class": "ByteTracker"},
    "bot-sort": {"class": "BoTSORTTracker"},    # NEW
    "oc-sort": {"class": "OCSORTTracker"},      # NEW
    "sam3-tracker": {"class": "SAM3Tracker"},   # NEW
    "boosttrack": {"class": "BoostTracker"},    # NEW
}
```

---

## 6. Fleet Management

### 6.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Fleet Manager (CLI/Web)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Device   │  │ Model    │  │ Deploy   │  │ Telemetry  │  │
│  │ Registry │  │ Registry │  │ Manager  │  │ Dashboard  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
│       │              │              │              │          │
├───────┼──────────────┼──────────────┼──────────────┼─────────┤
│       │         Communication Layer (MQTT/HTTP)     │          │
│       └──────────────┬──────────────────────────────┘          │
│                      │                                          │
├──────────────────────┼──────────────────────────────────────────┤
│    Edge Devices      │                                          │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐        │
│  │Dev-1│  │Dev-2│  │Dev-3│  │Dev-4│  │Dev-5│  │Dev-N│  ...    │
│  │Jetson│ │Pi 5 │ │Jetson│ │Intel│ │Hailo│ │Qualc│             │
│  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Device Protocol

```python
# src/fleet/protocol.py

@dataclass
class DeviceHeartbeat:
    device_id: str
    group: str                    # "warehouse-robots", "qa-line-1", etc.
    platform: str                 # "jetson-orin-nano"
    uptime_seconds: int
    fps: float
    latency_ms: float
    cpu_percent: float
    gpu_percent: float
    memory_percent: float
    temperature_c: float
    model_versions: dict[str, str]  # {"detection": "yolo26n-v1.2", ...}
    error_count: int
    timestamp: float


@dataclass
class ModelDeployment:
    model_name: str
    version: str
    checksum: str                 # SHA256 of model file
    download_url: str
    target_devices: list[str]     # ["*"] for all, or specific device IDs
    target_groups: list[str]      # Deploy to device groups
    rollback_version: str | None  # Version to rollback to on failure
    requires_restart: bool


class FleetClient:
    """Client running on each edge device for fleet communication."""

    def __init__(self, device_id: str, server_url: str):
        self.device_id = device_id
        self.server_url = server_url
        self._heartbeat_interval = 30  # seconds

    async def send_heartbeat(self, heartbeat: DeviceHeartbeat):
        ...

    async def receive_deployment(self) -> ModelDeployment | None:
        ...

    async def report_status(self, status: dict):
        ...
```

### 6.3 Fleet CLI Commands

```bash
# Device management
openeyes fleet register --name warehouse-robot-01 --group warehouse
openeyes fleet list
openeyes fleet list --group warehouse
openeyes fleet info <device-id>

# Model deployment
openeyes fleet models list
openeyes fleet models upload --model yolo26n-fp16.engine --version v1.2
openeyes fleet deploy --model yolo26n --version v1.2 --group warehouse
openeyes fleet deploy --model yolo26n --version v1.2 --device robot-01

# Telemetry
openeyes fleet telemetry --device robot-01 --last 1h
openeyes fleet telemetry --group warehouse --metric fps
openeyes fleet dashboard --port 8080

# OTA
openeyes fleet ota check
openeyes fleet ota update --version v2.0.1 --group all
openeyes fleet ota rollback --device robot-03
```

### 6.4 Fleet Dashboard (Web)

```
src/fleet/dashboard/
├── index.html          # Device overview
├── devices.html        # Per-device detail
├── models.html         # Model registry
├── deployments.html    # Deployment history
├── telemetry.html      # Real-time metrics
├── alerts.html         # Error alerts
└── static/
    ├── css/
    ├── js/
    └── charts.js       # Real-time FPS/latency charts
```

---

## 7. Edge-Cloud Split Inference

### 7.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Edge Device                           │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ Light Model   │───│ Confidence   │───│ Local Result   │  │
│  │ (YOLO26n-INT8)│    │ Router       │    │ (< 50ms)      │  │
│  └──────────────┘    │              │    └───────────────┘  │
│                      │ If conf < T  │                       │
│                      │ or anomaly   │                       │
│                      └──────┬───────┘                       │
│                             │                                │
├─────────────────────────────┼────────────────────────────────┤
│                             │ HTTPS/gRPC                     │
│                         ┌───┴────┐                           │
│                         │ Cloud   │                           │
│                         │ Heavy   │                           │
│                         │ Model   │                           │
│                         │ (VLA 7B)│                           │
│                         └───┬────┘                           │
│                             │                                │
│                         ┌───┴────┐                           │
│                         │ Cloud   │                           │
│                         │ Result  │                           │
│                         │ (< 500ms)│                          │
│                         └────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Implementation

```python
# src/pipeline/edge_cloud_router.py

@dataclass
class RouterConfig:
    edge_model: str = "yolo26n-int8"
    cloud_model: str = "openvla-7b"
    confidence_threshold: float = 0.7
    max_cloud_latency_ms: int = 500
    edge_only_on_disconnect: bool = True
    cloud_url: str = ""
    cloud_api_key: str = ""


class EdgeCloudRouter:
    """Routes inference between edge and cloud based on confidence.

    Strategy:
    1. Run lightweight edge model (every frame, < 50ms)
    2. If confidence >= threshold: use edge result
    3. If confidence < threshold: send to cloud model
    4. If cloud unavailable: fall back to edge result with warning
    5. Track routing statistics for optimization
    """

    def __init__(self, config: RouterConfig):
        self.config = config
        self.edge_model = load_model(config.edge_model)
        self.stats = RoutingStats()

    def infer(self, frame: np.ndarray) -> InferenceResult:
        # Step 1: Edge inference
        edge_result = self.edge_model.detect(frame)
        self.stats.edge_inferences += 1

        # Step 2: Confidence check
        if edge_result.max_confidence >= self.config.confidence_threshold:
            return edge_result

        # Step 3: Cloud inference
        self.stats.cloud_routed += 1
        try:
            cloud_result = self._cloud_infer(frame)
            return cloud_result
        except CloudError:
            self.stats.cloud_failures += 1
            if self.config.edge_only_on_disconnect:
                return edge_result  # Fallback
            raise
```

---

## 8. Production Deployment Toolkit

### 8.1 Docker Images

```dockerfile
# Dockerfile.jetson-orin-nano
FROM nvcr.io/nvidia/l4t-jetpack:r36.2.0

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV OPENEYES_BACKEND=tensorrt
ENV OPENEYES_PRECISION=fp16

CMD ["python", "-m", "src.main", "--camera", "0", "--ros2"]
```

```dockerfile
# Dockerfile.pi5-ai-hat
FROM balenalib/raspberrypi5-64-debian:trixie

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV OPENEYES_BACKEND=hailo_dfc
ENV OPENEYES_PRECISION=int8

CMD ["python", "-m", "src.main", "--camera", "0"]
```

```dockerfile
# Dockerfile.intel-npu
FROM intel/oneapi-basekit:latest

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV OPENEYES_BACKEND=openvino
ENV OPENEYES_PRECISION=int8

CMD ["python", "-m", "src.main", "--camera", "0"]
```

### 8.2 Systemd Service

```ini
# scripts/openeyes.service
[Unit]
Description=OpenEyes Vision System
After=network.target

[Service]
Type=simple
User=openeyes
WorkingDirectory=/opt/openeyes
ExecStart=/opt/openeyes/venv/bin/python -m src.main --camera 0 --ros2 --health-monitor
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryMax=2G
CPUQuota=80%

# Security
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/opt/openeyes/logs /opt/openeyes/models

[Install]
WantedBy=multi-user.target
```

### 8.3 Prometheus Metrics

```python
# src/metrics/prometheus.py

from prometheus_client import Counter, Histogram, Gauge

# Metrics
inference_latency = Histogram(
    'openeyes_inference_latency_seconds',
    'Inference latency per stage',
    ['stage', 'model']
)

pipeline_fps = Gauge(
    'openeyes_pipeline_fps',
    'Current pipeline FPS'
)

model_confidence = Histogram(
    'openeyes_model_confidence',
    'Model confidence scores',
    ['stage']
)

errors_total = Counter(
    'openeyes_errors_total',
    'Total errors by type',
    ['type']
)

gpu_temperature = Gauge(
    'openeyes_gpu_temperature_celsius',
    'GPU temperature'
)

cloud_routing_ratio = Gauge(
    'openeyes_cloud_routing_ratio',
    'Percentage of inferences routed to cloud'
)
```

---

## 9. CLI Updates

### 9.1 New CLI Commands

```bash
# Platform detection
openeyes platform-info              # Show detected platform and capabilities
openeyes platform list              # List all supported platforms

# Backend management
openeyes backend list               # Show available backends
openeyes backend benchmark --all    # Benchmark all backends
openeyes backend select tensorrt    # Force specific backend

# Model export
openeyes export --model yolo26n --backend openvino --precision int8
openeyes export --model da3-small --backend hailo_dfc
openeyes export --all-models --backend tensorrt --precision fp16

# Pipeline management
openeyes pipeline list              # List available pipeline configs
openeyes pipeline validate warehouse.yaml
openeyes pipeline visualize warehouse.yaml
openeyes pipeline run --config warehouse.yaml

# Fleet management
openeyes fleet register --name robot-01 --group warehouse
openeyes fleet list
openeyes fleet deploy --model yolo26n --version v1.2 --group warehouse
openeyes fleet telemetry --device robot-01
openeyes fleet dashboard --port 8080

# Benchmarking
openeyes benchmark --model yolo26n --iterations 1000
openeyes benchmark --all-models --report
openeyes benchmark --compare yolo11n yolo26n

# Industry templates
openeyes init --template warehouse
openeyes init --template manufacturing-qa
openeyes init --template agriculture
openeyes init --template retail

# Compliance
openeyes compliance report          # Generate EU AI Act compliance report
openeyes compliance check           # Check compliance status
```

### 9.2 Updated CLI Flags

```
# New flags for v2.0.0
--backend           str     auto        Inference backend (auto/tensorrt/openvino/tvm/hailo/qnn)
--platform          str     auto        Target platform for model export
--pipeline          str     None        Pipeline configuration file
--template          str     None        Industry template to initialize
--fleet-server      str     None        Fleet management server URL
--fleet-device-id   str     None        Device ID for fleet registration
--fleet-group       str     None        Device group for fleet operations
--edge-cloud        flag    False       Enable edge-cloud split inference
--cloud-url         str     None        Cloud inference endpoint
--export            flag    False       Export models for target backend
--benchmark         flag    False       Run benchmark suite
--compliance        flag    False       Generate compliance report
--metrics-port      int     9090        Prometheus metrics port
```

---

## 10. Configuration Updates

### 10.1 Updated config.yaml

```yaml
# Updated configuration for v2.0.0

camera:
  source: 0
  width: 640
  height: 480
  fps: 30

backend:
  type: auto                    # auto, tensorrt, openvino, tvm, hailo_dfc, qnn, onnxruntime
  precision: auto               # auto, fp32, fp16, int8, int4
  workspace_size_gb: 1
  dla_enabled: false
  dla_core: 0

models:
  detection:
    model: yolo26n              # yolo11n, yolo11s, yolo12n, yolo26n, yolo26s
    confidence: 0.5
    iou_threshold: 0.45
  depth:
    model: da3-small            # midas-small, da3-small, da3-base
    enabled: true
    skip_frames: 3
  segmentation:
    model: edgesam              # sam3, edgesam, fastsam
    enabled: false
    skip_frames: 5
    confidence: 0.4
  tracking:
    algorithm: bytetrack        # bytetrack, bot-sort, oc-sort, sam3-tracker
    enabled: true
    max_age: 30
    min_hits: 3
    iou_threshold: 0.3
  face:
    enabled: false
    skip_frames: 3
  gesture:
    enabled: false
    skip_frames: 3
    confidence: 0.1
  pose:
    enabled: false
    skip_frames: 3

pipeline:
  config_file: ""               # Path to pipeline YAML config
  max_workers: 4
  zero_copy: true
  adaptive_scheduling: true     # Auto-adjust skip frames based on FPS
  target_fps: 30

edge_cloud:
  enabled: false
  cloud_url: ""
  cloud_api_key: ""
  confidence_threshold: 0.7
  max_cloud_latency_ms: 500
  edge_only_on_disconnect: true

fleet:
  enabled: false
  server_url: ""
  device_id: ""
  group: ""
  heartbeat_interval_sec: 30

metrics:
  enabled: true
  prometheus_port: 9090
  stats_interval_sec: 5

output:
  format: json
  protocol: udp
  host: 127.0.0.1
  port: 5000

ros2:
  enabled: false
  node_name: "openeyes_vision"
  qos_profile: "default"
  topics:
    detections: "/vision/detections"
    depth: "/vision/depth"
    faces: "/vision/faces"
    gestures: "/vision/gestures"
    poses: "/vision/poses"
    tracking: "/vision/tracks"
    status: "/vision/status"

safety:
  enabled: false
  max_velocity: 0.5
  min_distance: 0.3

compliance:
  face_blurring: false
  license_plate_blurring: false
  audit_trail: true
  data_retention_days: 30

debug: false
```

---

## 11. Directory Structure (v2.0.0)

```
openeyes/
├── src/
│   ├── __init__.py
│   ├── exceptions.py
│   ├── main.py
│   │
│   ├── backends/                    # NEW - Hardware Abstraction Layer
│   │   ├── __init__.py
│   │   ├── base.py                  # Backend abstract interface
│   │   ├── registry.py              # Backend discovery and selection
│   │   ├── tensorrt_backend.py      # Refactored from tensorrt_optimizer.py
│   │   ├── openvino_backend.py      # NEW
│   │   ├── tvm_backend.py           # NEW
│   │   ├── hailo_backend.py         # NEW
│   │   ├── qnn_backend.py           # NEW
│   │   └── onnxruntime_backend.py   # NEW (fallback)
│   │
│   ├── platforms/                   # NEW - Platform Detection
│   │   ├── __init__.py
│   │   ├── detector.py              # Auto-detection logic
│   │   ├── types.py                 # PlatformInfo dataclass
│   │   └── profiles/                # Per-platform configs
│   │       ├── jetson-orin-nano.yaml
│   │       ├── jetson-orin-nx.yaml
│   │       ├── jetson-t4000.yaml
│   │       ├── pi5-ai-hat.yaml
│   │       ├── intel-npu.yaml
│   │       ├── hailo-8.yaml
│   │       └── qualcomm-rb5.yaml
│   │
│   ├── camera/                      # (existing, minor updates)
│   │   ├── __init__.py
│   │   ├── camera_handler.py
│   │   └── types.py
│   │
│   ├── models/                      # (updated with new models)
│   │   ├── __init__.py
│   │   ├── object_detector.py       # Updated with YOLO26
│   │   ├── yolo26_detector.py       # NEW
│   │   ├── depth_estimator.py       # Updated with DA3
│   │   ├── depth_anything_v3.py     # NEW
│   │   ├── sam3_segmenter.py        # NEW
│   │   ├── face_detector.py
│   │   ├── gesture_recognizer.py
│   │   ├── pose_estimator.py
│   │   ├── model_registry.py        # Updated registry
│   │   ├── specialized.py
│   │   ├── tensorrt_optimizer.py    # Moved to backends/
│   │   ├── tensorrt_detector.py     # Moved to backends/
│   │   ├── action_chunker.py
│   │   ├── diffusion_policy.py
│   │   ├── lora_finetuning.py
│   │   ├── vla.py
│   │   └── vla_models.py
│   │
│   ├── tracking/                    # NEW - Dedicated tracking module
│   │   ├── __init__.py
│   │   ├── base.py                  # Tracker interface
│   │   ├── bytetrack.py             # Existing, refactored
│   │   ├── bot_sort.py              # NEW
│   │   ├── oc_sort.py               # NEW
│   │   ├── sam3_tracker.py          # NEW
│   │   └── boosttrack.py            # NEW
│   │
│   ├── pipeline/                    # (major update)
│   │   ├── __init__.py
│   │   ├── unified_pipeline.py      # NEW - Main pipeline orchestrator
│   │   ├── scheduler.py             # NEW - Frame scheduler
│   │   ├── edge_cloud_router.py     # NEW - Edge-cloud split
│   │   └── vision_pipeline.py       # Legacy (deprecated)
│   │
│   ├── fleet/                       # NEW - Fleet Management
│   │   ├── __init__.py
│   │   ├── client.py                # Edge device client
│   │   ├── server.py                # Fleet management server
│   │   ├── protocol.py              # Heartbeat/deployment protocol
│   │   ├── model_registry.py        # Model version registry
│   │   └── dashboard/               # Web dashboard
│   │       ├── __init__.py
│   │       ├── app.py
│   │       └── templates/
│   │
│   ├── metrics/                     # NEW - Observability
│   │   ├── __init__.py
│   │   ├── prometheus.py            # Prometheus metrics
│   │   └── health.py                # Health check endpoints
│   │
│   ├── compliance/                  # NEW - EU AI Act Compliance
│   │   ├── __init__.py
│   │   ├── anonymizer.py            # Face/plate blurring
│   │   ├── audit.py                 # Audit trail
│   │   └── reporter.py              # Compliance report generation
│   │
│   ├── templates/                   # NEW - Industry Templates
│   │   ├── __init__.py
│   │   ├── warehouse.yaml
│   │   ├── manufacturing_qa.yaml
│   │   ├── agriculture.yaml
│   │   └── retail.yaml
│   │
│   ├── core/                        # (existing, updated)
│   │   ├── __init__.py
│   │   ├── vision_system.py
│   │   ├── frame_processor.py
│   │   ├── initialization.py
│   │   └── ros2_bridge.py
│   │
│   ├── ros2/                        # (existing, minor updates)
│   │   ├── __init__.py
│   │   ├── vision_node.py
│   │   ├── services.py
│   │   ├── actions.py
│   │   ├── lidar_processing.py
│   │   ├── sensor_fusion.py
│   │   ├── multi_camera.py
│   │   └── ...
│   │
│   ├── output/                      # (existing)
│   │   ├── __init__.py
│   │   ├── json_formatter.py
│   │   └── udp_sender.py
│   │
│   ├── utils/                       # (existing, minor updates)
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── tracker.py               # Moved to tracking/
│   │   ├── performance_monitor.py
│   │   ├── frame_skipper.py         # Moved to pipeline/scheduler.py
│   │   ├── safety_controller.py
│   │   ├── health_monitor.py
│   │   └── ota_update.py            # Superseded by fleet/
│   │
│   ├── cli/                         # (updated with new commands)
│   │   ├── __init__.py
│   │   └── argparse.py
│   │
│   └── deepstream/                  # (existing, deprecated)
│       ├── __init__.py
│       └── pipeline.py
│
├── pipelines/                       # NEW - Pipeline configurations
│   ├── warehouse.yaml
│   ├── manufacturing_qa.yaml
│   ├── agriculture.yaml
│   └── retail.yaml
│
├── benchmarks/                      # NEW - Benchmark suite
│   ├── __init__.py
│   ├── run_benchmarks.py
│   ├── models/                      # Benchmark model definitions
│   └── reports/                     # Generated reports
│
├── docker/                          # NEW - Docker configurations
│   ├── Dockerfile.jetson-orin-nano
│   ├── Dockerfile.pi5-ai-hat
│   ├── Dockerfile.intel-npu
│   └── docker-compose.yml
│
├── scripts/                         # (updated)
│   ├── deploy.sh                    # Production deployment script
│   ├── openeyes.service             # Systemd service file
│   └── jetson_helper.py
│
├── tests/                           # (expanded)
│   ├── test_backends.py             # NEW
│   ├── test_platforms.py            # NEW
│   ├── test_pipeline.py             # NEW
│   ├── test_fleet.py                # NEW
│   ├── test_tracking.py             # NEW
│   ├── test_models_yolo26.py        # NEW
│   ├── test_models_da3.py           # NEW
│   ├── test_camera.py
│   ├── test_object_detector.py
│   ├── test_config.py
│   └── test_output.py
│
├── config.yaml
├── requirements.txt
├── pyproject.toml                   # NEW - Modern Python packaging
└── ...
```

---

## 12. Requirements Updates

```txt
# requirements.txt - Updated for v2.0.0

# Core (Required)
opencv-python>=4.8.0
numpy>=1.24.0
PyYAML>=6.0
python-dotenv>=1.0.0

# AI/ML (Required)
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.4.0
onnxruntime>=1.15.0
mediapipe>=0.10.9
timm>=1.0.0
psutil>=5.9.0

# NEW: VLA and advanced models
transformers>=4.40.0
peft>=0.10.0
accelerate>=0.28.0
safetensors>=0.4.0

# NEW: Hardware backends
openvino>=2024.1.0; platform_machine == "x86_64"
tvm>=0.15.0; platform_machine == "aarch64"

# NEW: Fleet management
paho-mqtt>=2.0.0
aiohttp>=3.9.0
jinja2>=3.1.0

# NEW: Observability
prometheus-client>=0.20.0

# NEW: Compliance
pillow>=10.0.0

# Communication
pyserial>=3.5

# Development (optional)
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.23.0
black>=24.0.0
isort>=5.0.0
mypy>=1.0.0
pylint>=3.0.0
```

---

## 13. Performance Targets

### 13.1 Per-Platform Targets

| Platform | Detection | Depth | Tracking | Full Pipeline | Power |
|----------|-----------|-------|----------|---------------|-------|
| Jetson Orin Nano | 35-40 FPS | 10-15 FPS | 30+ FPS | 6-10 FPS | 8-12W |
| Jetson Orin NX | 60-80 FPS | 20-30 FPS | 50+ FPS | 15-20 FPS | 15-25W |
| Jetson T4000 | 200+ FPS | 60+ FPS | 100+ FPS | 40-60 FPS | 40-70W |
| Pi 5 + AI HAT+ 2 | 25-30 FPS | 8-12 FPS | 25+ FPS | 5-8 FPS | ~12W |
| Intel Core Ultra | 20-30 FPS | 5-10 FPS | 20+ FPS | 4-7 FPS | 15-45W |
| Hailo-8 + x86 | 30-40 FPS | 10-15 FPS | 30+ FPS | 8-12 FPS | ~5W |

### 13.2 Model-Specific Targets (Jetson Orin Nano)

| Model | Variant | Target FPS | Precision | Memory |
|-------|---------|-----------|-----------|--------|
| YOLO26n | Detection | 35-40 | FP16 | 50MB |
| YOLO26s | Detection | 20-25 | FP16 | 200MB |
| DA3-Small | Depth | 10-15 | FP16 | 100MB |
| EdgeSAM | Segmentation | 25-35 | FP16 | 80MB |
| ByteTrack | Tracking | 30+ | N/A | 10MB |
| BoT-SORT | Tracking | 20-25 | FP16 | 50MB |

### 13.3 Latency Budget

| Component | Budget | Notes |
|-----------|--------|-------|
| Camera capture | < 5ms | GStreamer zero-copy |
| Detection | < 30ms | YOLO26n FP16 |
| Depth | < 100ms | DA3-Small, every 3rd frame |
| Tracking | < 10ms | ByteTrack |
| Output (ROS2/UDP) | < 5ms | Zero-copy where possible |
| **Total (per frame)** | **< 50ms** | Target for 20+ FPS pipeline |

---

## 14. Migration Guide (v1.0.0 → v2.0.0)

### 14.1 Breaking Changes

1. **Model names updated**: `yolo11n` still works, but default is now `yolo26n`
2. **Depth model changed**: `midas-small` deprecated, default is `da3-small`
3. **Config structure**: `performance.tensorrt.*` moved to `backend.*`
4. **Backend selection**: `--precision` and `--dla` flags replaced by `--backend` and backend-specific config
5. **OTA updates**: `--ota-update` superseded by fleet management system

### 14.2 Backward Compatibility

All v1.0.0 CLI flags continue to work with deprecation warnings:

```
WARNING: --precision is deprecated. Use --backend tensorrt --backend-precision fp16
WARNING: --dla is deprecated. Use --backend tensorrt --backend-dla 0
WARNING: --ota-update is deprecated. Use openeyes fleet ota check
```

### 14.3 Migration Steps

```bash
# 1. Update config (automatic migration tool)
openeyes migrate-config --input config.yaml --output config-v2.yaml

# 2. Re-export models for new backend
openeyes export --model yolo26n --backend tensorrt --precision fp16
openeyes export --model da3-small --backend tensorrt --precision fp16

# 3. Test with new defaults
openeyes start --camera 0 --backend auto

# 4. Verify performance
openeyes benchmark --report
```

---

## 15. Testing Strategy

### 15.1 Test Categories

| Category | Coverage Target | Tools |
|----------|----------------|-------|
| Unit tests | 80%+ | pytest |
| Integration tests | All backends | pytest + mock hardware |
| Benchmark tests | All platforms | Custom benchmark suite |
| Performance tests | Regression detection | pytest-benchmark |
| End-to-end tests | Full pipeline | Real hardware |

### 15.2 New Test Files

```
tests/
├── test_backends/
│   ├── test_tensorrt_backend.py
│   ├── test_openvino_backend.py
│   ├── test_tvm_backend.py
│   ├── test_hailo_backend.py
│   ├── test_qnn_backend.py
│   └── test_onnxruntime_backend.py
├── test_platforms/
│   ├── test_detector.py
│   └── test_profiles.py
├── test_pipeline/
│   ├── test_unified_pipeline.py
│   ├── test_scheduler.py
│   └── test_edge_cloud_router.py
├── test_fleet/
│   ├── test_client.py
│   ├── test_protocol.py
│   └── test_model_registry.py
├── test_tracking/
│   ├── test_bytetrack.py
│   ├── test_bot_sort.py
│   └── test_oc_sort.py
├── test_models/
│   ├── test_yolo26.py
│   ├── test_depth_anything_v3.py
│   └── test_sam3.py
├── test_compliance/
│   ├── test_anonymizer.py
│   └── test_audit.py
├── test_camera.py              # Existing
├── test_object_detector.py     # Existing
├── test_config.py              # Existing
└── test_output.py              # Existing
```

### 15.3 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run unit tests
        run: pytest tests/ --cov=src --cov-report=xml
      - name: Check coverage
        run: coverage report --fail-under=80

  benchmark:
    runs-on: [self-hosted, jetson-orin-nano]
    steps:
      - uses: actions/checkout@v4
      - name: Run benchmarks
        run: python benchmarks/run_benchmarks.py --report
      - name: Check regression
        run: python benchmarks/check_regression.py

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run linters
        run: |
          black --check src/
          isort --check src/
          mypy src/
          pylint src/ --fail-under=8.0
```

---

## 16. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| TensorRT backend regression during refactor | Medium | High | Comprehensive benchmark suite before/after |
| OpenVINO/Hailo backends delayed | Medium | Medium | Ship with TensorRT + ONNXRuntime first |
| VLA models still too large for edge | High | Low | Focus on edge-cloud split, not edge-only VLA |
| EU AI Act requirements change | Low | Medium | Design compliance system to be configurable |
| Hardware supply chain issues | Low | Medium | Support multiple platforms, no single-vendor lock-in |
| Community adoption slower than expected | Medium | Low | Focus on developer experience, 5-minute setup |

---

## 17. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| GitHub stars | 5,000+ by Q4 2026 | GitHub API |
| Active contributors | 20+ | GitHub contributors |
| Supported platforms | 7+ | Platform detection tests |
| Pipeline FPS (Orin Nano) | 10+ with full stack | Benchmark suite |
| Setup time | < 5 minutes | Timed installation test |
| Test coverage | 80%+ | pytest-cov |
| Documentation completeness | 100% of public APIs documented | Docstring checker |
| Fleet management devices | 100+ in testing | Fleet server metrics |

---

## 18. Appendix

### A. Competitive Analysis Summary

| Feature | OpenEyes v2.0 | Isaac ROS | LeRobot | yolo_ros |
|---------|--------------|-----------|---------|----------|
| Hardware support | 7+ platforms | NVIDIA only | Any (training) | NVIDIA only |
| Backend abstraction | Yes (HAL) | No | No | No |
| Fleet management | Yes | No | No | No |
| Edge-cloud split | Yes | No | No | No |
| Industry templates | 4 templates | None | None | None |
| EU AI Act compliance | Yes | Partial | No | No |
| Setup time | 5 minutes | Hours-days | Days | 30 minutes |
| Test coverage target | 80%+ | Unknown | ~40% | ~20% |

### B. Model Benchmark Reference

All benchmarks run on Jetson Orin Nano (8GB, JetPack 6.0, TensorRT FP16):

| Model | Input Size | Params | FPS | mAP/Quality | Memory |
|-------|-----------|--------|-----|-------------|--------|
| YOLO26n | 640x640 | 2.4M | 35-40 | 40.9% mAP | 50MB |
| YOLO26s | 640x640 | 9.5M | 20-25 | 48.6% mAP | 200MB |
| YOLO11n | 640x640 | 2.6M | 30-35 | 39.5% mAP | 55MB |
| DA3-Small | 640x480 | 25M | 10-15 | SOTA depth | 100MB |
| EdgeSAM | 1024x1024 | ~5M | 25-35 | Concept-aware | 80MB |
| ByteTrack | N/A | N/A | 30+ | HOTA 36.3 | 10MB |

### C. Glossary

| Term | Definition |
|------|-----------|
| HAL | Hardware Abstraction Layer |
| VLA | Vision-Language-Action model |
| DLA | Deep Learning Accelerator (Jetson) |
| DFC | Dataflow Compiler (Hailo) |
| QNN | Qualcomm Neural Network SDK |
| INT8/INT4 | 8-bit/4-bit integer quantization |
| FP16/FP8/FP4 | 16/8/4-bit floating point |
| QAT | Quantization-Aware Training |
| PTQ | Post-Training Quantization |
| ESDF | Euclidean Signed Distance Field |
| MOT | Multi-Object Tracking |
| HOTA | Higher Order Tracking Accuracy |
| RaaS | Robotics-as-a-Service |
