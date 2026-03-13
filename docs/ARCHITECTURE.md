# ARCHITECTURE.md - System Architecture for PROJECT0

> **Project**: PROJECT0 - Robot Vision System  
> **Version**: v0.0.1  
> **Last Updated**: 2026-03-13

---

## 1. High-Level Architecture

### 1.1 System Overview

PROJECT0 follows a layered architecture designed for real-time edge AI processing:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PROJECT0                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Input     │    │  Processing  │    │   Output    │               │
│  │   Layer    │───▶│    Layer    │───▶│    Layer    │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        Hardware Layer                             │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │  │
│  │  │  Camera    │  │ Jetson     │  │  Network  │                  │  │
│  │  │  (USB)    │  │ Orin Nano  │  │  (UDP)    │                  │  │
│  │  └────────────┘  └────────────┘  └────────────┘                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow

```
Camera Frame (BGR)
       │
       ▼
┌─────────────────┐
│  Preprocessing │  ── Resize ── Normalize ── Convert to Tensor
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AI Inference  │  ── YOLOv8 ── MiDaS ── MediaPipe
│   (TensorRT)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Postprocessing  │  ── NMS ── Format ── Package
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   JSON Output  │  ── UDP Socket ── External System
└─────────────────┘
```

---

## 2. Component Architecture

### 2.1 Input Layer

```
┌─────────────────────────────────────────────┐
│              Input Layer                     │
├─────────────────────────────────────────────┤
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │       CameraHandler                  │    │
│  │  ─────────────────────────────────  │    │
│  │  • Open camera connection           │    │
│  │  • Read frames                      │    │
│  │  • Handle disconnects               │    │
│  │  • Manage multiple cameras          │    │
│  └─────────────────────────────────────┘    │
│                                              │
└─────────────────────────────────────────────┘
```

**Responsibilities:**
- Initialize camera (USB/RTSP)
- Capture frames at configured FPS
- Handle camera errors gracefully
- Buffer management

**Public Interface:**
```python
class CameraHandler:
    def __init__(self, source: int | str, width: int, height: int, fps: int)
    def read() -> np.ndarray | None
    def release() -> None
    @property
    def is_opened() -> bool
```

### 2.2 Processing Layer

```
┌─────────────────────────────────────────────┐
│            Processing Layer                  │
├─────────────────────────────────────────────┤
│                                              │
│  ┌─────────────┐ ┌─────────────┐            │
│  │  Object     │ │   Depth     │            │
│  │  Detector   │ │  Estimator  │            │
│  │  (YOLOv8)   │ │   (MiDaS)   │            │
│  └─────────────┘ └─────────────┘            │
│                                              │
│  ┌─────────────┐ ┌─────────────┐            │
│  │    Face    │ │  Gesture    │            │
│  │  Detector  │ │  Recognizer │            │
│  │ (MediaPipe)│ │ (MediaPipe) │            │
│  └─────────────┘ └─────────────┘            │
│                                              │
│  ┌─────────────┐ ┌─────────────┐            │
│  │    Pose    │ │  Inference  │            │
│  │ Estimator  │ │   Engine    │            │
│  │ (MediaPipe)│ │ (TensorRT)  │            │
│  └─────────────┘ └─────────────┘            │
│                                              │
└─────────────────────────────────────────────┘
```

#### 2.2.1 Object Detector

```
┌─────────────────────────────────────────────┐
│            ObjectDetector                   │
├─────────────────────────────────────────────┤
│                                              │
│  Input:  frame (H, W, 3) BGR                │
│  Output: List[Detection]                     │
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │ 1. Preprocess                       │    │
│  │    • Resize to model input          │    │
│  │    • Normalize (0-1)                │    │
│  │    • Convert to blob                │    │
│  └─────────────────────────────────────┘    │
│                 │                             │
│                 ▼                             │
│  ┌─────────────────────────────────────┐    │
│  │ 2. Inference (YOLOv8n)             │    │
│  │    • TensorRT optimized             │    │
│  │    • GPU acceleration               │    │
│  └─────────────────────────────────────┘    │
│                 │                             │
│                 ▼                             │
│  ┌─────────────────────────────────────┐    │
│  │ 3. Postprocess                      │    │
│  │    • Extract bounding boxes         │    │
│  │    • Filter by confidence           │    │
│  │    • NMS for overlapping boxes      │    │
│  └─────────────────────────────────────┘    │
│                                              │
└─────────────────────────────────────────────┘
```

#### 2.2.2 Depth Estimator

```
┌─────────────────────────────────────────────┐
│            DepthEstimator                   │
├─────────────────────────────────────────────┤
│                                              │
│  Input:  frame (H, W, 3) BGR                │
│  Output: depth_map (H, W) normalized        │
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │ 1. Preprocess                       │    │
│  │    • Resize to 384x384              │    │
│  │    • Normalize ImageNet stats        │    │
│  └─────────────────────────────────────┘    │
│                 │                             │
│                 ▼                             │
│  ┌─────────────────────────────────────┐    │
│  │ 2. Inference (MiDaS v2.1)          │    │
│  │    • Single forward pass            │    │
│  │    • Output depth map              │    │
│  └─────────────────────────────────────┘    │
│                 │                             │
│                 ▼                             │
│  ┌─────────────────────────────────────┐    │
│  │ 3. Postprocess                      │    │
│  │    • Resize to original resolution │    │
│  │    • Normalize to 0-1 range         │    │
│  └─────────────────────────────────────┘    │
│                                              │
└─────────────────────────────────────────────┘
```

#### 2.2.3 Inference Engine

```
┌─────────────────────────────────────────────┐
│             InferenceEngine                  │
├─────────────────────────────────────────────┤
│                                              │
│  Responsibilities:                          │
│  • Load ONNX/TFLite models                  │
│  • TensorRT optimization                    │
│  • GPU memory management                    │
│  • Batch inference                          │
│                                              │
│  Public Interface:                          │
│  ────────────────────                       │
│  class InferenceEngine:                     │
│      def load_model(path: str) -> None      │
│      def infer(input_data: Tensor) -> Tensor│
│      def optimize() -> None                  │
│                                              │
└─────────────────────────────────────────────┘
```

### 2.3 Output Layer

```
┌─────────────────────────────────────────────┐
│              Output Layer                    │
├─────────────────────────────────────────────┤
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │         VisionOutput                 │    │
│  │  ─────────────────────────────────  │    │
│  │  • JSON Formatter                   │    │
│  │  • UDP Publisher                    │    │
│  │  • WebSocket Server (optional)      │    │
│  │  • File Logger (optional)           │    │
│  └─────────────────────────────────────┘    │
│                                              │
└─────────────────────────────────────────────┘
```

**Output Format:**
```json
{
  "timestamp": 1699123456.123,
  "frame_id": 1234,
  "objects": [...],
  "depth": {...},
  "faces": [...],
  "gestures": [...],
  "pose": {...}
}
```

---

## 3. Module Architecture

### 3.1 Directory Structure

```
src/
├── camera/
│   ├── __init__.py
│   ├── camera_handler.py      # Camera abstraction
│   ├── types.py               # Type definitions
│   └── exceptions.py          # Camera exceptions
│
├── models/
│   ├── __init__.py
│   ├── object_detector.py     # YOLOv8 wrapper
│   ├── depth_estimator.py    # MiDaS wrapper
│   ├── face_detector.py      # MediaPipe face
│   ├── gesture_recognizer.py  # MediaPipe hands
│   └── pose_estimator.py     # MediaPipe pose
│
├── inference/
│   ├── __init__.py
│   ├── engine.py              # TensorRT wrapper
│   └── optimizer.py           # Model optimization
│
├── output/
│   ├── __init__.py
│   ├── json_formatter.py     # JSON formatting
│   ├── udp_sender.py         # UDP transmission
│   └── logger.py            # File logging
│
├── pipeline/
│   ├── __init__.py
│   └── vision_pipeline.py    # Main processing pipeline
│
├── utils/
│   ├── __init__.py
│   ├── logger.py            # Logging setup
│   ├── config.py            # Configuration
│   └── transforms.py        # Image transforms
│
└── main.py                  # Entry point
```

### 3.2 Key Class Diagrams

#### VisionPipeline

```
┌─────────────────────────────────────────────┐
│           VisionPipeline                    │
├─────────────────────────────────────────────┤
│ - camera: CameraHandler                      │
│ - detector: ObjectDetector                  │
│ - depth: DepthEstimator                      │
│ - face: FaceDetector                         │
│ - gesture: GestureRecognizer                 │
│ - pose: PoseEstimator                        │
│ - output: VisionOutput                       │
├─────────────────────────────────────────────┤
│ + __init__(config: Config)                  │
│ + start() -> None                           │
│ + stop() -> None                            │
│ + process_frame() -> VisionResult           │
│ + is_running() -> bool                      │
└─────────────────────────────────────────────┘
```

---

## 4. Integration Points

### 4.1 External Control System

```
PROJECT0                          External System
    │                                     │
    │  ◄──── UDP JSON (5000) ──────────  │
    │                                     │
    │                                     │
```

**Integration Protocol:**
- Protocol: UDP
- Port: 5000 (configurable)
- Format: JSON
- Rate: Configurable (default: every frame)

### 4.2 ROS2 Migration Path

Future ROS2 integration will use:

```
┌─────────────────────────────────────────────┐
│         ROS2 Integration (Future)           │
├─────────────────────────────────────────────┤
│                                              │
│  VisionPipeline → ROS2 Bridge → rclpy       │
│                                              │
│  Topics:                                     │
│  • /vision/objects                          │
│  • /vision/depth                            │
│  • /vision/faces                            │
│  • /vision/gestures                        │
│  • /vision/pose                             │
│                                              │
└─────────────────────────────────────────────┘
```

---

## 5. Configuration

### 5.1 Default Configuration

```python
CONFIG = {
    "camera": {
        "source": 0,
        "width": 640,
        "height": 480,
        "fps": 30
    },
    "models": {
        "yolo": {
            "path": "models/yolov8n.pt",
            "confidence": 0.5,
            "iou_threshold": 0.45
        },
        "depth": {
            "path": "models/depth_midas.pt",
            "enabled": True
        },
        "face": {
            "enabled": True,
            "model": "short"
        },
        "gesture": {
            "enabled": True
        },
        "pose": {
            "enabled": True
        }
    },
    "output": {
        "format": "json",
        "protocol": "udp",
        "host": "127.0.0.1",
        "port": 5000,
        "fps": 30
    },
    "performance": {
        "target_fps": 30,
        "max_latency_ms": 50,
        "use_tensorrt": True
    }
}
```

---

## 6. Error Handling

### 6.1 Error Categories

| Category | Handling |
|:---------|:---------|
| Camera Disconnect | Auto-reconnect with exponential backoff |
| Model Load Failure | Fallback to CPU inference |
| Inference Timeout | Skip frame, log warning |
| Output Send Failure | Log error, continue processing |

### 6.2 Graceful Degradation

If a specific vision module fails:
1. Disable that module
2. Continue with remaining modules
3. Log warning with module status
4. Attempt recovery periodically

---

## 7. Performance Optimization

### 7.1 GPU Optimization

- TensorRT FP16 inference
- CUDA stream management
- GPU memory pooling
- Model batching where applicable

### 7.2 CPU Optimization

- Multi-threaded frame capture
- Async I/O for output
- Efficient NumPy operations
- OpenCV GPU backend

### 7.3 Memory Optimization

- Frame buffer pooling
- Model lazy loading
- Result object recycling

---

## 8. Security

### 8.1 Network Security

- No inbound connections required
- Outbound only (UDP transmission)
- Optional TLS for UDP (future)

### 8.2 Data Privacy

- All processing on-device
- No video data transmitted
- Optional face embedding storage (encrypted)

---

## Appendix: Sequence Diagrams

### Frame Processing Sequence

```
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│ Camera │    │ Prepro │    │ YOLOv8 │    │  NMS  │    │ Output │
└───┬────┘    └───┬────┘    └───┬────┘    └───┬────┘    └───┬────┘
    │            │            │            │            │
    │ read()     │            │            │            │
    │───────────>│            │            │            │
    │            │ preprocess │            │            │
    │            │───────────>│            │            │
    │            │            │ infer()    │            │
    │            │            │───────────>│            │
    │            │            │            │ filter()   │
    │            │            │            │───────────>│
    │            │            │            │   format() │
    │            │            │            │───────────>│
    │            │            │            │   send()   │
    │            │            │            │───────────>│
    │            │            │            │            │
```
