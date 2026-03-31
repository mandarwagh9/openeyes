# Architecture

System architecture and design for OpenEyes.

---

## High-Level Architecture

OpenEyes follows a layered architecture designed for real-time edge AI processing:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           OpenEyes                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Input     │    │  Processing  │    │   Output    │               │
│  │   Layer    │───▶│    Layer    │───▶│    Layer    │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                        Hardware Layer                             │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │    │
│  │  │  Camera    │  │ Jetson     │  │  Network  │                  │    │
│  │  │  (USB)    │  │ Orin Nano  │  │  (UDP)    │                  │    │
│  │  └────────────┘  └────────────┘  └────────────┘                  │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

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
│  AI Inference  │  ── YOLO ── MiDaS ── MediaPipe
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

## Component Architecture

### Input Layer

- **CameraHandler**: Opens camera, reads frames, handles disconnects
- Supports CSI (IMX219) and USB webcams

### Processing Layer

| Component | Model | Purpose |
|:---------|:------|:--------|
| ObjectDetector | YOLO11n | Object detection |
| DepthEstimator | MiDaS v2.1 | Depth estimation |
| FaceDetector | MediaPipe Face | Face detection |
| GestureRecognizer | MediaPipe Hands | Gesture recognition |
| PoseEstimator | MediaPipe Pose | Body pose estimation |

### Output Layer

- **JSON Formatter**: Formats results to JSON
- **UDP Sender**: Sends JSON via UDP socket
- **ROS2 Publisher**: Publishes to ROS2 topics (optional)

---

## Directory Structure

```
src/
├── camera/
│   ├── camera_handler.py      # Camera abstraction
│   ├── types.py               # Type definitions
│   └── exceptions.py          # Camera exceptions
│
├── models/
│   ├── object_detector.py     # YOLO wrapper
│   ├── depth_estimator.py    # MiDaS wrapper
│   ├── face_detector.py      # MediaPipe face
│   ├── gesture_recognizer.py  # MediaPipe hands
│   └── pose_estimator.py     # MediaPipe pose
│
├── output/
│   ├── json_formatter.py      # JSON formatting
│   ├── udp_sender.py          # UDP transmission
│   └── logger.py              # File logging
│
├── ros2/
│   ├── vision_node.py         # ROS2 node
│   ├── actions.py             # Action server
│   └── services.py            # Service definitions
│
├── utils/
│   ├── config.py              # Configuration
│   ├── logger.py              # Logging setup
│   └── transforms.py          # Image transforms
│
├── cli/
│   └── argparse.py            # CLI parsing
│
├── core/
│   ├── vision_system.py       # Main vision system
│   ├── frame_processor.py    # Frame processing
│   ├── initialization.py     # Component init
│   └── ros2_bridge.py        # ROS2 abstraction
│
└── main.py                    # Entry point
```

---

## Key Classes

### VisionSystem

```python
class VisionSystem:
    def __init__(config: Config)
    def start() -> None
    def stop() -> None
    def process_frame(frame: np.ndarray) -> VisionResult
    @property
    def is_running() -> bool
```

### FrameProcessor

```python
class FrameProcessor:
    def __init__(detector, depth, face, gesture, pose, tracker)
    def process(frame: np.ndarray) -> ProcessingResult
    def _process_all_models(frame: np.ndarray) -> Dict
```

---

## Integration Points

### External Control System

```
OpenEyes                          External System
     │                                     │
     │  ◄──── UDP JSON (5000) ──────────  │
     │                                     │
```

**Integration Protocol:**
- Protocol: UDP
- Port: 5000 (configurable)
- Format: JSON
- Rate: Every frame

### ROS2 Integration

```
VisionPipeline → ROS2 Bridge → rclpy
```

**Topics:**
- `/vision/detections` - Object detections
- `/vision/depth` - Depth map
- `/vision/faces` - Face detections
- `/vision/gestures` - Gesture recognitions
- `/vision/poses` - Body poses

---

## Error Handling

| Category | Handling |
|:---------|:---------|
| Camera Disconnect | Auto-reconnect with exponential backoff |
| Model Load Failure | Fallback to CPU inference |
| Inference Timeout | Skip frame, log warning |
| Output Send Failure | Log error, continue processing |

### Graceful Degradation

If a specific vision module fails:
1. Disable that module
2. Continue with remaining modules
3. Log warning with module status
4. Attempt recovery periodically

---

## Performance Optimization

### GPU Optimization

- TensorRT FP16 inference
- CUDA stream management
- GPU memory pooling
- Model batching where applicable

### CPU Optimization

- Multi-threaded frame capture
- Async I/O for output
- Efficient NumPy operations

### Memory Optimization

- Frame buffer pooling
- Model lazy loading
- Result object recycling