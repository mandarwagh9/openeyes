# API_SPEC.md - API Documentation for PROJECT0

> **Project**: PROJECT0 - Robot Vision System  
> **Version**: v0.0.1  
> **Last Updated**: 2026-03-13

---

## Table of Contents

1. [Camera API](#1-camera-api)
2. [Object Detection API](#2-object-detection-api)
3. [Depth Estimation API](#3-depth-estimation-api)
4. [Face Detection API](#4-face-detection-api)
5. [Gesture Recognition API](#5-gesture-recognition-api)
6. [Pose Estimation API](#6-pose-estimation-api)
7. [Output API](#7-output-api)
8. [Pipeline API](#8-pipeline-api)

---

## 1. Camera API

### 1.1 CameraHandler

Main class for camera input management.

```python
from src.camera import CameraHandler

handler = CameraHandler(
    source=0,           # Camera index or RTSP URL
    width=640,          # Frame width
    height=480,         # Frame height
    fps=30              # Target FPS
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `source` | `int \| str` | `0` | Camera index (0, 1...) or RTSP URL |
| `width` | `int` | `640` | Frame width |
| `height` | `int` | `480` | Frame height |
| `fps` | `int` | `30` | Target frames per second |

#### Methods

##### `read() -> np.ndarray | None`

Read a frame from the camera.

```python
frame = handler.read()
if frame is not None:
    # Process frame
    pass
```

**Returns:** NumPy array (H, W, 3) in BGR format, or `None` if no frame available.

##### `release() -> None`

Release camera resources.

```python
handler.release()
```

##### `is_opened() -> bool`

Check if camera is open.

```python
if handler.is_opened():
    print("Camera ready")
```

#### Properties

| Property | Type | Description |
|:---------|:-----|:-----------|
| `width` | `int` | Frame width |
| `height` | `int` | Frame height |
| `fps` | `int` | Target FPS |
| `is_opened` | `bool` | Camera status |

---

## 2. Object Detection API

### 2.1 ObjectDetector

YOLOv8-based object detection.

```python
from src.models import ObjectDetector

detector = ObjectDetector(
    model_path="models/yolov8n.pt",
    conf=0.5,           # Confidence threshold
    iou_threshold=0.45, # NMS IoU threshold
    device="cuda"       # cuda or cpu
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `model_path` | `str` | Required | Path to YOLOv8 model (.pt) |
| `conf` | `float` | `0.5` | Confidence threshold (0-1) |
| `iou_threshold` | `float` | `0.45` | NMS IoU threshold |
| `device` | `str` | `"cuda"` | Device for inference |

#### Methods

##### `detect(frame: np.ndarray) -> List[Detection]`

Detect objects in a frame.

```python
detections = detector.detect(frame)
for det in detections:
    print(f"{det.label}: {det.confidence:.2f}")
```

**Parameters:**
- `frame`: Input image as NumPy array (H, W, 3) BGR

**Returns:** List of `Detection` objects

#### 2.2 Detection Class

```python
@dataclass
class Detection:
    label: str           # Object class name
    confidence: float   # Confidence score (0-1)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int        # Class ID
```

---

## 3. Depth Estimation API

### 3.1 DepthEstimator

MiDaS-based depth estimation.

```python
from src.models import DepthEstimator

depth = DepthEstimator(
    model_path="models/depth_midas.pt",
    model_type="vitb",   # Model variant
    device="cuda"
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `model_path` | `str` | Required | Path to MiDaS model |
| `model_type` | `str` | `"vitb"` | Model variant (vits, vitb, vitl) |
| `device` | `str` | `"cuda"` | Device for inference |

#### Methods

##### `estimate(frame: np.ndarray) -> np.ndarray`

Estimate depth map from frame.

```python
depth_map = depth.estimate(frame)
# depth_map shape: (H, W), values 0-1
```

**Parameters:**
- `frame`: Input image as NumPy array (H, W, 3) BGR

**Returns:** Depth map as NumPy array (H, W), normalized 0-1

##### `distance_to_depth(distance_m: float) -> float`

Convert real distance to depth value.

```python
depth_value = depth.distance_to_depth(1.5)  # 1.5 meters
```

---

## 4. Face Detection API

### 4.1 FaceDetector

MediaPipe-based face detection.

```python
from src.models import FaceDetector

face_detector = FaceDetector(
    model_selection=0,   # 0 or 1 (short/long range)
    min_confidence=0.5
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `model_selection` | `int` | `0` | 0 for short-range, 1 for long-range |
| `min_confidence` | `float` | `0.5` | Minimum detection confidence |

#### Methods

##### `detect(frame: np.ndarray) -> List[Face]`

Detect faces in a frame.

```python
faces = face_detector.detect(frame)
for face in faces:
    print(f"Face at {face.bbox}, confidence: {face.confidence}")
```

---

## 5. Gesture Recognition API

### 5.1 GestureRecognizer

MediaPipe-based hand gesture recognition.

```python
from src.models import GestureRecognizer

gesture = GestureRecognizer()
```

#### Methods

##### `recognize(frame: np.ndarray) -> List[Gesture]`

Recognize hand gestures.

```python
gestures = gesture.recognize(frame)
for g in gestures:
    print(f"Gesture: {g.gesture_type}, Hand: {g.handedness}")
```

#### 5.2 Gesture Class

```python
@dataclass
class Gesture:
    gesture_type: str   # "thumbs_up", "stop", "wave", "point", "open_palm"
    handedness: str      # "left" or "right"
    confidence: float    # Confidence score
    landmarks: List[Tuple[float, float, float]]  # 21 hand landmarks
```

---

## 6. Pose Estimation API

### 6.1 PoseEstimator

MediaPipe-based body pose estimation.

```python
from src.models import PoseEstimator

pose = PoseEstimator()
```

#### Methods

##### `estimate(frame: np.ndarray) -> Optional[PoseResult]`

Estimate body pose.

```python
result = pose.estimate(frame)
if result:
    print(f"Pose detected with {len(result.keypoints)} keypoints")
```

#### 6.2 PoseResult Class

```python
@dataclass
class PoseResult:
    keypoints: List[PoseKeypoint]  # 33 body keypoints
    score: float                    # Overall pose confidence
    bounding_box: Tuple[int, int, int, int]

@dataclass
class PoseKeypoint:
    name: str           # "nose", "left_shoulder", etc.
    x: float           # Normalized x (0-1)
    y: float           # Normalized y (0-1)
    z: float           # Depth estimate
    confidence: float  # Visibility score
```

---

## 7. Output API

### 7.1 VisionOutput

Handles output transmission.

```python
from src.output import VisionOutput

output = VisionOutput(
    format="json",       # "json" or "binary"
    protocol="udp",      # "udp" or "tcp"
    host="127.0.0.1",
    port=5000,
    fps=30
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `format` | `str` | `"json"` | Output format |
| `protocol` | `str` | `"udp"` | Network protocol |
| `host` | `str` | `"127.0.0.1"` | Target host |
| `port` | `int` | `5000` | Target port |
| `fps` | `int` | `30` | Output rate |

#### Methods

##### `send(data: VisionResult) -> None`

Send vision result.

```python
result = VisionResult(
    timestamp=time.time(),
    frame_id=123,
    objects=detections,
    depth=depth_map,
    faces=faces,
    gestures=gestures,
    pose=pose_result
)
output.send(result)
```

##### `close() -> None`

Close output connection.

---

## 8. Pipeline API

### 8.1 VisionPipeline

Main processing pipeline orchestrator.

```python
from src.pipeline import VisionPipeline

pipeline = VisionPipeline(config=config)
pipeline.start()

# Process frames
while pipeline.is_running():
    result = pipeline.process_frame(frame)
    # Handle result

pipeline.stop()
```

#### Constructor Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `config` | `Config` | Configuration object |

#### Methods

##### `start() -> None`

Start the pipeline.

##### `stop() -> None`

Stop the pipeline.

##### `process_frame(frame: np.ndarray) -> VisionResult`

Process a single frame.

##### `is_running() -> bool`

Check if pipeline is running.

#### 8.2 VisionResult

```python
@dataclass
class VisionResult:
    timestamp: float
    frame_id: int
    objects: List[Detection]
    depth: Optional[np.ndarray]
    faces: List[Face]
    gestures: List[Gesture]
    pose: Optional[PoseResult]
```

---

## 9. Type Definitions

### 9.1 Common Types

```python
# src/camera/types.py

from typing import TypedDict

class CameraConfig(TypedDict):
    source: int | str
    width: int
    height: int
    fps: int

class ModelConfig(TypedDict):
    path: str
    device: str
    confidence: float
```

---

## 10. Error Handling

### 10.1 Exceptions

```python
from src.camera.exceptions import CameraError
from src.models.exceptions import ModelError
from src.output.exceptions import OutputError

# Catch specific exceptions
try:
    detector = ObjectDetector("models/yolov8n.pt")
except ModelError as e:
    print(f"Failed to load model: {e}")
```

### 10.2 Exception Hierarchy

```
VisionError (base)
├── CameraError
│   ├── CameraNotFoundError
│   └── CameraReadError
├── ModelError
│   ├── ModelLoadError
│   └── InferenceError
└── OutputError
    ├── OutputConnectionError
    └── OutputSendError
```

---

## 11. Usage Examples

### 11.1 Simple Object Detection

```python
import cv2
from src.camera import CameraHandler
from src.models import ObjectDetector

# Initialize
camera = CameraHandler(source=0)
detector = ObjectDetector("models/yolov8n.pt")

# Process
frame = camera.read()
if frame is not None:
    detections = detector.detect(frame)
    for det in detections:
        # Draw bbox
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{det.label} {det.confidence:.2f}", 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

cv2.imwrite("output.jpg", frame)
camera.release()
```

### 11.2 Full Pipeline

```python
from src.pipeline import VisionPipeline
from src.utils.config import load_config

# Load configuration
config = load_config("config/default.yaml")

# Create and run pipeline
pipeline = VisionPipeline(config)
pipeline.start()

# Run for 100 frames
for i in range(100):
    result = pipeline.get_result()  # Non-blocking
    if result:
        print(f"Frame {result.frame_id}: {len(result.objects)} objects")

pipeline.stop()
```

---

## Appendix: Output JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "timestamp": { "type": "number" },
    "frame_id": { "type": "integer" },
    "objects": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "label": { "type": "string" },
          "confidence": { "type": "number" },
          "bbox": { "type": "array", "items": { "type": "integer" } }
        }
      }
    },
    "depth": {
      "type": "object",
      "properties": {
        "enabled": { "type": "boolean" },
        "min_distance": { "type": "number" },
        "max_distance": { "type": "number" }
      }
    },
    "faces": { "type": "array" },
    "gestures": { "type": "array" },
    "pose": { "type": "object" }
  }
}
```
