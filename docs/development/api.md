# API Reference

Complete API documentation for OpenEyes.

---

## Camera API

### CameraHandler

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

##### `is_opened() -> bool`

Check if camera is open.

---

## Object Detection API

### ObjectDetector

YOLO-based object detection.

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
| `model_path` | `str` | Required | Path to YOLO model (.pt) |
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

### Detection Class

```python
@dataclass
class Detection:
    label: str                              # Object class name
    confidence: float                       # Confidence score (0-1)
    bbox: Tuple[int, int, int, int]         # (x1, y1, x2, y2)
    class_id: int                           # Class ID
```

---

## Depth Estimation API

### DepthEstimator

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

---

## Face Detection API

### FaceDetector

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

---

## Gesture Recognition API

### GestureRecognizer

MediaPipe-based hand gesture recognition.

```python
from src.models import GestureRecognizer

gesture = GestureRecognizer()
```

#### Methods

##### `recognize(frame: np.ndarray) -> List[Gesture]`

Recognize hand gestures.

### Gesture Class

```python
@dataclass
class Gesture:
    gesture_type: str                                    # "thumbs_up", "stop", "wave", "open_palm"
    handedness: str                                      # "left" or "right"
    confidence: float                                    # Confidence score
    landmarks: List[Tuple[float, float, float]]        # 21 hand landmarks
```

---

## Pose Estimation API

### PoseEstimator

MediaPipe-based body pose estimation.

```python
from src.models import PoseEstimator

pose = PoseEstimator()
```

#### Methods

##### `estimate(frame: np.ndarray) -> Optional[PoseResult]`

Estimate body pose.

### PoseResult Class

```python
@dataclass
class PoseResult:
    keypoints: List[PoseKeypoint]     # 33 body keypoints
    score: float                      # Overall pose confidence
    bounding_box: Tuple[int, int, int, int]
```

---

## Output API

### VisionOutput

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

---

## Error Handling

### Exceptions

```python
from src.camera.exceptions import CameraError
from src.models.exceptions import ModelError
from src.output.exceptions import OutputError

try:
    detector = ObjectDetector("models/yolov8n.pt")
except ModelError as e:
    print(f"Failed to load model: {e}")
```

### Exception Hierarchy

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

## Usage Examples

### Simple Object Detection

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
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

camera.release()
```

---

## Output JSON Schema

```json
{
  "timestamp": 1699123456.123,
  "frame_id": 1234,
  "objects": [
    {
      "label": "person",
      "confidence": 0.95,
      "bbox": [100, 50, 300, 400]
    }
  ],
  "depth": {
    "enabled": true,
    "min_distance": 1.2,
    "max_distance": 5.0
  },
  "faces": [],
  "gestures": [
    {
      "type": "thumbs_up",
      "handedness": "right"
    }
  ],
  "pose": null
}
```