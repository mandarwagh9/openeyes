# Configuration

Detailed configuration options for OpenEyes.

---

## Config File

Create or edit `config.yaml`:

```yaml
camera:
  source: 0
  width: 640
  height: 480
  fps: 30

models:
  yolo:
    path: models/yolov8n.pt
    confidence: 0.5
    iou_threshold: 0.45
  depth:
    enabled: true
    path: models/depth_midas.pt
  face:
    enabled: true
    confidence: 0.5
  gesture:
    enabled: true
  pose:
    enabled: true

output:
  format: json
  protocol: udp
  host: 127.0.0.1
  port: 5000
  fps: 30

ros2:
  enabled: false
  node_name: "openeyes_vision"
  topics:
    detections: "/vision/detections"
    depth: "/vision/depth"
    faces: "/vision/faces"
    gestures: "/vision/gestures"
    poses: "/vision/poses"
    cmd: "/vision/cmd"
    status: "/vision/status"
  frame_id: "camera_link"
  confidence_threshold: 0.5
  max_depth_range: 5.0
```

---

## Camera Settings

| Setting | Description | Default |
|:--------|:------------|:--------|
| `source` | Camera index (0, 1...) or RTSP URL | 0 |
| `width` | Frame width | 640 |
| `height` | Frame height | 480 |
| `fps` | Target FPS | 30 |

---

## Model Settings

### YOLO

| Setting | Description | Default |
|:--------|:------------|:--------|
| `path` | Path to model file | models/yolov8n.pt |
| `confidence` | Detection threshold (0-1) | 0.5 |
| `iou_threshold` | NMS IoU threshold | 0.45 |

### Depth

| Setting | Description | Default |
|:--------|:------------|:--------|
| `enabled` | Enable depth estimation | true |
| `path` | Path to MiDaS model | models/depth_midas.pt |

### Face/Gesture/Pose

| Setting | Description | Default |
|:--------|:------------|:--------|
| `enabled` | Enable detection | true |
| `confidence` | Detection threshold | 0.5 |

---

## Output Settings

| Setting | Description | Default |
|:--------|:------------|:--------|
| `format` | Output format (json) | json |
| `protocol` | Network protocol (udp/tcp) | udp |
| `host` | Target host IP | 127.0.0.1 |
| `port` | Target port | 5000 |
| `fps` | Output rate | 30 |

---

## ROS2 Settings

| Setting | Description | Default |
|:--------|:------------|:--------|
| `enabled` | Enable ROS2 | false |
| `node_name` | ROS2 node name | openeyes_vision |
| `frame_id` | TF frame ID | camera_link |
| `confidence_threshold` | Detection threshold | 0.5 |
| `max_depth_range` | Max depth in meters | 5.0 |

---

## Performance Tuning

For best performance on Jetson:

```bash
# Run at maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Model Precision

| Precision | Speed | Accuracy |
|:----------|:------|:--------|
| fp32 | Slowest | Highest |
| fp16 | Fast | High |
| int8 | Fastest | Good |

Use precision in command:
```bash
python src/main.py --precision int8
```