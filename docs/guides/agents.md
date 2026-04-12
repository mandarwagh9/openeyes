# AGENTS.md - OpenEyes Developer Guidelines

> Hardware-agnostic edge robot vision framework with world models  
> **Version**: v2.5.0

---

## Critical Commands

```bash
# Entry point - MUST use -m flag
python -m src.main --camera 0 --debug

# Video processing (v2.5.0)
python -m src.main --video path/to/video.mp4 --output output.mp4

# Person following with world model
python -m src.main --world-model lewm --follow --turbo

# REST API server
python -m src.main --api --api-port 8000 --api-host 0.0.0.0

# Testing
pytest tests/ -v
pytest tests/test_camera.py -x  # Stop on first failure

# ROS2 launch
ros2 launch openeyes openeyes.launch.py device:=cuda ros2:=true
```

---

## Architecture

```
src/
├── main.py              # Entry point (use python -m src.main)
├── cli/argparse.py      # All CLI flags
├── camera/             # CameraHandler, video_source
├── core/               # VisionSystem, frame_processor
├── models/              # ObjectDetector, depth_estimator
├── ros2/                # VisionPublisher
├── utils/               # config, logger, tracker, safety_controller
└── world_model/         # LeWorldModel, V-JEPA
```

### Data Flow
Camera/Video → Detection → Tracking → Depth → [World Model] → Output

---

## Key Discoveries

### CSI Camera Not Available
- **Fix**: Check `/dev/video0` exists, add queue to GStreamer pipeline, reboot
- **File**: `src/camera/camera_handler.py`

### ROS2 Topics Not Publishing
- **Fix**: Use MultiThreadedExecutor in separate thread, add `time.sleep(0.5)` after init
- **File**: `src/ros2/vision_node.py`

### MediaPipe Empty Results
- **Fix**: Lower confidence to 0.3 for face/pose, 0.1 for hands, resize to 640x480
- **Files**: `src/models/face_detector.py`, `src/models/gesture_recognizer.py`

### Person Following Distance
- **Fix**: Use bounding box HEIGHT RATIO (% of frame): forward <60%, stop 60-95%, backward >95%
- **File**: `src/utils/tracker.py`

---

## Style

- Python 3.10+, type hints required
- Custom exceptions: `src/exceptions.py`
- Logging: `from src.utils.logger import get_logger`
- Entry point: ALWAYS `python -m src.main` (not `python src/main.py`)

---

## Testing

```python
# Quick test structure
import pytest
from src.models.object_detector import ObjectDetector

def test_detector():
    detector = ObjectDetector(model_path="models/yolov8n.pt")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert isinstance(result, list)
```

---

## ROS2 Topics (10 Publishers)

| Topic | Type |
|:------|:----|
| `/vision/detections` | JSON |
| `/vision/depth` | JSON |
| `/vision/faces` | JSON |
| `/vision/gestures` | JSON |
| `/vision/pose` | JSON |
| `/vision/status` | JSON |
| `/vision/image/debug` | JSON |
| `/vision/predictions` | JSON |
| `/vision/plan` | JSON |
| `/vision/safety` | JSON |

---

## Known Issues

- Motor control not integrated (commands print to console only)
- Target: 30-50 FPS (currently 10-12 without INT8/DLA)