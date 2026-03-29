# AGENTS.md - Developer Guidelines for OpenEyes

> AI agent guidelines for OpenEyes - vision system for humanoid robots running on Jetson Orin Nano with full ROS2 integration.

---

## Quick Commands

```bash
# Install & Run
pip install -r requirements.txt
python src/main.py --camera 0 --debug

# Run with ROS2 publishing
python src/main.py --ros2

# Show version
python src/main.py --version

# Show system info
python src/main.py --info

# Performance options (v0.2.0)
python src/main.py --precision int8 --batch-size 2
python src/main.py --dla

# Tracking options (v0.2.1)
python src/main.py --follow
python src/main.py --no-tracking
python src/main.py --follow --track-max-age 60

# Model selection (v0.3.0)
python src/main.py --list-models
python src/main.py --model yolo12n
python src/main.py --model rtmdet_nano

# Advanced AI (v0.4.0)
python src/main.py --vla
python src/main.py --event-camera
python src/main.py --advanced-ai

# Disable models for speed
python src/main.py --no-face --no-gesture --no-pose --no-depth

# Enable file logging with rotation
python src/main.py --log-file logs/openeyes.log

# Jetson optimization (requires sudo)
sudo bash scripts/jetson_perf.sh
python3 scripts/jetson_helper.py --check

# Run with ROS2 launch file
ros2 launch openeyes openeyes.launch.py device:=cuda camera:=0 ros2:=true

# Testing
pytest tests/                          # All tests
pytest tests/test_camera.py -v        # Single file (verbose)
pytest tests/ -x                      # Stop on first failure
pytest tests/ --cov=src --cov-report=html

# Linting & Formatting
pylint src/
mypy src/
black src/
isort src/
```

---

## Code Style

### Language
- Python 3.10+
- Type hints required on ALL functions

### Imports (use isort)
```python
# 1. Standard library
import os
from typing import List, Optional

# 2. Third-party
import cv2
import numpy as np

# 3. Internal modules
from src.camera import CameraHandler
from src.models import ObjectDetector
```

### Naming Conventions

| Element | Convention | Example |
|:--------|:-----------|:--------|
| Files | snake_case | `camera_handler.py` |
| Classes | PascalCase | `CameraHandler` |
| Functions | snake_case | `process_frame()` |
| Variables | snake_case | `frame_buffer` |
| Constants | UPPER_SNAKE | `MAX_WIDTH = 640` |

### Type Annotations Required
```python
def detect_objects(frame: np.ndarray, conf: float = 0.5) -> List[Detection]:
    """Detect objects in the given frame."""
    ...
```

### Error Handling
Use custom exceptions from `src/exceptions.py`:
```python
from src.exceptions import CameraError, ModelError, VisionError

try:
    detector.load()
except ModelError as e:
    logger.error(f"Failed to load model: {e}")
    raise
```

### Docstrings
Google-style, required on all public functions:
```python
def process_frame(frame: np.ndarray) -> VisionResult:
    """Process a single frame through the vision pipeline.

    Args:
        frame: Input frame as BGR numpy array

    Returns:
        VisionResult containing detections, depth, faces, gestures, pose

    Raises:
        CameraError: If frame capture fails
    """
```

### Logging
```python
from src.utils.logger import get_logger
logger = get_logger(__name__)

logger.info(f"Detected {len(detections)} objects")
```

---

## Architecture

```
src/
├── camera/           # CameraHandler, types, CSI device detection
├── models/           # ObjectDetector, depth_estimator, etc.
├── output/           # json_formatter, udp_sender
├── ros2/             # VisionPublisher with MultiThreadedExecutor, JSON fallback
├── utils/            # config (absolute path resolution), logger
└── main.py           # Entry point with --ros2 and --version flags
```

### Data Flow
Camera → ObjectDetector → JSON Formatter → UDP Sender + ROS2 Publisher

### Vision Pipeline
1. CSI Camera capture (IMX219 at 1920x1080)
2. YOLO11n object detection
3. MiDaS depth estimation (now wired to pipeline)
4. Face detection (MediaPipe - confidence lowered to 0.3)
5. Gesture recognition (MediaPipe - confidence lowered to 0.3)
6. Pose estimation
7. Output via UDP + ROS2 (7 topics + command subscription)

---

## Key Discoveries & Solutions

### CSI Camera Issues
- **Problem**: Camera not available
- **Solution**: Add `_check_csi_available()` method to check `/dev/video0`, add queue to GStreamer pipeline, reboot Jetson
- **File**: `src/camera/camera_handler.py`

### ROS2 vision_msgs Bug
- **Problem**: vision_msgs caused assertion failure
- **Solution**: Use JSON/std_msgs fallback mode, force `VISION_MSGS_AVAILABLE = False`
- **File**: `src/ros2/vision_node.py`

### ROS2 Not Publishing
- **Problem**: Topics weren't appearing
- **Solution**: Add MultiThreadedExecutor in separate thread, add `time.sleep(0.5)` after initialization
- **File**: `src/ros2/vision_node.py`

### Model Path Issue
- **Problem**: config.yaml model path was relative
- **Solution**: Make `yolo_path` property resolve absolute paths
- **File**: `src/utils/config.py`

### Parameter Validation
- Added validation for camera constructor parameters and VisionPublisher parameters

### Face/Gesture Detection Issues
- **Problem**: MediaPipe face/gesture detection returning empty results
- **Solution**: Lower confidence threshold from 0.5 to 0.3, added debug logging
- **Files**: `src/models/face_detector.py`, `src/models/gesture_recognizer.py`

### Depth Estimation Not Publishing
- **Problem**: DepthEstimator loaded but never called, no depth_map in DepthData
- **Solution**: Added depth_map field to DepthData, wired estimator in pipeline, tracks last depth for frame skipping
- **Files**: `src/camera/types.py`, `src/main.py`

---

## ROS2 Integration

### Publishers (7 Topics)
| Topic | Type | Description |
|:------|:-----|:------------|
| `/vision/detections` | JSON | Object detections |
| `/vision/depth` | JSON | Depth map data |
| `/vision/faces` | JSON | Face detections |
| `/vision/gestures` | JSON | Gesture recognitions |
| `/vision/pose` | JSON | Body pose landmarks |
| `/vision/status` | JSON | System status with timestamp |
| `/vision/image/debug` | JSON | Debug image (annotated frame) |

### Command Subscription
| Command | Action |
|:--------|:-------|
| `forward` | Move forward |
| `backward` | Move backward |
| `stop` | Stop all motion |
| `left` | Turn left |
| `right` | Turn right |
| `follow` | Follow detected person |

**Note**: Commands currently print to console (not motor control)

### ROS2 Message Format
- Use JSON format for all messages (vision_msgs compatibility issues)
- Depth normalized to 0-1 meters (32FC1)
- Timestamp included in status messages

---

## Testing

### Test Structure
```python
import pytest
from src.models.object_detector import ObjectDetector

class TestObjectDetector:
    @pytest.fixture
    def detector(self):
        return ObjectDetector(model_path="models/yolov8n.pt")

    def test_detect_returns_list(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert isinstance(result, list)
```

---

## Git Workflow

### Branch Naming
- `feat/<feature>` - New features
- `fix/<issue>` - Bug fixes
- `docs/<feature>` - Documentation

### Commit Messages
```
<type>: <short description>

Closes #<issue>
```
Types: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

---

## Common Tasks

### Add New AI Model
1. Create class in `src/models/`
2. Implement `load()` and `detect()` methods
3. Add type hints and docstrings
4. Add tests in `tests/`

### Modify Output
1. Edit `src/output/json_formatter.py`
2. Ensure output matches API_SPEC.md schema

---

## Modified Files Reference

### Core Code Files
| File | Changes |
|:-----|:--------|
| `src/main.py` | Entry point with ROS2 integration, --ros2 and --version flags, depth estimator wired |
| `src/ros2/vision_node.py` | VisionPublisher with all publishers, command callback, JSON fallback |
| `src/ros2/__init__.py` | Try/except for services import |
| `src/ros2/services.py` | Try/except for std_srvs import |
| `src/camera/camera_handler.py` | CSI device detection, queue in pipeline, validation |
| `src/camera/types.py` | Added bbox, landmarks to PoseData, depth_map to DepthData |
| `src/models/face_detector.py` | Lowered confidence to 0.3, added debug logging |
| `src/models/gesture_recognizer.py` | Lowered confidence to 0.3, added debug logging |
| `src/utils/config.py` | YOLO path resolution, logger added |
| `config.yaml` | ROS2 configuration section |
| `launch/openeyes.launch.py` | ROS2 launch file (NEW) |
| `package.xml` | ROS2 package manifest (NEW) |

### Documentation Files
| File | Changes |
|:-----|:--------|
| `CHANGELOG.md` | v0.1.0 with all changes |
| `README.md` | Version badge, command subscription feature |
| `USER_GUIDE.md` | ROS2 Integration section, CLI options |
| `QUICKSTART.md` | --ros2/--version examples |
| `TROUBLESHOOTING.md` | ROS2 Issues section |
| `AGENTS.md` | Architecture, CLI commands, launch file (this file) |
| `ROADMAP.md` | Version history |
| `INSTALL.md` | Version update |
| `DOCUMENTATION.md` | Version table |

---

## Performance Targets

| Metric | Target | v0.1.1 |
|:-------|:-------|:-------|
| FPS | 20-30 | 10-12 (all), 22-25 (minimal) |
| Latency | <50ms | ~40ms |
| Memory | <2GB | ~1.2GB |

---

## Known Issues

- Robot motor control not integrated (commands just print to console)
- Performance needs optimization (target 20-30 FPS)

---

## Resources

- [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md)
- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
