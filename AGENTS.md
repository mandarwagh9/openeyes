# AGENTS.md - Developer Guidelines for OpenEyes

> AI agent guidelines for OpenEyes - hardware-agnostic edge robot vision framework with world models
> **Version**: v2.5.0-dev

---

## Quick Commands

```bash
# Install & Run
pip install -r requirements.txt
python -m src.main --camera 0 --debug

# Run with ROS2 publishing
python -m src.main --ros2

# Show version
python -m src.main --version

# Show system info
python -m src.main --info

# Performance options (v0.2.0)
python -m src.main --precision int8 --batch-size 2
python -m src.main --dla

# Video file processing (v2.5.0)
python -m src.main --video path/to/video.mp4 --output output.mp4
python -m src.main --video path/to/video.mp4 --output demo.mp4 --follow

# Tracking options (v0.2.1)
python -m src.main --follow
python -m src.main --no-tracking
python -m src.main --follow --track-max-age 60

# Model selection (v0.3.0)
python -m src.main --list-models
python -m src.main --model yolo12n
python -m src.main --model rtmdet_nano
python -m src.main --model yolo26n

# World Models (v2.5.0)
python -m src.main --world-model lewm
python -m src.main --world-model vjepa2
python -m src.main --world-model lewm --follow --turbo

# Industry Templates (v2.5.0)
python -m src.main --template warehouse
python -m src.main --template manufacturing-qa
python -m src.main --template agriculture
python -m src.main --template retail

# Advanced AI (v0.4.0)
python -m src.main --vla
python -m src.main --event-camera
python -m src.main --advanced-ai

# Depth model selection (v2.5.0)
python -m src.main --depth-model da3-small
python -m src.main --depth-model da3-base
python -m src.main --depth-model midas-small

# Disable models for speed
python -m src.main --no-face --no-gesture --no-pose --no-depth

# Enable file logging with rotation
python -m src.main --log-file logs/openeyes.log

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

# SLAM & Navigation (v0.5.0)
python -m src.main --visual-odom          # Enable visual odometry
python -m src.main --depth-to-scan      # Convert depth to laser scan
python -m src.main --slam              # Enable full SLAM mode

# Launch SLAM with Isaac ROS
ros2 launch openeyes cuvslam.launch.py

# Launch Nav2 navigation
ros2 launch openeyes nav2.launch.py map:=/path/to/map.yaml

# VLA Commands (v0.5.0)
python -m src.main --vla               # Enable VLA processing
python -m src.main --advanced-ai       # Enable all AI features

# Real VLA Models (v0.6.0)
python -m src.main --real-vla smolvla  # Use SmolVLA (~450M params)
python -m src.main --real-vla openvla  # Use OpenVLA (7B params, needs AGX)
python -m src.main --real-vla octo     # Use Octo (~93M params)

# Navigation (v0.6.0)
python -m src.main --nav2              # Enable Nav2 with obstacle avoidance
ros2 launch openeyes unified.launch.py # Full autonomous navigation

# Navigation goals
ros2 topic pub /navigation/goal std_msgs/String "data: '2.0 1.0 0.0'"  # x, y, yaw

# Multi-Modal Sensing (v0.7.0)
python -m src.main --lidar                         # Enable LIDAR processing
python -m src.main --lidar-topic /scan             # LIDAR topic
python -m src.main --realsense                     # Enable RealSense D455
python -m src.main --multi-camera                  # Multi-camera mode

# VLA & Performance (v0.8.0)
python -m src.main --int8                         # INT8 quantization
python -m src.main --dla                          # DLA offloading
python -m src.main --diffusion-policy             # Enable Diffusion Policy
python -m src.main --action-chunking              # Enable action chunking
python -m src.main --control-freq 20              # Control frequency (Hz)

# Safety & Reliability (v1.0.0)
python -m src.main --safety                      # Enable safety controller
python -m src.main --health-monitor              # Enable health monitoring
python -m src.main --max-velocity 1.0            # Max velocity (m/s)
python -m src.main --min-distance 0.3            # Min distance (m)
python -m src.main --ota-update                   # Enable OTA updates
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
│   ├── video_source.py      # v2.5.0 - Video file input support
├── cli/              # argparse.py - All CLI flags
├── core/             # VisionSystem - Main pipeline orchestration
├── models/           # ObjectDetector, depth_estimator, VLA, etc.
│   ├── action_chunker.py      # v0.8.0 - Real-time control
│   ├── lora_finetuning.py     # v0.8.0 - VLA customization
│   ├── diffusion_policy.py    # v1.0.0 - Robot manipulation
│   └── tensorrt_optimizer.py  # v0.8.0 - INT8/DLA optimization
├── output/           # json_formatter, udp_sender
├── ros2/             # VisionPublisher with MultiThreadedExecutor
│   ├── lidar_processing.py    # v0.7.0 - LIDAR point cloud
│   ├── sensor_fusion.py       # v0.7.0 - Multi-sensor fusion
│   └── multi_camera.py        # v0.7.0 - Multi-camera support
├── utils/            # config, logger, tracker
│   ├── health_monitor.py      # v1.0.0 - 24/7 operation
│   ├── ota_update.py          # v1.0.0 - Model updates
│   ├── safety_controller.py  # v1.0.0 - Safety features
│   └── tracker.py             # Person following with distance logic
├── world_model/      # v2.5.0 - Predictive planning
│   ├── base.py              # WorldModel abstract interface
│   ├── lewm.py              # LeWorldModel (15M params)
│   ├── planner.py           # CEM planner
│   └── safety_evaluator.py  # Predictive safety checks
└── main.py           # Entry point with all CLI flags
```

### Entry Point
- **Use**: `python -m src.main` (NOT `python src/main.py`)
- Video mode: `--video <path>` + `--output <path>`
- Follow mode: `--follow` with `--turbo` for max FPS

### Data Flow
Camera/Video → ObjectDetector → Tracker → Depth → [World Model] → JSON/UDP/ROS2

### Vision Pipeline
1. CSI Camera or Video file capture (1920x1080 or configured)
2. YOLO11n/12n/26n object detection (TensorRT optimized)
3. ByteTrack tracking with IoU association
4. MiDaS/DA3 depth estimation
5. Face detection (MediaPipe - confidence 0.3)
6. Gesture recognition (MediaPipe - confidence 0.1, resized to 640x480)
7. Pose estimation (MediaPipe - confidence 0.3)
8. World model prediction (optional, v2.5.0)
9. Output via UDP + ROS2 (7+ topics) + Video writer (if --output)

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

### Gesture Detection Not Working (v0.4.4)
- **Problem**: MediaPipe Hands wasn't detecting hands at high resolution
- **Solution**: Added image resizing to 640x480, lowered confidence to 0.1
- **File**: `src/models/gesture_recognizer.py`

### Person Following Distance Issues (v0.4.4)
- **Problem**: System used bbox size for distance, relied on tracking continuity
- **Solution**: Use bounding box HEIGHT RATIO (% of frame) for distance:
  - forward: < 60% (person small = far away)
  - stop: 60-95% (person medium = just right)
  - backward: > 95% (person large = too close)
- **File**: `src/utils/tracker.py`

### Gesture-Based Owner Selection (v0.4.4)
- **Problem**: Needed way to designate who robot should follow
- **Solution**: Show open_palm gesture to become "owner", robot follows that person
- **File**: `src/utils/tracker.py`

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

### Core Code Files (v2.5.0)
| File | Changes |
|:-----|:--------|
| `src/cli/argparse.py` | Added `--video` and `--output` flags |
| `src/camera/video_source.py` | New VideoSource class for video file input |
| `src/core/vision_system.py` | Wired video mode, video writer support |
| `src/main.py` | Entry point with video/output path support |
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

### Demo Files (v2.5.0)
| File | Changes |
|:-----|:--------|
| `demo/process_demo.py` | Standalone demo processing script |
| `demo/demo1.gif` | Warehouse follow demo (user-provided) |
| `demo/demo2.gif` | Multi-object tracking demo (user-provided) |

### Documentation Files
| File | Changes |
|:-----|:--------|
| `CHANGELOG.md` | v2.5.0 with all changes |
| `README.md` | System architecture diagram, Demos section, Star History |
| `index.html` | v2.5.0 updates with new features, performance, hardware |
| `getting-started.html` | v2.5.0 commands, templates, video processing |
| `commands.html` | World models, templates, video, fleet sections |
| `hardware.html` | Supported platforms table |
| `USER_GUIDE.md` | ROS2 Integration section, CLI options |
| `QUICKSTART.md` | --ros2/--version examples |
| `TROUBLESHOOTING.md` | ROS2 Issues section |
| `AGENTS.md` | Architecture, CLI commands, launch file (this file) |
| `ROADMAP.md` | Version history |
| `INSTALL.md` | Version update |
| `DOCUMENTATION.md` | Version table |

---

## Performance Targets

| Metric | Target | v1.0.0 |
|:-------|:-------|:-------|
| FPS | 30-50 | 10-12 (all), 40-50 (INT8+DLA) |
| Latency | <50ms | ~40ms |
| Memory | <2GB | ~1.2GB |
| Control Frequency | 10-30 Hz | Configurable |

---

## v2.5.0 New Features

### Video Processing (v2.5.0)
- `--video` flag for video file input (mp4, avi, mkv)
- `--output` flag for annotated output recording
- `VideoSource` class in `src/camera/video_source.py`
- Wired into `VisionSystem` with OpenCV VideoWriter

### Demo Pipeline
- `demo/process_demo.py` - Standalone demo processing script
- Full pipeline: detection → tracking → depth → decision → overlay
- Decision logic: FORWARD/STOP/LEFT/RIGHT/BACKWARD based on bbox height ratio
- VLA-style reasoning overlay with confidence scores

### Industry Templates (v2.5.0)
- `--template warehouse` - Warehouse person following
- `--template manufacturing-qa` - Manufacturing QA inspection
- `--template agriculture` - Agricultural monitoring
- `--template retail` - Retail analytics

### World Models (v2.5.0)
- LeWM (15M params) for 100-200 Hz predictive planning
- V-JEPA 2 for spatiotemporal perception
- CEM planner with configurable horizon/samples
- Predictive safety evaluation

### Depth Anything V3 (v2.5.0)
- `--depth-model da3-small` or `da3-base`
- 35.7% better than MiDaS
- Smaller model for edge deployment

### Hardware Abstraction Layer
- TensorRT (NVIDIA), OpenVINO (Intel), TVM, Hailo, QNN

### Supported Platforms
- Jetson Orin Nano/AGX
- Raspberry Pi 5
- Intel NPU
- Hailo-8L
- Qualcomm Snapdragon

---

## Demo Processing

### Create Demo Video
```bash
# Process video with full overlay
python demo/process_demo.py --input video.mp4 --output demo_out.mp4

# Convert to optimized GIF for GitHub
ffmpeg -i demo_out.mp4 -vf "fps=8,scale=400:-1" -loop 0 demo.gif

# Or with imageio (better optimization)
python -c "
import imageio
reader = imageio.get_reader('demo_out.mp4')
frames = []
for i, frame in enumerate(reader):
    if i % 8 == 0:  # 8 FPS
        from PIL import Image
        img = Image.fromarray(frame).resize((400, int(400 * frame.shape[0] / frame.shape[1])))
        frames.append(np.array(img))
imageio.mimsave('demo.gif', frames, fps=8, loop=0)
"
```

### GIF Optimization for GitHub (<2MB)
- Width: 400px
- FPS: 6-8
- Colors: max_colors=64
- Dither: bayer
- Use: `python -m imageio --mpeg-rewrite filename.gif output.gif`

---

## Known Issues

- Robot motor control not integrated (commands just print to console)
- Performance needs optimization (target 20-30 FPS)

---

## Resources

- [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md)
- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [WORLD_MODELS.md](docs/WORLD_MODELS.md)
- [WORLD_MODELS_PLAN.md](WORLD_MODELS_PLAN.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)

---

## World Model Development Guidelines (v2.5.0+)

### Architecture Principles

1. **Augment, don't replace**: World models add prediction/planning on top of existing detection/tracking/depth
2. **Latency budget**: World model inference must stay under 10ms for planning, 50ms for perception
3. **Memory budget**: World models must stay under 1GB additional memory on Jetson Orin Nano
4. **Power budget**: World model stack must stay under 10W additional power

### Code Structure

```
src/world_model/
├── __init__.py
├── base.py              # WorldModel abstract interface
├── lewm.py              # LeWorldModel (15M params)
├── planner.py           # CEM planner
├── safety_evaluator.py  # Predictive safety checks
└── types.py             # Prediction, Plan data types
```

### World Model Interface

All world models must implement the `WorldModel` abstract interface:

```python
from src.world_model.base import WorldModel

class MyWorldModel(WorldModel):
    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Encode frame to latent state."""
        ...
    
    def predict(self, latent: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict next latent state given action."""
        ...
    
    def plan(self, current: np.ndarray, goal: np.ndarray, 
             horizon: int = 10) -> list[np.ndarray]:
        """Plan action sequence to reach goal."""
        ...
```

### Performance Requirements

| Metric | LeWM (15M) | V-JEPA 2 ViT-B (80M) |
|--------|-----------|---------------------|
| Latency | <10ms | <100ms |
| Memory | <100MB | <1GB |
| Power | 3-5W | 6-9W |
| Control rate | 100-200 Hz | 10-20 Hz |

### Testing Requirements

- All world model tests must pass on Jetson Orin Nano
- Latency tests: `assert latency_ms < 10` for LeWM
- Memory tests: `assert memory_mb < 100` for LeWM
- Prediction accuracy: `assert iou > 0.8` for occlusion handling

### Key References

- **LeWorldModel**: arXiv:2603.19312
- **V-JEPA 2**: https://github.com/facebookresearch/vjepa2
- **DINO-WM**: https://github.com/gaoyuezhou/dino_wm
- **Full docs**: [docs/WORLD_MODELS.md](docs/WORLD_MODELS.md)
