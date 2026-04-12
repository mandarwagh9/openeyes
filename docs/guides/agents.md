# AGENTS.md - OpenEyes Developer Guidelines

> Hardware-agnostic edge robot vision framework with world models  
> **Version**: v2.5.0

---

## Entry Point

```bash
# MUST use -m flag (not python src/main.py)
python -m src.main --camera 0 --debug
```

---

## Key Commands

```bash
# Video processing
python -m src.main --video path/to/video.mp4 --output output.mp4

# World model + person following
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
├── main.py           # Entry point
├── cli/argparse.py  # All CLI flags
├── camera/         # CameraHandler
├── core/          # VisionSystem, frame_processor
├── models/        # ObjectDetector, depth_estimator
├── ros2/          # VisionPublisher
├── utils/         # config, logger, tracker, safety_controller
└── world_model/  # LeWorldModel, planner, safety_evaluator
```

---

## Hard-Earned Discoveries

- **CSI camera not available**: Check `/dev/video0`, add queue to GStreamer pipeline, reboot
- **ROS2 topics not publishing**: Use MultiThreadedExecutor in separate thread, add `time.sleep(0.5)` after init
- **MediaPipe empty results**: Lower confidence to 0.3 for face/pose, 0.1 for hands, resize to 640x480
- **Person following distance**: Use bounding box HEIGHT RATIO (% of frame): forward <60%, stop 60-95%, backward >95%

---

## Style

- Python 3.10+, type hints required
- Custom exceptions: `src/exceptions.py`
- Logging: `from src.utils.logger import get_logger`

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