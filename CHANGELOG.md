# CHANGELOG.md - Version History for OpenEyes

> **Version**: v0.1.2  
> **Last Updated**: 2026-03-29

---

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.1.2] - 2026-03-29

### Added

- **--info CLI Flag**
  - Show system information and OpenEyes recommendations
  - Displays Jetson-specific optimization tips
  - Quick reference for performance flags

- **--log-file CLI Flag**
  - Enable file logging with automatic rotation
  - Default: 5MB max file size, 3 backup files
  - Useful for debugging and production monitoring

- **Jetson Optimization Scripts**
  - `scripts/jetson_perf.sh` - One-command performance optimization
  - `scripts/jetson_info.sh` - Detailed system information
  - `scripts/jetson_helper.py` - Python helper (--info, --optimize, --check)

- **Log Rotation Support**
  - Added RotatingFileHandler to logger
  - Configurable max bytes and backup count
  - Prevents disk from filling up during long runs

### Changed

- **Version Bump**: v0.1.1 → v0.1.2

---

## [v0.1.1] - 2026-03-29

### Added

- **--no-depth CLI Flag**
  - New flag to disable depth estimation for maximum FPS
  - Depth estimation is computationally expensive

- **Model Enable/Disable Flags Now Working**
  - `--no-face`, `--no-gesture`, `--no-pose` were defined but not wired
  - Now properly skip model initialization when specified

- **Jetson Optimization Hint**
  - Startup message: "Run 'sudo nvpmodel -m 0 && sudo jetson_clocks' for max performance"
  - Auto-detects Jetson platform via `/proc/device-tree/model`

- **Model Status Logging**
  - Startup logs show which models are enabled/disabled
  - Helps verify configuration at startup

### Changed

- **More Aggressive Frame Skipping (Default)**
  - depth: 4 → 8 (every 8th frame)
  - face: 4 → 6 (every 6th frame)
  - gesture: 4 → 6 (every 6th frame)
  - pose: 4 → 6 (every 6th frame)

- **Adaptive Skipper Parameters**
  - base_skip: 3 → 2
  - min_skip: 2 → 1
  - max_skip: 5 → 4

### Performance

| Configuration | Expected FPS |
|:-------------|:------------|
| All models enabled (default) | ~10-12 |
| --no-face --no-gesture --no-pose | ~18-22 |
| --no-face --no-gesture --no-pose --no-depth | ~22-25 |
| + Jetson max performance (sudo nvpmodel -m 0 && sudo jetson_clocks) | +20-30% |

### CLI New Flags

```bash
# Disable specific models for speed
python src/main.py --no-face              # Skip face detection
python src/main.py --no-gesture           # Skip gesture recognition  
python src/main.py --no-pose              # Skip pose estimation
python src/main.py --no-depth             # Skip depth estimation (NEW)

# Disable multiple models
python src/main.py --no-face --no-gesture --no-pose --no-depth
```

---

## [v0.1.0] - 2026-03-28

### Added

- **ROS2 Configuration**
  - New `ros2` section in config.yaml
  - Configurable topics: detections, depth, faces, gestures, poses, cmd, status
  - Frame ID and confidence threshold settings

- **CSI Camera Improvements**
  - Device detection via `/dev/video*` check
  - Queue element added to GStreamer pipeline for stability
  - 1080p native resolution preferred
  - Retry logic with initialization delays

- **PoseData Enhancements**
  - Added `bbox` field for bounding box
  - Added `landmarks` field for pose landmarks

- **YOLO Path Resolution**
  - Fixed to use absolute path resolution from config directory

- **Complete ROS2 Vision Integration**
  - VisionPublisher with all vision modality publishers
  - Detections, depth, faces, gestures, poses topics
  - JSON fallback mode using std_msgs/String (avoids vision_msgs issues)

- **Command Subscription**
  - New `/vision/cmd` topic for robot commands
  - Valid commands: forward, backward, stop, left, right, follow
  - Command callback system for robot control integration

- **Parameter Validation**
  - Camera parameter validation in constructor
  - VisionPublisher parameter validation
  - Meaningful error messages for invalid inputs

- **Status Message Enhancement**
  - Timestamps added to vision status messages

- **CLI Enhancements**
  - `--ros2` flag to enable ROS2 publishing
  - `--version` flag to display version

### Changed

- Updated default version to v0.1.0
- Vision status now includes face and gesture counts
- Command field added to status output

---

## [v0.0.3] - 2026-03-26

### Added

- **YOLO11n Model**
  - YOLO11n model with better performance than YOLOv10n
  - ONNX export for TensorRT deployment
  - Expected FPS: 139 (FP16), 180 (INT8)

- **Adaptive Frame Skipping**
  - Universal frame skipper for all models
  - Adaptive skipping based on motion detection
  - Multi-model frame scheduler
  - Configurable skip intervals per model

- **ROS2 Integration**
  - VisionPublisher node for publishing detections
  - VisionControlNode for robot control
  - VisionWrapperNode for OpenEyes integration
  - Support for vision_msgs (Detection2DArray)

- **DeepStream SDK Integration**
  - DeepStream-Yolo custom parser library
  - GStreamer pipeline for CSI camera
  - Configuration files for YOLOv10

- **Performance Optimizations**
  - Motion-based adaptive processing
  - Result caching across all models
  - Frame interpolation for skipped frames

### Changed

- Updated version to v0.0.3
- Model path configuration supports YOLO11n
- Default frame scheduler intervals: detector(1), depth(2), face(2), gesture(2), pose(2)

### Dependencies Added

- pyds (DeepStream Python bindings)

---

## [v0.0.2] - 2026-03-25

### Added

- **Object Detection**
  - YOLOv10n model with PyTorch + CUDA acceleration
  - ONNX Runtime support with TensorRT provider
  - Automatic CUDA/ONNX fallback detection

- **Depth Estimation**
  - MiDaS_small model integration
  - GPU acceleration support
  - Depth map estimation and distance calculation

- **Face Detection**
  - MediaPipe FaceMesh integration
  - Multi-face support (up to 3)

- **Gesture Recognition**
  - MediaPipe Hands integration
  - Real-time hand tracking

- **Pose Estimation**
  - MediaPipe Pose integration
  - Body keypoint detection

- **Performance Optimizations**
  - Parallel processing with ThreadPoolExecutor
  - Frame skipping for pose estimation
  - Result caching for face/gesture

- **Camera Support**
  - CSI camera (IMX219) via nvarguscamerasrc
  - GStreamer pipeline integration
  - Auto-detection of Jetson platform

- **Display**
  - Auto-display detection (DISPLAY=:0 fallback)
  - Debug visualization with bounding boxes

### Changed

- Updated version to v0.0.2
- Performance: 5-6 FPS → 7-10 FPS with all models
- Default model: YOLOv8n → YOLOv10n
- CLI options added: --no-parallel, --pose-every

### Dependencies Updated

- Added `timm` for depth estimation
- Added `onnxruntime-gpu` for TensorRT support
- Downgraded MediaPipe to 0.10.9 for stability

### Known Issues

- MediaPipe may crash with certain frame sizes (workaround: use frame skipping)
- TensorRT engine build may timeout on low-memory systems (use ONNX fallback)

---

## [v0.0.1] - 2026-03-15

### Added

- **Documentation**
  - README.md with project overview
  - AGENTS.md developer guidelines
  - TECHNICAL_SPEC.md technical specifications
  - ARCHITECTURE.md system architecture
  - HARDWARE.md hardware specifications
  - API_SPEC.md API documentation
  - QUICKSTART.md quick start guide
  - INSTALL.md detailed installation
  - USER_GUIDE.md user guide
  - TROUBLESHOOTING.md common issues
  - CONTRIBUTING.md contribution guidelines
  - ROADMAP.md project roadmap
  - CHANGELOG.md version history

- **Project Structure**
  - Directory structure for src/, models/, docs/
  - requirements.txt with dependencies
  - LICENSE (Apache 2.0)

- **Source Code**
  - config.yaml with default configuration
  - camera/ module with CameraHandler
  - models/ module with ObjectDetector (YOLOv8)
  - output/ module with JSON formatter and UDP sender
  - utils/ module with config loader and logger
  - main.py entry point

- **Testing**
  - Unit tests for config, camera, models, output (36 tests)

### Changed

- Initial repository setup
- Project named "OpenEyes"
- License set to Apache 2.0

---

## [v0.0.3] - 2026-03-25

### Added

- **DeepStream SDK Integration**
  - DeepStream 7.1 installation and setup
  - Python bindings (pyds) for DeepStream
  - DeepStream-Yolo custom parser library
  - GStreamer pipeline for CSI camera
  - Test scripts for DeepStream pipeline

- **Documentation**
  - DEEPSTREAM.md integration guide
  - DeepStream configuration files

### Known Issues

- TensorRT engine build requires significant time (~5-10 minutes)
- Hybrid DeepStream + MediaPipe integration requires further testing

---

## [Unreleased]

### Planned for v1.0.0

- [ ] Multi-camera support
- [ ] Production hardening
- [ ] Motor control integration
- [ ] Further FPS optimization (target: 25-30 FPS)

### Planned for v1.1.0

- [ ] YOLOv10s for higher accuracy
- [ ] Custom model training
- [ ] Stereo vision

---

## Version Format

Given a version number `MAJOR.MINOR.PATCH`:

- **MAJOR** - Incompatible API changes
- **MINOR** - New functionality (backwards compatible)
- **PATCH** - Bug fixes

---

## Upgrade Guide

### From v0.0.1 to v0.0.2

1. Update requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Download new models (if not already included):
   ```bash
   # YOLOv10n is included in models/ folder
   ```

3. Run vision system:
   ```bash
   python src/main.py --debug
   ```

4. For optimal performance, enable Jetson max mode:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

---

## Release Cycle

| Version | Type | Target |
|:--------|:-----|:-------|
| v0.0.1 | Initial | March 2026 |
| v0.0.2 | Minor | March 2026 |
| v1.0.0 | Major | June 2026 |

---

## Acknowledgments

This CHANGELOG format is based on [Keep a Changelog](https://keepachangelog.com).
