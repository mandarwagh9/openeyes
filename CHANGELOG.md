# CHANGELOG.md - Version History for OpenEyes

> **Version**: v0.0.3  
> **Last Updated**: 2026-03-26

---

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.0.3] - 2026-03-26

### Added

- **YOLO11n Model**
  - YOLO11n model with better performance than YOLOv10n
  - ONNX export for TensorRT deployment
  - Expected FPS: 139 (FP16), 180 (INT8)

- **TensorRT Engine Integration**
  - Native TensorRT FP16 inference (~51 FPS on Jetson Orin Nano)
  - Export script: `scripts/export_tensorrt.py`
  - Pre-built engine: `models/yolo11n.engine` (8.2 MB)
  - Uses Ultralytics YOLO with TensorRT backend

- **Adaptive Frame Skipping**
  - Universal frame skipper for all models
  - Adaptive skipping based on motion detection
  - Multi-model frame scheduler
  - Configurable skip intervals per model
  - Default skip increased to 4 frames for CPU models

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
- Model path configuration supports YOLO11n + TensorRT engine
- Default frame scheduler intervals: detector(1), depth(4), face(4), gesture(4), pose(4)
- Depth estimation disabled by default (CPU bottleneck)
- Config file now supports per-model skip_frames setting

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

- [ ] Performance optimization (target: 15+ FPS)
- [ ] ROS2 integration
- [ ] Multi-camera support
- [ ] Production hardening

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
