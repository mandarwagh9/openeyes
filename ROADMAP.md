# ROADMAP.md - Project Roadmap for OpenEyes

> **Version**: v0.0.3  
> **Last Updated**: 2026-03-26

---

## Overview

This roadmap outlines the development plan for OpenEyes - a vision system for humanoid robots.

---

## Version History

| Version | Status | Date | Description |
|:--------|:-------|:-----|:------------|
| v0.0.3 | Current | 2026-03-26 | Performance + ROS2 |
| v0.0.2 | Released | 2026-03-25 | Full Vision Pipeline |
| v0.0.1 | Released | 2026-03-15 | Object Detection |

---

## Release Timeline

### v0.0.x - Foundation Phase

#### v0.0.3 - Performance + ROS2 ✓
**Status:** Complete

- [x] YOLO11n integration (better than YOLOv10n)
- [x] TensorRT FP16 engine integration (~51 FPS YOLO inference)
- [x] Export script for TensorRT (scripts/export_tensorrt.py)
- [x] MiDaS depth estimation
- [x] MediaPipe Face detection
- [x] MediaPipe Gesture recognition
- [x] MediaPipe Pose estimation
- [x] Parallel processing optimization
- [x] CSI camera support (IMX219)
- [x] Auto-display detection
- [x] Adaptive frame skipping (4-frame default)
- [x] ROS2 integration (vision nodes, services)
- [x] DeepStream SDK integration

**Released:** March 2026

**Performance:**
- 10-15 FPS with all models (TensorRT + MediaPipe)
- 30-50 FPS with object detection only

---

### v1.0.x - Integration Phase

#### v1.0.0 - Full Integration
**Status:** In Progress

- [ ] Unified vision pipeline optimization
- [ ] Performance tuning (target: 15+ FPS)
- [ ] ROS2 integration (optional)
- [ ] Production ready
- [ ] Multi-camera support
- [ ] Model switching (YOLOv10s/m for accuracy)

**Target:** June 2026

---

### v1.1.x - Advanced Features

#### v1.1.0 - Enhanced Models
**Status:** Planned

- [ ] YOLOv10s (higher accuracy)
- [ ] Better depth estimation
- [ ] Custom model training

#### v1.2.0 - Multi-Camera
**Status:** Planned

- [ ] Stereo vision
- [ ] 360° coverage
- [ ] Camera calibration

#### v1.3.0 - Navigation
**Status:** Planned

- [ ] SLAM integration
- [ ] Path planning
- [ ] Obstacle avoidance

---

## Feature Roadmap

### Phase 1: Core Vision (v0.0.x)

| Feature | Priority | Version | Status |
|:--------|:---------|:--------|:-------|
| Object Detection (YOLOv10) | P0 | v0.0.1 | Complete |
| Depth Estimation (MiDaS) | P0 | v0.0.2 | Complete |
| Face Detection | P1 | v0.0.2 | Complete |
| Gesture Recognition | P1 | v0.0.2 | Complete |
| Pose Estimation | P1 | v0.0.2 | Complete |

### Phase 2: Integration (v1.0.x)

| Feature | Priority | Version |
|:--------|:---------|:--------|
| Unified Pipeline | P0 | v1.0.0 |
| Performance (15+ FPS) | P0 | v1.0.0 |
| ROS2 Bridge | P2 | v1.0.0 |
| Multi-Camera | P1 | v1.1.0 |

### Phase 3: Advanced (v1.1.x - v2.0.x)

| Feature | Priority | Version |
|:--------|:---------|:--------|
| Custom Training | P2 | v1.1.0 |
| SLAM | P1 | v1.2.0 |
| Navigation | P1 | v1.3.0 |
| Voice Commands | P2 | v2.0.0 |

---

## Milestones

### M1: First Detection ✓
**Goal:** Get object detection working
**Status:** Complete

### M2: Depth Perception ✓
**Goal:** Add 3D understanding
**Status:** Complete

### M3: Human Interaction ✓
**Goal:** Detect faces, gestures, poses
**Status:** Complete

### M4: Production Ready
**Goal:** Stable, optimized, documented
**Target:** June 2026

---

## Long-term Vision

### v2.0+ - Autonomous

- Full SLAM implementation
- Navigation and path planning
- Real-time obstacle avoidance
- Voice command integration
- Integration with robot control systems

---

## Contributing to Roadmap

Want to suggest features? Please [open an issue](https://github.com/mandarwagh9/openeyes/issues) with:
- Feature description
- Use case
- Priority suggestion

---

## Notes

- Timeline is approximate and may change based on resources and feedback
- Priorities may shift based on user requirements
- Community contributions can accelerate development
- YOLOv10 uses AGPL-3.0 license - consider RTMDet for commercial use
