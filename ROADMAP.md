# ROADMAP.md - Project Roadmap for OpenEyes

> **Version**: v0.7.0 (Phase 1 Complete)
> **Last Updated**: 2026-04-01

---

## Overview

This roadmap outlines the development plan for OpenEyes - a vision system for humanoid robots running on NVIDIA Jetson Orin Nano.

---

## Version History

| Version | Status | Date | Description |
|:--------|:-------|:-----|:------------|
| v0.7.0 | Current | 2026-04-01 | Multi-Modal Sensing + LIDAR + Sensor Fusion |
| v0.6.0 | Released | 2026-03-30 | Navigation + Obstacle Avoidance |
| v0.5.0 | Released | 2026-03-30 | SLAM + Nav2 + VLA Integration |
| v0.4.4 | Released | 2026-03-30 | Person Following + Gesture Owner |
| v0.4.0 | Released | 2026-03-28 | VLA + Event Camera |
| v0.3.0 | Released | 2026-03-27 | Model Selection |
| v0.2.x | Released | 2026-03-26 | Tracking + ROS2 |
| v0.1.0 | Released | 2026-03-28 | Command Subscription + Full ROS2 |

---

## Industry Standard Roadmap (18-Month Plan)

### Phase 1: Foundation (Months 1-6) ✅ COMPLETE

#### v0.7.0 - Multi-Modal Sensing (April 2026) ✅
**Status:** Complete

- [x] Isaac ROS VSLAM integration (GPU-accelerated visual odometry)
- [x] Cartographer support for 2D LiDAR mapping
- [x] Depth-to-LaserScan conversion (pointcloud_to_laserscan)
- [x] Nav2 behavior tree customization support
- [x] LIDAR integration for obstacle detection (`src/ros2/lidar_processing.py`)
- [x] RealSense D455 support (stereo depth + IMU)
- [x] Sensor fusion module (camera + depth + LIDAR) (`src/ros2/sensor_fusion.py`)
- [x] Multi-camera support (`src/ros2/multi_camera.py`)
- [x] CLI args: `--lidar`, `--lidar-topic`, `--realsense`, `--multi-camera`

---

### Phase 2: AI & Performance (Months 7-12)

#### v0.8.0 - VLA Integration (July 2026)
**Status:** Planned

- [ ] SmolVLA integration (~450M params, Orin Nano optimized)
- [ ] Isaac GR00T support for humanoid control
- [ ] Action chunking for real-time control (10-30 Hz)
- [ ] LoRA fine-tuning support for VLA customization

#### v0.9.0 - Performance Optimization (September 2026)
**Status:** Planned

- [ ] Full INT8 quantization with calibration
- [ ] DLA (Deep Learning Accelerator) offloading
- [ ] DeepStream multi-camera pipeline
- [ ] Memory optimization for unified memory
- [ ] Target: 30 FPS with all models on Orin Nano

#### v1.0.0 - Diffusion Policies (December 2026)
**Status:** Planned

- [ ] Diffusion Policy integration for manipulation
- [ ] ACT (Action Chunking with Transformers)
- [ ] On-device VLA inference optimization

---

### Phase 3: Safety & Certification (Months 13-18)

#### v1.1.0 - Reliability (March 2027)
**Status:** Planned

- [ ] 24/7 operation with auto-recovery
- [ ] Comprehensive error handling and logging
- [ ] Health monitoring and diagnostics
- [ ] OTA model updates

#### v1.2.0 - Safety Certification (June 2027)
**Status:** Planned

- [ ] ISO 10218 compliance preparation
- [ ] Emergency stop integration
- [ ] Safe speed/position monitoring
- [ ] Functional safety documentation

---

## Feature Roadmap

### Phase 1: Multi-Modal Sensing (v0.7.x) ✅

| Feature | Priority | Status |
|:--------|:---------|:-------|
| Isaac ROS VSLAM | P0 | Complete |
| LIDAR Processing | P0 | Complete |
| Sensor Fusion | P0 | Complete |
| Multi-Camera | P1 | Complete |
| RealSense Support | P1 | Complete |

### Phase 2: VLA & Performance (v0.8.x - v0.9.x)

| Feature | Priority | Version | Status |
|:--------|:---------|:--------|:-------|
| SmolVLA | P0 | v0.8.0 | Planned |
| Isaac GR00T | P1 | v0.8.0 | Planned |
| INT8 Quantization | P0 | v0.9.0 | Planned |
| DLA Offloading | P0 | v0.9.0 | Planned |
| DeepStream Pipeline | P1 | v0.9.0 | Planned |

### Phase 3: Safety & Reliability (v1.0.x - v1.2.x)

| Feature | Priority | Version | Status |
|:--------|:---------|:--------|:-------|
| Diffusion Policy | P1 | v1.0.0 | Planned |
| 24/7 Operation | P0 | v1.1.0 | Planned |
| Health Monitoring | P0 | v1.1.0 | Planned |
| ISO 10218 Prep | P1 | v1.2.0 | Planned |
| Emergency Stop | P0 | v1.2.0 | Planned |

---

## Industry Standard Requirements

| Category | Requirements | Status |
|:---------|:-------------|:-------|
| Technical | Multi-modal sensing, <100ms latency, >99% accuracy | v0.7.0 ✅ |
| AI/ML | Deep learning, VLA, continuous learning | v0.8.0+ |
| Standards | ROS2 compliance, ISO 10218 | v1.2.0 |
| Reliability | MTBF >50K hours, 24/7 operation | v1.1.0 |
| Safety | SIL 2/PLd compliance, emergency integration | v1.2.0 |

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
- All new modules follow ROS2 standards for interoperability