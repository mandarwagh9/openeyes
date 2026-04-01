# ROADMAP.md - Project Roadmap for OpenEyes

> **Version**: v1.0.0 (Industry Standard)
> **Last Updated**: 2026-04-01

---

## Overview

This roadmap outlines the development plan for OpenEyes - a vision system for humanoid robots running on NVIDIA Jetson Orin Nano.

---

## Version History

| Version | Status | Date | Description |
|:--------|:-------|:-----|:------------|
| v1.0.0 | Current | 2026-04-01 | Safety & Reliability + Diffusion Policy |
| v0.8.0 | Released | 2026-04-01 | VLA Integration + Action Chunking + TensorRT |
| v0.7.0 | Released | 2026-04-01 | Multi-Modal Sensing + LIDAR + Sensor Fusion |
| v0.6.0 | Released | 2026-03-30 | Navigation + Obstacle Avoidance |
| v0.5.0 | Released | 2026-03-30 | SLAM + Nav2 + VLA Integration |
| v0.4.4 | Released | 2026-03-30 | Person Following + Gesture Owner |
| v0.4.0 | Released | 2026-03-28 | VLA + Event Camera |
| v0.3.0 | Released | 2026-03-27 | Model Selection |
| v0.2.x | Released | 2026-03-26 | Tracking + ROS2 |
| v0.1.0 | Released | 2026-03-28 | Command Subscription + Full ROS2 |

---

## Industry Standard Roadmap (18-Month Plan) ✅ COMPLETE

All phases of the industry standard roadmap have been implemented:

### Phase 1: Foundation (Months 1-6) ✅ COMPLETE

#### v0.7.0 - Multi-Modal Sensing (April 2026) ✅ COMPLETE
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

### Phase 2: AI & Performance (Months 7-12) ✅ COMPLETE

#### v0.8.0 - VLA Integration (April 2026) ✅ COMPLETE
**Status:** Complete

- [x] Action chunking for real-time control (10-30 Hz) (`src/models/action_chunker.py`)
- [x] LoRA fine-tuning support for VLA customization (`src/models/lora_finetuning.py`)
- [x] TensorRT INT8 quantization with calibration (`src/models/tensorrt_optimizer.py`)
- [x] DLA (Deep Learning Accelerator) offloading
- [x] CLI args: `--int8`, `--dla`, `--action-chunking`, `--control-freq`

#### v1.0.0 - Diffusion Policies (April 2026) ✅ COMPLETE
**Status:** Complete

- [x] Diffusion Policy integration for manipulation (`src/models/diffusion_policy.py`)
- [x] Action Chunking with Transformers
- [x] On-device VLA inference optimization

### Phase 3: Safety & Certification (Months 13-18) ✅ COMPLETE

#### v1.0.0 - Safety & Reliability (April 2026) ✅ COMPLETE
**Status:** Complete

- [x] 24/7 operation with auto-recovery (`src/utils/health_monitor.py`)
- [x] Comprehensive error handling and logging
- [x] Health monitoring and diagnostics
- [x] OTA model updates with rollback (`src/utils/ota_update.py`)
- [x] Emergency stop integration (`src/utils/safety_controller.py`)
- [x] Safe speed/position monitoring
- [x] Functional safety documentation
- [x] CLI args: `--safety`, `--health-monitor`, `--max-velocity`, `--min-distance`, `--ota-update`

---

## Feature Roadmap (Complete)

### Phase 1: Multi-Modal Sensing (v0.7.x) ✅ COMPLETE

| Feature | Priority | Status |
|:--------|:---------|:-------|
| Isaac ROS VSLAM | P0 | Complete |
| LIDAR Processing | P0 | Complete |
| Sensor Fusion | P0 | Complete |
| Multi-Camera | P1 | Complete |
| RealSense Support | P1 | Complete |

### Phase 2: VLA & Performance (v0.8.x - v0.9.x) ✅ COMPLETE

| Feature | Priority | Version | Status |
|:--------|:---------|:--------|:-------|
| Action Chunker | P0 | v0.8.0 | Complete |
| LoRA Fine-tuning | P1 | v0.8.0 | Complete |
| TensorRT Optimizer | P0 | v0.8.0 | Complete |
| Diffusion Policy | P1 | v1.0.0 | Complete |
| INT8 Quantization | P0 | v0.8.0 | Complete |
| DLA Offloading | P0 | v0.8.0 | Complete |

### Phase 3: Safety & Reliability (v1.0.x) ✅ COMPLETE

| Feature | Priority | Version | Status |
|:--------|:---------|:--------|:-------|
| Health Monitor | P0 | v1.0.0 | Complete |
| OTA Updates | P1 | v1.0.0 | Complete |
| Safety Controller | P0 | v1.0.0 | Complete |
| Emergency Stop | P0 | v1.0.0 | Complete |

---

## Future Development

Now that the 18-month industry standard roadmap is complete, the project moves to maintenance and enhancement mode.

### Potential Next Steps

| Area | Description | Priority |
|:-----|:------------|:---------|
| Integration Testing | Real robot hardware integration | Medium |
| VLA Model Loading | SmolVLA, Isaac GR00T actual weights | Medium |
| DeepStream Pipeline | Multi-camera DeepStream implementation | Low |
| ROS2 Package | Full ROS2 package for distribution | Low |
| Community | Documentation, examples, tutorials | Ongoing |

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