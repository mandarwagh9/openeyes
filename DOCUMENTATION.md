# OpenEyes Documentation Index

Welcome to the OpenEyes documentation. This index helps you find the right document for your needs.

> **Version**: v2.5.0

## Quick Links

| Your Goal | Start Here |
|:----------|:-----------|
| I want to run the system quickly | [QUICKSTART.md](QUICKSTART.md) |
| I need detailed installation instructions | [INSTALL.md](INSTALL.md) |
| I want to understand how to use it | [USER_GUIDE.md](USER_GUIDE.md) |
| I want to optimize performance | [OPTIMIZATION.md](OPTIMIZATION.md) |

---

## For Users

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide for first-time users
- **[INSTALL.md](INSTALL.md)** - Complete installation instructions
- **[USER_GUIDE.md](USER_GUIDE.md)** - How to run and configure the system
- **[CONFIG.md](CONFIG.md)** - Configuration options (if available)

### Using the System
- **Camera setup**: CSI camera (IMX219) or USB webcam
- **Output formats**: JSON over UDP, ROS2 topics
- **Debug mode**: Visual display with bounding boxes

---

## For Developers

### Contributing
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute code
- **[AGENTS.md](AGENTS.md)** - Developer guidelines and code style
- **Coding standards**: Type hints, docstrings, testing requirements

### Technical Documentation
- **[TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md)** - Complete technical specification
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and design
- **[HARDWARE.md](docs/HARDWARE.md)** - Hardware requirements and setup
- **[API_SPEC.md](docs/API_SPEC.md)** - API documentation

### Advanced Topics
- **[OPTIMIZATION.md](OPTIMIZATION.md)** - Performance tuning (15+ FPS target)
- **[DEEPSTREAM.md](DEEPSTREAM.md)** - DeepStream SDK integration
- **[ROADMAP.md](ROADMAP.md)** - Project roadmap and future plans

---

## Troubleshooting

### Common Issues
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

### Known Issues
- MediaPipe may require specific versions (0.10.9 recommended)
- TensorRT engine building requires GPU and may timeout on low-memory systems
- CSI camera requires Jetson platform with camera connector

---

## Version Information

### Current Version: v1.0.0

| Version | Release Date | Key Changes |
|:--------|:-------------|:------------|
| v1.0.0 | 2026-04-01 | Safety & Reliability, Diffusion Policy, Health Monitor, OTA |
| v0.8.0 | 2026-04-01 | Action Chunker, LoRA, TensorRT Optimizer, INT8, DLA |
| v0.7.0 | 2026-04-01 | LIDAR, Sensor Fusion, Multi-Camera, RealSense |
| v0.6.0 | 2026-03-30 | Navigation + Obstacle Avoidance |
| v0.5.0 | 2026-03-30 | SLAM + Nav2 + VLA |
| v0.4.4 | 2026-03-30 | Person following, gesture owner |
| v0.4.0 | 2026-03-28 | VLA models, event camera |
| v0.3.0 | 2026-03-27 | Model selection |
| v0.2.x | 2026-03-26 | Tracking, ROS2, performance |
| v0.1.0 | 2026-03-28 | Command subscription, full ROS2 |
| v0.0.3 | 2026-03-26 | YOLO11n, frame skipping, ROS2 |
| v0.0.2 | 2026-03-25 | Depth, face, gesture, pose |
| v0.0.1 | 2026-03-15 | Initial release |

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

---

## Model Information

| Model | Purpose | Size | FPS |
|:------|:--------|-----:|----:|
| YOLO11n | Object detection | 5.4MB | 139+ |
| MiDaS | Depth estimation | 46MB | 20-30 |
| MediaPipe Face | Face detection | 1.8MB | 30-40 |
| MediaPipe Hands | Gesture recognition | 1.8MB | 20-30 |
| MediaPipe Pose | Pose estimation | 9MB | 15-25 |

---

## Directory Structure

```
openeyes/
├── src/
│   ├── camera/           # Camera handlers
│   ├── models/           # AI models
│   ├── output/           # Output formatters
│   ├── ros2/             # ROS2 integration
│   ├── deepstream/       # DeepStream scripts
│   └── utils/            # Utilities
├── models/               # Model weights
├── docs/                 # Technical docs
├── tests/                # Test suites
├── README.md            # This file
└── requirements.txt     # Dependencies
```

---

## Getting Help

1. **Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)** for common issues
2. **Search existing issues** on GitHub
3. **Open a new issue** with details about your problem

---

<p align="center">
  <sub>OpenEyes v0.0.3 - Robot Vision System</sub>
</p>
