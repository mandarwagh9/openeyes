# OpenEyes

**v0.6.0** · 🤖 We Give Robots Vision

[![GitHub stars](https://img.shields.io/github/stars/mandarwagh9/openeyes?style=social)](https://github.com/mandarwagh9/openeyes)
[![PyPI](https://img.shields.io/pypi/v/openeyes)](https://pypi.org/project/openeyes/)
[![License](https://img.shields.io/github/license/mandarwagh9/openeyes)](LICENSE)

---

## What is OpenEyes?

OpenEyes is an open-source vision system for humanoid robots. It runs entirely on **NVIDIA Jetson Orin Nano** — no cloud, no lag, no dependencies.

A humanoid robot needs to see the world like a human does. Not just pixels — but understanding. Distance. Intent. Action.

---

## ✨ Features

| Capability | Description |
|:-----------|:------------|
| 🔍 **Object Detection** | Real-time detection of 80+ object classes |
| 📏 **Depth Estimation** | Measure distance to everything in the scene |
| 👤 **Face Detection** | Identify and track faces |
| 👋 **Gesture Recognition** | Understand hand signals |
| 🦴 **Pose Estimation** | Detect body positions |
| 🎯 **Object Tracking** | Follow specific objects |
| 🚶 **Person Following** | Autonomous person tracking |
| 🗺️ **Visual SLAM** | Build maps and navigate |
| 🤖 **VLA Models** | Vision-Language-Action (SmolVLA, OpenVLA, Octo) |

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes
pip install -r requirements.txt

# Run with debug window
python src/main.py --debug
```

### First Time on Jetson?

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## ⚡ Performance

| Configuration | FPS | Use Case |
|:--------------|:----|:---------|
| All models | 10-15 | Full capability |
| Minimal | 25-30 | Speed critical |
| Optimized INT8 | 30-40 | Production |

---

## 💻 Hardware

- **Platform**: NVIDIA Jetson Orin Nano (4GB / 8GB)
- **Camera**: CSI (IMX219) or USB Webcam
- **OS**: Ubuntu 22.04 + JetPack 5.1+

---

## 📖 Documentation

**Live Site**: https://mandarwagh9.github.io/openeyes/

| Guide | Description |
|:------|:------------|
| [Quick Start](docs/getting-started/quickstart.md) | Get up and running in 5 minutes |
| [Installation](docs/getting-started/installation.md) | Detailed installation guide |
| [Commands](docs/user-guide/commands.md) | All CLI options |
| [ROS2](docs/user-guide/ros2.md) | ROS2 integration |
| [Hardware](docs/reference/hardware.md) | Hardware specifications |

---

## 🤖 ROS2 Integration

```bash
# Enable ROS2 publishing
python src/main.py --ros2

# Topics published:
# /vision/detections  - Object detections
# /vision/depth       - Depth map
# /vision/faces       - Face detections
# /vision/gestures   - Gesture recognition
# /vision/poses      - Body poses
# /vision/status     - System status
```

---

## 📦 Models

| Model | Type | Size | Platform |
|:------|:-----|:-----|:---------|
| YOLO11n | Detection | 5.4MB | Orin Nano |
| MiDaS v2.1 | Depth | 350MB | Orin Nano |
| MediaPipe | Face/Gesture/Pose | ~20MB | Orin Nano |
| SmolVLA | VLA | ~450M | Orin Nano (optimized) |
| OpenVLA | VLA | 7B | Orin AGX |

---

## 📅 Version History

| Version | Milestone |
|:--------|:----------|
| v0.6.0 | Real VLA models, Nav2, SLAM |
| v0.5.0 | Visual odometry, SLAM |
| v0.4.x | VLA, event camera |
| v0.3.x | Model selection |
| v0.2.x | Tracking, ROS2, performance |
| v0.1.x | Core vision |

---

## 🤝 Contribute

OpenEyes is built by people like you. See [CONTRIBUTING.md](CONTRIBUTING.md) to join us.

---

## 📄 License

Apache 2.0 — See [LICENSE](LICENSE).

> The future of robotics is open. Let's build it together. 🤖