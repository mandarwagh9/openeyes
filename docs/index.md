---
title: Home
template: home.html
---

# OpenEyes

**Open-source vision system for humanoid robots**

<p align="center">
  <img src="assets/images/logo.svg" alt="OpenEyes Logo" width="150"/>
</p>

---

## 🤖 We Give Robots Vision

A humanoid robot needs to see the world like a human does. Not just pixels — but **understanding**. **Distance**. **Intent**. **Action**.

OpenEyes runs entirely on **NVIDIA Jetson** — no cloud, no lag, no dependencies.

<p align="center">
  <a href="getting-started/quickstart.md" class="btn btn-primary">🚀 Quick Start</a>
  <a href="https://github.com/mandarwagh9/openeyes" class="btn btn-secondary">⭐ Star on GitHub</a>
</p>

---

## ✨ Features

<div class="grid cards">

- **🔍 Object Detection** — Real-time detection of 80+ object classes

- **📏 Depth Estimation** — Measure distance to everything in the scene

- **👤 Face Detection** — Who's in the room? Identify and track faces

- **👋 Gesture Recognition** — Understand hand signals — stop, wave, point

- **🦴 Pose Estimation** — Detect body positions and movements

- **🎯 Object Tracking** — Follow specific objects across frames

- **🚶 Person Following** — Autonomous person tracking and following

- **🗺️ Visual SLAM** — Build maps and navigate autonomously

</div>

---

## 🚀 Performance

| Configuration | FPS | Use Case |
|:--------------|:----|:---------|
| All models enabled | 10-15 | Full capability |
| Minimal (no depth/face/gesture/pose) | 25-30 | Speed critical |
| Optimized INT8 | 30-40 | Production deployment |

---

## 💻 Hardware

- **Platform**: NVIDIA Jetson Orin Nano (4GB / 8GB)
- **Camera**: CSI (IMX219) or USB Webcam
- **OS**: Ubuntu 22.04 + JetPack 5.1+
- **Power**: 7-15W

---

## 🛠️ Quick Start

```bash
# Clone and install
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes
pip install -r requirements.txt

# Run with debug window
python src/main.py --debug
```

### ⚡ Jetson Optimization

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## 📦 Supported Models

| Model | Type | Size | Purpose |
|:------|:-----|:-----|:--------|
| YOLO11n | Object Detection | 5.4MB | Real-time detection |
| MiDaS v2.1 | Depth Estimation | 350MB | Monocular depth |
| MediaPipe | Face/Gesture/Pose | ~20MB | Multi-modal ML |
| SmolVLA | VLA | ~450M params | Vision-Language-Action |
| OpenVLA | VLA | 7B params | State-of-the-art VLA |

---

## 📅 The Journey

| Version | Milestone |
|:--------|:----------|
| v0.1.x | Core vision (detection, depth, face, gesture, pose) |
| v0.2.x | Tracking, performance, ROS2 |
| v0.3.x | Model selection |
| v0.4.x | VLA, event camera |
| v0.5.x | Visual odometry, SLAM, Nav2 |
| v0.6.x | Real VLA models (SmolVLA, OpenVLA, Octo) |

---

## 🤝 Contribute

OpenEyes is built by people like you. Developers, researchers, hobbyists, dreamers.

See [Contributing Guide](development/contributing.md) to join us.

---

## 📄 License

Apache 2.0 — See [LICENSE](https://github.com/mandarwagh9/openeyes/blob/main/LICENSE).

> The future of robotics is open. Let's build it together. 🤖