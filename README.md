# OpenEyes 🤖

[![GitHub stars](https://img.shields.io/github/stars/mandarwagh9/openeyes?style=flat&color=ffffff)](https://github.com/mandarwagh9/openeyes)
[![GitHub forks](https://img.shields.io/github/forks/mandarwagh9/openeyes?style=flat&color=ffffff)](https://github.com/mandarwagh9/openeyes)
[![Tests](https://img.shields.io/badge/tests-119%20passing-brightgreen)](https://github.com/mandarwagh9/openeyes/actions)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Discord](https://img.shields.io/discord/123456789?color=5865F2&label=Discord)](https://discord.gg/openeyes)

**v2.5.0** · Hardware-agnostic edge robot vision framework with world models for predictive intelligence.

---

## The Problem

Current robot vision is **reactive** — robots see, then react. But by the time a robot sees an obstacle, it's often too late. True robot intelligence needs to **predict** and **plan**.

## The Solution

OpenEyes delivers the full perception-to-planning pipeline:

```
Camera → Detection → Tracking → Depth → [World Model] → Planning → Control
                                            ↓
                                    Predictive Safety
```

A robot doesn't just see — it anticipates what's coming and plans around it.

---

## ✨ Features

| Category | Capabilities |
|:---------|:-------------|
| **Perception** | Object detection, depth estimation, face/gesture/pose detection, visual SLAM |
| **Intelligence** | World models (LeWM), V-JEPA 2, predictive tracking, occlusion handling |
| **Control** | Action chunking (10-30 Hz), diffusion policy, person following |
| **Safety** | Predictive collision avoidance, E-STOP, velocity/distance limits, health monitoring |
| **Integration** | ROS2 (7+ topics), REST API, Prometheus metrics, UDP/JSON output |
| **Hardware** | TensorRT, OpenVINO, Hailo DFC, ONNXRuntime — Jetson, Pi, Intel, Hailo, Qualcomm |
| **Industry** | Warehouse, manufacturing QA, agriculture, retail templates |

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes
pip install -r requirements.txt

# 2. Run with camera
python -m src.main --debug

# 3. Enable predictive tracking
python -m src.main --world-model lewm --follow --debug
```

That's it. The vision pipeline starts automatically.

---

## 📊 Performance

On **Jetson Orin Nano** (40 TOPS):

| Mode | FPS | Notes |
|:-----|:----|:------|
| Detection only (TensorRT) | 35-40 | YOLO11n |
| Full pipeline | 4-6 | All models |
| Turbo mode | 8-12 | Frame skipping |
| World model (planning) | 100-200 Hz | <10ms latency |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      INPUT                                    │
│  Camera (CSI/USB/Realsense) · Video file · ROS2 topic        │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│                    PERCEPTION LAYER                         │
│  ┌──────────┐  ┌────────┐  ┌─────┐  ���────────┐       │
│  │ YOLO11n  │  │ MiDaS  │  │MediaPipe    │  │ByteTrack│       │
│  │Detect   │  │Depth  │  │Face/Gest/Pose│  │Track  │       │
│  └──────────┘  └────────┘  └─────┘  └────────┘       │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│                   WORLD MODEL LAYER                         │
│  ┌──────────────────────────────────┐                   │
│  │      LeWorldModel (15M params)     │                   │
│  │  • Predictive tracking            │                   │
│  │  • Occlusion handling          │                   │
│  │  • Safety evaluation          │                   │
│  │  • CEM planner              │                   │
│  └──────────────────────────────────┘                   │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                          │
│  Debug window · JSON · UDP · ROS2 · Video file · REST API      │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ Supported Hardware

| Platform | TOPS | Power | Price |
|:---------|:----:|:-----:|:-----:|
| Jetson Orin Nano | 40 | 5-15W | $199 |
| Jetson Orin NX | 100 | 10-25W | $399 |
| Raspberry Pi 5 + AI HAT | 40 | ~12W | $150 |
| Intel Core Ultra (NPU) | 48 | 15-45W | $400 |
| Hailo-8 | 26 | 3.5W | $150 |
| Qualcomm RB5 | 15-30 | 5-15W | $700 |

---

## 📦 Production

### Docker

```bash
cd docker
docker compose up -d
```

### ROS2 Launch

```bash
ros2 launch openeyes openeyes.launch.py
```

---

## 🔧 CLI Options

```bash
# Camera
python -m src.main --camera 0 --width 640 --height 480

# Models
python -m src.main --model yolo12n --depth-model da3-base

# World models
python -m src.main --world-model lewm --follow --safety-predict

# Performance
python -m src.main --int8 --dla --turbo

# Video processing
python -m src.main --video input.mp4 --output output.mp4

# REST API
python -m src.main --api --api-port 8000
```

See [COMMANDS.md](COMMANDS.md) for the full reference.

---

## 📚 Documentation

| Guide | Description |
|:-----|:------------|
| [Getting Started](QUICKSTART.md) | 5-minute quick start |
| [Installation](INSTALL.md) | Full installation guide |
| [World Models](docs/WORLD_MODELS.md) | Predictive intelligence docs |
| [Hardware](hardware.html) | Platform setup guides |
| [ROS2](ros2.html) | ROS2 integration |
| [Safety](safety.html) | Safety features |

---

## 🧪 Test Suite

```bash
pytest tests/ -v
# 119 tests passing ✓
```

---

## 🙏 Thanks To

- [Ultralytics](https://ultralytics.com) — YOLO
- [Meta FAIR](https://ai.facebook.com/research) — V-JEPA 2, DINOv2
- [ByteDance](https://bytedance.com) — Depth Anything
- [NVIDIA](https://nvidia.com) — TensorRT, Jetson
- [MediaPipe](https://google.google.dev/mediapipe) — Face, gesture, pose
- [Mila](https://mila.quebec) — LeWorldModel

---

## 📄 License

[Apache 2.0](LICENSE) — © 2024-2025 Mandar Wagh

---

## ⭐ If OpenEyes Helps You

[Star us on GitHub](https://github.com/mandarwagh9/openeyes) · [Join Discord](https://discord.gg/openeyes) · [Follow @mandarwagh9](https://twitter.com/mandarwagh9)