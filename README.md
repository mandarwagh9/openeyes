<p align="center">
  <img src="assets/images/logo.svg" width="80" alt="OpenEyes">
</p>

<h1 align="center">OpenEyes</h1>

<p align="center">
  <strong>v3.0.0</strong> · Robot Vision for Edge Devices<br>
  <br>
  <a href="https://github.com/mandarwagh9/openeyes/stargazers"><img src="https://img.shields.io/github/stars/mandarwagh9/openeyes?style=flat&color=ffffff" alt="Stars"></a>
  <a href="https://github.com/mandarwagh9/openeyes/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python"></a>
  <a href="https://discord.gg/openeyes"><img src="https://img.shields.io/discord/123456789?color=5865F2&label=Discord" alt="Discord"></a>
</p>

---

## What is OpenEyes?

OpenEyes is an open-source robot vision framework for edge devices. It runs on **NVIDIA Jetson**, **Raspberry Pi + AI HAT**, **Intel NPU**, and **Hailo** — giving robots the ability to see, track, and follow people in real-time.

Built for production: TensorRT optimization, ROS2 integration, and Docker deployment out of the box.

```
Camera → Detection → Tracking → Depth → Control
```

---

## Demos

| | |
|---|---|
| ![Demo 1](demo/demo1.gif) | ![Demo 2](demo/demo2.gif) |

---

## Features

| Capability | Description |
|:-----------|:------------|
| 🚀 **DeepStream** | Hardware-accelerated pipeline (30-60 FPS on Jetson) |
| 🔍 **Object Detection** | YOLO11n/12n with TensorRT optimization |
| 📏 **Depth Estimation** | MiDaS + Depth Anything V3 |
| 👤 **Face Detection** | MediaPipe FaceMesh |
| 👋 **Gesture Recognition** | MediaPipe Hands |
| 🦴 **Pose Estimation** | MediaPipe Pose |
| 🎯 **Object Tracking** | ByteTrack with occlusion handling |
| 🚶 **Person Following** | Autonomous person tracking |
| 📡 **ROS2** | Full integration with 10+ topics |
| 🐳 **Docker** | Production-ready containerized deployment |

---

## Quick Start

### Install
```bash
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes
pip install -r requirements.txt
```

### Run
```bash
# Basic vision pipeline
python -m src.main --debug

# DeepStream pipeline (NEW - 30 FPS)
python -m src.main --deepstream --camera 0

# With person following
python -m src.main --follow --debug

# Turbo mode for maximum FPS
python -m src.main --turbo --follow --debug

# ROS2 mode
python -m src.main --ros2 --debug
```

### DeepStream Quick Start
```bash
# One-time setup (with internet)
python setup_plug_and_play.py

# Run DeepStream pipeline
python -m src.main --deepstream --camera 0

# Run all demos
python demo_all_features.py
```

### Optimize (Jetson)
```bash
sudo bash scripts/jetson_perf.sh
```

---

## Performance

| Configuration | FPS (Orin Nano) | Notes |
|:--------------|:----------------|:------|
| **DeepStream pipeline** | **30-40** | YOLOv10n + TensorRT |
| Detection only (INT8) | 50-80 | YOLO11n INT8 + TensorRT |
| Full pipeline + INT8 | 15-25 | All models with INT8 |
| Full pipeline + INT8 + Turbo | 25-35 | Aggressive frame skipping |
| Minimal (no face/gesture/pose) | 25-40 | Detection + depth + tracking |
| DLA mode | 20-30 | GPU + DLA offload |

### How We Achieved 30 FPS (vs 2 FPS)

The original OpenCV pipeline was slow because:
```
cv2.VideoCapture → CPU decode → YOLO (CPU) → cv2.rectangle (CPU)
= ~2 FPS on Jetson
```

DeepStream uses **hardware-accelerated** GPU pipeline:
```
nvarguscamerasrc → NVMM → nvinfer (TensorRT/GPU) → nvdsosd (GPU) → nv3dsink
= ~30 FPS on Jetson
```

**Key optimizations:**
1. **nvarguscamerasrc** - CSI camera hardware decode
2. **NVMM** - Zero-copy video memory
3. **nvinfer** - TensorRT inference (not ONNX Runtime)
4. **nvdsosd** - GPU-accelerated drawing
5. **nv3dsink** - Hardware display output

Run benchmark:
```bash
python -m benchmarks.run_deepstream_benchmark --compare
```

---

## Run Commands

```bash
# Default (~8-15 FPS)
python -m src.main --debug

# With INT8 (~15-25 FPS)
python -m src.main --int8 --debug

# INT8 + Turbo (~25-35 FPS)
python -m src.main --int8 --turbo --debug

# Minimal (~25-40 FPS)
python -m src.main --int8 --no-face --no-gesture --no-pose --debug

# DLA mode
python -m src.main --dla --debug

# Run optimization script first
sudo bash scripts/jetson_perf.sh
```

---

## Supported Platforms

| Platform | Backend | Notes |
|:---------|:--------|:------|
| Jetson Orin Nano/NX | TensorRT | Primary target |
| Raspberry Pi 5 + AI HAT | Hailo DFC | ~40 TOPS |
| Intel Core Ultra (NPU) | OpenVINO | ~48 TOPS |
| Hailo-8 | Hailo DFC | ~26 TOPS, 3.5W |

---

## CLI Reference

| Flag | Description |
|:-----|:------------|
| `--camera N` | Camera source (default: 0) |
| `--video FILE` | Process video file |
| `--debug` | Show annotated debug window |
| `--follow` | Enable person following |
| `--ros2` | Enable ROS2 publishing |
| `--turbo` | Aggressive frame skipping |
| `--model NAME` | Detection model (yolo11n, yolo12n, yolo26n) |
| `--depth-model NAME` | Depth model (midas-small, da3-small, da3-base) |
| `--no-face`, `--no-gesture`, `--no-pose`, `--no-depth`, `--no-tracking` | Disable specific models |
| `--list-models` | List available models |

---

## ROS2 Topics

| Topic | Type |
|:------|:----|
| `/vision/detections` | JSON |
| `/vision/depth` | JSON |
| `/vision/faces` | JSON |
| `/vision/gestures` | JSON |
| `/vision/pose` | JSON |
| `/vision/status` | JSON |
| `/vision/predictions` | JSON |
| `/vision/safety` | JSON |

---

## Docker

```bash
cd docker
docker compose up -d
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Documentation

| Document | Location |
|:---------|:---------|
| Getting Started | [docs/getting-started/](docs/getting-started/) |
| Troubleshooting | [docs/troubleshooting/](docs/troubleshooting/) |
| Technical Spec | [docs/concepts/technical-spec.md](docs/concepts/technical-spec.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

## Acknowledgments

- [Ultralytics](https://ultralytics.com) — YOLO models
- [MediaPipe](https://google.google.dev/mediapipe) — Face, gesture, pose models
- [Depth Anything](https://github.com/DepthAnything/Depth-Anything-V3) — Depth estimation
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — Object tracking
- [NVIDIA](https://nvidia.com) — TensorRT, Jetson platform

---

<p align="center">
  If OpenEyes helps your work, please <a href="https://github.com/mandarwagh9/openeyes">star us</a> · <a href="https://discord.gg/openeyes">join Discord</a>
</p>
