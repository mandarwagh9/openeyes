<p align="center">
  <img src="assets/images/logo.svg" width="80" alt="OpenEyes">
</p>

<h1 align="center">OpenEyes</h1>

<p align="center">
  <strong>v2.5.0</strong> · Hardware-agnostic Edge Robot Vision with World Models<br>
  <br>
  <a href="https://github.com/mandarwagh9/openeyes/stargazers"><img src="https://img.shields.io/github/stars/mandarwagh9/openeyes?style=flat&color=ffffff" alt="Stars"></a>
  <a href="https://github.com/mandarwagh9/openeyes/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://github.com/mandarwagh9/openeyes/actions"><img src="https://img.shields.io/badge/tests-119%20passing-brightgreen" alt="Tests"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python"></a>
  <a href="https://discord.gg/openeyes"><img src="https://img.shields.io/discord/123456789?color=5865F2&label=Discord" alt="Discord"></a>
</p>

---

## What is OpenEyes?

OpenEyes is an open-source, hardware-agnostic robot vision framework for edge AI. It runs on **NVIDIA Jetson**, **Raspberry Pi**, **Intel NPU**, **Hailo**, and **Qualcomm** platforms — with world models for predictive intelligence.

A robot needs to see, predict, and plan — not just react. OpenEyes delivers the full pipeline:

```
Camera → Detection → Tracking → Depth → [World Model] → Planning → Control
                                           ↓
                                   Predictive Safety
```

---

## Features

| Capability | Description |
|:-----------|:------------|
| 🔍 **Object Detection** | YOLO11n/12n/26n with TensorRT optimization (35-40 FPS) |
| 📏 **Depth Estimation** | MiDaS + Depth Anything V3 (35.7% better than MiDaS) |
| 👤 **Face Detection** | MediaPipe FaceMesh (optimized: complexity=0) |
| 👋 **Gesture Recognition** | MediaPipe Hands (optimized: max_hands=1) |
| 🦴 **Pose Estimation** | MediaPipe Pose (optimized: complexity=0) |
| 🎯 **Object Tracking** | ByteTrack with occlusion handling via world models |
| 🚶 **Person Following** | Autonomous person tracking with predictive following |
| 🧠 **World Models** | LeWM (15M) for 100-200 Hz predictive planning |
| 🔮 **V-JEPA 2** | Spatiotemporal features for perception enhancement |
| 🛡️ **Safety** | Predictive collision avoidance, E-STOP, health monitoring |
| 📡 **ROS2** | Full integration with 10+ topics |
| 🚀 **Fleet Management** | Multi-device deployment with OTA updates |
| 🏭 **Industry Templates** | Warehouse, Manufacturing, Agriculture, Retail |
| 🐳 **Docker** | Production-ready containerized deployment |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenEyes v2.5.0 Pipeline                  │
│                                                              │
│  Camera → Detection → Tracking → Depth → [World Model]      │
│                                              ↓               │
│                                    Predictive Planning       │
│                                    Safety Evaluation         │
│                                    Occlusion Handling        │
│                                                              │
│  Hardware Abstraction Layer                                  │
│  TensorRT │ OpenVINO │ TVM │ Hailo │ QNN │ ONNXRuntime       │
│                                                              │
│  Platforms: Jetson Orin │ Pi 5 │ Intel NPU │ Hailo │ Qualcomm│
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install
```bash
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes
pip install -r requirements.txt
```

### 2. Run
```bash
# Basic vision pipeline
python -m src.main --debug

# With person following
python -m src.main --follow --debug

# With world model (predictive tracking)
python -m src.main --world-model lewm --follow --debug

# Turbo mode for maximum FPS
python -m src.main --turbo --world-model lewm --follow --debug

# Industry template (warehouse)
python -m src.main --template warehouse --debug
```

### 3. Optimize (Jetson)
```bash
# One-command performance optimization
sudo bash scripts/jetson_perf.sh
# Expected: 8-12 FPS with full pipeline in turbo mode
```

---

## Performance

| Configuration | FPS (Orin Nano) | Notes |
|:--------------|:----------------|:------|
| Detection only (TensorRT) | 35-40 | YOLO11n FP16 |
| Full pipeline (default) | 4-6 | All models enabled |
| Full pipeline + turbo | 8-12 | Aggressive frame skipping |
| Minimal | 15-20 | Detection + depth + tracking |
| World model (LeWM 15M) | 100-200 Hz | Planning only, <10ms |

---

## Demos

| Demo | Description |
|:----|:------------|
| ![Demo 1](demo/demo1.gif) | Warehouse person following |
| ![Demo 2](demo/demo2.gif) | Multi-object tracking |

Run demo yourself:
```bash
python demo/process_demo.py --input video.mp4 --output output.mp4
```

---

## CLI Reference

### Core Commands

| Flag | Description |
|:-----|:------------|
| `--camera N` | Camera source (default: 0) |
| `--debug` | Show annotated debug window |
| `--follow` | Enable person following |
| `--ros2` | Enable ROS2 publishing |
| `--model NAME` | Detection model (yolo11n, yolo12n, yolo26n) |
| `--list-models` | List available models |

### World Models

| Flag | Description |
|:-----|:------------|
| `--world-model TYPE` | World model (none, lewm, vjepa2) |
| `--plan-horizon N` | Planning horizon in steps (default: 10) |
| `--plan-samples N` | CEM sample count (default: 100) |
| `--prediction-fps N` | Prediction update rate (default: 30) |
| `--occlusion-frames N` | Max frames to predict through occlusion |
| `--safety-predict` | Enable predictive safety evaluation |

### Performance

| Flag | Description |
|:-----|:------------|
| `--turbo` | Aggressive frame skipping for max FPS |
| `--no-face` | Disable face detection |
| `--no-gesture` | Disable gesture recognition |
| `--no-pose` | Disable pose estimation |
| `--no-depth` | Disable depth estimation |
| `--no-tracking` | Disable object tracking |
| `--depth-model M` | Depth model (midas-small, da3-small, da3-base) |
| `--int8` | INT8 quantization |
| `--dla` | DLA offloading |

### Industry Templates

| Flag | Description |
|:-----|:------------|
| `--template warehouse` | Logistics/fulfillment |
| `--template manufacturing-qa` | Quality assurance |
| `--template agriculture` | Farming/crop monitoring |
| `--template retail` | Store analytics |

### Fleet Management

| Command | Description |
|:--------|:------------|
| `openeyes fleet register --name robot-01 --group warehouse` | Register device |
| `openeyes fleet list` | List all devices |
| `openeyes fleet deploy --model yolo26n --version v1.2 --group warehouse` | Deploy model |
| `openeyes fleet telemetry --device robot-01` | View device telemetry |

### REST API

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/health` | GET | Health check with uptime, FPS |
| `/health/live` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |
| `/models` | GET | List loaded models |
| `/models` | POST | Load new model |
| `/control` | GET | Get control state |
| `/control` | POST | Control vision system |
| `/control/start` | POST | Start vision |
| `/control/stop` | POST | Stop vision |

Start API server:
```bash
python -m src.main --api --api-port 8000 --api-host 0.0.0.0
```

---

## Industry Templates

Pre-configured pipelines for high-demand industries:

| Template | Use Case | Key Features |
|:---------|:---------|:-------------|
| **warehouse** | Logistics, fulfillment | Package detection, pallet counting, forklift safety |
| **manufacturing-qa** | Quality assurance | Defect detection, PPE compliance, assembly verification |
| **agriculture** | Farming, crop monitoring | Weed detection, crop health, yield estimation |
| **retail** | Store analytics | Shelf monitoring, inventory counting, customer analytics |

```bash
python -m src.main --template warehouse --debug
```

---

## World Models

OpenEyes integrates world models for **predictive intelligence** — going beyond reactive vision to anticipate and plan.

### LeWorldModel (15M params)
- Latent-space planning at **100-200 Hz**
- Predictive tracking through occlusions (5-10 frames)
- Safety evaluation before action execution
- Online learning from observation history
- Memory: <100MB, Power: 3-5W
- Reference: [arXiv:2603.19312](https://arxiv.org/abs/2603.19312)

### V-JEPA 2 (80M-600M params)
- Spatiotemporal feature extraction from video clips
- Enhances detection accuracy with temporal context
- ViT-B: 10-20 FPS, ViT-L: 3-6 FPS on Orin Nano
- Reference: [Meta V-JEPA 2](https://github.com/facebookresearch/vjepa2)

```bash
# Enable world model with predictive tracking
python -m src.main --world-model lewm --follow

# With safety evaluation
python -m src.main --world-model lewm --safety-predict

# With V-JEPA 2 for enhanced perception
python -m src.main --world-model vjepa2
```

See [docs/WORLD_MODELS.md](docs/WORLD_MODELS.md) for complete documentation.

---

## Hardware Support

| Platform | TOPS | Power | Price | Backend |
|:---------|:----:|:-----:|:-----:|:--------|
| Jetson Orin Nano | 40 | 5-15W | $199 | TensorRT |
| Jetson Orin NX | 100 | 10-25W | $399 | TensorRT |
| Raspberry Pi 5 + AI HAT | 40 | ~12W | ~$150 | Hailo DFC |
| Intel Core Ultra (NPU) | 48 | 15-45W | $400 | OpenVINO |
| Hailo-8 | 26 | 3.5W | $150 | Hailo DFC |
| Qualcomm RB5/RB6 | 15-30 | 5-15W | $700 | QNN |

See [hardware.html](hardware.html) for detailed setup guides.

---

## Production Deployment

### Docker
```bash
cd docker
docker compose up -d
```

### Systemd
```bash
sudo cp docker/openeyes.service /etc/systemd/system/
sudo systemctl enable openeyes
sudo systemctl start openeyes
```

### ROS2
```bash
ros2 launch openeyes openeyes.launch.py
```

---

## Documentation

| Document | Description |
|:---------|:------------|
| [QUICKSTART.md](QUICKSTART.md) | Getting started guide |
| [INSTALL.md](INSTALL.md) | Installation instructions |
| [COMMANDS.md](COMMANDS.md) | Complete CLI reference |
| [USER_GUIDE.md](USER_GUIDE.md) | User guide |
| [OPTIMIZATION.md](OPTIMIZATION.md) | Performance optimization |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions |
| [ROADMAP.md](ROADMAP.md) | Development roadmap |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [docs/WORLD_MODELS.md](docs/WORLD_MODELS.md) | World models documentation |
| [docs/TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md) | Technical specification |
| [WORLD_MODELS_PLAN.md](WORLD_MODELS_PLAN.md) | World models implementation plan |
| [AGENTS.md](AGENTS.md) | Developer guidelines (for AI agents) |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# All 119 tests passing ✓
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

## Acknowledgments

- [Ultralytics](https://ultralytics.com) — YOLO models
- [Meta FAIR](https://ai.facebook.com/research) — V-JEPA 2, DINOv2, SAM 3
- [ByteDance](https://bytedance.com) — Depth Anything V3
- [Hugging Face](https://huggingface.co) — LeRobot, transformers
- [NVIDIA](https://nvidia.com) — TensorRT, Jetson platform
- [MediaPipe](https://google.google.dev/mediapipe) — Face, gesture, pose models
- [Mila/NYU](https://mila.quebec) — LeWorldModel

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mandarwagh9/openeyes&type=Date)](https://www.star-history.com/#mandarwagh9/openeyes&Date)

---

<p align="center">
  If OpenEyes helps your work, please <a href="https://github.com/mandarwagh9/openeyes">star us</a> · <a href="https://discord.gg/openeyes">join Discord</a>
</p>