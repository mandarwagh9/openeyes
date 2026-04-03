# INSTALL.md - Detailed Installation Guide for OpenEyes

> **Version**: v2.5.0-dev  
> **Last Updated**: 2026-04-03

---

## Table of Contents

1. [Hardware Setup](#1-hardware-setup)
2. [OS Installation](#2-os-installation)
3. [Python Environment](#3-python-environment)
4. [Project Setup](#4-project-setup)
5. [Model Download](#5-model-download)
6. [Testing](#6-testing)
7. [Jetson Performance](#7-jetson-performance)
8. [CSI Camera Setup](#8-csi-camera-setup)
9. [Docker Installation](#9-docker-installation)
10. [Multi-Platform Support](#10-multi-platform-support)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Hardware Setup

### 1.1 Supported Platforms

| Platform | TOPS | Power | Price | Backend |
|:---------|:-----|:------|:------|:--------|
| **Jetson Orin Nano** | 40 | 5-15W | $199-249 | TensorRT |
| **Jetson Orin NX** | 100 | 10-25W | $399-499 | TensorRT |
| **Raspberry Pi 5 + AI HAT+ 2** | 40 | ~12W | ~$150 | Hailo DFC |
| **Intel Core Ultra** | 48 | 15-45W | $300-600 | OpenVINO |
| **Hailo-8** | 26 | 3.5W | $150-200 | Hailo DFC |
| **Qualcomm RB5/RB6** | 15-30 | 5-15W | $600-800 | QNN |

### 1.2 Required Hardware

| Item | Specification |
|:-----|:--------------|
| **Board** | Any supported platform above |
| **Camera** | CSI Camera (IMX219) or USB Webcam (720p/1080p) |
| **Storage** | 64GB+ microSD or NVMe |
| **Power** | Appropriate power supply for your platform |

---

## 2. OS Installation

### 2.1 Jetson Orin Nano

1. Download JetPack 6.2 from [NVIDIA Developer](https://developer.nvidia.com/embedded/jetpack)
2. Flash using NVIDIA SDK Manager or `jetson-disk-image-creator.sh`
3. Boot and complete initial setup

### 2.2 Raspberry Pi 5

1. Install Raspberry Pi OS (64-bit) using Raspberry Pi Imager
2. Enable camera: `sudo raspi-config` → Interface Options → Camera
3. Install Hailo AI HAT+ 2 drivers per Hailo documentation

### 2.3 Intel Core Ultra

1. Install Ubuntu 22.04 or 24.04
2. Install Intel NPU drivers from [Intel](https://github.com/intel/linux-npu-driver)

---

## 3. Python Environment

```bash
# Python 3.10+ required
python3 --version

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

## 4. Project Setup

```bash
# Clone repository
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m src.main --version
```

---

## 5. Model Download

Models are included in the `models/` directory:
- `yolo11n.engine` - TensorRT engine (FP16)
- `yolo11n.onnx` - ONNX format
- `yolo26n.pt` - PyTorch format (latest SOTA)

### Download Additional Models

```bash
# Download YOLO26n
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"

# Rebuild TensorRT engine with optimizations
python scripts/export_tensorrt_optimized.py --model models/yolo11n.pt
```

---

## 6. Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Current: 119 tests passing
```

---

## 7. Jetson Performance

```bash
# One-command optimization (recommended)
sudo bash scripts/jetson_perf.sh

# Verify
nvpmodel -q
tegrastats
```

---

## 8. CSI Camera Setup

### 8.1 Verify Camera

```bash
# Check for CSI camera device
ls /dev/video*

# Test GStreamer pipeline
gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! video/x-raw,width=640,height=480 ! xvimagesink
```

### 8.2 Troubleshooting

```bash
# If camera not detected:
sudo systemctl restart nvargus-daemon

# Check camera status
v4l2-ctl --list-devices
```

---

## 9. Docker Installation

### 9.1 Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 9.2 Run with Docker

```bash
cd docker
docker compose up -d

# View logs
docker compose logs -f
```

### 9.3 Systemd Service

```bash
# Install service
sudo cp docker/openeyes.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable openeyes
sudo systemctl start openeyes

# View logs
sudo journalctl -u openeyes -f
```

---

## 10. Multi-Platform Support

### 10.1 Hardware Abstraction Layer

OpenEyes automatically detects your platform and selects the optimal backend:

```bash
# Auto-detect and run
python -m src.main --camera 0

# Force specific backend
python -m src.main --camera 0 --backend tensorrt
python -m src.main --camera 0 --backend openvino
python -m src.main --camera 0 --backend hailo_dfc
```

### 10.2 Platform Detection

```bash
# Show detected platform info
python -c "from src.platforms import PlatformDetector; print(PlatformDetector.detect())"
```

---

## 11. Troubleshooting

### No Camera Detected
```bash
ls /dev/video*
sudo systemctl restart nvargus-daemon  # Jetson
```

### Low FPS
```bash
sudo bash scripts/jetson_perf.sh
python -m src.main --camera 0 --turbo
```

### Out of Memory
```bash
# GStreamer pipeline now captures at 1280x720 (fixed in v1.5.0)
# If still OOM, increase swap:
sudo fallocate -l 4G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
```

### Depth Anything V3 Requires HuggingFace Token
```bash
# Use MiDaS by default (works offline)
python -m src.main --camera 0 --depth-model midas-small

# Or login for DA3
huggingface-cli login
python -m src.main --camera 0 --depth-model da3-small
```

### Docker GPU Not Available
```bash
# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```
