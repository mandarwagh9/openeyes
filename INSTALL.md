# INSTALL.md - Detailed Installation Guide for OpenEyes

> **Version**: v0.0.1  
> **Last Updated**: 2026-03-13

---

## Table of Contents

1. [Hardware Setup](#1-hardware-setup)
2. [OS Installation](#2-os-installation)
3. [Python Environment](#3-python-environment)
4. [Project Setup](#4-project-setup)
5. [Model Download](#5-model-download)
6. [Testing](#6-testing)
7. [Optional: TensorRT](#7-optional-tensorrt)

---

## 1. Hardware Setup

### 1.1 Required Hardware

| Item | Specification |
|:-----|:--------------|
| **Jetson** | NVIDIA Jetson Orin Nano (4GB or 8GB) |
| **Camera** | USB Webcam (720p or 1080p) |
| **Storage** | 64GB+ microSD or NVMe |
| **Power** | 5V/4A barrel jack or USB-C PD |

### 1.2 Connect Hardware

```
1. Insert microSD card into Jetson
2. Connect USB webcam to Jetson USB port
3. Connect Ethernet cable (or configure WiFi)
4. Connect power supply
5. Press power button
```

---

## 2. OS Installation

### 2.1 Download JetPack

1. Download **NVIDIA JetPack SDK Manager** from:
   https://developer.nvidia.com/embedded/jetpack

2. Run SDK Manager:
   ```bash
   sdkmanager
   ```

3. Select:
   - Jetson Orin Nano
   - JetPack 5.1+
   - Ubuntu 22.04

4. Flash to device

### 2.2 Post-Installation Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y git python3-pip python3-venv
```

---

## 3. Python Environment

### 3.1 Create Virtual Environment

```bash
# Navigate to project
cd /path/to/openeyes

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate
```

### 3.2 Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 3.3 Requirements.txt

```
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
onnxruntime>=1.15.0
mediapipe>=0.10.0
pyserial>=3.5
PyYAML>=6.0
python-dotenv>=1.0.0
```

---

## 4. Project Setup

### 4.1 Clone Repository

```bash
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes
```

### 4.2 Create Directories

```bash
mkdir -p models output logs
```

### 4.3 Configuration

Create `config.yaml`:

```yaml
camera:
  source: 0
  width: 640
  height: 480
  fps: 30

models:
  yolo:
    path: models/yolov8n.pt
    confidence: 0.5
  depth:
    enabled: true
    path: models/depth_midas.pt

output:
  format: json
  protocol: udp
  host: 127.0.0.1
  port: 5000
```

---

## 5. Model Download

### 5.1 Download YOLOv8

```bash
# Using ultralytics
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx')"

# Or download directly
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
mv yolov8n.pt models/
```

### 5.2 Download MiDaS

```bash
# Download MiDaS v2.1
wget https://github.com/intel-isd/MiDaS/releases/download/v2.1/midas_v21_onnx.pt
mv midas_v21_onnx.pt models/depth_midas.pt
```

---

## 6. Testing

### 6.1 Test Camera

```bash
python -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('✓ Camera detected')
    ret, frame = cap.read()
    if ret:
        print(f'✓ Frame captured: {frame.shape}')
    cap.release()
else:
    print('✗ Camera not found')
"
```

### 6.2 Test Object Detection

```bash
python -c "
from src.models import ObjectDetector
detector = ObjectDetector('models/yolov8n.pt')
import cv2, numpy as np
frame = np.zeros((480, 640, 3), dtype=np.uint8)
result = detector.detect(frame)
print(f'✓ Object detector works')
"
```

### 6.3 Run Full System

```bash
python src/main.py --debug
```

Expected output:
```
[INFO] Camera initialized: 640x480 @ 30fps
[INFO] YOLOv8n loaded successfully
[INFO] Starting vision pipeline...
[INFO] Processing frames...
```

---

## 7. Optional: TensorRT

For better performance, convert models to TensorRT:

### 7.1 Install TensorRT

```bash
# TensorRT comes with JetPack
# Verify installation
python -c "import tensorrt; print(tensorrt.__version__)"
```

### 7.2 Convert YOLO to TensorRT

```bash
python scripts/convert_to_tensorrt.py --model yolov8n.pt
```

---

## 8. Troubleshooting

### 8.1 Camera Issues

```bash
# List video devices
ls -la /dev/video*

# Check camera with v4l2
v4l2-ctl --list-devices
```

### 8.2 Memory Issues

```bash
# Check available memory
free -h

# Check GPU memory
tegrastats
```

### 8.3 Import Errors

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

---

## 9. Uninstall

```bash
# Deactivate virtual environment
deactivate

# Remove directory
cd ..
rm -rf openeyes
```

---

## Next Steps

- Read [USER_GUIDE.md](USER_GUIDE.md) for usage guide
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
