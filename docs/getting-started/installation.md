# Installation

Detailed installation guide for OpenEyes on NVIDIA Jetson Orin Nano.

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
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Hardware Setup

### Required Hardware

| Item | Specification |
|:-----|:--------------|
| **Jetson** | NVIDIA Jetson Orin Nano (4GB or 8GB) |
| **Camera** | CSI Camera (IMX219) or USB Webcam (720p/1080p) |
| **Storage** | 64GB+ microSD or NVMe |
| **Power** | 5V/4A barrel jack or USB-C PD |

### Connect Hardware (CSI Camera)

```
1. Insert microSD card into Jetson
2. Connect CSI camera ribbon to CAM0 connector
3. Connect Ethernet cable (or configure WiFi)
4. Connect power supply
5. Press power button
```

### Connect Hardware (USB Webcam)

```
1. Insert microSD card into Jetson
2. Connect USB webcam to Jetson USB port
3. Connect Ethernet cable (or configure WiFi)
4. Connect power supply
5. Press power button
```

---

## 2. OS Installation

### Download JetPack

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

### Post-Installation Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y git python3-pip python3-venv
```

---

## 3. Python Environment

### Create Virtual Environment

```bash
# Navigate to project
cd /path/to/openeyes

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate
```

### Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Requirements.txt

```
# Core
opencv-python>=4.8.0
numpy>=1.24.0
PyYAML>=6.0
python-dotenv>=1.0.0

# AI/ML
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
onnxruntime>=1.15.0
onnxruntime-gpu>=1.15.0
mediapipe==0.10.9
timm>=1.0.0

# Communication
pyserial>=3.5
```

---

## 4. Project Setup

### Clone Repository

```bash
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes
```

### Create Directories

```bash
mkdir -p models output logs
```

### Configuration

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

### Download YOLOv8

```bash
# Using ultralytics
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx')"

# Or download directly
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
mv yolov8n.pt models/
```

### Download MiDaS

```bash
# Download MiDaS v2.1
wget https://github.com/intel-isd/MiDaS/releases/download/v2.1/midas_v21_onnx.pt
mv midas_v21_onnx.pt models/depth_midas.pt
```

---

## 6. Testing

### Test Camera

```python
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
```

### Test Object Detection

```python
from src.models import ObjectDetector
detector = ObjectDetector('models/yolov8n.pt')
import cv2, numpy as np
frame = np.zeros((480, 640, 3), dtype=np.uint8)
result = detector.detect(frame)
print(f'✓ Object detector works')
```

### Run Full System

```bash
python src/main.py --debug
```

Expected output:
```
[INFO] Camera initialized: 640x480 @ 30fps
[INFO] YOLOv10 loaded successfully
[INFO] Starting vision pipeline...
[INFO] Processing frames...
```

---

## 7. Jetson Performance

### Enable Maximum Performance Mode

For best AI performance on Jetson Orin Nano:

```bash
# Enable 15W power mode (MAXN)
sudo nvpmodel -m 0

# Lock CPU/GPU clocks to maximum
sudo jetson_clocks

# Verify performance mode
sudo nvpmodel -q
```

### Verify CUDA

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Performance Tips

- Use CSI camera instead of USB for lower latency
- Disable unused models for higher FPS
- Use `--no-parallel` if experiencing stability issues
- Use `--pose-every 3` to reduce pose estimation frequency

---

## 8. CSI Camera Setup

### Verify CSI Camera

```bash
# Check camera device
ls -la /dev/video*

# Test with GStreamer
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! fakesink
```

### Camera Device Tree

If camera is not detected, you may need to enable it in extlinux.conf:

```bash
sudo nano /boot/extlinux/extlinux.conf
```

Add to APPEND line:
```
FDT /boot/kernel_og_tegra234-p3768-0000+p3767-0000-nv.dtb
```

!!! note
    Modern JetPack versions usually auto-detect CSI cameras.

---

## 9. Troubleshooting

### Camera Issues

```bash
# List video devices
ls -la /dev/video*

# Restart Argus daemon
sudo systemctl restart nvargus-daemon

# Check GStreamer
gst-inspect-1.0 nvarguscamerasrc
```

### Memory Issues

```bash
# Check available memory
free -h

# Check GPU memory
tegrastats
```

### OpenCV Issues

If OpenCV doesn't work with GStreamer:
```bash
# Uninstall pip OpenCV to use system OpenCV
pip uninstall -y opencv-contrib-python opencv-python
```

### Import Errors

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Display Issues

If display doesn't show:
```bash
# Set display manually
export DISPLAY=:0
```

---

## Uninstall

```bash
# Deactivate virtual environment
deactivate

# Remove directory
cd ..
rm -rf openeyes
```

---

## Next Steps

- Read [User Guide](../user-guide/commands.md) for usage guide
- Check [Troubleshooting Guide](../troubleshooting/common-issues.md) for issues