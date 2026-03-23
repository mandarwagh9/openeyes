# TROUBLESHOOTING.md - Common Issues and Solutions for OpenEyes

> **Version**: v0.0.1  
> **Last Updated**: 2026-03-13

---

## Table of Contents

1. [Camera Issues](#1-camera-issues)
2. [Model Issues](#2-model-issues)
3. [Performance Issues](#3-performance-issues)
4. [Output Issues](#4-output-issues)
5. [Installation Issues](#5-installation-issues)
6. [Hardware Issues](#6-hardware-issues)

---

## 1. Camera Issues

### Issue: Camera Not Detected

**Symptoms:**
```
ERROR: Camera not found
```

**Solutions:**

1. Check camera is connected:
   ```bash
   ls -la /dev/video*
   ```

2. Verify camera works on host:
   ```bash
   # Test with cheese (GNOME)
   cheese
   
   # Or test with v4l2
   v4l2-ctl --list-devices
   ```

3. Check USB power:
   ```bash
   lsusb
   ```

4. Try different USB port (USB 2.0 recommended)

---

### Issue: Camera Permission Denied

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: /dev/video0
```

**Solutions:**

1. Add user to video group:
   ```bash
   sudo usermod -a -G video $USER
   # Log out and back in
   ```

2. Or temporarily:
   ```bash
   sudo chmod 666 /dev/video0
   ```

---

### Issue: Camera Works but No Frame

**Symptoms:**
```
WARNING: Camera opened but no frame received
```

**Solutions:**

1. Check camera is not in use:
   ```bash
   lsof /dev/video0
   ```

2. Try different resolution:
   ```bash
   python src/main.py --width 640 --height 480
   ```

3. Update camera firmware

---

## 2. Model Issues

### Issue: Model File Not Found

**Symptoms:**
```
FileNotFoundError: models/yolov8n.pt not found
```

**Solutions:**

1. Download model:
   ```bash
   # YOLOv8
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   mv yolov8n.pt models/
   
   # MiDaS
   wget https://github.com/intel-isd/MiDaS/releases/download/v2.1/midas_v21_onnx.pt
   mv midas_v21_onnx.pt models/depth_midas.pt
   ```

2. Verify file exists:
   ```bash
   ls -la models/
   ```

---

### Issue: Model Inference Fails

**Symptoms:**
```
RuntimeError: Inference failed
```

**Solutions:**

1. Check GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

2. Try CPU fallback:
   ```bash
   python src/main.py --device cpu
   ```

3. Re-download model (file may be corrupted)

---

### Issue: Low Detection Accuracy

**Symptoms:**
- Few or no detections
- Wrong classifications

**Solutions:**

1. Lower confidence threshold:
   ```yaml
   # config.yaml
   models:
     yolo:
       confidence: 0.3
   ```

2. Improve lighting

3. Use appropriate model (YOLOv8m instead of YOLOv8n)

---

## 3. Performance Issues

### Issue: Low FPS

**Symptoms:**
```
FPS: 5.2 (target: 30)
```

**Solutions:**

1. Reduce resolution:
   ```bash
   python src/main.py --width 640 --height 480
   ```

2. Disable unused models:
   ```yaml
   models:
     depth:
       enabled: false
     face:
       enabled: false
   ```

3. Enable TensorRT:
   ```bash
   python src/main.py --use-tensorrt
   ```

4. Check thermal throttling:
   ```bash
   sudo tegrastats
   ```

---

### Issue: High Memory Usage

**Symptoms:**
- System becomes slow
- Out of memory errors

**Solutions:**

1. Check memory usage:
   ```bash
   free -h
   ```

2. Reduce buffer sizes in config

3. Disable unused models

4. Add swap space:
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

---

### Issue: Thermal Throttling

**Symptoms:**
```
WARNING: GPU throttling detected
```

**Solutions:**

1. Improve cooling:
   - Add heatsink
   - Install 40mm fan
   - Use NVIDIA Active Cooler

2. Reduce power mode:
   ```bash
   sudo nvpmodel -m 1
   sudo jetson_clocks
   ```

3. Reduce FPS target

---

## 4. Output Issues

### Issue: No UDP Output

**Symptoms:**
- External system not receiving data

**Solutions:**

1. Verify network:
   ```bash
   ping 192.168.1.100
   ```

2. Check port is not blocked:
   ```bash
   sudo netstat -tulpn | grep 5000
   ```

3. Test with localhost first:
   ```bash
   python src/main.py --host 127.0.0.1 --port 5000
   
   # In another terminal
   nc -ul 127.0.0.1 5000
   ```

4. Check firewall:
   ```bash
   sudo ufw allow 5000/udp
   ```

---

### Issue: JSON Parse Error

**Symptoms:**
```
JSONDecodeError: Expecting value
```

**Solutions:**

1. Check output format matches

2. Enable debug mode for raw JSON:
   ```bash
   python src/main.py --debug
   ```

3. Check network latency (may be packet loss)

---

## 5. Installation Issues

### Issue: pip Install Fails

**Symptoms:**
```
ERROR: Could not build wheels for opencv-python
```

**Solutions:**

1. Update pip:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. Install system dependencies:
   ```bash
   sudo apt install -y python3-dev libopencv-dev
   ```

3. Use pre-built wheels:
   ```bash
   pip install opencv-python-headless
   ```

---

### Issue: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Solutions:**

1. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Reinstall requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Check Python version:
   ```bash
   python --version  # Should be 3.10+
   ```

---

## 6. Hardware Issues

### Issue: Jetson Won't Boot

**Symptoms:**
- No display
- Power LED off

**Solutions:**

1. Check power supply (5V/4A minimum)

2. Verify microSD is properly inserted

3. Try recovery mode:
   - Press and hold force recovery button
   - Press power button
   - Release recovery button

4. Re-flash JetPack

---

### Issue: USB Device Not Recognized

**Symptoms:**
- Webcam not in lsusb

**Solutions:**

1. Check USB power:
   ```bash
   dmesg | grep usb
   ```

2. Try powered USB hub

3. Check for driver issues

---

## Debug Commands

### Collect Debug Info

```bash
# System info
uname -a
cat /etc/os-release

# Python environment
python --version
pip list

# GPU status
tegrastats

# Camera
v4l2-ctl --all

# Network
ip addr show

# Memory
free -h
```

---

## Getting More Help

| Resource | Link |
|:---------|:-----|
| GitHub Issues | https://github.com/mandarwagh9/openeyes/issues |
| NVIDIA Forums | https://forums.developer.nvidia.com/ |
| OpenCV Q&A | https://answers.opencv.org/ |

---

## Report a Bug

When reporting issues, include:

1. Hardware setup
2. Software version
3. Full error message
4. Steps to reproduce
5. Debug output
