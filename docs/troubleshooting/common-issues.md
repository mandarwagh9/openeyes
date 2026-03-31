# Common Issues

Common problems and solutions for OpenEyes.

---

## Camera Issues

### Camera Not Detected

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
   v4l2-ctl --list-devices
   ```

3. Check USB power:
   ```bash
   lsusb
   ```

4. Try different USB port (USB 2.0 recommended)

### Camera Permission Denied

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

## Model Issues

### Model File Not Found

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

### Model Inference Fails

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

## Performance Issues

### Low FPS

**Symptoms:**
```
FPS: 5.2 (target: 30)
```

**Solutions:**

1. Reduce resolution:
   ```bash
   python src/main.py --width 640 --height 480
   ```

2. Disable unused models in `config.yaml`:
   ```yaml
   models:
     depth:
       enabled: false
     face:
       enabled: false
   ```

3. Enable Jetson max performance:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

4. Check thermal throttling:
   ```bash
   sudo tegrastats
   ```

### Thermal Throttling

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

---

## Output Issues

### No UDP Output

**Solutions:**

1. Verify network:
   ```bash
   ping 192.168.1.100
   ```

2. Test with localhost first:
   ```bash
   python src/main.py --host 127.0.0.1 --port 5000
   
   # In another terminal
   nc -ul 127.0.0.1 5000
   ```

3. Check firewall:
   ```bash
   sudo ufw allow 5000/udp
   ```

---

## Installation Issues

### pip Install Fails

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

### Import Errors

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

## Display Issues

### Window Doesn't Show

**Solutions:**

1. Set DISPLAY environment variable:
   ```bash
   export DISPLAY=:0
   python src/main.py --debug
   ```

2. Check permissions:
   ```bash
   xhost +local:*
   ```

### GTK Errors

These are harmless warnings. Install the module or ignore:
```bash
sudo apt install libcanberra-gtk-module
```

---

## OpenCV Issues

### GStreamer Not Available

**Solutions:**

1. Use system OpenCV instead of pip OpenCV:
   ```bash
   pip uninstall -y opencv-python opencv-contrib-python
   ```

2. Verify system OpenCV has GStreamer:
   ```bash
   python -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer
   ```

---

## MediaPipe Issues

### MediaPipe Crashes

**Solutions:**

1. Downgrade MediaPipe to stable version:
   ```bash
   pip install mediapipe==0.10.9
   ```

2. Disable parallel processing:
   ```bash
   python src/main.py --no-parallel
   ```

3. Use frame skipping:
   ```bash
   python src/main.py --pose-every 3
   ```

---

## ROS2 Issues

### ROS2 Not Available

**Solutions:**

1. Install ROS2 Humble:
   ```bash
   sudo apt update
   sudo apt install ros-humble-rclpy ros-humble-vision-msgs ros-humble-std-msgs
   ```

2. Source ROS2 environment:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

3. Verify installation:
   ```bash
   ros2 topic list
   ```

### Topics Not Appearing

**Solutions:**

1. Check ROS2 is running:
   ```bash
   ros2 node list
   ```

2. The system uses MultiThreadedExecutor automatically.

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

---

## Report a Bug

When reporting issues, include:

1. Hardware setup
2. Software version
3. Full error message
4. Steps to reproduce
5. Debug output