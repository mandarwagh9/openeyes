# USER_GUIDE.md - User Guide for OpenEyes

> **Version**: v0.2.2  
> **Last Updated**: 2026-03-29

---

## Table of Contents

1. [Overview](#1-overview)
2. [Basic Usage](#2-basic-usage)
3. [Configuration](#3-configuration)
4. [Running Modes](#4-running-modes)
5. [Interpreting Output](#5-interpreting-output)
6. [Advanced Features](#6-advanced-features)
7. [Jetson Optimization](#7-jetson-optimization)

---

## 1. Overview

OpenEyes provides vision capabilities for humanoid robots:

| Capability | Description |
|:-----------|:------------|
| Object Detection | Find and identify objects |
| Depth Estimation | Measure distance to objects |
| Face Detection | Locate faces in frame |
| Gesture Recognition | Understand hand signals |
| Pose Estimation | Detect body positions |

---

## 2. Basic Usage

### 2.1 Start the System

```bash
# Activate environment
source venv/bin/activate

# Run with defaults
python src/main.py
```

### 2.2 Controls

| Key | Action |
|:----|:-------|
| `q` | Quit |
| `s` | Save screenshot |
| `d` | Toggle debug info |
| `space` | Pause/Resume |

---

## 3. Configuration

### 3.1 Command Line Options

```bash
python src/main.py --help
```

```
usage: main.py [-h] [--camera CAMERA] [--width WIDTH] [--height HEIGHT]
               [--fps FPS] [--host HOST] [--port PORT] [--debug]
               [--config CONFIG] [--no-face] [--no-gesture] [--no-pose]
               [--no-depth] [--no-parallel] [--pose-every POSE_EVERY] [--ros2]
               [--version] [--info] [--log-file LOG_FILE]

optional arguments:
  --camera CAMERA       Camera index (0, 1...) or RTSP URL
  --width WIDTH         Frame width (default: 640)
  --height HEIGHT       Frame height (default: 480)
  --fps FPS             Target FPS (default: 30)
  --host HOST           Output host IP (default: 127.0.0.1)
  --port PORT           Output port (default: 5000)
  --debug               Enable debug mode (shows video window)
  --config CONFIG       Config file path
  --no-face             Disable face detection
  --no-gesture          Disable gesture recognition
  --no-pose             Disable pose estimation
  --no-depth            Disable depth estimation
  --no-parallel         Disable parallel processing
  --pose-every N        Run pose every N frames (default: 2)
  --ros2                Enable ROS2 publishing
  --version             Show version
  --info                Show system info and recommendations
  --log-file PATH       Log file path (with rotation)
  --no-pose             Disable pose estimation
  --no-depth            Disable depth estimation (NEW - saves ~2 FPS)
  --no-parallel         Disable parallel processing (more stable)
  --pose-every POSE_EVERY  Run pose estimation every N frames (default: 2)
  --ros2                Enable ROS2 publishing (requires ROS2 installation)
  --version             Show version number and exit
```

### 3.2 Config File

Edit `config.yaml`:

```yaml
camera:
  source: 0
  width: 640
  height: 480
  fps: 30

models:
  yolo:
    confidence: 0.5
    iou_threshold: 0.45
  depth:
    enabled: true
  face:
    enabled: true
  gesture:
    enabled: true
  pose:
    enabled: true

output:
  format: json
  protocol: udp
  host: 127.0.0.1
  port: 5000
  fps: 30
```

---

## 4. Running Modes

### 4.1 Standalone Mode

```bash
# Show visualization window
python src/main.py --camera 0
```

### 4.2 Headless Mode

```bash
# No display, output only via UDP
python src/main.py --camera 0 --no-display
```

### 4.3 Server Mode

```bash
# Run as network server
python src/main.py --server --port 5000
```

---

## 5. Interpreting Output

### 5.1 Console Output

```
[2026-03-13 10:30:45] FPS: 25.3 | Objects: 3 | Faces: 1 | Latency: 38ms
```

| Metric | Description |
|:-------|:------------|
| FPS | Frames per second |
| Objects | Number of detected objects |
| Faces | Number of detected faces |
| Latency | Processing time per frame |

### 5.2 Visual Output

The display shows:
- Bounding boxes around objects
- Labels with confidence scores
- Depth map (if enabled)
- Face/gesture/pose annotations

### 5.3 Network Output

JSON sent via UDP:

```json
{
  "timestamp": 1699123456.123,
  "frame_id": 1234,
  "objects": [
    {"label": "person", "confidence": 0.95, "bbox": [100, 50, 300, 400]}
  ],
  "depth": {"enabled": true, "min_distance": 1.2, "max_distance": 5.0},
  "faces": [],
  "gestures": [{"type": "thumbs_up", "handedness": "right"}],
  "pose": null
}
```

---

## 6. Advanced Features

### 6.1 Enable/Disable Models

```yaml
models:
  yolo:
    enabled: true
  depth:
    enabled: true
  face:
    enabled: false    # Disable face detection
  gesture:
    enabled: true
  pose:
    enabled: false   # Disable pose estimation
```

### 6.2 Adjust Confidence

```yaml
models:
  yolo:
    confidence: 0.7   # Higher = fewer detections but more confident
```

### 6.3 Network Output to Robot

```bash
# Robot controller listens on port 5000
python src/main.py --host 192.168.1.100 --port 5000
```

### 6.4 Multiple Cameras

```bash
# Camera 0 - front view
python src/main.py --camera 0 --name front

# Camera 1 - back view (separate terminal)
python src/main.py --camera 1 --name back
```

### 6.5 Performance Tuning

For optimal performance on Jetson Orin Nano:

```bash
# Run with all optimizations enabled
python src/main.py --debug

# Disable parallel processing if unstable
python src/main.py --no-parallel

# Run pose less frequently for higher FPS
python src/main.py --pose-every 3

# Disable all extra models for maximum object detection FPS
python src/main.py --no-face --no-gesture --no-pose
```

### Performance Examples

| Command | Expected FPS |
|:--------|:------------|
| `python src/main.py` | ~10-12 FPS (all models) |
| `python src/main.py --debug` | ~10-12 FPS |
| `python src/main.py --no-parallel` | 4-6 FPS (more stable) |
| `python src/main.py --pose-every 3` | ~12-14 FPS |
| `python src/main.py --no-face --no-gesture --no-pose` | ~18-22 FPS |
| `python src/main.py --no-face --no-gesture --no-pose --no-depth` | ~22-25 FPS |
| `python src/main.py --no-face --no-gesture --no-pose` + `sudo nvpmodel -m 0 && sudo jetson_clocks` | ~22-28 FPS |

> **Tip**: Run `sudo nvpmodel -m 0 && sudo jetson_clocks` for maximum Jetson performance!

---

## 7. Maintenance

### 7.1 Update Models

```bash
# Download latest YOLOv8
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Update MiDaS
# Download from GitHub releases
```

### 7.2 Check System Health

```bash
# View resource usage
htop

# Check GPU
tegrastats

# View logs
tail -f logs/openeyes.log
```

---

## 7. ROS2 Integration

### 7.1 Enabling ROS2

```bash
python src/main.py --ros2
```

### 7.2 ROS2 Topics

| Topic | Type | Description |
|:------|:-----|:-----------|
| `/vision/detections` | `std_msgs/String` (JSON) | Object detections |
| `/vision/depth` | `sensor_msgs/Image` | Depth map (32FC1, 0-1 meters) |
| `/vision/faces` | `std_msgs/String` (JSON) | Face detections |
| `/vision/gestures` | `std_msgs/String` (JSON) | Gesture recognition results |
| `/vision/poses` | `std_msgs/String` (JSON) | Body pose estimations |
| `/vision/cmd` | `std_msgs/String` | Robot commands (subscribe) |
| `/vision/status` | `std_msgs/String` | System status (FPS, counts) |

### 7.3 Command Subscription

Send commands to `/vision/cmd` topic:

```bash
# Send a command
ros2 topic pub /vision/cmd std_msgs/String "data: 'forward'" -1

# Valid commands: forward, backward, stop, left, right, follow
ros2 topic pub /vision/cmd std_msgs/String "data: 'stop'" -1
ros2 topic pub /vision/cmd std_msgs/String "data: 'left'" -1
ros2 topic pub /vision/cmd std_msgs/String "data: 'right'" -1
```

### 7.4 ROS2 Configuration

Edit `config.yaml`:

```yaml
ros2:
  enabled: false
  node_name: "openeyes_vision"
  topics:
    detections: "/vision/detections"
    depth: "/vision/depth"
    faces: "/vision/faces"
    gestures: "/vision/gestures"
    poses: "/vision/poses"
    cmd: "/vision/cmd"
    status: "/vision/status"
  frame_id: "camera_link"
  confidence_threshold: 0.5
  max_depth_range: 5.0
```

### 7.5 Testing ROS2

```bash
# Check available topics
ros2 topic list

# Monitor detections
ros2 topic echo /vision/detections

# Monitor status
ros2 topic echo /vision/status
```

---

## 8. Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

---

## 7. Jetson Optimization

### Quick Optimization

```bash
# One-command optimization (requires sudo)
sudo bash scripts/jetson_perf.sh
```

### Manual Optimization

```bash
# Set MAX power mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Check System Status

```bash
# Using shell script
bash scripts/jetson_info.sh

# Using Python helper
python3 scripts/jetson_helper.py

# Check optimization status
python3 scripts/jetson_helper.py --check

# Run optimization (requires sudo)
sudo python3 scripts/jetson_helper.py --optimize
```

### System Info Command

```bash
# Shows device info and recommendations
python src/main.py --info
```

---

## 8. Logging

### Enable File Logging

```bash
# Log to file with rotation (5MB max, 3 backups)
python src/main.py --log-file logs/openeyes.log
```

### View Logs

```bash
# View current log
cat logs/openeyes.log

# Follow log in real-time
tail -f logs/openeyes.log
```

---

## 9. Getting Help

| Resource | Link |
|:---------|:-----|
| GitHub Issues | https://github.com/mandarwagh9/openeyes/issues |
| Documentation | See `docs/` folder |
| Discussions | GitHub Discussions |

---

## Appendix: Keyboard Shortcuts

| Key | Action |
|:----|:-------|
| `q` | Quit |
| `s` | Save screenshot |
| `d` | Toggle debug overlay |
| `p` | Pause/Resume |
| `h` | Show help |
| `f` | Toggle fullscreen |
| `1-5` | Switch visualization mode |
