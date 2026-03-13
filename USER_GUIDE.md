# USER_GUIDE.md - User Guide for PROJECT0

> **Version**: v0.0.1  
> **Last Updated**: 2026-03-13

---

## Table of Contents

1. [Overview](#1-overview)
2. [Basic Usage](#2-basic-usage)
3. [Configuration](#3-configuration)
4. [Running Modes](#4-running-modes)
5. [Interpreting Output](#5-interpreting-output)
6. [Advanced Features](#6-advanced-features)

---

## 1. Overview

PROJECT0 provides vision capabilities for humanoid robots:

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
               [--config CONFIG]

optional arguments:
  --camera CAMERA       Camera index (0, 1...) or RTSP URL
  --width WIDTH         Frame width (default: 640)
  --height HEIGHT       Frame height (default: 480)
  --fps FPS             Target FPS (default: 30)
  --host HOST           Output host IP (default: 127.0.0.1)
  --port PORT           Output port (default: 5000)
  --debug               Enable debug mode
  --config CONFIG       Config file path
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
tail -f logs/project0.log
```

---

## 8. Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

---

## 9. Getting Help

| Resource | Link |
|:---------|:-----|
| GitHub Issues | https://github.com/mandarwagh9/project0/issues |
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
