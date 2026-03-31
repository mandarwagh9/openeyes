# Technical Specifications

Detailed technical specifications for OpenEyes.

---

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|:----------|:--------|:------------|
| **Platform** | Jetson Orin Nano 4GB | Jetson Orin Nano 8GB |
| **Camera** | USB Webcam 720p | USB Webcam 1080p |
| **RAM** | 4GB | 8GB |
| **Storage** | 32GB microSD | 64GB+ NVMe |
| **Power** | 5V/3A | 5V/4A |

### Software Requirements

| Component | Version |
|:----------|:--------|
| **OS** | Ubuntu 22.04 LTS |
| **JetPack** | 5.1+ |
| **Python** | 3.10+ |
| **CUDA** | 11.8+ |
| **cuDNN** | 8.6+ |

---

## AI Models

### Primary Models

| Capability | Model | Version | Size |
|:-----------|:------|:--------|:-----|
| Object Detection | YOLO11n | 11.0 | 5.4MB |
| Depth Estimation | MiDaS v2.1 | 2.1 | 350MB |
| Face Detection | MediaPipe Face | 0.10 | ~5MB |
| Gesture | MediaPipe Hands | 0.10 | ~5MB |
| Pose | MediaPipe Pose | 0.10 | ~8MB |

### VLA Models (Advanced)

| Model | Parameters | Platform |
|:------|:-----------|:---------|
| SmolVLA | ~450M | Orin Nano (optimized) |
| OpenVLA | 7B | Orin AGX |
| Octo | ~93M | Orin Nano |

---

## Performance Targets

| Metric | Target | Minimum |
|:-------|:-------|:-------|
| **FPS** | 30 | 20 |
| **Latency** | 30ms | 50ms |
| **Memory Usage** | 1.5GB | 2GB |
| **Model Size** | 25MB | 50MB |

### Real-World Performance

| Configuration | FPS |
|:--------------|:----|
| All models enabled | 10-15 |
| Without face/gesture/pose | 18-22 |
| Without all extras + Jetson max | 22-28 |
| Optimized (INT8 + minimal) | 30-40 |

---

## Data Specifications

### Input

| Parameter | Value |
|:----------|:-----|
| **Resolution** | 640x480 or 1280x720 |
| **Format** | BGR (OpenCV) |
| **Frame Rate** | 30 FPS (capture) |
| **Source** | USB Camera / CSI / RTSP |

### Output

| Parameter | Value |
|:----------|:-----|
| **Format** | JSON over UDP |
| **Protocol** | UDP (default port 5000) |
| **Rate** | Match inference rate |

### ROS2 Message Format

Depth normalized to 0-1 meters (32FC1 format).

---

## Configuration Schema

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
    iou_threshold: 0.45
  depth:
    enabled: true
    path: models/depth_midas.pt
  face:
    enabled: true
    confidence: 0.5
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

ros2:
  enabled: false
  node_name: "openeyes_vision"
  topics:
    detections: "/vision/detections"
    depth: "/vision/depth"
    faces: "/vision/faces"
    gestures: "/vision/gestures"
    poses: "/vision/poses"
```

---

## Glossary

| Term | Definition |
|:----|:----------|
| **Edge AI** | AI processing done on local devices |
| **Inference** | Running AI model to get predictions |
| **TensorRT** | NVIDIA's inference optimizer |
| **ONNX** | Open Neural Network Exchange format |
| **YOLO** | You Only Look Once (object detection) |
| **MiDaS** | Monocular Depth Estimation |
| **MediaPipe** | Google's ML pipeline framework |
| **VLA** | Vision-Language-Action model |
| **SLAM** | Simultaneous Localization and Mapping |
| **Nav2** | ROS2 Navigation Stack |