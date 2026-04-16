# DeepStream SDK Integration Guide

This document describes the DeepStream SDK integration for OpenEyes vision system.

## Overview

DeepStream SDK provides GPU-accelerated video analytics pipelines using GStreamer. The integration enables:
- Hardware-accelerated video capture (CSI cameras)
- TensorRT-optimized inference
- Multi-stream processing
- Production-ready video analytics

## Installation

### DeepStream SDK 7.1

```bash
sudo apt update
sudo apt install -y deepstream-7.1
```

### Python Bindings

```bash
pip3 install pyds
```

## Components

### DeepStream-Yolo Parser

Custom library for YOLO model inference in DeepStream:
- Location: `deepstream/libnvdsinfer_custom_impl_Yolo.so`
- Built from: `https://github.com/marcoslucianops/DeepStream-Yolo`

### Configuration Files

| File | Description |
|------|-------------|
| `deepstream/config_yolov10.txt` | YOLOv10 inference config |
| `deepstream/labels.txt` | COCO class labels |

### Test Scripts

| Script | Description |
|--------|-------------|
| `src/deepstream/test_deepstream.py` | Basic GStreamer pipeline test |
| `src/deepstream/pipeline.py` | DeepStream pipeline class |

## Running

### Basic GStreamer Test

```bash
cd ~/openeyes
python3 src/deepstream/test_deepstream.py
```

### DeepStream Pipeline (with YOLO inference)

```bash
cd ~/openeyes
python -m src.main --deepstream --camera 0
```

### DeepStream with All Models

```bash
# Face, gesture, and pose detection (60 FPS)
python -m src.main --deepstream --camera 0 --enable-face --enable-gesture --enable-pose
```

## Architecture

### Hybrid Approach

```
┌─────────────────────────────────────────────────────────────┐
│                   GStreamer Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  CSI Camera → nvarguscamerasrc → NVDEC → nvinfer (YOLO)   │
│                                              ↓              │
│                                         TensorRT            │
└─────────────────────────────────────────────────────────────┘
                                ↓
                     ┌─────────────────────┐
                     │  nvdsosd (OSD)    │
                     │  + bounding boxes  │
                     └─────────────────────┘
                                ↓
                     ┌─────────────────────┐
                     │  appsink          │
                     │  (Python models)  │
                     └─────────────────────┘
                                ↓
              ┌──────────────────────────────────────┐
              │ MediaPipe: FaceMesh, Hands, Pose │
              └──────────────────────────────────────┘
```

### appsink Integration

For face, gesture, and pose detection, DeepStream uses appsink to extract frames to Python:

1. **nvdsosd** draws YOLO boxes
2. **appsink** extracts RGB frames
3. **MediaPipe** processes face/gesture/pose
4. Results merged with detection output

## Performance

Expected performance on Jetson Orin Nano:

| Configuration | FPS |
|--------------|-----|
| YOLOv10n (TensorRT) | 60+ |
| + Face + Gesture + Pose | 20-40 |

## Notes

- TensorRT engine pre-building requires significant time on first run
- The current implementation uses ONNX with TensorRT runtime
- For maximum performance, pre-build the engine:
  ```bash
  /usr/src/tensorrt/bin/trtexec \
    --onnx=models/yolov10n.onnx \
    --saveEngine=models/yolov10n.engine \
    --fp16 --memPoolSize=workspace:2048
  ```

## Troubleshooting

### Missing Libraries

If you see errors about missing libraries:
```bash
sudo apt install -y librtspserver-1.0-dev librivermax0.7
```

### Camera Not Available

Ensure CSI camera is connected and detected:
```bash
gst-inspect-1.0 nvarguscamerasrc
```

## References

- [DeepStream SDK Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)
- [DeepStream-Yolo GitHub](https://github.com/marcoslucianops/DeepStream-Yolo)
- [NVIDIA Jetson Platform](https://developer.nvidia.com/embedded/jetson)
