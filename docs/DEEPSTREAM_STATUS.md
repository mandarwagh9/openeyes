# DeepStream Notes for OpenEyes

## Current Status

### Working (OpenCV Pipeline)
- Regular pipeline: ~12 FPS with YOLO11n
- Use: `python -m src.main --no-depth`

### DeepStream Pipeline
- Creates successfully but **output parsing fails**
- Issue: Pre-built library has NvDsInferParseYolo but not for YOLOv10's output format
- Expected FPS: 40-70 FPS once fixed

## Root Cause

The TensorRT engine outputs tensors that DeepStream's built-in parsers don't understand:
```
Could not find output coverage layer for parsing objects
```

## Solutions

### Option 1: Rebuild with Custom Parser (Recommended)

1. Clone the DeepStream-Yolo repository:
```bash
git clone https://github.com/marcoslucianops/DeepStream-Yolo
```

2. Follow their guide to build a custom parser for your YOLO version

### Option 2: Use Official NVIDIA Models

Download pre-built models from NVIDIA NGC that include parsers:
```bash
# Traffic Cam Net (4 classes)
wget https://ngc.nvidia.com/resource/fb/trafficcamnet.onnx
# Then convert to engine with trtexec
```

### Option 3: Wait for Future Updates

We're tracking this issue. Subscribe to OpenEyes updates for when proper YOLOv10/YOLO11n support is added.

## Quick Test Commands

```bash
# Regular pipeline (works)
python -m src.main --no-depth --debug

# DeepStream (needs parser fix)
python -m src.main --deepstream
```

## Expected Performance (once fixed)

| Platform | Expected FPS |
|----------|--------------|
| Jetson Orin Nano 8GB | 40-70 FPS |
| Jetson Orin NX 16GB | 80-120 FPS |
| AGX Orin 64GB | 150+ FPS |