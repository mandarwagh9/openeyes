# DeepStream Model Conversion Guide

This guide explains how to convert the downloaded models to TensorRT engines for DeepStream.

## On Jetson Orin (recommended)

### 1. Face Detection (ONNX → TensorRT)
```bash
trtexec --onnx=models/yolov8n-face.onnx \
        --saveEngine=models/yolov8n-face.engine \
        --fp16
```

### 2. Hand Pose (PyTorch → ONNX → TensorRT)
Requires trt_pose library:
```bash
pip install trt_pose
python -c "
import torch
from trt_pose.coco import load_model
model = load_model('hand_pose_resnet18_att_244_244.pth')
model.eval()
dummy = torch.randn(1, 3, 244, 244)
torch.onnx.export(model, dummy, 'hand_pose.onnx', input_names=['input'], output_names=['output'])
"
trtexec --onnx=hand_pose.onnx --saveEngine=models/hand_pose.engine --fp16
```

### 3. Body Pose (PyTorch → ONNX → TensorRT)
```bash
trtexec --onnx=models/densenet121_body.onnx \
        --saveEngine=models/body_pose.engine \
        --fp16
```

### 4. Depth Estimation
```bash
# Using MiDaS (simpler, works on any device)
python -c "
import torch
import torch.hub
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.eval()
torch.onnx.export(midas, torch.randn(1,3,384,384), 'models/midas.onnx')
"
trtexec --onnx=models/midas.onnx --saveEngine=models/depth.engine --fp16
```

## Using Docker (for faster conversion)
```bash
docker run -it --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/tensorrt:24.03-py3 \
    trtexec --onnx=models/yolov8n-face.onnx --saveEngine=models/yolov8n-face.engine --fp16
```

## Quick Check
```bash
ls -la models/*.engine
```

## Performance Tips
- Use `--fp16` for half-precision (2x faster)
- Use `--int8` for maximum speed (requires calibration data)
- Pre-generate engines, don't build at runtime