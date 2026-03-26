# Performance Optimization Guide

This guide covers optimization techniques to achieve 15+ FPS on Jetson Orin Nano.

## Quick Start - Enable Max Performance

```bash
# Maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Verify
nvpmodel -q
jetson_clocks --show
```

## Model Selection

| Model | Size | FP16 FPS | INT8 FPS | Recommended |
|-------|------|----------|----------|-------------|
| YOLO11n | 5.4MB | 139 | 180 | Best speed |
| YOLOv10n | 4.7MB | 139 | 180 | Good alternative |
| YOLO11s | 18.6MB | 100 | 133 | Higher accuracy |

### Export Commands

```bash
# YOLO11n - FP16 (recommended)
yolo export model=yolo11n.pt format=engine half=True

# YOLO11n - INT8 (fastest)
yolo export model=yolo11n.pt format=engine int8=True
```

## TensorRT Engine Export

Using the pre-built TensorRT engine is recommended for maximum performance on Jetson.

### Why TensorRT?

| Mode | Inference FPS | Overall FPS Impact |
|------|---------------|-------------------|
| ONNX Runtime (CPU) | ~5 FPS | 4-5 FPS overall |
| PyTorch + CUDA | ~20 FPS | 10-15 FPS overall |
| **TensorRT FP16** | **~51 FPS** | **10-15 FPS overall** |

The YOLO inference is now 10x faster with TensorRT. The overall FPS is limited by MediaPipe CPU models.

### Pre-built Engine

The project includes a pre-built TensorRT engine:
- **File**: `models/yolo11n.engine`
- **Size**: 8.2 MB
- **Precision**: FP16
- **Performance**: ~51 FPS inference

### Export New Engine

To export a new engine (e.g., after retraining):

```bash
# Using the export script (recommended)
python3 scripts/export_tensorrt.py --model models/yolo11n.pt

# Or using Ultralytics CLI
yolo export model=models/yolo11n.pt format=engine half=True imgsz=640
```

### Using the Engine

Update `config.yaml` to use the engine:

```yaml
models:
  yolo:
    path: models/yolo11n.engine  # TensorRT engine
    confidence: 0.5
```

The system automatically detects `.engine` files and uses Ultralytics' TensorRT backend.

## Frame Skipping

The system includes adaptive frame skipping that automatically adjusts based on motion:

### Default Intervals

```python
skip_intervals = {
    'detector': 1,   # Process every frame (TensorRT - very fast)
    'depth': 4,      # Process every 4th frame (CPU - heaviest)
    'face': 4,       # Process every 4th frame (CPU)
    'gesture': 4,    # Process every 4th frame (CPU)
    'pose': 4        # Process every 4th frame (CPU)
}
```

> **Note:** Skip intervals increased from 2 to 4 to reduce CPU load. MediaPipe models run on CPU and are the main bottleneck.

### Adaptive Skipping

The `AdaptiveFrameSkipper` automatically adjusts skip interval based on motion:
- Low motion (static scene): skip more frames (up to 5)
- High motion (active scene): process more frames

### Manual Configuration

Edit `src/main.py` to customize:

```python
self._adaptive_skipper = AdaptiveFrameSkipper(
    base_skip=3,           # Default skip interval
    motion_threshold=5000, # Motion detection threshold
    min_skip=2,            # Minimum skip (2 = every other frame)
    max_skip=5             # Maximum skip
)
```

## Memory Management

### Enable Memory Optimization

```python
import gc
import torch

# After each inference cycle
gc.collect()
torch.cuda.empty_cache()
```

### System Memory

Increase swap space for model loading:
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Input Resolution

Lower resolution = faster inference:

| Resolution | FPS Multiplier | Accuracy |
|------------|----------------|----------|
| 640x640 | 1x | Full |
| 512x512 | 1.4x | -2% |
| 416x416 | 1.8x | -4% |

Modify in `config.yaml`:
```yaml
camera:
  width: 416
  height: 416
```

## Performance Comparison

| Configuration | All Models FPS | Object Only FPS |
|--------------|----------------|-----------------|
| YOLO11n ONNX (CPU) | 4-5 | 10-15 |
| YOLO11n PyTorch + CUDA | 7-10 | 20-30 |
| **YOLO11n TensorRT FP16** | **10-15** | **30-50** |
| YOLO11n TensorRT + All Skip | 15-20 | 50+ |

> **Actual Results on Jetson Orin Nano**: With all models (face, gesture, pose) running, we achieve **10-15 FPS** with TensorRT. With aggressive frame skipping (skip=4), we achieve **15-20 FPS**.

## Troubleshooting

### Low FPS

1. Check Jetson power mode: `nvpmodel -q`
2. Verify clocks: `jetson_clocks --show`
3. Check memory: `tegrastats`

### Out of Memory

1. Reduce input resolution
2. Use smaller model (YOLO11n vs YOLO11s)
3. Enable frame skipping
4. Increase swap space

### Model Loading Errors

1. Ensure ONNX file is in `models/` directory
2. Check TensorRT version compatibility
3. Try PyTorch fallback (slower but more compatible)
