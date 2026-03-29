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

## Frame Skipping

The system includes adaptive frame skipping that automatically adjusts based on motion:

### Default Intervals (v0.1.1)

```python
skip_intervals = {
    'detector': 1,   # Process every frame (most important)
    'depth': 8,       # Process every 8th frame (expensive)
    'face': 6,       # Process every 6th frame
    'gesture': 6,    # Process every 6th frame
    'pose': 6        # Process every 6th frame
}
```

### Adaptive Skipping

The `AdaptiveFrameSkipper` automatically adjusts skip interval based on motion:
- Low motion (static scene): skip more frames (up to 4)
- High motion (active scene): process every frame

### Manual Configuration

Edit `src/main.py` to customize:

```python
self._adaptive_skipper = AdaptiveFrameSkipper(
    base_skip=2,           # Default skip interval (v0.1.1: was 3)
    motion_threshold=5000, # Motion detection threshold
    min_skip=1,            # Minimum skip (1 = every frame) (v0.1.1: was 2)
    max_skip=4             # Maximum skip (v0.1.1: was 5)
)
```

## Disable Models for Speed

The fastest way to increase FPS is to disable models you don't need:

```bash
# Disable face detection (~+2 FPS)
python src/main.py --no-face

# Disable gesture recognition (~+2 FPS)
python src/main.py --no-gesture

# Disable pose estimation (~+2 FPS)
python src/main.py --no-pose

# Disable depth estimation (~+2 FPS, NEW in v0.1.1)
python src/main.py --no-depth

# Maximum speed - disable all extra models
python src/main.py --no-face --no-gesture --no-pose --no-depth
```

### Model Combinations

| Command | Expected FPS |
|:--------|:------------|
| All models (default) | ~10-12 |
| --no-face | ~12-14 |
| --no-face --no-gesture --no-pose | ~18-22 |
| --no-face --no-gesture --no-pose --no-depth | ~22-25 |
| + Jetson max performance | +20-30% |

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
| YOLO11n + Default (v0.1.1) | 10-12 | 25-35 |
| YOLO11n + --no-face/gesture/pose | 18-22 | 40-50 |
| YOLO11n + All disabled | 22-25 | 50-60 |
| YOLO11n + INT8 + All disabled | 30-40 | 80-100 |
| YOLO11n + Jetson max + INT8 | 40-50 | 100-120 |

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
