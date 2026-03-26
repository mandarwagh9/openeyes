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

### Default Intervals

```python
skip_intervals = {
    'detector': 1,   # Process every frame
    'depth': 2,      # Process every 2nd frame
    'face': 2,       # Process every 2nd frame
    'gesture': 2,    # Process every 2nd frame
    'pose': 2        # Process every 2nd frame
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
    base_skip=2,           # Default skip interval
    motion_threshold=5000, # Motion detection threshold
    min_skip=1,            # Minimum skip (1 = every frame)
    max_skip=4             # Maximum skip
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
| YOLOv10n + No skipping | 7-10 | 25-35 |
| YOLO11n + Adaptive skip | 15-25 | 50-70 |
| YOLO11n + INT8 + Skip | 25-35 | 80-100 |

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
