# Performance Optimization Guide

> **Version**: v2.5.0 | **Last Updated**: 2026-04-10

---

## Quick Start

```bash
# One-command optimization (recommended)
sudo bash scripts/jetson_perf.sh

# Run with turbo mode for max FPS
python -m src.main --camera 0 --turbo --debug
```

---

## Performance Overview

| Configuration | FPS (Orin Nano) | Latency | Power |
|:--------------|:----------------|:--------|:------|
| Detection only (TensorRT FP16) | 35-40 | ~28ms | 5-8W |
| Full pipeline (default) | 4-6 | ~200ms | 10-15W |
| Full pipeline + turbo | 8-12 | ~100ms | 10-15W |
| Minimal (--no-face --no-gesture --no-pose) | 15-20 | ~60ms | 8-12W |
| World model planning (LeWM) | 100-200 Hz | <10ms | 3-5W |

---

## 1. Jetson System Optimization

### MAXN SUPER Mode + jetson_clocks

```bash
# One-command optimization (recommended)
sudo bash scripts/jetson_perf.sh
```

This script:
- Sets MAXN SUPER power mode (GPU 1020MHz, CPU 1.7GHz)
- Locks all clocks with `jetson_clocks` (prevents DVFS dips)
- Sets CPU governor to performance
- Optimizes memory (swappiness=10, overcommit=1)
- Disables unnecessary services (cups, bluetooth, ModemManager)

**Expected gain: +50-80% FPS**

### Manual Optimization

```bash
# Power mode
sudo nvpmodel -m 2  # MAXN SUPER (or -m 0 for MAX 15W)

# Lock clocks
sudo jetson_clocks

# Check status
tegrastats  # Monitor thermals and performance
```

---

## 2. TensorRT Engine Optimization

### Rebuild with SOTA Flags

```bash
# FP16 engine with --best and --useCudaGraph
python scripts/export_tensorrt_optimized.py --model models/yolo11n.pt

# INT8 engine with calibration (1.6-1.9x faster than FP16)
python scripts/export_tensorrt_optimized.py --model models/yolo11n.pt --int8 --calib-dir /path/to/images
```

### Optimization Flags

| Flag | Effect | Speedup |
|------|--------|---------|
| `--best` | Exhaustive tactic search | +5-15% |
| `--useCudaGraph` | CUDA graph capture (0.5ms → 0.02ms enqueue) | +10-20% |
| `--fp16` | Half-precision inference | +2x vs FP32 |
| `--int8` | INT8 quantization | +1.6-1.9x vs FP16 |
| `--noDataTransfers` | Skip host-device transfers | +5-10% |
| `--useSpinWait` | Reduce latency variance | Lower p99 |

---

## 3. Turbo Mode

```bash
python -m src.main --turbo
```

Turbo mode uses aggressive frame skipping:

| Model | Default Skip | Turbo Skip |
|-------|-------------|------------|
| Detection | 1 (every frame) | 1 (every frame) |
| Depth | 8 | 16 |
| Face | 6 | 12 |
| Gesture | 6 | 12 |
| Pose | 6 | 12 |

**Expected gain: +50-100% FPS**

---

## 4. MediaPipe Optimization

### Applied Optimizations (Default)

| Setting | Before | After | Impact |
|---------|--------|-------|--------|
| Face max_faces | 3 | 1 | -60% face detection time |
| Face confidence | 0.3 | 0.5 | Fewer false positives |
| Gesture model_complexity | 1 | 0 | -50% gesture time |
| Gesture max_hands | 2 | 1 | -50% gesture time |
| Pose model_complexity | 1 | 0 | -40% pose time |

These are applied automatically — no flags needed.

---

## 5. GStreamer Pipeline Optimization

### Optimized Pipeline

The camera pipeline now:
- Captures at 1280x720 (saves NVMM memory vs 1920x1080)
- Uses hardware scaling via nvvidconv (VIC engine, zero CPU)
- `sync=false drop=true max-buffers=2` for lowest latency

**Fixes**: NvMapMemAllocInternalTagged OOM errors on Orin Nano

---

## 6. Model Selection

### Detection Models (Orin Nano, TensorRT FP16)

| Model | Params | GFLOPs | FPS | mAP |
|-------|--------|--------|-----|-----|
| YOLO26n | 2.57M | 6.1 | 35-40 | 40.9% |
| YOLO11n | 2.6M | 6.5 | 30-35 | 39.5% |
| YOLO12n | 2.6M | 6.5 | 30-35 | 40.0% |
| YOLO11s | 9.5M | 20.7 | 15-20 | 45.5% |

### Depth Models (Orin Nano)

| Model | Params | FPS | Quality |
|-------|--------|-----|---------|
| MiDaS Small | 5M | 15-20 | Good |
| Depth Anything V3 Small | 25M | 10-15 | **Best** |
| Depth Anything V3 Base | 98M | 5-8 | SOTA |

---

## 7. Disable Models for Speed

```bash
# Disable face detection
python -m src.main --no-face

# Disable gesture recognition
python -m src.main --no-gesture

# Disable pose estimation
python -m src.main --no-pose

# Disable depth estimation
python -m src.main --no-depth

# Maximum speed
python -m src.main --no-face --no-gesture --no-pose --no-depth
```

### Model Combinations

| Command | Expected FPS |
|:--------|:------------|
| All models (default) | 4-6 |
| --no-face | 5-7 |
| --no-face --no-gesture --no-pose | 8-12 |
| --no-face --no-gesture --no-pose --no-depth | 15-20 |
| + turbo mode | +50-100% |
| + jetson_perf.sh | +50-80% |

---

## 8. World Model Performance

### LeWorldModel (15M params)

| Metric | Value |
|--------|-------|
| Encoding latency | 1-2ms |
| Prediction latency | 0.5ms |
| Planning (100 samples) | 3-5ms |
| Total loop | 5-10ms |
| Control rate | 100-200 Hz |
| Memory | <100MB |
| Power | 3-5W |

### V-JEPA 2 (80M params)

| Frames | FPS (TensorRT FP16) | Memory |
|--------|---------------------|--------|
| 7 | 20-30 | ~710MB |
| 16 | 10-20 | ~710MB |
| 32 | 3-5 | ~710MB |

---

## 9. Input Resolution

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

---

## 10. Benchmarking

```bash
# All models
python -m benchmarks.run_benchmarks --all

# Specific model
python -m benchmarks.run_benchmarks --model yolo11n

# Generate report
python -m benchmarks.run_benchmarks --report
```

---

## 11. Memory Management

### System Memory

```bash
# Increase swap space for model loading
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### GPU Memory

```bash
# Monitor GPU memory
tegrastats

# Clear cache (if needed)
python -c "import torch; torch.cuda.empty_cache()"
```

---

## 12. Troubleshooting

### Low FPS

1. Run `sudo bash scripts/jetson_perf.sh`
2. Use `--turbo` mode
3. Disable unused models: `--no-face --no-gesture --no-pose`
4. Check thermals: `watch -n 1 tegrastats`

### Out of Memory (NvMapMemAllocInternalTagged error 12)

1. GStreamer pipeline now captures at 1280x720 (fixed)
2. Reduce input resolution in config.yaml
3. Use smaller models
4. Increase swap space

### Model Loading Errors

1. Ensure model file is in `models/` directory
2. Check TensorRT version compatibility
3. Try PyTorch fallback (slower but more compatible)
