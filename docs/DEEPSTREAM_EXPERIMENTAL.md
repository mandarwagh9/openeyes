# DeepStream Status - Experimental

## Current Issue

YOLOv10/YOLO11n output format `[1, 300, 6]` (center format with class predictions) is not compatible with DeepStream's built-in parsers.

```
Could not find output coverage layer for parsing objects
```

This requires a custom C++ parser library to be compiled for the specific YOLO version's output format.

## Working Alternatives

### OpenCV Pipeline (Works Now - ~12 FPS)
```bash
# Standard pipeline
python -m src.main --no-depth --debug

# Faster with lower resolution  
python -m src.main --no-depth --low-res
```

### Expected DeepStream Performance (40-70 FPS) when fixed

## What's Implemented

1. DeepStream pipeline code in `src/deepstream/pipeline.py`
2. Config files in `deepstream/config_*.txt`
3. CLI flag `--deepstream` 

## To Fix - Required Steps

1. Clone DeepStream-Yolo repository for parser source:
```bash
git clone https://github.com/marcoslucianops/DeepStream-Yolo
```

2. Build custom parser for your YOLO version

3. Or use NVIDIA's pre-trained models (TrafficCamNet, etc.)

## Quick Test
```bash
# Current working pipeline  
python -m src.main --no-depth
```

## Files Modified This Session

- `src/deepstream/pipeline.py` - Full pipeline implementation
- `deepstream/config_yolov10n.txt` - Model config
- `src/main.py` - CLI integration
- `models/yolov10n.engine` - Rebuilt TensorRT engine