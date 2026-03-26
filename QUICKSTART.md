# QUICKSTART.md - Quick Start Guide for OpenEyes

> **Version**: v0.0.3  
> **Estimated Time**: 5 minutes

---

## Prerequisites

| Requirement | Check |
|:------------|:------|
| NVIDIA Jetson Orin Nano | ☐ |
| CSI Camera (IMX219) or USB Webcam | ☐ |
| Ubuntu 22.04 installed | ☐ |
| Internet connection | ☐ |

---

## Step 1: Clone the Project

```bash
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes
```

---

## Step 2: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## Step 3: Enable Maximum Performance (Recommended)

For best performance on Jetson Orin Nano:

```bash
# Enable 15W power mode
sudo nvpmodel -m 0

# Lock CPU/GPU clocks to maximum
sudo jetson_clocks
```

---

## Step 4: TensorRT Engine (Optional - Already Included)

The project includes a pre-built TensorRT engine for maximum GPU performance:

```bash
# The engine is already at: models/yolo11n.engine (8.2 MB)
# To verify it exists:
ls -la models/yolo11n.engine

# To export a new engine (e.g., after retraining):
python3 scripts/export_tensorrt.py --model models/yolo11n.pt
```

> **Why TensorRT?** Provides ~51 FPS YOLO inference vs ~5 FPS on CPU - a 10x speedup!

---

## Step 5: Verify Camera

### For CSI Camera (IMX219)
```bash
# Check camera device
ls -la /dev/video*

# Test with GStreamer
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! fakesink
```

### For USB Webcam
```bash
# Check camera is detected
ls /dev/video*

# Test with Python
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

---

## Step 6: Run the Vision System

```bash
# Run with debug/display output
python src/main.py --camera 0 --debug

# Or run headless (no display)
python src/main.py --camera 0
```

You should see:
- Camera feed window (with --debug)
- Object detection boxes (green)
- Face detection boxes (blue)
- Console output with FPS (10-20 FPS with all models)

---

## Step 7: Customize (Optional)

### Change Camera

```bash
# Use different CSI sensor (CAM1)
python src/main.py --camera 1

# Or USB camera
python src/main.py --camera 0
```

### Enable Debug Mode

```bash
python src/main.py --debug
```

### Performance Tuning

```bash
# Disable parallel processing (more stable, slower)
python src/main.py --no-parallel

# Run pose estimation every 3 frames (faster)
python src/main.py --pose-every 3
```

### Change Output Target

```bash
# Send to different host
python src/main.py --host 192.168.1.100 --port 5000
```

---

## Performance Info

| Configuration | Expected FPS |
|:--------------|:------------|
| All models enabled | 15-25 FPS |
| Object detection only | 50-70 FPS |
| YOLO11n TensorRT INT8 | 80-100 FPS |
| Without display | Slightly higher |

> **Tip**: See [OPTIMIZATION.md](OPTIMIZATION.md) for more performance tuning options.

---

## Common Issues

| Issue | Solution |
|:------|:---------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Camera not found | Check `ls /dev/video*` or restart nvargus-daemon |
| Low FPS | Use `--no-parallel` or reduce resolution |
| Display not showing | Ensure DISPLAY=:0 is set (auto-detected) |
| GTK errors | These are harmless warnings, ignore them |

---

## Next Steps

- Read [INSTALL.md](INSTALL.md) for detailed installation
- Read [USER_GUIDE.md](USER_GUIDE.md) for usage guide
- Check [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md) for technical details
