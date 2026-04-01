# QUICKSTART.md - Quick Start Guide for OpenEyes

> **Version**: v1.0.0  
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
# One-command optimization (recommended)
sudo bash scripts/jetson_perf.sh

# Or manually:
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## Step 4: Verify Camera

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

## Step 5: Run the Vision System

```bash
# Run with debug/display output
python src/main.py --debug

# Run headless (no display)
python src/main.py

# Run with ROS2 publishing
python src/main.py --ros2

# Show version
python src/main.py --version
```

You should see:
- Camera feed window (with --debug)
- Object detection boxes (green)
- Face detection boxes (blue)
- Console output with FPS (15-25 FPS with all models)

---

## Step 6: Customize (Optional)

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

# Disable all extra models for maximum FPS
python src/main.py --no-face --no-gesture --no-pose --no-depth

# Enable max Jetson performance (recommended)
sudo nvpmodel -m 0 && sudo jetson_clocks
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
| All models enabled (default) | 10-12 FPS |
| Without face/gesture/pose | 18-22 FPS |
| Without all extras + Jetson max | 22-28 FPS |
| Object detection only | 40-60 FPS |
| INT8 (v0.8.0+) | 30-40 FPS |
| INT8 + DLA (v0.8.0+) | 40-50 FPS |
| YOLO11n TensorRT INT8 | 80-100 FPS |

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
