# QUICKSTART.md - Quick Start Guide for OpenEyes

> **Version**: v0.0.1  
> **Estimated Time**: 5 minutes

---

## Prerequisites

| Requirement | Check |
|:------------|:------|
| NVIDIA Jetson Orin Nano | ☐ |
| USB Webcam | ☐ |
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

## Step 3: Verify Camera

```bash
# Check camera is detected
ls /dev/video*

# Test with Python
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

---

## Step 4: Run the Vision System

```bash
# Run with default settings
python src/main.py
```

You should see:
- Camera feed window
- Object detection boxes
- Console output with FPS

---

## Step 5: Customize (Optional)

### Change Camera

```bash
# Use different camera
python src/main.py --camera 1
```

### Enable Debug Mode

```bash
python src/main.py --debug
```

### Change Output Target

```bash
# Send to different host
python src/main.py --host 192.168.1.100 --port 5000
```

---

## Common Issues

| Issue | Solution |
|:------|:---------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Camera not found | Check `ls /dev/video*` |
| Low FPS | Reduce resolution with `--width 640 --height 480` |

---

## Next Steps

- Read [INSTALL.md](INSTALL.md) for detailed installation
- Read [USER_GUIDE.md](USER_GUIDE.md) for usage guide
- Check [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md) for technical details
