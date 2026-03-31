# FAQ

Frequently asked questions about OpenEyes.

---

## General

### What is OpenEyes?

OpenEyes is an open-source vision system for humanoid robots. It runs entirely on NVIDIA Jetson devices with full ROS2 integration.

### What can OpenEyes do?

- Object Detection (YOLO)
- Depth Estimation (MiDaS)
- Face Detection (MediaPipe)
- Gesture Recognition (MediaPipe)
- Pose Estimation (MediaPipe)
- Person Following
- Visual SLAM
- VLA (Vision-Language-Action) models

### What hardware do I need?

- NVIDIA Jetson Orin Nano (4GB or 8GB)
- USB webcam or CSI camera
- Ubuntu 22.04 + JetPack

---

## Installation

### How do I install OpenEyes?

```bash
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes
pip install -r requirements.txt
python src/main.py --debug
```

### Do I need ROS2?

No. ROS2 is optional. OpenEyes works standalone with UDP output, or with ROS2 for full robot integration.

---

## Performance

### Why is my FPS low?

1. Enable Jetson max performance:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

2. Disable unused models:
   ```bash
   python src/main.py --no-face --no-gesture --no-pose --no-depth
   ```

3. Use smaller resolution:
   ```bash
   python src/main.py --width 640 --height 480
   ```

### What FPS can I expect?

| Configuration | FPS |
|:--------------|:----|
| All models | 10-15 |
| Minimal | 25-30 |
| Optimized (INT8) | 30-40 |

---

## Troubleshooting

### Camera not detected

```bash
ls -la /dev/video*
sudo usermod -a -G video $USER
```

### Import errors

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Thermal throttling

Use better cooling (fan/heatsink) or reduce power mode:
```bash
sudo nvpmodel -m 1
```

---

## ROS2

### How do I enable ROS2?

```bash
python src/main.py --ros2
```

### What topics are available?

- `/vision/detections` - Object detections
- `/vision/depth` - Depth map
- `/vision/faces` - Face detections
- `/vision/gestures` - Gestures
- `/vision/poses` - Body poses
- `/vision/status` - System status

---

## Contributing

### How do I contribute?

1. Fork the repo
2. Create a feature branch
3. Make changes following our coding standards
4. Run tests and linting
5. Submit a pull request

See [Contributing Guide](../development/contributing.md) for details.

---

## More Questions?

- [GitHub Issues](https://github.com/mandarwagh9/openeyes/issues)
- [GitHub Discussions](https://github.com/mandarwagh9/openeyes/discussions)