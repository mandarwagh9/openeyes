# OpenEyes - Robot Vision System

```
 ___  ____  _____ _   _ _______   _______ ____   
/ _ \|  _ \| ____| \ | | ____\ \ / / ____/ ___|  
| | | | |_) |  _| |  \| |  _|  \ V /|  _| \___ \  
| |_| |  __/| |___| |\  | |___  | | | |___ ___) | 
 \___/|_|   |_____|_| \_|_____| |_| |_____|____/ 
```

**Version**: v0.4.0 | **Platform**: NVIDIA Jetson Orin Nano

---

## What is OpenEyes?

OpenEyes is a vision system for humanoid robots - the "eyes" that enable robots to perceive the physical world. It runs entirely on-device (Edge AI) with no cloud required.

### What It Does

| Capability | Description |
|:-----------|:------------|
| Object Detection | Recognize 80+ everyday objects |
| Depth Estimation | Measure distance to objects |
| Face Detection | Locate and track faces |
| Gesture Recognition | Understand hand signals |
| Pose Estimation | Detect body poses |
| Object Tracking | Track objects across frames |
| Person Following | Follow a person autonomously |
| ROS2 Integration | Publish vision data to robot |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes
pip install -r requirements.txt

# Run (basic)
python src/main.py

# Run with debug window
python src/main.py --debug
```

### First Time on Jetson?

```bash
# Enable max performance (run once)
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## Common Commands

### Basic
```bash
python src/main.py                      # Default run
python src/main.py --camera 0           # Specific camera
python src/main.py --debug              # Show video window
```

### Speed Optimization
```bash
# Maximum speed (30+ FPS)
python src/main.py --no-face --no-gesture --no-pose --no-depth --precision int8

# Balanced (good speed + features)
python src/main.py --no-depth --precision fp16
```

### ROS2 Integration
```bash
# Enable ROS2 publishing
python src/main.py --ros2

# With actions and QoS
python src/main.py --ros2 --ros2-actions --ros2-qos sensor
```

### Tracking & Following
```bash
# Enable person following
python src/main.py --follow
```

### Select Model
```bash
# List available models
python src/main.py --list-models

# Use specific model
python src/main.py --model yolo12n
python src/main.py --model rtmdet_nano
```

---

## Performance

| Configuration | Expected FPS |
|:--------------|:-------------|
| All models enabled | 10-15 FPS |
| No depth/face/gesture/pose | 25-30 FPS |
| INT8 + minimal models | 30-40 FPS |

### Optimize Further

```bash
# TensorRT INT8 (~2x faster)
python src/main.py --precision int8

# Batch inference
python src/main.py --batch-size 4

# Use DLA (Jetson)
python src/main.py --dla
```

### Jetson Scripts

```bash
sudo bash scripts/jetson_perf.sh      # Optimize performance
python3 scripts/jetson_helper.py       # System info
python3 scripts/jetson_helper.py --check  # Check status
```

---

## Output

### UDP/JSON Format
Vision results sent to `--host` (default: 127.0.0.1:5000)

### ROS2 Topics
| Topic | Type |
|:------|:-----|
| `/vision/detections` | JSON |
| `/vision/depth` | JSON |
| `/vision/faces` | JSON |
| `/vision/gestures` | JSON |
| `/vision/poses` | JSON |
| `/vision/cmd` | Subscribe |
| `/vision/status` | JSON |

### Robot Commands (from `/vision/cmd`)
- `forward` / `backward` / `stop`
- `left` / `right` / `follow`

---

## Hardware

| Component | Specification |
|:----------|:--------------|
| Platform | NVIDIA Jetson Orin Nano |
| Camera | CSI (IMX219) or USB |
| Power | 5V/4A |
| OS | Ubuntu 22.04 + JetPack |

---

## Troubleshooting

**No camera detected**
```bash
ls /dev/video*
# If empty, check camera connection and reboot
```

**Low FPS**
```bash
# Disable unused models
python src/main.py --no-face --no-gesture --no-pose --no-depth

# Lower resolution
python src/main.py --width 480 --height 360
```

**ROS2 not working**
```bash
# Check ROS2 installation
ros2 doctor

# Verify topics
ros2 topic list
```

---

## All Commands

See [COMMANDS.md](COMMANDS.md) for the complete command reference.

---

## Documentation

- [COMMANDS.md](COMMANDS.md) - Full command list
- [USER_GUIDE.md](USER_GUIDE.md) - Detailed usage
- [INSTALL.md](INSTALL.md) - Installation guide
- [OPTIMIZATION.md](OPTIMIZATION.md) - Performance tuning

---

## License

Apache 2.0 - See [LICENSE](LICENSE) file.

> **Note**: YOLO models use AGPL-3.0 license. RTMDet (Apache 2.0) available as alternative.
