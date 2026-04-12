# ROS2 Installation Guide

> **Note:** ROS2 is **optional** for OpenEyes. The vision system works fully without ROS2.

## Prerequisites

- Ubuntu 22.04 (Jetson JetPack)
- Root/sudo access

## Installation Steps

### Step 1: Set Locale

```bash
sudo locale-gen en_US.UTF-8
```

### Step 2: Add ROS2 Repository

```bash
# Create directory for keys
sudo mkdir -p /etc/apt/keyrings

# Download ROS keyring
sudo wget https://packages.ros.org/ros.keyring -O /etc/apt/keyrings/ros.gpg --no-check-certificate

# Add ROS2 repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros.gpg] http://packages.ros.org/ros2/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/ros2.list
```

### Step 3: Install ROS2 Packages

```bash
sudo apt update
sudo apt install -y ros-humble-vision-msgs ros-humble-cv-bridge ros-humble-image-transport ros-humble-image-proc
```

### Step 4: Verify Installation

```bash
# Check packages
dpkg -l | grep ros-humble

# Test cv_bridge
python3 -c "from cv_bridge import CvBridge; print('cv_bridge OK')"
```

## Alternative: Docker Installation

If the repository approach fails, use Docker:

```bash
# Pull ROS2 Humble container
docker pull ros:humble-ros-base-jammy

# Run with GPU access
docker run --runtime nvidia -it ros:humble-ros-base-jammy bash
```

## Required Packages

| Package | Purpose |
|---------|---------|
| `ros-humble-vision-msgs` | Vision message types (Detection2D, etc.) |
| `ros-humble-cv-bridge` | Convert between ROS Image and OpenCV |
| `ros-humble-image-transport` | Efficient image transport |
| `ros-humble-image-proc` | Image processing (rectification, etc.) |

## Uninstall

```bash
sudo apt remove ros-humble-*
sudo rm /etc/apt/sources.list.d/ros2.list
sudo rm /etc/apt/keyrings/ros.gpg
```

## Without ROS2

OpenEyes works fully without ROS2. Use the Python API:

```python
from src.main import VisionSystem
from src.camera.types import VisionResult

# Run vision system
system = VisionSystem(config)
system.start()
```

Output is sent via UDP in JSON format by default.
