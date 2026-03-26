# OpenEyes - Robot Vision System

<p align="center">
  <pre style="font-family: monospace; font-size: 11px; font-weight: bold; color: #1a1a2e; background: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center;">
 ___  ____  _____ _   _ _______   _______ ____   
/ _ \|  _ \| ____| \ | | ____\ \ / / ____/ ___|  
| | | | |_) |  _| |  \| |  _|  \ V /|  _| \___ \  
| |_| |  __/| |___| |\  | |___  | | | |___ ___) | 
 \___/|_|   |_____|_| \_|_____| |_| |_____|____/ 
      OpenEyes
 Vision System for Humanoid Robots
  </pre>
</p>

<p align="center">
  <a href="#about">
    <img src="https://img.shields.io/badge/Version-v0.0.3-1a1a2e?style=for-the-badge&logo=version-control&logoColor=white" alt="Version" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-1a1a2e?style=for-the-badge" alt="License" />
  </a>
  <a href="#tech-stack">
    <img src="https://img.shields.io/badge/Tech-NVIDIA_Jetson-1a1a2e?style=for-the-badge&logo=nvidia&logoColor=white" alt="Tech Stack" />
  </a>
  <a href="CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/Welcome-Contributions-1a1a2e?style=for-the-badge" alt="Contributing" />
  </a>
</p>

---

## About

**OpenEyes** is a vision system for humanoid robots - the "eyes" that enable robots to perceive and understand the physical world. Built on NVIDIA Jetson Orin Nano, it provides real-time AI-powered computer vision capabilities running entirely on-device (Edge AI).

### Key Features

- 👁️ **Object Detection** - Recognize everyday objects in real-time (YOLO11n)
- 📊 **Depth Estimation** - Understand 3D environment from 2D camera (MiDaS)
- 👤 **Face Detection** - Locate and track faces
- 🖐️ **Gesture Recognition** - Understand human hand signals
- 🦵 **Pose Estimation** - Detect human body poses
- ⚡ **Real-time Performance** - 15+ FPS with adaptive frame skipping
- 🔒 **Privacy-first** - All processing done locally, no cloud required
- 🤖 **ROS2 Ready** - Vision data publishing for robot control

---

## Why OpenEyes?

A humanoid robot needs vision like humans need eyes. OpenEyes provides:

| Capability | Human Equivalent | Use Case |
|:-----------|:----------------|:---------|
| Object Detection | "That's a cup" | Find objects to grasp |
| Depth Estimation | "The table is 50cm away" | Navigation & avoidance |
| Face Detection | "Someone is there" | Presence detection |
| Gesture Recognition | "Stop sign" | Understand commands |
| Pose Estimation | "Person sitting" | Activity recognition |

---

## Hardware

| Component | Specification |
|:----------|:-------------|
| **Platform** | NVIDIA Jetson Orin Nano (4GB/8GB) |
| **Camera** | CSI Camera (IMX219) or USB Webcam |
| **OS** | JetPack (Ubuntu 22.04) |
| **Power** | 5V/4A barrel jack |
| **Storage** | MicroSD 64GB+ |

### Performance Targets

| Metric | Target |
|:-------|:-------|
| FPS | 15-25 FPS (all models), 50+ FPS (object only) |
| Latency | <50ms |
| Detection Range | 0.5m - 5m |
| Model Size | <10MB |

---

## Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,tensorflow,pytorch,opencv,docker" />
</p>

| Layer | Technology |
|:------|:-----------|
| **AI Framework** | PyTorch, ONNX Runtime |
| **Vision** | OpenCV, CUDA, GStreamer |
| **Models** | YOLOv10, MediaPipe, MiDaS |
| **Deployment** | TensorRT optimization |
| **Communication** | JSON, UDP, ROS2-ready |

> **Note:** YOLOv10 uses AGPL-3.0 license. For commercial use, consider RTMDet (Apache 2.0).

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes

# Install dependencies
pip install -r requirements.txt

# Enable max performance (Jetson)
sudo nvpmodel -m 0
sudo jetson_clocks

# Run the vision system
python src/main.py --debug
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

---

## Project Structure

```
openeyes/
├── src/
│   ├── camera/           # Camera input handling (CSI + USB)
│   ├── models/           # AI model wrappers
│   │   ├── object_detector.py    # YOLOv10
│   │   ├── depth_estimator.py   # MiDaS
│   │   ├── face_detector.py     # MediaPipe
│   │   ├── gesture_recognizer.py # MediaPipe
│   │   └── pose_estimator.py   # MediaPipe
│   ├── output/           # Output handlers
│   └── utils/            # Utilities
├── models/               # AI model weights
├── docs/                # Documentation
├── tests/                # Test suites
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## Documentation

### Getting Started
- [QUICKSTART.md](QUICKSTART.md) - 5-minute setup guide
- [INSTALL.md](INSTALL.md) - Detailed installation
- [USER_GUIDE.md](USER_GUIDE.md) - How to use the system

### Technical
- [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md) - Full technical specification
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
- [HARDWARE.md](docs/HARDWARE.md) - Hardware specifications
- [API_SPEC.md](docs/API_SPEC.md) - API documentation
- [HOW_HUMANOID_ROBOTS_SEE.md](docs/HOW_HUMANOID_ROBOTS_SEE.md) - How robot vision works

### Developer
- [AGENTS.md](AGENTS.md) - Developer guidelines
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [ROADMAP.md](ROADMAP.md) - Project roadmap
- [CHANGELOG.md](CHANGELOG.md) - Version history

---

## Capabilities Roadmap

| Phase | Capabilities | Status |
|:------|:------------|:-------|
| v0.0.1 | Object Detection | Complete |
| v0.0.2 | Depth, Face, Gesture, Pose | Complete |
| v1.0.0 | Full Integration | In Progress |

---

## Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) before getting started.

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

> **License Note:** YOLOv10 model used in this project is licensed under AGPL-3.0. The inference code itself does not impose additional restrictions, but if you distribute modified model weights, you may need to comply with AGPL-3.0 requirements.

---

## References

- [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/)
- [YOLOv10 Documentation](https://docs.ultralytics.com/)
- [MediaPipe](https://mediapipe.dev/)
- [MiDaS Depth Estimation](https://github.com/isl-org/MiDaS)
- [TensorRT](https://developer.nvidia.com/tensorrt)

---

<p align="center">
  <sub>Built with ⚡ for the future of humanoid robotics</sub>
</p>
