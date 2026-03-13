# PROJECT0 - Robot Vision System

<p align="center">
  <pre style="font-family: monospace; font-size: 14px; font-weight: bold; color: #1a1a2e; background: #f0f0f0; padding: 20px; border-radius: 8px;">
                  _           _    ___  
 _ __  _ __ ___ (_) ___  ___| |_ / _ \ 
| '_ \| '__/ _ \| |/ _ \/ __| __| | | |
| |_) | | | (_) | |  __/ (__| |_| |_| |
| .__/|_|  \___// |\___|\___|\__|\___/ 
|_|           |__/                     
  
                PROJECT0
              Vision System for Humanoid Robots
  </pre>
</p>

<p align="center">
  <a href="#about">
    <img src="https://img.shields.io/badge/Version-v0.0.1-1a1a2e?style=for-the-badge&logo=version-control&logoColor=white" alt="Version" />
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

**PROJECT0** is a vision system for humanoid robots - the "eyes" that enable robots to perceive and understand the physical world. Built on NVIDIA Jetson Orin Nano, it provides real-time AI-powered computer vision capabilities running entirely on-device (Edge AI).

### Key Features

- 👁️ **Object Detection** - Recognize everyday objects in real-time
- 📊 **Depth Estimation** - Understand 3D environment from 2D camera
- 👤 **Face Recognition** - Identify and track people
- 🖐️ **Gesture Recognition** - Understand human hand signals
- 🦵 **Pose Estimation** - Detect human body poses
- ⚡ **Real-time Performance** - 20-30 FPS on embedded hardware
- 🔒 **Privacy-first** - All processing done locally, no cloud required

---

## Why PROJECT0?

A humanoid robot needs vision like humans need eyes. PROJECT0 provides:

| Capability | Human Equivalent | Use Case |
|:-----------|:----------------|:---------|
| Object Detection | "That's a cup" | Find objects to grasp |
| Depth Estimation | "The table is 50cm away" | Navigation & avoidance |
| Face Recognition | "That's my owner" | Personal identification |
| Gesture Recognition | "Stop sign" | Understand commands |
| Pose Estimation | "Person sitting" | Activity recognition |

---

## Hardware

| Component | Specification |
|:----------|:-------------|
| **Platform** | NVIDIA Jetson Orin Nano (4GB/8GB) |
| **Camera** | USB Webcam 1080p |
| **OS** | JetPack (Ubuntu 22.04) |
| **Power** | 5V/4A barrel jack |
| **Storage** | MicroSD 64GB+ |

### Performance Targets

| Metric | Target |
|:-------|:-------|
| FPS | 20-30 FPS |
| Latency | <50ms |
| Detection Range | 0.5m - 5m |
| Model Size | <50MB |

---

## Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,tensorflow,pytorch,opencv,docker" />
</p>

| Layer | Technology |
|:------|:-----------|
| **AI Framework** | TensorFlow Lite, PyTorch, ONNX |
| **Vision** | OpenCV, CUDA |
| **Models** | YOLOv8, MediaPipe, MiDaS |
| **Deployment** | TensorRT optimization |
| **Communication** | JSON, UDP, ROS2-ready |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mandarwagh9/project0.git
cd project0

# Install dependencies
pip install -r requirements.txt

# Run the vision system
python src/main.py
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

---

## Project Structure

```
project0/
├── src/
│   ├── camera/           # Camera input handling
│   ├── models/           # AI model wrappers
│   ├── inference/        # TensorRT optimization
│   └── output/          # Output handlers
├── models/              # AI model weights
├── docs/                # Documentation
│   ├── TECHNICAL_SPEC.md
│   ├── ARCHITECTURE.md
│   ├── HARDWARE.md
│   └── API_SPEC.md
├── tests/               # Test suites
├── requirements.txt    # Python dependencies
└── README.md           # This file
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

### Developer
- [AGENTS.md](AGENTS.md) - Developer guidelines
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [ROADMAP.md](ROADMAP.md) - Project roadmap
- [CHANGELOG.md](CHANGELOG.md) - Version history

---

## Capabilities Roadmap

| Phase | Capabilities | Status |
|:------|:------------|:-------|
| v0.0.1 | Object Detection | Planned |
| v0.0.2 | Depth Estimation | Planned |
| v0.0.3 | Face Recognition | Planned |
| v0.0.4 | Gesture Recognition | Planned |
| v0.0.5 | Pose Estimation | Planned |
| v1.0.0 | Full Integration | Planned |

---

## Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) before getting started.

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## References

- [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [MediaPipe](https://mediapipe.dev/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)

---

<p align="center">
  <sub>Built with ⚡ for the future of humanoid robotics</sub>
</p>
