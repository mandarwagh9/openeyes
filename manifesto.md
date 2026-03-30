# OpenEyes Manifesto

## v0.4.4

## We Give Robots Vision

A humanoid robot needs to see the world like a human does. Not just pixels - but understanding. Distance. Intent. Action.

OpenEyes is an open-source vision system built for humanoid robots. It runs entirely on NVIDIA Jetson - no cloud, no lag, no dependencies.

---

## The Problem

Every day, millions of robots are deployed to help humans. But most of them are blind. Or dependent on cloud services that fail. Or so expensive only big companies can afford them.

We wanted to change that.

---

## Our Philosophy

| Principle | What It Means |
|:----------|:--------------|
| **Edge First** | All processing happens on the robot. No round-trip to cloud. No latency. |
| **Privacy First** | No data leaves the device. What the robot sees, stays with the robot. |
| **Real-time** | 30 FPS isn't a dream - it's what we optimize for. |
| **Open** | Built by the community, for the community. Anyone can use. Anyone can contribute. |

---

## Core Technical Vision

### AI + Embedded Systems = Intelligence Running Directly on the Robot

Instead of AI running only in large cloud servers, **AI models are embedded inside the robot** that interacts with the real world.

This means the robot itself can:

- perceive its environment
- interpret data
- make decisions
- trigger actions

without relying on a remote server.

### Why This Matters

Traditional AI architecture for robots looks like this:

```
Sensors → Internet → Cloud AI → Decision → Robot
```

**Problems:**

- latency (slow response)
- bandwidth costs
- privacy concerns
- dependency on internet

OpenEyes changes the architecture to:

```
Camera → On-device AI → Decision → Actuator
```

Now the robot **thinks locally**.

This allows:

- real-time decision making
- offline operation
- improved privacy
- lower network load

---

## What OpenEyes Sees

```
The robot looks at a room and understands:

- "There's a cup on the table, 40cm away"
- "A person is standing to my left"
- "They're waving at me - that's a greeting"
- "The person is sitting down - they might need help"
- "That's a stair - don't step there"
```

### Vision Modalities

| Modality | Technology | Purpose |
|:---------|:-----------|:--------|
| **Object Detection** | YOLO11n | What objects exist in the scene |
| **Depth Estimation** | MiDaS | How far away is everything |
| **Face Detection** | MediaPipe | Who's in the room |
| **Gesture Recognition** | MediaPipe Hands | What commands are being given |
| **Pose Estimation** | MediaPipe Pose | What positions are bodies in |
| **Object Tracking** | Custom tracker | Follow that object |
| **Person Following** | Bbox height ratio | Follow a person autonomously |

---

## Technical Architecture

### Layer 1: Hardware (Physical)

- **Platform**: NVIDIA Jetson Orin Nano
- **Camera**: CSI (IMX219) or USB webcam at 1920x1080
- **AI Accelerator**: Integrated GPU

### Layer 2: Edge Processing

All AI inference happens locally on the Jetson:

- YOLO11n for object detection
- MiDaS for depth estimation
- MediaPipe for face/gesture/pose

### Layer 3: ROS2 Integration

The robot communicates via ROS2:

| Topic | Type | Description |
|:------|:-----|:------------|
| `/vision/detections` | JSON | Object detections |
| `/vision/depth` | JSON | Depth map data |
| `/vision/faces` | JSON | Face detections |
| `/vision/gestures` | JSON | Gesture recognitions |
| `/vision/pose` | JSON | Body pose landmarks |
| `/vision/status` | JSON | System status with timestamp |
| `/vision/image/debug` | JSON | Debug image |

**Commands:**

| Command | Action |
|:--------|:-------|
| `forward` | Move forward |
| `backward` | Move backward |
| `stop` | Stop all motion |
| `left` | Turn left |
| `right` | Turn right |
| `follow` | Follow detected person |

---

## Key Technical Decisions

### Bounding Box Height Ratio for Distance

Instead of relying on depth maps or tracking continuity, OpenEyes uses **bounding box height ratio** (% of frame) to determine distance:

| Zone | Height Ratio | Action |
|:-----|:-------------|:-------|
| Forward | < 60% | Move forward (person is far) |
| Stop | 60-95% | Stay still (person at ideal distance) |
| Backward | > 95% | Move backward (person too close) |

This is more reliable for monocular cameras where depth estimates can be noisy.

### Gesture-Based Owner Selection

To designate who the robot should follow:

1. Show **open_palm** gesture to the camera
2. Robot recognizes the gesture
3. That person becomes the "owner" - the robot will follow them

This provides intuitive control without needing external interfaces.

### Image Resolution for MediaPipe

MediaPipe models work better with lower resolution:

- Input: 1920x1080 (camera)
- Processed: 640x480 (MediaPipe)
- This improves detection confidence significantly

### Depth Normalization

MiDaS outputs normalized depth (0-1 where 1.0 = closest). For display and processing:

- Normalize to 0-1 meters
- 1.0 = closest point
- 0.0 = farthest point

---

## Performance

| Configuration | FPS |
|:--------------|:----|
| All models | 10-15 |
| Minimal (no depth/face/gesture/pose) | 25-30 |
| Optimized (INT8 + minimal) | 30-40 |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/mandarwagh9/openeyes.git
cd openeyes
pip install -r requirements.txt

# Run
python src/main.py

# With debug window
python src/main.py --debug

# With person following
python src/main.py --follow

# With ROS2
python src/main.py --ros2
```

---

## The Journey

OpenEyes started with a simple question: *Why can't robots see like we do?*

We've come far. But there's more to do.

| Version | Milestone |
|:--------|:----------|
| v0.1.x | Core vision (detection, depth, face, gesture, pose) |
| v0.2.x | Tracking, performance, ROS2 |
| v0.3.x | Model selection, specialized models |
| v0.4.x | Person following, gesture owner selection |

---

## Contribute

OpenEyes is built by people like you. Developers, researchers, hobbyists, dreamers.

See [CONTRIBUTING.md](CONTRIBUTING.md) to join us.

---

## License

Apache 2.0 - See [LICENSE](LICENSE).

> The future of robotics is open. Let's build it together.
