# OpenEyes

**v0.4.1**

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

## What OpenEyes Sees

```
The robot looks at a room and understands:

- "There's a cup on the table, 40cm away"
- "A person is standing to my left"
- "They're waving at me - that's a greeting"
- "The person is sitting down - they might need help"
- "That's a stair - don't step there"
```

OpenEyes provides:

- **Object Detection** - What objects exist in the scene
- **Depth Estimation** - How far away is everything  
- **Face Detection** - Who's in the room
- **Gesture Recognition** - What commands are being given
- **Pose Estimation** - What positions are bodies in
- **Object Tracking** - Follow that object
- **Person Following** - Follow a person autonomously

---

## What It Enables

| Use Case | Why It Matters |
|:---------|:---------------|
| **Home Assistant** | Robots that navigate homes safely |
| **Elderly Care** | Detect falls, alert family |
| **Warehouse** | Autonomous navigation, object handling |
| **Research** | Open platform for vision experiments |

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
```

### First Time on Jetson?

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## Performance

| Configuration | FPS |
|:--------------|:----|
| All models | 10-15 |
| Minimal (no depth/face/gesture/pose) | 25-30 |
| Optimized (INT8 + minimal) | 30-40 |

---

## Common Commands

```bash
# Maximum speed
python src/main.py --no-face --no-gesture --no-pose --no-depth --precision int8

# With ROS2
python src/main.py --ros2

# Person following
python src/main.py --follow

# List models
python src/main.py --list-models
```

See [COMMANDS.md](COMMANDS.md) for all commands.

---

## Hardware

- **Platform**: NVIDIA Jetson Orin Nano
- **Camera**: CSI (IMX219) or USB webcam
- **OS**: Ubuntu 22.04 + JetPack

---

## The Journey

OpenEyes started with a simple question: *Why can't robots see like we do?*

We've come far. But there's more to do.

| Version | Milestone |
|:--------|:----------|
| v0.1.x | Core vision (detection, depth, face, gesture, pose) |
| v0.2.x | Tracking, performance, ROS2 |
| v0.3.x | Model selection, specialized models |
| v0.4.x | VLA, event camera |

We're just getting started.

---

## Contribute

OpenEyes is built by people like you. Developers, researchers, hobbyists, dreamers.

See [CONTRIBUTING.md](CONTRIBUTING.md) to join us.

---

## License

Apache 2.0 - See [LICENSE](LICENSE).

> The future of robotics is open. Let's build it together.
