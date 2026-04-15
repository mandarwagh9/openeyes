# OpenEyes Manifesto

**v2.0.0**

---

## The Problem

cRobots are everywhere, but most are blind. The ones that see depend on cloud services that lag, fail, and spy. The alternatives are locked behind proprietary hardware costing thousands. We wanted something different.

We asked: *Why can't a robot see like a human - locally, instantly, privately?*

---

## The Vision

A world where every robot sees. Not just detects - understands. Where a humanoid can navigate a room, recognize intent, and act without asking permission from a server. Where open-source isn't a niche but the default.

---

## Core Principles

| Principle | What It Means |
|:----------|:--------------|
| **Edge Native** | All AI runs on the robot. No round-trips. No excuses. |
| **Privacy by Architecture** | What the robot sees stays on the robot. Forever. |
| **30 FPS or Bust** | Real-time isn't aspirational - it's the baseline. |
| **Open Stack** | No proprietary blobs. No telemetry. No vendor lock-in. |
| **Hardware Agnostic** | Your robot, your choice of Jetson. |
| **Community First** | Built by users, for users. Not by corporations, for shareholders. |

---

## What OpenEyes Is Not

- **Not a cloud service** - Everything runs locally
- **Not a product** - It's a foundation you build on
- **Not single-model** - YOLO + MediaPipe + Depth Anything + world models work together
- **Not closed** - No telemetry, no proprietary layers, no "contact sales"
- **Not slow** - We optimize for real-time, not batch processing

---

## The Stance on Tradeoffs

We choose **speed over accuracy** when it matters. A robot needs to react in milliseconds, not seconds. We'd rather detect something at 30fps with 95% confidence than pause for 3fps at 99%.

We choose **simplicity over features**. Every new capability must justify its presence. Complexity is a tax on maintainability.

We choose **local over compatible**. If a feature requires cloud or telemetry, it doesn't belong.

---

## Who This Is For

You. The person building a robot in a garage. The researcher needing real-time vision. The company that refuses to depend on Big Tech's servers. The one who believes robotics should be open by default.

If you've ever been told "just use our API" - this is for you.

---

## What OpenEyes Does (2026)

```
Camera → On-device AI → Decision → Action
```

- **YOLO11n** - Object detection at edge
- **MediaPipe Holistic** - Unified face + pose + hands (30% faster)
- **Selfie Segmentation** - Obstacle avoidance
- **Depth Anything V3** - Real-time depth estimation
- **Motor Control** - Direct CmdVel integration
- **Pipeline Optimizer** - Targeting 30-50 FPS
- **ROS2** - Native vision_msgs, not JSON wrappers

---

## The Future

We've upgraded from 10-15 FPS to targeting 30-50 FPS. We've unified the pipeline. We've added motor control. But we're not done.

The goal: A robot that sees, understands, and acts - without ever touching the internet.

> Robotics shouldn't require permission. Build freely.
