# Demo Commands for Launch Video

These are the core commands to showcase OpenEyes capabilities in a launch video.

---

## 1. Basic Vision Pipeline

```bash
python -m src.main --camera 0 --debug
```

**What it shows:** Detection, tracking, depth, face, gesture, pose — all running at once.

---

## 2. Person Following

```bash
python -m src.main --camera 0 --follow --debug
```

**What it shows:** Robot autonomously tracks and follows a person.

---

## 3. Turbo Mode (Max FPS)

```bash
python -m src.main --camera 0 --turbo --debug
```

**What it shows:** Aggressive optimization for 30+ FPS on edge devices.

---

## 4. ROS2 Integration

```bash
python -m src.main --camera 0 --ros2 --debug
```

**What it shows:** Full ROS2 integration with 10+ topic publishing.

---

## 5. Video Processing

```bash
python -m src.main --video input.mp4 --output output.mp4 --debug
```

**What it shows:** Process video files and save annotated output.

---

## 6. TensorRT Optimization (Jetson)

```bash
python -m src.main --camera 0 --precision fp16 --debug
```

**What it shows:** Hardware-accelerated inference on NVIDIA Jetson.

---

## 7. Depth Estimation (MiDaS - works without auth)

```bash
python -m src.main --camera 0 --depth-model midas-small --debug
```

**What it shows:** Depth map overlay with colored visualization.

---

## 8. Object Tracking Only

```bash
python -m src.main --camera 0 --no-face --no-gesture --no-pose --no-depth --debug
```

**What it shows:** Lightweight tracking mode for max performance.

---

## 9. Industry Template (Warehouse)

```bash
python -m src.main --template warehouse --debug
```

**What it shows:** Pre-configured pipeline for logistics/fulfillment.

---

## 10. System Info

```bash
python -m src.main --info
```

**What it shows:** Hardware detection, model recommendations, FPS estimates.

---

## Quick Demo Sequence (30 seconds)

```bash
# 1. Show basic pipeline (5s)
python -m src.main --camera 0 --debug

# 2. Show person following (10s)
python -m src.main --camera 0 --follow --debug

# 3. Show ROS2 topics (5s)
python -m src.main --camera 0 --ros2 --debug

# 4. Show info (2s)
python -m src.main --info
```

---

## Visual Hooks for Video

| Command | Visual |
|:--------|:-------|
| `--debug` | Annotated window with boxes, labels, FPS |
| `--follow` | Green box on tracked person, distance info |
| `--ros2` | Terminal showing published topics |
| `--turbo` | Higher FPS counter, smoother video |
| `--depth-model da3-small` | Depth map overlay |
