# Commands

Complete reference for all CLI commands and options.

## Basic Commands

```bash
# Run with debug window
python src/main.py --debug

# Run headless (no display)
python src/main.py

# Show version
python src/main.py --version

# Show system info
python src/main.py --info
```

---

## Camera Options

| Option | Description | Default |
|:-------|:------------|:--------|
| `--camera CAMERA` | Camera index (0, 1...) or RTSP URL | 0 |
| `--width WIDTH` | Frame width | 640 |
| `--height HEIGHT` | Frame height | 480 |
| `--fps FPS` | Target FPS | 30 |

Examples:
```bash
python src/main.py --camera 0
python src/main.py --camera 1
python src/main.py --camera rtsp://192.168.1.100:8554/stream
python src/main.py --width 1280 --height 720
```

---

## Model Options

| Option | Description |
|:-------|:------------|
| `--no-face` | Disable face detection |
| `--no-gesture` | Disable gesture recognition |
| `--no-pose` | Disable pose estimation |
| `--no-depth` | Disable depth estimation |
| `--pose-every N` | Run pose every N frames (default: 2) |

Examples:
```bash
python src/main.py --no-face --no-gesture --no-pose
python src/main.py --pose-every 3
python src/main.py --no-depth
```

---

## Performance Options

| Option | Description |
|:-------|:------------|
| `--no-parallel` | Disable parallel processing |
| `--precision PRECISION` | Model precision (fp32, fp16, int8) |
| `--dla` | Use DLA (Deep Learning Accelerator) |
| `--batch-size N` | Batch size for inference |

Examples:
```bash
python src/main.py --precision int8
python src/main.py --dla
python src/main.py --batch-size 2
```

---

## Tracking Options

| Option | Description | Default |
|:-------|:------------|:--------|
| `--follow` | Enable person following | disabled |
| `--no-tracking` | Disable object tracking | - |
| `--track-max-age AGE` | Max tracking age in frames | 30 |

Examples:
```bash
python src/main.py --follow
python src/main.py --follow --track-max-age 60
python src/main.py --no-tracking
```

---

## Model Selection

| Option | Description |
|:-------|:------------|
| `--list-models` | List available models |
| `--model MODEL` | Select YOLO model (yolo11n, yolov8n, rtmdet_nano, etc.) |

Examples:
```bash
python src/main.py --list-models
python src/main.py --model yolo12n
python src/main.py --model rtmdet_nano
```

---

## Advanced AI

| Option | Description |
|:-------|:------------|
| `--vla` | Enable VLA (Vision-Language-Action) processing |
| `--advanced-ai` | Enable all AI features |
| `--real-vla MODEL` | Use real VLA model (smolvla, openvla, octo) |
| `--event-camera` | Enable event camera processing |

Examples:
```bash
python src/main.py --vla
python src/main.py --advanced-ai
python src/main.py --real-vla smolvla
python src/main.py --real-vla octo
```

---

## ROS2 Options

| Option | Description |
|:-------|:------------|
| `--ros2` | Enable ROS2 publishing |
| `--nav2` | Enable Nav2 navigation |

Examples:
```bash
python src/main.py --ros2
python src/main.py --nav2
```

---

## SLAM & Navigation

| Option | Description |
|:-------|:------------|
| `--visual-odom` | Enable visual odometry |
| `--depth-to-scan` | Convert depth to laser scan |
| `--slam` | Enable full SLAM mode |

Examples:
```bash
python src/main.py --visual-odom
python src/main.py --depth-to-scan
python src/main.py --slam
```

---

## Output Options

| Option | Description | Default |
|:-------|:------------|:--------|
| `--host HOST` | Output host IP | 127.0.0.1 |
| `--port PORT` | Output port | 5000 |
| `--config CONFIG` | Config file path | config.yaml |
| `--log-file PATH` | Log file path | - |

Examples:
```bash
python src/main.py --host 192.168.1.100 --port 5000
python src/main.py --config custom.yaml
python src/main.py --log-file logs/openeyes.log
```

---

## Keyboard Controls

When running with `--debug`:

| Key | Action |
|:----|:-------|
| `q` | Quit |
| `s` | Save screenshot |
| `d` | Toggle debug overlay |
| `p` | Pause/Resume |
| `space` | Pause/Resume |
| `f` | Toggle fullscreen |
| `h` | Show help |

---

## Performance Examples

| Command | Expected FPS |
|:--------|:------------|
| `python src/main.py` | ~10-12 FPS (all models) |
| `python src/main.py --no-face --no-gesture --no-pose` | ~18-22 FPS |
| `python src/main.py --no-face --no-gesture --no-pose --no-depth` | ~22-25 FPS |
| `python src/main.py --no-face --no-gesture --no-pose` + Jetson max | ~22-28 FPS |
| `python src/main.py --precision int8` | ~30-40 FPS (minimal) |

!!! tip
    Run `sudo nvpmodel -m 0 && sudo jetson_clocks` for maximum Jetson performance!