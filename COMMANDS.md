# OpenEyes Command Reference

## Quick Reference

| Goal | Command |
|:-----|:--------|
| Basic run | `python src/main.py` |
| List models | `--list-models` |
| Max speed | `--no-face --no-gesture --no-pose --no-depth` |
| With ROS2 | `--ros2` |
| Person follow | `--follow` |

---

## Run & Camera
```bash
python src/main.py                              # Default run
python src/main.py --camera 0                   # Camera index
python src/main.py --debug                       # Show video window
python src/main.py --width 480 --height 360      # Lower resolution
python src/main.py --fps 20                      # Target FPS
```

## Models
```bash
python src/main.py --list-models                 # Show all models
python src/main.py --model yolo12n               # YOLO12 (latest)
python src/main.py --model rtmdet_nano           # RTMDet
```

## Performance
```bash
python src/main.py --precision int8              # TensorRT INT8 (~2x faster)
python src/main.py --precision fp16              # TensorRT FP16
python src/main.py --batch-size 4                # Batch inference
python src/main.py --dla                         # Use DLA (Jetson)
python src/main.py --no-monitoring               # Disable stats
```

## Disable Models
```bash
python src/main.py --no-face                     # Disable face detection
python src/main.py --no-gesture                  # Disable gesture
python src/main.py --no-pose                     # Disable pose
python src/main.py --no-depth                    # Disable depth
python src/main.py --no-tracking                 # Disable tracking
python src/main.py --no-parallel                 # Disable parallel
```

## Tracking & Following
```bash
python src/main.py --follow                      # Enable person following
python src/main.py --track-max-age 60            # Tracking duration
```

## ROS2
```bash
python src/main.py --ros2                        # Enable ROS2
python src/main.py --ros2-qos sensor              # QoS: sensor/command/default
python src/main.py --ros2-actions                 # Enable action server
python src/main.py --multi-camera 0 1             # Multiple cameras
python src/main.py --ros2-time-sync               # Use ROS time sync
```

## Advanced AI
```bash
python src/main.py --vla                          # Vision-Language-Action
python src/main.py --event-camera                 # Event camera
python src/main.py --advanced-ai                   # All AI features
```

## Info & Logs
```bash
python src/main.py --version                      # Show version
python src/main.py --info                         # System info
python src/main.py --log-file logs/openeyes.log  # File logging
```

## Jetson Scripts
```bash
sudo bash scripts/jetson_perf.sh                 # Optimize performance
bash scripts/jetson_info.sh                      # System info
python3 scripts/jetson_helper.py --check         # Check status
sudo python3 scripts/jetson_helper.py --optimize # Apply optimization
```

## Common Combinations
```bash
# Maximum FPS
python src/main.py --no-face --no-gesture --no-pose --no-depth --precision int8

# Balanced (good FPS + features)
python src/main.py --no-depth --precision fp16

# Full features
python src/main.py --ros2 --follow --precision fp16

# Development
python src/main.py --debug --log-file logs/debug.log
```
