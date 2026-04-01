# OpenEyes Command Reference

## Quick Reference

| Goal | Command |
|:-----|:--------|
| Basic run | `python src/main.py` |
| List models | `--list-models` |
| Max speed | `--no-face --no-gesture --no-pose --no-depth` |
| With ROS2 | `--ros2` |
| Person follow | `--follow` |
| Version | `--version` |

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
python src/main.py --int8                        # INT8 quantization (v0.8.0+)
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

## Multi-Modal Sensing (v0.7.0+)
```bash
python src/main.py --lidar                        # Enable LIDAR processing
python src/main.py --lidar-topic /scan            # LIDAR topic (default: /scan)
python src/main.py --realsense                    # Enable RealSense D455
python src/main.py --multi-camera                 # Multi-camera mode
```

## VLA & Performance (v0.8.0+)
```bash
python src/main.py --int8                         # Enable INT8 quantization
python src/main.py --dla                         # Enable DLA offloading
python src/main.py --diffusion-policy             # Enable Diffusion Policy
python src/main.py --action-chunking              # Enable action chunking
python src/main.py --control-freq 20              # Control frequency (10-30 Hz)
```

## Safety & Reliability (v1.0.0+)
```bash
python src/main.py --safety                      # Enable safety controller
python src/main.py --health-monitor              # Enable health monitoring
python src/main.py --max-velocity 1.0            # Max velocity (m/s)
python src/main.py --min-distance 0.3            # Min obstacle distance (m)
python src/main.py --ota-update                   # Enable OTA updates
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
# Maximum FPS (v0.7.0+)
python src/main.py --no-face --no-gesture --no-pose --no-depth --int8

# Balanced (good FPS + features)
python src/main.py --no-depth --precision fp16

# Full features
python src/main.py --ros2 --follow --precision fp16

# Development
python src/main.py --debug --log-file logs/debug.log

# Full Safety Mode (v1.0.0+)
python src/main.py --safety --health-monitor --max-velocity 0.5 --min-distance 0.5

# Multi-Modal with Fusion (v0.7.0+)
python src/main.py --lidar --realsense --sensor-fusion

# High-Performance VLA (v0.8.0+)
python src/main.py --vla --int8 --dla --action-chunking --control-freq 30
```
