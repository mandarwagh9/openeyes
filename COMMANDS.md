# OpenEyes Command Reference

> **Version**: v2.5.0 | **Last Updated**: 2026-04-10

---

## Quick Reference

| Goal | Command |
|:-----|:--------|
| Basic run | `python -m src.main --camera 0 --debug` |
| World model | `--world-model lewm --follow` |
| Turbo mode | `--turbo` |
| Industry template | `--template warehouse` |
| Max speed | `--no-face --no-gesture --no-pose --turbo` |
| With ROS2 | `--ros2` |
| Benchmark | `python -m benchmarks.run_benchmarks --all` |
| Fleet register | `openeyes fleet register --name robot-01` |

---

## Run & Camera

```bash
python -m src.main                              # Default run
python -m src.main --camera 0                   # Camera index
python -m src.main --debug                       # Show annotated video window
python -m src.main --width 640 --height 480      # Resolution
python -m src.main --fps 30                      # Target FPS
```

## Models

```bash
python -m src.main --list-models                 # Show all available models
python -m src.main --model yolo11n               # YOLO11n (default, TensorRT)
python -m src.main --model yolo12n               # YOLO12n
python -m src.main --model yolo26n               # YOLO26n (latest SOTA, NMS-free)
python -m src.main --model rtmdet_nano           # RTMDet-nano
```

## Depth Estimation

```bash
python -m src.main --depth-model midas-small     # MiDaS Small (default, works offline)
python -m src.main --depth-model da3-small       # Depth Anything V3 Small (requires HF token)
python -m src.main --depth-model da3-base        # Depth Anything V3 Base
python -m src.main --no-depth                    # Disable depth estimation
```

## World Models

```bash
python -m src.main --world-model lewm            # Enable LeWorldModel (15M params)
python -m src.main --world-model vjepa2          # Enable V-JEPA 2 (not yet implemented)
python -m src.main --world-model none            # Disable world model (default)
python -m src.main --plan-horizon 10             # Planning horizon in steps
python -m src.main --plan-samples 100            # CEM sample count
python -m src.main --prediction-fps 30           # Prediction update rate
python -m src.main --occlusion-frames 5          # Max frames to predict through occlusion
python -m src.main --safety-predict              # Enable predictive safety evaluation
```

## Performance

```bash
python -m src.main --turbo                       # Turbo mode: aggressive frame skipping
python -m src.main --int8                        # INT8 quantization
python -m src.main --dla                         # DLA offloading (Jetson)
python -m src.main --no-monitoring               # Disable performance stats
```

## Disable Models

```bash
python -m src.main --no-face                     # Disable face detection
python -m src.main --no-gesture                  # Disable gesture recognition
python -m src.main --no-pose                     # Disable pose estimation
python -m src.main --no-depth                    # Disable depth estimation
python -m src.main --no-tracking                 # Disable object tracking
python -m src.main --no-parallel                 # Disable parallel processing
```

## Tracking & Following

```bash
python -m src.main --follow                      # Enable person following
python -m src.main --track-max-age 60            # Max frames to keep lost track
```

## Industry Templates

```bash
python -m src.main --template warehouse          # Warehouse/Logistics pipeline
python -m src.main --template manufacturing-qa   # Manufacturing QA pipeline
python -m src.main --template agriculture        # Agriculture pipeline
python -m src.main --template retail             # Retail pipeline
```

## ROS2

```bash
python -m src.main --ros2                        # Enable ROS2 publishing
python -m src.main --ros2-qos sensor             # QoS: sensor/command/default/best_effort/reliable
python -m src.main --ros2-actions                # Enable action server
python -m src.main --multi-camera 0 1            # Multiple cameras
python -m src.main --ros2-time-sync              # Use ROS time sync
```

## SLAM & Navigation

```bash
python -m src.main --slam                        # Enable SLAM mode
python -m src.main --visual-odom                 # Enable visual odometry
python -m src.main --depth-to-scan               # Convert depth to laser scan
python -m src.main --nav2                        # Enable Nav2 integration
```

## Multi-Modal Sensing

```bash
python -m src.main --lidar                       # Enable LIDAR processing
python -m src.main --lidar-topic /scan           # LIDAR topic (default: /scan)
python -m src.main --realsense                   # Enable RealSense D455
```

## VLA & Advanced AI

```bash
python -m src.main --vla                         # Enable VLA processing
python -m src.main --advanced-ai                 # Enable all AI features
python -m src.main --real-vla smolvla            # Use SmolVLA (~450M params)
python -m src.main --real-vla openvla            # Use OpenVLA (7B params)
python -m src.main --real-vla octo               # Use Octo (~93M params)
python -m src.main --diffusion-policy            # Enable Diffusion Policy
python -m src.main --action-chunking             # Enable action chunking
python -m src.main --control-freq 20             # Control frequency (10-30 Hz)
python -m src.main --event-camera                # Enable event camera
```

## Safety & Reliability

```bash
python -m src.main --safety                      # Enable safety controller
python -m src.main --health-monitor              # Enable health monitoring
python -m src.main --max-velocity 1.0            # Max velocity (m/s)
python -m src.main --min-distance 0.3            # Min obstacle distance (m)
python -m src.main --ota-update                  # Enable OTA updates
```

## Fleet Management

```bash
openeyes fleet register --name robot-01 --group warehouse   # Register device
openeyes fleet list                                          # List all devices
openeyes fleet list --group warehouse                        # List devices in group
openeyes fleet info robot-01                                 # Device details
openeyes fleet models list                                   # List available models
openeyes fleet models upload --model yolo26n --version v1.2  # Upload model
openeyes fleet deploy --model yolo26n --version v1.2 --group warehouse  # Deploy
openeyes fleet telemetry --device robot-01 --last 1h         # View telemetry
openeyes fleet dashboard --port 8080                         # Web dashboard
openeyes fleet ota check                                     # Check for updates
openeyes fleet ota update --version v2.0.1 --group all      # OTA update
```

## Benchmarking

```bash
python -m benchmarks.run_benchmarks --all                    # Benchmark all models
python -m benchmarks.run_benchmarks --model yolo11n          # Benchmark specific model
python -m benchmarks.run_benchmarks --iterations 200         # Custom iterations
python -m benchmarks.run_benchmarks --report                 # Generate JSON report
python -m benchmarks.run_benchmarks --output-dir ./results   # Custom output dir
```

## Info & Logs

```bash
python -m src.main --version                      # Show version
python -m src.main --info                         # Show system info
python -m src.main --log-file logs/openeyes.log   # File logging with rotation
python -m src.main --log-format json          # JSON structured logs
python -m src.main --log-format console       # Console logs (default)
```

## REST API (v2.5.0)

```bash
python -m src.main --api                        # Enable REST API server
python -m src.main --api-port 8000            # API port (default: 8000)
python -m src.main --api-host 0.0.0.0         # API host (default: 127.0.0.1)
```

## Video Processing (v2.5.0)

```bash
python -m src.main --video input.mp4          # Process video file
python -m src.main --video input.mp4 --output output.mp4  # Process and save
python -m src.main --video input.mp4 --follow  # Follow person in video
```

## Jetson Optimization

```bash
sudo bash scripts/jetson_perf.sh                 # MAXN SUPER + jetson_clocks (recommended)
bash scripts/jetson_info.sh                      # System information
python3 scripts/jetson_helper.py --check         # Check system status
sudo python3 scripts/jetson_helper.py --optimize # Apply optimization
```

## TensorRT Engine Optimization

```bash
# Rebuild engine with SOTA optimizations
python scripts/export_tensorrt_optimized.py --model models/yolo11n.pt

# With INT8 quantization (requires calibration images)
python scripts/export_tensorrt_optimized.py --model models/yolo11n.pt --int8 --calib-dir /path/to/images
```

---

## Common Combinations

```bash
# Maximum FPS
python -m src.main --no-face --no-gesture --no-pose --turbo

# Balanced (good FPS + features)
python -m src.main --turbo

# Full features with world model
python -m src.main --world-model lewm --follow --debug

# Warehouse deployment
python -m src.main --template warehouse --turbo --debug

# Manufacturing QA
python -m src.main --template manufacturing-qa --debug

# Safety-critical with world model
python -m src.main --world-model lewm --safety-predict --min-distance 0.5 --debug

# Development with logging
python -m src.main --debug --log-file logs/debug.log

# Full Safety Mode
python -m src.main --safety --health-monitor --max-velocity 0.5 --min-distance 0.5

# High-Performance VLA
python -m src.main --vla --int8 --dla --action-chunking --control-freq 30

# Multi-Modal with LIDAR
python -m src.main --lidar --realsense

# ROS2 with world model
python -m src.main --ros2 --world-model lewm --follow
```
