# CHANGELOG.md - Version History for OpenEyes

> **Version**: v3.0.0  
> **Last Updated**: 2026-04-16

---

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v3.0.0] - 2026-04-16

### Added - DeepStream Pipeline

- **DeepStream Pipeline**: New hardware-accelerated pipeline using NVIDIA DeepStream SDK
  - CSI camera input via nvarguscamerasrc
  - TensorRT inference via nvinfer (YOLOv10n)
  - On-screen display via nvdsosd (bounding boxes + labels)
  - Display output via nv3dsink
- **FPS Overlay**: Real-time FPS display on terminal and screen
- **UDP Output**: JSON detection output to 127.0.0.1:5000
- **ROS2 Integration**: VisionPublisher for ROS2 topics
- **pyds Bindings**: NVIDIA DeepStream Python bindings (v1.2.0)
- **Demo Script**: `demo_all_features.py` with 12 different demos
- **Plug & Play Setup**: `setup_plug_and_play.py`

### Performance

- **30 FPS** on Jetson Orin Nano (vs ~1.9 FPS with old OpenCV pipeline)
- Hardware-accelerated throughout entire pipeline

### Files Added

- `src/deepstream/pipeline.py` - Main DeepStream pipeline
- `src/deepstream/__init__.py` - Module exports
- `deepstream/config_yolov10n.txt` - YOLOv10n config
- `deepstream/labels.txt` - COCO class labels
- `tests/test_deepstream_pipeline.py` - Unit tests (18 tests)
- `tests/test_deepstream_integration.py` - Integration tests (13 tests)
- `demo_all_features.py` - Feature demos
- `setup_plug_and_play.py` - Setup script

### Fixed

- Metadata extraction using correct pyds API (`gst_buffer_get_nvds_batch_meta`)
- Probe attachment to src pad instead of sink
- Duplicate method definitions removed
- Frame counter for accurate FPS calculation

---

## [v2.6.0] - 2026-04-15

### Added - Performance Optimization

- **INT8 TensorRT Optimization** (`src/models/object_detector.py`)
  - `--int8` CLI flag for INT8 quantized inference
  - 2-3x speedup over FP16
  - Automatic engine file naming: `yolo11n-int8.engine`

- **DLA Offloading** (`src/models/object_detector.py`)
  - `--dla` CLI flag for Deep Learning Accelerator
  - Runs YOLO backbone on DLA, post-processing on GPU
  - Frees GPU for other models

- **Enhanced Jetson Perf Script** (`scripts/jetson_perf.sh`)
  - Added disk I/O optimization (noop scheduler)
  - Network TCP offload (TSO, GSO)
  - Expected FPS numbers for different modes

- **DeepStream Flag** (`src/cli/argparse.py`)
  - `--deepstream` CLI flag for DeepStream pipeline

- **Depth Overlay** (`src/core/vision_system.py`)
  - Real-time depth map visualization in debug mode
  - Color-coded (blue=near, red=far)

### Changed

- Updated version to v2.6.0
- Depth Anything V3 → Depth Anything V2 (V3 not available on HuggingFace)
- Fixed depth overlay size mismatch

### Performance Improvements

| Mode | FPS (Before) | FPS (After) |
|------|-------------|-------------|
| Default pipeline | 8-12 | 8-15 |
| +INT8 | 8-12 | 15-25 |
| +INT8 --turbo | 8-12 | 25-35 |
| Minimal (no face/gesture/pose) | 15-20 | 25-40 |

---

## [v2.5.0] - 2026-04-10

### Added - Production Infrastructure

- **Structured Logging** (`src/utils/logger.py`)
  - JSON logging with structlog
  - `--log-format json|console` CLI flag
  - Log fields: timestamp, level, logger, message, frame, detections, latency_ms

- **REST API** (`src/api/`)
  - FastAPI server with uvicorn
  - `--api`, `--api-port`, `--api-host` CLI flags
  - GET `/health` - Health check with uptime, FPS
  - GET `/metrics` - Prometheus metrics
  - GET `/models` - List registered models
  - POST `/models` - Register model
  - GET/POST `/control` - Get/update control state

- **Prometheus Metrics** (`src/utils/prometheus_exporter.py`)
  - openeyes_fps (gauge)
  - openeyes_latency_ms (histogram)
  - openeyes_detections_total (counter)
  - openeyes_frames_total (counter)
  - openeyes_memory_mb (gauge)
  - openeyes_errors_total (counter)
  - openeyes_model_inference_ms (histogram)

- **INT8 Conversion** (`scripts/convert_int8.py`)
  - TensorRT INT8 model conversion
  - `--model`, `--all`, `--calibrate` options
  - Calibration dataset generation

### Changed

- Updated version to v2.5.0
- Fixed `--version` CLI flag to show v2.5.0
- Added `VisionSystem._instance` singleton for API access

### Dependencies Added

- structlog>=24.0.0
- prometheus-client>=0.19.0
- fastapi>=0.109.0
- uvicorn>=0.27.0

---

## [v1.0.0] - 2026-04-01

### Added - Safety & Reliability

- **Health Monitor** (`src/utils/health_monitor.py`)
  - `HealthMonitor` class for 24/7 operation
  - Component health tracking with heartbeat
  - Auto-recovery from failures
  - Watchdog timer for stuck processes
  - System diagnostics (CPU, memory, GPU)

- **OTA Update System** (`src/utils/ota_update.py`)
  - `OTAUpdater` class for model updates
  - Version checking and download
  - Automatic rollback on failure
  - Safe update with verification

- **Safety Controller** (`src/utils/safety_controller.py`)
  - `SafetyController` class for robot safety
  - Emergency stop (E-STOP) integration
  - Safe velocity limits
  - Minimum distance monitoring
  - Collision avoidance triggers

### Added - Diffusion Policy & Action Chunking

- **Action Chunker** (`src/models/action_chunker.py`)
  - `ActionChunker` class for real-time control
  - 10-30 Hz control frequency support
  - Action sequence prediction
  - Smooth trajectory generation

- **Diffusion Policy** (`src/models/diffusion_policy.py`)
  - `DiffusionPolicy` class for robot manipulation
  - Denoising diffusion process
  - Multi-step action planning
  - Integration with VLA models

### Added - CLI Arguments

- `--health-monitor` - Enable health monitoring
- `--safety` - Enable safety controller
- `--max-velocity` - Set maximum velocity (m/s)
- `--min-distance` - Set minimum obstacle distance (m)
- `--ota-update` - Enable OTA updates
- `--diffusion-policy` - Enable Diffusion Policy
- `--action-chunking` - Enable action chunking
- `--control-freq` - Set control frequency (Hz)

---

## [v0.8.0] - 2026-04-01

### Added - VLA Integration & Performance

- **LoRA Fine-tuning** (`src/models/lora_finetuning.py`)
  - `LoRAAdapter` class for VLA customization
  - Low-rank adaptation layers
  - On-device fine-tuning support
  - Model checkpointing

- **TensorRT Optimizer** (`src/models/tensorrt_optimizer.py`)
  - `TensorRTOptimizer` class for model optimization
  - INT8 quantization with calibration
  - DLA (Deep Learning Accelerator) offloading
  - FP16/INT8 precision modes
  - Engine caching for fast startup

### Added - CLI Arguments

- `--int8` - Enable INT8 quantization
- `--dla` - Enable DLA offloading
- `--diffusion-policy` - Enable Diffusion Policy
- `--action-chunking` - Enable action chunking
- `--control-freq` - Set control frequency (10-30 Hz)

---

## [v0.7.0] - 2026-04-01

### Added - Multi-Modal Sensing

- **LIDAR Processing** (`src/ros2/lidar_processing.py`)
  - `LIDARProcessor` class for point cloud processing
  - Obstacle detection from LIDAR data
  - Cluster-based object detection
  - Configurable LIDAR topic subscription
  - Range and angle filtering

- **Sensor Fusion** (`src/ros2/sensor_fusion.py`)
  - `SensorFusion` class for multi-sensor integration
  - Camera + Depth + LIDAR fusion
  - 3D obstacle tracking
  - Confidence scoring

- **Multi-Camera Support** (`src/ros2/multi_camera.py`)
  - `MultiCameraManager` for handling multiple cameras
  - Synchronized capture option
  - Camera calibration support

### Added - CLI Arguments

- `--lidar` - Enable LIDAR processing
- `--lidar-topic` - Specify LIDAR topic (default: /scan)
- `--realsense` - Enable RealSense D455 support
- `--multi-camera` - Enable multi-camera mode

### Changed

- **Performance Targets**: Updated to 30+ FPS with INT8 optimization

---

## [v0.6.0] - 2026-03-30

### Added - Real VLA Models

- **VLA Model Wrappers** (`src/models/vla_models.py`)
  - `SmolVLAWrapper` - Lightweight VLA (~450M params) for Jetson
  - `OpenVLAWrapper` - Full VLA (7B params, needs AGX)
  - `OctoWrapper` - Generalist policy (~93M params)
  - Factory function `create_vla_model()` for easy instantiation

- **CLI Integration**
  - `--real-vla smolvla|openvla|octo` - Use real transformer-based VLA
  - Fallback to rule-based VLA if model fails to load
  - Config section in `config.yaml` for VLA settings

### Added - Navigation & Obstacle Avoidance (Phase 3)

- **Navigation Goal Node** (`src/ros2/navigation_goal.py`)
  - Send navigation goals to Nav2 via `/navigate_to_pose` action
  - Waypoint navigation with `/navigate_through_poses`
  - Goal cancellation support
  - Status feedback and monitoring

- **Vision Obstacle Avoidance** (`src/ros2/vision_obstacle_avoidance.py`)
  - Real-time obstacle detection from vision
  - Velocity override when obstacle detected
  - Distance estimation from bounding box height
  - Configurable obstacle classes and distances

- **Unified Launch** (`launch/unified.launch.py`)
  - Complete autonomous navigation stack
  - Vision + SLAM + Nav2 integration
  - Optional teleop and RViz support
  - All-in-one launch for production use

### Added - CLI Arguments

- `--nav2` - Enable Nav2 integration with obstacle avoidance

---

## [v0.5.0] - 2026-03-30

### Added - SLAM & Navigation Integration (Phase 1)

- **Visual Odometry** (`src/ros2/visual_odometry.py`)
  - New `VisualOdometry` class using Lucas-Kanade optical flow
  - Computes frame-to-frame motion for odometry
  - Total displacement and rotation tracking
  - Configurable focal length, baseline, and feature parameters

- **Depth to LaserScan** (`src/ros2/depth_to_laserscan.py`)
  - New `DepthToLaserScan` ROS2 node
  - Converts depth images to `/scan` topic for Nav2
  - Configurable range and angle limits
  - Supports obstacle avoidance in navigation stack

- **Isaac cuVSLAM Launch** (`launch/cuvslam.launch.py`)
  - NVIDIA Isaac ROS Visual SLAM integration
  - RealSense camera support (D435i recommended)
  - IMU fusion mode for visual-inertial odometry
  - Configurable tracking modes (stereo, VIO, RGBD)

- **Nav2 Launch** (`launch/nav2.launch.py`)
  - ROS2 Navigation2 stack integration
  - Controller server, planner server, behavior server
  - BT navigator for behavior tree navigation
  - Lifecycle manager for node orchestration

- **Nav2 Parameters** (`config/nav2_params.yaml`)
  - Complete Nav2 configuration
  - DWB controller with path following
  - SMAC planner for 2D path planning
  - Obstacle and inflation layers for costmaps

### Added - VLA Integration (Phase 2)

- **Enhanced VLA Processing** (`src/models/vla.py`)
  - Expanded context: depth, faces, pose, tracks, gesture
  - Natural language instruction processing
  - Gesture-based command recognition
  - Detection-based fallback actions
  - `_process_instruction()`, `_process_gesture()`, `_process_detection_based()`

- **VLA Pipeline Integration** (`src/main.py`)
  - VLA model initialization with `--vla` flag
  - Full context passing (detections, depth, faces, gesture, pose, tracks)
  - Command execution via `_execute_vla_command()`
  - Scene description generation

### Added - CLI Arguments

- `--slam` - Enable SLAM mode
- `--visual-odom` - Enable visual odometry publisher
- `--depth-to-scan` - Convert depth to laser scan
- `--vla` - Enable VLA processing
- `--advanced-ai` - Enable all AI features (VLA + event camera)

### Added - Configuration

- **config.yaml** - New SLAM and Nav2 configuration sections:
  - `slam.visual_odom_enabled`
  - `slam.depth_to_scan_enabled`
  - `slam.scan_topic`, `slam.odom_topic`
  - `slam.range_min`, `slam.range_max`
  - `nav2.enabled`, `nav2.map_file`, `nav2.params_file`

---

## [v0.4.4] - 2026-03-30

### Fixed

- **Gesture Detection** (`src/models/gesture_recognizer.py`)
  - Lowered min_detection_confidence to 0.1
  - Lowered min_tracking_confidence to 0.1
  - Added image resizing to 640x480 for better MediaPipe detection

### Enhanced

- **Person Following** (`src/utils/tracker.py`)
  - Added `get_follow_command_with_depth()` using bbox height ratio
  - Added `set_owner_from_gesture()` for gesture-based owner selection
  - Added `owner_track_id` property and `clear_owner()` method
  - Distance zones: forward (<60%), stop (60-95%), backward (>95%)

- **Configuration** (`config.yaml`, `src/utils/config.py`)
  - Added `gesture.confidence` parameter
  - Added `follow_distance_min` and `follow_distance_max` parameters

- **Documentation** (`manifesto.md`)
  - Complete rewrite for v0.4.4
  - Vision system philosophy and architecture
  - All 7 vision modalities documented

---

## [v0.4.3] - 2026-03-29

### Fixed

- **Gesture Detection** (`src/models/gesture_recognizer.py`)
  - Simplified classification logic
  - Added debug logging for troubleshooting

### Enhanced

- **Debug Display** (`src/main.py`)
  - All bounding boxes now same color (green)
  - Solid black background behind all text overlays
  - Removed track ID unique colors
  - Simplified pose display (keypoints only)

---

## [v0.4.2] - 2026-03-29

### Fixed

- **Gesture Detection** (`src/models/gesture_recognizer.py`)
  - Rewrote `_classify_gesture()` with correct MediaPipe landmark interpretation
  - Added "victory" gesture (index + middle extended)
  - Fixed "thumbs_up" detection (thumb above other fingers)
  - Added "ok_sign", "three", "two", "one", "thumbs_down" gestures
  - Fixed finger detection logic (Y increases downward in image coords)

### Enhanced

- **Debug Display** (`src/main.py`)
  - Increased display size from 640x360 to 1280x720

---

## [v0.4.1] - 2026-03-29

### Enhanced

- **Debug Display Overhaul** (`src/main.py`)
  - Added real-time FPS counter overlay (top-left)
  - Added latency display (avg/min/max in ms)
  - Added memory usage (used/total in MB)
  - Added frame and detection counters
  - Added per-model inference timing (top-right)
  - Added track ID visualization with unique colors
  - Added pose skeleton overlay
  - Gesture labels now shown at bottom of frame

---

## [v0.4.0] - 2026-03-29

### Added

- **VLA (Vision-Language-Action) Models**
  - New `VLAModel` class in `src/models/vla.py`
  - Intelligent action generation based on scene understanding
  - Person following with distance-based control
  - Gesture-responsive commands

- **Event Camera Integration**
  - New `EventCameraProcessor` class
  - Frame fusion with conventional camera
  - Motion detection and direction estimation
  - Fast event processing for low latency

- **Advanced AI Module**
  - `AdvancedAI` class combining VLA and event camera
  - Scene description generation

- **New CLI Flags**
  - `--vla` - Enable VLA model
  - `--event-camera` - Enable event camera processing
  - `--advanced-ai` - Enable all advanced AI features

### Changed

- **Version Bump**: v0.3.0 → v0.4.0

---

## [v0.3.0] - 2026-03-29

### Added

- **Model Registry**
  - New `ModelRegistry` class in `src/models/model_registry.py`
  - Supports YOLO11, YOLO12, RTMDet models
  - Lists available models with descriptions
  - `suggest_model()` for FPS-based recommendations

- **Specialized Models**
  - `GraspDetector` for robot manipulation
  - `FallDetector` for safety monitoring (pose-based fallback)
  - Located in `src/models/specialized.py`

- **New CLI Flags**
  - `--model` - Select detection model (yolo11n, yolo12n, rtmdet_nano)
  - `--list-models` - List all available models

### Changed

- **Model Options**
  - Added model selection support
  - Added specialized models (grasp, fall detection)

- **Version Bump**: v0.2.2 → v0.3.0

---

## [v0.2.2] - 2026-03-29

### Added

- **ROS2 Action Server**
  - New `VisionActionServer` class in `src/ros2/actions.py`
  - Support for follow, detect, track actions
  - Publishes to `/cmd_vel` for robot control

- **ROS2 QoS Configuration**
  - New `--ros2-qos` CLI flag
  - Configurable profiles: default, sensor, command, best_effort, reliable
  - `QoSConfig` helper class

- **Multi-Camera Support**
  - New `--multi-camera` CLI flag
  - `MultiCameraManager` for handling multiple cameras
  - Configurable in config.yaml under `multi_camera` section

- **Time Synchronization**
  - New `--ros2-time-sync` CLI flag
  - `TimeSyncManager` for synchronized timestamps

- **New CLI Flags**
  - `--ros2-qos` - QoS profile selection
  - `--ros2-actions` - Enable action server
  - `--multi-camera` - Multi-camera mode
  - `--ros2-time-sync` - Use ROS2 time sync

### Changed

- **Config Updates**
  - Added `ros2.qos_profile`, `ros2.actions_enabled`, `ros2.time_sync`
  - Added `multi_camera` section with enabled, sources

- **Version Bump**: v0.2.1 → v0.2.2

---

## [v0.2.1] - 2026-03-29

### Added

- **Object Tracking**
  - New `ObjectTracker` class in `src/utils/tracker.py`
  - IoU-based multi-object tracking
  - Tracks objects across frames with unique IDs
  - Configurable max_age, min_hits, IoU threshold

- **Person Following**
  - Automatic person selection based on frame center
  - Generates movement commands (forward, backward, left, right, stop)
  - Tracks follow target across frames

- **Track Data in Output**
  - Added `TrackData` to `VisionResult`
  - Includes track_id, class_name, bbox, centroid, age

- **New CLI Flags**
  - `--no-tracking` - Disable object tracking
  - `--follow` - Enable person following
  - `--track-max-age` - Configure tracking max age

### Changed

- **Config Updates**
  - Added `tracking` section with enabled, max_age, min_hits, iou_threshold, follow_enabled

- **Version Bump**: v0.2.0 → v0.2.1

---

## [v0.2.0] - 2026-03-29

### Added

- **Performance Monitoring**
  - New `PerformanceMonitor` class in `src/utils/performance_monitor.py`
  - Tracks FPS, latency, memory usage, per-model inference times
  - Configurable via config.yaml `performance.monitoring` section

- **TensorRT INT8 Support**
  - Added `--precision` CLI flag (fp32, fp16, int8)
  - Configurable via `performance.tensorrt.precision`
  - INT8 provides ~2x speedup over FP16 on Jetson

- **DLA Offloading Support**
  - Added `--dla` CLI flag for Deep Learning Accelerator
  - Configurable via `performance.tensorrt.dla_enabled`
  - Offloads inference to Jetson DLA for even lower latency

- **Batch Inference**
  - Added `--batch-size` CLI flag
  - Configurable via `performance.batch_inference` section
  - Supports dynamic batching for throughput optimization

- **New CLI Flags**
  - `--deepstream` - Use DeepStream pipeline
  - `--precision` - TensorRT precision (fp32/fp16/int8)
  - `--dla` - Use DLA for inference
  - `--batch-size` - Batch size for inference
  - `--no-monitoring` - Disable performance monitoring

### Changed

- **Config Updates**
  - Added `performance.tensorrt` section for precision, DLA settings
  - Added `performance.batch_inference` section for batching
  - Added `performance.monitoring` section for stats

- **Version Bump**: v0.1.2 → v0.2.0

---

## [v0.1.2] - 2026-03-29

### Added

- **--info CLI Flag**
  - Show system information and OpenEyes recommendations
  - Displays Jetson-specific optimization tips
  - Quick reference for performance flags

- **--log-file CLI Flag**
  - Enable file logging with automatic rotation
  - Default: 5MB max file size, 3 backup files
  - Useful for debugging and production monitoring

- **Jetson Optimization Scripts**
  - `scripts/jetson_perf.sh` - One-command performance optimization
  - `scripts/jetson_info.sh` - Detailed system information
  - `scripts/jetson_helper.py` - Python helper (--info, --optimize, --check)

- **Log Rotation Support**
  - Added RotatingFileHandler to logger
  - Configurable max bytes and backup count
  - Prevents disk from filling up during long runs

### Changed

- **Version Bump**: v0.1.1 → v0.1.2

---

## [v0.1.1] - 2026-03-29

### Added

- **--no-depth CLI Flag**
  - New flag to disable depth estimation for maximum FPS
  - Depth estimation is computationally expensive

- **Model Enable/Disable Flags Now Working**
  - `--no-face`, `--no-gesture`, `--no-pose` were defined but not wired
  - Now properly skip model initialization when specified

- **Jetson Optimization Hint**
  - Startup message: "Run 'sudo nvpmodel -m 0 && sudo jetson_clocks' for max performance"
  - Auto-detects Jetson platform via `/proc/device-tree/model`

- **Model Status Logging**
  - Startup logs show which models are enabled/disabled
  - Helps verify configuration at startup

### Changed

- **More Aggressive Frame Skipping (Default)**
  - depth: 4 → 8 (every 8th frame)
  - face: 4 → 6 (every 6th frame)
  - gesture: 4 → 6 (every 6th frame)
  - pose: 4 → 6 (every 6th frame)

- **Adaptive Skipper Parameters**
  - base_skip: 3 → 2
  - min_skip: 2 → 1
  - max_skip: 5 → 4

### Performance

| Configuration | Expected FPS |
|:-------------|:------------|
| All models enabled (default) | ~10-12 |
| --no-face --no-gesture --no-pose | ~18-22 |
| --no-face --no-gesture --no-pose --no-depth | ~22-25 |
| + Jetson max performance (sudo nvpmodel -m 0 && sudo jetson_clocks) | +20-30% |

### CLI New Flags

```bash
# Disable specific models for speed
python src/main.py --no-face              # Skip face detection
python src/main.py --no-gesture           # Skip gesture recognition  
python src/main.py --no-pose              # Skip pose estimation
python src/main.py --no-depth             # Skip depth estimation (NEW)

# Disable multiple models
python src/main.py --no-face --no-gesture --no-pose --no-depth
```

---

## [v0.1.0] - 2026-03-28

### Added

- **ROS2 Configuration**
  - New `ros2` section in config.yaml
  - Configurable topics: detections, depth, faces, gestures, poses, cmd, status
  - Frame ID and confidence threshold settings

- **CSI Camera Improvements**
  - Device detection via `/dev/video*` check
  - Queue element added to GStreamer pipeline for stability
  - 1080p native resolution preferred
  - Retry logic with initialization delays

- **PoseData Enhancements**
  - Added `bbox` field for bounding box
  - Added `landmarks` field for pose landmarks

- **YOLO Path Resolution**
  - Fixed to use absolute path resolution from config directory

- **Complete ROS2 Vision Integration**
  - VisionPublisher with all vision modality publishers
  - Detections, depth, faces, gestures, poses topics
  - JSON fallback mode using std_msgs/String (avoids vision_msgs issues)

- **Command Subscription**
  - New `/vision/cmd` topic for robot commands
  - Valid commands: forward, backward, stop, left, right, follow
  - Command callback system for robot control integration

- **Parameter Validation**
  - Camera parameter validation in constructor
  - VisionPublisher parameter validation
  - Meaningful error messages for invalid inputs

- **Status Message Enhancement**
  - Timestamps added to vision status messages

- **CLI Enhancements**
  - `--ros2` flag to enable ROS2 publishing
  - `--version` flag to display version

### Changed

- Updated default version to v0.1.0
- Vision status now includes face and gesture counts
- Command field added to status output

---

## [v0.0.3] - 2026-03-26

### Added

- **YOLO11n Model**
  - YOLO11n model with better performance than YOLOv10n
  - ONNX export for TensorRT deployment
  - Expected FPS: 139 (FP16), 180 (INT8)

- **Adaptive Frame Skipping**
  - Universal frame skipper for all models
  - Adaptive skipping based on motion detection
  - Multi-model frame scheduler
  - Configurable skip intervals per model

- **ROS2 Integration**
  - VisionPublisher node for publishing detections
  - VisionControlNode for robot control
  - VisionWrapperNode for OpenEyes integration
  - Support for vision_msgs (Detection2DArray)

- **DeepStream SDK Integration**
  - DeepStream-Yolo custom parser library
  - GStreamer pipeline for CSI camera
  - Configuration files for YOLOv10

- **Performance Optimizations**
  - Motion-based adaptive processing
  - Result caching across all models
  - Frame interpolation for skipped frames

### Changed

- Updated version to v0.0.3
- Model path configuration supports YOLO11n
- Default frame scheduler intervals: detector(1), depth(2), face(2), gesture(2), pose(2)

### Dependencies Added

- pyds (DeepStream Python bindings)

---

## [v0.0.2] - 2026-03-25

### Added

- **Object Detection**
  - YOLOv10n model with PyTorch + CUDA acceleration
  - ONNX Runtime support with TensorRT provider
  - Automatic CUDA/ONNX fallback detection

- **Depth Estimation**
  - MiDaS_small model integration
  - GPU acceleration support
  - Depth map estimation and distance calculation

- **Face Detection**
  - MediaPipe FaceMesh integration
  - Multi-face support (up to 3)

- **Gesture Recognition**
  - MediaPipe Hands integration
  - Real-time hand tracking

- **Pose Estimation**
  - MediaPipe Pose integration
  - Body keypoint detection

- **Performance Optimizations**
  - Parallel processing with ThreadPoolExecutor
  - Frame skipping for pose estimation
  - Result caching for face/gesture

- **Camera Support**
  - CSI camera (IMX219) via nvarguscamerasrc
  - GStreamer pipeline integration
  - Auto-detection of Jetson platform

- **Display**
  - Auto-display detection (DISPLAY=:0 fallback)
  - Debug visualization with bounding boxes

### Changed

- Updated version to v0.0.2
- Performance: 5-6 FPS → 7-10 FPS with all models
- Default model: YOLOv8n → YOLOv10n
- CLI options added: --no-parallel, --pose-every

### Dependencies Updated

- Added `timm` for depth estimation
- Added `onnxruntime-gpu` for TensorRT support
- Downgraded MediaPipe to 0.10.9 for stability

### Known Issues

- MediaPipe may crash with certain frame sizes (workaround: use frame skipping)
- TensorRT engine build may timeout on low-memory systems (use ONNX fallback)

---

## [v0.0.1] - 2026-03-15

### Added

- **Documentation**
  - README.md with project overview
  - AGENTS.md developer guidelines
  - TECHNICAL_SPEC.md technical specifications
  - ARCHITECTURE.md system architecture
  - HARDWARE.md hardware specifications
  - API_SPEC.md API documentation
  - QUICKSTART.md quick start guide
  - INSTALL.md detailed installation
  - USER_GUIDE.md user guide
  - TROUBLESHOOTING.md common issues
  - CONTRIBUTING.md contribution guidelines
  - ROADMAP.md project roadmap
  - CHANGELOG.md version history

- **Project Structure**
  - Directory structure for src/, models/, docs/
  - requirements.txt with dependencies
  - LICENSE (Apache 2.0)

- **Source Code**
  - config.yaml with default configuration
  - camera/ module with CameraHandler
  - models/ module with ObjectDetector (YOLOv8)
  - output/ module with JSON formatter and UDP sender
  - utils/ module with config loader and logger
  - main.py entry point

- **Testing**
  - Unit tests for config, camera, models, output (36 tests)

### Changed

- Initial repository setup
- Project named "OpenEyes"
- License set to Apache 2.0

---

## [v0.0.3] - 2026-03-25

### Added

- **DeepStream SDK Integration**
  - DeepStream 7.1 installation and setup
  - Python bindings (pyds) for DeepStream
  - DeepStream-Yolo custom parser library
  - GStreamer pipeline for CSI camera
  - Test scripts for DeepStream pipeline

- **Documentation**
  - DEEPSTREAM.md integration guide
  - DeepStream configuration files

### Known Issues

- TensorRT engine build requires significant time (~5-10 minutes)
- Hybrid DeepStream + MediaPipe integration requires further testing

---

## [Unreleased]

### Planned for v1.0.0

- [ ] Multi-camera support
- [ ] Production hardening
- [ ] Motor control integration
- [ ] Further FPS optimization (target: 25-30 FPS)

### Planned for v1.1.0

- [ ] YOLOv10s for higher accuracy
- [ ] Custom model training
- [ ] Stereo vision

---

## Version Format

Given a version number `MAJOR.MINOR.PATCH`:

- **MAJOR** - Incompatible API changes
- **MINOR** - New functionality (backwards compatible)
- **PATCH** - Bug fixes

---

## Upgrade Guide

### From v0.0.1 to v0.0.2

1. Update requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Download new models (if not already included):
   ```bash
   # YOLOv10n is included in models/ folder
   ```

3. Run vision system:
   ```bash
   python src/main.py --debug
   ```

4. For optimal performance, enable Jetson max mode:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

---

## Release Cycle

| Version | Type | Target |
|:--------|:-----|:-------|
| v0.0.1 | Initial | March 2026 |
| v0.0.2 | Minor | March 2026 |
| v1.0.0 | Major | April 2026 |
| v1.1.0 | Minor | April 2026 |
| v1.5.0 | Major | Q2 2026 |
| v2.0.0 | Major | Q3 2026 |
| v2.5.0 | Major | Q4 2026 |

---

## [v2.5.0] - In Development (Q4 2026)

### Added - World Models & Predictive Intelligence

- **World Models Module** (`src/world_model/`)
  - `WorldModel` abstract interface (encode/predict/plan)
  - `LeWorldModel` - 15M param latent-space world model
  - `CEMPlanner` - Cross-Entropy Method for latent-space MPC
  - `SafetyEvaluator` - Predictive safety evaluation before action execution
  - Predictive tracking with occlusion handling (5-10 frame prediction)
  - Bounding box trajectory prediction for all tracked objects
  - Online learning from observation history
  - Debug visualization: ghost boxes showing predicted future positions

- **V-JEPA 2 Perception** (`src/models/vjepa2_extractor.py`)
  - V-JEPA 2 ViT-B/L/H feature extractor (80M-600M params)
  - Spatiotemporal feature extraction from video clips
  - Frame buffer for streaming feature extraction
  - Ready for feature fusion with YOLO detection

- **Edge-Cloud Split Architecture** (planned)
  - Adaptive routing between edge and cloud inference
  - Cloud API for heavy model offloading

### Changed
- World model integrated into frame processor pipeline
- Tracker now supports predicted positions during occlusion
- VisionResult now includes predictions list

---

## [v2.0.0] - In Development (Q3 2026)

### Added - Hardware-Agnostic Edge Vision Framework

- **Hardware Abstraction Layer** (`src/backends/`)
  - `Backend` abstract interface for all inference engines
  - `BackendRegistry` with auto-selection
  - Support for TensorRT, OpenVINO, TVM, Hailo DFC, QNN, ONNXRuntime
  - Unified model export across backends

- **Platform Detection** (`src/platforms/`)
  - Auto-detects Jetson, Raspberry Pi, Intel NPU, Hailo, Qualcomm
  - Platform-specific optimization profiles
  - `PlatformInfo` dataclass with hardware capabilities

- **Industry Templates** (`src/templates/`)
  - `TemplateManager` with 4 pre-configured pipelines
  - Warehouse/Logistics: package detection, damage inspection, pallet counting
  - Manufacturing QA: defect detection, PPE compliance, assembly verification
  - Agriculture: weed detection, crop health, yield estimation
  - Retail: shelf monitoring, inventory counting, customer analytics
  - Template save/load from YAML files
  - CLI: `--template warehouse`

- **Fleet Management** (`src/fleet/`)
  - `DeviceHeartbeat` protocol with FPS, latency, CPU/GPU/memory/temperature
  - `ModelDeployment` with device/group targeting and rollback
  - `ModelRegistry` with SHA256 checksums and deployment tracking
  - `FleetClient` for edge device communication
  - CLI commands: `fleet register`, `fleet list`, `fleet deploy`, `fleet telemetry`

- **Benchmarking Suite** (`benchmarks/`)
  - Comprehensive FPS/latency benchmarking across all models
  - Mean, p50, p95, p99 latency measurements
  - JSON report generation with summary statistics
  - CLI: `python -m benchmarks.run_benchmarks --all --report`

- **Production Toolkit** (`docker/`)
  - `Dockerfile.jetson-orin-nano` for containerized deployment
  - `docker-compose.yml` for easy multi-container setup
  - `openeyes.service` systemd unit with security hardening
  - Prometheus metrics endpoint (port 9090)
  - Health checks, resource limits, auto-restart

### Changed
- Version: v1.0.0 → v2.0.0
- Hardware support expanded from Jetson-only to multi-platform

---

## [v1.5.0] - In Development (Q2 2026)

### Added - YOLO26 + Depth Anything V3 + Fleet Foundation

- **YOLO26n Support**
  - Latest SOTA detection model (2.57M params, 6.1 GFLOPs)
  - NMS-free end-to-end predictions
  - 43% faster CPU inference vs YOLO11
  - Added to model registry with `--model yolo26n`

- **Depth Anything V3 Integration**
  - `DepthAnythingV3` class with da3-small/base/large variants
  - 35.7% better camera pose accuracy vs MiDaS
  - Depth-ray representation for improved geometric accuracy
  - Multi-view depth support capability
  - Unified `DepthEstimator` supporting both MiDaS and DA3
  - CLI: `--depth-model da3-small` (default: midas-small)

- **Performance Optimizations**
  - `--turbo` mode: aggressive frame skipping (depth=16, face/gesture/pose=12)
  - GStreamer pipeline: hardware scaling via nvvidconv (1280x720 capture)
  - MediaPipe: complexity=0, max_hands=1, max_faces=1
  - Thread pool: optimized from 5 to 4 workers
  - `scripts/jetson_perf.sh`: MAXN SUPER mode, jetson_clocks, disable unnecessary services
  - `scripts/export_tensorrt_optimized.py`: `--best` and `--useCudaGraph` engine building

### Changed
- Default depth model: midas-small (DA3 requires HuggingFace token)
- GStreamer capture: 1920x1080 → 1280x720 (fixes NVMM OOM on Orin Nano)
- MediaPipe: max_faces 3→1, max_hands 2→1, complexity=0

---

## [v1.1.0] - 2026-04-02

### Added - World Models Phase 1

- **LeWorldModel Integration**
  - 15M parameter latent-space world model
  - CEM planner for goal-conditioned planning
  - Predictive tracking with occlusion handling
  - Safety evaluation before action execution
  - CLI: `--world-model lewm`, `--plan-horizon`, `--plan-samples`, `--safety-predict`
  - Debug visualization: predicted future positions as ghost boxes

- **World Model Documentation**
  - `docs/WORLD_MODELS.md`: Complete technical documentation
  - `WORLD_MODELS_PLAN.md`: 4-phase implementation plan
  - `AGENTS.md`: World Model Development Guidelines

### Changed
- 40 new tests for world model module (total: 76 passing)
- Tracker now supports `update_with_predictions()` for occlusion handling

---

---

## Acknowledgments

This CHANGELOG format is based on [Keep a Changelog](https://keepachangelog.com).
