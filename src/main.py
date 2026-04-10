#!/usr/bin/env python3
"""OpenEyes Vision System - Main Entry Point.

Modular architecture with separated concerns:
- CLI parsing: src/cli/argparse.py
- Core vision system: src/core/vision_system.py
- Frame processing: src/core/frame_processor.py
- Initialization: src/core/initialization.py
"""

import logging
import os
import platform
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

ros2_python_paths = [
    '/opt/ros/humble/local/lib/python3.10/dist-packages',
    '/opt/ros/humble/lib/python3/dist-packages',
    '/usr/lib/python3/dist-packages',
]
for p in ros2_python_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

if not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':0'

from src.cli.argparse import create_parser, parse_args, apply_args_to_config
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.core.vision_system import VisionSystem


def _start_api(host: str, port: int) -> None:
    """Start the REST API server."""
    import threading
    import uvicorn

    def run():
        from src.api import create_api
        app = create_api()
        uvicorn.run(app, host=host, port=port, log_level="info")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

try:
    from src.ros2.visual_odometry import VisualOdometry as VO
except ImportError:
    VO = None
VisualOdometry = VO


def show_system_info() -> None:
    """Display system information and OpenEyes recommendations."""
    print("=" * 50)
    print("OpenEyes System Information")
    print("=" * 50)
    print(f"  Version: v0.6.0")
    print(f"  Python: {platform.python_version()}")
    print(f"  Platform: {platform.system()} {platform.machine()}")

    is_jetson = False
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip()
            if "jetson" in model.lower() or "tegra" in model.lower():
                is_jetson = True
                print(f"  Device: {model}")
                print("\n[Jetson Optimization]")
                print("  Run: sudo bash scripts/jetson_perf.sh")
                print("  Info: bash scripts/jetson_info.sh")
    except Exception:
        print(f"  Device: {platform.machine()}")

    print("\n[OpenEyes Recommendations]")
    print("  Minimal (fastest): --no-face --no-gesture --no-pose --no-depth")
    print("  Balanced:         --no-depth")
    print("  Full:             (all models enabled)")
    print("  With ROS2:        --ros2")
    print("\n[Performance Tips]")
    if is_jetson:
        print("  - Run sudo jetson_clocks for max performance")
        print("  - Use TensorRT models (.engine) for 2x speedup")
        print("  - Consider disabling depth for >25 FPS")
    else:
        print("  - Use GPU-accelerated models when available")
        print("  - Lower resolution: --width 480 --height 360")
        print("  - Lower target FPS: --fps 20")
    print("=" * 50)


def main() -> None:
    """Main entry point for OpenEyes Vision System."""
    parser = create_parser()
    args = parser.parse_args()

    if args.list_models:
        from src.models.model_registry import ModelRegistry
        print("=" * 50)
        print("Available Models")
        print("=" * 50)
        print("\n[Detection Models]")
        for model in ModelRegistry.get_detection_models():
            info = ModelRegistry.get_model_info(model)
            print(f"  {model}: {info['description']}")
        print("\n[Specialized Models]")
        for model in ModelRegistry.get_specialized_models():
            info = ModelRegistry.get_model_info(model)
            print(f"  {model}: {info['description']}")
        print("=" * 50)
        sys.exit(0)

    if args.info:
        show_system_info()
        sys.exit(0)

    config = Config()
    apply_args_to_config(config, args)

    try:
        logger = setup_logger(
            "openeyes",
            level=logging.DEBUG if config.debug else logging.INFO,
            log_file=args.log_file,
            log_format=args.log_format,
        )

        if args.api:
            _start_api(args.api_host, args.api_port)
            print(f"REST API started at http://{args.api_host}:{args.api_port}")

        if args.video:
            print(f"Video mode: processing {args.video}")
            if args.output:
                print(f"Output video: {args.output}")

        system = VisionSystem(
            config, use_ros2=args.ros2, log_file=args.log_file,
            video_path=args.video, output_path=args.output,
        )

        if args.no_parallel:
            system._use_parallel = False
        if args.pose_every:
            system._pose_skip_frames = args.pose_every - 1
        if args.no_face:
            system._use_face = False
        if args.no_gesture:
            system._use_gesture = False
        if args.no_pose:
            system._use_pose = False
        if args.no_depth:
            system._use_depth = False

        if args.turbo:
            system._turbo_mode = True
            print("TURBO MODE: Aggressive frame skipping for max FPS")

        if args.template:
            from src.templates import TemplateManager
            tm = TemplateManager()
            template = tm.get_template(args.template)
            if template:
                print(f"Loading template: {template.name}")
                print(f"  {template.description}")
                system._use_face = template.face_enabled
                system._use_gesture = template.gesture_enabled
                system._use_pose = template.pose_enabled
                system._use_depth = template.depth_enabled
                system._use_tracking = template.tracking_enabled
                system._use_world_model = template.world_model_enabled
                config._config["models"]["yolo"]["confidence"] = template.confidence_threshold
                config._config["models"]["yolo"]["iou_threshold"] = template.iou_threshold
                config._config["depth"]["model"] = template.depth_model
                config._config["depth"]["enabled"] = template.depth_enabled
                config._config["depth"]["skip_frames"] = template.depth_skip_frames
                if template.world_model_enabled:
                    system._world_model_type = template.world_model_type
                    system._wm_horizon = template.plan_horizon
                    system._wm_samples = template.plan_samples
                if template.safety_enabled:
                    print(f"  Safety: max_vel={template.max_velocity}, min_dist={template.min_distance}")
                if template.classes_filter:
                    print(f"  Classes: {', '.join(template.classes_filter[:5])}...")
            else:
                print(f"Warning: Template '{args.template}' not found")
                print(f"Available: {', '.join(tm.list_templates())}")

        if args.no_monitoring and system._perf_monitor:
            system._perf_monitor.enabled = False

        if args.no_tracking:
            system._use_tracking = False
            system._tracker = None

        if args.follow:
            config._config["tracking"]["follow_enabled"] = True
            system._follow_target = True
            if not system._tracker:
                system._use_tracking = True
                from src.utils.tracker import ObjectTracker
                system._tracker = ObjectTracker(
                    max_age=args.track_max_age,
                    min_hits=3,
                    iou_threshold=0.3,
                )

        if args.visual_odom or args.slam:
            if VisualOdometry is not None:
                system._use_visual_odom = True
                system._visual_odom = VisualOdometry()
                config._config["ros2"]["enabled"] = True
                print("Visual odometry enabled")

        if args.depth_to_scan:
            config._config["ros2"]["enabled"] = True
            print("Depth to laser scan enabled")

        if args.nav2:
            config._config["ros2"]["enabled"] = True
            print("Nav2 integration enabled (obstacle avoidance)")

        if args.lidar:
            config._config["ros2"]["enabled"] = True
            print(f"LIDAR processing enabled (topic: {args.lidar_topic})")

        if args.realsense:
            config._config["ros2"]["enabled"] = True
            print("RealSense camera mode enabled (depth + IMU)")

        if args.int8:
            config._config["performance"]["tensorrt"]["precision"] = "int8"
            print("INT8 quantization enabled")

        if args.dla:
            config._config["performance"]["tensorrt"]["dla_enabled"] = True
            print("DLA offloading enabled")

        if args.diffusion_policy:
            from src.models.diffusion_policy import DiffusionPolicy
            print("Diffusion policy enabled")

        if args.action_chunking:
            from src.models.action_chunker import create_action_chunker
            print(f"Action chunking enabled ({args.control_freq} Hz)")

        if args.safety:
            from src.utils.safety_controller import create_safety_controller
            print(f"Safety controller enabled (max_vel={args.max_velocity}, min_dist={args.min_distance})")

        if args.health_monitor:
            from src.utils.health_monitor import create_health_monitor
            print("Health monitoring enabled for 24/7 operation")

        if args.ota_update:
            from src.utils.ota_update import create_ota_updater
            print("OTA update system enabled")

        if args.vla or args.advanced_ai:
            from src.models.vla import VLAModel
            if VLAModel is not None:
                system._use_vla = True
                config._config["ros2"]["enabled"] = True

                if args.real_vla:
                    from src.models.vla_models import create_vla_model
                    vla_model = create_vla_model(
                        model_type=args.real_vla,
                        device="cuda"
                    )
                    if vla_model and vla_model.is_loaded:
                        system._real_vla_model = vla_model
                        print(f"Real VLA enabled ({args.real_vla})")
                    else:
                        print(f"Failed to load {args.real_vla}, using rule-based VLA")
                else:
                    print(f"VLA enabled (mode: {'advanced-ai' if args.advanced_ai else 'basic'})")

        if args.world_model and args.world_model != "none":
            from src.world_model.lewm import LeWorldModel
            wm_device = "cuda" if platform.machine() == "aarch64" else "cpu"
            system._use_world_model = True
            system._wm_horizon = args.plan_horizon
            system._wm_samples = args.plan_samples
            system._prediction_fps = args.prediction_fps
            system._occlusion_frames = args.occlusion_frames
            system._safety_predict = args.safety_predict

            if args.world_model == "lewm":
                world_model = LeWorldModel(
                    device=wm_device,
                    precision="fp16",
                    latent_dim=384,
                    use_dinov2=False,
                )
                world_model.load()
                system._world_model = world_model
                logger.info(f"LeWorldModel loaded (device={wm_device}, latent_dim=384)")
            else:
                logger.warning(f"World model '{args.world_model}' not yet implemented")

        system.start()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
