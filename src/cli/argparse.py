import argparse
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="OpenEyes Vision System")

    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera source index",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Output host IP",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Output port",
    )
    parser.add_argument(
        "--no-face",
        action="store_true",
        help="Disable face detection",
    )
    parser.add_argument(
        "--no-gesture",
        action="store_true",
        help="Disable gesture recognition",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable pose estimation",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth estimation",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--deepstream",
        action="store_true",
        help="Use DeepStream pipeline for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="TensorRT precision (fp32, fp16, int8)",
    )
    parser.add_argument(
        "--dla",
        action="store_true",
        help="Use DLA for inference (Jetson only)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n",
        help="Model to use (yolo11n, yolo12n, rtmdet_nano, grasp_detector)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "--vla",
        action="store_true",
        help="Enable VLA (Vision-Language-Action) for intelligent control",
    )
    parser.add_argument(
        "--real-vla",
        type=str,
        default="",
        choices=["smolvla", "openvla", "octo", ""],
        help="Use real VLA model instead of rule-based (smolvla, openvla, octo)",
    )
    parser.add_argument(
        "--event-camera",
        action="store_true",
        help="Enable event camera processing",
    )
    parser.add_argument(
        "--advanced-ai",
        action="store_true",
        help="Enable all advanced AI features (VLA + event camera)",
    )
    parser.add_argument(
        "--no-monitoring",
        action="store_true",
        help="Disable performance monitoring",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable object tracking",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Enable person following (requires tracking)",
    )
    parser.add_argument(
        "--track-max-age",
        type=int,
        default=30,
        help="Max frames to keep lost track (default: 30)",
    )
    parser.add_argument(
        "--pose-every",
        type=int,
        default=2,
        help="Run pose estimation every N frames",
    )
    parser.add_argument(
        "--ros2",
        action="store_true",
        help="Enable ROS2 publishing (requires ros-humble-vision-msgs)",
    )
    parser.add_argument(
        "--ros2-qos",
        type=str,
        default="default",
        choices=["default", "sensor", "command", "best_effort", "reliable"],
        help="ROS2 QoS profile (default, sensor, command, best_effort, reliable)",
    )
    parser.add_argument(
        "--ros2-actions",
        action="store_true",
        help="Enable ROS2 action server for robot control",
    )
    parser.add_argument(
        "--multi-camera",
        type=int,
        nargs="+",
        default=None,
        help="Enable multi-camera mode (list camera indices)",
    )
    parser.add_argument(
        "--ros2-time-sync",
        action="store_true",
        help="Use ROS2 time for synchronization",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="OpenEyes v2.5.0",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show system information and OpenEyes recommendations",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (with rotation, 5MB max by default)",
    )
    parser.add_argument(
        "--slam",
        action="store_true",
        help="Enable SLAM (visual odometry for Nav2 integration)",
    )
    parser.add_argument(
        "--visual-odom",
        action="store_true",
        help="Enable visual odometry publisher (/odom topic)",
    )
    parser.add_argument(
        "--depth-to-scan",
        action="store_true",
        help="Convert depth to laser scan for Nav2",
    )
    parser.add_argument(
        "--nav2",
        action="store_true",
        help="Enable Nav2 integration with obstacle avoidance",
    )
    parser.add_argument(
        "--lidar",
        action="store_true",
        help="Enable LIDAR processing for obstacle detection",
    )
    parser.add_argument(
        "--lidar-topic",
        type=str,
        default="/scan",
        help="LIDAR scan topic (default: /scan)",
    )
    parser.add_argument(
        "--realsense",
        action="store_true",
        help="Use RealSense camera (depth + IMU) for SLAM",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="midas-small",
        choices=["da3-small", "da3-base", "midas-small"],
        help="Depth estimation model (default: midas-small)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Industry template (warehouse, manufacturing-qa, agriculture, retail)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 quantized models for faster inference",
    )
    parser.add_argument(
        "--diffusion-policy",
        action="store_true",
        help="Enable diffusion policy for manipulation",
    )
    parser.add_argument(
        "--action-chunking",
        action="store_true",
        help="Enable action chunking for smooth control",
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=20,
        help="Control frequency in Hz (default: 20)",
    )
    parser.add_argument(
        "--safety",
        action="store_true",
        help="Enable safety controller (emergency stop, speed limits)",
    )
    parser.add_argument(
        "--health-monitor",
        action="store_true",
        help="Enable health monitoring for 24/7 operation",
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=0.5,
        help="Maximum velocity in m/s (default: 0.5)",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.3,
        help="Minimum safe distance to obstacles in m (default: 0.3)",
    )
    parser.add_argument(
        "--ota-update",
        action="store_true",
        help="Check for OTA model updates",
    )

    parser.add_argument(
        "--world-model",
        type=str,
        default="none",
        choices=["none", "lewm", "vjepa2"],
        help="World model to use (none/lewm/vjepa2)",
    )
    parser.add_argument(
        "--plan-horizon",
        type=int,
        default=10,
        help="Planning horizon in steps",
    )
    parser.add_argument(
        "--plan-samples",
        type=int,
        default=100,
        help="CEM sample count for planning",
    )
    parser.add_argument(
        "--prediction-fps",
        type=int,
        default=30,
        help="Prediction update rate",
    )
    parser.add_argument(
        "--occlusion-frames",
        type=int,
        default=5,
        help="Max frames to predict through occlusion",
    )
    parser.add_argument(
        "--safety-predict",
        action="store_true",
        help="Enable predictive safety evaluation",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Turbo mode: aggressive frame skipping for max FPS (depth=16, face/gesture/pose=12)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file (mp4, avi, mkv) instead of live camera",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output video file for annotated demo recording",
    )
    parser.add_argument(
        "--log-format",
        type=str,
        default="console",
        choices=["console", "json"],
        help="Log format: console (human-readable) or json (structured)",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Enable REST API server",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="REST API server port (default: 8000)",
    )
    parser.add_argument(
        "--api-host",
        type=str,
        default="127.0.0.1",
        help="REST API server host (default: 127.0.0.1)",
    )

    return parser


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    return create_parser().parse_args()


def apply_args_to_config(config: "Config", args: argparse.Namespace) -> None:
    """Apply parsed CLI arguments to Config object.
    
    Args:
        config: Config object to modify
        args: Parsed command line arguments
    """
    from src.utils.config import Config
    
    if args.camera is not None:
        config._config["camera"]["source"] = args.camera
    if args.debug:
        config._config["debug"] = True
    if args.config:
        config._config_path = args.config
    if args.width:
        config._config["camera"]["width"] = args.width
    if args.height:
        config._config["camera"]["height"] = args.height
    if args.fps:
        config._config["camera"]["fps"] = args.fps

    if args.batch_size > 1:
        config._config["performance"]["batch_inference"]["enabled"] = True
        config._config["performance"]["batch_inference"]["batch_size"] = args.batch_size

    if args.precision:
        config._config["performance"]["tensorrt"]["precision"] = args.precision

    if args.dla:
        config._config["performance"]["tensorrt"]["dla_enabled"] = True
