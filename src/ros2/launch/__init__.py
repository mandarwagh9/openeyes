from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer
from pathlib import Path
import os


def generate_launch_description():
    pkg_dir = Path(__file__).parent.parent.parent
    config_dir = pkg_dir / "src" / "ros2" / "config"

    declare_output_topic = DeclareLaunchArgument(
        "output_topic",
        default_value="/vision/detections",
        description="Topic for publishing detections"
    )

    declare_camera_topic = DeclareLaunchArgument(
        "camera_topic",
        default_value="/camera/image_raw",
        description="Camera image topic"
    )

    declare_model_path = DeclareLaunchArgument(
        "model_path",
        default_value=str(pkg_dir / "models" / "yolo11n.onnx"),
        description="Path to YOLO model"
    )

    declare_confidence = DeclareLaunchArgument(
        "confidence_threshold",
        default_value="0.5",
        description="Confidence threshold for detections"
    )

    declare_device = DeclareLaunchArgument(
        "device",
        default_value="cuda",
        description="Device for inference (cuda/cpu)"
    )

    vision_publisher_node = Node(
        package="openeyes",
        executable="src.ros2.vision_node",
        name="vision_publisher",
        parameters=[{
            "output_topic": LaunchConfiguration("output_topic"),
            "camera_topic": LaunchConfiguration("camera_topic"),
            "model_path": LaunchConfiguration("model_path"),
            "confidence_threshold": LaunchConfiguration("confidence_threshold"),
            "device": LaunchConfiguration("device"),
            "frame_id": "camera_link",
            "enable_debug_image": True
        }],
        output="screen",
        emulate_tty=True
    )

    vision_control_node = Node(
        package="openeyes",
        executable="src.ros2.vision_node",
        name="vision_control",
        parameters=[{
            "target_class": "person",
            "follow_distance": 1.5,
            "max_linear_speed": 0.3,
            "max_angular_speed": 0.8,
            "detection_topic": LaunchConfiguration("output_topic")
        }],
        output="screen",
        emulate_tty=True,
        condition=None
    )

    vision_service_node = Node(
        package="openeyes",
        executable="src.ros2.services",
        name="vision_service",
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "confidence_threshold": LaunchConfiguration("confidence_threshold"),
            "device": LaunchConfiguration("device")
        }],
        output="screen",
        emulate_tty=True
    )

    return LaunchDescription([
        SetEnvironmentVariable("OPENCV_VIDEOIO_DEBUG", "1"),
        declare_output_topic,
        declare_camera_topic,
        declare_model_path,
        declare_confidence,
        declare_device,
        vision_publisher_node,
        vision_control_node,
        vision_service_node
    ])


def generate_vision_pipeline_launch():
    pkg_dir = Path(__file__).parent.parent.parent

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=str(pkg_dir / "src" / "ros2" / "config" / "vision_params.yaml"),
            description="Path to configuration YAML"
        ),

        Node(
            package="openeyes",
            executable="src.main",
            name="openeyes_vision",
            parameters=[
                str(pkg_dir / "config.yaml"),
                {"device": "cuda"}
            ],
            output="screen"
        ),

        Node(
            package="openeyes",
            executable="src.ros2.vision_node",
            name="vision_publisher",
            output="screen"
        ),

        Node(
            package="openeyes",
            executable="src.ros2.services",
            name="vision_service",
            output="screen"
        )
    ])
