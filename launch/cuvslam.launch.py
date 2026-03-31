#!/usr/bin/env python3
"""Isaac ROS cuVSLAM launch configuration for OpenEyes.

This launch file integrates NVIDIA Isaac ROS Visual SLAM (cuVSLAM) with OpenEyes.
cuVSLAM provides GPU-accelerated visual odometry and SLAM for robotics.

Hardware Requirements:
- NVIDIA Jetson Orin (Nano, NX, or AGX)
- Stereo camera (RealSense D435i recommended) or RGBD camera

Installation:
    sudo apt-get update
    sudo apt-get install -y ros-jazzy-isaac-ros-visual-slam
    sudo apt-get install -y ros-jazzy-realsense2-camera
    sudo apt-get install -y ros-jazzy-image-pipeline

Usage:
    ros2 launch openeyes cuvslam.launch.py
    ros2 launch openeyes cuvslam.launch.py camera:=zed
"""

import os
import sys
from typing import Optional

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def get_isaac_ros_package_path(package_name: str) -> Optional[str]:
    """Get the path to an Isaac ROS package."""
    try:
        return get_package_share_directory(package_name)
    except Exception:
        return None


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for Isaac ROS cuVSLAM."""
    
    pkg_name = "openeyes"
    
    declare_use_realsense = DeclareLaunchArgument(
        "use_realsense",
        default_value="true",
        description="Use RealSense camera for SLAM"
    )
    
    declare_camera = DeclareLaunchArgument(
        "camera",
        default_value="realsense",
        description="Camera type: realsense, zed, or usb"
    )
    
    declare_base_frame = DeclareLaunchArgument(
        "base_frame",
        default_value="base_link",
        description="Robot base frame ID"
    )
    
    declare_odom_frame = DeclareLaunchArgument(
        "odom_frame",
        default_value="odom",
        description="Odometry frame ID"
    )
    
    declare_map_frame = DeclareLaunchArgument(
        "map_frame",
        default_value="map",
        description="Map frame ID"
    )
    
    declare_slam_mode = DeclareLaunchArgument(
        "slam_mode",
        default_value="localization",
        description="SLAM mode: mapping, localization, or slam"
    )
    
    declare_use_imu = DeclareLaunchArgument(
        "use_imu",
        default_value="true",
        description="Use IMU for visual-inertial odometry"
    )
    
    use_realsense = LaunchConfiguration("use_realsense")
    camera = LaunchConfiguration("camera")
    base_frame = LaunchConfiguration("base_frame")
    odom_frame = LaunchConfiguration("odom_frame")
    map_frame = LaunchConfiguration("map_frame")
    slam_mode = LaunchConfiguration("slam_mode")
    use_imu = LaunchConfiguration("use_imu")
    
    cuvslam_params = {
        "base_frame": base_frame,
        "odom_frame": odom_frame,
        "map_frame": map_frame,
        "enable_localization_n_mapping": True,
        "publish_map_to_odom_tf": True,
        "publish_odom_to_base_tf": True,
        "tracking_mode": 1 if use_imu else 0,
        "depth_scale_factor": 1000.0,
        "enable_image_denoising": False,
        "num_cameras": 2,
        "slam_max_map_size": 10000,
    }
    
    nodes = []
    
    if use_realsense:
        nodes.append(
            Node(
                package="realsense2_camera",
                executable="realsense2_camera_node",
                name="realsense2_camera",
                parameters=[{
                    "enable_depth": True,
                    "depth_module.profile": "640x480x30",
                    "rgb_camera.profile": "640x480x30",
                    "enable_sync": True,
                    "sync_frames": True,
                    "enable_gyro": True,
                    "enable_accel": True,
                    "gyro_fps": 200,
                    "accel_fps": 200,
                }],
                output="screen",
                arguments=["--ros-args", "--log-level", "warn"],
            )
        )
        
        nodes.append(
            Node(
                package="isaac_ros_visual_slam",
                executable="isaac_ros_visual_slam_node",
                name="visual_slam",
                parameters=[cuvslam_params],
                output="screen",
                arguments=["--ros-args", "--log-level", "info"],
                remappings=[
                    ("/camera/left/image_raw", "/camera/color/image_raw"),
                    ("/camera/right/image_raw", "/camera/aligned_depth_to_color/image_raw"),
                    ("/camera/left/camera_info", "/camera/color/camera_info"),
                    ("/camera/right/camera_info", "/camera/aligned_depth_to_color/camera_info"),
                ],
            )
        )
    else:
        nodes.append(
            Node(
                package="isaac_ros_visual_slam",
                executable="isaac_ros_visual_slam_node",
                name="visual_slam",
                parameters=[cuvslam_params],
                output="screen",
                arguments=["--ros-args", "--log-level", "info"],
            )
        )
    
    nodes.extend([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            parameters=[{"robot_description": ""}],
            output="screen",
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_tf_publisher",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0.1",
                "--roll", "0", "--pitch", "0", "--yaw", "0",
                "--frame-id", "camera_link",
                "--child-frame-id", "base_link",
            ],
            output="screen",
        ),
    ])
    
    return LaunchDescription([
        declare_use_realsense,
        declare_camera,
        declare_base_frame,
        declare_odom_frame,
        declare_map_frame,
        declare_slam_mode,
        declare_use_imu,
        *nodes,
    ])


def main():
    """Main entry point for standalone testing."""
    from launch import LaunchService
    
    ld = generate_launch_description()
    ls = LaunchService()
    ls.include_launch_description(ld)
    
    try:
        ls.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
