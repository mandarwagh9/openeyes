#!/usr/bin/env python3
"""OpenEyes Vision System ROS2 Launch File

Launch file for starting OpenEyes with ROS2 integration on Jetson Orin Nano.

Usage:
    ros2 launch openeyes openeyes.launch.py

With parameters:
    ros2 launch openeyes openeyes.launch.py camera:=0 debug:=true
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess


def generate_launch_description():
    """Generate the launch description for OpenEyes vision system."""

    project_root = os.path.dirname(os.path.dirname(__file__))
    main_py = os.path.join(project_root, 'src/main.py')

    cmd = [
        'python3', main_py,
        '--camera', '0',
        '--width', '1280',
        '--height', '720',
        '--fps', '30',
        '--ros2',
        '--debug',
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera',
            default_value='0',
            description='Camera device number'
        ),
        DeclareLaunchArgument(
            'width',
            default_value='1280',
            description='Camera width'
        ),
        DeclareLaunchArgument(
            'height',
            default_value='720',
            description='Camera height'
        ),
        DeclareLaunchArgument(
            'fps',
            default_value='30',
            description='Target FPS'
        ),
        DeclareLaunchArgument(
            'debug',
            default_value='false',
            description='Enable debug mode'
        ),
        DeclareLaunchArgument(
            'ros2',
            default_value='true',
            description='Enable ROS2 publishing'
        ),
        ExecuteProcess(
            cmd=cmd,
            output='screen',
            shell=False,
        ),
    ])
