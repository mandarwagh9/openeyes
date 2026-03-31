#!/usr/bin/env python3
"""Unified launch file for OpenEyes - Vision + SLAM + Navigation.

This launch file starts all components needed for autonomous navigation:
1. OpenEyes vision system (detection, depth, tracking, VLA)
2. Isaac cuVSLAM for visual odometry and mapping
3. Nav2 navigation stack
4. Vision-based obstacle avoidance
5. Navigation goal client

Usage:
    # Full autonomous navigation
    ros2 launch openeyes unified.launch.py

    # With visualization
    ros2 launch openeyes unified.launch.py rviz:=true

    # With manual teleop
    ros2 launch openeyes unified.launch.py teleop:=true
"""

import os
from typing import List, Optional

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, 
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    RegisterLaunch,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackage


def generate_launch_description() -> LaunchDescription:
    """Generate unified launch description."""
    
    pkg_name = "openeyes"
    
    declare_teleop = DeclareLaunchArgument(
        "teleop",
        default_value="false",
        description="Enable keyboard teleop for manual control"
    )
    
    declare_rviz = DeclareLaunchArgument(
        "rviz",
        default_value="false",
        description="Launch RViz for visualization"
    )
    
    declare_map = DeclareLaunchArgument(
        "map",
        default_value="",
        description="Path to pre-built map (empty for SLAM)"
    )
    
    declare_use_realsense = DeclareLaunchArgument(
        "use_realsense",
        default_value="true",
        description="Use RealSense camera for SLAM"
    )
    
    teleop = LaunchConfiguration("teleop")
    rviz = LaunchConfiguration("rviz")
    map_file = LaunchConfiguration("map")
    use_realsense = LaunchConfiguration("use_realsense")
    
    launch_nodes = []
    
    if use_realsense:
        launch_nodes.append(
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
                }],
                output="screen",
                arguments=["--ros-args", "--log-level", "warn"],
            )
        )
    
    launch_nodes.extend([
        Node(
            package=pkg_name,
            executable="openeyes_vision",
            name="openeyes_vision",
            parameters=[{
                "ros2": True,
                "tracking": True,
                "follow": False,
            }],
            output="screen",
            arguments=["--ros-args", "--log-level", "info"],
        ),
        
        Node(
            package="isaac_ros_visual_slam",
            executable="isaac_ros_visual_slam_node",
            name="visual_slam",
            parameters=[{
                "base_frame": "base_link",
                "odom_frame": "odom",
                "map_frame": "map",
                "enable_localization_n_mapping": True,
                "publish_map_to_odom_tf": True,
                "publish_odom_to_base_tf": True,
                "tracking_mode": 1,
            }],
            output="screen",
            arguments=["--ros-args", "--log-level", "warn"],
            remappings=[
                ("/camera/left/image_raw", "/camera/color/image_raw"),
                ("/camera/right/image_raw", "/camera/aligned_depth_to_color/image_raw"),
            ],
        ),
        
        Node(
            package=pkg_name,
            executable="depth_to_laserscan",
            name="depth_to_laserscan",
            parameters=[{
                "depth_topic": "/vision/depth",
                "scan_topic": "/scan",
                "range_min": 0.1,
                "range_max": 5.0,
            }],
            output="screen",
        ),
        
        Node(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            output="screen",
            parameters=[
                PathJoinSubstitution([
                    FindPackage(pkg_name),
                    "config",
                    "nav2_params.yaml"
                ]),
            ],
        ),
        
        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            output="screen",
            parameters=[
                PathJoinSubstitution([
                    FindPackage(pkg_name),
                    "config",
                    "nav2_params.yaml"
                ]),
            ],
        ),
        
        Node(
            package="nav2_behaviors",
            executable="behavior_server",
            name="behavior_server",
            output="screen",
            parameters=[
                PathJoinSubstitution([
                    FindPackage(pkg_name),
                    "config",
                    "nav2_params.yaml"
                ]),
            ],
        ),
        
        Node(
            package="nav2_bt_navigator",
            executable="bt_navigator",
            name="bt_navigator",
            output="screen",
            parameters=[
                PathJoinSubstitution([
                    FindPackage(pkg_name),
                    "config",
                    "nav2_params.yaml"
                ]),
            ],
        ),
        
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            output="screen",
            parameters=[{
                "use_sim_time": False,
                "autostart": True,
                "node_names": [
                    "planner_server",
                    "controller_server",
                    "behavior_server",
                    "bt_navigator",
                ],
            }],
        ),
        
        Node(
            package=pkg_name,
            executable="vision_obstacle_avoidance",
            name="vision_obstacle_avoidance",
            parameters=[{
                "detection_topic": "/vision/detections",
                "cmd_vel_in_topic": "/diffbot_controller_cmd_vel",
                "cmd_vel_out_topic": "/cmd_vel",
                "stop_distance": 0.5,
                "slowdown_distance": 1.0,
            }],
            output="screen",
        ),
        
        Node(
            package=pkg_name,
            executable="navigation_goal",
            name="navigation_goal",
            output="screen",
        ),
        
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            parameters=[{
                "robot_description": "",
                "use_sim_time": False,
            }],
            output="screen",
        ),
    ])
    
    if teleop:
        launch_nodes.append(
            Node(
                package="teleop_twist_keyboard",
                executable="teleop_twist_keyboard",
                name="teleop_twist_keyboard",
                output="screen",
                remappings=[
                    ("/cmd_vel", "/nav2_cmd_vel"),
                ],
            )
        )
    
    if rviz:
        launch_nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("nav2_bringup"),
                        "launch",
                        "rviz_launch.py"
                    )
                ),
                launch_arguments={"use_sim_time": "false"}.items(),
            )
        )
    
    return LaunchDescription([
        SetEnvironmentVariable("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] [{name}]: {message}"),
        SetEnvironmentVariable("TURTLEBOT3_MODEL", "burger"),
        declare_teleop,
        declare_rviz,
        declare_map,
        declare_use_realsense,
        *launch_nodes,
    ])


def main():
    """Main entry point."""
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
