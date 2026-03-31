#!/usr/bin/env python3
"""Nav2 launch configuration for OpenEyes.

This launch file integrates ROS2 Navigation2 stack with OpenEyes vision system.
Nav2 provides path planning, obstacle avoidance, and navigation goals.

Installation:
    sudo apt-get update
    sudo apt-get install -y ros-jazzy-navigation2
    sudo apt-get install -y ros-jazzy-nav2-bringup
    sudo apt-get install -y ros-jazzy-ros2-control

Usage:
    ros2 launch openeyes nav2.launch.py
    ros2 launch openeyes nav2.launch.py map:=/path/to/map.yaml
"""

import os
from typing import Optional, Dict, Any

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def get_nav2_package_path() -> Optional[str]:
    """Get the path to Nav2 package."""
    try:
        return get_package_share_directory("nav2_bringup")
    except Exception:
        return None


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for Nav2 navigation."""
    
    pkg_name = "openeyes"
    nav2_pkg = "nav2_bringup"
    
    declare_map_yaml = DeclareLaunchArgument(
        "map",
        default_value="",
        description="Path to map YAML file (empty for SLAM)"
    )
    
    declare_params_file = DeclareLaunchArgument(
        "params_file",
        default_value="",
        description="Path to Nav2 params file"
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
    
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation time"
    )
    
    declare_autostart = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        description="Autostart Nav2"
    )
    
    map_yaml = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    base_frame = LaunchConfiguration("base_frame")
    odom_frame = LaunchConfiguration("odom_frame")
    map_frame = LaunchConfiguration("map_frame")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    
    default_params_file = os.path.join(
        os.path.dirname(__file__), "..", "config", "nav2_params.yaml"
    )
    
    params_file_path = params_file if params_file else default_params_file
    
    nav2_params: Dict[str, Any] = {
        "use_sim_time": use_sim_time,
        "autostart": autostart,
        "map_yaml_file": map_yaml,
        "topic_to_pose_goal": "/goal_pose",
        "bt_xml_filename": "navigate_w_replanning_and_recovery.xml",
        "plugin_lib_names": [
            "nav2_compute_path_to_pose_action",
            "nav2_follow_path_action",
            "nav2_back_up_action",
            "nav2_spin_action",
            "nav2_wait_action",
            "nav2_clear_costmap_service",
            "nav2_global_costmap_client",
            "nav2_local_costmap_client",
        ],
    }
    
    controller_params = {
        "use_sim_time": use_sim_time,
        "controller_server": {
            "ros__parameters": {
                "use_sim_time": use_sim_time,
                "FollowPath": {
                    "plugin": "nav2_controller::DwbController",
                    "ros__parameters": {
                        "use_sim_time": use_sim_time,
                        "critics": ["FollowPath", "GoalAdapter", "ObstacleFootprint", "ObstacleCostmap"],
                        "GoalAdapter": {
                            "plugin": "nav2_controller::GoalChecker",
                            "ros__parameters": {
                                "use_sim_time": use_sim_time,
                                "xy_goal_tolerance": 0.25,
                                "yaw_goal_tolerance": 0.25,
                                "check_time_allowance": 10.0,
                            }
                        },
                        "FollowPath": {
                            "ros__parameters": {
                                "use_sim_time": use_sim_time,
                                "max_vel_x": 0.5,
                                "max_vel_y": 0.0,
                                "max_vel_theta": 1.0,
                                "min_vel_x": 0.05,
                                "min_vel_theta": -1.0,
                                "frequent_cmd_timeout": 0.5,
                                "vx_samples": 20,
                                "vtheta_samples": 20,
                            }
                        },
                    }
                },
            }
        },
        "controller_server_rclcpp_node": {
            "ros__parameters": {
                "use_sim_time": use_sim_time,
            }
        },
    }
    
    nodes = [
        Node(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            output="screen",
            parameters=[params_file_path, controller_params],
        ),
        Node(
            package="nav2_controller",
            executable="smoother_server",
            name="smoother_server",
            output="screen",
            parameters=[params_file_path],
        ),
        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            output="screen",
            parameters=[params_file_path],
        ),
        Node(
            package="nav2_behaviors",
            executable="behavior_server",
            name="behavior_server",
            output="screen",
            parameters=[params_file_path],
        ),
        Node(
            package="nav2_bt_navigator",
            executable="bt_navigator",
            name="bt_navigator",
            output="screen",
            parameters=[params_file_path],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "autostart": autostart,
                "node_names": [
                    "planner_server",
                    "controller_server",
                    "smoother_server",
                    "behavior_server",
                    "bt_navigator",
                ],
            }],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ]
    
    return LaunchDescription([
        SetEnvironmentVariable("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] [{name}]: {message}"),
        declare_map_yaml,
        declare_params_file,
        declare_base_frame,
        declare_odom_frame,
        declare_map_frame,
        declare_use_sim_time,
        declare_autostart,
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
