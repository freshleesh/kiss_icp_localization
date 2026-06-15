"""KISS-ICP map-based localization (MID360 + IMU).

Usage:
    ros2 launch kiss_icp_localization localization.launch.py map:=hall_0609

`map:=` is the map directory name under the maps root; the node loads
<maps_root>/<map>/cloudGlobal.pcd. Override the root with the
KISS_LOC_MAPS_ROOT env var (falls back to FASTLIVO_MAPS_ROOT, then the
in-repo stack_master/maps path).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DEFAULT_MAPS_ROOT = os.environ.get(
    "KISS_LOC_MAPS_ROOT",
    os.environ.get(
        "FASTLIVO_MAPS_ROOT",
        "/Users/mini/ros2_ws/src/IFAC2026/src/system/stack_master/maps",
    ),
)


def launch_setup(context, *args, **kwargs):
    map_name = LaunchConfiguration("map").perform(context)
    map_pcd = os.path.join(DEFAULT_MAPS_ROOT, map_name, "cloudGlobal.pcd")
    share = get_package_share_directory("kiss_icp_localization")
    config = os.path.join(share, "config", "mid360_localization.yaml")
    use_custom_msg = (
        LaunchConfiguration("use_custom_msg").perform(context).lower() == "true"
    )
    nodes = [
        Node(
            package="kiss_icp_localization",
            executable="localization_node",
            name="kiss_icp_localization",
            output="screen",
            parameters=[
                config,
                {"map_pcd_path": map_pcd, "use_custom_msg": use_custom_msg},
            ],
        )
    ]
    if LaunchConfiguration("use_rviz").perform(context).lower() == "true":
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", os.path.join(share, "config", "kiss_loc.rviz")],
                output="log",
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("map", default_value="hall_0609"),
            # 라이브 센서(livox CustomMsg)는 true, bag 재생(PointCloud2)은 false
            DeclareLaunchArgument("use_custom_msg", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            OpaqueFunction(function=launch_setup),
        ]
    )
