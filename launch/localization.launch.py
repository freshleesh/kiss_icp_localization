"""KISS-ICP map-based localization (MID360 + IMU).

Usage:
    ros2 launch kiss_icp_localization localization.launch.py \
        map_pcd:=/abs/path/to/map.pcd

`map_pcd:=` is the absolute path to the map PCD and is loaded directly.

If `<map_pcd dir>/ground_lidar.yaml` exists (written by glim_map_pipeline.py), its
ground-crop params (crop_ground_mode/normal/offset, crop_z_min/max) are loaded
automatically so the localization input scan is cropped to the same z-band as the
map. Override the path with `ground_yaml:=`, or leave it absent to fall back to the
config's crop params.

`show_2d_map:=true` (default) also brings up nav2 map_server publishing the
2D occupancy grid (<map_pcd dir>/map_2d.yaml, e.g. GLIM's map_2d.pgm) on /map
as a lightweight RViz backdrop -- far cheaper to render than the 12M-pt cloud.
The server self-activates (no lifecycle_manager needed). Override the yaml with
map_2d:=/abs/path/to/foo.yaml.
"""
import os

import lifecycle_msgs.msg
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState


def launch_setup(context, *args, **kwargs):
    map_pcd = LaunchConfiguration("map_pcd").perform(context)
    share = get_package_share_directory("kiss_icp_localization")
    config = os.path.join(share, "config", "mid360_localization.yaml")

    # Auto-link the GLIM-derived ground crop: read ground_lidar.yaml from the
    # map's own folder (override with ground_yaml:=). It carries crop_ground_mode
    # + the LiDAR-frame ground plane (crop_ground_normal/offset, crop_z_min/max),
    # so the localization input is cropped to the same band as the map. Later
    # param files override earlier ones, so this wins over the config defaults.
    ground_yaml = LaunchConfiguration("ground_yaml").perform(context)
    if not ground_yaml:
        ground_yaml = os.path.join(os.path.dirname(map_pcd), "ground_lidar.yaml")
    params = [config]
    info = []
    if os.path.isfile(ground_yaml):
        params.append(ground_yaml)
        info.append(LogInfo(msg=f"[kiss_loc] ground crop auto-linked: {ground_yaml}"))
    else:
        info.append(LogInfo(
            msg=f"[kiss_loc] no ground_lidar.yaml at {ground_yaml} -> using config crop params"))
    params.append({"map_pcd_path": map_pcd})

    nodes = list(info) + [
        Node(
            package="kiss_icp_localization",
            executable="localization_node",
            name="kiss_icp_localization",
            output="screen",
            parameters=params,
        )
    ]

    if LaunchConfiguration("show_2d_map").perform(context).lower() == "true":
        map_2d = LaunchConfiguration("map_2d").perform(context)
        if not map_2d:
            map_2d = os.path.join(os.path.dirname(map_pcd), "map_2d.yaml")
        map_server = LifecycleNode(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            namespace="",
            output="screen",
            parameters=[{"yaml_filename": map_2d, "frame_id": "map"}],
        )
        # self-activate without nav2_lifecycle_manager:
        # configure on launch, then activate once it reaches 'inactive'.
        configure = EmitEvent(
            event=ChangeState(
                lifecycle_node_matcher=matches_action(map_server),
                transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
            )
        )
        activate = RegisterEventHandler(
            OnStateTransition(
                target_lifecycle_node=map_server,
                goal_state="inactive",
                entities=[
                    EmitEvent(
                        event=ChangeState(
                            lifecycle_node_matcher=matches_action(map_server),
                            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                        )
                    )
                ],
            )
        )
        nodes += [map_server, activate, configure]

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
            # ICP's per-iteration parallel region is small; many cores hurt
            # (fork/join overhead). Cap threads unless the user already set it.
            SetEnvironmentVariable(
                "OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "8")
            ),
            DeclareLaunchArgument(
                "map_pcd",
                default_value="/home/leesh/ros2_ws/maps_glim/ifac_1/map.pcd",
                description="Absolute path to the map PCD to load.",
            ),
            DeclareLaunchArgument(
                "show_2d_map",
                default_value="true",
                description="Bring up nav2 map_server for the 2D occupancy grid.",
            ),
            DeclareLaunchArgument(
                "map_2d",
                default_value="",
                description="2D map yaml. If empty, uses <map_pcd dir>/map_2d.yaml.",
            ),
            DeclareLaunchArgument(
                "ground_yaml",
                default_value="",
                description="Ground-crop param yaml. If empty, uses "
                "<map_pcd dir>/ground_lidar.yaml (GLIM output); skipped if absent.",
            ),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            OpaqueFunction(function=launch_setup),
        ]
    )
