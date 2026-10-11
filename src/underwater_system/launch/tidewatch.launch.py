"""ROS launch entry; the supervisor also works without a colcon installation."""
from pathlib import Path
import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share = Path(get_package_share_directory('underwater_system'))
    return LaunchDescription([
        DeclareLaunchArgument('domain',default_value='42'),
        ExecuteProcess(cmd=[sys.executable,str(share/'scripts/run_simulation.py'),
                            '--domain',LaunchConfiguration('domain')],output='screen'),
    ])
