#!/usr/bin/env python3
"""Run the whole simulation from a checkout without colcon or root access."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='Gazebo server without GUI')
    parser.add_argument('--rviz', action='store_true')
    parser.add_argument('--no-dashboard', action='store_true')
    parser.add_argument('--no-mqtt', action='store_true')
    parser.add_argument('--duration', type=float, help='Stop after this many wall seconds')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--mode', choices=('protect', 'explore'), help='Initial mission mode')
    parser.add_argument('--hydrophone-recording', help='Replay a local PCM16 hydrophone WAV')
    parser.add_argument('--domain', type=int, default=42)
    parser.add_argument('--partition', default='tidewatch_sim')
    parser.add_argument('--config')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root/'src/underwater_system'
    if not source.is_dir():
        source = root
    sys.path.insert(0, str(source))
    runtime = root/'.runtime'
    runtime.mkdir(exist_ok=True)
    pid_file = runtime/'run.pid'
    if pid_file.is_file():
        try:
            os.kill(int(pid_file.read_text()), 0)
        except (ProcessLookupError, ValueError):
            pass
        else:
            parser.error('TIDEWATCH is already running. Use scripts/stop_simulation.py first.')
    for binary in ('ros2','gz'):
        if not shutil.which(binary):
            parser.error(f'{binary} is missing. Source /opt/ros/<distribution>/setup.bash first.')
    config_path = Path(args.config) if args.config else root/'config/mission.yaml'
    config = yaml.safe_load(config_path.read_text())
    if args.seed is not None:
        config['seed'] = args.seed
    if args.mode is not None:
        config['mission_mode'] = args.mode
    if args.hydrophone_recording:
        config.setdefault('perception', {})['hydrophone_recording'] = str(Path(args.hydrophone_recording).expanduser().resolve())
    from underwater_system.mission import MissionPlanner
    from underwater_system.recordings import HydrophoneRecording
    try:
        for index in range(3):
            MissionPlanner(index, config)
        perception = config.get('perception', {})
        recording_path = perception.get('hydrophone_recording')
        if recording_path:
            recording = HydrophoneRecording(recording_path, perception.get('sample_rate_hz',8000),
                                             loop=perception.get('hydrophone_recording_loop',False))
            config['perception']['hydrophone_recording'] = str(recording.path.resolve())
    except (ValueError, OSError) as error:
        parser.error(str(error))
    config['world_path'] = str(source/'worlds/tidewatch.sdf')
    resolved = runtime/'mission.yaml'
    resolved.write_text(yaml.safe_dump(config))
    state,control = runtime/'state.json',runtime/'control.json'
    for path in (state,control):
        path.unlink(missing_ok=True)
    results = root/'results/live.jsonl'
    results.parent.mkdir(exist_ok=True)
    results.write_text('')
    env = dict(os.environ)
    env['GZ_PARTITION'] = args.partition
    env['ROS_DOMAIN_ID'] = str(args.domain)
    env['PYTHONPATH'] = str(source)+os.pathsep+env.get('PYTHONPATH','')
    env['PYTHONUNBUFFERED'] = '1'
    bridge = [
        {'ros_topic_name':'/clock','gz_topic_name':'/world/tidewatch/clock',
         'ros_type_name':'rosgraph_msgs/msg/Clock','gz_type_name':'gz.msgs.Clock','direction':'GZ_TO_ROS'},
        {'ros_topic_name':'/tidewatch/ground_truth','gz_topic_name':'/tidewatch/ground_truth',
         'ros_type_name':'tf2_msgs/msg/TFMessage','gz_type_name':'gz.msgs.Pose_V','direction':'GZ_TO_ROS'},
        {'ros_topic_name':'/ocean_current','gz_topic_name':'/ocean_current',
         'ros_type_name':'geometry_msgs/msg/Vector3','gz_type_name':'gz.msgs.Vector3d','direction':'ROS_TO_GZ'},
    ]
    for name in ('diver','intruder','marine_life','large_marine_life'):
        bridge.append({'ros_topic_name':f'/model/{name}/cmd_vel','gz_topic_name':f'/model/{name}/cmd_vel',
                       'ros_type_name':'geometry_msgs/msg/Twist','gz_type_name':'gz.msgs.Twist','direction':'ROS_TO_GZ'})
    for unit in range(3):
        for thruster in range(8):
            topic = f'/model/auv_{unit}/joint/thruster_{thruster}_joint/cmd_thrust'
            bridge.append({'ros_topic_name':topic,'gz_topic_name':topic,
                           'ros_type_name':'std_msgs/msg/Float64','gz_type_name':'gz.msgs.Double','direction':'ROS_TO_GZ'})
    bridge_config = runtime/'bridge.yaml'
    bridge_config.write_text(yaml.safe_dump(bridge))
    children,files = [],[]
    pid_file.write_text(str(os.getpid()))

    def spawn(name, command, child_env=env):
        log = (runtime/f'{name}.log').open('w')
        files.append(log)
        process = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=root)
        children.append((name,process))
        return process

    stopping = False
    def stop(signum=None, frame=None):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    try:
        if not args.no_mqtt:
            executable = shutil.which('mosquitto')
            local = root/'.tools/mqtt/usr/sbin/mosquitto'
            if not executable and local.is_file():
                executable = str(local)
            if executable:
                broker_config = runtime/'mosquitto.conf'
                broker_config.write_text(f"listener {config['mqtt']['port']} 127.0.0.1\nallow_anonymous true\npersistence false\n")
                broker_env = dict(env)
                broker_env['LD_LIBRARY_PATH'] = str(root/'.tools/mqtt/usr/lib/x86_64-linux-gnu')+os.pathsep+env.get('LD_LIBRARY_PATH','')
                spawn('mqtt',[executable,'-c',str(broker_config)],broker_env)
            else:
                print('MQTT broker missing; telemetry stays local. Optional: python3 scripts/bootstrap_mqtt.py',flush=True)
        gz = ['gz','sim','-r','-v','2']
        if args.headless:
            gz.append('-s')
        gz.append(config['world_path'])
        spawn('gazebo',gz)
        spawn('bridge',['ros2','run','ros_gz_bridge','parameter_bridge',
                        '/world/tidewatch/control@ros_gz_interfaces/srv/ControlWorld',
                        '--ros-args','-p',f'config_file:={bridge_config}'])
        base = [sys.executable,'-m','underwater_system.ros_nodes']
        spawn('instruments',base+['instruments','--config',str(resolved)])
        for index in range(3):
            spawn(f'auv_{index}',base+['auv','--index',str(index),'--config',str(resolved)])
        spawn('mesh',base+['mesh','--config',str(resolved)])
        spawn('station',base+['station','--config',str(resolved),'--state',str(state),
                             '--control',str(control),'--results',str(results)])
        if not args.no_dashboard:
            spawn('dashboard',[sys.executable,'-m','underwater_system.dashboard','--host',config['dashboard']['host'],
                               '--port',str(config['dashboard']['port']),'--state-file',str(state),'--control-file',str(control)])
        if args.rviz:
            spawn('rviz',['rviz2','-d',str(root/'config/tidewatch.rviz'),'--ros-args','-p','use_sim_time:=true'])
        print(f'TIDEWATCH running: http://127.0.0.1:{config["dashboard"]["port"]} | ROS_DOMAIN_ID={args.domain} | GZ_PARTITION={args.partition}',flush=True)
        print(f'Logs: {runtime} | Stop: Ctrl+C or python3 scripts/stop_simulation.py',flush=True)
        started = time.monotonic()
        while not stopping and (args.duration is None or time.monotonic()-started < args.duration):
            for name,process in children:
                if process.poll() is not None and name not in ('rviz','mqtt'):
                    print(f'{name} exited ({process.returncode}); see {runtime/name}.log',file=sys.stderr)
                    return 1
            time.sleep(0.5)
        return 0
    finally:
        for name,process in reversed(children):
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGINT)
        for name,process in reversed(children):
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
        for log in files:
            log.close()
        pid_file.unlink(missing_ok=True)


if __name__ == '__main__':
    sys.exit(main())
