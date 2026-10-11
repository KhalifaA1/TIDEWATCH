#!/usr/bin/env python3
"""Validate native AUV thrusters and underwater dynamics in isolated Gazebo.

Starts a separate, paused server in a unique transport partition. The active
TIDEWATCH mission, its ROS nodes, and its dashboard are never addressed.
Source the ROS / Gazebo environment before invoking this script.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET


class PhysicsProbe:
    def __init__(self, world, log_path, timeout):
        # Set GZ_PARTITION before importing transport or creating its first node.
        from gz.transport import Node
        from gz.msgs.clock_pb2 import Clock
        from gz.msgs.double_pb2 import Double
        from gz.msgs.pose_v_pb2 import Pose_V
        from gz.msgs.vector3d_pb2 import Vector3d
        from gz.msgs.world_control_pb2 import WorldControl

        self.Double = Double
        self.Vector3d = Vector3d
        self.WorldControl = WorldControl
        self.timeout = timeout
        self.log_path = log_path
        self.log = log_path.open('w')
        self.process = None
        self.clock = 0.0
        self.poses = {}
        self.lock = threading.Lock()
        root = ET.parse(world)
        self.step_s = float(root.findtext('./world/physics/max_step_size'))
        self.mass = sum(float(link.findtext('inertial/mass'))
                        for link in root.findall('./world/model[@name="auv_0"]/link'))
        self.node = Node()
        self.node.subscribe(Pose_V, '/tidewatch/ground_truth', self.pose_callback)
        self.node.subscribe(Clock, '/world/tidewatch/clock', self.clock_callback)
        self.publishers = {
            name: [self.node.advertise(
                f'/model/{name}/joint/thruster_{index}_joint/cmd_thrust', Double)
                for index in range(8)]
            for name in ('auv_0', 'auv_1', 'auv_2')
        }
        self.current_publisher = self.node.advertise('/ocean_current', Vector3d)
        try:
            self.process = subprocess.Popen(
                ['gz', 'sim', '-s', '-v', '3', str(world)],
                stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        except BaseException:
            self.log.close()
            raise

    def pose_callback(self, message):
        stamp = message.header.stamp.sec + message.header.stamp.nsec / 1e9
        with self.lock:
            for pose in message.pose:
                if pose.name in self.publishers:
                    self.poses[pose.name] = {
                        'position': [pose.position.x, pose.position.y, pose.position.z],
                        'quaternion': [pose.orientation.x, pose.orientation.y,
                                       pose.orientation.z, pose.orientation.w],
                        'time': stamp,
                    }

    def clock_callback(self, message):
        with self.lock:
            self.clock = message.sim.sec + message.sim.nsec / 1e9

    def snapshot(self):
        with self.lock:
            return self.clock, json.loads(json.dumps(self.poses, allow_nan=False))

    def wait_for(self, predicate, description):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f'Gazebo exited ({self.process.returncode}); see {self.log_path}')
            if predicate():
                return
            time.sleep(0.02)
        raise TimeoutError(f'Timed out waiting for {description}; see {self.log_path}')

    def ready(self):
        self.wait_for(
            lambda: '/world/tidewatch/control' in self.node.service_list()
            and self.current_publisher.has_connections()
            and all(publisher.has_connections()
                    for publishers in self.publishers.values() for publisher in publishers),
            'world control and all 24 native thruster subscriptions',
        )

    def step(self, seconds):
        from google.protobuf import text_format

        start, _ = self.snapshot()
        iterations = round(seconds / self.step_s)
        target = start + iterations * self.step_s
        control = self.WorldControl(pause=True, multi_step=iterations)
        # The CLI isolates the synchronous service request from Python transport
        # callbacks, which remain active while clock and model poses are delivered.
        response = subprocess.run(
            ['gz', 'service', '-s', '/world/tidewatch/control',
             '--reqtype', 'gz.msgs.WorldControl', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '2000', '--req', text_format.MessageToString(control)],
            capture_output=True, text=True, timeout=5,
        )
        if response.returncode or 'data: true' not in response.stdout:
            raise RuntimeError(f'Gazebo step request failed: {response.stdout} {response.stderr}')

        def completed():
            clock, poses = self.snapshot()
            # The first iteration is stamped at t=0, so an N-step run can end
            # one physics step before the requested elapsed duration.
            return (clock >= target - self.step_s * 1.5
                    and len(poses) == 3
                    and all(pose['time'] >= target - 0.06 for pose in poses.values()))

        self.wait_for(completed, f'simulation time {target:.3f}s and three fresh model poses')
        return self.snapshot()[1]

    def thrust(self, name, forces):
        for publisher, force in zip(self.publishers[name], forces):
            if not publisher.publish(self.Double(data=force)):
                raise RuntimeError(f'Thruster command publish failed for {name}')
        # Gazebo holds each command; this settles transport delivery while paused.
        time.sleep(0.1)

    def current(self, vector):
        if not self.current_publisher.publish(self.Vector3d(x=vector[0], y=vector[1], z=vector[2])):
            raise RuntimeError('Ocean-current publish failed')
        time.sleep(0.15)

    def close(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
        self.log.close()


def yaw(quaternion):
    x, y, z, w = quaternion
    return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def evaluate(probe):
    probe.ready()
    for name in probe.publishers:
        probe.thrust(name, [0] * 8)
    probe.current([0, 0, 0])
    initial = probe.step(0.1)
    neutral = probe.step(5)
    depth_error = max(abs(neutral[name]['position'][2] - initial[name]['position'][2])
                      for name in initial)
    neutral_displacement = max(math.dist(neutral[name]['position'], initial[name]['position'])
                               for name in initial)

    # A single command must remain active over multiple native physics steps.
    probe.thrust('auv_0', [40] * 4 + [0] * 4)
    surge_start = neutral['auv_0']['position']
    surge_half = probe.step(0.5)['auv_0']['position']
    surge_two = probe.step(1.5)['auv_0']['position']
    probe.thrust('auv_0', [0] * 8)
    coast = probe.step(1)['auv_0']['position']

    probe.thrust('auv_0', [0] * 4 + [20] * 4)
    heave_start = probe.snapshot()[1]['auv_0']['position']
    heave = probe.step(2)['auv_0']['position']
    probe.thrust('auv_0', [0] * 8)

    probe.current([0.35, 0.1, 0])
    drift_start = probe.snapshot()[1]['auv_1']['position']
    drift = probe.step(5)['auv_1']['position']
    probe.current([0, 0, 0])

    yaw_start = probe.snapshot()[1]['auv_2']['quaternion']
    probe.thrust('auv_2', [-5, 5, -5, 5, 0, 0, 0, 0])
    yaw_end = probe.step(1)['auv_2']['quaternion']
    probe.thrust('auv_2', [0] * 8)
    delta_yaw = math.atan2(math.sin(yaw(yaw_end) - yaw(yaw_start)),
                           math.cos(yaw(yaw_end) - yaw(yaw_start)))

    result = {
        'physics_step_s': probe.step_s,
        'total_mass_kg': probe.mass,
        'neutral_depth_error_m': depth_error,
        'neutral_max_displacement_m': neutral_displacement,
        'surge_displacement_after_0_5_s_m': surge_half[0] - surge_start[0],
        'surge_displacement_after_2_s_m': surge_two[0] - surge_start[0],
        'coast_displacement_after_zero_command_1_s_m': coast[0] - surge_two[0],
        'heave_displacement_after_2_s_m': heave[2] - heave_start[2],
        'unpowered_current_drift_after_5_s_m': [drift[i] - drift_start[i] for i in range(3)],
        'positive_yaw_response_rad': delta_yaw,
    }
    result['checks'] = {
        'neutral_buoyancy': depth_error < 0.001 and neutral_displacement < 0.001,
        'thrust_generates_acceleration': (surge_two[0] - surge_start[0] > 0.5
                                        and 0.01 < surge_half[0] - surge_start[0] < 0.15),
        'inertia_coasts_after_zero_command': coast[0] - surge_two[0] > 0.2,
        'vertical_thrusters_raise_vehicle': heave[2] - heave_start[2] > 0.25,
        'current_drives_unpowered_vehicle': math.dist(drift[:2], drift_start[:2]) > 0.1,
        'allocator_yaw_sign_matches_joints': delta_yaw > 0.03,
    }
    result['all_checks_passed'] = all(result['checks'].values())
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root / 'results/physical_world_validation.json')
    parser.add_argument('--timeout', type=float, default=25, help='Per-stage timeout in wall seconds (1-60)')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 1 <= args.timeout <= 60:
        parser.error('--timeout must be finite and between 1 and 60 seconds')
    if not shutil.which('gz'):
        parser.error('gz is missing; source the ROS / Gazebo setup.bash first')
    world = root / 'src/underwater_system/worlds/tidewatch.sdf'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix('.log')
    partition = f'tidewatch_physics_{uuid.uuid4().hex}'
    os.environ['GZ_PARTITION'] = partition
    print(f'Validating isolated Gazebo partition {partition}', flush=True)

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    probe = None
    report = {
        'source': 'Isolated native Gazebo underwater physics; held thruster commands',
        'partition': partition,
        'world': str(world.relative_to(root)),
        'all_checks_passed': False,
        'checks': {},
        'scope': 'Simulation dynamics smoke checks; uncalibrated hydrodynamics and actuator coefficients.',
    }
    try:
        probe = PhysicsProbe(world, log_path, args.timeout)
        report.update(evaluate(probe))
    except (Exception, KeyboardInterrupt) as error:
        report['error'] = str(error) or 'Validation interrupted'
    finally:
        if probe is not None:
            probe.close()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
