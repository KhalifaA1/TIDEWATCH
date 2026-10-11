"""ROS adapters: simulated instruments, independent edge agents, acoustic link,
and a read-only monitoring station. Gazebo owns every physical model pose.

Only the instrument emulator and evaluator subscribe to ground truth. AUV
controllers receive noisy instrument packets and delayed acoustic messages.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import yaml
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, TransformStamped, Point, Vector3
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32, read_points_numpy
from std_msgs.msg import String, Header, Float64
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from ros_gz_interfaces.srv import ControlWorld

from .core import (EdgeAgent, SensorModel, WorldObject, MultiObjectTracker,
                   Observation, PersistentMap, AcousticTDMA, assess_alerts, navigation_alerts)
from .controls import DiverPilot, validate_command
from .mission import canonical_mode
from .recordings import HydrophoneRecording
from .mqtt import MQTTPublisher
from .geometry import CollisionWorld
from .acoustics import synthesize_hydrophone
from .perception import EdgePerception, SonarFeatureModel
from .propulsion import ThrusterController, angular_velocity, rotation_matrix

IDS = ('auv_0', 'auv_1', 'auv_2')
MODELS = IDS + ('diver', 'intruder', 'marine_life', 'large_marine_life', 'rock_hazard', 'unknown_object')
KINDS = {'intruder': 'vehicle', 'marine_life': 'marine_life',
         'large_marine_life': 'marine_life', 'rock_hazard': 'obstacle', 'unknown_object': 'unknown'}
WALL = Clock(clock_type=ClockType.STEADY_TIME)
BEST_EFFORT = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)


def seconds(node):
    return node.get_clock().now().nanoseconds / 1e9


def encode(value):
    return String(data=json.dumps(value, separators=(',', ':'), allow_nan=False))


def compact(value):
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value]
    if isinstance(value, np.ndarray):
        return compact(value.tolist())
    return value


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y*q.y + q.z*q.z))


def twist(world_velocity, yaw, yaw_rate=0):
    # VelocityControl consumes model-frame velocities, not navigation-frame ones.
    c, s = math.cos(yaw), math.sin(yaw)
    msg = Twist()
    msg.linear.x = float(c*world_velocity[0] + s*world_velocity[1])
    msg.linear.y = float(-s*world_velocity[0] + c*world_velocity[1])
    msg.linear.z = float(world_velocity[2])
    msg.angular.z = float(yaw_rate)
    return msg


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False))
    os.replace(temporary, path)


class MissionNode(Node):
    def __init__(self, name, config):
        super().__init__(name, parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.config = config
        perception = config.get('perception', {})
        model_backends = [name for name in ('acoustic', 'sonar') if perception.get(name+'_model_path')]
        mode = canonical_mode(config.get('mission_mode', 'protect'))
        self.mission = {'status': 'running', 'mode': mode, 'mode_revision': 0, 'outage': False, 'epoch': 0,
                        'planner': 'predictive escort' if mode == 'protect' else 'cooperative frontier survey',
                        'diver_control': {'mode': 'scripted', 'velocity': [0,0,0]},
                        'protected_radius': config['protected_radius'], 'formation_radius': config['formation_radius'],
                        'current': config.get('ocean_current', [0,0,0]), 'propulsion':'native_thrusters',
                        'classifier_backend': '+'.join(name+'_torchscript' for name in model_backends)
                            if model_backends else 'numeric_feature_baseline'}
        self.create_subscription(String, '/tidewatch/mission', self.on_mission, 10)

    def on_mission(self, message):
        value = json.loads(message.data)
        reset = value.get('epoch', 0) != self.mission.get('epoch', 0)
        self.mission.update(value)
        if reset:
            self.reset()

    def reset(self):
        pass


class Instruments(MissionNode):
    """Instrument emulator. Truth is used only to synthesize noisy measurements.

    Sonar is a geometric range/bearing/elevation model, not acoustic propagation.
    Downward beams raycast the SDF terrain / static collision geometry.
    """
    def __init__(self, config):
        super().__init__('tidewatch_instruments', config)
        self.states = {}
        self.sensors = {key: SensorModel(seed=config['seed'] + i, range_m=config['sensor_range']) for i, key in enumerate(IDS)}
        self.sonar_features = {key: SonarFeatureModel(seed=config['seed']+500+i) for i,key in enumerate(IDS)}
        self.rng = np.random.default_rng(config['seed'])
        perception = config.get('perception', {})
        path = perception.get('hydrophone_recording')
        self.recording = HydrophoneRecording(path, perception.get('sample_rate_hz',8000),
                                            loop=perception.get('hydrophone_recording_loop',False)) if path else None
        self.drift = {key: np.zeros(3) for key in IDS}
        self.pub = {key: self.create_publisher(String, f'/{key}/instruments', 5) for key in IDS}
        self.cmd = {key: self.create_publisher(Twist, f'/model/{key}/cmd_vel', 5)
                    for key in ('diver', 'intruder', 'marine_life', 'large_marine_life')}
        self.eval_pub = self.create_publisher(String, '/tidewatch/evaluation', 5)
        self.create_subscription(TFMessage, '/tidewatch/ground_truth', self.truth, BEST_EFFORT)
        self.last_tick = None
        self.navigation_errors = []
        self.mapping_errors = []
        self.confusion = {}
        self.truth_labels = {}
        self.evaluated_uids = set()
        self.create_subscription(String, '/tidewatch/classifier_predictions', self.evaluate_predictions, 10)
        self.targets = {}
        for key in IDS:
            self.create_subscription(String, f'/{key}/edge_status', self.edge_target, 5)
        self.map_rmse = None
        self.map_samples = 0

        self.create_subscription(PointCloud2, '/tidewatch/map/points', self.evaluate_map, 5)
        self.last_states_at = time.monotonic()
        self.geometry = CollisionWorld(config['world_path'])
        self.create_timer(1/config['sensor_rate'], self.tick)
        self.get_logger().info('Noisy sonar/navigation emulator ready; waiting for Gazebo truth.')

    def on_mission(self, message):
        super().on_mission(message)
        if self.mission['status'] == 'paused':
            for publisher in self.cmd.values():
                publisher.publish(twist([0,0,0], 0))

    def edge_target(self, message):
        value = json.loads(message.data)
        if value.get('mission', {}).get('mode') == self.mission['mode']:
            self.targets[value['id']] = value['mission'].get('target')

    def reset(self):
        self.rng = np.random.default_rng(self.config['seed'])
        self.sensors = {key: SensorModel(seed=self.config['seed']+i, range_m=self.config['sensor_range'])
                        for i, key in enumerate(IDS)}
        self.sonar_features = {key: SonarFeatureModel(seed=self.config['seed']+500+i) for i,key in enumerate(IDS)}
        self.drift = {key: np.zeros(3) for key in IDS}
        self.navigation_errors.clear()
        self.mapping_errors.clear()
        self.confusion.clear()
        self.truth_labels.clear()
        self.evaluated_uids.clear()
        self.last_tick = None
        self.states.clear()
        self.map_rmse = None
        self.map_samples = 0
        self.targets.clear()

    def evaluate_predictions(self, message):
        packet = json.loads(message.data)
        for payload in packet.get('observations', []):
            uid = payload['u']
            if uid in self.evaluated_uids or uid not in self.truth_labels:
                continue
            truth = self.truth_labels[uid]
            if abs(truth['time']-payload['t'])>0.05:
                continue
            prediction = Observation.from_payload(payload).classification
            pair = f'{truth["kind"]}:{prediction}'
            self.confusion[pair] = self.confusion.get(pair,0)+1
            self.evaluated_uids.add(uid)

    def evaluate_map(self, message):
        points = read_points_numpy(message, field_names=('x','y','z'), skip_nans=True)
        errors = [self.geometry.surface_distance(point)**2 for point in points]
        self.map_rmse = math.sqrt(float(np.mean(errors))) if errors else None
        self.map_samples = len(errors)

    def truth(self, message):
        for tf in message.transforms:
            name = tf.child_frame_id.split('::')[0].split('/')[-1]
            if name not in MODELS:
                continue
            p = tf.transform.translation
            position = np.array([p.x, p.y, p.z])
            stamp = tf.header.stamp.sec + tf.header.stamp.nanosec/1e9
            old = self.states.get(name)
            velocity = np.zeros(3)
            q = tf.transform.rotation
            orientation = [q.x,q.y,q.z,q.w]
            omega = np.zeros(3)
            if old and 0 < stamp-old['time'] < 1:
                velocity = (position-old['position'])/(stamp-old['time'])
                omega = angular_velocity(old['orientation'],orientation,stamp-old['time'])
            self.states[name] = {'position': position, 'velocity': velocity,
                                 'yaw': yaw_of(tf.transform.rotation), 'orientation':orientation,
                                 'omega_body':omega, 'time': stamp}
        self.last_states_at = time.monotonic()

    def rays(self, origin, measured_origin, yaw, now, orientation=None):
        hits, errors = [], []
        rotation = rotation_matrix(orientation or [0,0,math.sin(yaw/2),math.cos(yaw/2)])
        for j in range(12):
            azimuth = 2*math.pi*j/12 + now*0.08
            direction = rotation@np.array([0.55*math.cos(azimuth), 0.55*math.sin(azimuth), -1.0])
            direction /= np.linalg.norm(direction)
            distance = self.geometry.first_hit(origin, direction, self.config['sensor_range'])
            if distance is None:
                continue
            if not 0 < distance <= self.config['sensor_range']:
                continue
            truth_hit = origin+direction*distance
            hit = measured_origin+direction*(distance+self.rng.normal(0, 0.06))
            hits.append(hit.tolist())
            errors.append(float(np.linalg.norm(hit-truth_hit)**2))
        return hits, errors

    def actor_commands(self, now):
        # Moving diver, a clearly approaching vehicle, and harmless marine life.
        targets = {
            'diver': np.array([3*math.sin(now/28), 2*math.sin(now/37), -10+0.7*math.sin(now/23)]),
            'intruder': np.array([18*math.cos(now/32), 4+5*math.sin(now/32), -10.0]),
            'marine_life': np.array([-11+4*math.cos(now/18), -8+4*math.sin(now/18), -8.0]),
            # A large wildlife crossing, not a species-recognition claim.
            'large_marine_life': np.array([12*math.cos(now/22), 7*math.sin(now/22), -10.5]),
        }
        for name, desired in targets.items():
            state = self.states.get(name)
            if not state:
                continue
            if self.mission['status'] == 'paused':
                self.cmd[name].publish(twist([0,0,0], state['yaw']))
                continue
            pilot = self.mission.get('diver_control', {})
            if name == 'diver' and pilot.get('mode') == 'manual':
                velocity = np.array(pilot.get('velocity', [0,0,0]), dtype=float)
                # Keep the operator inside this scene's test volume.
                for axis, limits in enumerate(((-27,27),(-27,27),(-16,-3))):
                    if ((state['position'][axis] <= limits[0] and velocity[axis] < 0) or
                            (state['position'][axis] >= limits[1] and velocity[axis] > 0)):
                        velocity[axis] = 0
                self.cmd[name].publish(twist(velocity, state['yaw']))
                continue
            error = desired-state['position']
            velocity = np.clip(error*0.8, -0.65, 0.65)
            desired_yaw = math.atan2(velocity[1], velocity[0])
            angular = np.clip(math.atan2(math.sin(desired_yaw-state['yaw']),
                                        math.cos(desired_yaw-state['yaw']))*0.8, -0.5, 0.5)
            self.cmd[name].publish(twist(velocity, state['yaw'], angular))

    def tick(self):
        now = seconds(self)
        if now < 0.001 or not all(key in self.states for key in IDS+('diver',)):
            return
        if self.last_tick is not None and now < self.last_tick:
            self.reset()
            return
        dt = 1/self.config['sensor_rate'] if self.last_tick is None else min(0.5, now-self.last_tick)
        if dt <= 0:
            return
        self.last_tick = now
        if time.monotonic()-self.last_states_at > 3:
            return
        self.actor_commands(now)
        objects = [WorldObject(name, KINDS[name], state['position'], state['velocity'])
                   for name, state in self.states.items() if name in KINDS]
        diver = self.states['diver']
        formation_errors = []
        for i, key in enumerate(IDS):
            state = self.states[key]
            self.drift[key] += self.rng.normal(0, self.config['navigation_drift_rate']*math.sqrt(dt), 3)
            measured = state['position']+self.drift[key]+self.rng.normal(0, self.config['navigation_noise'], 3)
            uncertainty = self.config['navigation_noise']**2 + self.config['navigation_drift_rate']**2*now
            self.navigation_errors.append(float(np.linalg.norm(measured-state['position'])**2))
            observations = self.sensors[key].observe(key, state['position'], state['yaw'], objects, now)
            payloads = []
            signatures = {}
            for observation in observations:
                # Convert noisy relative ranges through a noisy navigation frame.
                observation.position += measured-state['position']
                observation.covariance += np.eye(3)*uncertainty
                if observation.truth_id in KINDS:
                    true_kind = KINDS[observation.truth_id]
                    truth_state = self.states[observation.truth_id]
                    distance = float(np.linalg.norm(truth_state['position']-state['position']))
                    speed = float(np.linalg.norm(truth_state['velocity']))
                    signatures[observation.uid] = self.sonar_features[key].measure(true_kind,distance,speed)
                    self.truth_labels[observation.uid] = {'kind':true_kind,'time':now}
                # Geometry simulation may use templates internally, but semantic
                # labels never cross the instrument-to-edge boundary.
                observation.classification, observation.confidence = 'unknown', 0.0
                payloads.append(observation.to_payload())
            points, errors = self.rays(state['position'], measured, state['yaw'], now, state['orientation'])
            self.mapping_errors.extend(errors)
            desired = self.targets.get(key)
            if self.mission['mode'] == 'protect' and desired is not None:
                formation_errors.append(float(np.linalg.norm(state['position']-np.array(desired))**2))
            measured_velocity = state['velocity']+self.rng.normal(0,0.015,3)
            packet = {'time': now, 'position': measured.tolist(), 'velocity': measured_velocity.tolist(),
                      'yaw': state['yaw'], 'uncertainty': math.sqrt(uncertainty),
                      'orientation':state['orientation'], 'omega_body':state['omega_body'].tolist(),
                      'diver_position': (diver['position']+self.rng.normal(0, 0.12, 3)).tolist(),
                      'diver_velocity': diver['velocity'].tolist(), 'observations': payloads,
                      'sonar_features':signatures, 'map_points': points}
            if self.recording is None:
                packet['hydrophone_samples'] = synthesize_hydrophone(objects, state['position'], now,
                    seed=self.config['seed']+i, sample_rate=self.config.get('perception',{}).get('sample_rate_hz',8000))
            else:
                packet['hydrophone_samples'] = self.recording.frame(now)
                packet['hydrophone_source'] = dict(self.recording.metadata)
            self.pub[key].publish(encode(packet))
        total = sum(self.confusion.values())
        correct = sum(value for pair, value in self.confusion.items() if pair.split(':')[0] == pair.split(':')[1])
        categories = ('diver','marine_life','vehicle','obstacle','unknown')
        classification_scores = {}
        for category in categories:
            true_positive = self.confusion.get(f'{category}:{category}',0)
            support = sum(count for pair,count in self.confusion.items() if pair.split(':')[0]==category)
            predicted = sum(count for pair,count in self.confusion.items() if pair.split(':')[1]==category)
            classification_scores[category] = {'precision': true_positive/predicted if predicted else None,
                                                'recall': true_positive/support if support else None,
                                                'support': support, 'predicted': predicted}
        positions = [self.states[key]['position'] for key in IDS]
        minimum = min(float(np.linalg.norm(a-b)) for i, a in enumerate(positions) for b in positions[i+1:])
        metrics = {'localization_rmse': float(np.sqrt(np.mean(self.navigation_errors[-3000:]))),
                   'mapping_rmse': self.map_rmse, 'map_evaluated_points': self.map_samples,
                   'sonar_position_rmse': float(np.sqrt(np.mean(self.mapping_errors[-6000:]))) if self.mapping_errors else None,
                   'classification_accuracy': correct/total if total else None,
                   'classification_samples': total, 'classification_confusion': self.confusion,
                   'classification_per_class': classification_scores,
                   'formation_error': math.sqrt(float(np.mean(formation_errors))) if formation_errors else None,
                   'minimum_auv_separation': minimum, 'time': now}
        self.eval_pub.publish(encode(metrics))
        # Bound evaluation storage; the full confusion counts stay cumulative.
        self.navigation_errors = self.navigation_errors[-3000:]
        self.mapping_errors = self.mapping_errors[-6000:]
        self.truth_labels = {uid:value for uid,value in self.truth_labels.items() if now-value['time']<10}
        self.evaluated_uids.intersection_update(self.truth_labels)


class AUV(MissionNode):
    """One autonomous edge process; it cannot access simulator ground truth."""
    def __init__(self, index, config):
        self.index, self.unit = index, IDS[index]
        super().__init__(f'{self.unit}_edge', config)
        self.agent = EdgeAgent(index, seed=config['seed']+index,
                               mission_mode=self.mission['mode'], mission_config=config)
        self.agent.controller.radius = config['formation_radius']
        self.agent.controller.speed_limit = config['speed_limit']
        self.perception = EdgePerception(config.get('perception',{}))
        self.propulsion = ThrusterController(config.get('propulsion',{}))
        self.navigation = None
        self.desired_velocity = np.zeros(3)
        self.desired_yaw = 0.0
        self.last_actuation_time = None
        self.peers = {}
        self.last_instruments = 0
        self.last_transmit = -1
        self.sequence = 0
        self.observation_cursor = 0
        self.map_cursor = 0
        self.last_sensor_time = -1
        self.thruster_pub = [self.create_publisher(Float64,f'/model/{self.unit}/joint/thruster_{i}_joint/cmd_thrust',5)
                            for i in range(8)]
        self.predictions = self.create_publisher(String, '/tidewatch/classifier_predictions', 10)
        self.tx = self.create_publisher(String, '/tidewatch/acoustic/tx', 10)
        self.status = self.create_publisher(String, f'/{self.unit}/edge_status', 5)
        self.cloud = self.create_publisher(PointCloud2, f'/{self.unit}/sonar/points', 5)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(String, f'/{self.unit}/instruments', self.measurement, 5)
        self.create_subscription(String, f'/{self.unit}/acoustic/rx', self.received, 10)
        self.create_timer(0.2, self.watchdog, clock=WALL)
        self.create_timer(1/config.get('propulsion',{}).get('control_rate_hz',20),self.actuate)
        self.get_logger().info('Independent edge controller ready; no ground-truth subscription.')

    def on_mission(self, message):
        super().on_mission(message)
        if self.mission['status'] == 'paused':
            self.desired_velocity[:] = 0
            self.publish_thrust(self.propulsion.stop())

    def reset(self):
        self.agent = EdgeAgent(self.index, seed=self.config['seed']+self.index,
                               mission_mode=self.mission['mode'], mission_config=self.config)
        self.agent.controller.radius = self.config['formation_radius']
        self.agent.controller.speed_limit = self.config['speed_limit']
        self.peers.clear()
        self.last_transmit = -1
        self.sequence = 0
        self.observation_cursor = 0
        self.map_cursor = 0
        self.last_sensor_time = -1
        self.navigation = None
        self.propulsion.reset()
        self.desired_velocity[:] = 0
        self.last_actuation_time = None
        self.publish_thrust(np.zeros(8))

    def received(self, message):
        envelope = json.loads(message.data)
        source = envelope['source']
        if source in IDS and source != self.unit:
            value = envelope['payload']
            if value.get('t', 0) > seconds(self)+0.5:
                return
            previous_packet = self.peers.get(source)
            if previous_packet and (value.get('t',0) < previous_packet.get('t',0) or
                                    value.get('q',-1) <= previous_packet.get('q',-1)):
                return
            value['source'] = source
            if isinstance(value.get('m'), dict):
                value['mission'] = self.decode_mission(value['m'], source, value['t'])
            else:
                previous = self.peers.get(source, {}).get('mission')
                if previous and seconds(self)-previous.get('reported_at', 0) < 4:
                    value['mission'] = previous
            self.peers[source] = value

    @staticmethod
    def decode_mission(value, source, timestamp):
        mode = value.get('m', 'protect')
        return {'mode': mode, 'target': value.get('g'), 'phase': value.get('s', 'unknown'),
                'role': value.get('r', f"{'Guard' if mode == 'protect' else 'Survey sector'} {int(source[-1])+1}"),
                'observed_cells': value.get('o', 0), 'coverage': value.get('c', 0),
                'waypoints_completed': value.get('w', 0), 'reported_at': timestamp,
                'return_reason': value.get('e')}

    def watchdog(self):
        if time.monotonic()-self.last_instruments > 1.5:
            self.publish_thrust(self.propulsion.stop())

    def publish_thrust(self, values):
        for publisher,value in zip(self.thruster_pub,values):
            publisher.publish(Float64(data=float(value)))

    def actuate(self):
        if self.mission['status'] == 'paused':
            self.desired_velocity[:] = 0
            self.publish_thrust(self.propulsion.stop())
            return
        if self.navigation is None or time.monotonic()-self.last_instruments>1.5:
            return
        now = seconds(self)
        dt = .05 if self.last_actuation_time is None else now-self.last_actuation_time
        self.last_actuation_time = now
        if dt<=0:
            return
        p = self.navigation
        thrust = self.propulsion.command(self.desired_velocity,p['velocity'],p['orientation'],
                                          p['omega_body'],self.desired_yaw,dt)
        self.publish_thrust(thrust)

    def measurement(self, message):
        packet = json.loads(message.data)
        now = packet['time']
        if self.mission['status'] == 'paused':
            self.desired_velocity[:] = 0
            self.publish_thrust(self.propulsion.stop())
            return
        if now < self.last_sensor_time:
            self.reset()
        self.last_sensor_time = now
        self.last_instruments = time.monotonic()
        self.navigation = packet
        frame = self.perception.analyze(packet['hydrophone_samples'])
        classified = []
        for raw in packet['observations']:
            observation = Observation.from_payload(raw)
            prediction = self.perception.classify(packet['hydrophone_samples'],
                                                  packet['sonar_features'].get(observation.uid,{}),frame)
            observation.classification = prediction['classification']
            observation.confidence = prediction['confidence']
            classified.append(observation.to_payload())
        self.predictions.publish(encode({'time':now,'source':self.unit,'observations':classified}))
        peers = [value for value in self.peers.values() if now-value.get('t', 0) < 4]
        self.agent.set_mode(self.mission['mode'])
        result = self.agent.step(now, 1/self.config['sensor_rate'], packet['position'], packet['velocity'],
                                 packet['diver_position'], packet['diver_velocity'], peer_telemetry=peers,
                                 observations=classified, rays=packet['map_points'], uncertainty=packet['uncertainty'],
                                 diver_timestamp=packet.get('diver_timestamp', now))
        acoustic_features = frame['acoustic_features']
        velocity = np.array(result['command_velocity'])
        gateway = self.config['gateway']
        surfacing = gateway['require_surface'] and self.index == 0 and now % gateway['surface_interval'] < gateway['surface_duration'] and now > gateway['surface_interval']
        if surfacing:
            velocity[2] = np.clip((-0.5-packet['position'][2])*0.5, -1, 1)
            speed = float(np.linalg.norm(velocity))
            if speed > self.config['speed_limit']:
                velocity *= self.config['speed_limit']/speed
        desired_yaw = result['mission']['heading']
        self.desired_velocity, self.desired_yaw = velocity,desired_yaw
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id, tf.child_frame_id = 'map', f'{self.unit}/base_link'
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = packet['position']
        tf.transform.rotation.x,tf.transform.rotation.y,tf.transform.rotation.z,tf.transform.rotation.w = packet['orientation']
        self.tf.sendTransform(tf)
        header = Header(stamp=tf.header.stamp, frame_id='map')
        self.cloud.publish(create_cloud_xyz32(header, packet['map_points']))
        self.status.publish(encode({'id': self.unit, 'time': now, 'tracks': len(result.get('tracks', [])),
                                    'peer_count': len(peers), 'surfacing': surfacing,
                                    'acoustic_features': acoustic_features,
                                    'mission': result['mission'],
                                    'local_map_cells': len(self.agent.map.cells), 'propulsion':self.propulsion.last}))
        if now-self.last_transmit < 3*self.config['acoustic']['slot_duration']:
            return
        self.last_transmit = now
        outgoing = {'t': now, 'p': packet['position'], 'v': packet['velocity'],
                    'r': packet['uncertainty']**2, 'd': packet['diver_position'], 'q': self.sequence,
                    'a': packet['yaw'], 'tu':self.propulsion.last.get('thrust_utilization',0),
                    'pw':self.propulsion.last.get('power_estimate_w',0)}
        plan = result['mission']
        outgoing['m'] = {'m': plan['mode'], 'g': plan['target'], 's': plan['phase']}
        if plan['mode'] == 'explore':
            outgoing['m'].update(o=plan.get('observed_cells',0), c=plan.get('coverage',0),
                                 w=plan.get('waypoints_completed',0))
            if plan.get('return_reason'):
                outgoing['m']['e'] = plan['return_reason']
        self.sequence += 1
        observations = classified
        if observations:
            outgoing['o'] = [observations[self.observation_cursor % len(observations)]]
            self.observation_cursor += 1
        points = packet['map_points']
        if points:
            outgoing['mp'] = [points[(self.map_cursor+j) % len(points)] for j in range(3)]
            self.map_cursor += 3
        outgoing = compact(outgoing)
        outgoing['r'] = math.ceil(packet['uncertainty']**2*100)/100
        for observation in outgoing.get('o', []):
            # Include quantization error and round the covariance bound upward.
            original = next((o for o in observations if o['u'] == observation['u']), observation)
            observation['r'] = max(0.01, math.ceil((original['r']+0.0001)*100)/100)
        # Every packet must fit the TDMA slot. Reduction happens before queuing.
        budget = int(self.config['acoustic']['bitrate'] * self.config['acoustic']['slot_duration']/8)-90
        # Reserve one report in three for mission claims. The other reports give
        # contact observations priority, so mission metadata cannot starve sensing.
        if self.sequence % 3 and len(json.dumps(outgoing, separators=(',', ':')).encode()) > budget:
            outgoing.pop('m', None)
        while len(json.dumps(outgoing, separators=(',', ':')).encode()) > budget and outgoing.get('mp'):
            outgoing['mp'].pop()
        if len(json.dumps(outgoing, separators=(',', ':')).encode()) > budget:
            outgoing.pop('o', None)
        self.tx.publish(encode({'source': self.unit, 'payload': outgoing, 'time': now}))


class Mesh(MissionNode):
    """Shared channel emulator; slot scheduling never computes motion commands."""
    def __init__(self, config):
        super().__init__('tidewatch_acoustic_channel', config)
        self.channel = self.make_channel()
        self.rx = {key: self.create_publisher(String, f'/{key}/acoustic/rx', 10) for key in IDS}
        self.gateway = self.create_publisher(String, '/tidewatch/gateway/rx', 20)
        self.stats = self.create_publisher(String, '/tidewatch/network', 5)
        self.pending_gateway = []
        self.last_gateway = -1
        self.create_subscription(String, '/tidewatch/acoustic/tx', self.queue, 20)
        self.create_timer(0.05, self.tick)

    def make_channel(self):
        c = self.config['acoustic']
        return AcousticTDMA(IDS, seed=self.config['seed'], slot_duration=c['slot_duration'],
                            bitrate=c['bitrate'], loss_probability=c['loss_probability'],
                            delay_range=(c['delay_min'], c['delay_max']), ttl=c['ttl'])

    def reset(self):
        self.channel = self.make_channel()
        self.pending_gateway.clear()
        self.last_gateway = -1

    def queue(self, message):
        value = json.loads(message.data)
        if value['time'] > seconds(self)+0.5:
            return
        self.channel.enqueue(value['source'], value['payload'], value['time'])

    def tick(self):
        now = seconds(self)
        if now < self.channel.now:
            self.reset()
        self.channel.set_connected('auv_1', not self.mission.get('outage', False))
        for delivered in self.channel.step(now):
            for unit, pub in self.rx.items():
                if unit != delivered['source'] and not (unit == 'auv_1' and self.mission.get('outage')):
                    pub.publish(encode(delivered))
            # A gateway receives its own reports locally and remote ones by mesh.
            self.pending_gateway.append(delivered)
        gateway = self.config['gateway']
        if now-self.last_gateway >= gateway['burst_interval']:
            self.last_gateway = now
            surfaced = (not gateway['require_surface'] or
                        now % gateway['surface_interval'] < gateway['surface_duration'] and now > gateway['surface_interval'])
            if surfaced:
                for delivered in self.pending_gateway[-30:]:
                    self.gateway.publish(encode(delivered))
                self.pending_gateway.clear()
        self.pending_gateway = [value for value in self.pending_gateway if now-value['received_at'] < 30]
        self.stats.publish(encode(self.channel.snapshot()))


class Station(MissionNode):
    def __init__(self, config, state_path, control_path, results_path):
        super().__init__('tidewatch_monitoring_station', config)
        self.state_path, self.control_path = state_path, control_path
        self.results_path = Path(results_path)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.telemetry = {}
        self.unit_missions = {}
        self.exploration_cells = set()
        self.diver_pilot = DiverPilot()
        self.tracker = MultiObjectTracker(max_age=15.0)
        self.map = PersistentMap(resolution=config['map_resolution'])
        self.diver = [0, 0, -10]
        self.network = {}
        self.metrics = {}
        self.control_nonce = None
        self.last_time = -1
        self.mqtt = MQTTPublisher(**config['mqtt'])
        self.mission_pub = self.create_publisher(String, '/tidewatch/mission', 10)
        self.current_pub = self.create_publisher(Vector3, '/ocean_current', 5)
        self.marker_pub = self.create_publisher(MarkerArray, '/tidewatch/markers', 5)
        self.cloud_pub = self.create_publisher(PointCloud2, '/tidewatch/map/points', 5)
        self.control_client = self.create_client(ControlWorld, '/world/tidewatch/control')
        self.create_subscription(String, '/tidewatch/gateway/rx', self.delivered, 20)
        self.create_subscription(String, '/tidewatch/network', self.network_update, 5)
        self.create_subscription(String, '/tidewatch/evaluation', self.evaluation, 5)
        self.create_timer(0.5, self.tick, clock=WALL)
        self.get_logger().info(f'Monitoring station writes live state to {state_path}')

    def reset(self):
        self.telemetry.clear()
        self.unit_missions.clear()
        self.exploration_cells.clear()
        self.diver_pilot.stop()
        self.tracker = MultiObjectTracker(max_age=15.0)
        self.map = PersistentMap(resolution=self.config['map_resolution'])
        self.metrics.clear()

    def network_update(self, message):
        self.network = json.loads(message.data)

    def evaluation(self, message):
        self.metrics = json.loads(message.data)

    def delivered(self, message):
        envelope = json.loads(message.data)
        payload, source = envelope['payload'], envelope['source']
        if payload['t'] > seconds(self)+0.5:
            return
        previous = self.telemetry.get(source)
        if previous and payload.get('q', -1) <= previous.get('q', -1):
            return
        self.telemetry[source] = payload
        if isinstance(payload.get('m'), dict):
            self.unit_missions[source] = AUV.decode_mission(payload['m'], source, payload['t'])
        self.diver = payload.get('d', self.diver)
        observations = [Observation.from_payload(value) for value in payload.get('o', [])]
        self.tracker.update(observations, seconds(self))
        self.map.update(payload.get('mp', []), payload['p'], payload['t'])
        exploration = self.config.get('exploration', {})
        bounds = exploration.get('bounds', [[-24,24],[-24,24]])
        size = exploration.get('cell_size', 3)
        for point in payload.get('mp', []):
            if bounds[0][0] <= point[0] < bounds[0][1] and bounds[1][0] <= point[1] < bounds[1][1]:
                self.exploration_cells.add((int((point[0]-bounds[0][0])/size), int((point[1]-bounds[1][0])/size)))

    def controls(self):
        try:
            value = json.loads(Path(self.control_path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return
        nonce = value.get('nonce')
        if nonce is None or nonce == self.control_nonce:
            return
        self.control_nonce = nonce
        try:
            value = {**value, **validate_command(value)}
        except (ValueError, TypeError):
            self.get_logger().warning('Rejected invalid operator command')
            return
        command = value.get('command')
        if command == 'toggle_outage':
            self.mission['outage'] = not self.mission.get('outage', False)
        elif command == 'set_current':
            vector = value.get('vector')
            if (isinstance(vector,list) and len(vector)==3 and
                    all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and abs(x)<=.8 for x in vector)):
                self.mission['current'] = [float(x) for x in vector]
        elif command == 'set_mode':
            if self.mission['mode'] != value['mode']:
                self.mission['mode'] = value['mode']
                self.mission['mode_revision'] += 1
                self.mission['planner'] = 'predictive escort' if value['mode'] == 'protect' else 'cooperative frontier survey'
                self.diver_pilot.stop()
        elif command == 'set_diver_control':
            # A delayed key press cannot restart motion after the operator stops.
            issued = value.get('issued_at')
            if (type(issued) in (int,float) and 0 <= issued <= 1e12 and math.isfinite(issued) and
                    -.5 <= time.time()-issued <= self.diver_pilot.timeout):
                self.diver_pilot.command(value['mode'], value['velocity'], time.monotonic())
        elif command in ('start', 'pause', 'reset'):
            request = ControlWorld.Request()
            request.world_control.pause = command == 'pause'
            if command == 'reset':
                request.world_control.reset.all = True
                self.mission['epoch'] += 1
                self.reset()
            self.mission['status'] = 'paused' if command == 'pause' else 'running'
            if command in ('pause', 'reset'):
                self.diver_pilot.stop()
            if self.control_client.service_is_ready():
                self.control_client.call_async(request)
            else:
                self.get_logger().warning('Gazebo control service unavailable')
        self.mission_pub.publish(encode(self.mission))

    def marker(self, identifier, kind, position, scale, color, text=''):
        result = Marker()
        result.header.frame_id = 'map'
        result.header.stamp = self.get_clock().now().to_msg()
        result.ns = 'tidewatch'
        result.id, result.type, result.action = identifier, kind, Marker.ADD
        result.pose.position.x, result.pose.position.y, result.pose.position.z = [float(x) for x in position]
        result.pose.orientation.w = 1.0
        result.scale.x, result.scale.y, result.scale.z = [float(x) for x in scale]
        result.color.r, result.color.g, result.color.b, result.color.a = [float(x) for x in color]
        result.text = text
        return result

    def visualize(self, state):
        markers = MarkerArray()
        clear = Marker(action=Marker.DELETEALL)
        markers.markers.append(clear)
        if self.mission['mode'] == 'protect':
            markers.markers.append(self.marker(0, Marker.SPHERE, self.diver, [0.7]*3, [1,0.75,0.3,1]))
            markers.markers.append(self.marker(1, Marker.SPHERE, self.diver,
                                               [self.config['protected_radius']*2]*3, [1,0.75,0.3,0.08]))
        colors = [[0.15,0.85,0.95,1], [0.3,0.95,0.65,1], [0.7,0.55,1,1]]
        for i, unit in enumerate(state['auvs']):
            robot = self.marker(10+i, Marker.CUBE, unit['position'], [1.0,0.75,0.4], colors[i])
            yaw = unit.get('yaw',0)
            robot.pose.orientation.z,robot.pose.orientation.w = math.sin(yaw/2),math.cos(yaw/2)
            markers.markers.append(robot)
            p = list(unit['position'])
            p[2] += 0.8
            markers.markers.append(self.marker(20+i, Marker.TEXT_VIEW_FACING, p, [0,0,0.5], colors[i], unit['id']))
        for i, track in enumerate(state['tracks']):
            markers.markers.append(self.marker(100+i, Marker.SPHERE, track['position'], [0.7]*3,
                                               [1,0.35,0.2,0.9] if track.get('risk') == 'danger' else [1,0.7,0.2,0.8]))
        if self.mission['mode'] == 'protect' and len(state['auvs']) >= 3:
            perimeter = self.marker(2, Marker.LINE_STRIP, [0,0,0], [0.08,0,0], [0.2,0.9,0.8,0.7])
            positions = [unit['position'] for unit in state['auvs']]
            perimeter.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in positions+[positions[0]]]
            markers.markers.append(perimeter)
        self.marker_pub.publish(markers)
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='map')
        self.cloud_pub.publish(create_cloud_xyz32(header, state['map']['points']))

    def tick(self):
        self.controls()
        self.mission['diver_control'] = self.diver_pilot.snapshot(time.monotonic(), self.mission['status'] == 'paused')
        self.mission_pub.publish(encode(self.mission))
        current = self.mission['current']
        self.current_pub.publish(Vector3(x=float(current[0]),y=float(current[1]),z=float(current[2])))
        now = seconds(self)
        if now < self.last_time:
            self.reset()
        self.last_time = now
        self.tracker.update([], now)
        tracks = self.tracker.snapshot()
        if isinstance(tracks, dict):
            tracks = tracks.get('tracks', [])
        auvs = []
        for key in IDS:
            value = self.telemetry.get(key)
            if not value:
                continue
            age = max(0, now-value['t'])
            uncertainty = math.sqrt(max(value.get('r',0),0))+age*0.04
            # Dead reckon the UI position, retaining report age and growing error.
            displayed = (np.array(value['p'])+np.array(value['v'])*min(age,4)).tolist()
            auvs.append({'id': key, 'position': displayed, 'velocity': value['v'],
                         'uncertainty': uncertainty, 'online': age < self.config['acoustic']['ttl']+self.config['gateway']['burst_interval']+1,
                         'age': age, 'yaw':value.get('a',0),
                         'mission': self.unit_missions.get(key),
                         'propulsion':{'mode':'native_thrusters','thrust_utilization':value.get('tu',0),
                                       'power_estimate_w':value.get('pw',0)}})
        if self.mission['mode'] == 'protect':
            velocity = self.mission['diver_control']['velocity']
            alerts = assess_alerts(tracks, self.diver, velocity, protected_radius=self.config['protected_radius'])
        else:
            candidates = [alert for unit in auvs if unit['online']
                          for alert in navigation_alerts(tracks, unit['position'], unit['velocity'])]
            rank = {'critical':0, 'warning':1, 'watch':2, 'info':3}
            grouped = {}
            for alert in candidates:
                previous = grouped.get(alert['track_id'])
                if previous is None or rank[alert['severity']] < rank[previous['severity']]:
                    grouped[alert['track_id']] = alert
            alerts = sorted(grouped.values(), key=lambda item: rank[item['severity']])
        by_track = {alert['track_id']: alert for alert in alerts}
        for track in tracks:
            track['label'] = track.get('classification', 'unknown')
            track['uncertainty'] = track.get('position_std_m', 0.5)
            severity = by_track.get(track['id'], {}).get('severity', 'info')
            track['risk'] = {'critical':'danger','warning':'warning','watch':'warning'}.get(severity, 'info')
        mapping = self.map.snapshot()
        entries = mapping.get('points', [])
        points = [point['position'] if isinstance(point, dict) else point for point in entries]
        coverage = self.map.evaluate()['coverage_fraction']
        network = dict(self.network)
        network['sent'] = network.get('transmitted', 0)
        network['dropped'] = network.get('lost', 0)
        network['queued'] = sum(network.get('queue_depth', {}).values())
        network['loss_rate'] = network.get('lost', 0)/max(network.get('transmitted', 0), 1)
        network['latency_ms'] = network.get('mean_latency_s', 0)*1000
        exploration = self.config.get('exploration', {})
        bounds = exploration.get('bounds', [[-24,24],[-24,24]])
        cell_size = exploration.get('cell_size', 3)
        total_cells = math.ceil((bounds[0][1]-bounds[0][0])/cell_size)*math.ceil((bounds[1][1]-bounds[1][0])/cell_size)
        self.mission['exploration'] = {**exploration, 'bounds': bounds,
                'origin': exploration.get('origin', [0,0,-10]), 'depth_layers': exploration.get('depth_layers', [-10]),
                'cell_size': cell_size, 'observed_cells': len(self.exploration_cells), 'total_cells': total_cells,
                'coverage': len(self.exploration_cells)/total_cells, 'source': 'gateway-delivered measured returns'}
        state = {'time': now, 'updated_at': time.time(), 'mission': dict(self.mission), 'auvs': auvs,
                 'diver': {'position': self.diver}, 'tracks': tracks, 'alerts': alerts,
                 'map': {**mapping, 'points': points, 'coverage': coverage}, 'network': network,
                 'metrics': {**self.metrics, 'coverage': coverage}, 'mqtt': {'connected': self.mqtt.connected},
                 'simulation': {'sensing': 'synthetic geometric sonar',
                                'hydrophone': {'source':'recorded' if self.config.get('perception',{}).get('hydrophone_recording') else 'synthetic',
                                    'name':Path(self.config['perception']['hydrophone_recording']).name
                                        if self.config.get('perception',{}).get('hydrophone_recording') else None,
                                    'loop':self.config.get('perception',{}).get('hydrophone_recording_loop',False)},
                                'classifier': 'edge numeric sonar features and hydrophone STFT; synthetic baseline',
                                'motion': 'eight native force-driven thrusters, buoyancy and hydrodynamics',
                                'gateway': 'periodic surface backhaul emulation'}}
        atomic_json(self.state_path, state)
        self.visualize(state)
        self.mqtt.publish('tidewatch/status', state)
        with self.results_path.open('a') as file:
            file.write(json.dumps({'time': now, 'metrics': state['metrics'], 'network': network,
                                   'alerts': len(alerts)}, separators=(',', ':'))+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=['instruments', 'auv', 'mesh', 'station'])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--config', required=True)
    parser.add_argument('--state', default='/tmp/tidewatch-state.json')
    parser.add_argument('--control', default='/tmp/tidewatch-control.json')
    parser.add_argument('--results', default='results/live.jsonl')
    args, ros_args = parser.parse_known_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    rclpy.init(args=ros_args)
    if args.role == 'instruments':
        node = Instruments(config)
    elif args.role == 'auv':
        node = AUV(args.index, config)
    elif args.role == 'mesh':
        node = Mesh(config)
    else:
        node = Station(config, args.state, args.control, args.results)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
