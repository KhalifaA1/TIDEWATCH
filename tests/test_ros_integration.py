"""Run real ROS callbacks with in-memory publishers and no ROS initialization."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'underwater_system'))

try:
    import rclpy
except ModuleNotFoundError:
    ros = None
else:
    from builtin_interfaces.msg import Time
    from underwater_system import ros_nodes as ros

from underwater_system.controls import DiverPilot
from underwater_system.core import AcousticTDMA, EdgeAgent, MultiObjectTracker, Observation, PersistentMap
from underwater_system.perception import EdgePerception, SonarFeatureModel
from underwater_system.propulsion import ThrusterController


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def attach_clock(node, stamp=10.0):
    node.sim_time = stamp

    def now():
        nanos = int(round(node.sim_time * 1e9))
        return SimpleNamespace(nanoseconds=nanos,
                               to_msg=lambda: Time(sec=nanos // 10**9, nanosec=nanos % 10**9))

    node.get_clock = lambda: SimpleNamespace(now=now)
    node.get_logger = lambda: node.logger
    node.logger = Mock()


def mission(mode='protect'):
    return {'mode': mode, 'status': 'running', 'epoch': 0, 'mode_revision': 0,
            'outage': False, 'current': [0., 0., 0.],
            'diver_control': {'mode': 'scripted', 'velocity': [0., 0., 0.]}}


def edge_node(config, mode='protect'):
    # Bypass Node.__init__: all application callbacks and ROS messages remain real.
    node = object.__new__(ros.AUV)
    node.index, node.unit = 0, 'auv_0'
    node.config, node.mission = copy.deepcopy(config), mission(mode)
    node.agent = EdgeAgent(0, mission_mode=mode, mission_config=node.config)
    node.perception = EdgePerception(node.config['perception'])
    node.propulsion = ThrusterController(node.config['propulsion'])
    node.navigation = None
    node.desired_velocity = np.zeros(3)
    node.desired_yaw = 0.0
    node.last_actuation_time = None
    node.peers = {}
    node.last_instruments = 0.0
    node.last_transmit = -1.0
    node.sequence = node.observation_cursor = node.map_cursor = 0
    node.last_sensor_time = -1.0
    node.thruster_pub = [Publisher() for _ in range(8)]
    node.predictions, node.tx, node.status, node.cloud = [Publisher() for _ in range(4)]
    node.tf = SimpleNamespace(sendTransform=Mock())
    attach_clock(node)
    return node


def instrument_packet(now, sequence=0):
    source = SonarFeatureModel(seed=23 + sequence)
    observations, signatures = [], {}
    for index, kind in enumerate(('vehicle', 'marine_life')):
        observation = Observation(f'auv_0:{sequence * 2 + index}', 'auv_0', now,
                                  [12.34567, 6.89123 + index * 3, -10.12345],
                                  np.eye(3) * 0.0654321, 'unknown', 0.0)
        observations.append(observation.to_payload())
        signatures[observation.uid] = source.measure(kind, 9.0, 0.25)
    return {'time': now, 'position': [6.1234567, -.093434, -10.023456],
            'velocity': [.01234, -.02345, .00567], 'yaw': .132456,
            'uncertainty': .0834567, 'orientation': [0., 0., 0., 1.], 'omega_body': [0., 0., 0.],
            'diver_position': [.123456, -.06789, -10.0234], 'diver_velocity': [.1, .02, 0.],
            'diver_timestamp': now, 'observations': observations, 'sonar_features': signatures,
            'hydrophone_samples': np.zeros(2048).tolist(),
            'map_points': [[8.123456 + i, -.678912 + i / 3, -16.234567] for i in range(12)]}


def station_node(config, directory, mode='protect'):
    node = object.__new__(ros.Station)
    node.config, node.mission = copy.deepcopy(config), mission(mode)
    node.state_path = Path(directory) / 'state.json'
    node.control_path = Path(directory) / 'control.json'
    node.results_path = Path(directory) / 'live.jsonl'
    node.telemetry, node.unit_missions = {}, {}
    node.exploration_cells = set()
    node.diver_pilot = DiverPilot()
    node.tracker, node.map = MultiObjectTracker(), PersistentMap()
    node.diver, node.network, node.metrics = [0., 0., -10.], {}, {}
    node.control_nonce, node.last_time = None, -1
    node.mqtt = SimpleNamespace(connected=False, publish=Mock())
    node.mission_pub, node.current_pub = Publisher(), Publisher()
    node.control_client = SimpleNamespace(service_is_ready=lambda: True, call_async=Mock())
    node.visualize = Mock()
    attach_clock(node)
    return node


@unittest.skipIf(ros is None, 'Source the installed ROS setup to run adapter integration tests')
class RosIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((ROOT / 'config' / 'mission.yaml').read_text())

    def test_actual_packets_fit_slots_and_deliver_contacts_maps_and_claims(self):
        for mode in ('protect', 'explore'):
            with self.subTest(mode=mode):
                node = edge_node(self.config, mode)
                for index in range(12):
                    node.sim_time = 20 + index * 1.21
                    node.measurement(ros.encode(instrument_packet(node.sim_time, index)))
                    node.actuate()
                packets = [json.loads(message.data) for message in node.tx.messages]
                self.assertEqual(len(packets), 12)
                contact_packets = sum(bool(item['payload'].get('o')) for item in packets)
                map_packets = sum(bool(item['payload'].get('mp')) for item in packets)
                claims = sum(bool(item['payload'].get('m')) for item in packets)
                self.assertGreaterEqual(contact_packets, 6)
                self.assertGreaterEqual(map_packets, 4)
                self.assertGreaterEqual(claims, 3)
                channel = AcousticTDMA(loss_probability=0)
                for packet in packets:
                    payload = packet['payload']
                    self.assertLessEqual(len(json.dumps(payload, separators=(',', ':')).encode()), 390)
                    self.assertTrue(channel.enqueue(packet['source'], payload, packet['time']))
                    self.assertNotIn('hydrophone_samples', payload)
                predictions = json.loads(node.predictions.messages[-1].data)['observations']
                self.assertEqual({value['k'] for value in predictions}, {2, 3})
                self.assertEqual(node.agent.planner.mode, mode)
                self.assertLessEqual(np.linalg.norm(node.desired_velocity), self.config['speed_limit'] + 1e-8)

    def test_mode_change_uses_same_map_and_new_survey_goal(self):
        node = edge_node(self.config)
        node.measurement(ros.encode(instrument_packet(node.sim_time)))
        first_goal = node.agent.last_result['target']
        observed = len(node.agent.map.cells)
        node.on_mission(ros.encode({**node.mission, 'mode': 'explore', 'mode_revision': 1}))
        node.sim_time += 1.21
        node.measurement(ros.encode(instrument_packet(node.sim_time, 1)))
        self.assertEqual(node.agent.planner.mode, 'explore')
        self.assertGreaterEqual(len(node.agent.map.cells), observed)
        self.assertGreater(np.linalg.norm(np.array(node.agent.last_result['target']) - first_goal), 1)

    def test_pause_immediately_stops_thrust_and_reset_clears_transmit_cursors(self):
        node = edge_node(self.config)
        node.measurement(ros.encode(instrument_packet(node.sim_time)))
        node.actuate()
        self.assertTrue(any(abs(publisher.messages[-1].data) > 0 for publisher in node.thruster_pub))
        node.on_mission(ros.encode({**node.mission, 'status': 'paused'}))
        self.assertTrue(all(publisher.messages[-1].data == 0 for publisher in node.thruster_pub))
        np.testing.assert_array_equal(node.desired_velocity, np.zeros(3))
        node.sim_time += .2
        node.measurement(ros.encode(instrument_packet(node.sim_time, 1)))
        node.actuate()
        self.assertTrue(all(publisher.messages[-1].data == 0 for publisher in node.thruster_pub))
        node.observation_cursor, node.map_cursor = 7, 21
        node.peers['auv_1'] = {'t': node.sim_time}
        node.on_mission(ros.encode({**node.mission, 'epoch': 1, 'status': 'running'}))
        self.assertEqual((node.sequence, node.observation_cursor, node.map_cursor), (0, 0, 0))
        self.assertFalse(node.peers)
        self.assertIsNone(node.navigation)

    def test_reordered_peer_reports_do_not_rewind_navigation_or_claims(self):
        node = edge_node(self.config, 'explore')
        node.sim_time = 15
        fresh = {'t': 14., 'p': [-8, 8, -10], 'v': [.2, 0, 0], 'q': 8,
                 'm': {'m': 'explore', 'g': [-9, 9, -10], 's': 'survey_transit'}}
        node.received(ros.encode({'source': 'auv_1', 'payload': fresh}))
        stale = {**fresh, 't': 12., 'q': 7, 'p': [0, 0, -10]}
        node.received(ros.encode({'source': 'auv_1', 'payload': stale}))
        self.assertEqual(node.peers['auv_1']['q'], 8)
        self.assertEqual(node.peers['auv_1']['p'], [-8, 8, -10])
        refreshed = {key: value for key, value in fresh.items() if key != 'm'}
        refreshed.update(t=15., q=9)
        node.received(ros.encode({'source': 'auv_1', 'payload': refreshed}))
        self.assertEqual(node.peers['auv_1']['mission']['reported_at'], 14.)
        node.sim_time = 19
        node.measurement(ros.encode(instrument_packet(node.sim_time)))
        self.assertEqual(node.agent.last_result['peer_count'], 0)

    def test_navigation_heartbeats_do_not_extend_an_old_target_claim(self):
        baseline = edge_node(self.config, 'explore')
        baseline.sim_time = 19
        baseline.measurement(ros.encode(instrument_packet(19)))
        target = baseline.agent.last_result['target']
        old = edge_node(self.config, 'explore')
        old.sim_time = 15
        payload = {'t': 14., 'q': 1, 'p': [-18, 18, -10], 'v': [0, 0, 0],
                   'm': {'m': 'explore', 'g': target, 's': 'survey_transit'}}
        old.received(ros.encode({'source': 'auv_1', 'payload': payload}))
        old.sim_time = 17.9
        heartbeat = {key: value for key, value in payload.items() if key != 'm'}
        heartbeat.update(t=17.9, q=2)
        old.received(ros.encode({'source': 'auv_1', 'payload': heartbeat}))
        old.sim_time = 19
        old.measurement(ros.encode(instrument_packet(19)))
        np.testing.assert_allclose(old.agent.last_result['target'], target)
        fresh = edge_node(self.config, 'explore')
        fresh.sim_time = 19
        payload.update(t=18.9)
        fresh.received(ros.encode({'source': 'auv_1', 'payload': payload}))
        fresh.measurement(ros.encode(instrument_packet(19)))
        self.assertGreater(np.linalg.norm(np.array(fresh.agent.last_result['target']) - target), 3)

    def test_optional_surfacing_keeps_total_velocity_bounded(self):
        node = edge_node(self.config)
        node.config['gateway']['require_surface'] = True
        node.sim_time = 92
        packet = instrument_packet(92)
        packet['velocity'] = [1.3, 0., 0.]
        node.measurement(ros.encode(packet))
        self.assertGreater(node.desired_velocity[2], 0)
        self.assertLessEqual(np.linalg.norm(node.desired_velocity), self.config['speed_limit'] + 1e-8)

    def test_station_controls_current_pause_reset_and_reject_old_manual_input(self):
        with tempfile.TemporaryDirectory() as directory:
            node = station_node(self.config, directory)

            def command(nonce, **value):
                node.control_path.write_text(json.dumps({'nonce': nonce, **value}))
                node.controls()

            command(1, command='set_diver_control', mode='manual', velocity=[.4, 0, 0], issued_at=100)
            self.assertEqual(node.diver_pilot.velocity, [0, 0, 0])
            for index, issued in enumerate((None, 'wrong', math.inf, 10**400), start=2):
                command(index, command='set_diver_control', mode='manual', velocity=[.4, 0, 0], issued_at=issued)
                self.assertEqual(node.diver_pilot.velocity, [0, 0, 0])
            with patch.object(ros.time, 'time', return_value=1000):
                command(6, command='set_diver_control', mode='manual', velocity=[.4, 0, 0], issued_at=1002)
                self.assertEqual(node.diver_pilot.velocity, [0, 0, 0])
                command(7, command='set_diver_control', mode='manual', velocity=[.4, 0, 0], issued_at=1000)
            self.assertEqual(node.diver_pilot.velocity, [.4, 0, 0])
            command(8, command='pause')
            self.assertEqual(node.diver_pilot.velocity, [0, 0, 0])
            self.assertTrue(node.control_client.call_async.call_args.args[0].world_control.pause)
            command(9, command='set_current', vector=[.2, -.1, .05])
            node.tick()
            current = node.current_pub.messages[-1]
            self.assertEqual([current.x, current.y, current.z], [.2, -.1, .05])
            command(10, command='set_mode', mode='explore')
            self.assertEqual(node.mission['mode'], 'explore')
            self.assertEqual(node.mission['mode_revision'], 1)
            node.telemetry['auv_0'] = {'q': 1}
            command(11, command='reset')
            self.assertFalse(node.telemetry)
            self.assertEqual(node.mission['epoch'], 1)
            self.assertTrue(node.control_client.call_async.call_args.args[0].world_control.reset.all)

    def test_manual_diver_world_velocity_is_converted_to_model_frame(self):
        publisher = Publisher()
        node = SimpleNamespace(states={'diver': {'position': np.array([0., 0., -10.]), 'yaw': math.pi / 2}},
                               mission={'status': 'running', 'diver_control': {'mode': 'manual', 'velocity': [.6, 0, 0]}},
                               cmd={'diver': publisher})
        ros.Instruments.actor_commands(node, 10)
        velocity = publisher.messages[-1].linear
        self.assertAlmostEqual(velocity.x, 0)
        self.assertAlmostEqual(velocity.y, -.6)
        node.states['diver']['position'][0] = 27
        ros.Instruments.actor_commands(node, 10)
        self.assertAlmostEqual(publisher.messages[-1].linear.y, 0)

    def test_pause_immediately_stops_manual_and_scripted_actors(self):
        node = object.__new__(ros.Instruments)
        node.mission = mission()
        node.mission['diver_control'] = {'mode': 'manual', 'velocity': [.6, 0, 0]}
        names = ('diver', 'intruder', 'marine_life', 'large_marine_life')
        node.states = {name: {'position': np.array([0., 0., -10.]), 'yaw': .4} for name in names}
        node.cmd = {name: Publisher() for name in names}
        node.actor_commands(10)
        node.on_mission(ros.encode({**node.mission, 'status': 'paused'}))
        for publisher in node.cmd.values():
            velocity = publisher.messages[-1]
            self.assertEqual([velocity.linear.x, velocity.linear.y, velocity.linear.z, velocity.angular.z], [0, 0, 0, 0])
        node.actor_commands(10)
        self.assertTrue(all(publisher.messages[-1].linear.x == 0 for publisher in node.cmd.values()))

    def test_exploration_alerts_reference_vehicle_not_parked_diver(self):
        with tempfile.TemporaryDirectory() as directory:
            node = station_node(self.config, directory, 'explore')
            node.telemetry['auv_0'] = {'t': 10., 'p': [12., 0., -10.], 'v': [0., 0., 0.], 'r': .04}
            node.diver = [-20., -20., -7.]
            node.tracker.update([Observation('contact:1', 'auv_0', 10., [13., 0., -10.],
                                           np.eye(3) * .04, 'marine_life', .9)], 10.)
            node.tick()
            state = json.loads(node.state_path.read_text())
            self.assertEqual(state['alerts'][0]['severity'], 'critical')
            self.assertEqual(state['alerts'][0]['scope'], 'vehicle_navigation')
            self.assertNotIn('protected zone', state['alerts'][0]['message'])


if __name__ == '__main__':
    unittest.main()
