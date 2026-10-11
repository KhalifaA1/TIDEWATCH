"""Meaningful algorithm checks; runnable without ROS, Gazebo or GPU libraries."""
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "underwater_system"))
from underwater_system.core import (AcousticTDMA, EdgeAgent, FormationController,
                                    MultiObjectTracker, Observation, PersistentMap,
                                    SensorModel, SimulationCore, WorldObject,
                                    assess_alerts, local_to_world, world_to_local)


def observation(uid="a:1", source="a", timestamp=0, position=(6, 0, -7),
                classification="vehicle"):
    return Observation(uid, source, timestamp, position, np.eye(3) * 0.04,
                       classification, 0.90)


class CoreTests(unittest.TestCase):
    def test_coordinate_roundtrip_and_covariance(self):
        point, origin, yaw = [4, -2, 1], [7, 8, -9], 1.31
        np.testing.assert_allclose(world_to_local(local_to_world(point, origin, yaw), origin, yaw), point)
        sensor = SensorModel(seed=4)
        obj = WorldObject("vehicle", "vehicle", [11, 8, -9])
        samples = sensor.observe("auv_0", origin, 0, [obj], 0)
        self.assertEqual(len(samples), 1)
        self.assertGreater(np.min(np.linalg.eigvalsh(samples[0].covariance)), 0)
        self.assertLess(np.linalg.norm(samples[0].position - obj.position), 1)
        self.assertNotIn("truth_id", samples[0].to_payload())

    def test_unknown_and_passive_never_invent_position(self):
        sensor = SensorModel(seed=3)
        features = sensor.passive_features("unknown", 10)
        label, confidence = sensor.classify(features, "unknown")
        self.assertEqual(label, "unknown")
        self.assertLess(confidence, sensor.unknown_threshold)
        self.assertNotIn("position", features)

    def test_multisensor_tracking_and_repeat_dedup(self):
        tracker = MultiObjectTracker()
        a = observation()
        b = observation("b:1", "b", position=(6.05, 0, -7))
        tracker.update([a, b], 0)
        self.assertEqual(len(tracker.tracks), 1)
        tracker.update([a, Observation.from_payload(b.to_payload())], 0.1)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.duplicates, 2)
        self.assertEqual(next(iter(tracker.tracks.values())).hits, 2)
        tracker.update([observation("a:2", timestamp=1, position=(5.5, 0, -7))], 1)
        track = next(iter(tracker.tracks.values()))
        self.assertLess(track.velocity[0], -0.1)
        self.assertGreaterEqual(len(track.history), 3)
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(track.covariance)), 0)

    def test_distinct_same_source_detections_do_not_merge(self):
        tracker = MultiObjectTracker()
        tracker.update([observation("a:1"), observation("a:2", position=(6.5, 0, -7))], 0)
        self.assertEqual(len(tracker.tracks), 2)

    def test_delayed_reordered_bursts_preserve_moving_ascending_track(self):
        tracker = MultiObjectTracker()
        rng = np.random.default_rng(19)
        origin, velocity = np.array([5.0, 2.0, -20]), np.array([0.5, 0, 0.65])
        deliveries = {}
        for source_index, source in enumerate(("auv_0", "auv_1", "auv_2")):
            for index, stamp in enumerate(np.arange(0, 20.01, 0.6)):
                delay = (0.15, 1.4, 3.1)[(index + source_index) % 3]
                arrival = math.ceil((stamp + delay) / 3) * 3
                measured = origin + velocity * stamp + rng.normal(0, 0.07, 3)
                item = Observation(f"{source}:{index}", source, float(stamp), measured,
                                   np.eye(3) * 0.07 ** 2, "marine_life", 0.9)
                deliveries.setdefault(arrival, []).append(item)
        identity = None
        for now in sorted(deliveries):
            batch = deliveries[now]
            rng.shuffle(batch)
            tracker.update(batch, now)
            self.assertEqual(len(tracker.tracks), 1, f"Ghost track in burst at {now}s")
            track = next(iter(tracker.tracks.values()))
            identity = identity or track.track_id
            self.assertEqual(track.track_id, identity)
            self.assertLess(np.linalg.norm(track.position - (origin + velocity * now)), 0.6)
            self.assertLess(np.linalg.norm(track.velocity - velocity), 0.12)
            self.assertGreaterEqual(np.min(np.linalg.eigvalsh(track.covariance)), 0)

    def test_delayed_same_class_simultaneous_contacts_stay_separate(self):
        tracker = MultiObjectTracker()
        velocity = np.array([0.5, 0, 0.65])
        origins = [np.array([5, 1, -20.0]), np.array([5, 3.2, -20.0])]
        for now in (2, 4, 6, 8):
            batch = []
            for source in ("auv_0", "auv_1"):
                for index, origin in enumerate(origins):
                    stamp = now - 1.5
                    batch.append(Observation(f"{source}:{now}:{index}", source, stamp,
                                             origin + velocity * stamp, np.eye(3) * 0.04,
                                             "marine_life", 0.9))
            tracker.update(batch, now)
            self.assertEqual(len(tracker.tracks), 2)
        for origin in origins:
            self.assertLess(min(np.linalg.norm(track.position - origin - velocity * 8)
                                for track in tracker.tracks.values()), 0.2)

    def test_map_unknown_cells_stay_unobserved(self):
        mapping = PersistentMap()
        self.assertEqual(mapping.snapshot()["observed_voxels"], 0)
        mapping.update([[5, 0, -16]], [0, 0, -7], 1)
        self.assertGreater(mapping.snapshot()["observed_voxels"], 0)
        self.assertNotIn((25, 25, -16), mapping.cells)
        self.assertEqual(len(mapping.elevation), 1)

    def test_tdma_byte_limit_outage_recovery_and_duplicate_suppression(self):
        network = AcousticTDMA(seed=2, loss_probability=0, duplicate_probability=1,
                               slot_duration=0.2, delay_range=(0.01, 0.01))
        self.assertFalse(network.enqueue("auv_0", {"audio": [0.0]}, 0))
        self.assertFalse(network.enqueue("auv_0", {"hydrophone_samples": [0.0] * 32}, 0))
        self.assertFalse(network.enqueue("auv_0", {"large": "x" * 1000}, 0))
        network.set_connected("auv_0", False)
        self.assertTrue(network.enqueue("auv_0", {"p": [0, 0, -7]}, 0))
        self.assertEqual(network.step(0.5), [])
        network.set_connected("auv_0", True)
        network.step(0.6)
        packets = network.step(1.0)
        self.assertEqual(len(packets), 1)
        self.assertEqual(network.snapshot()["reconnections"], 1)
        self.assertEqual(network.snapshot()["duplicate_discarded"], 1)
        self.assertEqual(network.snapshot()["delivery_ratio"], 1)

    def test_tdma_expired_queue_does_not_replay_stale_reports(self):
        network = AcousticTDMA(loss_probability=0, ttl=0.2)
        network.set_connected("auv_0", False)
        network.enqueue("auv_0", {"p": [0, 0, -7]}, 0)
        network.step(1.3)
        network.set_connected("auv_0", True)
        self.assertEqual(network.step(3), [])
        self.assertEqual(network.snapshot()["expired"], 1)

    def test_reconnection_supersedes_backlog_with_latest_state(self):
        network = AcousticTDMA(loss_probability=0, duplicate_probability=0,
                               ttl=4, delay_range=(0.01, 0.01))
        network.set_connected("auv_0", False)
        for index in range(4):
            network.enqueue("auv_0", {"p": [index, 0, -7]}, index * 0.3)
        network.step(1)
        network.set_connected("auv_0", True)
        self.assertEqual(len(network.queues["auv_0"]), 1)
        self.assertEqual(network.snapshot()["superseded_on_reconnect"], 3)
        network.step(1.2)
        recovered = network.step(1.5)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["payload"]["p"][0], 3)

    def test_controller_speed_and_collision_separation(self):
        controllers = [FormationController(i) for i in range(3)]
        diver = np.array([0, 0, -7.0])
        positions = [controller.target(diver) for controller in controllers]
        velocities = [np.zeros(3) for _ in range(3)]
        for _ in range(100):
            commands = []
            for index, controller in enumerate(controllers):
                peers = [(positions[j], velocities[j]) for j in range(3) if j != index]
                command = controller.command(positions[index], velocities[index], diver,
                                             np.zeros(3), peers, dt=0.1)
                self.assertLessEqual(np.linalg.norm(command), 1.3 + 1e-9)
                commands.append(command)
            velocities = commands
            positions = [p + v * 0.1 for p, v in zip(positions, velocities)]
        for i in range(3):
            self.assertGreater(np.linalg.norm(positions[i] - diver), 5.5)
            for j in range(i + 1, 3):
                self.assertGreater(np.linalg.norm(positions[i] - positions[j]), 9)

    def test_risk_separate_from_classification(self):
        track = {"id": "track-1", "position": [10, 0, -7], "velocity": [-1, 0, 0],
                 "classification": "vehicle", "confidence": 0.95, "position_std_m": 0.2}
        alert = assess_alerts([track], [0, 0, -7])[0]
        self.assertEqual(alert["severity"], "warning")
        self.assertFalse(alert["hostility_determined"])
        track["velocity"] = [1, 0, 0]
        self.assertEqual(assess_alerts([track], [0, 0, -7])[0]["severity"], "info")

    def test_edge_uses_estimates_and_mesh_only(self):
        edge = EdgeAgent(0)
        result = edge.step(1, 0.1, [6, 0, -7], [0, 0, 0], [0, 0, -7], [0, 0, 0],
                           observations=[observation("auv_0:1", "auv_0", 1).to_payload()],
                           peer_telemetry=[{"source": "auv_1", "payload":
                                            {"t": 1, "p": [-3, 5.2, -7], "v": [0, 0, 0]}}])
        self.assertEqual(result["peer_count"], 1)
        self.assertEqual(len(result["tracks"]), 1)
        self.assertLessEqual(np.linalg.norm(result["command_velocity"]), 1.3 + 1e-9)
        json.dumps(result)

    def test_exploration_ignores_diver_beacon_and_moves_to_survey_target(self):
        agents = [EdgeAgent(0, mission_mode="explore") for _ in range(2)]
        position = np.array([6.0, 0, -7])
        velocities = [np.zeros(3), np.zeros(3)]
        poses = [position.copy(), position.copy()]
        for tick in range(45):
            for index, edge in enumerate(agents):
                result = edge.step(tick * 0.2, 0.2, poses[index], velocities[index],
                                   [index * 100, -index * 50, -7], [index, 0, 0], observations=[])
                velocities[index] = np.asarray(result["command_velocity"])
                poses[index] += velocities[index] * 0.2
                self.assertEqual(result["mission"]["mode"], "explore")
                self.assertLessEqual(np.linalg.norm(velocities[index]), 1.3 + 1e-9)
        np.testing.assert_allclose(poses[0], poses[1])
        self.assertGreater(np.linalg.norm(poses[0] - position), 2)

    def test_explorers_respect_decoded_ros_target_claims(self):
        args = (0, 0.2, [6, 0, -7], [0, 0, 0], [0, 0, -7], [0, 0, 0])
        unclaimed = EdgeAgent(0, mission_mode="explore").step(*args, observations=[])
        claimed_target = unclaimed["target"]
        peer = {"source": "auv_1", "t": 0, "p": [-3, 5, -7], "v": [0, 0, 0],
                "m": {"m": "explore", "g": claimed_target},
                "mission": {"mode": "explore", "target": claimed_target}}
        coordinated = EdgeAgent(0, mission_mode="explore").step(*args, observations=[], peer_telemetry=[peer])
        self.assertGreater(np.linalg.norm(np.asarray(coordinated["target"]) - claimed_target), 3)

    def test_mode_switch_preserves_pose_tracks_and_measured_map(self):
        core = SimulationCore(seed=3)
        core.step(1)
        positions = {name: state["position"].copy() for name, state in core.auvs.items()}
        mapped = len(core.map.cells)
        tracks = len(core.tracker.tracks)
        core.set_mode("explore")
        self.assertEqual(len(core.map.cells), mapped)
        self.assertEqual(len(core.tracker.tracks), tracks)
        for name, state in core.auvs.items():
            np.testing.assert_array_equal(state["position"], positions[name])
        snapshot = core.step(8)
        self.assertEqual(snapshot["mission"]["mode"], "explore")
        self.assertGreater(snapshot["mission"]["observed_cells"], 0)
        self.assertIsNone(snapshot["metrics"]["formation_rmse_m"])
        core.set_mode("protect")
        self.assertEqual(core.step(0.2)["mission"]["mode"], "protect")
        json.dumps(core.snapshot(), allow_nan=False)

    def test_predicted_wildlife_crossing_changes_bounded_command(self):
        controller = FormationController(0)
        args = ([0, 0, -10], [1, 0, 0], [0, 0, -7], [0, 0, 0])
        options = {"dt": 0.5, "navigation_goal": [12, 0, -10], "protect_diver": False}
        clear = controller.command(*args, **options)
        evasive = controller.command(*args, obstacles=[([4, -4, -10], 2.0, [0, 1, 0])], **options)
        self.assertLessEqual(np.linalg.norm(evasive), 1.3 + 1e-9)
        self.assertGreater(np.linalg.norm(evasive - clear), 0.2)
        self.assertGreater(evasive[1], clear[1])

    def test_exploration_core_has_actual_motion_and_safe_separation(self):
        core = SimulationCore(seed=8, mode="explore")
        initial = {name: state["position"].copy() for name, state in core.auvs.items()}
        for _ in range(60):
            core.step(0.2)
        for name, state in core.auvs.items():
            self.assertGreater(np.linalg.norm(state["position"] - initial[name]), 2)
        self.assertGreaterEqual(core.evaluate()["minimum_auv_separation_m"], 2.4)
        self.assertGreater(core.mission_status()["coverage_pct"], 0)

    def test_external_motion_authority_and_json_snapshot(self):
        core = SimulationCore(seed=3)
        states = {"auv_0": {"position": [6, 0, -7], "velocity": [0, 0, 0]}}
        core.step(0.1, model_states=states)
        core.step(1)
        np.testing.assert_allclose(core.auvs["auv_0"]["position"], [6, 0, -7])
        self.assertEqual(core.evaluate()["motion_authority"], "external Gazebo poses")
        json.dumps(core.snapshot(), allow_nan=False)

    def test_seed_reproducibility(self):
        a, b = SimulationCore(seed=8), SimulationCore(seed=8)
        for _ in range(10):
            a.step(0.1)
            b.step(0.1)
        self.assertEqual(a.snapshot(), b.snapshot())


if __name__ == "__main__":
    unittest.main()
