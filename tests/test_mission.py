"""Mission behaviour without ROS, Gazebo or access to world truth."""
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "underwater_system"))
from underwater_system.mission import MissionPlanner


def survey_config(**options):
    return {"mission_mode": "explore", "exploration": options}


def contact(now, *, kind="vehicle", position=(0, 14, -10), velocity=(0, -1, 0)):
    return {"id": "local-contact", "position": position, "velocity": velocity,
            "classification": kind, "confidence": 0.9, "position_std_m": 0.1,
            "last_seen": now}


class MissionTests(unittest.TestCase):
    def test_exploration_does_not_depend_on_diver(self):
        first = MissionPlanner(0, survey_config())
        second = MissionPlanner(0, survey_config())
        for now in (0, 1, 2):
            a = first.plan(now, [6, 0, -10], diver=[0, 0, -10], rays=[[8, 2, -20]])
            b = second.plan(now, [6, 0, -10], diver=[900, -300, 50], rays=[[8, 2, -20]])
            np.testing.assert_allclose(a.goal, b.goal)
            self.assertEqual(a.status, b.status)
        self.assertEqual(a.status["mode"], "explore")
        self.assertTrue(a.status["role"].startswith("survey_sector"))

    def test_independent_vehicles_get_distinct_tasks_during_outage(self):
        plans = []
        for index in range(3):
            angle = index * 2 * math.pi / 3
            position = [6 * math.cos(angle), 6 * math.sin(angle), -10]
            planner = MissionPlanner(index, survey_config())
            plans.append(planner.plan(0, position, peers=[]))
        for index, plan in enumerate(plans):
            bearing = math.atan2(plan.goal[1], plan.goal[0])
            difference = (bearing - index * 2 * math.pi / 3 + math.pi) % (2 * math.pi) - math.pi
            self.assertLess(abs(difference), math.pi / 3)
            for other in plans[index + 1:]:
                self.assertGreater(np.linalg.norm(plan.goal - other.goal), 4)

    def test_only_measured_hits_count_as_coverage(self):
        planner = MissionPlanner(0, survey_config())
        first = planner.plan(0, [6, 0, -10])
        self.assertEqual(first.status["observed_cells"], 0)
        self.assertEqual(first.status["progress"], 0)
        measured = planner.plan(1, [6, 0, -10], rays=[[8, 2, -20], [8.1, 2.1, -20],
                                                   [200, 100, -20], [float("nan"), 0, -20]])
        self.assertEqual(measured.status["observed_cells"], 1)
        self.assertEqual(len(planner.observed_cells), 1)
        self.assertGreater(measured.status["progress"], 0)
        same = planner.plan(2, [6, 0, -10], rays=[])
        self.assertEqual(same.status["observed_cells"], 1)
        json.dumps(same.status, allow_nan=False)

    def test_target_stays_stable_while_new_rays_arrive(self):
        planner = MissionPlanner(0, survey_config())
        first = planner.plan(0, [6, 0, -10])
        for now in np.arange(0.2, 4.5, 0.2):
            plan = planner.plan(float(now), [6.05, 0.03, -10], rays=[[10 + now, 4, -20]])
            np.testing.assert_allclose(plan.goal, first.goal)
        self.assertEqual(plan.status["waypoints_completed"], 0)

    def test_peer_claim_age_does_not_refresh_with_navigation_packets(self):
        baseline = MissionPlanner(0, survey_config()).plan(10, [6, 0, -10])
        claim = {"mode": "explore", "target": baseline.goal.tolist(), "reported_at": 9}
        peer = {"t": 10, "mission": claim}
        fresh = MissionPlanner(0, survey_config()).plan(10, [6, 0, -10], peers=[peer])
        self.assertGreater(np.linalg.norm(fresh.goal - baseline.goal), 3)
        claim["reported_at"] = 0
        expired = MissionPlanner(0, survey_config()).plan(10, [6, 0, -10], peers=[peer])
        np.testing.assert_allclose(expired.goal, baseline.goal)

    def test_arrival_requires_a_scan_dwell_before_advancing(self):
        planner = MissionPlanner(0, survey_config(dwell_time=2.5))
        first = planner.plan(0, [6, 0, -10])
        arriving = planner.plan(1, first.goal)
        self.assertEqual(arriving.status["phase"], "survey_dwell")
        waiting = planner.plan(2, first.goal + [1.7, 0, 0])
        np.testing.assert_allclose(waiting.goal, first.goal)
        self.assertEqual(waiting.status["waypoints_completed"], 0)
        next_task = planner.plan(4, first.goal)
        self.assertEqual(next_task.status["waypoints_completed"], 1)
        self.assertGreater(np.linalg.norm(next_task.goal - first.goal), 3)

    def test_stalled_task_eventually_replans(self):
        planner = MissionPlanner(0, survey_config(target_timeout=8))
        first = planner.plan(0, [6, 0, -10])
        held = planner.plan(4, [6, 0, -10])
        np.testing.assert_allclose(held.goal, first.goal)
        replacement = planner.plan(9, [6, 0, -10])
        self.assertGreater(np.linalg.norm(replacement.goal - first.goal), 3)
        self.assertEqual(replacement.status["waypoints_completed"], 0)

    def test_time_reserve_returns_to_estimated_launch_position(self):
        planner = MissionPlanner(0, survey_config(max_duration=12))
        home = np.array([6, 0, -10.0])
        first = planner.plan(0, home)
        returning = planner.plan(11, first.goal)
        self.assertEqual(returning.status["phase"], "return")
        self.assertEqual(returning.status["return_reason"], "time_reserve")
        np.testing.assert_allclose(returning.goal, home)
        arrived = planner.plan(12, home)
        self.assertEqual(arrived.status["phase"], "complete")

    def test_distance_budget_reserves_the_trip_home(self):
        planner = MissionPlanner(0, survey_config(distance_budget=20))
        home = [6, 0, -10]
        planner.plan(0, home)
        returning = planner.plan(1, [19, 0, -10])
        self.assertEqual(returning.status["return_reason"], "distance_reserve")
        np.testing.assert_allclose(returning.goal, home)

    def test_uncertainty_holds_then_returns_without_resuming_survey(self):
        planner = MissionPlanner(0, survey_config())
        planner.plan(0, [6, 0, -10])
        hold = planner.plan(1, [10, 0, -10], uncertainty=6)
        self.assertEqual(hold.status["phase"], "localization_hold")
        np.testing.assert_allclose(hold.goal, [10, 0, -10])
        returning = planner.plan(2, [10, 0, -10], uncertainty=0.2)
        self.assertEqual(returning.status["phase"], "return")
        self.assertEqual(returning.status["return_reason"], "localization_uncertainty")
        np.testing.assert_allclose(returning.goal, [6, 0, -10])

    def test_escort_turns_toward_approach_and_keeps_ring_spacing(self):
        planners = [MissionPlanner(index) for index in range(3)]
        beacon = {"position": [0, 0, -10], "velocity": [0, 0, 0]}
        last_goals = None
        for now in np.arange(0, 5.5, 0.5):
            beacon["timestamp"] = float(now)
            plans = [planner.plan(float(now), [6, 0, -10], diver=beacon,
                                  observations=[contact(float(now))]) for planner in planners]
            if last_goals is not None:
                for before, after in zip(last_goals, plans):
                    self.assertLess(np.linalg.norm(after.goal - before), 0.7)
            last_goals = [plan.goal for plan in plans]
        sentinel = plans[1]
        self.assertEqual(sentinel.status["role"], "predictive_sentinel")
        self.assertGreater(sentinel.goal[1], 5.8)
        self.assertLess(abs(sentinel.goal[0]), 0.5)
        self.assertGreater(sentinel.status["risk"]["score"], 0.18)
        for index, plan in enumerate(plans):
            self.assertGreaterEqual(np.linalg.norm(plan.goal - beacon["position"]), 6 - 1e-9)
            for other in plans[index + 1:]:
                self.assertGreater(np.linalg.norm(plan.goal - other.goal), 9)

    def test_threat_crossing_a_sector_edge_does_not_swing_the_ring_back(self):
        # A threat moving from 50 to 70 degrees crosses the 60 degree edge
        # between guard sectors. The ring should keep turning the short way
        # (+20 degrees), not reverse by about 100 degrees.
        planner = MissionPlanner(0)
        beacon = {"position": [0, 0, -10], "velocity": [0, 0, 0]}

        def threat(now, degrees):
            angle = math.radians(degrees)
            heading = np.array([math.cos(angle), math.sin(angle), 0.0])
            return [{"id": "threat", "position": (14 * heading + [0, 0, -10]).tolist(),
                     "velocity": (-0.8 * heading).tolist(), "classification": "vehicle",
                     "confidence": 0.9, "position_std_m": 0.1, "last_seen": now}]

        now = 0.0
        for _ in range(80):  # settle facing 50 degrees
            beacon["timestamp"] = now
            planner.plan(now, [6, 0, -10], diver=beacon, observations=threat(now, 50))
            now += 0.25
        self.assertAlmostEqual(planner._rotation, math.radians(50), delta=0.05)
        previous = planner._rotation
        for _ in range(80):  # threat now at 70 degrees, past the sector edge
            beacon["timestamp"] = now
            planner.plan(now, [6, 0, -10], diver=beacon, observations=threat(now, 70))
            self.assertGreaterEqual(planner._rotation - previous, -1e-9)
            previous = planner._rotation
            now += 0.25
        self.assertAlmostEqual(planner._rotation, math.radians(70), delta=0.05)

    def test_ring_rotation_is_reported_as_slot_velocity(self):
        planner = MissionPlanner(1)
        beacon = {"position": [0, 0, -10], "velocity": [0, 0, 0]}
        turning = None
        for now in np.arange(0, 3, 0.5):
            beacon["timestamp"] = float(now)
            turning = planner.plan(float(now), [6, 0, -10], diver=beacon,
                                   observations=[contact(float(now))])
        ring = np.array(turning.status["ring_velocity"])
        radial = turning.goal - np.array(beacon["position"])
        self.assertGreater(np.linalg.norm(ring), 0.1)
        cosine = float(np.dot(ring, radial)) / (np.linalg.norm(ring) * np.linalg.norm(radial))
        self.assertLess(abs(cosine), 1e-3)  # sideways (tangential) only
        # With no threat and the ring back home, the slot is not moving.
        idle = MissionPlanner(1).plan(0, [6, 0, -10], diver=[0, 0, -10])
        self.assertEqual(idle.status["ring_velocity"], [0.0, 0.0, 0.0])

    def test_similar_contacts_do_not_make_the_ring_flip_between_threats(self):
        planner = MissionPlanner(0)
        beacon = {"position": [0, 0, -10], "velocity": [0, 0, 0]}
        chosen = set()
        for step, now in enumerate(np.arange(0, 6, 0.5)):
            beacon["timestamp"] = float(now)
            # Two approaching contacts whose confidence alternates slightly.
            north = {**contact(float(now)), "id": "north",
                     "confidence": 0.90 if step % 2 else 0.86}
            east = {**contact(float(now), position=(13, -5, -10), velocity=(-0.9, 0.35, 0)),
                    "id": "east", "confidence": 0.86 if step % 2 else 0.90}
            plan = planner.plan(float(now), [6, 0, -10], diver=beacon, observations=[north, east])
            chosen.add(plan.status["risk"]["contact_id"])
        self.assertEqual(len(chosen), 1)

    def test_receding_contact_and_stale_contact_have_less_risk(self):
        approaching = MissionPlanner(1).plan(0, [6, 0, -10], diver=[0, 0, -10],
                                             observations=[contact(0)])
        receding = MissionPlanner(1).plan(0, [6, 0, -10], diver=[0, 0, -10],
                                         observations=[contact(0, velocity=(0, 1, 0))])
        stale = MissionPlanner(1).plan(6, [6, 0, -10], diver=[0, 0, -10],
                                      observations=[contact(0)])
        self.assertGreater(approaching.status["risk"]["score"], receding.status["risk"]["score"])
        self.assertEqual(stale.status["risk"]["score"], 0)

    def test_approaching_marine_life_triggers_geometric_risk(self):
        plan = MissionPlanner(1).plan(0, [-3, 5.2, -10], diver=[0, 0, -10],
                                      observations=[contact(0, kind="marine_life")])
        self.assertEqual(plan.status["phase"], "predictive_watch")
        self.assertGreater(plan.status["risk"]["score"], 0.18)
        self.assertNotIn("hostile", json.dumps(plan.status))

    def test_escort_does_not_place_a_goal_on_a_nearby_animal(self):
        position = np.array([6, 0, -10.0])
        animal = contact(0, kind="marine_life", position=position, velocity=(0, 0, 0))
        plan = MissionPlanner(0).plan(0, [3, -4, -10], diver=[0, 0, -10], observations=[animal])
        self.assertGreaterEqual(np.linalg.norm(plan.goal - position), 3.2)
        self.assertGreaterEqual(np.linalg.norm(plan.goal - [0, 0, -10]), 6 - 1e-9)

    def test_stale_diver_beacon_holds_instead_of_chasing_it(self):
        planner = MissionPlanner(0)
        beacon = {"position": [0, 0, -10], "timestamp": 0}
        stale = planner.plan(5, [8, 0, -10], diver=beacon)
        self.assertEqual(stale.status["phase"], "beacon_hold")
        np.testing.assert_allclose(stale.goal, [8, 0, -10])

    def test_measured_terrain_raises_goal_to_preserve_altitude(self):
        planner = MissionPlanner(0, survey_config())
        first = planner.plan(0, [6, 0, -10])
        hit = first.goal.copy()
        hit[2] = -11
        adjusted = planner.plan(1, [6, 0, -10], rays=[hit])
        self.assertGreaterEqual(adjusted.goal[2], -8.8)
        self.assertLessEqual(adjusted.goal[2], -3)

    def test_switching_modes_resets_budget_and_preserves_measurements(self):
        planner = MissionPlanner(0, survey_config())
        planner.plan(0, [6, 0, -10], rays=[[8, 2, -20]])
        planner.plan(1, [10, 0, -10])
        planner.set_mode("protect")
        protect = planner.plan(2, [10, 0, -10], diver=[0, 0, -10])
        self.assertEqual(protect.status["estimated_distance_m"], 0)
        self.assertEqual(protect.status["observed_cells"], 1)
        planner.reset("explore")
        fresh = planner.plan(0, [6, 0, -10])
        self.assertEqual(fresh.status["observed_cells"], 0)

    def test_invalid_configuration_and_navigation_are_rejected(self):
        for config in ({"mission_mode": "wandering"}, {"formation_radius": 2},
                       survey_config(bounds=[[10, -10], [-20, 20]]),
                       survey_config(depth_layers=[2]), survey_config(uncertainty_limit=float("nan")),
                       survey_config(cell_size=0), survey_config(target_coverage=1.1)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                MissionPlanner(0, config)
        planner = MissionPlanner(0)
        with self.assertRaises(ValueError):
            planner.plan(0, [float("nan"), 0, -10])
        with self.assertRaises(ValueError):
            planner.plan(0, [0, 0, -10], uncertainty=-1)


if __name__ == "__main__":
    unittest.main()
