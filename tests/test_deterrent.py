import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "underwater_system"))
from underwater_system.core import SimulationCore, assess_alerts
from underwater_system.deterrent import (DeterrentSpeaker, WildlifeResponse,
                                         received_level_db, source_level_for, time_to_enter)


def fish(position, velocity, label="marine_life", confidence=0.9):
    return {"id": "t1", "position": position, "velocity": velocity,
            "classification": label, "confidence": confidence}


class DeterrentTests(unittest.TestCase):
    def test_level_round_trip_and_spreading(self):
        self.assertAlmostEqual(received_level_db(source_level_for(110, 12), 12), 110, places=6)
        self.assertGreater(received_level_db(160, 5), received_level_db(160, 20))

    def test_time_to_enter(self):
        self.assertAlmostEqual(time_to_enter([20, 0, 0], [-1, 0, 0], 4), 16.0)
        self.assertIsNone(time_to_enter([20, 0, 0], [1, 0, 0], 4))
        self.assertEqual(time_to_enter([1, 0, 0], [0, 0, 0], 4), 0.0)

    def test_speaker_fires_on_approaching_wildlife_only(self):
        speaker = DeterrentSpeaker()
        diver = np.array([0.0, 0, -8])
        me = np.array([-5.0, 0, -8])
        coming = fish([-15, 0, -8], [0.8, 0, 0])
        self.assertTrue(speaker.decide(0, me, diver, np.zeros(3), [coming])["active"])
        speaker = DeterrentSpeaker()
        self.assertFalse(speaker.decide(0, me, diver, np.zeros(3), [fish([-15, 0, -8], [-0.8, 0, 0])])["active"])
        self.assertFalse(speaker.decide(0, me, diver, np.zeros(3), [fish([-15, 0, -8], [0.8, 0, 0], "vehicle")])["active"])

    def test_diver_exposure_is_capped(self):
        speaker = DeterrentSpeaker()
        diver = np.array([0.0, 0, -8])
        result = speaker.decide(0, [-4.0, 0, -8], diver, np.zeros(3), [fish([-18, 0, -8], [0.8, 0, 0])])
        if result["active"]:
            self.assertLessEqual(result["received_at_diver_db"], speaker.config.diver_limit_db + 0.1)

    def test_duty_cycle(self):
        speaker = DeterrentSpeaker()
        diver, me = np.array([0.0, 0, -8]), np.array([-5.0, 0, -8])
        threat = [fish([-15, 0, -8], [0.8, 0, 0])]
        states = [speaker.decide(t, me, diver, np.zeros(3), threat)["active"] for t in np.arange(0, 14, 1.0)]
        self.assertTrue(states[0] and not all(states))

    def test_wildlife_flees_and_habituates(self):
        response = WildlifeResponse()
        speakers = [{"position": [0, 0, -8], "level_db": 165.0}]
        v = response.velocity("fish", [-6, 0, -8], speakers, 1.0)
        self.assertLess(v[0], 0)
        for _ in range(60):
            response.velocity("fish", [-6, 0, -8], speakers, 1.0)
        self.assertLess(np.linalg.norm(response.velocity("fish", [-6, 0, -8], speakers, 1.0)), np.linalg.norm(v))
        self.assertTrue(np.allclose(response.velocity("far", [-600, 0, -8], speakers, 1.0), 0))

    def test_early_warning_reports_time_to_zone(self):
        track = {"id": "x", "position": [30, 0, -7], "velocity": [-1, 0, 0], "classification": "vehicle",
                 "confidence": 0.9, "position_std_m": 0.3}
        alert = assess_alerts([track], [0, 0, -7])[0]
        self.assertEqual(alert["severity"], "warning")
        self.assertAlmostEqual(alert["time_to_zone_s"], 25.4, delta=0.7)
        self.assertEqual(alert["warning_stage"], "early")

    def test_speakers_keep_animals_farther_from_diver(self):
        def closest(enabled):
            sim = SimulationCore(seed=7, config={"deterrent": {"enabled": enabled}})
            best = np.inf
            for _ in range(600):
                sim.step(0.2)
                best = min(best, min(np.linalg.norm(o.position - sim.objects["diver"].position)
                                     for o in sim.objects.values() if o.kind == "marine_life"))
            return best, sim.evaluate()
        off, _ = closest(False)
        on, metrics = closest(True)
        self.assertGreater(on, off + 2.0)
        self.assertGreater(metrics["deterrent_activations"], 0)


if __name__ == "__main__":
    unittest.main()
