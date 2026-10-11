"""Evaluation checks cover reproducibility and an observable blackout/rejoin."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "underwater_system"))
from underwater_system.evaluate import run_mission_evaluation, run_scenario


class EvaluationTests(unittest.TestCase):
    def test_seeded_report_is_reproducible(self):
        first = run_scenario("nominal", duration=6, seed=11)
        second = run_scenario("nominal", duration=6, seed=11)
        self.assertEqual(first, second)
        self.assertGreater(first["classification"]["observations"], 10)
        self.assertTrue(first["all_applicable_checks_passed"])
        self.assertGreater(first["uncertainty"]["observations"], 10)

    def test_blackout_expires_peers_and_delivers_after_rejoin(self):
        report = run_scenario("outage_recovery", duration=14, seed=7)
        events = report["fault_events"]
        self.assertEqual(events[0]["event"], "acoustic_link_disconnected")
        self.assertEqual(events[1]["event"], "acoustic_link_reconnected")
        self.assertEqual(events[2]["event"], "first_post_reconnection_delivery")
        self.assertGreaterEqual(events[2]["time_s"], events[1]["time_s"])
        checks = {entry["name"]: entry for entry in report["checks"]}
        self.assertEqual(checks["outage_recovery"]["status"], "passed")
        self.assertEqual(checks["stale_peer_expiry"]["status"], "passed")
        self.assertGreaterEqual(report["metrics"]["minimum_auv_separation_m"], 2.4)

    def test_dual_mission_report_measures_motion_and_switching(self):
        report = run_mission_evaluation(duration=6, seed=11)
        self.assertTrue(report["passed"], report["scenarios"])
        self.assertEqual(set(report["scenarios"]), {"protect", "explore", "mode_switch"})
        exploring = report["scenarios"]["explore"]
        self.assertGreater(exploring["mission"]["coverage_pct"], 0)
        self.assertIsNone(exploring["metrics"]["formation_rmse_m"])
        self.assertEqual([event["mode"] for event in report["scenarios"]["mode_switch"]["mode_events"]],
                         ["explore", "protect"])


if __name__ == "__main__":
    unittest.main()
