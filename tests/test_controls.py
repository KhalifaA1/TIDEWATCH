"""Operator input validation and wall-clock stop behavior."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'underwater_system'))
from underwater_system.controls import DiverPilot, validate_command


class PilotTests(unittest.TestCase):
    def test_movement_expires_without_a_heartbeat(self):
        pilot = DiverPilot()
        pilot.command('manual', [.65,0,0], 10)
        self.assertEqual(pilot.snapshot(11)['velocity'], [.65,0,0])
        self.assertEqual(pilot.snapshot(11.3)['velocity'], [0,0,0])
        self.assertTrue(pilot.snapshot(11.3)['input_expired'])
        pilot.command('manual', [0,0,-.3], 12)
        self.assertEqual(pilot.snapshot(12.1)['velocity'], [0,0,-.3])

    def test_pause_clears_held_movement_before_resume(self):
        pilot = DiverPilot()
        pilot.command('manual', [0,.6,0], 3)
        self.assertEqual(pilot.snapshot(3.1, paused=True)['velocity'], [0,0,0])
        self.assertEqual(pilot.snapshot(3.2)['velocity'], [0,0,0])

    def test_scripted_mode_cannot_retain_manual_velocity(self):
        pilot = DiverPilot()
        pilot.command('manual', [0,.5,0], 3)
        pilot.command('scripted', [0,.5,0], 4)
        self.assertEqual(pilot.snapshot(4)['velocity'], [0,0,0])

    def test_invalid_motion_and_modes(self):
        for mode in ('unknown', None, [], 1):
            with self.assertRaises(ValueError):
                validate_command({'command':'set_mode', 'mode':mode})
        for velocity in ([.65,.65,0], [True,0,0], [float('nan'),0,0], [10**400,0,0], [0,0]):
            with self.assertRaises(ValueError):
                validate_command({'command':'set_diver_control', 'mode':'manual', 'velocity':velocity})


if __name__ == '__main__':
    unittest.main()
