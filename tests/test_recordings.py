"""Local WAV playback preserves samples and obeys clip and frame limits."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "underwater_system"))
from underwater_system.recordings import HydrophoneRecording


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "clip.wav"

    def tearDown(self):
        self.directory.cleanup()

    def write_clip(self, samples, *, rate=8000, width=2):
        samples = np.asarray(samples)
        channels = samples.shape[1] if samples.ndim == 2 else 1
        with wave.open(str(self.path), "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(width)
            writer.setframerate(rate)
            writer.writeframes(samples.astype("<i2" if width == 2 else "u1").tobytes())
        return self.path

    def test_pcm_scale_and_metadata_without_calibration_or_labels(self):
        values = np.tile([-32768, 32767], 16)
        recording = HydrophoneRecording(self.write_clip(values), frame_samples=32)
        np.testing.assert_allclose(recording.frame(0), values / 32768)
        self.assertEqual(recording.frame(0)[0], -1)
        self.assertLess(recording.frame(0)[1], 1)
        metadata = recording.metadata
        self.assertEqual(metadata["source"], "recorded")
        self.assertEqual(metadata["amplitude_units"], "normalized_pcm")
        self.assertEqual(metadata["sample_rate_hz"], 8000)
        self.assertEqual(metadata["name"], "clip.wav")
        self.assertFalse(metadata["loop"])
        self.assertNotIn("classification", metadata)
        self.assertNotIn("spl_db", metadata)
        json.dumps(metadata, allow_nan=False)
        metadata["source"] = "changed"
        self.assertEqual(recording.metadata["source"], "recorded")

    def test_stereo_channels_are_averaged(self):
        values = np.array([[-16384, 16384], [32767, 0]] * 16)
        recording = HydrophoneRecording(self.write_clip(values), frame_samples=32)
        expected = values.mean(axis=1) / 32768
        np.testing.assert_allclose(recording.frame(0), expected)
        self.assertEqual(recording.metadata["channels"], 2)

    def test_time_selects_sequential_frames_and_resets_deterministically(self):
        values = np.arange(96)
        recording = HydrophoneRecording(self.write_clip(values), frame_samples=32)
        first = recording.frame(0)
        second = recording.frame(32 / 8000)
        np.testing.assert_allclose(first, values[:32] / 32768)
        np.testing.assert_allclose(second, values[32:64] / 32768)
        np.testing.assert_allclose(recording.frame(0), first)

    def test_nonloop_playback_pads_partial_and_finished_frames(self):
        values = np.arange(1, 65)
        recording = HydrophoneRecording(self.write_clip(values), frame_samples=32)
        partial = recording.frame(48 / 8000)
        np.testing.assert_allclose(partial[:16], values[48:] / 32768)
        np.testing.assert_allclose(partial[16:], 0)
        self.assertEqual(recording.metadata["padded_samples"], 16)
        self.assertFalse(recording.metadata["looped"])
        np.testing.assert_allclose(recording.frame(64 / 8000), 0)
        self.assertTrue(recording.metadata["exhausted"])
        np.testing.assert_allclose(recording.frame(1e300), 0)

    def test_explicit_loop_wraps_within_a_frame(self):
        values = np.arange(64)
        recording = HydrophoneRecording(self.write_clip(values), loop=True, frame_samples=32)
        expected = np.r_[values[48:], values[:16]] / 32768
        np.testing.assert_allclose(recording.frame(48 / 8000), expected)
        self.assertTrue(recording.metadata["looped"])
        self.assertFalse(recording.metadata["exhausted"])
        np.testing.assert_allclose(recording.frame(64 / 8000), values[:32] / 32768)
        self.assertEqual(recording.metadata["padded_samples"], 0)

    def test_sample_rate_mismatch_requires_explicit_conversion(self):
        self.write_clip(np.zeros(32), rate=16000)
        with self.assertRaisesRegex(ValueError, "ffmpeg.*-ar 8000"):
            HydrophoneRecording(self.path)

    def test_unsupported_width_and_empty_clip_are_rejected(self):
        self.write_clip(np.zeros(32), width=1)
        with self.assertRaisesRegex(ValueError, "PCM16"):
            HydrophoneRecording(self.path)
        self.write_clip(np.array([]))
        with self.assertRaisesRegex(ValueError, "no samples"):
            HydrophoneRecording(self.path)

    def test_long_clip_rejection_gives_trim_instruction(self):
        self.write_clip(np.zeros(960))
        with self.assertRaisesRegex(ValueError, "-t 0.1"):
            HydrophoneRecording(self.path, max_seconds=0.1)

    def test_truncated_clip_is_rejected(self):
        self.write_clip(np.arange(64))
        with self.path.open("r+b") as output:
            output.truncate(self.path.stat().st_size - 8)
        with self.assertRaisesRegex(ValueError, "truncated"):
            HydrophoneRecording(self.path)

    def test_file_size_limit_applies_before_loading(self):
        self.write_clip(np.zeros(32))
        with self.path.open("r+b") as output:
            output.truncate(65 * 1024 * 1024)
        with self.assertRaisesRegex(ValueError, "64 MiB"):
            HydrophoneRecording(self.path)

    def test_invalid_times_and_oversized_frames_are_rejected(self):
        recording = HydrophoneRecording(self.write_clip(np.zeros(2048)))
        self.assertEqual(len(recording.frame(0)), 1024)
        self.assertEqual(len(recording.frame(0, duration=0.064)), 512)
        for now in (-1, float("nan"), float("inf")):
            with self.subTest(now=now), self.assertRaises(ValueError):
                recording.frame(now)
        for duration in (0, -1, 0.001, 0.2, float("inf"), 1e300):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                recording.frame(0, duration=duration)
        with self.assertRaises(ValueError):
            HydrophoneRecording(self.path, frame_samples=100000)
        with self.assertRaises(ValueError):
            HydrophoneRecording(self.path, loop="false")


if __name__ == "__main__":
    unittest.main()
