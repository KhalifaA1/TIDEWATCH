from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "underwater_system"))
from underwater_system.acoustics import (DualPipelineClassifier, analyze_hydrophone,
                                         spectrogram, synthesize_hydrophone)
from underwater_system.core import WorldObject


class AcousticsTests(unittest.TestCase):
    def test_reproducible_frames_and_spatial_attenuation(self):
        engine = WorldObject("engine", "vehicle", [10, 0, -7])
        first = synthesize_hydrophone([engine], [0, 0, -7], 2, 7)
        self.assertEqual(first, synthesize_hydrophone([engine], [0, 0, -7], 2, 7))
        far = synthesize_hydrophone([engine], [-35, 0, -7], 2, 7)
        self.assertGreater(analyze_hydrophone(first)["rms"], analyze_hydrophone(far)["rms"] * 2)
        self.assertTrue(np.all(np.abs(first) <= 1))

    def test_engine_harmonics_distinguish_noise_and_single_tone(self):
        engine = WorldObject("engine", "vehicle", [12, 0, -7])
        animal = WorldObject("animal", "marine_life", [5, 0, -7])
        engine_features = analyze_hydrophone(synthesize_hydrophone([engine], [0, 0, -7], 0, 7))
        animal_features = analyze_hydrophone(synthesize_hydrophone([animal], [0, 0, -7], 0, 7))
        noise_features = analyze_hydrophone(synthesize_hydrophone([], [0, 0, -7], 0, 7))
        self.assertGreater(engine_features["engine_score"], 0.75)
        self.assertLess(animal_features["engine_score"], 0.30)
        self.assertLess(noise_features["engine_score"], 0.30)
        self.assertLess(engine_features["broadband_energy"], noise_features["broadband_energy"])

    def test_local_spectrogram_shape_and_finite_zero_audio(self):
        values = spectrogram(np.zeros(1024))
        self.assertEqual(values.shape, (257, 5))
        self.assertTrue(np.all(np.isfinite(values)))
        features = analyze_hydrophone(np.zeros(1024))
        self.assertEqual(features["engine_score"], 0)
        self.assertEqual(features["rms"], 0)

    def test_unknown_rejection_and_no_passive_position(self):
        classifier = DualPipelineClassifier()
        unknown = classifier.classify(np.zeros(1024), {"range_m": 10, "echo_strength": 0.2})
        self.assertEqual(unknown["classification"], "unknown")
        self.assertTrue(unknown["unknown_rejected"])
        self.assertFalse(unknown["position_from_passive_audio"])
        self.assertEqual(unknown["backend"], "untrained_feature_baseline")
        engine = WorldObject("engine", "vehicle", [12, 0, -7])
        vehicle = classifier.classify(synthesize_hydrophone([engine], [0, 0, -7], 1, 7),
                                     {"range_m": 12, "echo_strength": 0.7})
        self.assertEqual(vehicle["classification"], "vehicle")
        self.assertGreater(vehicle["confidence"], 0.62)

    def test_explicit_models_require_existing_artifacts(self):
        with self.assertRaises(FileNotFoundError):
            DualPipelineClassifier(acoustic_model_path="/no/trained_model.pt")
        with self.assertRaises(ValueError):
            DualPipelineClassifier(class_labels=("vehicle", "diver"))

    def test_loaded_pipeline_probabilities_are_used_and_disagreement_rejects(self):
        classifier = DualPipelineClassifier()
        acoustic, sonar = object(), object()
        classifier.acoustic_model = acoustic
        # A lightweight inference fixture tests fusion independently of PyTorch.
        classifier._infer = lambda model, values: (np.array([0.01, 0.96, 0.01, 0.01, 0.01])
                                                   if model is acoustic else
                                                   np.array([0.01, 0.01, 0.01, 0.96, 0.01]))
        single = classifier.classify(np.zeros(1024), {})
        self.assertEqual(single["classification"], "diver")
        self.assertEqual(single["backend"], "acoustic_torchscript")
        classifier.sonar_model = sonar
        disagreement = classifier.classify(np.zeros(1024), {})
        self.assertEqual(disagreement["classification"], "unknown")
        self.assertTrue(disagreement["unknown_rejected"])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            analyze_hydrophone([float("nan")] * 1024)
        with self.assertRaises(ValueError):
            synthesize_hydrophone([], [0, 0, 0], 0, 7, n=10)
        with self.assertRaises(ValueError):
            DualPipelineClassifier().classify(np.zeros(1024), {"range_m": float("nan")})


if __name__ == "__main__":
    unittest.main()
