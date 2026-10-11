from pathlib import Path
import sys
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "underwater_system"))
from underwater_system.acoustics import synthesize_hydrophone
from underwater_system.core import CLASS_NAMES, WorldObject
from underwater_system.perception import (EdgePerception, SONAR_FEATURE_NAMES,
                                          SonarFeatureModel)


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.edge = EdgePerception()
        self.sensor = SonarFeatureModel(7)
        self.silent = np.zeros(1024)

    def test_sensor_is_reproducible_and_contains_only_numeric_measurements(self):
        first = self.sensor.measure("vehicle", 12, 0.6)
        self.assertEqual(first, SonarFeatureModel(7).measure("vehicle", 12, 0.6))
        self.assertEqual(set(first), set(SONAR_FEATURE_NAMES))
        self.assertTrue(all(isinstance(value, (int, float)) for value in first.values()))
        self.assertFalse(set(first) & {"class", "kind", "label", "sonar_label", "baseline_kind", "truth_id"})

    def test_known_signatures_survive_range_and_measurement_noise(self):
        # Numerical measurements, including changed object size, are consumed
        # independently of source identity and the truth-side generator call.
        context = self.edge.analyze(self.silent)
        for kind in CLASS_NAMES:
            if kind == "unknown":
                continue
            for distance in (3, 15, 28):
                with self.subTest(kind=kind, distance=distance):
                    predictions = [self.edge.classify(self.silent,
                        self.sensor.measure(kind, distance, 0.5, size_m=1.4), context)
                        for _ in range(50)]
                    self.assertGreaterEqual(sum(p["classification"] == kind for p in predictions), 48)
                    self.assertTrue(all(p["backend"] == "synthetic_echo_template_baseline" for p in predictions))

    def test_unfamiliar_numeric_echo_is_rejected(self):
        prediction = self.edge.classify(self.silent, self.sensor.measure("unknown", 12, 0.2))
        self.assertEqual(prediction["classification"], "unknown")
        self.assertTrue(prediction["unknown_rejected"])
        self.assertEqual(prediction["rejection_reason"], "out_of_distribution_echo")
        self.assertFalse(prediction["position_from_passive_audio"])
        self.assertGreater(prediction["sonar_distance"], self.edge.distance_threshold)

    def test_out_of_distribution_numeric_values_and_missing_cues_are_rejected(self):
        features = self.sensor.measure("vehicle", 10, 0.5)
        features.update(elongation=7.8, micro_doppler_hz=48, echo_spread_m=14)
        self.assertEqual(self.edge.classify(self.silent, features)["classification"], "unknown")
        missing = self.edge.classify(self.silent, {"range_m": 10, "echo_strength": 0.8})
        self.assertEqual(missing["classification"], "unknown")
        self.assertEqual(missing["rejection_reason"], "incomplete_sonar_features")

    def test_semantic_hints_never_change_classification(self):
        features = self.sensor.measure("marine_life", 12, 0.3)
        first = self.edge.classify(self.silent, features)
        features.update(kind="vehicle", classification="vehicle", sonar_label="vehicle",
                        baseline_kind="vehicle", truth_id="engine")
        self.assertEqual(self.edge.classify(self.silent, features), first)

    def test_shared_engine_waveform_cannot_turn_unknowns_into_vehicles(self):
        engine = WorldObject("unseen_engine", "vehicle", [8, 0, -7])
        samples = synthesize_hydrophone([engine], [0, 0, -7], 1, 7)
        echoes = [self.sensor.measure(kind, 12, 0.4)
                  for kind in ("unknown", "diver", "marine_life", "obstacle", "vehicle")]
        predictions = self.edge.classify_frame(samples, echoes)
        self.assertEqual([p["classification"] for p in predictions],
                         ["unknown", "diver", "marine_life", "obstacle", "vehicle"])
        self.assertGreater(predictions[0]["acoustic_features"]["engine_score"], 0.7)
        self.assertTrue(all(p["acoustic_scope"] == "shared_scene" for p in predictions))

    def test_weak_measurement_quality_rejects_known_shape(self):
        features = self.sensor.measure("diver", 15, 0.2)
        features["measurement_quality"] = 0.2
        result = self.edge.classify(self.silent, features)
        self.assertEqual(result["classification"], "unknown")
        self.assertEqual(result["rejection_reason"], "low_sonar_confidence")

    def test_batch_analyzes_one_hydrophone_frame_only_once(self):
        echoes = [self.sensor.measure("diver", 8, 0.2) for _ in range(5)]
        with patch.object(self.edge, "analyze", wraps=self.edge.analyze) as analyze:
            outputs = self.edge.classify_frame(self.silent, echoes)
            self.assertEqual(len(outputs), 5)
            analyze.assert_called_once()

    def test_sonar_model_preprocessing_has_documented_numerical_order(self):
        features = dict(zip(SONAR_FEATURE_NAMES, (15, 0.8, 1.5, 1, 4, 0.6, 0.3, 25, 1.5, 0.7)))
        values = self.edge.sonar_tensor_features(features)
        np.testing.assert_allclose(values, (0.5, 0.8, 0.5, 0.5, 0.5, 0.6, 0.3, 0.5, 0.5, 0.7))
        self.assertEqual(values.dtype, np.float32)

    def test_optional_model_adapter_uses_inputs_and_logits_class_order(self):
        acoustic, sonar = object(), object()
        self.edge.acoustic_model, self.edge.sonar_model = acoustic, sonar
        seen = []

        def infer(model, values):
            seen.append((model, values.shape))
            probs = np.full(len(CLASS_NAMES), 0.01)
            probs[CLASS_NAMES.index("diver")] = 0.96
            return probs

        self.edge._infer = infer
        result = self.edge.classify(self.silent, self.sensor.measure("vehicle", 12, 0.3))
        self.assertEqual(result["classification"], "diver")
        self.assertEqual(result["backend"], "sonar_torchscript+acoustic_torchscript")
        self.assertEqual(seen, [(acoustic, (1, 1, 257, 5)), (sonar, (1, 10))])

    def test_explicit_associated_pipeline_disagreement_rejects(self):
        edge = EdgePerception({"acoustic_target_associated": True})
        acoustic, sonar = object(), object()
        edge.acoustic_model, edge.sonar_model = acoustic, sonar

        def infer(model, values):
            probs = np.full(len(CLASS_NAMES), 0.01)
            probs[CLASS_NAMES.index("obstacle" if model is acoustic else "vehicle")] = 0.96
            return probs

        edge._infer = infer
        result = edge.classify(self.silent, self.sensor.measure("vehicle", 12, 0.3))
        self.assertEqual(result["classification"], "unknown")
        self.assertEqual(result["rejection_reason"], "associated_pipeline_disagreement")

    def test_shared_scene_pipeline_disagreement_does_not_assign_unrelated_audio(self):
        acoustic, sonar = object(), object()
        self.edge.acoustic_model, self.edge.sonar_model = acoustic, sonar

        def infer(model, values):
            probs = np.full(len(CLASS_NAMES), 0.01)
            probs[CLASS_NAMES.index("marine_life" if model is acoustic else "vehicle")] = 0.96
            return probs

        self.edge._infer = infer
        result = self.edge.classify(self.silent, self.sensor.measure("vehicle", 12, 0.3))
        self.assertEqual(result["classification"], "vehicle")
        self.assertFalse(result["unknown_rejected"])

    def test_models_reject_low_confidence_and_explicit_unknown(self):
        self.edge.sonar_model = object()
        self.edge._infer = lambda model, values: np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES)
        self.assertEqual(self.edge.classify(self.silent, {})["classification"], "unknown")
        self.edge._infer = lambda model, values: np.array([0.96, 0.01, 0.01, 0.01, 0.01])
        self.assertEqual(self.edge.classify(self.silent, {})["rejection_reason"], "explicit_unknown")

    def test_confidence_describes_returned_unknown_label(self):
        self.edge.sonar_model = object()
        self.edge._infer = lambda model, values: np.array([0.1, 0.61, 0.1, 0.1, 0.09])
        result = self.edge.classify(self.silent, self.sensor.measure("diver", 12, 0.2))
        self.assertEqual(result["classification"], "unknown")
        self.assertEqual(result["confidence"], result["class_probabilities"]["unknown"])
        self.assertEqual(result["candidate_confidence"], 0.61)

    def test_shared_audio_cannot_promote_below_threshold_sonar_prediction(self):
        acoustic, sonar = object(), object()
        self.edge.acoustic_model, self.edge.sonar_model = acoustic, sonar
        self.edge._infer = lambda model, values: (
            np.array([0.01, 0.01, 0.01, 0.96, 0.01]) if model is acoustic else
            np.array([0.10, 0.10, 0.10, 0.61, 0.09]))
        result = self.edge.classify(self.silent, self.sensor.measure("vehicle", 12, 0.3))
        self.assertEqual(result["classification"], "unknown")
        self.assertEqual(result["candidate_confidence"], 0.61)

    def test_invalid_measurements_and_model_configuration(self):
        for kwargs in ({"range_m": -1, "speed_m_s": 0}, {"range_m": 2, "speed_m_s": float("nan")},
                       {"range_m": 2, "speed_m_s": 0, "size_m": 0}):
            with self.assertRaises(ValueError):
                self.sensor.measure("diver", **kwargs)
        features = self.sensor.measure("vehicle", 10, 0.5)
        features["echo_strength"] = float("nan")
        with self.assertRaises(ValueError):
            self.edge.classify(self.silent, features)
        with self.assertRaises(ValueError):
            self.edge.classify([float("nan")] * 1024, {})
        for config in ({"class_labels": ["diver"]}, {"unknown_threshold": 0},
                       {"sonar_distance_threshold": float("nan")}):
            with self.assertRaises(ValueError):
                EdgePerception(config)
        with self.assertRaises(FileNotFoundError):
            EdgePerception({"sonar_model_path": "/no/model.pt"})

    @unittest.skipUnless(importlib.util.find_spec("torch"),
                         "PyTorch optional; no classifier artifacts bundled")
    def test_real_torchscript_artifact_loading(self):
        import torch

        class FixedLogits(torch.nn.Module):
            def forward(self, values):
                return values[:, :1] * 0 + torch.tensor([[0., 0., 0., 8., 0.]])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sonar.pt"
            torch.jit.trace(FixedLogits(), torch.zeros(1, 10)).save(str(path))
            edge = EdgePerception({"sonar_model_path": str(path)})
            result = edge.classify(self.silent, self.sensor.measure("vehicle", 12, 0.3))
            self.assertEqual(result["classification"], "vehicle")
            self.assertEqual(result["backend"], "sonar_torchscript")


if __name__ == "__main__":
    unittest.main()
