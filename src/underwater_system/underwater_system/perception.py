"""Local dual-pipeline perception with a transparent synthetic baseline.

``SonarFeatureModel`` is a truth-side simulation sensor, not an edge classifier.
It generates noisy numerical echo-shape/micro-Doppler surrogates from a source
category. These are illustrative signatures, not measured sonar returns.
``EdgePerception`` sees only those numbers and a local hydrophone waveform. It
has no source category, object identity or position-truth input. Its default
template classifier is untrained and has no real-world accuracy claim.

A single hydrophone hears the whole scene: its engine evidence cannot identify
which echo belongs to that engine. Shared acoustic evidence only gives bounded
support to an already compatible sonar return. Target-associated audio may be
enabled explicitly when an array/beamformer supplies that association.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .acoustics import analyze_hydrophone, spectrogram
from .core import CLASS_NAMES


# Five morphology cues used by the illustrative open-set template classifier:
# elongation, specularity, echo roughness, micro-Doppler / 20, spread / size.
_TEMPLATES = {
    "diver": np.array([3.2, 0.35, 0.35, 0.45, 0.68]),
    "vehicle": np.array([4.8, 0.85, 0.15, 0.10, 0.30]),
    "marine_life": np.array([2.4, 0.15, 0.25, 0.85, 0.78]),
    "obstacle": np.array([1.35, 0.48, 0.82, 0.015, 0.95]),
}
_DISTANCE_SCALE = np.array([0.55, 0.115, 0.115, 0.14, 0.14])
SONAR_FEATURE_NAMES = (
    "range_m", "echo_strength", "size_m", "speed_m_s", "elongation",
    "specular_ratio", "echo_roughness", "micro_doppler_hz", "echo_spread_m",
    "measurement_quality",
)


def synthesize_sonar_features(kind: str, range_m: float, speed_m_s: float,
                              rng: np.random.Generator,
                              size_m: float | None = None) -> dict:
    """Generate a numerical synthetic echo; call only in the sensor emulator.

    Feature distributions differ by illustrative material/shape/articulation.
    Range degrades SNR and morphological measurements. No semantic source field
    is returned. Unknown signatures lie outside known template regions, with
    variable morphology; real unknowns can still resemble known objects.
    """
    distance, speed = float(range_m), float(speed_m_s)
    if not np.all(np.isfinite([distance, speed])) or distance < 0 or speed < 0:
        raise ValueError("Sonar range and speed must be finite and nonnegative")
    kind = {"environmental_hazard": "obstacle", "person": "diver"}.get(kind, kind)
    default_sizes = {"diver": 1.3, "vehicle": 2.2, "marine_life": 1.0,
                     "obstacle": 2.8, "unknown": 1.6}
    size = float(default_sizes.get(kind, 1.6) if size_m is None else size_m)
    if not math.isfinite(size) or size <= 0:
        raise ValueError("Sonar object size must be finite and positive")
    quality = float(np.clip(0.98 - distance * 0.006, 0.30, 0.98))
    if kind in _TEMPLATES:
        center = _TEMPLATES[kind]
    else:
        # A deliberately unfamiliar combination, not an edge-side unknown tag.
        center = np.array([1.8, 0.76, 0.48, 0.70, 0.12])
    jitter = _DISTANCE_SCALE * (0.20 + (1 - quality) * 1.2)
    cues = center + rng.normal(0, jitter)
    measured_size = max(0.1, size + rng.normal(0, 0.03 + distance * 0.004))
    strength = np.clip((0.35 + 0.60 * cues[1]) * math.exp(-distance / 45)
                       + rng.normal(0, 0.025), 0, 1)
    return {
        "range_m": max(0.0, distance + rng.normal(0, 0.03 + distance * 0.005)),
        "echo_strength": float(strength),
        "size_m": float(measured_size),
        "speed_m_s": max(0.0, speed + rng.normal(0, 0.015 + distance * 0.001)),
        "elongation": float(np.clip(cues[0], 1, 8)),
        "specular_ratio": float(np.clip(cues[1], 0, 1)),
        "echo_roughness": float(np.clip(cues[2], 0, 1)),
        "micro_doppler_hz": float(np.clip(cues[3] * 20, 0, 50)),
        "echo_spread_m": float(max(0.02, cues[4] * measured_size)),
        "measurement_quality": quality,
    }


class SonarFeatureModel:
    """Stateful reproducible truth-side echo-signature emulator."""

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)

    def measure(self, kind: str, range_m: float, speed_m_s: float,
                size_m: float | None = None) -> dict:
        return synthesize_sonar_features(kind, range_m, speed_m_s, self.rng, size_m)


class EdgePerception:
    """Fuse raw local acoustics with numerical sonar returns on one edge node.

    Config keys: ``sample_rate_hz`` (8000), ``unknown_threshold`` (0.62),
    ``acoustic_model_path`` and ``sonar_model_path`` (optional local TorchScript
    artifacts), ``class_labels`` (CLASS_NAMES), ``acoustic_target_associated``
    (False), and ``sonar_distance_threshold`` (1.8).

    Acoustic models consume float32 [N,1,257,T] normalized STFTs. Sonar models
    consume float32 [N,10] in SONAR_FEATURE_NAMES order, normalized by
    [30,1,3,2,8,1,1,50,3,1]. Models return class logits in class_labels order.
    No pretrained artifacts are shipped or downloaded automatically.
    """

    def __init__(self, config: Mapping | None = None):
        config = dict(config or {})
        self.sample_rate = int(config.get("sample_rate_hz", 8000))
        self.unknown_threshold = float(config.get("unknown_threshold", 0.62))
        self.distance_threshold = float(config.get("sonar_distance_threshold", 1.8))
        self.class_labels = tuple(config.get("class_labels", CLASS_NAMES))
        self.acoustic_target_associated = bool(config.get("acoustic_target_associated", False))
        if (self.sample_rate <= 0 or not 0 < self.unknown_threshold <= 1
                or not math.isfinite(self.distance_threshold) or self.distance_threshold <= 0):
            raise ValueError("Invalid edge perception sampling/rejection parameters")
        if "unknown" not in self.class_labels or len(set(self.class_labels)) != len(self.class_labels):
            raise ValueError("Classifier labels must be unique and include unknown")
        self.torch = None
        self.acoustic_model = None
        self.sonar_model = None
        paths = {"acoustic_model": config.get("acoustic_model_path"),
                 "sonar_model": config.get("sonar_model_path")}
        for path in paths.values():
            if path and not Path(path).is_file():
                raise FileNotFoundError(f"TorchScript classifier model not found: {path}")
        if any(paths.values()):
            try:
                import torch
            except ImportError as error:
                raise RuntimeError("TorchScript artifacts require PyTorch; the baseline needs only NumPy") from error
            self.torch = torch
            for attribute, path in paths.items():
                if path:
                    setattr(self, attribute, torch.jit.load(str(path), map_location="cpu").eval())

    @staticmethod
    def sonar_tensor_features(features: Mapping) -> np.ndarray:
        """Documented numerical preprocessing for optional sonar models."""
        values = np.array([float(features.get(key, 0)) for key in SONAR_FEATURE_NAMES], dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("Sonar features must be finite")
        return np.clip(values / np.array([30, 1, 3, 2, 8, 1, 1, 50, 3, 1],
                                         dtype=np.float32), 0, 2)

    def _infer(self, model, values: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            logits = model(self.torch.from_numpy(values))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits = logits.reshape(-1)
            if logits.numel() != len(self.class_labels):
                raise ValueError("TorchScript output class count does not match class_labels")
            if not self.torch.all(self.torch.isfinite(logits)):
                raise ValueError("TorchScript classifier returned non-finite logits")
            return self.torch.softmax(logits, dim=0).cpu().numpy()

    def analyze(self, samples) -> dict:
        """Analyze one shared hydrophone frame once before handling its echoes."""
        context = {"acoustic_features": analyze_hydrophone(samples, self.sample_rate),
                   "acoustic_probabilities": None}
        if self.acoustic_model is not None:
            values = spectrogram(samples, self.sample_rate)[None, None, :, :]
            context["acoustic_probabilities"] = self._infer(self.acoustic_model, values)
        return context

    def _baseline(self, features: Mapping) -> tuple[np.ndarray, dict]:
        unknown_index = self.class_labels.index("unknown")
        probabilities = np.zeros(len(self.class_labels), dtype=float)
        if not all(key in features for key in SONAR_FEATURE_NAMES):
            probabilities[unknown_index] = 1.0
            return probabilities, {"rejection_reason": "incomplete_sonar_features",
                                   "sonar_distance": None, "sonar_candidate": "unknown"}
        values = {key: float(features[key]) for key in SONAR_FEATURE_NAMES}
        if (values["size_m"] <= 0 or values["range_m"] < 0 or values["speed_m_s"] < 0
                or values["echo_spread_m"] < 0 or values["micro_doppler_hz"] < 0
                or values["elongation"] < 1
                or any(not 0 <= values[key] <= 1 for key in
                       ("echo_strength", "specular_ratio", "echo_roughness", "measurement_quality"))):
            probabilities[unknown_index] = 1.0
            return probabilities, {"rejection_reason": "invalid_echo_geometry",
                                   "sonar_distance": None, "sonar_candidate": "unknown"}
        morphology = np.array([values["elongation"], values["specular_ratio"],
                               values["echo_roughness"], values["micro_doppler_hz"] / 20,
                               values["echo_spread_m"] / values["size_m"]])
        distances = sorted((float(np.sqrt(np.mean(((morphology - template) / _DISTANCE_SCALE) ** 2))), kind)
                           for kind, template in _TEMPLATES.items() if kind in self.class_labels)
        if not distances:
            probabilities[unknown_index] = 1.0
            return probabilities, {"rejection_reason": "no_known_templates",
                                   "sonar_distance": None, "sonar_candidate": "unknown"}
        distance, candidate = distances[0]
        quality = values["measurement_quality"] * min(1.0, values["echo_strength"] / 0.12)
        score = float(quality * math.exp(-0.5 * (distance / 1.30) ** 2))
        reason = None
        if distance > self.distance_threshold:
            reason = "out_of_distribution_echo"
        elif len(distances) > 1 and distances[1][0] - distance < 0.35:
            reason = "ambiguous_echo_templates"
        elif score < self.unknown_threshold:
            reason = "low_sonar_confidence"
        if reason:
            probabilities[unknown_index] = max(0.55, 1.0 - score)
            probabilities[self.class_labels.index(candidate)] = 1 - probabilities[unknown_index]
        else:
            probabilities[self.class_labels.index(candidate)] = score
            probabilities[unknown_index] = 1 - score
        return probabilities, {"rejection_reason": reason, "sonar_distance": round(distance, 4),
                               "sonar_candidate": candidate}

    def classify(self, samples, sonar_features: Mapping, frame_context: Mapping | None = None) -> dict:
        """Classify one echo without source identity or semantic feature hints."""
        sonar_vector = self.sonar_tensor_features(sonar_features)
        context = self.analyze(samples) if frame_context is None else frame_context
        acoustic_features = dict(context["acoustic_features"])
        acoustic_probs = context.get("acoustic_probabilities")
        probabilities, details = self._baseline(sonar_features)
        backends = ["synthetic_echo_template_baseline"]
        if self.sonar_model is not None:
            probabilities = self._infer(self.sonar_model, sonar_vector[None, :])
            details = {"rejection_reason": None, "sonar_distance": None,
                       "sonar_candidate": self.class_labels[int(np.argmax(probabilities))]}
            backends = ["sonar_torchscript"]
        if acoustic_probs is not None:
            backends.append("acoustic_torchscript")
            acoustic_probs = np.asarray(acoustic_probs, dtype=float)
            if self.acoustic_target_associated:
                probabilities = 0.65 * probabilities + 0.35 * acoustic_probs
                sonar_winner = details["sonar_candidate"]
                acoustic_winner = self.class_labels[int(np.argmax(acoustic_probs))]
                if (sonar_winner != acoustic_winner and max(acoustic_probs) >= 0.8
                        and max(probabilities) >= 0.6):
                    # This rejection requires explicit target association. A
                    # shared-scene animal tone beside a vehicle is no conflict.
                    probabilities *= 0.4
                    probabilities[self.class_labels.index("unknown")] += 0.6
                    details["rejection_reason"] = "associated_pipeline_disagreement"
            else:
                # This waveform is shared by all echoes, so only a small bonus
                # supports the sonar candidate; audio cannot change identity.
                candidate = int(np.argmax(probabilities))
                if (self.class_labels[candidate] != "unknown"
                        and probabilities[candidate] >= self.unknown_threshold
                        and details["rejection_reason"] is None):
                    support = float(acoustic_probs[candidate]) * 0.03
                    probabilities *= 1 - support
                    probabilities[candidate] += support
        elif ("vehicle" in self.class_labels and details["rejection_reason"] is None
              and details["sonar_candidate"] == "vehicle"):
            support = max(0.0, float(acoustic_features["engine_score"]) - 0.5) * 0.06
            candidate = self.class_labels.index("vehicle")
            probabilities *= 1 - support
            probabilities[candidate] += support
        index = int(np.argmax(probabilities))
        candidate = self.class_labels[index]
        score = float(probabilities[index])
        rejected = candidate == "unknown" or score < self.unknown_threshold
        label = "unknown" if rejected else candidate
        if rejected and details["rejection_reason"] is None:
            details["rejection_reason"] = "explicit_unknown" if candidate == "unknown" else "low_fused_confidence"
        # Confidence belongs to the returned class, including unknown. Preserve
        # the maximum candidate score separately when threshold rejection maps
        # a weak known prediction to unknown.
        label_confidence = float(probabilities[self.class_labels.index(label)])
        return {"classification": label, "confidence": round(label_confidence, 4),
                "candidate_confidence": round(score, 4),
                "unknown_rejected": rejected, "backend": "+".join(backends),
                "acoustic_features": acoustic_features,
                "class_probabilities": {key: round(float(value), 4)
                                        for key, value in zip(self.class_labels, probabilities)},
                "acoustic_scope": "target_associated" if self.acoustic_target_associated else "shared_scene",
                "position_from_passive_audio": False, **details}

    def classify_frame(self, samples, sonar_returns) -> list[dict]:
        """Batch a node's returns while computing acoustic STFT features once."""
        context = self.analyze(samples)
        return [self.classify(samples, features, context) for features in sonar_returns]
