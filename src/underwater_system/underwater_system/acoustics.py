"""Local passive acoustic STFT and an optional dual TorchScript classifier.

``synthesize_hydrophone`` is a simulation sensor using truth objects. Everything
else consumes measured waveforms/scalar sonar features only. Raw waveforms and
spectrograms stay local; ``analyze_hydrophone`` returns compact feature values
that may be telemetered. The default classifier is an untrained feature baseline,
not an acoustic AI model validated on real hydrophone recordings.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np

from .core import CLASS_NAMES, WorldObject, vec3


def synthesize_hydrophone(objects: Iterable[WorldObject], observer_position,
                          timestamp: float, seed: int, sample_rate: int = 8000,
                          n: int = 1024) -> list[float]:
    """Generate a reproducible local hydrophone frame from nearby truth sources.

    Vehicle sources contain motor harmonics/cavitation, animals a swept tone,
    divers weak breath-modulated noise, and unknown sources irregular broadband
    energy. Spherical attenuation and propagation delay are approximations;
    this generator does not model measured transducers/ocean multipath.
    """
    if sample_rate < 2000 or n < 128:
        raise ValueError("Hydrophone synthesis requires sample_rate >= 2000 and n >= 128")
    if not math.isfinite(timestamp):
        raise ValueError("Hydrophone timestamp must be finite")
    frame_index = int(round(timestamp * sample_rate / n))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) % 2**32,
                                                       frame_index % 2**32]))
    origin = vec3(observer_position)
    times = timestamp + np.arange(n, dtype=float) / sample_rate
    waveform = rng.normal(0, 0.018, n)
    for obj in objects:
        position = vec3(obj.position)
        distance = float(np.linalg.norm(position - origin))
        if distance > 60:
            continue
        local_times = times - distance / 1500.0
        attenuation = 1.0 / max(distance, 2.0)
        kind = obj.kind
        if kind == "vehicle":
            # Integrate slow RPM modulation to preserve continuous source phase.
            phase = 2 * math.pi * (120 * local_times - 5 / 0.07 * np.cos(0.07 * local_times))
            source = sum(amplitude * np.sin(harmonic * phase + harmonic * 0.2)
                         for harmonic, amplitude in enumerate((1.0, 0.55, 0.30, 0.18), start=1))
            source += rng.normal(0, 0.13, n)
            waveform += 3.0 * attenuation * source
        elif kind == "marine_life":
            phase = 2 * math.pi * (520 * local_times - 18 / 0.6 * np.cos(0.6 * local_times))
            waveform += 0.6 * attenuation * np.sin(phase)
        elif kind == "diver":
            envelope = 0.4 + 0.6 * max(0.0, math.sin(2 * math.pi * 0.3 * timestamp))
            waveform += 0.35 * attenuation * envelope * rng.normal(0, 1, n)
        elif kind == "unknown":
            source = rng.normal(0, 0.75, n)
            source += 0.25 * np.sin(2 * math.pi * (193 * local_times + 15 * local_times ** 2))
            waveform += 0.7 * attenuation * source
    return np.clip(waveform, -1.0, 1.0).astype(float).tolist()


def _stft(samples, sample_rate: int = 8000, frame_size: int = 512,
          hop_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    waveform = np.asarray(samples, dtype=float).reshape(-1)
    if waveform.size < 32 or not np.all(np.isfinite(waveform)):
        raise ValueError("Hydrophone frame needs at least 32 finite samples")
    if sample_rate <= 0 or frame_size < 32 or hop_size <= 0:
        raise ValueError("Invalid STFT sampling/window parameters")
    if waveform.size < frame_size:
        waveform = np.pad(waveform, (0, frame_size - waveform.size))
    frames = np.lib.stride_tricks.sliding_window_view(waveform, frame_size)[::hop_size]
    window = np.hanning(frame_size)
    spectra = np.fft.rfft((frames - frames.mean(axis=1, keepdims=True)) * window, axis=1)
    power = (np.abs(spectra) ** 2 / max(float(np.sum(window ** 2)), 1e-9)).T
    frequencies = np.fft.rfftfreq(frame_size, 1 / sample_rate)
    return frequencies, power


def spectrogram(samples, sample_rate: int = 8000, frame_size: int = 512,
                hop_size: int = 128) -> np.ndarray:
    """Return a local frequency-by-time normalized log-power STFT array.

    The fixed preprocessing is log10 power relative to the frame's maximum,
    clamped to [-80, 0] dB and scaled into [-1, 1]. Model training must use this
    same preprocessing. No network payload includes this array.
    """
    _, power = _stft(samples, sample_rate, frame_size, hop_size)
    decibels = 10 * np.log10(np.maximum(power, 1e-12) / max(float(power.max()), 1e-12))
    return (np.clip(decibels, -80, 0) / 40 + 1).astype(np.float32)


def analyze_hydrophone(samples, sample_rate: int = 8000) -> dict:
    """Extract local engine harmonic evidence and broadband/rms measurements.

    A high engine_score means multiple harmonic bands support an engine-like
    signature. It does not identify a specific object or determine hostility.
    Pure noise and an isolated animal tone should have little harmonic evidence.
    """
    frequencies, power_frames = _stft(samples, sample_rate)
    power = np.mean(power_frames, axis=1)
    usable = frequencies >= 40
    total = float(np.sum(power[usable]))
    peak_index = int(np.argmax(np.where(usable, power, -1)))
    best_score, best_ratio = 0.0, 0.0
    resolution = sample_rate / 512
    # Scan plausible engine fundamentals rather than assuming a true class.
    for fundamental in frequencies[(frequencies >= 75) & (frequencies <= 220)]:
        energies = []
        included = set()
        for harmonic in range(1, 5):
            center = fundamental * harmonic
            half_width = max(resolution * 1.05, center * 0.035)
            indices = np.flatnonzero(abs(frequencies - center) <= half_width)
            included.update(indices.tolist())
            energies.append(float(np.sum(power[indices])))
        tonal_ratio = sum(power[index] for index in included) / max(total, 1e-12)
        primary = max(energies[0], 1e-12)
        # Genuine harmonics need energy at both the second and third bands.
        presence = min(1.0, energies[1] / primary / 0.10) * min(1.0, energies[2] / primary / 0.04)
        fundamental_support = min(1.0, primary / max(total, 1e-12) / 0.30)
        score = float(np.clip((tonal_ratio - 0.10) / 0.72, 0, 1) * presence * fundamental_support)
        if score > best_score:
            best_score, best_ratio = score, float(tonal_ratio)
    tonal_bins = np.partition(power[usable], -min(8, int(np.sum(usable))))[-8:]
    tonal_energy_ratio = float(np.sum(tonal_bins) / max(total, 1e-12))
    broadband_mask = frequencies >= 900
    broadband_energy = float(np.sum(power[broadband_mask]) / max(total, 1e-12))
    return {"engine_score": round(best_score, 4),
            "peak_frequency_hz": round(float(frequencies[peak_index]), 2),
            "broadband_energy": round(broadband_energy, 4),
            "tonal_energy_ratio": round(tonal_energy_ratio, 4),
            "harmonic_ratio": round(best_ratio, 4),
            "rms": round(float(np.sqrt(np.mean(np.asarray(samples, dtype=float) ** 2))), 5),
            "sample_rate_hz": int(sample_rate), "pipeline": "numpy_stft_feature_baseline"}


class DualPipelineClassifier:
    """Optional local TorchScript acoustic/sonar inference with open-set rejection.

    Models are optional and are never downloaded automatically. Acoustic model
    input is [batch, 1, frequency, time] from ``spectrogram``; sonar model input
    is [batch, 5] with normalized range, echo strength, size, speed and elevation.
    Both models must output logits in ``class_labels`` order. The baseline uses
    engine harmonics and an existing sonar_label estimate/geometric features;
    it explicitly reports ``backend='untrained_feature_baseline'``.
    """
    def __init__(self, acoustic_model_path: Optional[str] = None,
                 sonar_model_path: Optional[str] = None, unknown_threshold: float = 0.62,
                 class_labels=CLASS_NAMES):
        self.class_labels = tuple(class_labels)
        if "unknown" not in self.class_labels:
            raise ValueError("Classifier labels must include an unknown class")
        if not 0 < unknown_threshold <= 1:
            raise ValueError("Unknown rejection threshold must be in (0,1]")
        self.unknown_threshold = unknown_threshold
        self.torch = None
        self.acoustic_model = None
        self.sonar_model = None
        if acoustic_model_path or sonar_model_path:
            for path in (acoustic_model_path, sonar_model_path):
                if path and not Path(path).is_file():
                    raise FileNotFoundError(f"TorchScript classifier model not found: {path}")
            try:
                import torch
            except ImportError as error:
                raise RuntimeError("TorchScript paths require PyTorch; the NumPy baseline needs no PyTorch") from error
            self.torch = torch
            if acoustic_model_path:
                self.acoustic_model = torch.jit.load(str(acoustic_model_path), map_location="cpu").eval()
            if sonar_model_path:
                self.sonar_model = torch.jit.load(str(sonar_model_path), map_location="cpu").eval()

    @staticmethod
    def sonar_tensor_features(features: Mapping) -> np.ndarray:
        """Scalar echo features, not truth labels; same preprocessing for training."""
        values = np.array([float(features.get("range_m", 0)) / 30,
                           float(features.get("echo_strength", 0)),
                           float(features.get("size_m", 0)) / 3,
                           float(features.get("speed_m_s", 0)) / 2,
                           float(features.get("elevation_rad", 0)) / math.pi], dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("Sonar features must be finite")
        return np.clip(values, -2, 2)

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

    def classify(self, samples, sonar_features: Mapping, sample_rate: int = 8000) -> dict:
        acoustic_features = analyze_hydrophone(samples, sample_rate)
        sonar_vector = self.sonar_tensor_features(sonar_features)
        engine_score = acoustic_features["engine_score"]
        label = str(sonar_features.get("sonar_label", sonar_features.get("baseline_kind", "unknown")))
        label = {"environmental_hazard": "obstacle"}.get(label, label)
        if label not in self.class_labels:
            label = "unknown"
        confidence = 0.80 if label != "unknown" else 0.45
        if engine_score > 0.60 and label in ("unknown", "vehicle") and "vehicle" in self.class_labels:
            label, confidence = "vehicle", 0.68 + 0.25 * engine_score
        elif label == "vehicle" and engine_score < 0.20:
            confidence = 0.58  # No supporting acoustic evidence: reject certainty.
        probabilities = np.full(len(self.class_labels),
                                (1 - confidence) / max(len(self.class_labels) - 1, 1))
        probabilities[self.class_labels.index(label)] = confidence
        used_models = []
        acoustic_probs, sonar_probs = None, None
        if self.acoustic_model is not None:
            acoustic_input = spectrogram(samples, sample_rate)[None, None, :, :]
            acoustic_probs = self._infer(self.acoustic_model, acoustic_input)
            used_models.append("acoustic_torchscript")
        if self.sonar_model is not None:
            sonar_probs = self._infer(self.sonar_model, sonar_vector[None, :])
            used_models.append("sonar_torchscript")
        if acoustic_probs is not None and sonar_probs is not None:
            probabilities = 0.4 * acoustic_probs + 0.6 * sonar_probs
        elif acoustic_probs is not None:
            probabilities = acoustic_probs
        elif sonar_probs is not None:
            probabilities = sonar_probs
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        label = self.class_labels[index]
        rejected = confidence < self.unknown_threshold
        if rejected:
            label = "unknown"
        return {"classification": label, "confidence": round(confidence, 4),
                "unknown_rejected": rejected,
                "backend": "+".join(used_models) if used_models else "untrained_feature_baseline",
                "acoustic_features": acoustic_features,
                "class_probabilities": {key: round(float(value), 4)
                                        for key, value in zip(self.class_labels, probabilities)},
                "position_from_passive_audio": False}
