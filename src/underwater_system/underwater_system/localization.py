"""Passive source localization from hydrophone time differences (TDOA).

This module adds the passive localization step described in the literature
review: each node estimates the arrival-time difference of the same sound
between two hydrophones with GCC-PHAT, and several time differences are
combined by least-squares multilateration into a source position.

Coordinates are ENU metres, time is seconds. Only measured waveforms and node
positions are used; no ground truth is needed. Accuracy depends on clock
synchronization and on how well each node knows its own position: with
c = 1500 m/s, a 1 ms timing error equals a 1.5 m range-difference error.
"""
from __future__ import annotations

import math

import numpy as np

SOUND_SPEED_M_S = 1500.0


def gcc_phat(signal, reference, sample_rate: float, max_tau: float | None = None,
             interp: int = 16) -> tuple[float, float]:
    """Estimate how much ``signal`` lags ``reference`` with GCC-PHAT.

    Returns ``(tau_s, sharpness)``. A positive ``tau_s`` means the sound
    reached ``signal``'s hydrophone later. ``sharpness`` is the correlation
    peak divided by the mean correlation level: a simple quality score where
    low values suggest noise or multipath rather than one clear arrival.
    ``max_tau`` limits the search to physically possible delays, for example
    the hydrophone spacing divided by the speed of sound.
    """
    x = np.asarray(signal, dtype=float).reshape(-1)
    y = np.asarray(reference, dtype=float).reshape(-1)
    if x.size < 2 or y.size < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("GCC-PHAT needs two finite waveforms of at least two samples")
    if sample_rate <= 0 or interp < 1:
        raise ValueError("sample_rate and interp must be positive")
    n = x.size + y.size
    spectrum = np.fft.rfft(x, n=n) * np.conj(np.fft.rfft(y, n=n))
    # PHAT weighting keeps only phase, which sharpens the peak in reverberation.
    spectrum /= np.maximum(np.abs(spectrum), 1e-12)
    correlation = np.fft.irfft(spectrum, n=interp * n)
    max_shift = interp * n // 2
    if max_tau is not None:
        max_shift = min(int(interp * sample_rate * max_tau), max_shift)
    correlation = np.concatenate((correlation[-max_shift:], correlation[:max_shift + 1]))
    index = int(np.argmax(np.abs(correlation)))
    tau = (index - max_shift) / float(interp * sample_rate)
    magnitude = np.abs(correlation)
    sharpness = float(magnitude[index] / max(float(np.mean(magnitude)), 1e-12))
    return tau, sharpness


def multilaterate(positions, tdoas, reference_index: int = 0,
                  sound_speed: float = SOUND_SPEED_M_S, initial=None,
                  iterations: int = 50) -> dict:
    """Estimate a source position from time differences of arrival.

    ``positions`` is an (N, D) array of hydrophone positions, D = 2 or 3.
    ``tdoas`` holds N arrival times relative to the reference hydrophone
    (``tdoas[reference_index]`` is ignored). Needs N >= D + 1. Uses
    Gauss-Newton least squares and returns the position, the RMS residual
    in metres and whether the solver converged.
    """
    nodes = np.asarray(positions, dtype=float)
    delays = np.asarray(tdoas, dtype=float).reshape(-1)
    if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
        raise ValueError("positions must be an (N, 2) or (N, 3) array")
    if delays.size != nodes.shape[0] or not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(delays)):
        raise ValueError("Need one finite time difference per hydrophone")
    if nodes.shape[0] < nodes.shape[1] + 1:
        raise ValueError("Need at least D + 1 hydrophones for D-dimensional localization")
    others = [i for i in range(nodes.shape[0]) if i != reference_index]
    reference = nodes[reference_index]
    measured = (delays[others] - delays[reference_index]) * sound_speed
    estimate = np.mean(nodes, axis=0) if initial is None else np.asarray(initial, dtype=float).copy()
    converged = False
    for _ in range(iterations):
        to_reference = estimate - reference
        d_ref = max(float(np.linalg.norm(to_reference)), 1e-9)
        residuals, jacobian = [], []
        for i in others:
            to_node = estimate - nodes[i]
            d_i = max(float(np.linalg.norm(to_node)), 1e-9)
            residuals.append((d_i - d_ref) - measured[len(residuals)])
            jacobian.append(to_node / d_i - to_reference / d_ref)
        step, *_ = np.linalg.lstsq(np.array(jacobian), -np.array(residuals), rcond=None)
        estimate = estimate + step
        if float(np.linalg.norm(step)) < 1e-6:
            converged = True
            break
    rms = math.sqrt(float(np.mean(np.square(residuals)))) if residuals else 0.0
    return {"position": estimate.tolist(), "rms_residual_m": rms, "converged": converged}


def bearing_from_pair(tau_s: float, spacing_m: float,
                      sound_speed: float = SOUND_SPEED_M_S) -> float:
    """Angle (radians) from the broadside of a two-hydrophone pair.

    A single pair cannot tell front from back; that mirror ambiguity is why
    at least three non-collinear hydrophones are needed for a unique bearing.
    """
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    return math.asin(float(np.clip(tau_s * sound_speed / spacing_m, -1.0, 1.0)))
