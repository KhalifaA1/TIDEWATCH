import math

import numpy as np
import pytest

from underwater_system.localization import (SOUND_SPEED_M_S, bearing_from_pair,
                                            gcc_phat, multilaterate)


def delayed_noise(delay_s, sample_rate=48000, n=4096, seed=3):
    rng = np.random.default_rng(seed)
    source = rng.normal(0, 1, n + 400)
    shift = int(round(delay_s * sample_rate))
    reference = source[200:200 + n]
    signal = source[200 - shift:200 - shift + n]
    return signal, reference


def test_gcc_phat_recovers_known_delay():
    signal, reference = delayed_noise(40 / 48000)
    tau, sharpness = gcc_phat(signal, reference, 48000, max_tau=0.005)
    assert abs(tau - 40 / 48000) < 1 / 48000
    assert sharpness > 5


def test_gcc_phat_negative_delay():
    signal, reference = delayed_noise(-25 / 48000)
    tau, _ = gcc_phat(signal, reference, 48000, max_tau=0.005)
    assert abs(tau + 25 / 48000) < 1 / 48000


def test_multilateration_2d_four_nodes():
    nodes = np.array([[0, 0], [30, 0], [0, 30], [30, 30]], dtype=float)
    source = np.array([12.0, 21.0])
    arrival = np.linalg.norm(nodes - source, axis=1) / SOUND_SPEED_M_S
    result = multilaterate(nodes, arrival - arrival[0])
    assert result["converged"]
    assert np.linalg.norm(np.array(result["position"]) - source) < 0.01


def test_multilateration_3d_with_timing_noise():
    rng = np.random.default_rng(1)
    nodes = np.array([[0, 0, -5], [25, 0, -8], [0, 25, -6], [25, 25, -9], [12, 12, -2]], dtype=float)
    source = np.array([9.0, 14.0, -7.0])
    arrival = np.linalg.norm(nodes - source, axis=1) / SOUND_SPEED_M_S
    noisy = arrival + rng.normal(0, 20e-6, arrival.size)  # 20 microsecond clock error
    result = multilaterate(nodes, noisy - noisy[0], initial=[12, 12, -5])
    assert np.linalg.norm(np.array(result["position"]) - source) < 1.0


def test_multilateration_rejects_too_few_nodes():
    with pytest.raises(ValueError):
        multilaterate([[0, 0], [1, 0]], [0, 0])


def test_bearing_from_pair():
    spacing = 0.5
    angle = math.radians(30)
    tau = spacing * math.sin(angle) / SOUND_SPEED_M_S
    assert abs(bearing_from_pair(tau, spacing) - angle) < 1e-9
