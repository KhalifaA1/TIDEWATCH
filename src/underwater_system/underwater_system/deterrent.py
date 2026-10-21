"""Acoustic deterrent speakers and the wildlife response they provoke.

``DeterrentSpeaker`` runs on each AUV and decides from *estimates only* (tracked
contacts and the diver beacon) whether to sound a deterrent. It picks the lowest
source level that should reach the approaching animal, and never exceeds a
received-level cap at the diver. A duty cycle limits exposure and habituation.

``WildlifeResponse`` is truth-side: it turns the sound field of active speakers
into a flee velocity for simulated marine life, with habituation and recovery.

Levels are dB re 1 uPa with spherical spreading plus a constant absorption
term. Thresholds are illustrative simulation values, not species audiograms or
diver-safety limits; real use needs measured hearing data and safety review.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Optional

import numpy as np

ABSORPTION_DB_PER_M = 0.02


def received_level_db(source_level_db: float, range_m: float) -> float:
    """Spherical spreading plus absorption, one metre reference."""
    r = max(float(range_m), 1.0)
    return float(source_level_db) - 20.0 * math.log10(r) - ABSORPTION_DB_PER_M * r


def source_level_for(target_level_db: float, range_m: float) -> float:
    """Source level needed for a given received level at ``range_m``."""
    r = max(float(range_m), 1.0)
    return float(target_level_db) + 20.0 * math.log10(r) + ABSORPTION_DB_PER_M * r


def time_to_enter(relative_position, relative_velocity, radius: float) -> Optional[float]:
    """Earliest t >= 0 at which |p + v t| <= radius, or None if never."""
    p = np.asarray(relative_position, dtype=float)
    v = np.asarray(relative_velocity, dtype=float)
    c = float(p @ p - radius ** 2)
    if c <= 0:
        return 0.0
    a = float(v @ v)
    b = float(p @ v)
    if a < 1e-9 or b >= 0:
        return None
    disc = b * b - a * c
    if disc < 0:
        return None
    return max(0.0, (-b - math.sqrt(disc)) / a)


@dataclass
class DeterrentConfig:
    enabled: bool = True
    min_source_level_db: float = 135.0
    max_source_level_db: float = 175.0
    range_m: float = 24.0
    deterrence_level_db: float = 110.0   # received level the animal should hear
    diver_limit_db: float = 135.0        # received-level cap at the diver
    warning_radius_m: float = 9.0        # animals predicted inside this are deterred
    horizon_s: float = 30.0
    max_on_s: float = 6.0
    cooldown_s: float = 4.0
    min_confidence: float = 0.5

    @classmethod
    def from_mapping(cls, value: Optional[Mapping]) -> "DeterrentConfig":
        value = dict(value or {})
        known = {key: value[key] for key in cls.__dataclass_fields__ if key in value}
        config = cls(**known)
        if not (0 < config.min_source_level_db <= config.max_source_level_db):
            raise ValueError("Deterrent source levels must be positive and ordered")
        if config.range_m <= 0 or config.max_on_s <= 0 or config.cooldown_s < 0:
            raise ValueError("Deterrent range and duty cycle must be positive")
        return config


class DeterrentSpeaker:
    """One AUV's speaker controller; uses estimated tracks, never truth."""

    def __init__(self, config: Optional[Mapping | DeterrentConfig] = None):
        self.config = config if isinstance(config, DeterrentConfig) else DeterrentConfig.from_mapping(config)
        self.on_since: Optional[float] = None
        self.cooldown_until = -math.inf
        self.activations = 0
        self.last = self._idle("idle")

    @staticmethod
    def _idle(reason: str) -> dict:
        return {"active": False, "level_db": 0.0, "target_id": None, "target_class": None,
                "reason": reason, "limited_by_diver_safety": False,
                "received_at_target_db": None, "received_at_diver_db": None}

    def _threats(self, position, diver, diver_velocity, tracks: Iterable[Mapping]):
        cfg = self.config
        for track in tracks:
            label = track.get("classification", "unknown")
            if label != "marine_life" or float(track.get("confidence", 0)) < cfg.min_confidence:
                continue
            p = np.asarray(track["position"], dtype=float)
            v = np.asarray(track.get("velocity", [0, 0, 0]), dtype=float)
            eta = time_to_enter(p - diver, v - diver_velocity, cfg.warning_radius_m)
            if eta is None or eta > cfg.horizon_s:
                continue
            reach = float(np.linalg.norm(p - position))
            if reach > cfg.range_m:
                continue
            yield eta, reach, track

    def decide(self, now: float, position, diver_position, diver_velocity, tracks: Iterable[Mapping]) -> dict:
        cfg = self.config
        position = np.asarray(position, dtype=float)
        diver = np.asarray(diver_position, dtype=float)
        diver_v = np.asarray(diver_velocity, dtype=float)
        if not cfg.enabled:
            self.last = self._idle("disabled")
            return self.last
        if self.on_since is not None and now - self.on_since >= cfg.max_on_s:
            self.on_since, self.cooldown_until = None, now + cfg.cooldown_s
        if now < self.cooldown_until:
            self.last = self._idle("cooldown")
            return self.last
        threats = sorted(self._threats(position, diver, diver_v, tracks), key=lambda item: item[0])
        if not threats:
            self.on_since = None
            self.last = self._idle("no_threat")
            return self.last
        eta, reach, track = threats[0]
        wanted = source_level_for(cfg.deterrence_level_db, reach)
        diver_range = float(np.linalg.norm(position - diver))
        diver_cap = source_level_for(cfg.diver_limit_db, diver_range)
        level = min(max(wanted, cfg.min_source_level_db), cfg.max_source_level_db)
        limited = level > diver_cap
        level = min(level, diver_cap)
        if level < cfg.min_source_level_db * 0.9:
            self.on_since = None
            self.last = self._idle("diver_safety_inhibit")
            return self.last
        if self.on_since is None:
            self.on_since = now
            self.activations += 1
        self.last = {"active": True, "level_db": round(level, 1), "target_id": track.get("id"),
                     "target_class": track.get("classification"), "eta_s": round(eta, 1),
                     "reason": "wildlife_approaching_diver", "limited_by_diver_safety": limited,
                     "received_at_target_db": round(received_level_db(level, reach), 1),
                     "received_at_diver_db": round(received_level_db(level, diver_range), 1)}
        return self.last


class WildlifeResponse:
    """Truth-side flee behaviour driven by the sound field of active speakers."""

    def __init__(self, hearing_threshold_db: float = 100.0, flee_speed: float = 1.4,
                 habituation_per_s: float = 0.015, recovery_per_s: float = 0.004):
        self.hearing_threshold_db = hearing_threshold_db
        self.flee_speed = flee_speed
        self.habituation_per_s = habituation_per_s
        self.recovery_per_s = recovery_per_s
        self.habituation: dict[str, float] = {}
        self.startled_s: dict[str, float] = {}

    def velocity(self, animal_id: str, animal_position, speakers: Iterable[Mapping], dt: float) -> np.ndarray:
        """Flee velocity (ENU) for one animal; speakers have position, level_db."""
        animal = np.asarray(animal_position, dtype=float)
        sensitivity = self.habituation.get(animal_id, 1.0)
        push = np.zeros(3)
        loudest = 0.0
        for speaker in speakers:
            level = float(speaker.get("level_db", 0.0))
            if level <= 0:
                continue
            offset = animal - np.asarray(speaker["position"], dtype=float)
            heard = received_level_db(level, float(np.linalg.norm(offset)))
            excess = heard - self.hearing_threshold_db
            if excess <= 0:
                continue
            loudest = max(loudest, excess)
            direction = offset / max(float(np.linalg.norm(offset)), 1e-6)
            push += direction * min(1.0, excess / 25.0)
        if loudest > 0:
            self.habituation[animal_id] = max(0.25, sensitivity - self.habituation_per_s * dt)
            self.startled_s[animal_id] = self.startled_s.get(animal_id, 0.0) + dt
            magnitude = float(np.linalg.norm(push))
            if magnitude > 1e-9:
                return push / magnitude * min(1.0, magnitude) * self.flee_speed * sensitivity
        else:
            self.habituation[animal_id] = min(1.0, sensitivity + self.recovery_per_s * dt)
        return np.zeros(3)
