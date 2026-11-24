"""Deterministic, ROS-independent algorithms for the TIDEWATCH demonstrator.

All vectors use ENU metres: x east, y north, z up; underwater z is negative.
Ground truth belongs only in ``SensorModel`` and the offline ``SimulationCore``.
``EdgeAgent`` accepts estimated navigation and observations, and does not require
ground truth. Classification is a synthetic feature/template baseline, not a
trained neural network or a validated real-world acoustic classifier.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import heapq
import json
import math
from typing import Iterable, Mapping, Optional

import numpy as np

from .deterrent import DeterrentSpeaker, WildlifeResponse, time_to_enter
from .mission import MissionPlanner

CLASS_NAMES = ("unknown", "diver", "marine_life", "vehicle", "obstacle")
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}
SOUND_SPEED_M_S = 1500.0


def vec3(value) -> np.ndarray:
    """Copy a finite three-component vector, rejecting invalid telemetry."""
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError("A 3D vector must contain finite values")
    return vector.copy()


def yaw_rotation(yaw: float) -> np.ndarray:
    """Rotation from a body's x-forward, y-left, z-up frame into ENU."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def local_to_world(point, position, yaw: float) -> np.ndarray:
    return yaw_rotation(yaw) @ vec3(point) + vec3(position)


def world_to_local(point, position, yaw: float) -> np.ndarray:
    return yaw_rotation(yaw).T @ (vec3(point) - vec3(position))


def bounded(vector, magnitude: float) -> np.ndarray:
    vector = vec3(vector)
    length = float(np.linalg.norm(vector))
    return vector * min(1.0, magnitude / max(length, 1e-12))


def seabed_height(x: float, y: float) -> float:
    """Analytic simulation seabed; use only in sensors and truth evaluation."""
    return -16.0 + 0.6 * math.sin(x * 0.12) * math.cos(y * 0.10)


@dataclass
class WorldObject:
    object_id: str
    kind: str
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 1.0

    def __post_init__(self):
        self.position = vec3(self.position)
        self.velocity = vec3(self.velocity)
        self.kind = {"environmental_hazard": "obstacle", "person": "diver"}.get(self.kind, self.kind)


@dataclass
class Observation:
    """A sensor estimate. ``truth_id`` is private evaluation metadata only.

    Network conversion deliberately discards truth IDs and transmits an
    isotropic upper covariance bound, retaining uncertainty under quantization.
    """
    uid: str
    source: str
    timestamp: float
    position: np.ndarray
    covariance: np.ndarray
    classification: str
    confidence: float
    modality: str = "fused"
    truth_id: Optional[str] = None

    def __post_init__(self):
        self.position = vec3(self.position)
        self.covariance = np.asarray(self.covariance, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(self.covariance)):
            raise ValueError("Non-finite observation covariance")
        self.covariance = (self.covariance + self.covariance.T) * 0.5
        if np.min(np.linalg.eigvalsh(self.covariance)) <= 0:
            raise ValueError("Observation covariance must be positive definite")
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if self.classification not in CLASS_IDS:
            self.classification = "unknown"

    def to_payload(self) -> dict:
        variance = float(np.max(np.linalg.eigvalsh(self.covariance))) + 1e-6
        return {"u": self.uid, "s": self.source, "t": round(self.timestamp, 3),
                "p": np.round(self.position, 3).tolist(), "r": round(variance, 5),
                "k": CLASS_IDS[self.classification], "f": round(self.confidence, 2),
                "m": self.modality}

    @classmethod
    def from_payload(cls, payload: Mapping) -> "Observation":
        label_id = int(payload.get("k", 0))
        label = CLASS_NAMES[label_id] if 0 <= label_id < len(CLASS_NAMES) else "unknown"
        variance = max(float(payload.get("r", 0.25)), 1e-5)
        return cls(str(payload["u"]), str(payload["s"]), float(payload["t"]),
                   payload["p"], np.eye(3) * variance, label,
                   float(payload.get("f", 0.0)), str(payload.get("m", "fused")))


class SensorModel:
    """Synthetic active sonar geometry fused with acoustic/silhouette features.

    ``observe(source, observer_position, observer_yaw, objects, timestamp)``
    returns independent noisy estimates. Active sonar supplies range, bearing
    and elevation; passive acoustic features supply classification evidence
    and never claim to provide a 3D position from a single hydrophone.
    ``sample_environment`` supplies finite sonar beam endpoints for mapping.
    """
    def __init__(self, seed: int = 7, range_m: float = 28.0,
                 horizontal_fov: float = math.radians(220),
                 unknown_threshold: float = 0.62):
        self.rng = np.random.default_rng(seed)
        self.range_m = range_m
        self.horizontal_fov = horizontal_fov
        self.unknown_threshold = unknown_threshold
        self.sequence = defaultdict(int)
        # Echo-signature emulator and template classifier shared with the ROS
        # edge path, so offline accuracy no longer reads the true object type.
        from .perception import EdgePerception, SonarFeatureModel
        self.echo_model = SonarFeatureModel(seed + 331)
        self.edge_classifier = EdgePerception({"unknown_threshold": unknown_threshold})

    def classify_echo(self, kind: str, distance: float, speed: float, size: float) -> tuple[str, float]:
        """Classify a synthesized echo from measurable features only."""
        features = self.echo_model.measure(kind, distance, speed, size)
        probabilities, details = self.edge_classifier._baseline(features)
        labels = self.edge_classifier.class_labels
        index = int(np.argmax(probabilities))
        label, confidence = labels[index], float(probabilities[index])
        if label == "unknown" or confidence < self.unknown_threshold:
            return "unknown", float(probabilities[labels.index("unknown")])
        return label, confidence

    def passive_features(self, kind: str, distance: float) -> dict:
        """Synthetic narrowband/tonality features, explicitly not raw audio."""
        templates = {"vehicle": (0.85, 0.82, 0.70), "diver": (0.18, 0.35, 0.12),
                     "marine_life": (0.10, 0.18, 0.55), "obstacle": (0.02, 0.02, 0.03),
                     "unknown": (0.47, 0.52, 0.42)}
        noise = 0.035 + distance * 0.0015
        values = np.clip(np.array(templates.get(kind, templates["unknown"])) +
                         self.rng.normal(0, noise, 3), 0.0, 1.0)
        return {"engine_tonality": float(values[0]), "harmonic_ratio": float(values[1]),
                "broadband_ratio": float(values[2]),
                "relative_snr_db": float(24 - 20 * math.log10(max(distance, 1)))}

    def classify(self, features: Mapping, sonar_kind: str = "unknown") -> tuple[str, float]:
        """Transparent feature rules for the simulation's untrained baseline."""
        tone = float(features.get("engine_tonality", 0))
        harmonic = float(features.get("harmonic_ratio", 0))
        if tone > 0.65 and harmonic > 0.60:
            label, confidence = "vehicle", 0.82 + 0.12 * min(tone, harmonic)
        elif sonar_kind in ("diver", "marine_life", "obstacle"):
            label, confidence = sonar_kind, 0.82
        else:
            label, confidence = "unknown", 0.46
        confidence = float(np.clip(confidence + self.rng.normal(0, 0.035), 0, 1))
        if confidence < self.unknown_threshold:
            label = "unknown"
        return label, confidence

    def observe(self, source: str, observer_position, observer_yaw: float,
                objects: Iterable[WorldObject], timestamp: float) -> list[Observation]:
        origin = vec3(observer_position)
        rotation = yaw_rotation(observer_yaw)
        observations = []
        for obj in objects:
            local = world_to_local(obj.position, origin, observer_yaw)
            distance = float(np.linalg.norm(local))
            if distance < 0.4 or distance > self.range_m:
                continue
            bearing = math.atan2(local[1], local[0])
            if abs(bearing) > self.horizontal_fov / 2:
                continue
            # Miss probability rises with range. It is separate from link loss.
            if self.rng.random() < 0.015 + 0.07 * (distance / self.range_m) ** 2:
                continue
            elevation = math.atan2(local[2], np.linalg.norm(local[:2]))
            range_sigma, angle_sigma = 0.045 + 0.006 * distance, math.radians(0.7)
            measured_r = max(0.1, distance + self.rng.normal(0, range_sigma))
            b = bearing + self.rng.normal(0, angle_sigma)
            e = elevation + self.rng.normal(0, angle_sigma)
            ce, se, cb, sb = math.cos(e), math.sin(e), math.cos(b), math.sin(b)
            measured = np.array([measured_r * ce * cb, measured_r * ce * sb,
                                 measured_r * se])
            jacobian = np.array([[ce * cb, -measured_r * ce * sb, -measured_r * se * cb],
                                 [ce * sb, measured_r * ce * cb, -measured_r * se * sb],
                                 [se, 0.0, measured_r * ce]])
            local_cov = jacobian @ np.diag([range_sigma ** 2, angle_sigma ** 2,
                                           angle_sigma ** 2]) @ jacobian.T
            covariance = rotation @ local_cov @ rotation.T + np.eye(3) * 0.01
            label, confidence = self.classify_echo(obj.kind, distance, float(np.linalg.norm(obj.velocity)),
                                                   max(0.3, 2 * obj.radius))
            self.sequence[source] += 1
            observations.append(Observation(f"{source}:{self.sequence[source]}", source,
                                timestamp, origin + rotation @ measured, covariance,
                                label, confidence, "sonar+acoustic", obj.object_id))
        return observations

    def sample_environment(self, observer_position, observer_yaw: float,
                           timestamp: float = 0.0, max_samples: int = 24) -> np.ndarray:
        """Sample finite downward sonar beams; unobserved ground stays unknown."""
        origin = vec3(observer_position)
        points = []
        for index in range(max_samples):
            bearing = observer_yaw + self.rng.uniform(-1.4, 1.4)
            horizontal = self.rng.uniform(1.0, 11.0)
            x = origin[0] + horizontal * math.cos(bearing)
            y = origin[1] + horizontal * math.sin(bearing)
            point = np.array([x, y, seabed_height(x, y)])
            if np.linalg.norm(point - origin) <= self.range_m:
                points.append(point + self.rng.normal(0, 0.06, 3))
        return np.asarray(points, dtype=float).reshape(-1, 3)


class FormationController:
    """A bounded decentralized formation/Boids velocity controller.

    ``command`` uses this node's estimate, a diver beacon estimate, delayed
    peer estimates and locally observed obstacles. It never reads world truth.
    Obstacle repulsion and velocity half-space projections protect separation;
    Gazebo physics and an emergency supervisor remain authoritative for motion.
    """
    def __init__(self, index: int, radius: float = 6.0, speed_limit: float = 1.3,
                 acceleration_limit: float = 0.8, minimum_separation: float = 2.4):
        self.index = index
        self.radius = radius
        self.speed_limit = speed_limit
        self.acceleration_limit = acceleration_limit
        self.minimum_separation = minimum_separation

    def target(self, diver_position) -> np.ndarray:
        angle = self.index * 2 * math.pi / 3
        return vec3(diver_position) + np.array([self.radius * math.cos(angle),
                                               self.radius * math.sin(angle), 0.0])

    def command(self, position, velocity, diver_position, diver_velocity,
                peers=(), obstacles=(), dt: float = 0.1, navigation_goal=None,
                protect_diver: bool = True, goal_velocity=None) -> np.ndarray:
        position, velocity = vec3(position), vec3(velocity)
        if navigation_goal is None:
            goal = self.target(diver_position)
            feedforward = vec3(diver_velocity)
        else:
            goal = vec3(navigation_goal)
            feedforward = np.zeros(3) if goal_velocity is None else vec3(goal_velocity)
        desired = feedforward + 0.62 * (goal - position)
        # Alignment/cohesion receive only acoustic mesh peer states.
        peer_positions, peer_velocities = [], []
        safety_constraints = []
        for peer_position, peer_velocity in peers:
            peer_p, peer_v = vec3(peer_position), vec3(peer_velocity)
            difference = position - peer_p
            distance = float(np.linalg.norm(difference))
            if distance < 1e-6:
                difference = np.array([math.cos(self.index * 2 * math.pi / 3),
                                       math.sin(self.index * 2 * math.pi / 3), 0.0])
                distance = 1e-6
            normal = difference / max(distance, 1e-6)
            peer_positions.append(peer_p)
            peer_velocities.append(peer_v)
            if distance < 4.0:
                desired += normal * min(2.5, 1.2 * (4.0 - distance))
                safety_constraints.append((normal, peer_v,
                                           -0.7 * (distance - self.minimum_separation)))
        if peer_positions:
            desired += 0.10 * (np.mean(peer_velocities, axis=0) - velocity)
            desired += 0.025 * (np.mean(peer_positions, axis=0) - position)
        # Keep a clearance around the diver as well as environmental objects.
        all_obstacles = list(obstacles)
        if protect_diver:
            all_obstacles.append((vec3(diver_position), 1.4))
        for obstacle in all_obstacles:
            obstacle_position, radius = obstacle[:2]
            obstacle_velocity = vec3(obstacle[2]) if len(obstacle) > 2 else np.zeros(3)
            difference = position - vec3(obstacle_position)
            distance = float(np.linalg.norm(difference))
            safe = float(radius) + 1.0
            # Moving wildlife and vehicles are physical clearance hazards.
            # Predict a short-horizon crossing using the commanded intent,
            # instead of waiting until the contact reaches the vehicle.
            relative_velocity = bounded(desired, self.speed_limit) - obstacle_velocity
            cpa, encounter_time = closest_approach(difference, relative_velocity, horizon=5.0)
            if 0.05 < encounter_time and cpa < safe + 0.7:
                future_difference = difference + relative_velocity * encounter_time
                normal = future_difference / max(float(np.linalg.norm(future_difference)), 1e-6)
                if np.linalg.norm(normal) < 0.5:
                    # A vertical escape is deterministic for head-on horizontal
                    # crossings and avoids choosing an arbitrary side of a peer.
                    normal = np.array([0.0, 0.0, -1.0])
                desired += normal * min(2.6, (safe + 0.7 - cpa) * 1.3)
            if distance < safe + 2.0:
                normal = difference / max(distance, 1e-6)
                if distance < 1e-6:
                    normal = np.array([math.cos(self.index * 2 * math.pi / 3),
                                       math.sin(self.index * 2 * math.pi / 3), 0.0])
                desired += normal * min(3.0, (safe + 2 - distance) * 1.4)
                safety_constraints.append((normal, obstacle_velocity, -0.7 * (distance - safe)))
        desired = bounded(desired, self.speed_limit)
        commanded = velocity + bounded(desired - velocity,
                                       self.acceleration_limit * max(dt, 1e-3))
        # Safety projection takes precedence over ordinary acceleration limiting.
        for _ in range(3):
            for normal, other_v, minimum_outward in safety_constraints:
                outward = float(np.dot(commanded - other_v, normal))
                if outward < minimum_outward:
                    commanded += normal * (minimum_outward - outward)
            commanded = bounded(commanded, self.speed_limit)
        return commanded


@dataclass
class Track:
    track_id: str
    state: np.ndarray
    covariance: np.ndarray
    last_seen: float
    predicted_at: float
    classification: str
    confidence: float
    hits: int = 1
    history: deque = field(default_factory=lambda: deque(maxlen=80))
    class_evidence: dict = field(default_factory=dict)
    sources: set = field(default_factory=set)

    @property
    def position(self):
        return self.state[:3]

    @property
    def velocity(self):
        return self.state[3:]

    def snapshot(self) -> dict:
        std = float(math.sqrt(np.max(np.linalg.eigvalsh(self.covariance[:3, :3]))))
        return {"id": self.track_id, "position": np.round(self.position, 3).tolist(),
                "velocity": np.round(self.velocity, 3).tolist(),
                "covariance": np.round(self.covariance[:3, :3], 5).tolist(),
                "position_std_m": std, "uncertainty": std,
                "classification": self.classification, "label": self.classification,
                "confidence": round(self.confidence, 3),
                "last_seen": round(self.last_seen, 3), "hits": self.hits,
                "sources": sorted(self.sources), "history": list(self.history)}


class MultiObjectTracker:
    """Constant-velocity Kalman tracks with statistical gating and UID dedup.

    ``update(observations, now)`` accepts Observation instances or wire payloads.
    Different sensors can update one track; observations from the same source
    and timestamp cannot consume the same track twice. Association uses only
    measured geometry/classification, never a simulation object's truth ID.
    """
    def __init__(self, prefix: str = "track", max_age: float = 8.0,
                 gate: float = 16.27):
        self.prefix, self.max_age, self.gate = prefix, max_age, gate
        self.tracks: dict[str, Track] = {}
        self.counter = 0
        self.seen: dict[str, float] = {}
        self.duplicates = 0
        self.rejected = 0

    @staticmethod
    def _predict(track: Track, now: float):
        dt = max(0.0, now - track.predicted_at)
        if dt == 0:
            return
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        # White acceleration noise supports maneuvering without zero uncertainty.
        q = 0.08 ** 2
        process = np.zeros((6, 6))
        process[:3, :3] = np.eye(3) * dt ** 4 * q / 4
        process[:3, 3:] = process[3:, :3] = np.eye(3) * dt ** 3 * q / 2
        process[3:, 3:] = np.eye(3) * dt ** 2 * q
        track.state = transition @ track.state
        track.covariance = transition @ track.covariance @ transition.T + process
        track.predicted_at = now

    def update(self, observations: Iterable, now: float):
        for track in self.tracks.values():
            self._predict(track, now)
        assigned = defaultdict(set)
        normalized = [Observation.from_payload(o) if isinstance(o, Mapping) else o
                      for o in observations]
        for observation in sorted(normalized, key=lambda o: (o.timestamp, o.source, o.uid)):
            if observation.uid in self.seen:
                self.duplicates += 1
                continue
            if now - observation.timestamp > self.max_age or observation.timestamp > now + 0.1:
                self.rejected += 1
                continue
            self.seen[observation.uid] = now
            batch_key = (observation.source, round(observation.timestamp, 3))
            candidates = []
            age = max(0.0, now - observation.timestamp)
            # A delayed report measures past position, not the present position
            # with a supposedly exact velocity correction. This observation
            # matrix retains velocity uncertainty and position/velocity cross
            # covariance during both association and the Kalman update.
            measurement = np.zeros((3, 6))
            measurement[:, :3] = np.eye(3)
            measurement[:, 3:] = -np.eye(3) * age
            # Allow unobserved acceleration between acquisition and reception.
            delay_variance = (0.08 * age ** 2 / 2) ** 2
            noise = observation.covariance + np.eye(3) * delay_variance
            for track in self.tracks.values():
                if track.track_id in assigned[batch_key]:
                    continue
                innovation = observation.position - measurement @ track.state
                total_cov = measurement @ track.covariance @ measurement.T + noise
                squared = float(innovation @ np.linalg.solve(total_cov, innovation))
                incompatible = (track.classification != observation.classification and
                                "unknown" not in (track.classification, observation.classification))
                if incompatible and track.confidence > 0.7 and observation.confidence > 0.7:
                    squared += 6.0
                if squared <= self.gate and np.linalg.norm(innovation) < 3.5:
                    candidates.append((squared, track, innovation, noise))
            if candidates:
                _, track, innovation, noise = min(candidates, key=lambda item: item[0])
                total_cov = measurement @ track.covariance @ measurement.T + noise
                gain = np.linalg.solve(total_cov, measurement @ track.covariance).T
                track.state += gain @ innovation
                residual = np.eye(6) - gain @ measurement
                # Joseph form preserves covariance positive semidefiniteness.
                track.covariance = residual @ track.covariance @ residual.T + gain @ noise @ gain.T
                track.last_seen = max(track.last_seen, observation.timestamp)
                track.hits += 1
                evidence = track.class_evidence
                evidence[observation.classification] = evidence.get(observation.classification, 0) + observation.confidence
                track.classification = max(evidence, key=evidence.get)
                track.confidence = 0.8 * track.confidence + 0.2 * observation.confidence
            else:
                self.counter += 1
                track_id = f"{self.prefix}-{self.counter:03d}"
                covariance = np.eye(6) * 0.6 ** 2
                covariance[:3, :3] = observation.covariance
                track = Track(track_id, np.r_[observation.position, np.zeros(3)], covariance,
                              observation.timestamp, observation.timestamp, observation.classification,
                              observation.confidence)
                # A first delayed measurement initializes at acquisition time,
                # then propagates its entire covariance to reception time.
                self._predict(track, now)
                track.class_evidence[observation.classification] = observation.confidence
                self.tracks[track_id] = track
            track.sources.add(observation.source)
            assigned[batch_key].add(track.track_id)
            track.history.append({"t": round(now, 2), "p": np.round(track.position, 3).tolist()})
        self.tracks = {key: track for key, track in self.tracks.items()
                       if now - track.last_seen <= self.max_age}
        self.seen = {uid: at for uid, at in self.seen.items() if now - at <= 60.0}
        return self.snapshot()

    def snapshot(self) -> list[dict]:
        return [track.snapshot() for track in self.tracks.values()]


class PersistentMap:
    """Sparse log-odds occupancy plus observed elevation/coverage, in ENU.

    ``update(points, origin, timestamp=0)`` marks occupied sonar endpoints and
    free ray cells. Unseen cells are absent (unknown), never initialized from
    ground truth. ``snapshot`` limits visualization points, retaining all cells
    internally. ``evaluate`` compares observed elevation to a truth callback.
    """
    def __init__(self, resolution: float = 1.0, max_cells: int = 60000,
                 bounds=(-30.0, 30.0, -30.0, 30.0)):
        if resolution <= 0:
            raise ValueError("Map resolution must be positive")
        self.resolution, self.max_cells = resolution, max_cells
        self.bounds = tuple(float(v) for v in bounds)  # x_min, x_max, y_min, y_max of the coverage region
        self.cells: dict[tuple, dict] = {}
        self.elevation: dict[tuple, dict] = {}

    def _key(self, point):
        return tuple(np.floor(vec3(point) / self.resolution).astype(int).tolist())

    def _touch(self, key, increment: float, timestamp: float):
        if key not in self.cells and len(self.cells) >= self.max_cells:
            return
        cell = self.cells.setdefault(key, {"log_odds": 0.0, "samples": 0, "last_seen": 0.0})
        cell["log_odds"] = float(np.clip(cell["log_odds"] + increment, -4.0, 4.0))
        cell["samples"] += 1
        cell["last_seen"] = timestamp

    def update(self, points, origin, timestamp: float = 0.0):
        origin = vec3(origin)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        for point in points:
            if not np.all(np.isfinite(point)):
                continue
            distance = float(np.linalg.norm(point - origin))
            count = max(2, int(distance / (self.resolution * 0.8)))
            end_key = self._key(point)
            ray_keys = {self._key(origin + (point - origin) * fraction)
                        for fraction in np.linspace(0.0, 1.0, count, endpoint=False)}
            for key in ray_keys - {end_key}:
                self._touch(key, -0.28, timestamp)
            self._touch(end_key, 0.85, timestamp)
            horizontal_key = end_key[:2]
            entry = self.elevation.setdefault(horizontal_key,
                                              {"z": float(point[2]), "n": 0, "m2": 0.0, "last_seen": 0.0})
            entry["n"] += 1
            difference = float(point[2]) - entry["z"]
            entry["z"] += difference / entry["n"]
            entry["m2"] += difference * (float(point[2]) - entry["z"])
            entry["last_seen"] = timestamp

    observe = update

    def snapshot(self, max_points: int = 1800) -> dict:
        occupied = [(key, value) for key, value in self.cells.items() if value["log_odds"] > 0.7]
        stride = max(1, math.ceil(len(occupied) / max(max_points, 1)))
        points = []
        for key, value in occupied[::stride]:
            center = (np.array(key) + 0.5) * self.resolution
            probability = 1.0 / (1.0 + math.exp(-value["log_odds"]))
            points.append({"position": center.tolist(), "confidence": round(probability, 3),
                           "samples": value["samples"]})
        elevation = [{"x": (key[0] + 0.5) * self.resolution,
                      "y": (key[1] + 0.5) * self.resolution, "z": round(value["z"], 3),
                      "samples": value["n"], "std_m": round(math.sqrt(value["m2"] / max(value["n"] - 1, 1)), 3)}
                     for key, value in list(self.elevation.items())[::max(1, math.ceil(len(self.elevation) / 1800))]]
        coverage = self.evaluate(bounds=self.bounds)["coverage_fraction"]
        return {"resolution_m": self.resolution, "observed_voxels": len(self.cells),
                "observed_cells": len(self.elevation), "coverage": coverage,
                "coverage_fraction": coverage,
                "occupied_voxels": len(occupied), "observed_area_m2": len(self.elevation) * self.resolution ** 2,
                "points": [point["position"] for point in points], "occupied_cells": points,
                "elevation": elevation,
                "unknown_policy": "Absent cells are unobserved; points contain measured returns only."}

    def evaluate(self, truth_height=seabed_height, bounds=None) -> dict:
        bounds = self.bounds if bounds is None else bounds
        errors = []
        for key, value in self.elevation.items():
            x, y = (key[0] + 0.5) * self.resolution, (key[1] + 0.5) * self.resolution
            errors.append(value["z"] - truth_height(x, y))
        area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
        observed_in_bounds = sum(1 for key in self.elevation
                                 if bounds[0] <= (key[0] + 0.5) * self.resolution < bounds[1]
                                 and bounds[2] <= (key[1] + 0.5) * self.resolution < bounds[3])
        return {"mapping_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                "coverage_fraction": min(1.0, observed_in_bounds * self.resolution ** 2 / max(area, 1)),
                "elevation_samples": len(errors)}


def closest_approach(relative_position, relative_velocity, horizon: float = 20.0) -> tuple[float, float]:
    position, velocity = vec3(relative_position), vec3(relative_velocity)
    speed_squared = float(velocity @ velocity)
    time = float(np.clip(-(position @ velocity) / speed_squared, 0, horizon)) if speed_squared > 1e-9 else 0.0
    return float(np.linalg.norm(position + time * velocity)), time


def assess_alerts(tracks: Iterable, diver_position, diver_velocity=None,
                  protected_radius: float = 4.0, horizon: float = 40.0) -> list[dict]:
    """Classify measurable approach risk separately from object identity.

    A vehicle is not inherently hostile. Confidence, position uncertainty,
    predicted closest approach and speed determine review/proximity alerts.
    """
    diver = vec3(diver_position)
    diver_v = np.zeros(3) if diver_velocity is None else vec3(diver_velocity)
    alerts = []
    for item in tracks:
        track = item.snapshot() if isinstance(item, Track) else item
        label = track.get("classification", "unknown")
        if label == "diver":
            continue
        relative = vec3(track["position"]) - diver
        relative_v = vec3(track.get("velocity", [0, 0, 0])) - diver_v
        distance = float(np.linalg.norm(relative))
        cpa, time_to_cpa = closest_approach(relative, relative_v, horizon)
        std = float(track.get("position_std_m", 0.5))
        lower_distance = max(0.0, distance - 2 * std)
        lower_cpa = max(0.0, cpa - 2 * std)
        speed = float(np.linalg.norm(relative_v))
        approaching = float(relative @ relative_v) < -0.05
        eta = time_to_enter(relative, relative_v, protected_radius + 2 * std)
        if eta is not None and eta > horizon:
            eta = None
        if distance > 30 and (not approaching or lower_cpa > protected_radius + 2):
            continue
        severity, reason = "info", "Object observed outside the protected zone"
        if lower_distance < protected_radius:
            severity, reason = "critical", "Position uncertainty overlaps the protected zone"
        elif approaching and lower_cpa < protected_radius and speed > 0.15:
            severity, reason = "warning", "Predicted trajectory enters the protected zone"
            if eta is not None:
                reason = f"Predicted to reach the protected zone in {eta:.0f} s"
        elif lower_distance < protected_radius + 4 or label == "unknown":
            severity, reason = "watch", "Close or uncertain observation requires human review"
        categories = {"vehicle": "approaching_vehicle", "marine_life": "marine_life",
                      "obstacle": "environmental_hazard", "unknown": "unknown_object"}
        level = {"critical": "danger", "warning": "warning", "watch": "warning", "info": "info"}[severity]
        if isinstance(item, dict):
            item["risk"] = level
        alerts.append({"id": track["id"], "track_id": track["id"], "classification": label,
                       "level": level, "message": reason, "time": track.get("last_seen", 0),
                       "classification_confidence": track.get("confidence", 0.0),
                       "category": categories.get(label, "unknown_object"), "severity": severity,
                       "reason": reason, "distance_m": round(distance, 2),
                       "closest_approach_m": round(cpa, 2), "time_to_closest_s": round(time_to_cpa, 2),
                       "time_to_zone_s": None if eta is None else round(eta, 1),
                       "warning_stage": ("inside" if eta == 0 else "imminent" if eta is not None and eta <= 10
                                         else "early" if eta is not None else None),
                       "deterrable": label == "marine_life",
                       "position_std_m": round(std, 3), "relative_speed_m_s": round(speed, 3),
                       "requires_review": severity in ("watch", "warning", "critical"),
                       "hostility_determined": False})
    rank = {"critical": 0, "warning": 1, "watch": 2, "info": 3}
    return sorted(alerts, key=lambda item: (rank[item["severity"]], item["distance_m"]))


def navigation_alerts(tracks: Iterable, position, velocity=None) -> list[dict]:
    """Report measured approach risk around an exploring vehicle."""
    alerts = assess_alerts(tracks, position, velocity, protected_radius=2.4, horizon=12.0)
    for alert in alerts:
        alert["scope"] = "vehicle_navigation"
        alert["reason"] = alert["reason"].replace("the protected zone", "vehicle clearance")
        alert["message"] = alert["reason"]
    return alerts


class AcousticTDMA:
    """A deterministic packet-level acoustic TDMA channel emulator.

    ``enqueue(source, payload, timestamp) -> bool`` enforces the slot byte cap.
    ``step(now)`` advances slots and returns newly delivered packet dictionaries.
    ``set_connected(source, False/True)`` exercises outages/reconnection. This
    simulator models packet delay/loss and capacity, not acoustic propagation
    waveforms or modem hardware. Payloads never contain raw audio.
    """
    def __init__(self, node_ids=("auv_0", "auv_1", "auv_2"), seed: int = 7,
                 slot_duration: float = 0.4, bitrate: float = 9600,
                 loss_probability: float = 0.12, delay_range=(0.15, 0.8),
                 ttl: float = 4.0, max_queue: int = 24, duplicate_probability: float = 0.02):
        if not node_ids or slot_duration <= 0 or bitrate <= 0:
            raise ValueError("TDMA needs nodes and positive slot duration/bitrate")
        self.node_ids = tuple(node_ids)
        self.rng = np.random.default_rng(seed)
        self.slot_duration, self.bitrate = slot_duration, bitrate
        self.loss_probability, self.delay_range = loss_probability, delay_range
        self.ttl, self.max_queue, self.duplicate_probability = ttl, max_queue, duplicate_probability
        self.max_packet_bytes = max(1, int(bitrate * slot_duration / 8) - 24)
        self.queues = {source: deque() for source in self.node_ids}
        self.connected = {source: True for source in self.node_ids}
        self.sequences = defaultdict(int)
        self.inflight = []
        self.event_counter = 0
        self.next_slot = 0
        self.seen = set()
        self.stats = defaultdict(int)
        self.latencies = deque(maxlen=500)
        self.last_delivery = {source: None for source in self.node_ids}
        self.now = 0.0

    def enqueue(self, source: str, payload: Mapping, timestamp: float) -> bool:
        if source not in self.queues:
            raise ValueError(f"Unregistered acoustic source: {source}")
        def forbidden(value):
            if isinstance(value, Mapping):
                return any(str(key).lower() in {"raw_audio", "audio", "samples", "waveform",
                                               "hydrophone_samples", "audio_samples", "raw_waveform",
                                               "spectrogram"} or forbidden(child)
                           for key, child in value.items())
            if isinstance(value, (list, tuple)):
                return any(forbidden(child) for child in value)
            return False
        if forbidden(payload):
            self.stats["raw_audio_rejected"] += 1
            return False
        self.sequences[source] += 1
        packet = {"s": source, "q": self.sequences[source], "t": round(timestamp, 3), "d": dict(payload)}
        encoded = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) > self.max_packet_bytes:
            self.stats["oversize_rejected"] += 1
            return False
        if len(self.queues[source]) >= self.max_queue:
            self.queues[source].popleft()
            self.stats["queue_overflow"] += 1
        self.queues[source].append((packet, len(encoded)))
        self.stats["enqueued"] += 1
        return True

    def set_connected(self, source: str, connected: bool):
        if source not in self.connected:
            raise ValueError(f"Unregistered acoustic source: {source}")
        if connected and not self.connected[source]:
            self.stats["reconnections"] += 1
            # Telemetry is a state update, not a command requiring replay. An
            # outage backlog can otherwise consume every restored slot with
            # reports that expire in flight, delaying recovery indefinitely.
            queue = self.queues[source]
            if len(queue) > 1:
                newest = max(queue, key=lambda item: item[0]["t"])
                self.stats["superseded_on_reconnect"] += len(queue) - 1
                queue.clear()
                queue.append(newest)
        if not connected and self.connected[source]:
            self.stats["outages"] += 1
        self.connected[source] = bool(connected)

    def step(self, now: float) -> list[dict]:
        if now < self.now:
            raise ValueError("Network simulation time must be monotonic")
        self.now = now
        while self.next_slot * self.slot_duration <= now + 1e-9:
            slot_time = self.next_slot * self.slot_duration
            source = self.node_ids[self.next_slot % len(self.node_ids)]
            self.next_slot += 1
            queue = self.queues[source]
            while queue and slot_time - queue[0][0]["t"] > self.ttl:
                queue.popleft()
                self.stats["expired"] += 1
            if not self.connected[source] or not queue or queue[0][0]["t"] > slot_time:
                continue
            packet, size = queue.popleft()
            self.stats["transmitted"] += 1
            self.stats["bytes_transmitted"] += size
            if self.rng.random() < self.loss_probability:
                self.stats["lost"] += 1
                continue
            airtime = (size + 24) * 8 / self.bitrate
            delivery = slot_time + airtime + self.rng.uniform(*self.delay_range)
            self.event_counter += 1
            heapq.heappush(self.inflight, (delivery, self.event_counter, slot_time, packet))
            if self.rng.random() < self.duplicate_probability:
                self.event_counter += 1
                heapq.heappush(self.inflight, (delivery + 0.05, self.event_counter, slot_time, packet))
                self.stats["duplicate_injected"] += 1
        delivered = []
        while self.inflight and self.inflight[0][0] <= now:
            at, _, sent_at, packet = heapq.heappop(self.inflight)
            key = (packet["s"], packet["q"])
            if key in self.seen:
                self.stats["duplicate_discarded"] += 1
                continue
            self.seen.add(key)
            if at - packet["t"] > self.ttl:
                self.stats["expired_inflight"] += 1
                continue
            latency = at - packet["t"]
            self.latencies.append(latency)
            self.stats["delivered"] += 1
            self.last_delivery[packet["s"]] = at
            delivered.append({"source": packet["s"], "sequence": packet["q"],
                              "sent_at": sent_at, "received_at": at,
                              "latency": latency, "payload": packet["d"]})
        return delivered

    def snapshot(self) -> dict:
        stats = {name: int(self.stats[name]) for name in ("enqueued", "transmitted", "delivered", "lost",
                 "expired", "expired_inflight", "queue_overflow", "oversize_rejected", "raw_audio_rejected",
                 "outages", "reconnections", "superseded_on_reconnect", "duplicate_injected", "duplicate_discarded", "bytes_transmitted")}
        stats.update({"protocol": "acoustic_tdma", "bitrate_bps": self.bitrate,
                      "slot_duration_s": self.slot_duration,
                      "max_packet_bytes": self.max_packet_bytes,
                      "loss_probability": self.loss_probability,
                      "loss_rate": self.stats["lost"] / max(self.stats["transmitted"], 1),
                      "queued": sum(len(queue) for queue in self.queues.values()),
                      "delivery_ratio": self.stats["delivered"] / max(self.stats["transmitted"], 1),
                      "mean_latency_s": float(np.mean(self.latencies)) if self.latencies else 0.0,
                      "queue_depth": {source: len(queue) for source, queue in self.queues.items()},
                      "connected": dict(self.connected), "last_delivery": dict(self.last_delivery)})
        return stats


class EdgeAgent:
    """One independent node's sensing, tracking and decentralized control.

    ``step(timestamp, dt, position, velocity, diver_position, diver_velocity,
    objects=None, peer_telemetry=(), observations=None)`` uses estimated pose.
    If ``observations`` is supplied, no truth sensor simulation occurs; ROS edge
    nodes can subscribe only to noisy navigation, observations and mesh output.
    Peer telemetry is a delivered ``AcousticTDMA`` packet or its payload with
    ``source`` supplied. Return includes command_velocity and compact telemetry.
    """
    def __init__(self, index: int, seed: int = 7, mission_mode: str = "protect",
                 mission_config: Optional[Mapping] = None):
        self.index, self.name = index, f"auv_{index}"
        self.sensor = SensorModel(seed + index * 997)
        self.controller = FormationController(index)
        config = dict(mission_config or {})
        config["mission_mode"] = mission_mode
        self.planner = MissionPlanner(index, config)
        self.tracker = MultiObjectTracker(prefix=f"{self.name}-track")
        self.map = PersistentMap()
        self.speaker = DeterrentSpeaker(config.get("deterrent"))
        self.peers = {}
        self.pending_observations = deque(maxlen=30)
        self.last_telemetry = -math.inf
        self.last_result = {}

    def set_mode(self, mode: str):
        """Change autonomy mission without resetting tracks, map or vehicle pose."""
        self.planner.set_mode(mode)

    def step(self, timestamp: float, dt: float, position, velocity, diver_position,
             diver_velocity, objects=None, peer_telemetry=(), observations=None,
             rays=(), diver_timestamp=None, uncertainty: float = 0.0) -> dict:
        position, velocity = vec3(position), vec3(velocity)
        remote_observations = []
        for packet in peer_telemetry:
            source = packet.get("source", "")
            payload = packet.get("payload", packet)
            source = source or payload.get("source", "")
            if source == self.name:
                continue
            if source and "p" in payload:
                self.peers[source] = {"position": vec3(payload["p"]),
                                      "velocity": vec3(payload.get("v", [0, 0, 0])),
                                      "timestamp": float(payload.get("t", timestamp)),
                                      "mission": payload.get("mission", payload.get("m", {}))}
            for item in payload.get("o", []):
                try:
                    remote_observations.append(Observation.from_payload(item))
                except (ValueError, KeyError, TypeError):
                    continue
        if self.planner.mode == "protect":
            relative = position - vec3(diver_position)
            yaw = math.atan2(relative[1], relative[0])
        else:
            yaw = float(self.last_result.get("yaw", self.index * 2 * math.pi / 3))
        if observations is None:
            observations = self.sensor.observe(self.name, position, yaw, objects or [], timestamp)
        normalized = [Observation.from_payload(o) if isinstance(o, Mapping) else o for o in observations]
        self.tracker.update(normalized + remote_observations, timestamp)
        for observation in normalized:
            self.pending_observations.append(observation.to_payload())
        peers, mission_peers = [], []
        for source in list(self.peers):
            peer = self.peers[source]
            age = timestamp - peer["timestamp"]
            if age > 4.0 or age < -0.1:
                del self.peers[source]
                continue
            peers.append((peer["position"] + peer["velocity"] * max(age, 0), peer["velocity"]))
            mission_peers.append({"source": source, "payload": {
                "p": peer["position"].tolist(), "v": peer["velocity"].tolist(),
                "t": peer["timestamp"], "m": peer["mission"]}})
        rays = np.asarray(rays, dtype=float).reshape(-1, 3)
        if len(rays):
            self.map.update(rays, position, timestamp)
        tracks = self.tracker.snapshot()
        plan = self.planner.plan(timestamp, position,
                                 diver={"position": diver_position, "velocity": diver_velocity,
                                        "timestamp": timestamp if diver_timestamp is None else diver_timestamp},
                                 peers=mission_peers, observations=tracks, rays=rays,
                                 uncertainty=uncertainty)
        obstacles = [(track.position,
                      (2.0 if track.classification == "marine_life" else 0.8)
                      + min(0.7, math.sqrt(np.trace(track.covariance[:3, :3]))),
                      track.velocity)
                     for track in self.tracker.tracks.values()
                     if track.classification != "diver"]
        goal_velocity = None
        if self.planner.mode == "protect" and plan.status.get("phase") in ("escort", "predictive_watch"):
            # Follow the slot's full motion: the diver's drift plus the ring's
            # rotation toward a threat (feedforward; feedback still corrects).
            goal_velocity = vec3(diver_velocity) + vec3(plan.status.get("ring_velocity", [0, 0, 0]))
        command = self.controller.command(position, velocity, diver_position, diver_velocity,
                                          peers=peers, obstacles=obstacles, dt=dt,
                                          navigation_goal=plan.goal,
                                          protect_diver=self.planner.mode == "protect",
                                          goal_velocity=goal_velocity)
        # Navigation heartbeat plus one observation fits the 456-byte default cap.
        deterrent = (self.speaker.decide(timestamp, position, diver_position, diver_velocity, tracks)
                     if self.planner.mode == "protect" else self.speaker._idle("survey_mode"))
        telemetry = {"t": round(timestamp, 3), "p": np.round(position, 2).tolist(),
                     "v": np.round(velocity, 2).tolist(), "r": 0.04, "o": [],
                     "m": {"mode": self.planner.mode, "target": np.round(plan.goal, 1).tolist()}}
        if deterrent["active"]:
            telemetry["dt"] = [int(round(deterrent["level_db"])), deterrent["target_class"]]
        if self.pending_observations:
            telemetry["o"].append(self.pending_observations.popleft())
        mission = dict(plan.status)
        mission.update({"target": plan.goal.tolist(), "anchor": plan.anchor.tolist(),
                        "heading": float(plan.heading)})
        alerts = (assess_alerts(self.tracker.tracks.values(), diver_position, diver_velocity)
                  if self.planner.mode == "protect" else
                  navigation_alerts(self.tracker.tracks.values(), position, velocity))
        result = {"name": self.name, "command_velocity": command.tolist(), "yaw": float(plan.heading),
                  "target": plan.goal.tolist(), "mission": mission,
                  "observations": [o.to_payload() for o in normalized],
                  "tracks": tracks, "telemetry": telemetry,
                  "peer_count": len(peers), "deterrent": deterrent,
                  "alerts": alerts}
        self.last_result = result
        return result


class SimulationCore:
    """Offline harness and bridge-compatible orchestration of modular algorithms.

    ``set_truth(states)`` accepts actual Gazebo model positions/velocities.
    ``step(dt, model_states=states)`` consumes those poses and returns commanded
    velocities without integrating vehicle poses. With no external truth set,
    it runs the deterministic offline harness for repeatable algorithm tests.
    For deployment, instantiate separate EdgeAgents in separate ROS processes.
    """
    def __init__(self, seed: int = 7, config: Optional[Mapping] = None,
                 mode: Optional[str] = None):
        config = dict(config or {})
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.seed = seed
        mission_mode = mode or config.get("mission_mode", "protect")
        self.agents = [EdgeAgent(i, seed, mission_mode=mission_mode, mission_config=config)
                       for i in range(3)]
        self.mode = self.agents[0].planner.mode
        self.network = AcousticTDMA(seed=seed + 19,
                                   loss_probability=float(config.get("packet_loss", 0.12)))
        self.tracker = MultiObjectTracker(prefix="global")
        region = config.get("exploration", {}).get("bounds", [[-30, 30], [-30, 30]])
        self.map = PersistentMap(resolution=float(config.get("map_resolution", 1)),
                                 bounds=(*region[0], *region[1]))
        self.objects = {
            "diver": WorldObject("diver", "diver", [0, 0, -7], [0.10, 0, 0], 0.65),
            "intruder": WorldObject("intruder", "vehicle", [24, -1.5, -7], [-0.50, 0.04, 0], 1.1),
            "marine_life": WorldObject("marine_life", "marine_life", [-12, 9, -8], [0.20, -0.04, 0], 0.5),
            "large_marine_life": WorldObject("large_marine_life", "marine_life", [12, 0, -10.5],
                                             [0, 7 / 22, 0], 0.85),
            "rock_hazard": WorldObject("rock_hazard", "obstacle", [8, 8, -9], [0, 0, 0], 1.4),
            "unknown_object": WorldObject("unknown_object", "unknown", [-15, -8, -8], [0.13, 0.07, 0], 0.8),
        }
        self.auvs = {}
        for agent in self.agents:
            self.auvs[agent.name] = {"position": agent.controller.target(self.objects["diver"].position),
                                     "velocity": np.zeros(3), "yaw": agent.index * 2 * math.pi / 3}
        self.external_truth = False
        self.wildlife = WildlifeResponse()
        self.flee_offset = defaultdict(lambda: np.zeros(3))  # displacement from each animal's nominal route
        self.speakers = {}
        self.deterrent_active_s = 0.0
        self.min_wildlife_diver_distance = math.inf
        self.zone_breach_s = 0.0
        self.first_warned = {}
        self.first_entry = {}
        self.drift = {agent.name: np.zeros(3) for agent in self.agents}
        self.mailboxes = {agent.name: [] for agent in self.agents}
        self.last_sense = -math.inf
        self.last_mapping = -math.inf
        self.last_broadcast = -math.inf
        self.commands = {}
        self.observation_error_sq = deque(maxlen=30000)
        self.localization_error_sq = deque(maxlen=30000)
        self.classification_count = 0
        self.classification_correct = 0
        self.unknown_count = 0
        self.minimum_separation = math.inf
        self.minimum_wildlife_clearance = math.inf
        self.minimum_seabed_clearance = math.inf
        self.formation_errors = deque(maxlen=1000)
        self.mission_errors = deque(maxlen=1000)
        self.last_snapshot = {}

    def set_mode(self, mode: str):
        """Switch missions in place; keep sensing, communication and map history."""
        for agent in self.agents:
            agent.set_mode(mode)
        new_mode = self.agents[0].planner.mode
        if new_mode != self.mode:
            self.formation_errors.clear()
            self.mission_errors.clear()
            for agent in self.agents:
                agent.last_result = {}
            self.commands.clear()
        self.mode = new_mode

    def set_truth(self, states: Mapping):
        """Install actual simulator poses, permanently disabling offline motion."""
        self.external_truth = True
        for name, value in states.items():
            if name in self.auvs:
                self.auvs[name]["position"] = vec3(value["position"])
                self.auvs[name]["velocity"] = vec3(value.get("velocity", [0, 0, 0]))
                self.auvs[name]["yaw"] = float(value.get("yaw", self.auvs[name]["yaw"]))
            elif name in self.objects:
                self.objects[name].position = vec3(value["position"])
                self.objects[name].velocity = vec3(value.get("velocity", [0, 0, 0]))

    def _offline_motion(self, dt: float):
        diver = self.objects["diver"]
        diver.velocity = np.array([0.10, 0.06 * math.cos(self.time * 0.07), 0.0])
        for obj in self.objects.values():
            if obj.kind == "marine_life":
                flee = self.wildlife.velocity(obj.object_id, obj.position, self.speakers.values(), dt)
                offset = self.flee_offset[obj.object_id]
                offset += flee * dt
                offset *= math.exp(-0.04 * dt)  # animals drift back toward their route
                if obj.object_id == "large_marine_life":
                    angle = self.time / 22
                    nominal = np.array([12 * math.cos(angle), 7 * math.sin(angle), -10.5])
                    obj.velocity = np.array([-12 / 22 * math.sin(angle), 7 / 22 * math.cos(angle), 0.0]) + flee
                    obj.position = nominal + offset
                    continue
                obj.position += (obj.velocity + flee) * dt
                continue
            if obj.object_id == "intruder" and obj.position[0] < -14:
                obj.velocity[0] = 0.50
            elif obj.object_id == "intruder" and obj.position[0] > 24:
                obj.velocity[0] = -0.50
            obj.position += obj.velocity * dt
        current = np.array([0.035 * math.sin(self.time * 0.11), 0.025, 0.0])
        for name, state in self.auvs.items():
            command = vec3(self.commands.get(name, [0, 0, 0]))
            # First-order thruster response and a modest ocean current.
            state["velocity"] += (command + current - state["velocity"]) * min(dt / 0.8, 1)
            state["position"] += state["velocity"] * dt
            # Sensor orientation follows the estimate-side mission planner.
            agent = self.agents[int(name.rsplit("_", 1)[1])]
            state["yaw"] = float(agent.last_result.get("yaw", state["yaw"]))

    def step(self, dt: float = 0.1, model_states: Optional[Mapping] = None) -> dict:
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError("Simulation dt must be finite and positive")
        if model_states is not None:
            self.set_truth(model_states)
        remaining = dt
        while remaining > 1e-9:
            interval = min(remaining, 0.2)
            remaining -= interval
            self.time += interval
            if not self.external_truth:
                self._offline_motion(interval)
            received = self.network.step(self.time)
            for packet in received:
                for name in self.mailboxes:
                    if name != packet["source"]:
                        self.mailboxes[name].append(packet)
                self.tracker.update([Observation.from_payload(item) for item in packet["payload"].get("o", [])], self.time)
            sensing = self.time - self.last_sense >= 0.5 - 1e-6
            mapping = self.time - self.last_mapping >= 0.8 - 1e-6
            broadcasting = self.time - self.last_broadcast >= 1.2 - 1e-6
            diver = self.objects["diver"]
            for agent in self.agents:
                state = self.auvs[agent.name]
                self.drift[agent.name] += self.rng.normal(0, 0.001 * math.sqrt(interval), 3)
                nav_position = state["position"] + self.drift[agent.name] + self.rng.normal(0, 0.015, 3)
                self.localization_error_sq.append(float(np.sum((nav_position - state["position"]) ** 2)))
                observations = agent.sensor.observe(agent.name, state["position"], state["yaw"],
                                                     self.objects.values(), self.time) if sensing else []
                for observation in observations:
                    truth = self.objects[observation.truth_id]
                    self.observation_error_sq.append(float(np.sum((observation.position - truth.position) ** 2)))
                    self.classification_count += 1
                    self.classification_correct += observation.classification == truth.kind
                    self.unknown_count += observation.classification == "unknown"
                mapped = np.empty((0, 3))
                if mapping:
                    points = agent.sensor.sample_environment(state["position"], state["yaw"], self.time)
                    mapped = points + nav_position - state["position"]
                    self.map.update(mapped, nav_position, self.time)
                result = agent.step(self.time, interval, nav_position, state["velocity"],
                                    diver.position, diver.velocity, peer_telemetry=self.mailboxes[agent.name],
                                    observations=observations, rays=mapped,
                                    uncertainty=float(np.linalg.norm(self.drift[agent.name]) + 0.03))
                self.mailboxes[agent.name] = []
                self.commands[agent.name] = result["command_velocity"]
                self.speakers[agent.name] = {"position": state["position"].copy(),
                                             "level_db": result["deterrent"]["level_db"] if result["deterrent"]["active"] else 0.0}
                if agent.index == 0:  # Gateway's own edge estimates need no acoustic hop.
                    self.tracker.update(observations, self.time)
                if broadcasting:
                    self.network.enqueue(agent.name, result["telemetry"], self.time)
                error = float(np.linalg.norm(state["position"] - vec3(result["target"])))
                self.mission_errors.append(error)
                if self.mode == "protect":
                    self.formation_errors.append(error)
            positions = [state["position"] for state in self.auvs.values()]
            # Truth-side safety measurement only; none of these distances feed
            # back into the estimate-side planner or velocity controller.
            for position in positions:
                self.minimum_seabed_clearance = min(self.minimum_seabed_clearance,
                    float(position[2] - seabed_height(position[0], position[1]) - 0.4))
                for obj in self.objects.values():
                    if obj.kind == "marine_life":
                        self.minimum_wildlife_clearance = min(self.minimum_wildlife_clearance,
                            float(np.linalg.norm(position - obj.position) - obj.radius - 0.4))
            for i in range(3):
                for j in range(i + 1, 3):
                    self.minimum_separation = min(self.minimum_separation, float(np.linalg.norm(positions[i] - positions[j])))
            self._protection_bookkeeping(interval)
            self.tracker.update([], self.time)
            if sensing:
                self.last_sense = self.time
            if mapping:
                self.last_mapping = self.time
            if broadcasting:
                self.last_broadcast = self.time
        self.last_snapshot = self.snapshot()
        return self.last_snapshot

    def _protection_bookkeeping(self, dt: float):
        """Truth-side measures of how early and how well the diver was protected."""
        if self.mode != "protect":
            return
        diver = self.objects["diver"]
        if any(speaker["level_db"] > 0 for speaker in self.speakers.values()):
            self.deterrent_active_s += dt
        warned = {a["track_id"] for a in assess_alerts(self.tracker.tracks.values(), diver.position, diver.velocity)
                  if a["severity"] in ("warning", "critical")}
        inside = False
        for obj in self.objects.values():
            if obj.kind in ("diver",):
                continue
            distance = float(np.linalg.norm(obj.position - diver.position))
            if obj.kind in ("marine_life", "vehicle", "unknown"):
                self.min_wildlife_diver_distance = min(self.min_wildlife_diver_distance, distance)
            for track_id in warned:
                track = self.tracker.tracks.get(track_id)
                if track is not None and np.linalg.norm(track.position - obj.position) < 4.0:
                    self.first_warned.setdefault(obj.object_id, self.time)
            if distance < 4.0 and obj.kind != "obstacle":
                inside = True
                self.first_entry.setdefault(obj.object_id, self.time)
        if inside:
            self.zone_breach_s += dt

    def model_states(self) -> dict:
        states = {name: {"position": state["position"].tolist(), "velocity": state["velocity"].tolist(),
                         "yaw": state["yaw"]} for name, state in self.auvs.items()}
        states.update({name: {"position": obj.position.tolist(), "velocity": obj.velocity.tolist(),
                              "yaw": math.atan2(obj.velocity[1], obj.velocity[0])}
                       for name, obj in self.objects.items()})
        return states

    def evaluate(self) -> dict:
        metrics = self.map.evaluate()
        errors = []
        correct_tracks, matches = 0, 0
        truth_objects = list(self.objects.values())
        for track in self.tracker.tracks.values():
            truth = min(truth_objects, key=lambda obj: np.linalg.norm(track.position - obj.position))
            error = float(np.linalg.norm(track.position - truth.position))
            if error < 4.0:
                errors.append(error ** 2)
                matches += 1
                correct_tracks += track.classification == truth.kind
        metrics.update({"elapsed_s": round(self.time, 3), "seed": self.seed,
                        "localization_rmse_m": float(np.sqrt(np.mean(self.localization_error_sq))) if self.localization_error_sq else None,
                        "sonar_position_rmse_m": float(np.sqrt(np.mean(self.observation_error_sq))) if self.observation_error_sq else None,
                        "tracking_rmse_m": float(np.sqrt(np.mean(errors))) if errors else None,
                        "classification_accuracy": self.classification_correct / max(self.classification_count, 1),
                        "matched_track_classification_accuracy": correct_tracks / max(matches, 1),
                        "sensor_observations": self.classification_count,
                        "unknown_observations": self.unknown_count,
                        "mission_mode": self.mode,
                        "deterrent_activations": sum(agent.speaker.activations for agent in self.agents),
                        "deterrent_active_s": round(self.deterrent_active_s, 1),
                        "min_animal_diver_distance_m": (self.min_wildlife_diver_distance
                                                        if math.isfinite(self.min_wildlife_diver_distance) else None),
                        "protected_zone_breach_s": round(self.zone_breach_s, 1),
                        "warning_lead_times_s": {name: round(self.first_entry[name] - at, 1)
                                                 for name, at in self.first_warned.items() if name in self.first_entry},
                        "formation_rmse_m": float(np.sqrt(np.mean(np.square(self.formation_errors)))) if self.mode == "protect" and self.formation_errors else None,
                        "mission_target_rmse_m": float(np.sqrt(np.mean(np.square(self.mission_errors)))) if self.mission_errors else None,
                        "exploration_coverage_pct": self.mission_status()["coverage_pct"],
                        "minimum_auv_separation_m": self.minimum_separation if math.isfinite(self.minimum_separation) else None,
                        "minimum_wildlife_clearance_m": self.minimum_wildlife_clearance if math.isfinite(self.minimum_wildlife_clearance) else None,
                        "minimum_seabed_clearance_m": self.minimum_seabed_clearance if math.isfinite(self.minimum_seabed_clearance) else None,
                        "packet_delivery_ratio": self.network.stats["delivered"] / max(self.network.stats["transmitted"], 1),
                        "classifier": "synthetic feature/template baseline; no trained AI",
                        "motion_authority": "external Gazebo poses" if self.external_truth else "offline numerical harness"})
        return metrics

    def mission_status(self) -> dict:
        """Fleet summary; coverage uses the union of actually observed cells."""
        plans = [agent.last_result.get("mission", {}) for agent in self.agents]
        cells = set().union(*(agent.planner.observed_cells for agent in self.agents))
        total = max((int(plan.get("total_cells", 0)) for plan in plans), default=0)
        measured = len(cells)
        coverage = min(1.0, measured / total) if total else 0.0
        phases = {plan.get("phase", "ready") for plan in plans}
        state = phases.pop() if len(phases) == 1 else "running" if self.time else "ready"
        return {"mode": self.mode, "state": state,
                "strategy": "predictive escort" if self.mode == "protect" else "distributed survey",
                "observed_cells": measured, "total_cells": total,
                "coverage": coverage, "coverage_pct": round(coverage * 100, 2),
                "waypoints_completed": sum(int(plan.get("waypoints_completed", 0)) for plan in plans),
                "units": {agent.name: plan for agent, plan in zip(self.agents, plans)}}

    def snapshot(self) -> dict:
        diver = self.objects["diver"]
        units = []
        for agent in self.agents:
            state = self.auvs[agent.name]
            units.append({"id": agent.name, "position": np.round(state["position"], 3).tolist(),
                          "velocity": np.round(state["velocity"], 3).tolist(), "yaw": state["yaw"],
                          "command_velocity": self.commands.get(agent.name, [0, 0, 0]),
                          "target": agent.last_result.get("target", state["position"].tolist()),
                          "mission": agent.last_result.get("mission", {"mode": self.mode, "state": "ready"}),
                          "localization_std_m": float(np.linalg.norm(self.drift[agent.name]) + 0.03),
                          "peer_count": agent.last_result.get("peer_count", 0), "gateway": agent.index == 0,
                          "deterrent": agent.last_result.get("deterrent", {"active": False, "level_db": 0.0})})
        alerts = assess_alerts(self.tracker.tracks.values(), diver.position, diver.velocity)
        if self.mode == "explore":
            by_track = {}
            rank = {"critical": 0, "warning": 1, "watch": 2, "info": 3}
            for state in self.auvs.values():
                for alert in navigation_alerts(self.tracker.tracks.values(), state["position"], state["velocity"]):
                    previous = by_track.get(alert["track_id"])
                    if previous is None or (rank[alert["severity"]], alert["distance_m"]) < (rank[previous["severity"]], previous["distance_m"]):
                        by_track[alert["track_id"]] = alert
            alerts = sorted(by_track.values(), key=lambda alert: (rank[alert["severity"]], alert["distance_m"]))
        return {"time": round(self.time, 3), "frame": "map/ENU", "auvs": units,
                "mission": self.mission_status(),
                "diver": {"position": diver.position.tolist(), "velocity": diver.velocity.tolist(),
                           "protected_radius_m": 4.0, "formation_radius_m": 6.0},
                "objects": [{"id": obj.object_id, "classification": obj.kind,
                             "position": obj.position.tolist(), "velocity": obj.velocity.tolist()}
                            for obj in self.objects.values() if obj.kind != "diver"],
                "tracks": self.tracker.snapshot(),
                "alerts": alerts,
                "map": self.map.snapshot(), "network": self.network.snapshot(), "metrics": self.evaluate(),
                "limitations": ["Synthetic sonar and acoustic feature sensors", "Untrained classification baseline",
                                "Packet-level acoustic emulator; no waveform propagation"]}
