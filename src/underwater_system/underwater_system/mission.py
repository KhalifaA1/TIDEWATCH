"""Mission goals from navigation estimates, sonar hits and acoustic messages.

Escort slots turn toward approaching contacts. Survey vehicles choose frontiers
in separate sectors and return before their time or travel budget runs out.
Positions are ENU metres, time seconds, and headings radians. The velocity
controller handles actual movement and collision avoidance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


_MODES = {
    "protect": "protect", "diver_escort": "protect", "escort": "protect",
    "explore": "explore", "autonomous_exploration": "explore", "exploration": "explore",
}


def canonical_mode(mode: str) -> str:
    """Accept the dashboard names and older escort/exploration names."""
    try:
        return _MODES[str(mode).lower()]
    except KeyError as error:
        raise ValueError("Mission mode must be 'protect' or 'explore'") from error


def _vector(value) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(result)):
        raise ValueError("Mission vectors must have three finite components")
    return result.copy()


def _positive(value, name, *, zero=False) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not zero and result == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return result


def _angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class MissionPlan:
    """A finite goal and scan heading, plus JSON-compatible mission telemetry."""
    goal: np.ndarray
    anchor: np.ndarray
    heading: float
    status: dict


class MissionPlanner:
    """Plan for one vehicle using the full mission configuration.

    Diver beacons contain position, velocity and timestamp; uncertainty is the
    navigation standard deviation in metres. Peers provide delivered telemetry,
    observations provide estimated contacts, and rays contain measured world
    endpoints. Coverage counts measured XY cells, independent of flight depth.
    """

    def __init__(self, agent_index: int, config: Mapping | None = None):
        config = dict(config or {})
        mission = config.get("mission", {})
        if not isinstance(mission, Mapping):
            raise ValueError("mission configuration must be a mapping")
        exploration = config.get("exploration", mission.get("exploration", {}))
        protection = config.get("protection", mission.get("protection", {}))
        if not isinstance(exploration, Mapping) or not isinstance(protection, Mapping):
            raise ValueError("exploration and protection must be mappings")
        self.index = int(agent_index)
        self.fleet_size = int(config.get("fleet_size", 3))
        if self.index != agent_index or self.fleet_size < 1 or not 0 <= self.index < self.fleet_size:
            raise ValueError("Agent index must belong to the configured fleet")
        self.mode = canonical_mode(config.get("mission_mode", mission.get("mode", "protect")))
        self.radius = _positive(config.get("formation_radius", 6), "formation_radius")
        self.protected_radius = _positive(config.get("protected_radius", 4), "protected_radius")
        self.speed_limit = _positive(config.get("speed_limit", 1.3), "speed_limit")
        self.sensor_range = _positive(config.get("sensor_range", 28), "sensor_range")
        self.diver_clearance = _positive(protection.get("diver_clearance", 2.5), "diver_clearance")
        self.contact_clearance = _positive(protection.get("contact_clearance", 3.0), "contact_clearance")
        if self.radius <= max(self.diver_clearance, self.protected_radius):
            raise ValueError("formation_radius must exceed protected radius and diver clearance")
        self.prediction_horizon = _positive(protection.get("prediction_horizon", 10), "prediction_horizon")
        self.beacon_timeout = _positive(protection.get("beacon_timeout", 4), "beacon_timeout")
        self.rotation_rate = _positive(protection.get("rotation_rate", 0.12), "rotation_rate")
        self.contact_timeout = _positive(protection.get("contact_timeout", 4), "contact_timeout")
        self.bounds = np.asarray(exploration.get("bounds", [[-24, 24], [-24, 24]]), dtype=float)
        if (self.bounds.shape != (2, 2) or not np.all(np.isfinite(self.bounds))
                or np.any(self.bounds[:, 1] <= self.bounds[:, 0])):
            raise ValueError("Exploration bounds must be two increasing finite [min,max] pairs")
        self.origin = _vector(exploration.get("origin", [0, 0, -10]))
        if np.any(self.origin[:2] <= self.bounds[:, 0]) or np.any(self.origin[:2] >= self.bounds[:, 1]):
            raise ValueError("Exploration origin must be inside the survey bounds")
        self.depth_bounds = np.asarray(exploration.get("depth_bounds", [-16, -3]), dtype=float)
        if (self.depth_bounds.shape != (2,) or not np.all(np.isfinite(self.depth_bounds))
                or self.depth_bounds[0] >= self.depth_bounds[1] or self.depth_bounds[1] >= 0):
            raise ValueError("depth_bounds must be increasing underwater ENU depths")
        self.depth_layers = np.asarray(exploration.get("depth_layers", [self.origin[2]]), dtype=float)
        if (self.depth_layers.ndim != 1 or not 1 <= len(self.depth_layers) <= 16
                or not np.all(np.isfinite(self.depth_layers))
                or np.any(self.depth_layers < self.depth_bounds[0])
                or np.any(self.depth_layers > self.depth_bounds[1])):
            raise ValueError("depth_layers must be finite flight depths within depth_bounds")
        self.cell_size = _positive(exploration.get("cell_size", 3), "cell_size")
        self.spacing = _positive(exploration.get("waypoint_spacing", 6), "waypoint_spacing")
        if self.cell_size > np.min(np.diff(self.bounds, axis=1)) / 2 or self.spacing < self.cell_size:
            raise ValueError("Survey cells must fit the bounds and waypoint spacing must cover a cell")
        self.max_duration = _positive(exploration.get("max_duration", 240), "max_duration")
        self.distance_budget = _positive(exploration.get("distance_budget", 180), "distance_budget")
        self.uncertainty_limit = _positive(exploration.get("uncertainty_limit", 2.5), "uncertainty_limit")
        self.tolerance = _positive(exploration.get("target_tolerance", 1.5), "target_tolerance")
        self.dwell_time = _positive(exploration.get("dwell_time", 2.5), "dwell_time", zero=True)
        self.minimum_hold = _positive(exploration.get("minimum_target_hold", 5), "minimum_target_hold")
        self.target_timeout = _positive(exploration.get("target_timeout", 25), "target_timeout")
        self.scan_radius = _positive(exploration.get("scan_radius", 6), "scan_radius")
        if self.scan_radius > self.sensor_range:
            raise ValueError("scan_radius cannot exceed the instrument range")
        self.altitude_clearance = _positive(exploration.get("altitude_clearance", 2.2), "altitude_clearance")
        self.minimum_survey_radius = _positive(exploration.get("minimum_survey_radius", 4.5), "minimum_survey_radius")
        self.target_coverage = _positive(exploration.get("target_coverage", 0.9), "target_coverage")
        if self.target_coverage > 1:
            raise ValueError("target_coverage cannot exceed 1")
        self.reserve_ratio = float(exploration.get("reserve_ratio", 0.15))
        if not math.isfinite(self.reserve_ratio) or not 0 <= self.reserve_ratio < 0.8:
            raise ValueError("reserve_ratio must be between 0 and 0.8")
        # This is the survey grid. Terrain enters only through measured hits.
        shape = np.ceil((self.bounds[:, 1] - self.bounds[:, 0]) / self.cell_size).astype(int)
        if int(np.prod(shape)) > 20000:
            raise ValueError("Exploration planning grid exceeds 20000 cells")
        self._cells = {}
        for x in range(int(shape[0])):
            for y in range(int(shape[1])):
                point = self.bounds[:, 0] + (np.array([x, y]) + 0.5) * self.cell_size
                if np.all(point < self.bounds[:, 1]):
                    self._cells[(x, y)] = point
        self._sector_cells = {key for key, point in self._cells.items() if self._sector(point) == self.index}
        step = max(1, int(round(self.spacing / self.cell_size)))
        self._candidates = []
        for key in sorted(self._sector_cells):
            point = self._cells[key]
            relative = point - self.origin[:2]
            sector_angle = abs(_angle(math.atan2(relative[1], relative[0]) - self.index * 2 * math.pi / self.fleet_size))
            # Leave a small margin at shared sector boundaries for separation.
            margin = min(math.radians(8), math.pi / self.fleet_size * 0.12)
            if (np.linalg.norm(relative) >= self.minimum_survey_radius
                    and sector_angle <= math.pi / self.fleet_size - margin
                    and key[0] % step == 0 and key[1] % step == 0):
                self._candidates.append(key)
        if not self._candidates:
            raise ValueError("Survey bounds/spacing leave no waypoints in this vehicle's sector")
        reach = int(math.ceil(self.scan_radius / self.cell_size))
        offsets = [(x, y) for x in range(-min(reach, int(shape[0])), min(reach, int(shape[0])) + 1)
                   for y in range(-min(reach, int(shape[1])), min(reach, int(shape[1])) + 1)
                   if math.hypot(x, y) * self.cell_size <= self.scan_radius]
        self._footprints = {key: tuple((key[0] + x, key[1] + y) for x, y in offsets
                                      if (key[0] + x, key[1] + y) in self._sector_cells)
                            for key in self._candidates}
        self.reset()

    @property
    def observed_cells(self) -> frozenset:
        """Measured XY cell IDs for the evaluator's fleet coverage count."""
        return frozenset(self._observed)

    def _sector(self, point) -> int:
        relative = np.asarray(point)[:2] - self.origin[:2]
        bearing = math.atan2(relative[1], relative[0])
        width = 2 * math.pi / self.fleet_size
        return int(math.floor((bearing + width / 2) % (2 * math.pi) / width))

    def reset(self, mode: str | None = None):
        """Clear a simulation epoch, including measured coverage and budgets."""
        if mode is not None:
            self.mode = canonical_mode(mode)
        self._observed = {}
        self._start = None
        self._last_time = None
        self._last_position = None
        self._home = None
        self._distance = 0.0
        self._rotation = 0.0
        self._ring_velocity = np.zeros(3)
        self._threat_id = None
        self._target = None
        self._target_key = None
        self._target_score = 0.0
        self._assigned_at = None
        self._progress_at = None
        self._best_distance = math.inf
        self._arrived_at = None
        self._visited = set()
        self._completed = 0
        self._return_reason = None

    def set_mode(self, mode: str):
        """Start a new mission budget on change; retain useful measured cells."""
        normalized = canonical_mode(mode)
        if normalized != self.mode:
            observed = self._observed
            self.reset(normalized)
            self._observed = observed

    def _ingest_rays(self, rays, position, uncertainty):
        for ray in rays if rays is not None else ():
            try:
                point = _vector(ray.get("endpoint", ray.get("position")) if isinstance(ray, Mapping) else ray)
            except (ValueError, TypeError, KeyError):
                continue
            if np.linalg.norm(point - position) > self.sensor_range + 3 * uncertainty:
                continue
            key = tuple(np.floor((point[:2] - self.bounds[:, 0]) / self.cell_size).astype(int))
            if key in self._cells:
                self._observed[key] = max(self._observed.get(key, -math.inf), float(point[2]))

    def _safe_goal(self, goal) -> tuple[np.ndarray, bool]:
        goal = _vector(goal)
        goal[:2] = np.clip(goal[:2], self.bounds[:, 0] + 0.1, self.bounds[:, 1] - 0.1)
        floor = self.depth_bounds[0]
        center = np.floor((goal[:2] - self.bounds[:, 0]) / self.cell_size).astype(int)
        for x in range(int(center[0]) - 2, int(center[0]) + 3):
            for y in range(int(center[1]) - 2, int(center[1]) + 3):
                key = (x, y)
                if key in self._observed and np.linalg.norm(self._cells[key] - goal[:2]) <= self.cell_size * 1.5:
                    floor = max(floor, self._observed[key] + self.altitude_clearance)
        limited = floor > self.depth_bounds[1]
        goal[2] = np.clip(max(goal[2], floor), self.depth_bounds[0], self.depth_bounds[1])
        return goal, limited

    def _contacts(self, observations, now):
        for item in observations if observations is not None else ():
            try:
                if hasattr(item, "snapshot"):
                    item = item.snapshot()
                if isinstance(item, Mapping):
                    position = _vector(item.get("position", item.get("p")))
                    velocity = _vector(item.get("velocity", item.get("v", [0, 0, 0])))
                    seen = float(item.get("last_seen", item.get("timestamp", item.get("t", now))))
                    kind = item.get("classification", item.get("label", "unknown"))
                    confidence = float(item.get("confidence", item.get("f", 0.5)))
                    uncertainty = float(item.get("position_std_m", item.get("uncertainty", 0)))
                    identifier = str(item.get("id", item.get("uid", "contact")))
                else:
                    position, velocity = _vector(item.position), _vector(getattr(item, "velocity", [0, 0, 0]))
                    seen, kind = float(item.timestamp), item.classification
                    confidence, identifier = float(item.confidence), str(item.uid)
                    uncertainty = math.sqrt(float(np.max(np.linalg.eigvalsh(item.covariance))))
                if (not np.all(np.isfinite([seen, confidence, uncertainty]))
                        or now - seen > self.contact_timeout or seen > now + 0.5
                        or kind == "diver" or confidence < 0.3):
                    continue
                yield position, velocity, str(kind), min(1, confidence), max(0, uncertainty), identifier
            except (ValueError, TypeError, KeyError, AttributeError, np.linalg.LinAlgError):
                continue

    def _peer_claims(self, peers, now):
        for peer in peers if peers is not None else ():
            try:
                payload = peer.get("payload", peer)
                at = float(payload.get("t", payload.get("timestamp", now)))
                if not math.isfinite(at) or now - at > 4 or at > now + 0.5:
                    continue
                claim = payload.get("mission", payload.get("m", {}))
                if not isinstance(claim, Mapping):
                    continue
                reported_at = float(claim.get("reported_at", at))
                if (not math.isfinite(reported_at) or now - reported_at > 4
                        or reported_at > now + 0.5):
                    continue
                if canonical_mode(claim.get("mode", "explore")) == "explore":
                    yield _vector(claim.get("target", claim.get("goal")))
            except (ValueError, TypeError, AttributeError):
                continue

    def _protect(self, now, dt, position, diver, observations, uncertainty):
        try:
            if isinstance(diver, Mapping):
                beacon = _vector(diver.get("position", diver.get("p")))
                velocity = _vector(diver.get("velocity", diver.get("v", [0, 0, 0])))
                at = float(diver.get("timestamp", diver.get("t", now)))
            else:
                beacon, velocity, at = _vector(diver), np.zeros(3), now
            if not math.isfinite(at) or at > now + 0.5 or now - at > self.beacon_timeout:
                raise ValueError("Stale beacon")
        except (ValueError, TypeError):
            return position.copy(), position.copy(), 0.0, "beacon_hold", "beacon_recovery", {}
        # Compensate for the beacon's transmission delay.
        beacon += velocity * max(0, now - at)
        if uncertainty > self.uncertainty_limit * 2:
            return position.copy(), beacon, 0.0, "localization_hold", "localization_recovery", {}
        best = current = None
        watch_radius = self.protected_radius + self.radius + 8
        contacts = list(self._contacts(observations, now))
        for contact, contact_v, kind, confidence, std, identifier in contacts:
            relative, relative_v = contact - beacon, contact_v - velocity
            speed_sq = float(np.dot(relative_v, relative_v))
            closest_t = float(np.clip(-np.dot(relative, relative_v) / max(speed_sq, 1e-9), 0, self.prediction_horizon))
            closest_distance = float(np.linalg.norm(relative + relative_v * closest_t))
            closing = -float(np.dot(relative, relative_v)) / max(float(np.linalg.norm(relative)), 1e-6)
            effective_distance = max(0, closest_distance - min(std * 2, 4))
            # Weight closing speed smoothly (0.25 when receding, 1.0 when
            # closing at >= 0.15 m/s). A hard 0.05 m/s switch made risk jump
            # 4x on small speed changes and the ring flip between threats.
            approach = 0.25 + 0.75 * float(np.clip(closing / 0.15, 0.0, 1.0))
            risk = confidence * max(0, 1 - effective_distance / watch_radius) * approach
            direction = relative + relative_v * min(closest_t, self.prediction_horizon / 2)
            if np.linalg.norm(direction[:2]) < 1e-5:
                direction = relative
            candidate = (risk, closest_distance, closest_t, direction, identifier)
            if best is None or risk > best[0]:
                best = candidate
            if identifier == self._threat_id:
                current = candidate
        # Hysteresis: keep facing the current threat unless another contact is
        # clearly (30%) riskier, so similar contacts do not make the ring swing.
        if (best is not None and current is not None and current[4] != best[4]
                and current[0] >= 0.18 and best[0] < current[0] * 1.3):
            best = current
        self._threat_id = best[4] if best is not None and best[0] >= 0.18 else None
        desired_rotation = 0.0
        risk_info = {"score": 0.0, "contact_id": None, "closest_approach_m": None, "time_to_closest_s": None}
        if best is not None:
            risk, closest, closest_t, direction, identifier = best
            risk_info = {"score": round(risk, 3), "contact_id": identifier,
                         "closest_approach_m": round(closest, 2), "time_to_closest_s": round(closest_t, 2)}
            if risk >= 0.18:
                desired_rotation = math.atan2(direction[1], direction[0])
        # The ring has fleet_size-fold symmetry, so any guard can face the
        # threat. Turn by the smallest equivalent angle (within +/- half a
        # sector). Wrapping the target into a fixed +/-60 degree window made
        # the ring swing up to 120 degrees backwards whenever a threat crossed
        # a sector edge; now the neighbouring guard takes over instead.
        width = 2 * math.pi / self.fleet_size
        radius = self.radius + min(uncertainty, 1.0) * 0.35
        rate = min(self.rotation_rate, self.speed_limit * 0.65 / radius)
        difference = (desired_rotation - self._rotation + width / 2) % width - width / 2
        step = float(np.clip(difference, -rate * dt, rate * dt))
        self._rotation = _angle(self._rotation + step)
        bearing = self.index * 2 * math.pi / self.fleet_size + self._rotation
        radial = np.array([math.cos(bearing), math.sin(bearing), 0.0])
        goal, limited = self._safe_goal(beacon + radius * radial)
        # When the ring turns toward a threat, each slot moves sideways at
        # omega * radius. Reporting that motion lets the velocity controller
        # follow the slot instead of only reacting once it has fallen behind.
        omega = step / dt if dt > 0 else 0.0
        self._ring_velocity = omega * radius * np.array([-math.sin(bearing), math.cos(bearing), 0.0])

        def contact_margin(candidate):
            margin = math.inf
            for contact, contact_v, _, _, std, _ in contacts:
                difference = contact - candidate
                closest_t = float(np.clip(-np.dot(difference, contact_v) / max(np.dot(contact_v, contact_v), 1e-9), 0, 3))
                margin = min(margin, float(np.linalg.norm(difference + contact_v * closest_t))
                             - self.contact_clearance - min(std * 2, 1))
            return margin

        contact_hold = False
        if contact_margin(goal) < 0:
            # Move a slot away from the contact's next three seconds of travel.
            # The velocity controller still enforces separation during motion.
            candidates = []
            maximum_offset = min(0.65, math.pi / self.fleet_size * 0.62)
            for offset in np.linspace(-maximum_offset, maximum_offset, 9):
                candidate_bearing = bearing + float(offset)
                candidate_radial = np.array([math.cos(candidate_bearing), math.sin(candidate_bearing), 0.0])
                candidate_goal, candidate_limited = self._safe_goal(beacon + radius * candidate_radial)
                if not candidate_limited and contact_margin(candidate_goal) >= 0:
                    candidates.append((abs(offset), candidate_bearing, candidate_radial, candidate_goal))
            if candidates:
                _, bearing, radial, goal = min(candidates, key=lambda candidate: candidate[0])
                risk_info["clearance_detour"] = True
            else:
                goal, contact_hold = position.copy(), True
        # Clipping at the survey boundary must preserve diver clearance.
        if np.linalg.norm(goal - beacon) < self.diver_clearance:
            goal = position.copy()
            phase, role = "clearance_hold", "perimeter_guard"
        else:
            phase, role = "escort", "perimeter_guard"
            if risk_info["score"] >= 0.18 and best is not None:
                phase = "predictive_watch"
                toward = best[3][:2] / max(np.linalg.norm(best[3][:2]), 1e-9)
                if float(np.dot(radial[:2], toward)) > 0.7:
                    role = "predictive_sentinel"
        if limited:
            phase = "clearance_hold"
            goal = position.copy()
        if contact_hold:
            phase = "contact_clearance_hold"
        risk_info["formation_rotation_rad"] = round(self._rotation, 3)
        return goal, beacon, bearing, phase, role, risk_info

    def _choose_target(self, position, now, peers):
        claims = list(self._peer_claims(peers, now))
        layer = min(len(self.depth_layers) - 1, self._completed // max(1, len(self._candidates)))
        options = []
        for key in self._candidates:
            task = (key, layer)
            if task in self._visited:
                continue
            goal, limited = self._safe_goal(np.r_[self._cells[key], self.depth_layers[layer]])
            if limited:
                continue
            footprint = self._footprints[key]
            unknown = sum(other not in self._observed for other in footprint)
            gain = unknown / max(len(footprint), 1)
            measured = len(footprint) - unknown
            frontier = 0.3 if measured and unknown else 0.0
            distance = float(np.linalg.norm(goal - position))
            # Sector ownership survives outages; fresh peer claims avoid overlap.
            contention = sum(max(0, 1 - np.linalg.norm(goal - claim) / self.spacing) * 3 for claim in claims)
            score = gain + frontier - 0.045 * distance - contention
            if self._home is not None:
                score -= 0.006 * np.linalg.norm(goal - self._home)
            options.append((score, -distance, key, layer, goal))
        if not options:
            return False
        score, _, key, layer, goal = max(options, key=lambda item: (item[0], item[1], item[2]))
        self._target, self._target_key, self._target_score = goal, (key, layer), float(score)
        self._assigned_at = self._progress_at = now
        self._best_distance = float(np.linalg.norm(goal - position))
        self._arrived_at = None
        return True

    def _explore(self, now, position, peers, uncertainty):
        elapsed = now - self._start
        home_distance = float(np.linalg.norm(position - self._home))
        home_time = home_distance / max(self.speed_limit * 0.65, 0.1)
        reserve_distance = max(self.distance_budget * self.reserve_ratio, home_distance * 1.2)
        progress = len(self._sector_cells & self._observed.keys()) / max(len(self._sector_cells), 1)
        if self._return_reason is None:
            if uncertainty > self.uncertainty_limit:
                self._return_reason = "localization_uncertainty"
            elif elapsed >= self.max_duration - max(home_time, self.max_duration * self.reserve_ratio):
                self._return_reason = "time_reserve"
            elif self._distance >= self.distance_budget - reserve_distance:
                self._return_reason = "distance_reserve"
            elif progress >= self.target_coverage:
                self._return_reason = "sector_coverage_reached"
        if self._return_reason is not None:
            if uncertainty > self.uncertainty_limit * 2:
                goal, phase, role = position.copy(), "localization_hold", "localization_recovery"
            else:
                goal, limited = self._safe_goal(self._home)
                phase = "complete" if home_distance <= self.tolerance else "return"
                role = "return_home"
                if limited:
                    goal, phase = position.copy(), "clearance_hold"
            direction = goal - position
            heading = math.atan2(direction[1], direction[0]) if np.linalg.norm(direction[:2]) > 0.1 else self.index * 2 * math.pi / self.fleet_size
            return goal, self.origin.copy(), heading, phase, role, {}
        if self._target is not None:
            distance = float(np.linalg.norm(self._target - position))
            if distance < self._best_distance - 0.25:
                self._best_distance, self._progress_at = distance, now
            if distance <= self.tolerance:
                if self._arrived_at is None:
                    self._arrived_at = now
                if now - self._arrived_at >= self.dwell_time:
                    self._visited.add(self._target_key)
                    self._completed += 1
                    self._target = None
            elif distance > self.tolerance * 1.6:
                self._arrived_at = None
            if (self._target is not None and now - self._assigned_at >= self.minimum_hold
                    and now - self._progress_at >= self.target_timeout):
                # A stalled waypoint cannot monopolize the mission forever.
                self._visited.add(self._target_key)
                self._target = None
        if self._target is None and not self._choose_target(position, now, peers):
            self._return_reason = "survey_route_exhausted"
            return self._explore(now, position, peers, uncertainty)
        goal, limited = self._safe_goal(self._target)
        phase = "survey_dwell" if self._arrived_at is not None else "survey_transit"
        if limited:
            goal, phase = position.copy(), "clearance_hold"
        direction = goal - position
        heading = math.atan2(direction[1], direction[0]) if phase == "survey_transit" else _angle(self.index * 2 * math.pi / self.fleet_size + (now - self._assigned_at) * 0.16)
        return goal, self.origin.copy(), heading, phase, f"survey_sector_{self.index + 1}", {}

    def plan(self, now: float, position, diver=None, peers=(), observations=(), rays=(),
             uncertainty: float = 0.0) -> MissionPlan:
        """Update one estimate-only mission and return a stable bounded goal."""
        now = float(now)
        position = _vector(position)
        uncertainty = _positive(uncertainty, "uncertainty", zero=True)
        if not math.isfinite(now):
            raise ValueError("Mission time must be finite")
        if self._last_time is not None and now < self._last_time:
            self.reset()
        dt = 0.0 if self._last_time is None else min(now - self._last_time, 1.0)
        if self._start is None:
            self._start = now
            self._home, _ = self._safe_goal(position)
        if self._last_position is not None:
            distance = float(np.linalg.norm(position - self._last_position))
            if distance >= 0.03:
                self._distance += distance
        self._last_time, self._last_position = now, position.copy()
        self._ingest_rays(rays, position, uncertainty)
        self._ring_velocity = np.zeros(3)
        if self.mode == "protect":
            goal, anchor, heading, phase, role, risk = self._protect(now, dt, position, diver, observations, uncertainty)
        else:
            goal, anchor, heading, phase, role, risk = self._explore(now, position, peers, uncertainty)
        observed = len(self._observed)
        coverage = observed / max(len(self._cells), 1)
        progress = len(self._sector_cells & self._observed.keys()) / max(len(self._sector_cells), 1)
        status = {"mode": self.mode, "phase": phase, "state": phase, "role": role,
                  "target": goal.tolist(), "sector": self.index,
                  "observed_cells": observed, "total_cells": len(self._cells),
                  "coverage": round(coverage, 5), "coverage_pct": round(coverage * 100, 2),
                  "progress": round(progress, 5), "sector_progress": round(progress, 5),
                  "waypoints_completed": self._completed, "target_score": round(self._target_score, 3),
                  "estimated_distance_m": round(self._distance, 2),
                  "elapsed_s": round(now - self._start, 2),
                  "remaining_s": round(max(0, self.max_duration - (now - self._start)), 2),
                  "localization_std_m": round(uncertainty, 3), "return_reason": self._return_reason,
                  "risk": risk, "ring_velocity": np.round(self._ring_velocity, 4).tolist(),
                  "depth_layer": self._target_key[1] if self._target_key is not None else 0,
                  "exploration": {"bounds": self.bounds.tolist(), "origin": self.origin.tolist(),
                                  "depth_layers": self.depth_layers.tolist(),
                                  "observed_cells": observed, "total_cells": len(self._cells),
                                  "coverage": round(coverage, 5)}}
        return MissionPlan(_vector(goal), _vector(anchor), float(heading), status)
