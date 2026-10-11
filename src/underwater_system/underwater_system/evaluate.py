"""Seeded algorithm evaluation, deliberately separate from Gazebo validation.

Run ``python -m underwater_system.evaluate --duration 60 --seed 7
--output results/evaluation.json``. All three scenarios use the same sensor
seed and software harness. This exercises synthetic sensors, numerical motion,
tracking, occupancy mapping and the packet-level acoustic channel. It does not
measure Gazebo physics, real hydrophones, trained AI, modem hardware or MQTT.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .core import CLASS_NAMES, SimulationCore


SCENARIOS = {
    "nominal": {"packet_loss": 0.12},
    "high_packet_loss": {"packet_loss": 0.65},
    "outage_recovery": {"packet_loss": 0.12},
}
CHI_SQUARED_3D_95 = 7.814727903251179


def _validate(duration: float, seed: int, dt: float):
    if not math.isfinite(duration) or duration < 6:
        raise ValueError("duration must be finite and at least 6 seconds")
    if not math.isfinite(dt) or not 0 < dt <= 0.2:
        raise ValueError("dt must be finite and between 0 and 0.2 seconds")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _check(name: str, passed: Optional[bool], measured, criterion: str) -> dict:
    return {"name": name,
            "status": "not_evaluated" if passed is None else "passed" if passed else "failed",
            "measured": measured, "criterion": criterion}


def _classification(confusion: dict) -> dict:
    matrix = {truth: {prediction: int(confusion[truth][prediction])
                      for prediction in CLASS_NAMES} for truth in CLASS_NAMES}
    precision, recall = {}, {}
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[label][label] for label in CLASS_NAMES)
    for label in CLASS_NAMES:
        predicted = sum(row[label] for row in matrix.values())
        actual = sum(matrix[label].values())
        precision[label] = matrix[label][label] / predicted if predicted else None
        recall[label] = matrix[label][label] / actual if actual else None
    return {"labels": list(CLASS_NAMES), "confusion_matrix": matrix,
            "precision_by_class": precision, "recall_by_class": recall,
            "micro_precision": correct / total if total else None,
            "observations": total,
            "scope": "Synthetic acoustic templates and class-tag silhouette cues; untrained baseline."}


def _tracking_matches(core: SimulationCore) -> dict:
    """Greedy, one-to-one geometric evaluation, with duplicates counted explicitly."""
    tracks = list(core.tracker.tracks.values())
    truths = list(core.objects.values())
    candidates = []
    nearby = Counter()
    for ti, track in enumerate(tracks):
        distances = [float(np.linalg.norm(track.position - obj.position)) for obj in truths]
        if distances and min(distances) <= 4:
            nearby[int(np.argmin(distances))] += 1
        candidates.extend((distance, ti, oi) for oi, distance in enumerate(distances)
                          if distance <= 4)
    used_tracks, used_objects, errors = set(), set(), []
    for distance, ti, oi in sorted(candidates):
        if ti not in used_tracks and oi not in used_objects:
            used_tracks.add(ti)
            used_objects.add(oi)
            errors.append(distance)
    return {"association": "Greedy one-to-one position association within 4 m; truth IDs unused by tracker.",
            "tracks": len(tracks), "truth_objects": len(truths),
            "matched_tracks": len(errors), "unmatched_tracks": len(tracks) - len(errors),
            "unmatched_objects": len(truths) - len(errors),
            "duplicate_candidates": sum(max(0, count - 1) for count in nearby.values()),
            "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "precision": len(errors) / len(tracks) if tracks else None,
            "recall": len(errors) / len(truths) if truths else None}


def run_scenario(name: str, duration: float = 60.0, seed: int = 7,
                 dt: float = 0.2) -> dict:
    """Return measurements and explicit checks, retaining failed measurements."""
    _validate(duration, seed, dt)
    if name not in SCENARIOS:
        raise ValueError(f"Unknown evaluation scenario: {name}")
    core = SimulationCore(seed=seed, config=SCENARIOS[name])
    confusion = defaultdict(Counter)
    uncertainty_covered = 0
    uncertainty_samples = 0

    # Instrument only the truth-side sensor fixture. No extra random draws and
    # no truth identity is passed to the controller or tracker by this wrapper.
    def instrument(original):
        def observe(*args, **kwargs):
            nonlocal uncertainty_covered, uncertainty_samples
            observations = original(*args, **kwargs)
            for observation in observations:
                truth = core.objects[observation.truth_id]
                confusion[truth.kind][observation.classification] += 1
                error = observation.position - truth.position
                squared = float(error @ np.linalg.solve(observation.covariance, error))
                uncertainty_covered += squared <= CHI_SQUARED_3D_95
                uncertainty_samples += 1
            return observations
        return observe

    for agent in core.agents:
        agent.sensor.observe = instrument(agent.sensor.observe)

    outage_start, outage_end = duration * 0.30, duration * 0.70
    events = []
    is_outage = name == "outage_recovery"
    outage_started, reconnected = False, False
    reconnect_at = None
    first_recovered_delivery = None
    stale_peer_removed = None
    max_queue_depth = 0
    max_command_speed = 0.0
    minimum_diver_clearance = math.inf
    intruder_entry_at = None
    first_intruder_alert = None
    alert_counts = Counter()
    finite_commands = True
    history = []
    next_history = 0.0
    first_delivery_after_outage = None

    while core.time < duration - 1e-9:
        # Split steps on fault boundaries so the reported event times are exact.
        if is_outage and not outage_started and core.time >= outage_start - 1e-9:
            core.network.set_connected("auv_1", False)
            outage_started = True
            events.append({"time_s": round(core.time, 6), "node": "auv_1", "event": "acoustic_link_disconnected"})
        if is_outage and outage_started and not reconnected and core.time >= outage_end - 1e-9:
            stale_peer_removed = all("auv_1" not in agent.peers for agent in core.agents if agent.name != "auv_1")
            core.network.set_connected("auv_1", True)
            reconnected = True
            reconnect_at = core.time
            events.append({"time_s": round(core.time, 6), "node": "auv_1", "event": "acoustic_link_reconnected"})
        interval = min(dt, duration - core.time)
        if is_outage:
            for boundary in (outage_start, outage_end):
                if core.time < boundary - 1e-9:
                    interval = min(interval, boundary - core.time)
        state = core.step(interval)
        for queue in core.network.queues.values():
            max_queue_depth = max(max_queue_depth, len(queue))
        for command in core.commands.values():
            finite_commands = finite_commands and bool(np.all(np.isfinite(command)))
            max_command_speed = max(max_command_speed, float(np.linalg.norm(command)))
        diver_position = core.objects["diver"].position
        for vehicle in core.auvs.values():
            minimum_diver_clearance = min(minimum_diver_clearance,
                                         float(np.linalg.norm(vehicle["position"] - diver_position)))
        intruder = core.objects["intruder"]
        if intruder_entry_at is None and np.linalg.norm(intruder.position - diver_position) < 4:
            intruder_entry_at = core.time
        for alert in state["alerts"]:
            alert_counts[alert["severity"]] += 1
            track = core.tracker.tracks.get(alert["track_id"])
            if (first_intruder_alert is None and track is not None
                    and alert["severity"] in ("warning", "critical")
                    and np.linalg.norm(track.position - intruder.position) <= 4):
                first_intruder_alert = core.time
        last_delivery = core.network.last_delivery["auv_1"]
        if outage_started and first_delivery_after_outage is None and last_delivery is not None and last_delivery > outage_start:
            # Already in-flight packets may arrive after disconnect. This is
            # measured explicitly rather than incorrectly treating it as failure.
            first_delivery_after_outage = last_delivery
        if reconnected and first_recovered_delivery is None and last_delivery is not None and last_delivery >= reconnect_at:
            first_recovered_delivery = last_delivery
            events.append({"time_s": round(last_delivery, 6), "node": "auv_1", "event": "first_post_reconnection_delivery"})
        if core.time >= next_history - 1e-9:
            history.append({"time_s": round(core.time, 3),
                            "packet_delivery_ratio": state["metrics"]["packet_delivery_ratio"],
                            "formation_rmse_m": state["metrics"]["formation_rmse_m"],
                            "occupied_voxels": state["map"]["occupied_voxels"],
                            "auv_1_connected": core.network.connected["auv_1"],
                            "peer_counts": {agent.name: len(agent.peers) for agent in core.agents}})
            next_history += max(1.0, duration / 60)

    metrics = core.evaluate()
    network = core.network.snapshot()
    slots = math.floor(core.time / core.network.slot_duration + 1e-8) + 1
    byte_budget = slots * core.network.slot_duration * core.network.bitrate / 8
    bytes_with_headers = network["bytes_transmitted"] + 24 * network["transmitted"]
    recovery_latency = None if first_recovered_delivery is None else first_recovered_delivery - reconnect_at
    sample_coverage = uncertainty_covered / uncertainty_samples if uncertainty_samples else None
    checks = [
        _check("finite_bounded_commands", finite_commands and max_command_speed <= 1.3 + 1e-8,
               max_command_speed, "All command components finite; maximum commanded speed <= 1.3 m/s."),
        _check("auv_separation", metrics["minimum_auv_separation_m"] >= 2.4,
               metrics["minimum_auv_separation_m"], "Minimum pairwise AUV centre separation >= 2.4 m."),
        _check("wildlife_clearance", metrics["minimum_wildlife_clearance_m"] >= 0.8,
               metrics["minimum_wildlife_clearance_m"], "Synthetic wildlife and AUV surfaces stay >= 0.8 m apart."),
        _check("seabed_clearance", metrics["minimum_seabed_clearance_m"] >= 1.0,
               metrics["minimum_seabed_clearance_m"], "Synthetic AUV surface stays >= 1 m above the analytic seabed."),
        _check("diver_clearance", minimum_diver_clearance >= 2.4,
               minimum_diver_clearance, "Every AUV stays at least 2.4 m from the diver centre."),
        _check("mapping_accuracy", metrics["mapping_rmse_m"] is not None and metrics["mapping_rmse_m"] <= core.map.resolution,
               metrics["mapping_rmse_m"], "Observed analytic-harness seabed elevation RMSE <= one map voxel."),
        _check("sensor_position_accuracy", metrics["sonar_position_rmse_m"] is not None and metrics["sonar_position_rmse_m"] <= 1,
               metrics["sonar_position_rmse_m"], "Synthetic sonar detection position RMSE <= 1 m."),
        _check("synthetic_classifier", metrics["sensor_observations"] >= 10 and metrics["classification_accuracy"] >= 0.75,
               metrics["classification_accuracy"], "At least 10 synthetic detections and classification accuracy >= 0.75."),
        _check("capacity_and_queues", bytes_with_headers <= byte_budget + 1e-8 and max_queue_depth <= core.network.max_queue,
               {"bytes_including_headers": bytes_with_headers, "slot_budget_bytes": byte_budget, "maximum_queue_depth": max_queue_depth},
               "Transmitted bytes including headers fit scheduled slot capacity; queues <= configured maximum."),
        _check("delivery_ratio", 0 <= metrics["packet_delivery_ratio"] <= 1,
               metrics["packet_delivery_ratio"], "Deduplicated delivered / transmitted packets is within [0, 1]."),
    ]
    if intruder_entry_at is not None:
        checks.append(_check("approach_alert_before_entry", first_intruder_alert is not None and first_intruder_alert <= intruder_entry_at,
                             {"first_alert_s": first_intruder_alert, "protected_zone_entry_s": intruder_entry_at},
                             "A warning/critical alert associated geometrically with the approaching object precedes its 4 m zone entry."))
    else:
        checks.append(_check("approach_alert_before_entry", None,
                             {"first_alert_s": first_intruder_alert, "protected_zone_entry_s": None},
                             "Requires a run long enough for the scenario object to enter the protected zone (about 34 s)."))
    if is_outage:
        checks.append(_check("outage_recovery", network["outages"] == 1 and network["reconnections"] == 1 and recovery_latency is not None,
                             {"outages": network["outages"], "reconnections": network["reconnections"], "first_delivery_delay_s": recovery_latency},
                             "One forced acoustic link outage, one reconnection, and a fresh post-reconnection delivery before run end."))
        checks.append(_check("stale_peer_expiry", stale_peer_removed if outage_end - outage_start > 4.4 else None,
                             stale_peer_removed,
                             "Other nodes remove auv_1 telemetry after its 4 s freshness limit; evaluated only for outage windows > 4.4 s."))
    classification = _classification(confusion)
    return {"name": name, "duration_s": duration, "seed": seed, "integration_step_s": dt,
            "configured_packet_loss": SCENARIOS[name]["packet_loss"],
            "all_applicable_checks_passed": all(check["status"] != "failed" for check in checks),
            "metrics": metrics, "network": network, "classification": classification,
            "tracking_at_end": _tracking_matches(core),
            "uncertainty": {"nominal_position_ellipsoid_probability": 0.95,
                            "mahalanobis_squared_threshold": CHI_SQUARED_3D_95,
                            "empirical_position_coverage": sample_coverage,
                            "observations": uncertainty_samples},
            "safety": {"minimum_auv_diver_clearance_m": minimum_diver_clearance,
                       "maximum_command_speed_m_s": max_command_speed,
                       "first_intruder_alert_s": first_intruder_alert,
                       "intruder_protected_zone_entry_s": intruder_entry_at,
                       "alert_lead_time_s": intruder_entry_at - first_intruder_alert if intruder_entry_at is not None and first_intruder_alert is not None else None,
                       "alert_time_steps_by_severity": dict(alert_counts)},
            "fault_events": events, "recovery_latency_s": recovery_latency,
            "inflight_first_delivery_after_disconnect_s": first_delivery_after_outage,
            "checks": checks, "time_series": history}


def run_evaluation(duration: float = 60.0, seed: int = 7, dt: float = 0.2) -> dict:
    """Compare seeded nominal, lossy and temporarily disconnected channels."""
    _validate(duration, seed, dt)
    results = {name: run_scenario(name, duration, seed, dt) for name in SCENARIOS}
    nominal = results["nominal"]["metrics"]
    lossy = results["high_packet_loss"]["metrics"]
    enough_packets = all(results[name]["network"]["transmitted"] >= 20
                         for name in ("nominal", "high_packet_loss"))
    delivery_decreased = lossy["packet_delivery_ratio"] < nominal["packet_delivery_ratio"]
    comparison_check = _check("loss_stress_changes_delivery", delivery_decreased if enough_packets else None,
                              {"nominal": nominal["packet_delivery_ratio"], "high_loss": lossy["packet_delivery_ratio"]},
                              "High-loss observed packet delivery ratio is lower for runs with at least 20 transmitted packets per scenario.")
    return {"schema_version": 1, "evaluation_type": "seeded_offline_software_harness",
            "seed": seed, "duration_s": duration,
            "passed": all(result["all_applicable_checks_passed"] for result in results.values()) and comparison_check["status"] != "failed",
            "method": {"coordinates": "ENU metres; z up", "motion": "Numerical first-order response with a modest current, not Gazebo",
                       "mapping_truth": "Analytic seabed_height fixture; not the Gazebo seabed/structures",
                       "network_fault": "auv_1 acoustic-link blackout; its autonomy continues running",
                       "classification": "Untrained synthetic feature/template baseline with synthetic class-tag silhouette cues",
                       "reproducibility": "Same sensor and navigation seeds; network paths diverge after packet-loss draws.",
                       "claim_limit": "These measurements do not validate real sensing, trained AI, physical protection, Gazebo physics, or hardware."},
            "scenarios": results,
            "comparisons": {"packet_delivery_ratio_drop": nominal["packet_delivery_ratio"] - lossy["packet_delivery_ratio"],
                            "formation_rmse_change_m": lossy["formation_rmse_m"] - nominal["formation_rmse_m"],
                            "outage_recovery_latency_s": results["outage_recovery"]["recovery_latency_s"],
                            "checks": [comparison_check]}}


def run_mission_scenario(mode: str, duration: float = 60.0, seed: int = 7,
                         dt: float = 0.2, switch_modes: bool = False) -> dict:
    """Exercise actual dual-mission motion, measured coverage and mode changes."""
    _validate(duration, seed, dt)
    core = SimulationCore(seed=seed, mode=mode)
    initial = {name: state["position"].copy() for name, state in core.auvs.items()}
    previous = {name: position.copy() for name, position in initial.items()}
    distances = {name: 0.0 for name in initial}
    max_speed, finite, map_preserved, mode_continuity = 0.0, True, True, True
    events, history = [], []
    switches = [(duration / 3, "explore"), (duration * 2 / 3, "protect")] if switch_modes else []
    next_history = 0.0
    while core.time < duration - 1e-9:
        if switches and core.time >= switches[0][0] - 1e-9:
            _, new_mode = switches.pop(0)
            before = {name: state["position"].copy() for name, state in core.auvs.items()}
            mapped_cells = len(core.map.cells)
            core.set_mode(new_mode)
            map_preserved = map_preserved and len(core.map.cells) == mapped_cells
            mode_continuity = mode_continuity and all(np.array_equal(before[name], state["position"])
                                                     for name, state in core.auvs.items())
            events.append({"time_s": round(core.time, 3), "mode": core.mode})
        interval = min(dt, duration - core.time)
        if switches:
            interval = min(interval, switches[0][0] - core.time)
        state = core.step(interval)
        for name, vehicle in core.auvs.items():
            distances[name] += float(np.linalg.norm(vehicle["position"] - previous[name]))
            previous[name] = vehicle["position"].copy()
        for command in core.commands.values():
            finite = finite and bool(np.all(np.isfinite(command)))
            max_speed = max(max_speed, float(np.linalg.norm(command)))
        if core.time >= next_history - 1e-9:
            history.append({"time_s": round(core.time, 3), "mode": core.mode,
                            "coverage_pct": state["mission"]["coverage_pct"],
                            "positions": {name: np.round(vehicle["position"], 3).tolist()
                                          for name, vehicle in core.auvs.items()},
                            "targets": {unit["id"]: unit["target"] for unit in state["auvs"]}})
            next_history += max(1.0, duration / 60)
    metrics = core.evaluate()
    mission = core.mission_status()
    checks = [
        _check("finite_bounded_commands", finite and max_speed <= 1.3 + 1e-8,
               max_speed, "All command components finite; speed <= 1.3 m/s."),
        _check("auv_separation", metrics["minimum_auv_separation_m"] >= 2.4,
               metrics["minimum_auv_separation_m"], "Pairwise AUV centre separation >= 2.4 m."),
        _check("wildlife_clearance", metrics["minimum_wildlife_clearance_m"] >= 0.8,
               metrics["minimum_wildlife_clearance_m"], "Synthetic wildlife and AUV surfaces stay >= 0.8 m apart."),
        _check("seabed_clearance", metrics["minimum_seabed_clearance_m"] >= 1.0,
               metrics["minimum_seabed_clearance_m"], "Synthetic AUV surface stays >= 1 m above the analytic seabed."),
        _check("measured_mapping", metrics["coverage_fraction"] > 0 and metrics["mapping_rmse_m"] <= 1,
               {"coverage": metrics["coverage_fraction"], "rmse_m": metrics["mapping_rmse_m"]},
               "Actual sonar return endpoints populate the map; observed seabed error <= 1 m."),
    ]
    if mode == "explore" and not switch_modes:
        checks.extend([
            _check("autonomous_survey_motion", min(distances.values()) >= duration * 0.15,
                   distances, "Every exploring AUV travels at least 0.15 m per simulated second."),
            _check("survey_progress_measured", mission["observed_cells"] > 0 and 0 < mission["coverage_pct"] <= 100,
                   {"observed_cells": mission["observed_cells"], "coverage_pct": mission["coverage_pct"]},
                   "Survey coverage comes from finite measured sonar returns and remains within [0, 100]."),
            _check("formation_metric_not_applied", metrics["formation_rmse_m"] is None,
                   metrics["formation_rmse_m"], "Diver formation error is undefined during exploration."),
        ])
    if switch_modes:
        checks.append(_check("mode_switch_preserves_state", map_preserved and mode_continuity and len(events) == 2,
                             {"map_preserved": map_preserved, "pose_continuity": mode_continuity, "events": events},
                             "Both mode changes preserve the observed map and all physical vehicle poses."))
    return {"name": "mode_switch" if switch_modes else mode, "duration_s": duration, "seed": seed,
            "all_applicable_checks_passed": all(check["status"] != "failed" for check in checks),
            "metrics": metrics, "network": core.network.snapshot(), "mission": mission,
            "distance_travelled_m": distances, "mode_events": events,
            "checks": checks, "time_series": history}


def run_mission_evaluation(duration: float = 60.0, seed: int = 7, dt: float = 0.2) -> dict:
    """Compare protect, explore and protect/explore/protect with the same seed."""
    scenarios = {mode: run_mission_scenario(mode, duration, seed, dt)
                 for mode in ("protect", "explore")}
    scenarios["mode_switch"] = run_mission_scenario("protect", duration, seed, dt, switch_modes=True)
    return {"schema_version": 2, "evaluation_type": "seeded_dual_mission_software_harness",
            "seed": seed, "duration_s": duration,
            "passed": all(scenario["all_applicable_checks_passed"] for scenario in scenarios.values()),
            "method": {"motion": "Numerical first-order thruster response plus modest current",
                       "exploration": "Distributed frontier scoring from measured sonar-return coverage",
                       "protection": "Predictive threat-facing guard envelope with local separation control",
                       "claim_limit": "Synthetic software simulation; real sensors, Gazebo physics and physical protection require separate validation."},
            "scenarios": scenarios}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation seconds per scenario, minimum 6 (default: 60).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dt", type=float, default=0.2, help="Numerical integration step, maximum 0.2 s.")
    parser.add_argument("--mode", choices=("protect", "explore", "both"), default="protect",
                        help="protect runs network stress scenarios; explore runs survey checks; both checks both missions and mode switching.")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    args = parser.parse_args(argv)
    try:
        if args.mode == "both":
            report = run_mission_evaluation(args.duration, args.seed, args.dt)
        elif args.mode == "explore":
            result = run_mission_scenario("explore", args.duration, args.seed, args.dt)
            report = {"schema_version": 2, "evaluation_type": "seeded_exploration_software_harness",
                      "passed": result["all_applicable_checks_passed"], "scenarios": {"explore": result}}
        else:
            report = run_evaluation(args.duration, args.seed, args.dt)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Seeded software harness; synthetic sensors and untrained classifier; Gazebo validation is separate.")
    print("scenario             delivery    map RMSE    sonar RMSE   min separation")
    for name, result in report["scenarios"].items():
        m = result["metrics"]
        print(f"{name:20} {m['packet_delivery_ratio']:8.1%} {m['mapping_rmse_m']:9.3f} m {m['sonar_position_rmse_m']:10.3f} m {m['minimum_auv_separation_m']:12.3f} m")
        for check in result["checks"]:
            if check["status"] == "failed":
                print(f"  FAILED {check['name']}: {check['measured']}")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: applicable checks. Report: {args.output.resolve()}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
