"""Validated operator commands and a wall-clock deadman for simulated diver input.

This module has no ROS or simulator dependency. A future adapter can reuse the
command validation without taking ownership of mission decisions.
"""
from __future__ import annotations

import math

COMMANDS = frozenset({"start", "pause", "reset", "toggle_outage", "set_current",
                      "set_mode", "set_diver_control"})
MODES = frozenset({"protect", "explore"})


def vector(value, limit, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three numbers: east, north, up")
    if any(type(x) not in (int, float) or abs(x) > limit
           or (type(x) is float and not math.isfinite(x)) for x in value):
        raise ValueError(f"{name} components must be finite numbers within ±{limit}")
    return [float(x) for x in value]


def validate_command(payload):
    if not isinstance(payload, dict) or payload.get("command") not in COMMANDS:
        raise ValueError("Unknown mission command")
    result = {"command": payload["command"]}
    if result["command"] == "set_current":
        result["vector"] = vector(payload.get("vector"), .8, "Current vector")
    elif result["command"] == "set_mode":
        if not isinstance(payload.get("mode"), str) or payload["mode"] not in MODES:
            raise ValueError("Mission mode must be protect or explore")
        result["mode"] = payload["mode"]
    elif result["command"] == "set_diver_control":
        mode = payload.get("mode")
        if not isinstance(mode, str) or mode not in ("manual", "scripted"):
            raise ValueError("Diver control must be manual or scripted")
        velocity = vector(payload.get("velocity", [0, 0, 0]), .65, "Diver velocity")
        if math.hypot(*velocity) > .650001:
            raise ValueError("Combined diver speed must be at most 0.65 m/s")
        result.update(mode=mode, velocity=velocity if mode == "manual" else [0., 0., 0.])
    return result


class DiverPilot:
    """Held input expires even when the simulator clock is paused."""
    def __init__(self, timeout=1.2):
        self.timeout = timeout
        self.mode = "scripted"
        self.velocity = [0., 0., 0.]
        self.last_input = -math.inf

    def command(self, mode, velocity, now):
        accepted = validate_command({"command": "set_diver_control", "mode": mode,
                                     "velocity": velocity})
        self.mode, self.velocity = accepted["mode"], accepted["velocity"]
        self.last_input = now

    def stop(self):
        self.velocity = [0., 0., 0.]

    def snapshot(self, now, paused=False):
        expired = self.mode == "manual" and now - self.last_input > self.timeout
        if expired or paused:
            self.stop()
        return {"mode": self.mode, "velocity": list(self.velocity),
                "input_expired": expired, "timeout_s": self.timeout}
