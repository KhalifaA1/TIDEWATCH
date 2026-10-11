"""Dependency-free monitoring server for the TIDEWATCH simulation.

The simulation writes its telemetry JSON atomically. This process only reads
telemetry and writes validated mission commands, keeping UI and node logic apart.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import uuid


from .controls import COMMANDS, validate_command


def dashboard_root() -> Path:
    """Find assets in either a source checkout or an installed ROS package."""
    source = Path(__file__).resolve().parent.parent / "dashboard"
    if source.is_dir():
        return source
    try:
        from ament_index_python.packages import get_package_share_directory
        installed = Path(get_package_share_directory("underwater_system")) / "dashboard"
        if installed.is_dir():
            return installed
    except (ImportError, LookupError):
        pass
    raise FileNotFoundError("TIDEWATCH dashboard assets are not installed")


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, state_path: Path, control_path: Path, static_root: Path):
        super().__init__(address, DashboardHandler)
        self.state_path = state_path
        self.control_path = control_path
        self.static_root = static_root.resolve()
        self.command_lock = threading.Lock()


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def log_message(self, format, *args):
        # Suppress frequent state polling, retaining failures for diagnostics.
        if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
            super().log_message(format, *args)

    def respond(self, status: int, content: bytes, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(content)

    def json_response(self, status: int, value):
        self.respond(status, json.dumps(value, allow_nan=False).encode("utf-8"))

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/api/state":
            try:
                raw = self.server.state_path.read_bytes()
                state = json.loads(raw)
                if not isinstance(state, dict):
                    raise ValueError("State must be a JSON object")
                state["dashboard"] = {
                    "connected": True,
                    "state_age_seconds": max(0.0, time.time() - self.server.state_path.stat().st_mtime),
                }
                self.json_response(200, state)
            except FileNotFoundError:
                self.json_response(503, {"error": "Waiting for simulation telemetry", "dashboard": {"connected": False}})
            except (OSError, ValueError, TypeError) as error:
                self.json_response(503, {"error": "Simulation telemetry is temporarily unavailable", "detail": str(error)})
            return
        if route == "/api/health":
            self.json_response(200, {"service": "tidewatch-dashboard", "state_available": self.server.state_path.is_file()})
            return
        filename = "index.html" if route == "/" else route.lstrip("/")
        candidate = (self.server.static_root / filename).resolve()
        if not candidate.is_relative_to(self.server.static_root) or not candidate.is_file():
            self.json_response(404, {"error": "Not found"})
            return
        try:
            mime = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
            if mime.startswith("text/") or mime == "application/javascript":
                mime += "; charset=utf-8"
            self.respond(200, candidate.read_bytes(), mime)
        except OSError:
            self.json_response(500, {"error": "Unable to load dashboard asset"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/control":
            self.json_response(404, {"error": "Not found"})
            return
        # This endpoint is local by default, and rejects browser requests from
        # other origins. Do not expose it on a public interface without auth.
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            self.json_response(403, {"error": "Cross-origin commands are not allowed"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            command = {**validate_command(payload), "nonce": uuid.uuid4().hex, "issued_at": time.time()}
            with self.server.command_lock:
                self.server.control_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.server.control_path.with_name(self.server.control_path.name + ".tmp")
                temporary.write_text(json.dumps(command), encoding="utf-8")
                os.replace(temporary, self.server.control_path)
            self.json_response(202, {"accepted": True, **command})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.json_response(400, {"error": str(error)})
        except OSError:
            self.json_response(500, {"error": "Unable to queue mission command"})


def main(args=None):
    parser = argparse.ArgumentParser(description="TIDEWATCH live monitoring station")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--state-file", type=Path, default=Path("/tmp/tidewatch-state.json"))
    parser.add_argument("--control-file", type=Path, default=Path("/tmp/tidewatch-control.json"))
    parser.add_argument("--assets", type=Path)
    options, _ = parser.parse_known_args(args)
    server = DashboardServer((options.host, options.port), options.state_file, options.control_file, options.assets or dashboard_root())
    print(f"TIDEWATCH monitoring station: http://{options.host}:{options.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
