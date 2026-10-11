"""Dashboard HTTP controls, including restricted environments without sockets."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "underwater_system"))
from underwater_system.dashboard import DashboardHandler, DashboardServer, dashboard_root


class DashboardControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.control = root / "control.json"
        try:
            self.server = DashboardServer(("127.0.0.1", 0), root / "state.json", self.control, dashboard_root())
        except PermissionError:
            # The same handler can parse HTTP bytes without a listening socket.
            self.server = SimpleNamespace(state_path=root/'state.json', control_path=self.control,
                                          static_root=dashboard_root(), command_lock=threading.Lock())
            self.thread = None
            self.url = 'http://127.0.0.1:8080'
            return
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        if self.thread is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()
        self.temporary.cleanup()

    def post(self, payload, origin=None):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        if self.thread is None:
            body = json.dumps(payload).encode()
            headers.update(Host='127.0.0.1:8080', **{'Content-Length':str(len(body))})
            raw = ('POST /api/control HTTP/1.0\r\n' +
                   ''.join(f'{key}: {value}\r\n' for key,value in headers.items()) + '\r\n').encode()+body
            class RequestStream:
                def __init__(self):
                    self.input = io.BytesIO(raw)
                    self.output = bytearray()
                def makefile(self, *args):
                    return self.input
                def sendall(self, content):
                    self.output.extend(content)
            stream = RequestStream()
            DashboardHandler(stream, ('127.0.0.1',0), self.server)
            head, data = bytes(stream.output).split(b'\r\n\r\n',1)
            return int(head.split()[1]), json.loads(data)
        request = Request(self.url + "/api/control", data=json.dumps(payload).encode(), headers=headers)
        try:
            response = urlopen(request, timeout=3)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read())

    def test_current_vector_preserved_with_unique_command_envelope(self):
        for vector in ([0, 0, 0], [0.25, -0.15, 0], [-0.8, 0.8, 0.1]):
            with self.subTest(vector=vector):
                status, response = self.post({"command": "set_current", "vector": vector})
                self.assertEqual(status, 202)
                saved = json.loads(self.control.read_text())
                self.assertEqual(saved["vector"], vector)
                self.assertEqual(saved["command"], "set_current")
                self.assertEqual(saved["nonce"], response["nonce"])
                self.assertIsInstance(saved["issued_at"], float)
                self.assertEqual(len(saved["nonce"]), 32)

    def test_invalid_current_does_not_replace_accepted_command(self):
        self.post({"command": "pause"})
        previous = self.control.read_bytes()
        invalid = [None, [], [0, 0], [0, 0, 0, 0], "0,0,0", [True, 0, 0],
                   ["0.2", 0, 0], [None, 0, 0], [[0], 0, 0], [float("nan"), 0, 0],
                   [float("inf"), 0, 0], [10**400, 0, 0], [0.8001, 0, 0], [0, -0.81, 0], [0, 0, 1]]
        for vector in invalid:
            with self.subTest(vector=vector):
                status, response = self.post({"command": "set_current", "vector": vector})
                self.assertEqual(status, 400)
                self.assertIn("error", response)
                self.assertEqual(self.control.read_bytes(), previous)

    def test_current_control_retains_same_origin_requirement(self):
        status, _ = self.post({"command": "set_current", "vector": [0, 0, 0]}, "http://example.org")
        self.assertEqual(status, 403)
        self.assertFalse(self.control.exists())
        status, _ = self.post({"command": "set_current", "vector": [0, 0, 0]}, self.url)
        self.assertEqual(status, 202)

    def test_existing_controls_keep_the_command_contract(self):
        previous_nonce = None
        for command in ("start", "pause", "reset", "toggle_outage"):
            status, response = self.post({"command": command})
            self.assertEqual(status, 202)
            self.assertNotEqual(previous_nonce, response["nonce"])
            self.assertNotIn("vector", json.loads(self.control.read_text()))
            previous_nonce = response["nonce"]

    def test_mode_switch_and_manual_pilot_are_queued_with_fields(self):
        for mode in ('protect', 'explore'):
            status, response = self.post({'command':'set_mode', 'mode':mode})
            self.assertEqual(status, 202)
            self.assertEqual(response['mode'], mode)
            self.assertEqual(json.loads(self.control.read_text())['mode'], mode)
        status, response = self.post({'command':'set_diver_control', 'mode':'manual', 'velocity':[.4,0,-.3]})
        self.assertEqual(status, 202)
        self.assertEqual(response['velocity'], [.4,0,-.3])
        previous = self.control.read_bytes()
        for payload in ({'command':'set_mode','mode':[]},
                        {'command':'set_mode','mode':'invalid'},
                        {'command':'set_diver_control','mode':'manual','velocity':[.65,.65,0]}):
            status, _ = self.post(payload)
            self.assertEqual(status, 400)
            self.assertEqual(self.control.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
