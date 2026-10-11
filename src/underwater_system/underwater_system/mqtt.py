"""Small MQTT 3.1.1 QoS-0 publisher for the simulation's surface gateway.

No acoustic data is routed through MQTT. Only delivered, summarized surface
telemetry is published. This client reconnects after a broker outage.
"""
import json
import socket
import struct
import time


def _string(value):
    value = value.encode('utf-8')
    return struct.pack('!H', len(value)) + value


def _packet(header, data):
    size, remaining = len(data), bytearray()
    while True:
        byte = size % 128
        size //= 128
        remaining.append(byte | (128 if size else 0))
        if not size:
            return bytes([header]) + remaining + data


class MQTTPublisher:
    def __init__(self, host='127.0.0.1', port=1883):
        self.host, self.port = host, port
        self.sock = None
        self.connected = False
        self.retry_at = 0

    def publish(self, topic, value):
        if not self.sock and time.monotonic() < self.retry_at:
            return False
        try:
            if not self.sock:
                self.sock = socket.create_connection((self.host, self.port), timeout=0.15)
                self.sock.settimeout(0.15)
                self.sock.sendall(_packet(0x10, _string('MQTT') + bytes([4, 2]) +
                                          struct.pack('!H', 15) + _string('tidewatch-gateway')))
                reply = bytearray()
                while len(reply) < 4:
                    chunk = self.sock.recv(4 - len(reply))
                    if not chunk:
                        raise OSError('MQTT broker closed connection')
                    reply.extend(chunk)
                if reply != b'\x20\x02\x00\x00':
                    raise OSError('MQTT connection rejected')
            payload = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
            self.sock.sendall(_packet(0x30, _string(topic) + payload))
            self.connected = True
            return True
        except (OSError, ValueError):
            self.close()
            self.retry_at = time.monotonic() + 3
            return False

    def close(self):
        if self.sock:
            self.sock.close()
        self.sock = None
        self.connected = False
