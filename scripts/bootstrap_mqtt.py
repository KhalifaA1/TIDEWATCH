#!/usr/bin/env python3
"""Extract Ubuntu's MQTT broker into .tools; no sudo or system changes."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
destination = root/'.tools/mqtt'
destination.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='tidewatch-mqtt-') as work:
    subprocess.run(['apt','download','mosquitto','libmosquitto1','libdlt3','libcjson1','libwebsockets19t64'],
                   cwd=work,check=True)
    for package in Path(work).glob('*.deb'):
        subprocess.run(['dpkg-deb','-x',str(package),str(destination)],check=True)
print(f'MQTT broker installed locally in {destination}')
