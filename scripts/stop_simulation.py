#!/usr/bin/env python3
"""Stop only the TIDEWATCH supervisor and its child processes."""
import os
from pathlib import Path
import signal

path = Path(__file__).resolve().parents[1]/'.runtime/run.pid'
if path.is_file():
    try:
        os.kill(int(path.read_text()),signal.SIGTERM)
        print('Stopping TIDEWATCH.')
    except ProcessLookupError:
        path.unlink(missing_ok=True)
        print('TIDEWATCH is already stopped.')
else:
    print('TIDEWATCH is not running.')
