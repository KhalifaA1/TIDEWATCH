"""Read a short local hydrophone WAV clip for the existing acoustic pipeline."""
from __future__ import annotations

import math
from pathlib import Path
import wave

import numpy as np


class HydrophoneRecording:
    """PCM16 audio indexed by clip time, without classification or SPL labels.

    ``frame(now)`` returns 1024 samples by default. ``now`` is seconds from the
    clip's start. Playback pads with zeros after the end unless ``loop=True``.
    Multichannel files are averaged. Convert files to the configured sample
    rate before loading; this adapter does not resample or normalize loudness.
    """

    def __init__(self, path, sample_rate=8000, max_seconds=60, *, loop=False,
                 frame_samples=1024):
        self.path = Path(path).expanduser()
        self.sample_rate = int(sample_rate)
        self.frame_samples = int(frame_samples)
        max_seconds = float(max_seconds)
        if self.sample_rate != sample_rate or not 1 <= self.sample_rate <= 192000:
            raise ValueError("Recording sample rate must be an integer from 1 to 192000 Hz")
        if self.frame_samples != frame_samples or not 32 <= self.frame_samples <= 1024:
            raise ValueError("Recording frames must contain 32 to 1024 samples")
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")
        if not isinstance(loop, bool):
            raise ValueError("Recording loop must be True or False")
        if not self.path.is_file():
            raise FileNotFoundError(f"Hydrophone WAV not found: {self.path}")
        if self.path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Hydrophone WAV exceeds 64 MiB; convert a shorter clip first")
        self.loop = loop
        try:
            with wave.open(str(self.path), "rb") as reader:
                channels, rate = reader.getnchannels(), reader.getframerate()
                if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
                    raise ValueError("Hydrophone WAV must be uncompressed PCM16; convert with ffmpeg -c:a pcm_s16le")
                if rate != self.sample_rate:
                    raise ValueError(f"Hydrophone WAV is {rate} Hz; expected {self.sample_rate} Hz. "
                                     f"Convert with ffmpeg -i input.wav -ar {self.sample_rate} -c:a pcm_s16le output.wav")
                if not 1 <= channels <= 16:
                    raise ValueError("Hydrophone WAV must have 1 to 16 channels")
                count = reader.getnframes()
                if count <= 0:
                    raise ValueError("Hydrophone WAV has no samples")
                if count / rate > max_seconds:
                    raise ValueError(f"Hydrophone WAV exceeds {max_seconds:g} seconds; "
                                     f"convert a short clip with ffmpeg -i input.wav -t {max_seconds:g} "
                                     f"-ar {self.sample_rate} -c:a pcm_s16le output.wav")
                if count > 8_000_000 or count * channels * 2 > 64 * 1024 * 1024:
                    raise ValueError("Hydrophone WAV has too many samples; convert a shorter clip first")
                data = reader.readframes(count)
                if len(data) != count * channels * 2:
                    raise ValueError("Hydrophone WAV is truncated")
        except (wave.Error, EOFError) as error:
            raise ValueError("Cannot read hydrophone WAV; convert it to uncompressed PCM16 first") from error
        pcm = np.frombuffer(data, dtype="<i2").reshape(-1, channels)
        self._samples = pcm.mean(axis=1, dtype=np.float64) / 32768.0
        self.duration = count / self.sample_rate
        self._metadata = {"source": "recorded", "name": self.path.name,
                          "sample_rate_hz": self.sample_rate, "file_rate_hz": rate,
                          "channels": channels, "duration_s": self.duration,
                          "clip_seconds": self.duration, "loop": self.loop,
                          "amplitude_units": "normalized_pcm", "offset_s": 0.0,
                          "frame_samples": self.frame_samples, "exhausted": False,
                          "looped": False, "padded_samples": 0}

    @property
    def metadata(self) -> dict:
        """Clip details and the last frame's position; never a source label."""
        return dict(self._metadata)

    def frame(self, now: float, duration: float | None = None) -> list[float]:
        """Return a bounded frame at a deterministic offset in the clip."""
        now = float(now)
        if not math.isfinite(now) or now < 0:
            raise ValueError("Recording time must be finite and nonnegative")
        count = self.frame_samples
        if duration is not None:
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Frame duration must be finite and positive")
            samples = duration * self.sample_rate
            if samples < 31.5 or samples > 1024.5:
                raise ValueError("Frame duration must produce 32 to 1024 samples")
            count = int(round(samples))
            if not 32 <= count <= 1024:
                raise ValueError("Frame duration must produce 32 to 1024 samples")
        available = len(self._samples)
        exhausted, looped, padded = False, False, 0
        if self.loop:
            offset = min(int(math.floor((now % self.duration) * self.sample_rate)), available - 1)
            indices = (offset + np.arange(count)) % available
            result = self._samples[indices]
            looped = now >= self.duration or offset + count > available
        else:
            offset = available if now >= self.duration else int(math.floor(now * self.sample_rate))
            length = min(count, available - offset)
            result = np.zeros(count, dtype=np.float64)
            result[:length] = self._samples[offset:offset + length]
            exhausted, padded = offset >= available, count - length
        self._metadata.update(offset_s=offset / self.sample_rate, frame_samples=count,
                              exhausted=exhausted, looped=looped, padded_samples=padded)
        return result.tolist()
