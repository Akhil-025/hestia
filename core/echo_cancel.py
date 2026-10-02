# core/echo_cancel.py

"""
Basic acoustic echo cancellation for barge-in (backlog #175).

Barge-in listens on the microphone *while Hestia is speaking*, so on a laptop
her own voice leaks from the speakers into the mic and can register as the
user talking over her. ``barge_in.min_rms`` only mitigates that. This module
adds a real (if simple) adaptive-filter stage in front of barge-in's VAD:

* ``EchoReference`` — a thread-safe buffer of the audio that was actually sent
  to the speakers, resampled to the mic rate (16 kHz).
* ``NLMSEchoCanceller`` — a block normalised-LMS FIR filter that learns the
  speaker→mic path from that reference and subtracts the predicted echo from
  each 30 ms mic frame. Adaptation freezes when the mic is much louder than
  the reference (the user is talking — "double talk"), so the filter doesn't
  learn to cancel *you*.

Limits worth knowing (it is deliberately basic)
-----------------------------------------------
* **Piper only.** The reference is the PCM Hestia writes to the output
  stream, which only the Piper path exposes. pyttsx3 plays through the OS
  speech engine, so there is nothing to feed the filter; main.py logs that and
  leaves AEC off.
* **Delay is configurable, not measured.** The filter covers ``filter_len``
  samples (1024 taps = 64 ms at 16 kHz) starting ``delay_ms`` behind the
  newest reference sample. Output-device latency varies by machine; if the
  echo isn't being reduced, raise ``barge_in.echo_cancel.delay_ms``.
* **Not validated on real hardware** in this repo's test suite — the tests use
  synthetic echo. Treat it as an experiment you can turn on, and compare
  against ``min_rms`` tuning or, best of all, a headset.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Union

import numpy as np

MIC_RATE = 16000


class EchoReference:
    """Rolling record of the audio sent to the speakers (mono, MIC_RATE)."""

    def __init__(self, samplerate: int = MIC_RATE, max_seconds: float = 2.0) -> None:
        self.samplerate = samplerate
        self._cap = int(samplerate * max_seconds)
        self._buf = np.zeros(0, dtype=np.float64)
        self._lock = threading.Lock()
        self._last_push = 0.0

    def push(self, pcm: Union[bytes, np.ndarray], source_rate: int = 22050) -> None:
        """Append int16 PCM (raw bytes or an array) played at *source_rate*."""
        if isinstance(pcm, (bytes, bytearray)):
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        else:
            samples = np.asarray(pcm).reshape(-1).astype(np.float64)
        if samples.size == 0:
            return
        if source_rate != self.samplerate:
            n_out = max(1, int(round(samples.size * self.samplerate / source_rate)))
            x_old = np.linspace(0.0, 1.0, samples.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
            samples = np.interp(x_new, x_old, samples)
        with self._lock:
            buf = np.concatenate([self._buf, samples])
            if buf.size > self._cap:
                buf = buf[-self._cap:]
            self._buf = buf
            self._last_push = time.monotonic()

    def tail(self, n: int) -> np.ndarray:
        """The newest *n* samples, zero-padded on the left if fewer exist."""
        with self._lock:
            buf = self._buf
            if buf.size >= n:
                return buf[-n:].copy()
            out = np.zeros(n, dtype=np.float64)
            if buf.size:
                out[-buf.size:] = buf
            return out

    def is_active(self, stale_after: float = 1.0) -> bool:
        """True if audio was pushed within the last *stale_after* seconds —
        i.e. Hestia is (probably) still talking. Outside that window the
        canceller passes the mic through untouched."""
        with self._lock:
            return self._last_push > 0 and (time.monotonic() - self._last_push) <= stale_after

    def clear(self) -> None:
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float64)
            self._last_push = 0.0


class NLMSEchoCanceller:
    """Block-NLMS echo canceller. ``process()`` takes one int16 mic frame
    and returns the echo-reduced int16 frame of the same length."""

    def __init__(
        self,
        reference: EchoReference,
        filter_len: int = 1024,
        mu: float = 0.4,
        delay_ms: float = 0.0,
        double_talk_ratio: float = 1.0,
        stale_after: float = 1.0,
        eps: float = 1.0,
    ) -> None:
        self.reference = reference
        self.filter_len = max(8, int(filter_len))
        self.mu = float(mu)
        self.delay = max(0, int(reference.samplerate * delay_ms / 1000.0))
        self.double_talk_ratio = float(double_talk_ratio)
        self.stale_after = stale_after
        self.eps = eps
        self.w = np.zeros(self.filter_len, dtype=np.float64)

    def reset(self) -> None:
        self.w[:] = 0.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        d = np.asarray(frame).reshape(-1)
        n = d.size
        if n == 0 or not self.reference.is_active(self.stale_after):
            return np.asarray(frame)

        L = self.filter_len
        d64 = d.astype(np.float64)
        total = n + L - 1
        u = self.reference.tail(total + self.delay)[:total]

        y = np.convolve(u, self.w, mode="valid")            # length n
        e = d64 - y

        u_peak = float(np.max(np.abs(u))) if u.size else 0.0
        double_talk = float(np.max(np.abs(d64))) > self.double_talk_ratio * max(u_peak, 1.0)
        if u_peak > 0.0 and not double_talk:
            sigma2 = float(np.mean(u ** 2))
            grad = np.correlate(u, e, mode="valid")[::-1]   # length L
            # Summed (not averaged) block gradient, normalised by the energy
            # of everything the update touches: stable for 0 < mu < 2 and
            # converges in a handful of frames rather than hundreds.
            self.w += self.mu * grad / ((L + n) * sigma2 + self.eps)

        return np.clip(np.rint(e), -32768, 32767).astype(np.int16)


def echo_return_loss_db(mic: np.ndarray, cleaned: np.ndarray) -> float:
    """ERLE-style figure: how many dB quieter *cleaned* is than *mic*.
    Positive = echo reduced. Handy for tests and for tuning delay_ms."""
    m = float(np.mean(np.asarray(mic, dtype=np.float64) ** 2))
    c = float(np.mean(np.asarray(cleaned, dtype=np.float64) ** 2))
    if m <= 0.0:
        return 0.0
    return 10.0 * float(np.log10(m / max(c, 1e-12)))
