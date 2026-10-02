# core/mic_calibration.py

"""
Per-device microphone calibration (backlog #174).

``barge_in.min_rms`` and the VAD aggressiveness settings used to be tuned by
hand ("raise this if Hestia interrupts herself, lower it if she misses you").
This module measures the device instead, in three short steps:

1. **Quiet** — record a few seconds of the room with nobody talking, which
   gives the ambient noise floor.
2. **Speech** — you say a sentence at normal volume, which gives the level a
   real interruption has to clear.
3. **Echo** (optional, when a TTS engine is supplied) — Hestia speaks a test
   phrase while the mic records, which measures how loudly her own voice
   bleeds back in on *this* speaker/mic pair. This is the number that matters
   for the self-interruption problem.

From those it recommends ``barge_in.min_rms``, ``barge_in.vad_aggressiveness``
and ``wake_word.sensitivity`` and saves them to ``data/mic_calibration.json``
together with the input device's name. ``main.py`` reads that file at startup
and uses the values for any setting you have NOT pinned in the config, so an
explicit config value always wins, and a calibration made on a different mic
(different device name) is ignored rather than misapplied.

Run it with ``python main.py --calibrate-mic`` (or
``python -m core.mic_calibration``).

The recommendation maths (``recommend_settings``) is pure and unit-tested; the
recording helpers are thin wrappers over sounddevice and take an injectable
recorder so the whole flow can be tested without a microphone.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

DEFAULT_PATH = os.path.join("data", "mic_calibration.json")
FRAME = 480            # 30 ms at 16 kHz — what webrtcvad requires
SAMPLERATE = 16000

# int16-scale RMS bounds for the recommended barge-in floor.
MIN_RMS_FLOOR = 100.0
MIN_RMS_CEILING = 3000.0
# Headroom multipliers over the measured noise / echo levels.
NOISE_HEADROOM = 2.0
ECHO_HEADROOM = 1.25
# Fraction of the (quiet end of) speech level the floor must stay under, or a
# normal-volume interruption would be ignored.
SPEECH_CEILING_FRACTION = 0.8


@dataclass
class CalibrationResult:
    device: str = ""
    created: float = 0.0
    noise_rms_mean: float = 0.0
    noise_rms_p95: float = 0.0
    speech_rms_p25: Optional[float] = None
    echo_rms_p95: Optional[float] = None
    min_rms: float = 300.0
    vad_aggressiveness: int = 2
    wake_sensitivity: str = "normal"
    notes: list = field(default_factory=list)

    def suggested(self) -> dict:
        return {
            "min_rms": self.min_rms,
            "vad_aggressiveness": self.vad_aggressiveness,
            "wake_sensitivity": self.wake_sensitivity,
        }


# ---------------------------------------------------------------------------
# Pure maths
# ---------------------------------------------------------------------------

def frame_rms_values(audio: np.ndarray, frame: int = FRAME) -> list[float]:
    """Per-frame RMS of an int16-scale mono array."""
    audio = np.asarray(audio).reshape(-1)
    out: list[float] = []
    for i in range(0, len(audio) - frame + 1, frame):
        chunk = audio[i:i + frame].astype(np.float64)
        out.append(float(np.sqrt(np.mean(chunk ** 2))))
    return out


def _pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _speech_active_p25(speech: Sequence[float], noise_p95: float) -> Optional[float]:
    """25th percentile of the frames that are actually speech. The speech
    recording includes pauses and tails, so frames near the noise floor are
    excluded before taking the percentile."""
    active = [v for v in speech if v > max(noise_p95 * 1.5, 1.0)]
    if len(active) < 3:
        return None
    return _pct(active, 25)


def recommend_settings(
    noise_rms: Sequence[float],
    speech_rms: Optional[Sequence[float]] = None,
    echo_rms: Optional[Sequence[float]] = None,
    device: str = "",
) -> CalibrationResult:
    """Turn measured per-frame RMS lists into recommended settings.

    *noise_rms* is required; *speech_rms* and *echo_rms* sharpen the answer
    when available. Never raises on odd input — an empty noise list falls
    back to defaults with a note.
    """
    notes: list[str] = []
    result = CalibrationResult(device=device, created=time.time())

    if not noise_rms:
        notes.append("No quiet-room audio was captured; keeping default settings.")
        result.notes = notes
        return result

    noise_mean = float(np.mean(noise_rms))
    noise_p95 = _pct(noise_rms, 95)
    result.noise_rms_mean = round(noise_mean, 1)
    result.noise_rms_p95 = round(noise_p95, 1)

    candidates = [noise_p95 * NOISE_HEADROOM]
    if echo_rms:
        echo_p95 = _pct(echo_rms, 95)
        result.echo_rms_p95 = round(echo_p95, 1)
        candidates.append(echo_p95 * ECHO_HEADROOM)
    min_rms = max(MIN_RMS_FLOOR, *candidates)

    if speech_rms:
        speech_p25 = _speech_active_p25(speech_rms, noise_p95)
        if speech_p25 is None:
            notes.append(
                "Your speech was too quiet to measure against the room noise; "
                "try again closer to the mic."
            )
        else:
            result.speech_rms_p25 = round(speech_p25, 1)
            ceiling = speech_p25 * SPEECH_CEILING_FRACTION
            if min_rms > ceiling:
                notes.append(
                    "Room noise / Hestia's own echo is nearly as loud as your "
                    "voice, so no amplitude floor can separate them cleanly. "
                    "A headset is the reliable fix."
                )
                min_rms = max(MIN_RMS_FLOOR, ceiling)

    min_rms = float(min(MIN_RMS_CEILING, max(MIN_RMS_FLOOR, min_rms)))
    result.min_rms = round(min_rms, 0)

    # Noisier rooms want a stricter VAD; webrtcvad 2 is right for quiet ones.
    result.vad_aggressiveness = 2 if noise_p95 < 250 else 3

    if noise_p95 < 150:
        result.wake_sensitivity = "quiet"
    elif noise_p95 < 500:
        result.wake_sensitivity = "normal"
    else:
        result.wake_sensitivity = "noisy"

    result.notes = notes
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def current_input_device_name() -> str:
    """Name of the default input device, or "" if it can't be determined."""
    try:
        import sounddevice as sd
        info = sd.query_devices(kind="input")
        if isinstance(info, dict):
            return str(info.get("name", ""))
    except Exception:
        pass
    return ""


def save_calibration(result: CalibrationResult, path: str = DEFAULT_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh, indent=2)
    return path


def load_calibration(
    path: str = DEFAULT_PATH, device: Optional[str] = None
) -> Optional[dict]:
    """Load saved suggestions, or None if there's no usable calibration.

    If *device* is given and the saved calibration was made on a different
    (non-empty) device name, it is ignored — thresholds measured on a laptop
    mic say nothing about a USB headset. Never raises.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    saved_device = data.get("device") or ""
    if device and saved_device and saved_device != device:
        return None
    try:
        return {
            "min_rms": float(data["min_rms"]),
            "vad_aggressiveness": int(data["vad_aggressiveness"]),
            "wake_sensitivity": str(data.get("wake_sensitivity", "normal")),
        }
    except (KeyError, TypeError, ValueError):
        return None


def format_report(result: CalibrationResult) -> str:
    lines = [
        "Mic calibration" + (f" — {result.device}" if result.device else ""),
        f"  noise floor      mean {result.noise_rms_mean}, p95 {result.noise_rms_p95}",
    ]
    if result.speech_rms_p25 is not None:
        lines.append(f"  your voice       p25 {result.speech_rms_p25}")
    if result.echo_rms_p95 is not None:
        lines.append(f"  Hestia's echo    p95 {result.echo_rms_p95}")
    lines += [
        "",
        "Recommended settings (used automatically unless your config sets them):",
        "  barge_in:",
        f"    min_rms: {result.min_rms:g}",
        f"    vad_aggressiveness: {result.vad_aggressiveness}",
        "  wake_word:",
        f"    sensitivity: {result.wake_sensitivity}",
    ]
    for note in result.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recording (hardware-bound; injectable for tests)
# ---------------------------------------------------------------------------

Recorder = Callable[[float], list]


def record_rms(seconds: float, samplerate: int = SAMPLERATE) -> list[float]:
    """Record *seconds* from the default mic and return per-frame RMS."""
    import sounddevice as sd

    n_frames = max(1, int(seconds * samplerate / FRAME))
    values: list[float] = []
    with sd.InputStream(
        samplerate=samplerate, channels=1, dtype="int16", blocksize=FRAME
    ) as stream:
        for _ in range(n_frames):
            data, _ = stream.read(FRAME)
            values.extend(frame_rms_values(np.asarray(data).reshape(-1), FRAME))
    return values


def run_calibration(
    tts=None,
    recorder: Recorder = record_rms,
    say: Callable[[str], None] = print,
    wait: Callable[[str], object] = input,
    quiet_seconds: float = 3.0,
    speech_seconds: float = 4.0,
    echo_seconds: float = 4.0,
    echo_phrase: str = "This is a short test of my voice, played through the speakers.",
    device: Optional[str] = None,
) -> CalibrationResult:
    """Interactive three-step calibration. Returns the recommendation (not
    saved — the caller decides). *recorder*, *say* and *wait* are injectable
    so tests can drive it without hardware or a terminal."""
    if device is None:
        device = current_input_device_name()

    say("Step 1/3 — stay quiet. Measuring the room...")
    wait("Press Enter, then keep silent for a few seconds: ")
    noise = recorder(quiet_seconds)

    say("Step 2/3 — speak normally.")
    wait("Press Enter, then say a sentence at your normal volume: ")
    speech = recorder(speech_seconds)

    echo = None
    if tts is not None:
        say("Step 3/3 — Hestia will speak while the mic listens (stay quiet).")
        wait("Press Enter to start the speaker test: ")
        try:
            tts.speak(echo_phrase)
            echo = recorder(echo_seconds)
        finally:
            try:
                tts.stop()
            except Exception:
                pass
    else:
        say("Step 3/3 — skipped (no TTS engine available for the echo test).")

    return recommend_settings(noise, speech, echo, device=device or "")


def main(argv: Optional[list] = None) -> int:
    """Standalone entry point: ``python -m core.mic_calibration``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    path = DEFAULT_PATH
    if "--path" in argv:
        path = argv[argv.index("--path") + 1]
    try:
        result = run_calibration(tts=None)
    except (KeyboardInterrupt, EOFError):
        print("\nCalibration cancelled.")
        return 1
    except Exception as exc:  # no mic, PortAudio missing, ...
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1
    print()
    print(format_report(result))
    save_calibration(result, path)
    print(f"\nSaved to {path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
