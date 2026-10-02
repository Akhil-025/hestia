# core/wake_word.py

import os
import queue
import time
import json
import sys
from difflib import SequenceMatcher
import sounddevice as sd
import vosk
from core.event_bus import bus


# ---------------------------------------------------------------------------
# Sensitivity presets (backlog #176)
# ---------------------------------------------------------------------------
# Vosk gives text, not a confidence "threshold" knob, so sensitivity is built
# from three levers applied to each recognised utterance:
#
#   fuzzy_threshold      — accept near-miss spellings of a wake word (how
#                          speech recognisers actually mis-hear "hestia") when
#                          every word's similarity ratio is at least this.
#                          None = exact token match only. Only wake words of
#                          5+ letters are fuzzy-matched; short ones would
#                          false-trigger.
#   min_word_conf        — require Vosk's own per-word confidence for the
#                          matched words to be at least this (0 = ignore).
#   max_utterance_tokens — ignore utterances longer than this many words, so
#                          TV/podcast chatter that happens to contain "hestia"
#                          mid-sentence doesn't trigger (None = no limit).
#
#   quiet  — quiet room: be generous, catch soft/mis-heard wake words.
#   normal — the original behaviour: exact match, anywhere in the utterance.
#   noisy  — noisy room: be strict, fewer false wakes from background audio.
SENSITIVITY_PRESETS = {
    "quiet":  {"fuzzy_threshold": 0.85, "min_word_conf": 0.0,  "max_utterance_tokens": None},
    "normal": {"fuzzy_threshold": None, "min_word_conf": 0.0,  "max_utterance_tokens": None},
    "noisy":  {"fuzzy_threshold": None, "min_word_conf": 0.75, "max_utterance_tokens": 6},
}
DEFAULT_SENSITIVITY = "normal"
_SENSITIVITY_ALIASES = {
    "high": "quiet", "sensitive": "quiet", "low": "noisy", "strict": "noisy",
    "default": "normal", "medium": "normal",
}
_FUZZY_MIN_WORD_LEN = 5


def normalise_sensitivity(name) -> str:
    """Map a user-supplied level to a preset name; unknown -> "normal"."""
    key = str(name or "").strip().lower()
    key = _SENSITIVITY_ALIASES.get(key, key)
    return key if key in SENSITIVITY_PRESETS else DEFAULT_SENSITIVITY


class WakeWordDetector:
    """Vosk-based wake word detection for phrases like 'hey hestia'."""

    def __init__(self, model_path: str = "models/vosk-model-small-en-us-0.15",
                 wake_words: list = None, sensitivity: str = DEFAULT_SENSITIVITY):
        """Initialize Vosk model, recognizer, audio queue, and event bus.

        sensitivity: "quiet", "normal" or "noisy" (see SENSITIVITY_PRESETS).
        """
        if wake_words is None:
            wake_words = [
                "hestia",
                "hey hestia",
                "hastia",
                "hey hastia",
                "estia",
                "hey estia",
                "hasta",
                "hey hasta",
            ]
        self.wake_words = [w.lower() for w in wake_words]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at '{model_path}'")

        try:
            self.model = vosk.Model(model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Vosk model at {model_path!r}. "
                "The model may be corrupted or the wrong version. "
                "Re-download from https://alphacephei.com/vosk/models"
            ) from exc
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.q = queue.Queue()

        requested = str(sensitivity or "").strip().lower()
        self.set_sensitivity(sensitivity)
        if requested and requested not in SENSITIVITY_PRESETS \
                and requested not in _SENSITIVITY_ALIASES:
            print(
                f"[Wake] Unknown sensitivity {sensitivity!r}; using "
                f"'{DEFAULT_SENSITIVITY}'. Valid: {', '.join(SENSITIVITY_PRESETS)}.",
                file=sys.stderr,
            )

    def set_sensitivity(self, level: str) -> str:
        """Switch sensitivity preset at runtime ("I'm in a noisy room").
        Returns the preset name actually applied."""
        self.sensitivity = normalise_sensitivity(level)
        preset = SENSITIVITY_PRESETS[self.sensitivity]
        self._fuzzy_threshold = preset["fuzzy_threshold"]
        self._min_word_conf = preset["min_word_conf"]
        self._max_tokens = preset["max_utterance_tokens"]
        # Per-word confidences are only produced when asked for, and only
        # the strict preset uses them.
        try:
            self.recognizer.SetWords(self._min_word_conf > 0)
        except Exception:
            pass
        return self.sensitivity

    def _match_window(self, text_tokens: list, res: dict):
        """Return the start index of the first wake-word window in
        *text_tokens* that passes this preset's rules, or None."""
        if self._max_tokens is not None and len(text_tokens) > self._max_tokens:
            return None

        for ww in self.wake_words:
            ww_tokens = ww.split()
            n = len(ww_tokens)
            for i in range(len(text_tokens) - n + 1):
                window = text_tokens[i:i + n]
                exact = window == ww_tokens
                fuzzy = (
                    not exact
                    and self._fuzzy_threshold is not None
                    and len(ww) >= _FUZZY_MIN_WORD_LEN
                    # Word by word, not on the joined phrase: a shared "hey "
                    # would otherwise inflate the ratio ("hey asia" scores
                    # ~0.89 against "hey hastia" as a whole string).
                    and all(
                        SequenceMatcher(None, got, want).ratio() >= self._fuzzy_threshold
                        for got, want in zip(window, ww_tokens)
                    )
                )
                if not (exact or fuzzy):
                    continue
                if self._confidence_ok(res, i, n, len(text_tokens)):
                    return i
        return None

    def _confidence_ok(self, res: dict, start: int, n: int, total: int) -> bool:
        """Apply the min_word_conf rule. If Vosk gave no per-word results
        (or they don't line up with the text), there is nothing to gate on,
        so the match stands."""
        if self._min_word_conf <= 0:
            return True
        words = res.get("result") if isinstance(res, dict) else None
        if not isinstance(words, list) or len(words) != total:
            return True
        confs = [w.get("conf") for w in words[start:start + n] if isinstance(w, dict)]
        confs = [c for c in confs if isinstance(c, (int, float))]
        return not confs or min(confs) >= self._min_word_conf

    def _audio_callback(self, indata, frames, time_, status):
        """Callback for sounddevice to push audio bytes into the queue."""
        self.q.put(bytes(indata))

    def flush_audio_queue(self) -> None:
        """Drain all pending audio from the queue."""
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except Exception:
                break

    def listen_for_wake_word(self, timeout: float = None) -> bool:
        """Listen for wake word until timeout, emit event on detection, return bool."""
        self.flush_audio_queue()
        start_time = time.time()

        print("Listening for wake word...")

        with sd.RawInputStream(samplerate=16000, blocksize=8000,
                               dtype='int16', channels=1,
                               callback=self._audio_callback):
            while True:
                if timeout and (time.time() - start_time) > timeout:
                    return False

                try:
                    data = self.q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self.recognizer.AcceptWaveform(data):
                    result = self.recognizer.Result()
                    try:
                        res = json.loads(result)
                    except Exception:
                        continue

                    text = res.get('text', '').lower().strip() if isinstance(res, dict) else ''
                    if not text:
                        continue

                    print(f"[Wake] Heard: {text}")

                    # Check if any wake word appears anywhere in the heard text
                    # (sliding window match), rather than rejecting longer
                    # utterances outright — a wake phrase can appear
                    # mid-sentence. The sensitivity preset decides how
                    # strict that match is (see SENSITIVITY_PRESETS).
                    matched = self._match_window(text.split(), res) is not None

                    if matched:
                        self.flush_audio_queue()
                        bus.emit("wake_detected", {"text": text})
                        return True

        return False


if __name__ == "__main__":
    detector = WakeWordDetector()
    print("Say 'Hey Hestia' to test...")
    try:
        result = detector.listen_for_wake_word(timeout=15)
        if result:
            print("Wake word detected!")
        else:
            print("Timed out.")
    except KeyboardInterrupt:
        print("Stopped.")