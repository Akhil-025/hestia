# core/wake_word.py

import os
import queue
import time
import json
import sounddevice as sd
import vosk
from core.event_bus import bus


class WakeWordDetector:
    """Vosk-based wake word detection for phrases like 'hey hestia'."""

    def __init__(self, model_path: str = "models/vosk-model-small-en-us-0.15",
                 wake_words: list = None):
        """Initialize Vosk model, recognizer, audio queue, and event bus."""
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

        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.q = queue.Queue()

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

                    words = text.split()
                    if len(words) > 4:
                        continue

                    # Match exact or prefix (allow trailing partial words)
                    matched = False
                    for ww in self.wake_words:
                        if text == ww or text.startswith(ww):
                            matched = True
                            break

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