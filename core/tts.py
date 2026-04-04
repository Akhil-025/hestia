# core/tts.py

import os
import sys
import threading
import queue
import subprocess
import pyttsx3
import sounddevice as sd


class HestiaTTS:
    """Text-to-speech module with queue-based non-blocking interface."""

    def __init__(
        self,
        engine: str = "pyttsx3",
        rate: int = 175,
        volume: float = 1.0,
        piper_model_path: str = None,
    ):
        """
        Initialize TTS engine.

        Args:
            engine: "pyttsx3" or "piper"
            rate: Speech rate in words per minute (pyttsx3)
            volume: Volume level 0.0-1.0
            piper_model_path: Path to Piper TTS model file (required if engine="piper")
        """
        self.rate = rate
        self.volume = max(0.0, min(volume, 1.0))
        self._queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        # Determine engine and voice
        self._engine = "pyttsx3"
        self._voice_id = None
        self._piper_model_path = None

        # Detect pyttsx3 voice (always needed for fallback)
        temp_engine = pyttsx3.init()
        voices = temp_engine.getProperty("voices")
        selected_voice = None
        for voice in voices:
            name = voice.name.lower()
            if "zira" in name:
                self._voice_id = voice.id
                selected_voice = voice.name
                break
            elif "hazel" in name:
                self._voice_id = voice.id
                selected_voice = voice.name
                break
            elif "female" in name:
                self._voice_id = voice.id
                selected_voice = voice.name
                break
        if not self._voice_id and voices:
            self._voice_id = voices[0].id
            selected_voice = voices[0].name
        print(f"Selected pyttsx3 voice: {selected_voice}")

        # Optional Piper engine
        if engine == "piper" and piper_model_path and os.path.exists(piper_model_path):
            self._engine = "piper"
            self._piper_model_path = piper_model_path
            print(f"TTS engine: piper (model: {piper_model_path})")
        else:
            if engine == "piper":
                print(
                    "Piper model not found, falling back to pyttsx3",
                    file=sys.stderr,
                )
            print("TTS engine: pyttsx3")

    def speak(self, text: str) -> None:
        """Queue text for speech (non-blocking)."""
        if not text or not text.strip():
            return
        self._queue.put(text)

    def wait_until_done(self) -> None:
        """Block until all queued speech has finished."""
        self._queue.join()

    def set_rate(self, rate: int) -> None:
        """Set speech rate (pyttsx3 only)."""
        self.rate = rate

    def set_volume(self, volume: float) -> None:
        """Set volume level 0.0-1.0 (pyttsx3 only)."""
        self.volume = max(0.0, min(volume, 1.0))

    def _worker_loop(self) -> None:
        """Daemon worker thread consuming queue."""
        while True:
            text = self._queue.get()
            self._speak_blocking(text)
            self._queue.task_done()

    def _speak_blocking(self, text: str) -> None:
        """Internal blocking speech call."""
        if self._engine == "piper":
            try:
                self._speak_piper(text)
                return
            except Exception as e:
                print(f"Piper TTS failed: {e}, falling back to pyttsx3", file=sys.stderr)
        self._speak_pyttsx3(text)

    def _speak_pyttsx3(self, text: str) -> None:
        """Speak using pyttsx3 (fresh engine per call)."""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"pyttsx3 TTS error: {e}", file=sys.stderr)

    def _speak_piper(self, text: str) -> None:
        """Speak using Piper TTS via subprocess and sounddevice."""
        # Launch piper process
        proc = subprocess.Popen(
            [
                "piper",
                "--model",
                self._piper_model_path,
                "--output-raw",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        # Send text to stdin
        proc.stdin.write(text.encode("utf-8"))
        proc.stdin.close()
        # Read raw audio data
        audio_data = proc.stdout.read()
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"Piper exited with code {proc.returncode}")

        # Play via sounddevice (22050 Hz, mono, int16)
        stream = sd.RawOutputStream(
            samplerate=22050,
            channels=1,
            dtype="int16",
            blocksize=2048,
        )
        stream.write(audio_data)
        stream.stop()
        stream.close()