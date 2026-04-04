# core/stt.py

import time
import sys
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from core.noise_filter import NoiseFilter




class HestiaSTT:
    """Speech-to-text using faster-whisper with VAD and optional noise filtering."""

    def __init__(self, model_size: str = "base.en", device: str = "cuda",
                 compute_type: str = "int8", samplerate: int = 16000,
                 noise_filter: bool = True):
        """Initialize Whisper model, VAD, and noise filter."""
        self.samplerate = samplerate
        import os

        base_cache = os.path.join(os.getcwd(), "data", "hf_cache")
        hub_cache = os.path.join(base_cache, "hub")

        os.makedirs(hub_cache, exist_ok=True)

        os.environ["HF_HOME"] = base_cache

        print(f"Loading Whisper model '{model_size}' on {device}...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print("STT ready")
        except Exception as e:
            print(f"Model load failed: {e}")
            raise

        self.vad = webrtcvad.Vad(2)  # 0–3 (higher = stricter)
        self.noise_filter = NoiseFilter(enabled=noise_filter)

    def listen_once(self, max_duration: int = 10) -> str:
        """Record an utterance with VAD, apply noise filter, and transcribe."""
        audio = self._record_until_silence(max_duration)

        if audio is None or len(audio) == 0:
            return ""

        audio = self.noise_filter.filter(audio, self.samplerate)
        return self._transcribe(audio)

    def _record_until_silence(self, max_duration: int) -> np.ndarray:
        """Record audio until silence or timeout, return float32 array normalized to [-1, 1]."""
        frames = []
        silence_counter = 0
        speech_started = False
        start_time = time.time()

        # 30ms frames at 16kHz = 480 samples (webrtcvad requires 10/20/30ms)
        chunk_size = 480

        try:
            with sd.InputStream(samplerate=self.samplerate, channels=1, dtype='int16') as stream:
                while True:
                    data, _ = stream.read(chunk_size)
                    audio_bytes = data.tobytes()

                    is_speech = self.vad.is_speech(audio_bytes, self.samplerate)

                    if is_speech:
                        if not speech_started:
                            print("[Speech detected]")
                            speech_started = True
                        silence_counter = 0
                        frames.append(data)
                    else:
                        if speech_started:
                            silence_counter += 1
                            frames.append(data)

                    # Stop after ~1 second of silence (10 * 100ms? No: 30ms * 10 = 300ms? Wait.)
                    # VAD frame is 30ms. 10 frames = 300ms, not 1s. Use 33 frames ≈ 1s.
                    # But original code used 10 * 100ms chunks. Let's use 33 for ~1s.
                    if speech_started and silence_counter > 33:  # ~1 second (33 * 30ms = 990ms)
                        print("[Silence detected → stopping]")
                        break

                    if time.time() - start_time > max_duration:
                        print("[Timeout reached]")
                        break

        except Exception as e:
            print(f"Recording error: {e}", file=sys.stderr)
            return np.array([], dtype="float32")

        if not frames:
            return np.array([], dtype="float32")

        audio = np.concatenate(frames).astype("float32") / 32768.0
        return audio

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio array using Whisper."""
        try:
            segments, _ = self.model.transcribe(
                audio,
                language="en",
                beam_size=3
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            print(f"Got: {text}")
            return text
        except Exception as e:
            print(f"Transcription error: {e}", file=sys.stderr)
            return ""