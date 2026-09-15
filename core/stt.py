# core/stt.py

import threading
import time
import sys
from typing import Callable, Optional
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from core.noise_filter import NoiseFilter


def _cuda_available() -> bool:
    """Best-effort GPU detection.

    Checked via ctranslate2 first since that's the actual inference backend
    faster-whisper uses (a CUDA-capable torch install doesn't guarantee
    ctranslate2 was built with CUDA support, and vice versa isn't needed at
    all — ctranslate2 doesn't require torch). Falls back to torch, then to
    "no GPU" if neither is importable or usable.
    """
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        pass
    return False


def _resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    """Resolve a requested (device, compute_type) pair against real hardware.

    - device="auto" picks cuda if available, else cpu.
    - device="cuda" with no usable GPU falls back to cpu instead of letting
      WhisperModel(...) raise and crash startup.
    - float16/bfloat16 compute types aren't supported on CPU, so they're
      downgraded to int8 whenever the resolved device is cpu.
    """
    requested_device = device

    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    elif device == "cuda" and not _cuda_available():
        print(
            "[STT] device='cuda' was requested but no usable GPU was detected "
            "(no NVIDIA GPU, or faster-whisper's CUDA backend isn't installed) "
            "— falling back to CPU. Set stt.device to 'cpu' in your config to "
            "skip this check next time."
        )
        device = "cpu"

    if device == "cpu" and compute_type in ("float16", "bfloat16"):
        if requested_device != "cpu":
            print(
                f"[STT] compute_type='{compute_type}' isn't supported on CPU "
                f"— using 'int8' instead."
            )
        compute_type = "int8"

    return device, compute_type


class HestiaSTT:
    """Speech-to-text using faster-whisper with VAD and optional noise filtering."""

    def __init__(self, model_size: str = "base.en", device: str = "cuda",
                 compute_type: str = "int8", samplerate: int = 16000,
                 noise_filter: bool = True, silence_frames: int = 33):
        """Initialize Whisper model, VAD, and noise filter.

        silence_frames: number of consecutive non-speech 30ms VAD frames that
        must elapse before recording stops (33 frames ≈ 990ms of silence).

        device: "cuda", "cpu", or "auto". "cuda" degrades gracefully to
        "cpu" when no usable GPU is detected, rather than raising — see
        _resolve_device().
        """
        self.samplerate = samplerate
        self.silence_frames = silence_frames
        import os

        base_cache = os.path.join(os.getcwd(), "data", "hf_cache")
        hub_cache = os.path.join(base_cache, "hub")

        os.makedirs(hub_cache, exist_ok=True)

        os.environ["HF_HOME"] = base_cache

        device, compute_type = _resolve_device(device, compute_type)

        print(f"Loading Whisper model '{model_size}' on {device}...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print("STT ready")
        except Exception as e:
            print(f"Model load failed: {e}")
            raise

        self.vad = webrtcvad.Vad(2)  # 0–3 (higher = stricter)
        self.noise_filter = NoiseFilter(enabled=noise_filter)

    def listen_once(self, max_duration: int = 10, on_partial: Optional[Callable[[str], None]] = None) -> str:
        """Record an utterance with VAD, apply noise filter, and transcribe.

        on_partial: optional callback invoked with a best-effort partial
        transcript every ~1.2s while still recording (see
        _partial_transcribe_loop for the caveats — it's a periodic
        re-transcription of the buffer-so-far, not true incremental
        decoding). Purely additive: the final return value is always the
        one full-utterance transcription of the complete recording, done
        after silence is detected, same as before. Pass None (the
        default) to skip spinning up the extra background thread
        entirely — existing callers are unaffected.
        """
        audio = self._record_until_silence(max_duration, on_partial=on_partial)

        if audio is None or len(audio) == 0:
            return ""

        return self.transcribe_audio(audio)

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """Apply the configured noise filter and transcribe a pre-recorded
        audio array (float32, normalized to [-1, 1]).

        Split out from listen_once() so callers that already have a full
        recorded utterance in hand — notably the voice loop's barge-in
        continuation path (core/barge_in.py's BargeInListener records the
        user's follow-up itself once triggered) — can transcribe it
        directly instead of going through listen_once(), which would open
        a *second*, fresh microphone stream and miss whatever audio
        arrived in the gap before that stream started.
        """
        if audio is None or len(audio) == 0:
            return ""
        audio = self.noise_filter.filter(audio, self.samplerate)
        return self._transcribe(audio)

    def _record_until_silence(
        self, max_duration: int, on_partial: Optional[Callable[[str], None]] = None
    ) -> np.ndarray:
        """Record audio until silence or timeout, return float32 array normalized to [-1, 1]."""
        frames = []
        silence_counter = 0
        speech_started = False
        start_time = time.time()

        # 30ms frames at 16kHz = 480 samples (webrtcvad requires 10/20/30ms)
        chunk_size = 480

        partial_thread = None
        partial_stop = threading.Event()
        if on_partial is not None:
            partial_thread = threading.Thread(
                target=self._partial_transcribe_loop,
                args=(frames, partial_stop, on_partial),
                daemon=True,
                name="STTPartialTranscribe",
            )
            partial_thread.start()

        try:
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

                        # Stop after `silence_frames` consecutive non-speech VAD
                        # frames (each frame is 30ms), i.e. ~silence_frames*30ms
                        # of continuous silence following detected speech.
                        if speech_started and silence_counter > self.silence_frames:
                            print("[Silence detected → stopping]")
                            break

                        if time.time() - start_time > max_duration:
                            print("[Timeout reached]")
                            break

            except Exception as e:
                print(f"Recording error: {e}", file=sys.stderr)
                return np.array([], dtype="float32")
        finally:
            partial_stop.set()
            if partial_thread is not None:
                partial_thread.join(timeout=2.0)

        if not frames:
            return np.array([], dtype="float32")

        audio = np.concatenate(frames).astype("float32") / 32768.0
        return audio

    def _partial_transcribe_loop(
        self,
        frames: list,
        stop_event: threading.Event,
        on_partial: Callable[[str], None],
        interval: float = 1.2,
    ) -> None:
        """Background loop for listen_once(on_partial=...): periodically
        re-transcribes whatever's been recorded so far and reports it,
        so a caller (e.g. a live-caption display) can show recognition
        progress well before the user stops talking.

        Best-effort and approximate by design: faster-whisper has no
        true incremental/streaming decode mode here, so this just re-runs
        a full transcription over the growing buffer each pass — it costs
        real CPU/GPU time on top of the eventual final transcription, and
        wording can change between passes as more context arrives (the
        same way live captions elsewhere sometimes correct themselves).
        Runs on a separate thread so it never blocks the tight audio-read
        loop in _record_until_silence(); never raises — a transcription
        error here is skipped and does not affect the real recording.
        """
        last_len = 0
        min_samples = int(self.samplerate * 0.3)
        while not stop_event.wait(interval):
            snapshot = list(frames)  # frames.append() is atomic in CPython
            if not snapshot or len(snapshot) == last_len:
                continue
            last_len = len(snapshot)
            try:
                audio = np.concatenate(snapshot).astype("float32") / 32768.0
                if len(audio) < min_samples:
                    continue
                text = self._transcribe(audio)
                if text:
                    on_partial(text)
            except Exception:
                continue

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