# core/barge_in.py

import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import webrtcvad


class BargeInListener:
    """
    Listens on the microphone *while Hestia is talking* and fires a
    callback the moment it hears the user start speaking over her — the
    mechanism that makes barge-in ("interrupt mid-sentence, like ChatGPT's
    voice mode") possible at all, instead of Hestia only ever being
    interruptible in the gap between turns.

    Deliberately a separate, much simpler listener from WakeWordDetector /
    HestiaSTT rather than a reuse of either: those only ever run *between*
    turns (waiting for the wake word, then recording one utterance), so
    there's never a device conflict with this one owning the input stream
    — it's only ever active for the duration of a single TTS turn (see
    Hestia._speak_streaming / _speak_with_barge_in in main.py), which is
    exactly when the other two are idle.

    Two-phase behaviour
    --------------------
    Phase 1 (detection): raw VAD + an amplitude gate on a rolling window
    of frames, answering "is the user talking right now?". A small
    pre-roll ring buffer is kept at all times so the ~1-2 frames of audio
    right before the trigger fires (the very start of the user's
    sentence) aren't lost.

    Phase 2 (continuation): once triggered, the listener does NOT tear
    down the input stream — it keeps recording on the *same* stream,
    starting from the pre-roll buffer, until it sees trailing silence or
    hits a hard cap. That recording is exposed via ``captured_audio`` /
    ``consume_captured_audio()`` so the voice loop can feed it straight
    to transcription instead of closing this stream and opening a brand
    new one via HestiaSTT.listen_once() — which would otherwise miss
    whatever the user said in the gap between "barge-in noticed speech"
    and "a second microphone stream finished spinning up" (see
    KNOWN_GAPS.md).

    Self-echo caveat
    -----------------
    This is plain energy+VAD detection with no acoustic echo
    cancellation (AEC). On a laptop with the mic and speakers a few
    inches apart, Hestia's own voice played back through the speakers
    can bleed into the mic and register as "the user is talking",
    triggering a false self-interruption. ``min_rms`` below is a partial
    mitigation (requiring the incoming signal to clear an amplitude
    floor cuts down on quieter echo bleed vs. a person actually talking
    close to the mic) but it is a heuristic, not real AEC — it will not
    fully solve the problem on shared mic/speaker hardware. The reliable
    fix is a headset (mic physically isolated from the speaker output)
    or routing playback + capture through a real AEC stage (e.g.
    WebRTC's or Speex's echo canceller) if/when that's wired in. See
    config/laptop_config.yaml's `barge_in` section for tuning guidance.
    """

    def __init__(
        self,
        samplerate: int = 16000,
        vad_aggressiveness: int = 2,
        speech_frames_to_trigger: int = 3,
        min_rms: float = 300.0,
        pre_roll_frames: int = 10,
        post_trigger_silence_frames: int = 25,
        max_capture_seconds: float = 12.0,
    ):
        """
        vad_aggressiveness: webrtcvad's 0-3 scale (higher = stricter,
        fewer false positives from background noise, but slower to
        notice quiet speech). 2 matches core/stt.py's own VAD setting.

        speech_frames_to_trigger: consecutive 30ms VAD-positive frames
        required before this counts as "the user is actually talking"
        rather than a single stray noise spike. 3 frames ≈ 90ms.

        min_rms: an int16 RMS amplitude floor (0-32768 scale) a frame
        must also clear, on top of being VAD-positive, to count toward
        the trigger. This is the heuristic self-echo mitigation
        described in the class docstring — raise it (try 600-1200) if
        Hestia is cutting herself off on laptop speakers; lower it (or
        rely on VAD alone by setting to 0) if real interruptions aren't
        being noticed. It is not a substitute for a headset.

        pre_roll_frames: how many 30ms frames of audio to always keep
        buffered before a trigger, so the recording that follows a
        barge-in includes the lead-in instead of starting mid-word.
        10 frames ≈ 300ms.

        post_trigger_silence_frames: like HestiaSTT's own
        `silence_frames` — how many consecutive non-speech frames after
        a trigger mark the end of the user's follow-up utterance. 25
        frames ≈ 750ms (shorter than STT's default 33/~990ms since the
        user is already mid-sentence, not starting cold).

        max_capture_seconds: hard cap on how long phase 2 will keep
        recording, in case silence is never detected (e.g. a VAD that
        never settles) — keeps a stuck barge-in from listening forever.
        """
        self._vad = webrtcvad.Vad(vad_aggressiveness)
        self.samplerate = samplerate
        self._speech_frames_to_trigger = speech_frames_to_trigger
        self._min_rms = max(0.0, min_rms)
        self._pre_roll_frames = max(1, pre_roll_frames)
        self._post_trigger_silence_frames = post_trigger_silence_frames
        self._max_capture_seconds = max_capture_seconds

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._triggered = threading.Event()
        self._callback: Optional[Callable[[], None]] = None
        self._captured_audio: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the triggered flag and any leftover captured audio ahead
        of a new TTS turn."""
        self._triggered.clear()
        self._captured_audio = None

    def consume_triggered(self) -> bool:
        """Return whether barge-in fired since the last reset(), clearing
        the flag as a side effect (test-and-clear), so callers can ask
        "did the user interrupt me?" exactly once per turn."""
        fired = self._triggered.is_set()
        self._triggered.clear()
        return fired

    @property
    def triggered(self) -> bool:
        return self._triggered.is_set()

    @property
    def captured_audio(self) -> Optional[np.ndarray]:
        """The audio recorded during phase 2 (continuation) after the
        most recent trigger, as a float32 array normalized to [-1, 1] —
        or None if there was no trigger, or phase 2 hasn't finished /
        captured anything yet. Left in place until reset()/
        consume_captured_audio() clears it."""
        return self._captured_audio

    def consume_captured_audio(self) -> Optional[np.ndarray]:
        """Return and clear the captured audio (test-and-clear, same
        pattern as consume_triggered())."""
        audio = self._captured_audio
        self._captured_audio = None
        return audio

    def start(self, on_barge_in: Callable[[], None]) -> None:
        """
        Begin listening in the background. Call stop() once the TTS turn
        this is guarding has finished (whether it finished normally or was
        itself the thing that got interrupted).

        No-op if already running — callers don't need to guard against
        calling start() twice.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._callback = on_barge_in
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="BargeInListener"
        )
        self._thread.start()

    def stop(self, force: bool = False) -> None:
        """
        Stop listening. Safe to call even if start() was never called, or
        was already stopped.

        If this turn was never interrupted (not triggered), tears down
        immediately — there's nothing to wait for. If it WAS triggered,
        the background thread has already moved into phase 2
        (recording the user's follow-up utterance) and is left to finish
        on its own — forcing it to stop here would cut the user off
        exactly the audio this class exists to capture. Pass force=True
        (e.g. from application shutdown) to tear down immediately
        regardless of phase.
        """
        if force or not self._triggered.is_set():
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._max_capture_seconds + 2.0)
        self._thread = None

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_rms(data: np.ndarray) -> float:
        if data.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))

    def _run(self) -> None:
        # 30ms frames at 16kHz = 480 samples (webrtcvad requires 10/20/30ms
        # frames) — same framing core/stt.py's VAD uses.
        chunk_size = 480

        try:
            with sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype="int16",
                blocksize=chunk_size,
            ) as stream:
                triggered = self._detect(stream, chunk_size)
                if triggered:
                    self._capture_followup(stream, chunk_size)
        except Exception:
            # No usable input device, permission denied, device busy,
            # etc. Barge-in is a nice-to-have on top of the normal
            # wake-word turn-taking flow — a failure here must never take
            # down the voice loop, it just means this turn can't be
            # interrupted early.
            pass

    def _detect(self, stream, chunk_size: int) -> bool:
        """Phase 1: watch for sustained, loud-enough speech. Returns True
        (and fires the callback) the moment it triggers."""
        pre_roll: deque = deque(maxlen=self._pre_roll_frames)
        consecutive_speech = 0

        while not self._stop_event.is_set():
            try:
                data, _ = stream.read(chunk_size)
            except Exception:
                return False

            pre_roll.append(data)

            is_speech = self._vad.is_speech(data.tobytes(), self.samplerate)
            loud_enough = self._frame_rms(data) >= self._min_rms

            if is_speech and loud_enough:
                consecutive_speech += 1
            else:
                consecutive_speech = 0

            if consecutive_speech >= self._speech_frames_to_trigger:
                self._triggered.set()
                self._pre_roll_snapshot = list(pre_roll)
                if self._callback is not None:
                    try:
                        self._callback()
                    except Exception:
                        pass
                return True

        return False

    def _capture_followup(self, stream, chunk_size: int) -> None:
        """Phase 2: keep recording — starting from the pre-roll captured
        during phase 1 — until trailing silence or the hard cap."""
        frames = list(getattr(self, "_pre_roll_snapshot", []))
        silence_counter = 0
        # Phase 1's trigger already established the user is mid-speech —
        # unlike HestiaSTT's own recording loop (which waits for a first
        # speech frame before it starts counting silence at all), phase 2
        # must be ready to end on trailing silence from its very first
        # frame. Otherwise a short follow-up ("wait!" then nothing) would
        # never see another speech frame and capture would run all the
        # way to max_capture_seconds instead of stopping at the normal
        # ~750ms of silence.
        start_time = time.time()

        while not self._stop_event.is_set():
            if time.time() - start_time > self._max_capture_seconds:
                break
            try:
                data, _ = stream.read(chunk_size)
            except Exception:
                break

            frames.append(data)
            is_speech = self._vad.is_speech(data.tobytes(), self.samplerate)

            if is_speech:
                silence_counter = 0
            else:
                silence_counter += 1

            if silence_counter > self._post_trigger_silence_frames:
                break

        if frames:
            self._captured_audio = np.concatenate(frames).astype("float32") / 32768.0


if __name__ == "__main__":
    listener = BargeInListener()
    print("Say something (loudly, within RMS floor) to trigger barge-in...")
    try:
        listener.start(on_barge_in=lambda: print("Barge-in triggered!"))
        time.sleep(15)
        listener.stop()
        if listener.captured_audio is not None:
            print(f"Captured {len(listener.captured_audio) / listener.samplerate:.1f}s of follow-up audio.")
    except KeyboardInterrupt:
        listener.stop(force=True)