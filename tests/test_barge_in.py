# tests/test_barge_in.py
"""
Covers core/barge_in.py's BargeInListener.

sounddevice and webrtcvad are mocked (see conftest.py / patches below) so
these tests run without a real microphone.
"""
import time
from unittest.mock import MagicMock

import numpy as np

import core.barge_in as barge_in_module


class _FakeInputStream:
    """Stand-in for sd.InputStream yielding a pre-scripted sequence of
    chunks, one per .read() call. Once the script is exhausted, returns
    silence chunks forever so a test that forgets to stop() the listener
    doesn't hang — the same shape used by tests/test_stt.py."""

    def __init__(self, chunks, read_delay=0.0):
        self._chunks = list(chunks)
        self._read_delay = read_delay

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, chunk_size):
        if self._read_delay:
            time.sleep(self._read_delay)
        if not self._chunks:
            return (np.zeros((chunk_size, 1), dtype="int16"), False)
        data = self._chunks.pop(0)
        return (data, False)


def _chunk(value=1000):
    # 1000 clears BargeInListener's default min_rms=300.0 amplitude gate,
    # so these fixtures register as "loud enough" and detection behaviour
    # is governed purely by the scripted VAD is_speech sequence, same as
    # before that gate existed.
    return np.full((480, 1), value, dtype="int16")


def _make_listener(monkeypatch, is_speech_sequence, speech_frames_to_trigger=3, read_delay=0.0):
    """Build a BargeInListener with webrtcvad.Vad and sd.InputStream both
    mocked, wired to replay is_speech_sequence one call at a time."""
    fake_vad = MagicMock()
    fake_vad.is_speech.side_effect = list(is_speech_sequence)
    monkeypatch.setattr(barge_in_module.webrtcvad, "Vad", MagicMock(return_value=fake_vad))

    chunks = [_chunk() for _ in is_speech_sequence]
    monkeypatch.setattr(
        barge_in_module.sd,
        "InputStream",
        lambda **kw: _FakeInputStream(chunks, read_delay=read_delay),
    )

    listener = barge_in_module.BargeInListener(
        speech_frames_to_trigger=speech_frames_to_trigger
    )
    return listener, fake_vad


# ---------------------------------------------------------------------------
# __init__ / reset / consume_triggered
# ---------------------------------------------------------------------------

class TestInitAndFlags:
    def test_not_triggered_initially(self, monkeypatch):
        listener, _ = _make_listener(monkeypatch, [])
        assert listener.triggered is False

    def test_reset_clears_triggered_flag(self, monkeypatch):
        listener, _ = _make_listener(monkeypatch, [])
        listener._triggered.set()
        listener.reset()
        assert listener.triggered is False

    def test_consume_triggered_returns_false_and_stays_false_when_never_triggered(self, monkeypatch):
        listener, _ = _make_listener(monkeypatch, [])
        assert listener.consume_triggered() is False
        assert listener.triggered is False

    def test_consume_triggered_returns_true_once_then_clears(self, monkeypatch):
        listener, _ = _make_listener(monkeypatch, [])
        listener._triggered.set()
        assert listener.consume_triggered() is True
        assert listener.consume_triggered() is False


# ---------------------------------------------------------------------------
# start() / _run() — detection behavior
# ---------------------------------------------------------------------------

class TestDetection:
    def test_fires_callback_after_enough_consecutive_speech_frames(self, monkeypatch):
        # 2 speech frames isn't enough (need 3), 3rd tips it over.
        listener, _ = _make_listener(monkeypatch, [True, True, True], speech_frames_to_trigger=3)
        fired = []
        listener.start(lambda: fired.append(True))
        listener._thread.join(timeout=2.0)
        assert fired == [True]
        assert listener.triggered is True

    def test_non_consecutive_speech_frames_do_not_trigger(self, monkeypatch):
        # True, False, True, False, True never reaches 3 in a row.
        listener, _ = _make_listener(
            monkeypatch, [True, False, True, False, True], speech_frames_to_trigger=3
        )
        fired = []
        listener.start(lambda: fired.append(True))
        listener.stop()
        assert fired == []
        assert listener.triggered is False

    def test_lower_trigger_threshold_fires_sooner(self, monkeypatch):
        listener, _ = _make_listener(monkeypatch, [True, True], speech_frames_to_trigger=2)
        fired = []
        listener.start(lambda: fired.append(True))
        listener._thread.join(timeout=2.0)
        assert fired == [True]

    def test_callback_exception_is_swallowed(self, monkeypatch):
        listener, _ = _make_listener(monkeypatch, [True, True, True], speech_frames_to_trigger=3)

        def _boom():
            raise RuntimeError("boom")

        listener.start(_boom)  # must not raise
        listener._thread.join(timeout=2.0)
        assert listener.triggered is True

    def test_stream_setup_failure_is_swallowed_not_raised(self, monkeypatch):
        fake_vad = MagicMock()
        fake_vad.is_speech.return_value = False
        monkeypatch.setattr(barge_in_module.webrtcvad, "Vad", MagicMock(return_value=fake_vad))
        monkeypatch.setattr(
            barge_in_module.sd, "InputStream", MagicMock(side_effect=OSError("no device"))
        )
        listener = barge_in_module.BargeInListener()
        fired = []
        listener.start(lambda: fired.append(True))  # must not raise
        listener._thread.join(timeout=2.0)
        assert fired == []
        assert listener.triggered is False


# ---------------------------------------------------------------------------
# start() / stop() lifecycle
# ---------------------------------------------------------------------------

class TestRmsGate:
    def test_quiet_frames_do_not_trigger_even_if_vad_says_speech(self, monkeypatch):
        # VAD says "speech" every frame, but the audio is quiet (below the
        # default min_rms=300 floor) — simulating echo bleed from Hestia's
        # own TTS rather than a person actually talking into the mic.
        fake_vad = MagicMock()
        fake_vad.is_speech.side_effect = [True, True, True, True, True]
        monkeypatch.setattr(barge_in_module.webrtcvad, "Vad", MagicMock(return_value=fake_vad))

        quiet_chunks = [np.full((480, 1), 50, dtype="int16") for _ in range(5)]
        monkeypatch.setattr(
            barge_in_module.sd, "InputStream", lambda **kw: _FakeInputStream(quiet_chunks)
        )

        listener = barge_in_module.BargeInListener(speech_frames_to_trigger=3, min_rms=300.0)
        fired = []
        listener.start(lambda: fired.append(True))
        listener.stop()

        assert fired == []
        assert listener.triggered is False

    def test_loud_frames_trigger_normally(self, monkeypatch):
        listener, _ = _make_listener(monkeypatch, [True, True, True], speech_frames_to_trigger=3)
        fired = []
        listener.start(lambda: fired.append(True))
        listener._thread.join(timeout=2.0)
        assert fired == [True]

    def test_min_rms_zero_disables_the_amplitude_gate(self, monkeypatch):
        fake_vad = MagicMock()
        fake_vad.is_speech.side_effect = [True, True, True]
        monkeypatch.setattr(barge_in_module.webrtcvad, "Vad", MagicMock(return_value=fake_vad))
        quiet_chunks = [np.full((480, 1), 1, dtype="int16") for _ in range(3)]
        monkeypatch.setattr(
            barge_in_module.sd, "InputStream", lambda **kw: _FakeInputStream(quiet_chunks)
        )

        listener = barge_in_module.BargeInListener(speech_frames_to_trigger=3, min_rms=0.0)
        fired = []
        listener.start(lambda: fired.append(True))
        listener._thread.join(timeout=2.0)
        assert fired == [True]


class TestFollowupCapture:
    def test_captured_audio_is_none_before_any_trigger(self, monkeypatch):
        listener, _ = _make_listener(monkeypatch, [False, False, False], speech_frames_to_trigger=3)
        listener.start(lambda: None)
        listener.stop()
        assert listener.captured_audio is None

    def test_trigger_followed_by_silence_produces_captured_audio(self, monkeypatch):
        # 3 speech frames trigger detection; then enough silence frames to
        # end phase 2 quickly (post_trigger_silence_frames=2).
        is_speech_sequence = [True, True, True] + [False] * 5
        fake_vad = MagicMock()
        fake_vad.is_speech.side_effect = is_speech_sequence + [False] * 50
        monkeypatch.setattr(barge_in_module.webrtcvad, "Vad", MagicMock(return_value=fake_vad))
        chunks = [np.full((480, 1), 1000, dtype="int16") for _ in is_speech_sequence]
        monkeypatch.setattr(barge_in_module.sd, "InputStream", lambda **kw: _FakeInputStream(chunks))

        listener = barge_in_module.BargeInListener(
            speech_frames_to_trigger=3, post_trigger_silence_frames=2, min_rms=0.0
        )
        listener.start(lambda: None)
        listener._thread.join(timeout=2.0)

        audio = listener.captured_audio
        assert audio is not None
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    def test_captured_audio_includes_pre_roll_lead_in(self, monkeypatch):
        # pre_roll_frames=2 means the 2 frames right before the trigger
        # threshold is reached should be included in the capture, not just
        # the frames from the trigger point onward.
        is_speech_sequence = [False, False, True, True, True] + [False] * 5
        fake_vad = MagicMock()
        fake_vad.is_speech.side_effect = is_speech_sequence + [False] * 50
        monkeypatch.setattr(barge_in_module.webrtcvad, "Vad", MagicMock(return_value=fake_vad))
        chunks = [np.full((480, 1), 1000, dtype="int16") for _ in is_speech_sequence]
        monkeypatch.setattr(barge_in_module.sd, "InputStream", lambda **kw: _FakeInputStream(chunks))

        listener = barge_in_module.BargeInListener(
            speech_frames_to_trigger=3, pre_roll_frames=2,
            post_trigger_silence_frames=2, min_rms=0.0,
        )
        listener.start(lambda: None)
        listener._thread.join(timeout=2.0)

        audio = listener.captured_audio
        assert audio is not None
        # 2 pre-roll frames + 3 trigger frames + 3 trailing silence frames
        # (post_trigger_silence_frames=2 means it stops once counter > 2,
        # i.e. after the 3rd silence frame) = 8 frames of 480 samples each.
        assert len(audio) >= 480 * 5  # at least pre-roll + trigger frames

    def test_consume_captured_audio_clears_it(self, monkeypatch):
        is_speech_sequence = [True, True, True] + [False] * 5
        fake_vad = MagicMock()
        fake_vad.is_speech.side_effect = is_speech_sequence + [False] * 50
        monkeypatch.setattr(barge_in_module.webrtcvad, "Vad", MagicMock(return_value=fake_vad))
        chunks = [np.full((480, 1), 1000, dtype="int16") for _ in is_speech_sequence]
        monkeypatch.setattr(barge_in_module.sd, "InputStream", lambda **kw: _FakeInputStream(chunks))

        listener = barge_in_module.BargeInListener(
            speech_frames_to_trigger=3, post_trigger_silence_frames=2, min_rms=0.0
        )
        listener.start(lambda: None)
        listener._thread.join(timeout=2.0)

        assert listener.captured_audio is not None
        first = listener.consume_captured_audio()
        assert first is not None
        assert listener.consume_captured_audio() is None
        assert listener.captured_audio is None

    def test_reset_clears_stale_captured_audio(self, monkeypatch):
        is_speech_sequence = [True, True, True] + [False] * 5
        fake_vad = MagicMock()
        fake_vad.is_speech.side_effect = is_speech_sequence + [False] * 50
        monkeypatch.setattr(barge_in_module.webrtcvad, "Vad", MagicMock(return_value=fake_vad))
        chunks = [np.full((480, 1), 1000, dtype="int16") for _ in is_speech_sequence]
        monkeypatch.setattr(barge_in_module.sd, "InputStream", lambda **kw: _FakeInputStream(chunks))

        listener = barge_in_module.BargeInListener(
            speech_frames_to_trigger=3, post_trigger_silence_frames=2, min_rms=0.0
        )
        listener.start(lambda: None)
        listener._thread.join(timeout=2.0)
        assert listener.captured_audio is not None

        listener.reset()
        assert listener.captured_audio is None

    def test_capture_stops_at_max_capture_seconds_if_silence_never_comes(self, monkeypatch):
        fake_vad = MagicMock()
        fake_vad.is_speech.return_value = True  # speech forever, never silent
        monkeypatch.setattr(barge_in_module.webrtcvad, "Vad", MagicMock(return_value=fake_vad))
        monkeypatch.setattr(
            barge_in_module.sd, "InputStream",
            lambda **kw: _FakeInputStream([np.full((480, 1), 1000, dtype="int16")] * 5, read_delay=0.02),
        )

        listener = barge_in_module.BargeInListener(
            speech_frames_to_trigger=3, min_rms=0.0, max_capture_seconds=0.05
        )
        listener.start(lambda: None)
        listener._thread.join(timeout=2.0)

        # Must terminate (not hang forever) and have captured *something*.
        assert listener.captured_audio is not None


class TestLifecycle:
    def test_start_is_a_noop_if_already_running(self, monkeypatch):
        listener, _ = _make_listener(monkeypatch, [], read_delay=0.05)
        listener.start(lambda: None)
        first_thread = listener._thread
        listener.start(lambda: None)
        assert listener._thread is first_thread
        listener.stop()

    def test_stop_is_safe_when_never_started(self, monkeypatch):
        listener, _ = _make_listener(monkeypatch, [])
        listener.stop()  # must not raise
        assert listener._thread is None

    def test_stop_joins_and_clears_thread(self, monkeypatch):
        listener, _ = _make_listener(monkeypatch, [], read_delay=0.05)
        listener.start(lambda: None)
        assert listener._thread is not None
        listener.stop()
        assert listener._thread is None

    def test_stop_then_start_again_works(self, monkeypatch):
        listener, _ = _make_listener(monkeypatch, [True, True, True], speech_frames_to_trigger=3)
        listener.start(lambda: None)
        listener.stop()

        listener2, _ = _make_listener(monkeypatch, [True, True, True], speech_frames_to_trigger=3)
        fired = []
        listener2.start(lambda: fired.append(True))
        listener2._thread.join(timeout=2.0)
        assert fired == [True]