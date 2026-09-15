"""
tests/test_stt.py

Covers core/stt.py's HestiaSTT.

faster_whisper.WhisperModel and webrtcvad.Vad are mocked (see conftest.py
and per-test monkeypatches below), so no real model download, GPU, or
microphone is required. sounddevice.InputStream is mocked per-test to
supply canned audio frames.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

import core.stt as stt_module


def _make_stt(monkeypatch, tmp_path, model_side_effect=None, **kwargs):
    monkeypatch.chdir(tmp_path)
    model_ctor = MagicMock(side_effect=model_side_effect) if model_side_effect else MagicMock()
    monkeypatch.setattr(stt_module, "WhisperModel", model_ctor)
    monkeypatch.setattr(stt_module.webrtcvad, "Vad", MagicMock(return_value=MagicMock()))
    return stt_module.HestiaSTT(**kwargs), model_ctor


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_hf_cache_directories(self, monkeypatch, tmp_path):
        _make_stt(monkeypatch, tmp_path)
        assert (tmp_path / "data" / "hf_cache" / "hub").is_dir()

    def test_sets_hf_home_env_var(self, monkeypatch, tmp_path):
        import os as real_os  # core/stt.py does `import os` locally inside __init__
        _make_stt(monkeypatch, tmp_path)
        assert real_os.environ["HF_HOME"] == str(tmp_path / "data" / "hf_cache")

    def test_constructs_whisper_model_with_given_params(self, monkeypatch, tmp_path):
        _, model_ctor = _make_stt(
            monkeypatch, tmp_path, model_size="small.en", device="cpu", compute_type="int8"
        )
        model_ctor.assert_called_once_with("small.en", device="cpu", compute_type="int8")

    def test_reraises_and_logs_when_model_load_fails(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(RuntimeError, match="corrupt"):
            _make_stt(monkeypatch, tmp_path, model_side_effect=RuntimeError("corrupt weights"))
        assert "Model load failed" in capsys.readouterr().out

    def test_creates_noise_filter_enabled_by_default(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path)
        assert stt.noise_filter.is_enabled is True

    def test_noise_filter_disabled_when_requested(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path, noise_filter=False)
        assert stt.noise_filter.is_enabled is False

    def test_vad_constructed_with_aggressiveness_2(self, monkeypatch, tmp_path):
        vad_ctor = MagicMock(return_value=MagicMock())
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(stt_module, "WhisperModel", MagicMock())
        monkeypatch.setattr(stt_module.webrtcvad, "Vad", vad_ctor)
        stt_module.HestiaSTT()
        vad_ctor.assert_called_once_with(2)


# ---------------------------------------------------------------------------
# listen_once
# ---------------------------------------------------------------------------

class TestListenOnce:
    def test_returns_empty_string_when_no_audio_recorded(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path)
        monkeypatch.setattr(stt, "_record_until_silence", lambda max_duration, on_partial=None: np.array([], dtype="float32"))
        assert stt.listen_once() == ""

    def test_returns_empty_string_when_recording_is_none(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path)
        monkeypatch.setattr(stt, "_record_until_silence", lambda max_duration, on_partial=None: None)
        assert stt.listen_once() == ""

    def test_pipes_recording_through_noise_filter_then_transcribe(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path)
        raw_audio = np.array([0.1, 0.2], dtype="float32")
        filtered_audio = np.array([0.05, 0.1], dtype="float32")

        monkeypatch.setattr(stt, "_record_until_silence", lambda max_duration, on_partial=None: raw_audio)
        filter_mock = MagicMock(return_value=filtered_audio)
        monkeypatch.setattr(stt.noise_filter, "filter", filter_mock)
        transcribe_mock = MagicMock(return_value="hello world")
        monkeypatch.setattr(stt, "_transcribe", transcribe_mock)

        result = stt.listen_once(max_duration=7)

        assert result == "hello world"
        filter_args, _ = filter_mock.call_args
        np.testing.assert_array_equal(filter_args[0], raw_audio)
        assert filter_args[1] == stt.samplerate
        transcribe_args, _ = transcribe_mock.call_args
        np.testing.assert_array_equal(transcribe_args[0], filtered_audio)


# ---------------------------------------------------------------------------
# _record_until_silence
# ---------------------------------------------------------------------------

class _FakeInputStream:
    """Stand-in for sd.InputStream yielding a pre-scripted sequence of
    (chunk, is_speech) pairs, one per .read() call."""

    def __init__(self, chunks, read_delay=0.0):
        self._chunks = list(chunks)
        self._read_delay = read_delay

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, chunk_size):
        if self._read_delay:
            import time as _time
            _time.sleep(self._read_delay)
        if not self._chunks:
            # Ran out of scripted frames — return silence forever so the
            # test's timeout/silence-counter path can still terminate.
            return (np.zeros((chunk_size, 1), dtype="int16"), False)
        data, _is_speech = self._chunks.pop(0)
        return (data, False)


class TestRecordUntilSilence:
    def _chunk(self, value=100):
        return np.full((480, 1), value, dtype="int16")

    def test_stops_after_required_consecutive_silence_frames(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path, silence_frames=2)
        # 1 speech frame, then 4 silence frames (only 3 needed: >2 means
        # silence_counter must exceed 2, i.e. reach 3).
        is_speech_sequence = [True, False, False, False, False]
        chunks = [(self._chunk(), flag) for flag in is_speech_sequence]
        monkeypatch.setattr(stt_module.sd, "InputStream", lambda **kw: _FakeInputStream(chunks))
        stt.vad.is_speech.side_effect = is_speech_sequence

        audio = stt._record_until_silence(max_duration=10)

        assert audio is not None
        assert len(audio) > 0
        assert audio.dtype == np.float32

    def test_ignores_silence_before_any_speech_detected(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path, silence_frames=1)
        # Leading silence should not be recorded or count toward stopping;
        # only after speech starts does silence get appended/counted.
        is_speech_sequence = [False, False, True, False, False]
        chunks = [(self._chunk(), flag) for flag in is_speech_sequence]
        monkeypatch.setattr(stt_module.sd, "InputStream", lambda **kw: _FakeInputStream(chunks))
        stt.vad.is_speech.side_effect = is_speech_sequence

        audio = stt._record_until_silence(max_duration=10)
        # frames appended: 1 speech + 2 silence = 3 chunks of 480 samples
        assert len(audio) == 480 * 3

    def test_stops_on_timeout_when_no_silence_threshold_reached(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path, silence_frames=1000)
        monkeypatch.setattr(stt_module.time, "time", MagicMock(side_effect=[0, 0.1, 0.2, 11]))
        chunks = [(self._chunk(), True)] * 3
        monkeypatch.setattr(stt_module.sd, "InputStream", lambda **kw: _FakeInputStream(chunks))
        stt.vad.is_speech.side_effect = [True, True, True]

        audio = stt._record_until_silence(max_duration=10)
        assert audio is not None  # should return whatever was captured before timing out

    def test_returns_empty_array_on_stream_exception(self, monkeypatch, tmp_path, capsys):
        stt, _ = _make_stt(monkeypatch, tmp_path)

        def _raise(**kw):
            raise OSError("no audio device")

        monkeypatch.setattr(stt_module.sd, "InputStream", _raise)
        audio = stt._record_until_silence(max_duration=5)
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 0
        assert "Recording error" in capsys.readouterr().err

    def test_returns_empty_array_when_no_frames_captured(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path, silence_frames=1)
        monkeypatch.setattr(stt_module.time, "time", MagicMock(side_effect=[0, 20]))
        monkeypatch.setattr(stt_module.sd, "InputStream", lambda **kw: _FakeInputStream([]))
        stt.vad.is_speech.return_value = False

        audio = stt._record_until_silence(max_duration=10)
        assert len(audio) == 0


# ---------------------------------------------------------------------------
# _transcribe
# ---------------------------------------------------------------------------

class TestTranscribe:
    def test_joins_segment_texts_stripped(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path)
        seg1, seg2 = MagicMock(text="  hello "), MagicMock(text="world  ")
        stt.model.transcribe.return_value = ([seg1, seg2], MagicMock())

        result = stt._transcribe(np.zeros(10, dtype="float32"))

        assert result == "hello world"
        stt.model.transcribe.assert_called_once()
        _, kwargs = stt.model.transcribe.call_args
        assert kwargs["language"] == "en"
        assert kwargs["beam_size"] == 3

    def test_returns_empty_string_on_transcription_error(self, monkeypatch, tmp_path, capsys):
        stt, _ = _make_stt(monkeypatch, tmp_path)
        stt.model.transcribe.side_effect = RuntimeError("cuda oom")

        result = stt._transcribe(np.zeros(10, dtype="float32"))

        assert result == ""
        assert "Transcription error" in capsys.readouterr().err

    def test_empty_segments_returns_empty_string(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path)
        stt.model.transcribe.return_value = ([], MagicMock())
        assert stt._transcribe(np.zeros(10, dtype="float32")) == ""


# ---------------------------------------------------------------------------
# transcribe_audio()
# ---------------------------------------------------------------------------

class TestTranscribeAudio:
    def test_pipes_audio_through_noise_filter_then_transcribe(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path)
        raw_audio = np.array([0.1, 0.2], dtype="float32")
        filtered_audio = np.array([0.05, 0.1], dtype="float32")

        filter_mock = MagicMock(return_value=filtered_audio)
        monkeypatch.setattr(stt.noise_filter, "filter", filter_mock)
        transcribe_mock = MagicMock(return_value="hello world")
        monkeypatch.setattr(stt, "_transcribe", transcribe_mock)

        result = stt.transcribe_audio(raw_audio)

        assert result == "hello world"
        filter_mock.assert_called_once()
        transcribe_args, _ = transcribe_mock.call_args
        np.testing.assert_array_equal(transcribe_args[0], filtered_audio)

    def test_returns_empty_string_for_none(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path)
        assert stt.transcribe_audio(None) == ""

    def test_returns_empty_string_for_empty_array(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path)
        assert stt.transcribe_audio(np.array([], dtype="float32")) == ""

    def test_listen_once_delegates_to_transcribe_audio(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path)
        raw_audio = np.array([0.1, 0.2], dtype="float32")
        monkeypatch.setattr(stt, "_record_until_silence", lambda max_duration, on_partial=None: raw_audio)
        transcribe_audio_mock = MagicMock(return_value="from transcribe_audio")
        monkeypatch.setattr(stt, "transcribe_audio", transcribe_audio_mock)

        result = stt.listen_once(max_duration=7)

        assert result == "from transcribe_audio"
        transcribe_audio_mock.assert_called_once()
        np.testing.assert_array_equal(transcribe_audio_mock.call_args[0][0], raw_audio)


# ---------------------------------------------------------------------------
# listen_once(on_partial=...) — background partial transcription
# ---------------------------------------------------------------------------

class TestPartialTranscription:
    def test_on_partial_none_skips_partial_thread_entirely(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path, silence_frames=1)
        is_speech_sequence = [True, False, False]
        chunks = [(np.full((480, 1), 1000, dtype="int16"), flag) for flag in is_speech_sequence]
        monkeypatch.setattr(stt_module.sd, "InputStream", lambda **kw: _FakeInputStream(chunks))
        stt.vad.is_speech.side_effect = is_speech_sequence
        monkeypatch.setattr(stt, "_transcribe", MagicMock(return_value="final"))

        thread_ctor = MagicMock(wraps=stt_module.threading.Thread)
        monkeypatch.setattr(stt_module.threading, "Thread", thread_ctor)

        stt.listen_once(max_duration=5, on_partial=None)

        # No background thread should have been spun up for the partial loop.
        names = [c.kwargs.get("name") for c in thread_ctor.call_args_list]
        assert "STTPartialTranscribe" not in names

    def test_on_partial_is_invoked_with_transcribed_text_during_recording(self, monkeypatch, tmp_path):
        stt, _ = _make_stt(monkeypatch, tmp_path, silence_frames=50)
        # Long-ish scripted recording so the 1.2s-interval partial loop
        # has time to fire at least once via a shortened interval below.
        is_speech_sequence = [True] * 60 + [False] * 60
        chunks = [(np.full((480, 1), 1000, dtype="int16"), flag) for flag in is_speech_sequence]
        monkeypatch.setattr(stt_module.sd, "InputStream", lambda **kw: _FakeInputStream(chunks, read_delay=0.01))
        stt.vad.is_speech.side_effect = is_speech_sequence + [False] * 200
        monkeypatch.setattr(stt, "_transcribe", MagicMock(return_value="partial so far"))

        # Speed up the partial loop's polling interval so the test doesn't
        # need to wait a full 1.2s.
        real_loop = stt._partial_transcribe_loop
        def _fast_loop(frames, stop_event, on_partial, interval=1.2):
            return real_loop(frames, stop_event, on_partial, interval=0.05)
        monkeypatch.setattr(stt, "_partial_transcribe_loop", _fast_loop)

        seen = []
        stt.listen_once(max_duration=5, on_partial=lambda text: seen.append(text))

        assert "partial so far" in seen

    def test_partial_loop_swallows_transcription_errors(self, monkeypatch, tmp_path):
        # Exercise _partial_transcribe_loop directly (rather than through
        # listen_once) so mocking _transcribe to always raise doesn't also
        # blow up the unrelated final transcription call.
        stt, _ = _make_stt(monkeypatch, tmp_path)
        monkeypatch.setattr(stt, "_transcribe", MagicMock(side_effect=RuntimeError("boom")))

        frames = [np.full((480, 1), 1000, dtype="int16") for _ in range(20)]
        seen = []

        class _FakeStopEvent:
            """wait() returns False exactly once (one pass through the
            loop body), then True (stop) — so the loop body runs
            exactly once before exiting."""
            def __init__(self):
                self._calls = 0

            def wait(self, timeout):
                self._calls += 1
                return self._calls > 1

        stt._partial_transcribe_loop(frames, _FakeStopEvent(), lambda text: seen.append(text), interval=0.01)

        assert seen == []  # errors were swallowed, callback never invoked