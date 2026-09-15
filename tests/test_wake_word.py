"""
tests/test_wake_word.py

Covers core/wake_word.py's WakeWordDetector.

vosk and sounddevice are mocked (see conftest.py / patches below) so these
tests run without a real Vosk model file or a microphone.
"""
import json
import queue as queue_module
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import core.wake_word as wake_word


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector(monkeypatch, wake_words=None):
    """Construct a WakeWordDetector with vosk fully mocked and no real
    filesystem check for the model path."""
    monkeypatch.setattr(wake_word.os.path, "exists", lambda p: True)
    fake_model = MagicMock(name="vosk.Model instance")
    fake_recognizer = MagicMock(name="vosk.KaldiRecognizer instance")
    monkeypatch.setattr(wake_word.vosk, "Model", MagicMock(return_value=fake_model))
    monkeypatch.setattr(wake_word.vosk, "KaldiRecognizer", MagicMock(return_value=fake_recognizer))
    detector = wake_word.WakeWordDetector(wake_words=wake_words)
    return detector, fake_recognizer


def _disable_initial_flush(monkeypatch, detector):
    """listen_for_wake_word() unconditionally flushes the queue as its
    first action (to discard stale audio from a previous call). Tests here
    pre-load the queue *before* calling listen_for_wake_word, so that first
    flush must be neutered or it wipes the fixture data before the loop
    ever runs. The later, post-detection flush is left untouched."""
    real_flush = detector.flush_audio_queue
    state = {"n": 0}

    def patched():
        state["n"] += 1
        if state["n"] == 1:
            return
        return real_flush()

    monkeypatch.setattr(detector, "flush_audio_queue", patched)


class _NullStream:
    """Stand-in for sd.RawInputStream — a no-op context manager."""

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_raises_file_not_found_when_model_path_missing(self, monkeypatch):
        monkeypatch.setattr(wake_word.os.path, "exists", lambda p: False)
        with pytest.raises(FileNotFoundError, match="Vosk model not found"):
            wake_word.WakeWordDetector(model_path="models/does-not-exist")

    def test_wraps_vosk_load_failure_in_runtime_error(self, monkeypatch):
        monkeypatch.setattr(wake_word.os.path, "exists", lambda p: True)
        monkeypatch.setattr(
            wake_word.vosk, "Model", MagicMock(side_effect=Exception("bad model file"))
        )
        with pytest.raises(RuntimeError, match="Failed to load Vosk model"):
            wake_word.WakeWordDetector()

    def test_default_wake_words_are_lowercased(self, monkeypatch):
        detector, _ = _make_detector(monkeypatch, wake_words=["HeStIa", "Hey HESTIA"])
        assert detector.wake_words == ["hestia", "hey hestia"]

    def test_default_wake_word_list_used_when_none_given(self, monkeypatch):
        detector, _ = _make_detector(monkeypatch, wake_words=None)
        assert "hestia" in detector.wake_words
        assert "hey hestia" in detector.wake_words

    def test_recognizer_built_with_model_and_sample_rate(self, monkeypatch):
        monkeypatch.setattr(wake_word.os.path, "exists", lambda p: True)
        fake_model = MagicMock()
        model_ctor = MagicMock(return_value=fake_model)
        recognizer_ctor = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(wake_word.vosk, "Model", model_ctor)
        monkeypatch.setattr(wake_word.vosk, "KaldiRecognizer", recognizer_ctor)
        wake_word.WakeWordDetector()
        recognizer_ctor.assert_called_once_with(fake_model, 16000)


# ---------------------------------------------------------------------------
# flush_audio_queue
# ---------------------------------------------------------------------------

class TestFlushAudioQueue:
    def test_drains_all_pending_items(self, monkeypatch):
        detector, _ = _make_detector(monkeypatch)
        detector.q.put(b"a")
        detector.q.put(b"b")
        detector.q.put(b"c")
        detector.flush_audio_queue()
        assert detector.q.empty()

    def test_noop_on_already_empty_queue(self, monkeypatch):
        detector, _ = _make_detector(monkeypatch)
        detector.flush_audio_queue()  # should not raise
        assert detector.q.empty()


# ---------------------------------------------------------------------------
# _audio_callback
# ---------------------------------------------------------------------------

class TestAudioCallback:
    def test_pushes_bytes_copy_of_indata_onto_queue(self, monkeypatch):
        detector, _ = _make_detector(monkeypatch)
        detector._audio_callback(bytearray(b"\x01\x02"), 2, None, None)
        assert detector.q.get_nowait() == b"\x01\x02"


# ---------------------------------------------------------------------------
# listen_for_wake_word
# ---------------------------------------------------------------------------

class TestListenForWakeWord:
    def _run_with_frames(self, monkeypatch, detector, recognizer, frame_texts, timeout=None):
        """Feed `frame_texts` (list of decoded text results, one per audio
        frame) through AcceptWaveform/Result, then let listen_for_wake_word
        run out of frames -> loop reads real queue via a fake stream that
        pre-populates the queue.

        listen_for_wake_word() calls flush_audio_queue() as its very first
        line, which would wipe out anything queued before the call — so
        that initial flush is neutered here (the final flush on detection,
        later in the method, is left intact and still exercised).
        """
        monkeypatch.setattr(wake_word.sd, "RawInputStream", _NullStream)
        _disable_initial_flush(monkeypatch, detector)

        for t in frame_texts:
            detector.q.put(b"chunk")

        results_iter = iter(frame_texts)
        recognizer.AcceptWaveform.side_effect = lambda data: True
        recognizer.Result.side_effect = lambda: json.dumps({"text": next(results_iter)})

        return detector.listen_for_wake_word(timeout=timeout)

    def test_returns_true_and_emits_event_on_exact_match(self, monkeypatch):
        detector, recognizer = _make_detector(monkeypatch, wake_words=["hestia"])
        emitted = {}
        monkeypatch.setattr(
            wake_word.bus, "emit", lambda event, data=None: emitted.update(event=event, data=data)
        )
        result = self._run_with_frames(monkeypatch, detector, recognizer, ["hestia"])
        assert result is True
        assert emitted["event"] == "wake_detected"
        assert emitted["data"] == {"text": "hestia"}

    def test_matches_wake_phrase_embedded_mid_sentence(self, monkeypatch):
        detector, recognizer = _make_detector(monkeypatch, wake_words=["hey hestia"])
        monkeypatch.setattr(wake_word.bus, "emit", MagicMock())
        result = self._run_with_frames(
            monkeypatch, detector, recognizer, ["can you hey hestia turn on the lights"]
        )
        assert result is True

    def test_does_not_match_partial_token_overlap(self, monkeypatch):
        # "hestias" (plural) should NOT match the "hestia" token via substring
        # matching, since matching is token-based, not substring-based.
        detector, recognizer = _make_detector(monkeypatch, wake_words=["hestia"])
        monkeypatch.setattr(wake_word.sd, "RawInputStream", _NullStream)
        _disable_initial_flush(monkeypatch, detector)
        detector.q.put(b"chunk1")
        recognizer.AcceptWaveform.side_effect = [True, False]
        recognizer.Result.return_value = json.dumps({"text": "hestias"})
        result = detector.listen_for_wake_word(timeout=0.3)
        assert result is False

    def test_times_out_and_returns_false_with_no_speech(self, monkeypatch):
        detector, recognizer = _make_detector(monkeypatch, wake_words=["hestia"])
        monkeypatch.setattr(wake_word.sd, "RawInputStream", _NullStream)
        recognizer.AcceptWaveform.return_value = False
        result = detector.listen_for_wake_word(timeout=0.2)
        assert result is False

    def test_ignores_empty_transcription_and_keeps_listening(self, monkeypatch):
        detector, recognizer = _make_detector(monkeypatch, wake_words=["hestia"])
        monkeypatch.setattr(wake_word.sd, "RawInputStream", _NullStream)
        _disable_initial_flush(monkeypatch, detector)
        detector.q.put(b"silence")
        detector.q.put(b"chunk")
        recognizer.AcceptWaveform.return_value = True
        recognizer.Result.side_effect = [
            json.dumps({"text": ""}),
            json.dumps({"text": "hestia"}),
        ]
        monkeypatch.setattr(wake_word.bus, "emit", MagicMock())
        result = detector.listen_for_wake_word(timeout=2)
        assert result is True

    def test_malformed_json_result_is_skipped_without_raising(self, monkeypatch):
        detector, recognizer = _make_detector(monkeypatch, wake_words=["hestia"])
        monkeypatch.setattr(wake_word.sd, "RawInputStream", _NullStream)
        _disable_initial_flush(monkeypatch, detector)
        detector.q.put(b"garbled")
        detector.q.put(b"chunk")
        recognizer.AcceptWaveform.return_value = True
        recognizer.Result.side_effect = ["not valid json{{{", json.dumps({"text": "hestia"})]
        monkeypatch.setattr(wake_word.bus, "emit", MagicMock())
        result = detector.listen_for_wake_word(timeout=2)
        assert result is True

    def test_flushes_queue_before_and_after_listening(self, monkeypatch):
        detector, recognizer = _make_detector(monkeypatch, wake_words=["hestia"])
        monkeypatch.setattr(wake_word.sd, "RawInputStream", _NullStream)
        _disable_initial_flush(monkeypatch, detector)
        detector.q.put(b"stale-leftover-from-last-call")
        recognizer.AcceptWaveform.return_value = True
        recognizer.Result.return_value = json.dumps({"text": "hestia"})
        monkeypatch.setattr(wake_word.bus, "emit", MagicMock())
        detector.listen_for_wake_word(timeout=2)
        # After detection, flush_audio_queue() is called again, so the
        # queue should end up empty (no leftover frames from mid-utterance).
        assert detector.q.empty()
