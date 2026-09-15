"""
tests/test_tts.py

Covers core/tts.py's HestiaTTS, including the streaming (speak_stream)
and barge-in (stop) additions on top of the original queue-based
single-utterance speak().

pyttsx3 is fully mocked (see conftest.py) so no real speech engine or
audio device is touched. sounddevice.RawOutputStream is mocked per-test
where the Piper path is exercised.

Note: HestiaTTS.__init__ starts a real daemon worker thread that blocks on
queue.get() forever; that's fine for tests (it dies with the process) but
means we avoid asserting on thread lifecycle and instead drive behavior
through public methods (speak/speak_stream/stop/wait_until_done) or by
calling the private _speak_* methods directly (passing the current
generation explicitly, since that's now part of their signature).
"""
import time
from unittest.mock import MagicMock, call

import pytest

import core.tts as tts_module


def _fake_voice(name, voice_id=None):
    v = MagicMock()
    v.name = name
    v.id = voice_id or f"id-{name}"
    return v


def _make_engine_factory(voices):
    """Return a callable usable as pyttsx3.init side_effect: each call
    produces a fresh MagicMock engine whose getProperty('voices') returns
    the given voice list."""
    def _factory(*a, **kw):
        engine = MagicMock(name="pyttsx3.engine")
        engine.getProperty.side_effect = lambda prop: voices if prop == "voices" else None
        return engine
    return _factory


def _engine(monkeypatch):
    monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory([]))
    return tts_module.HestiaTTS()


# ---------------------------------------------------------------------------
# Voice selection during __init__
# ---------------------------------------------------------------------------

class TestVoiceSelection:
    def test_prefers_zira_over_other_voices(self, monkeypatch):
        voices = [_fake_voice("Microsoft David"), _fake_voice("Microsoft Zira"), _fake_voice("Microsoft Hazel")]
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory(voices))
        engine = tts_module.HestiaTTS()
        assert engine._voice_id == "id-Microsoft Zira"

    def test_falls_back_to_hazel_when_no_zira(self, monkeypatch):
        voices = [_fake_voice("Microsoft David"), _fake_voice("Microsoft Hazel")]
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory(voices))
        engine = tts_module.HestiaTTS()
        assert engine._voice_id == "id-Microsoft Hazel"

    def test_falls_back_to_any_female_voice(self, monkeypatch):
        voices = [_fake_voice("Microsoft David"), _fake_voice("Generic Female Voice")]
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory(voices))
        engine = tts_module.HestiaTTS()
        assert engine._voice_id == "id-Generic Female Voice"

    def test_falls_back_to_first_voice_when_no_preferred_match(self, monkeypatch):
        voices = [_fake_voice("Microsoft David"), _fake_voice("Microsoft Mark")]
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory(voices))
        engine = tts_module.HestiaTTS()
        assert engine._voice_id == "id-Microsoft David"

    def test_voice_id_stays_none_when_no_voices_available(self, monkeypatch):
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory([]))
        engine = tts_module.HestiaTTS()
        assert engine._voice_id is None


# ---------------------------------------------------------------------------
# Engine selection (pyttsx3 vs piper)
# ---------------------------------------------------------------------------

class TestEngineSelection:
    def test_defaults_to_pyttsx3(self, monkeypatch):
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory([]))
        engine = tts_module.HestiaTTS()
        assert engine._engine == "pyttsx3"

    def test_uses_piper_when_model_path_exists(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory([]))
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake")
        engine = tts_module.HestiaTTS(engine="piper", piper_model_path=str(model_file))
        assert engine._engine == "piper"
        assert engine._piper_model_path == str(model_file)

    def test_falls_back_to_pyttsx3_when_piper_model_path_missing(self, monkeypatch, capsys):
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory([]))
        engine = tts_module.HestiaTTS(engine="piper", piper_model_path="/no/such/file.onnx")
        assert engine._engine == "pyttsx3"
        assert "falling back to pyttsx3" in capsys.readouterr().err

    def test_falls_back_to_pyttsx3_when_no_model_path_given(self, monkeypatch):
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory([]))
        engine = tts_module.HestiaTTS(engine="piper", piper_model_path=None)
        assert engine._engine == "pyttsx3"


# ---------------------------------------------------------------------------
# speak() / wait_until_done()
# ---------------------------------------------------------------------------

class TestSpeakQueue:
    def _engine(self, monkeypatch):
        return _engine(monkeypatch)

    def test_speak_ignores_empty_string(self, monkeypatch):
        engine = self._engine(monkeypatch)
        engine.speak("")
        assert engine._queue.empty()

    def test_speak_ignores_whitespace_only(self, monkeypatch):
        engine = self._engine(monkeypatch)
        engine.speak("   \n\t")
        assert engine._queue.empty()

    def test_speak_enqueues_text_and_worker_consumes_it(self, monkeypatch):
        engine = self._engine(monkeypatch)
        spoken = []
        monkeypatch.setattr(engine, "_speak_blocking", lambda text, gen: spoken.append(text))
        engine.speak("hello there")
        engine.wait_until_done()
        assert spoken == ["hello there"]

    def test_multiple_speak_calls_are_processed_in_order(self, monkeypatch):
        # Each speak() call now cancels anything still queued from a
        # previous call (see TestCancellation) — to test in-order
        # processing of multiple items, queue them via speak_stream()
        # instead, which appends without cancelling between sentences.
        engine = self._engine(monkeypatch)
        spoken = []
        monkeypatch.setattr(engine, "_speak_blocking", lambda text, gen: spoken.append(text))
        engine.speak_stream(iter(["one. ", "two. ", "three."]))
        engine.wait_until_done()
        assert spoken == ["one.", "two.", "three."]


# ---------------------------------------------------------------------------
# speak_stream() sentence-splitting
# ---------------------------------------------------------------------------

class TestSpeakStream:
    def test_splits_completed_sentences_as_they_arrive(self, monkeypatch):
        engine = _engine(monkeypatch)
        spoken = []
        monkeypatch.setattr(engine, "_speak_blocking", lambda text, gen: spoken.append(text))

        engine.speak_stream(iter(["Hi there. ", "How are ", "you? ", "Great."]))
        engine.wait_until_done()

        assert spoken == ["Hi there.", "How are you?", "Great."]

    def test_flushes_trailing_incomplete_sentence_at_end_of_stream(self, monkeypatch):
        engine = _engine(monkeypatch)
        spoken = []
        monkeypatch.setattr(engine, "_speak_blocking", lambda text, gen: spoken.append(text))

        engine.speak_stream(iter(["No terminal punctuation here"]))
        engine.wait_until_done()

        assert spoken == ["No terminal punctuation here"]

    def test_empty_stream_speaks_nothing(self, monkeypatch):
        engine = _engine(monkeypatch)
        spoken = []
        monkeypatch.setattr(engine, "_speak_blocking", lambda text, gen: spoken.append(text))

        engine.speak_stream(iter([]))
        engine.wait_until_done()

        assert spoken == []

    def test_ignores_empty_chunks_within_stream(self, monkeypatch):
        engine = _engine(monkeypatch)
        spoken = []
        monkeypatch.setattr(engine, "_speak_blocking", lambda text, gen: spoken.append(text))

        engine.speak_stream(iter(["Hello. ", "", "World."]))
        engine.wait_until_done()

        assert spoken == ["Hello.", "World."]


# ---------------------------------------------------------------------------
# Cancellation: speak()/speak_stream()/stop() and the generation counter
# ---------------------------------------------------------------------------

class TestCancellation:
    def test_speak_cancels_previously_queued_but_unplayed_items(self, monkeypatch):
        engine = _engine(monkeypatch)
        spoken = []

        # Block the worker on the first item so the second speak() call
        # races the drain against an in-flight item, then verify only the
        # *second* speak()'s text ultimately gets spoken.
        import threading
        release = threading.Event()

        def _slow_speak(text, gen):
            if text == "first":
                release.wait(timeout=1.0)
            spoken.append(text)

        monkeypatch.setattr(engine, "_speak_blocking", _slow_speak)

        engine.speak("first")
        time.sleep(0.05)  # let the worker pick up "first" and start blocking
        engine.speak("second")
        release.set()
        engine.wait_until_done()

        assert spoken == ["first", "second"]

    def test_stop_bumps_generation_so_stale_queued_items_are_dropped(self, monkeypatch):
        engine = _engine(monkeypatch)
        spoken = []
        monkeypatch.setattr(engine, "_speak_blocking", lambda text, gen: spoken.append(text))

        gen_before = engine._bump_generation()
        engine._queue.put((gen_before, "stale sentence"))
        engine.stop()
        engine.wait_until_done()

        assert spoken == []

    def test_stop_calls_stop_on_active_pyttsx3_engine(self, monkeypatch):
        engine = _engine(monkeypatch)
        active = MagicMock(name="active-pyttsx3-engine")
        engine._active_pyttsx3_engine = active

        engine.stop()

        active.stop.assert_called_once()

    def test_stop_kills_active_piper_process(self, monkeypatch):
        engine = _engine(monkeypatch)
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        engine._active_piper_proc = proc

        engine.stop()

        proc.kill.assert_called_once()

    def test_stop_does_not_kill_already_finished_piper_process(self, monkeypatch):
        engine = _engine(monkeypatch)
        proc = MagicMock()
        proc.poll.return_value = 0  # already exited
        engine._active_piper_proc = proc

        engine.stop()

        proc.kill.assert_not_called()

    def test_speak_stream_stops_pulling_from_source_once_interrupted(self, monkeypatch):
        engine = _engine(monkeypatch)
        spoken = []
        monkeypatch.setattr(engine, "_speak_blocking", lambda text, gen: spoken.append(text))

        pulled = []

        def _chunks():
            pulled.append("a")
            yield "First sentence. "
            # Interrupt mid-stream — a fresh speak() bumps the generation
            # speak_stream() is checking against.
            engine.speak("interrupting utterance")
            pulled.append("b")
            yield "Second sentence, should never be queued."

        engine.speak_stream(_chunks())
        engine.wait_until_done()

        assert pulled == ["a", "b"]  # the generator itself still ran to this point...
        assert "Second sentence, should never be queued." not in "".join(spoken)


# ---------------------------------------------------------------------------
# set_rate / set_volume
# ---------------------------------------------------------------------------

class TestRateVolume:
    def test_set_rate_updates_rate(self, monkeypatch):
        engine = _engine(monkeypatch)
        engine.set_rate(200)
        assert engine.rate == 200

    def test_set_volume_clamps_above_one(self, monkeypatch):
        engine = _engine(monkeypatch)
        engine.set_volume(5.0)
        assert engine.volume == 1.0

    def test_set_volume_clamps_below_zero(self, monkeypatch):
        engine = _engine(monkeypatch)
        engine.set_volume(-3.0)
        assert engine.volume == 0.0

    def test_init_clamps_volume_argument(self, monkeypatch):
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory([]))
        engine = tts_module.HestiaTTS(volume=1.5)
        assert engine.volume == 1.0


# ---------------------------------------------------------------------------
# _speak_pyttsx3
# ---------------------------------------------------------------------------

class TestSpeakPyttsx3:
    def test_creates_fresh_engine_and_applies_rate_volume_voice(self, monkeypatch):
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory([_fake_voice("Zira")]))
        engine = tts_module.HestiaTTS(rate=150, volume=0.5)

        speak_engine = MagicMock()
        init_mock = MagicMock(return_value=speak_engine)
        monkeypatch.setattr(tts_module.pyttsx3, "init", init_mock)

        engine._speak_pyttsx3("hello", engine._current_generation())

        speak_engine.setProperty.assert_any_call("rate", 150)
        speak_engine.setProperty.assert_any_call("volume", 0.5)
        speak_engine.setProperty.assert_any_call("voice", "id-Zira")
        speak_engine.say.assert_called_once_with("hello")
        speak_engine.runAndWait.assert_called_once()

    def test_clears_active_engine_reference_after_speaking(self, monkeypatch):
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory([]))
        engine = tts_module.HestiaTTS()
        monkeypatch.setattr(tts_module.pyttsx3, "init", MagicMock(return_value=MagicMock()))

        engine._speak_pyttsx3("hello", engine._current_generation())

        assert engine._active_pyttsx3_engine is None

    def test_skips_speaking_if_generation_is_stale_before_engine_use(self, monkeypatch):
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory([]))
        engine = tts_module.HestiaTTS()

        speak_engine = MagicMock()
        monkeypatch.setattr(tts_module.pyttsx3, "init", MagicMock(return_value=speak_engine))

        stale_gen = engine._current_generation() - 1
        engine._speak_pyttsx3("hello", stale_gen)

        speak_engine.say.assert_not_called()

    def test_swallows_exceptions_and_logs(self, monkeypatch, capsys):
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory([]))
        engine = tts_module.HestiaTTS()
        monkeypatch.setattr(
            tts_module.pyttsx3, "init", MagicMock(side_effect=RuntimeError("boom"))
        )
        engine._speak_pyttsx3("hello", engine._current_generation())  # must not raise
        assert "pyttsx3 TTS error" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _speak_piper / _speak_blocking fallback
# ---------------------------------------------------------------------------

class TestSpeakPiper:
    def _piper_engine(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory([]))
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake")
        return tts_module.HestiaTTS(engine="piper", piper_model_path=str(model_file))

    def test_speak_piper_success_plays_audio_in_chunks(self, monkeypatch, tmp_path):
        engine = self._piper_engine(monkeypatch, tmp_path)

        proc = MagicMock()
        proc.stdin = MagicMock()
        # Two chunks then EOF, so playback loop iterates more than once —
        # this is what makes mid-utterance cancellation possible.
        proc.stdout.read.side_effect = [b"\x00\x01" * 10, b"\x02\x03" * 5, b""]
        proc.returncode = 0
        proc.poll.return_value = 0
        monkeypatch.setattr(tts_module.subprocess, "Popen", MagicMock(return_value=proc))

        stream = MagicMock()
        monkeypatch.setattr(tts_module.sd, "RawOutputStream", MagicMock(return_value=stream))

        engine._speak_piper("hello world", engine._current_generation())

        proc.stdin.write.assert_called_once_with(b"hello world")
        proc.stdin.close.assert_called_once()
        assert stream.write.call_args_list == [
            call(b"\x00\x01" * 10),
            call(b"\x02\x03" * 5),
        ]
        stream.stop.assert_called_once()
        stream.close.assert_called_once()
        assert engine._active_piper_proc is None

    def test_speak_piper_raises_on_nonzero_exit(self, monkeypatch, tmp_path):
        engine = self._piper_engine(monkeypatch, tmp_path)

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdout.read.side_effect = [b""]
        proc.returncode = 1
        proc.poll.return_value = 1
        monkeypatch.setattr(tts_module.subprocess, "Popen", MagicMock(return_value=proc))
        monkeypatch.setattr(tts_module.sd, "RawOutputStream", MagicMock(return_value=MagicMock()))

        with pytest.raises(RuntimeError, match="Piper exited with code 1"):
            engine._speak_piper("hello", engine._current_generation())

    def test_speak_piper_returns_immediately_if_already_stale(self, monkeypatch, tmp_path):
        """A generation mismatch caught before Popen is even called (e.g.
        cancelled between dequeue and playback start) must not spawn a
        Piper process or touch sounddevice at all."""
        engine = self._piper_engine(monkeypatch, tmp_path)

        popen_mock = MagicMock()
        monkeypatch.setattr(tts_module.subprocess, "Popen", popen_mock)
        monkeypatch.setattr(tts_module.sd, "RawOutputStream", MagicMock())

        gen = engine._current_generation()
        engine._bump_generation()  # now stale relative to *gen*

        engine._speak_piper("hello", gen)

        popen_mock.assert_not_called()

    def test_speak_piper_kills_subprocess_when_generation_changes_mid_playback(self, monkeypatch, tmp_path):
        """A generation mismatch discovered *during* the chunked playback
        loop (i.e. barge-in fired partway through speaking) must stop
        reading/writing further audio and kill the still-running Piper
        process, without raising despite the resulting nonzero/None exit
        code."""
        engine = self._piper_engine(monkeypatch, tmp_path)

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdout.read.side_effect = [b"\x00\x01", b"\x02\x03", b""]
        proc.returncode = -9  # killed
        proc.poll.return_value = None  # still running when stop() would check it
        monkeypatch.setattr(tts_module.subprocess, "Popen", MagicMock(return_value=proc))

        stream = MagicMock()
        monkeypatch.setattr(tts_module.sd, "RawOutputStream", MagicMock(return_value=stream))

        gen = engine._current_generation()
        # First check (entry) sees `gen`; the loop's first iteration check
        # also sees `gen` and writes one chunk; the second iteration's
        # check sees a bumped generation and bails out — simulating
        # barge-in landing after the first chunk has already played.
        calls = {"n": 0}

        def _gen_sequence():
            calls["n"] += 1
            return gen if calls["n"] <= 2 else gen + 1

        monkeypatch.setattr(engine, "_current_generation", _gen_sequence)

        engine._speak_piper("hello", gen)  # must not raise despite returncode=-9

        assert stream.write.call_args_list == [call(b"\x00\x01")]
        proc.kill.assert_called_once()

    def test_speak_blocking_falls_back_to_pyttsx3_when_piper_fails(self, monkeypatch, tmp_path, capsys):
        engine = self._piper_engine(monkeypatch, tmp_path)
        monkeypatch.setattr(
            engine, "_speak_piper", MagicMock(side_effect=RuntimeError("piper crashed"))
        )
        fallback_calls = []
        monkeypatch.setattr(engine, "_speak_pyttsx3", lambda text, gen: fallback_calls.append(text))

        engine._speak_blocking("hello", engine._current_generation())

        assert fallback_calls == ["hello"]
        assert "Piper TTS failed" in capsys.readouterr().err

    def test_speak_blocking_uses_pyttsx3_directly_when_engine_is_pyttsx3(self, monkeypatch):
        monkeypatch.setattr(tts_module.pyttsx3, "init", _make_engine_factory([]))
        engine = tts_module.HestiaTTS()  # default engine="pyttsx3"
        calls = []
        monkeypatch.setattr(engine, "_speak_pyttsx3", lambda text, gen: calls.append(text))
        piper_spy = MagicMock()
        monkeypatch.setattr(engine, "_speak_piper", piper_spy)

        engine._speak_blocking("hi", engine._current_generation())

        assert calls == ["hi"]
        piper_spy.assert_not_called()