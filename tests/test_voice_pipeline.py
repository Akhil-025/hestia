"""
tests/test_voice_pipeline.py

Backlog section 16 (Voice Pipeline): items #170-#176, #178, #179.

  #170 per-module TTS voices        #174 mic calibration
  #171 "repeat that"                #175 echo cancellation (synthetic audio)
  #172 streaming sentence splitter  #176 wake-word sensitivity
  #173 do-not-disturb               #178 listening-state indicator (API)
                                    #179 typed-input fallback / NullTTS

No microphone, speaker or speech engine is touched: pyttsx3 / sounddevice /
vosk are faked by conftest.py or per-test mocks.
"""
import json
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

import core.tts as tts_module
from core import mic_calibration as mc
from core.echo_cancel import (
    EchoReference,
    NLMSEchoCanceller,
    echo_return_loss_db,
)
from core.tts import NullTTS, split_speakable
from core.voice_commands import (
    DND_OFF,
    DND_ON,
    DND_STATUS,
    REPEAT,
    SET_SENSITIVITY,
    parse_duration_minutes,
    parse_voice_command,
)
from core.voice_state import (
    MAX_HELD_NOTIFICATIONS,
    STATE_LISTENING,
    STATE_SPEAKING,
    STATE_WAKE,
    VoiceState,
)

import main


# ===========================================================================
# #172 streaming sentence splitter
# ===========================================================================

class TestSplitSpeakable:
    def test_complete_sentence_and_remainder(self):
        assert split_speakable("Hello there. How are ") == (["Hello there."], "How are ")

    def test_abbreviation_is_not_a_boundary(self):
        sentences, rest = split_speakable("Dr. Patel is here. Then")
        assert sentences == ["Dr. Patel is here."]
        assert rest == "Then"

    def test_initials_are_not_boundaries(self):
        sentences, _ = split_speakable("J. R. Smith went home. Next")
        assert sentences == ["J. R. Smith went home."]

    def test_eg_is_not_a_boundary(self):
        sentences, rest = split_speakable("Use e.g. a list. And")
        assert sentences == ["Use e.g. a list."]

    def test_decimal_number_is_not_a_boundary(self):
        sentences, rest = split_speakable("Version 3.5 is out! Great")
        assert sentences == ["Version 3.5 is out!"]
        assert rest == "Great"

    def test_line_breaks_split_bullet_lists(self):
        sentences, rest = split_speakable("- one\n- two\n- thr")
        assert sentences == ["- one", "- two"]
        assert rest == "- thr"

    def test_long_unpunctuated_run_flushes_at_clause_break(self):
        long = ("This is a long sentence with many words, and it keeps going "
                "without any end punctuation at all, which would normally "
                "delay speech until the very end of the sentence arrives here")
        sentences, rest = split_speakable(long)
        assert sentences and sentences[0].endswith(",")
        assert (" ".join(sentences) + " " + rest).split() == long.split()

    def test_short_text_without_boundary_is_kept(self):
        assert split_speakable("no end yet") == ([], "no end yet")

    def test_empty(self):
        assert split_speakable("") == ([], "")


# ===========================================================================
# #170 per-module voice profiles + #179 NullTTS
# ===========================================================================

def _fake_voice(name):
    v = MagicMock()
    v.name = name
    v.id = f"id-{name}"
    return v


def _pyttsx3_engine(monkeypatch, voices, voices_cfg=None):
    def factory(*a, **kw):
        eng = MagicMock()
        eng.getProperty.side_effect = lambda p: voices if p == "voices" else None
        return eng
    monkeypatch.setattr(tts_module.pyttsx3, "init", factory)
    return tts_module.HestiaTTS(voices=voices_cfg)


class TestVoiceProfiles:
    def test_profile_resolves_named_voice_and_overrides(self, monkeypatch):
        voices = [_fake_voice("Microsoft Zira"), _fake_voice("Microsoft Hazel")]
        t = _pyttsx3_engine(monkeypatch, voices,
                            {"calm": {"voice_name": "hazel", "rate": 150, "volume": 5}})
        assert t.has_voice("calm")
        assert t.voice_profiles == ["calm"]
        p = t._profile("calm")
        assert p["voice_id"] == "id-Microsoft Hazel"
        assert p["rate"] == 150
        assert p["volume"] == 1.0          # clamped

    def test_unknown_voice_name_falls_back_with_warning(self, monkeypatch, capsys):
        t = _pyttsx3_engine(monkeypatch, [_fake_voice("Microsoft Zira")],
                            {"x": {"voice_name": "Nonexistent", "rate": 160}})
        assert t._profile("x")["voice_id"] is None
        assert t._profile("x")["rate"] == 160     # rest of profile kept
        assert "no pyttsx3 voice matching" in capsys.readouterr().err

    def test_missing_piper_model_in_profile_is_dropped(self, monkeypatch, capsys):
        t = _pyttsx3_engine(monkeypatch, [], {"x": {"piper_model_path": "/nope.onnx"}})
        assert t._profile("x")["piper_model_path"] is None
        assert "not found" in capsys.readouterr().err

    def test_unknown_or_none_profile_is_empty(self, monkeypatch):
        t = _pyttsx3_engine(monkeypatch, [])
        assert t._profile(None) == {}
        assert t._profile("ghost") == {}
        assert not t.has_voice("ghost")

    def test_non_dict_profile_ignored(self, monkeypatch):
        t = _pyttsx3_engine(monkeypatch, [], {"bad": "oops"})
        assert t.voice_profiles == []

    def test_speak_carries_voice_to_worker(self, monkeypatch):
        t = _pyttsx3_engine(monkeypatch, [], {"calm": {"rate": 150}})
        seen = []
        monkeypatch.setattr(t, "_speak_blocking", lambda *a: seen.append(a))
        # speak() starts a new generation and cancels the previous one, so
        # let each finish before the next.
        t.speak("hello", voice="calm")
        t.wait_until_done()
        t.speak("plain")
        t.wait_until_done()
        assert seen[0][0] == "hello" and seen[0][2] == "calm"
        assert len(seen[1]) == 2             # default path keeps (text, gen)

    def test_speak_stream_carries_voice(self, monkeypatch):
        t = _pyttsx3_engine(monkeypatch, [], {"calm": {"rate": 150}})
        seen = []
        monkeypatch.setattr(t, "_speak_blocking", lambda *a: seen.append(a))
        t.speak_stream(iter(["One. ", "Two."]), voice="calm")
        t.wait_until_done()
        assert [a[0] for a in seen] == ["One.", "Two."]
        assert all(a[2] == "calm" for a in seen)

    def test_worker_survives_an_engine_exception(self, monkeypatch):
        t = _pyttsx3_engine(monkeypatch, [])
        calls = []

        def boom(text, gen, *a):
            calls.append(text)
            if text == "first":
                raise RuntimeError("audio device gone")
        monkeypatch.setattr(t, "_speak_blocking", boom)
        t.speak("first")
        t.wait_until_done()
        t.speak("second")
        t.wait_until_done()
        assert calls == ["first", "second"]


class TestNullTTS:
    def test_is_silent_and_unavailable(self):
        n = NullTTS()
        assert n.available is False
        n.speak("x"); n.speak("x", voice="v"); n.stop(); n.wait_until_done()
        assert n.has_voice("anything") is False
        assert n.synthesize_wav_bytes("x") == b""

    def test_speak_stream_still_drains_iterator(self):
        consumed = []

        def gen():
            for c in ("a", "b", "c"):
                consumed.append(c)
                yield c
        NullTTS().speak_stream(gen())
        assert consumed == ["a", "b", "c"]


# ===========================================================================
# #173 / #178 VoiceState
# ===========================================================================

class _Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


class TestVoiceState:
    def test_mic_open_derivation(self):
        vs = VoiceState()
        assert vs.snapshot()["mic_open"] is False and vs.snapshot()["active"] is False
        vs.set_state(STATE_WAKE)
        assert vs.snapshot()["mic_open"] is True
        vs.set_state(STATE_LISTENING)
        assert vs.snapshot()["mic_open"] is True
        vs.set_state(STATE_SPEAKING)
        assert vs.snapshot()["mic_open"] is False      # not armed
        vs.set_barge_in_armed(True)
        assert vs.snapshot()["mic_open"] is True       # barge-in listening
        vs.set_state(STATE_WAKE)
        vs.set_state(STATE_SPEAKING)
        assert vs.snapshot()["mic_open"] is False      # arming reset on leave

    def test_unknown_state_rejected(self):
        with pytest.raises(ValueError):
            VoiceState().set_state("dancing")

    def test_dnd_open_ended(self):
        vs = VoiceState()
        vs.set_dnd(True)
        assert vs.dnd_active() and vs.dnd_remaining_minutes() is None

    def test_timed_dnd_expires_lazily(self):
        clk = _Clock()
        vs = VoiceState(clock=clk)
        vs.set_dnd(True, 30)
        assert vs.dnd_active()
        assert vs.dnd_remaining_minutes() == pytest.approx(30)
        clk.t += 31 * 60
        assert not vs.dnd_active()
        assert vs.snapshot()["dnd"] is False

    def test_hold_and_release_in_order(self):
        vs = VoiceState()
        vs.hold("a"); vs.hold("  "); vs.hold("b")
        assert vs.held_count() == 2
        assert vs.release_held() == ["a", "b"]
        assert vs.held_count() == 0

    def test_held_notifications_are_capped(self):
        vs = VoiceState()
        for i in range(MAX_HELD_NOTIFICATIONS + 5):
            vs.hold(f"n{i}")
        held = vs.release_held()
        assert len(held) == MAX_HELD_NOTIFICATIONS
        assert held[-1] == f"n{MAX_HELD_NOTIFICATIONS + 4}"
        assert held[0] == "n5"                          # oldest dropped

    def test_turning_dnd_off_keeps_held(self):
        vs = VoiceState()
        vs.set_dnd(True); vs.hold("x"); vs.set_dnd(False)
        assert vs.held_count() == 1


# ===========================================================================
# #171 / #173 / #176 local voice commands
# ===========================================================================

class TestParseVoiceCommand:
    @pytest.mark.parametrize("text", [
        "repeat that", "Repeat that.", "say it again please", "say that again",
        "what did you say", "can you repeat that", "Hey Hestia, say that again",
        "pardon",
    ])
    def test_repeat(self, text):
        assert parse_voice_command(text).name == REPEAT

    @pytest.mark.parametrize("text", [
        "can you repeat that recipe for pancakes",
        "repeat after me hello",
        "what's the weather",
        "quiet",
        "I'm back from the shops, add milk to my list",
        "",
    ])
    def test_ordinary_queries_not_hijacked(self, text):
        assert parse_voice_command(text) is None

    @pytest.mark.parametrize("text", [
        "do not disturb", "enable do not disturb mode", "don't disturb me",
        "dnd", "quiet mode", "mute notifications", "silence my reminders",
    ])
    def test_dnd_on_open_ended(self, text):
        c = parse_voice_command(text)
        assert c.name == DND_ON and c.minutes is None and c.unparsed is None

    @pytest.mark.parametrize("text,minutes", [
        ("do not disturb for 30 minutes", 30),
        ("mute notifications for an hour", 60),
        ("quiet mode for half an hour", 30),
        ("pause notifications for two hours", 120),
        ("do not disturb for 1 hour 30 minutes", 90),
        ("do not disturb for 45 mins", 45),
    ])
    def test_dnd_on_with_duration(self, text, minutes):
        c = parse_voice_command(text)
        assert c.name == DND_ON and c.minutes == minutes

    def test_dnd_unreadable_duration_is_flagged_not_guessed(self):
        c = parse_voice_command("mute notifications for the afternoon")
        assert c.name == DND_ON and c.minutes is None and c.unparsed == "the afternoon"

    @pytest.mark.parametrize("text", [
        "resume notifications", "turn off do not disturb", "unmute reminders",
        "I'm back", "disable quiet mode",
    ])
    def test_dnd_off(self, text):
        assert parse_voice_command(text).name == DND_OFF

    def test_dnd_status(self):
        assert parse_voice_command("is do not disturb on?").name == DND_STATUS

    @pytest.mark.parametrize("text,level", [
        ("I'm in a noisy room", "noisy"),
        ("I'm in a quiet room", "quiet"),
        ("wake word sensitivity quiet", "quiet"),
        ("set sensitivity to noisy", "noisy"),
        ("noisy room mode", "noisy"),
    ])
    def test_sensitivity(self, text, level):
        c = parse_voice_command(text)
        assert c.name == SET_SENSITIVITY and c.level == level

    def test_duration_parser_rejects_garbage(self):
        assert parse_duration_minutes("soon") is None
        assert parse_duration_minutes("5 minutes banana") is None
        assert parse_duration_minutes("") is None


# ===========================================================================
# #174 mic calibration
# ===========================================================================

class TestMicCalibration:
    def _rng(self):
        return np.random.default_rng(0)

    def test_recommend_quiet_room(self):
        r = self._rng()
        noise = list(r.normal(60, 10, 100))
        speech = list(r.normal(60, 10, 20)) + list(r.normal(2500, 400, 60))
        rec = mc.recommend_settings(noise, speech, device="Mic A")
        assert rec.device == "Mic A"
        assert rec.vad_aggressiveness == 2
        assert rec.wake_sensitivity == "quiet"
        assert mc.MIN_RMS_FLOOR <= rec.min_rms < 500
        assert rec.speech_rms_p25 and rec.speech_rms_p25 > 1000

    def test_echo_raises_floor_above_noise_only_value(self):
        r = self._rng()
        noise = list(r.normal(60, 10, 100))
        speech = list(r.normal(2500, 400, 60))
        echo = list(r.normal(500, 60, 100))
        without = mc.recommend_settings(noise, speech)
        with_echo = mc.recommend_settings(noise, speech, echo)
        assert with_echo.min_rms > without.min_rms
        assert with_echo.echo_rms_p95 is not None

    def test_noisy_room_gets_stricter_vad_and_sensitivity(self):
        noise = list(self._rng().normal(900, 100, 100))
        rec = mc.recommend_settings(noise)
        assert rec.vad_aggressiveness == 3 and rec.wake_sensitivity == "noisy"

    def test_floor_never_exceeds_ceiling_and_note_when_echo_rivals_voice(self):
        r = self._rng()
        noise = list(r.normal(60, 10, 100))
        speech = list(r.normal(1000, 100, 80))
        echo = list(r.normal(1200, 100, 100))
        rec = mc.recommend_settings(noise, speech, echo)
        assert rec.min_rms <= rec.speech_rms_p25 * mc.SPEECH_CEILING_FRACTION + 1
        assert any("headset" in n for n in rec.notes)

    def test_too_quiet_speech_adds_note(self):
        noise = [60.0] * 50
        rec = mc.recommend_settings(noise, [61.0] * 50)
        assert rec.speech_rms_p25 is None
        assert any("too quiet" in n for n in rec.notes)

    def test_empty_noise_returns_defaults(self):
        rec = mc.recommend_settings([])
        assert rec.min_rms == 300.0 and rec.notes

    def test_frame_rms_values(self):
        audio = np.full(960, 100, dtype=np.int16)
        assert mc.frame_rms_values(audio) == [pytest.approx(100.0)] * 2

    def test_save_load_round_trip(self, tmp_path):
        p = str(tmp_path / "cal.json")
        rec = mc.recommend_settings([60.0] * 50, device="USB Mic")
        mc.save_calibration(rec, p)
        loaded = mc.load_calibration(p, device="USB Mic")
        assert loaded == {
            "min_rms": rec.min_rms,
            "vad_aggressiveness": rec.vad_aggressiveness,
            "wake_sensitivity": rec.wake_sensitivity,
        }

    def test_load_ignores_calibration_from_other_device(self, tmp_path):
        p = str(tmp_path / "cal.json")
        mc.save_calibration(mc.recommend_settings([60.0] * 50, device="Laptop Mic"), p)
        assert mc.load_calibration(p, device="USB Headset") is None
        assert mc.load_calibration(p, device="") is not None   # unknown device: trust it

    @pytest.mark.parametrize("content", ["", "not json", "[]", '{"min_rms": "x"}', "{}"])
    def test_load_never_raises_on_bad_file(self, tmp_path, content):
        p = tmp_path / "cal.json"
        p.write_text(content)
        assert mc.load_calibration(str(p)) is None

    def test_load_missing_file(self, tmp_path):
        assert mc.load_calibration(str(tmp_path / "nope.json")) is None

    def test_run_calibration_with_fake_recorder_and_tts(self):
        r = self._rng()
        seq = iter([
            list(r.normal(60, 10, 100)),                                   # quiet
            list(r.normal(60, 10, 10)) + list(r.normal(2500, 300, 60)),    # speech
            list(r.normal(400, 40, 100)),                                  # echo
        ])
        tts = MagicMock()
        said = []
        rec = mc.run_calibration(
            tts=tts, recorder=lambda secs: next(seq),
            say=said.append, wait=lambda _p: "", device="Fake Mic",
        )
        tts.speak.assert_called_once()
        tts.stop.assert_called_once()
        assert rec.echo_rms_p95 is not None and rec.device == "Fake Mic"
        assert len(said) == 3

    def test_run_calibration_without_tts_skips_echo(self):
        seq = iter([[60.0] * 50, [2000.0] * 50])
        rec = mc.run_calibration(
            tts=None, recorder=lambda s: next(seq), say=lambda m: None,
            wait=lambda p: "", device="x",
        )
        assert rec.echo_rms_p95 is None

    def test_format_report_mentions_settings(self):
        text = mc.format_report(mc.recommend_settings([60.0] * 50, device="Mic"))
        assert "min_rms:" in text and "vad_aggressiveness:" in text and "sensitivity:" in text


# ===========================================================================
# #175 echo cancellation (synthetic echo)
# ===========================================================================

def _simulate_echo(delay, gain, noise=0.0, filt_len=256, filt_delay_ms=0.0, secs=3, seed=1):
    rng = np.random.default_rng(seed)
    ref = EchoReference(samplerate=16000)
    ec = NLMSEchoCanceller(ref, filter_len=filt_len, mu=0.5, delay_ms=filt_delay_ms)
    sig = rng.normal(0, 3000, 16000 * secs)
    mic = np.zeros_like(sig)
    mic[delay:] = gain * sig[:-delay]
    mic += rng.normal(0, noise, len(sig)) if noise else 0
    out = []
    for i in range(0, len(sig) - 480, 480):
        ref.push(sig[i:i + 480].astype(np.int16), 16000)
        d = mic[i:i + 480].astype(np.int16)
        out.append(echo_return_loss_db(d, ec.process(d)))
    return out


class TestEchoCancel:
    def test_converges_on_pure_echo(self):
        erle = _simulate_echo(delay=20, gain=0.5)
        assert np.mean(erle[-10:]) > 30

    def test_reduces_echo_with_noise_present(self):
        erle = _simulate_echo(delay=300, gain=0.5, noise=100, filt_len=1024)
        assert np.mean(erle[-10:]) > 15

    def test_delay_beyond_filter_needs_delay_ms(self):
        assert np.mean(_simulate_echo(delay=500, gain=0.5)[-5:]) < 3
        assert np.mean(_simulate_echo(delay=500, gain=0.5, filt_delay_ms=25)[-5:]) > 30

    def test_passthrough_when_nothing_is_playing(self):
        ref = EchoReference()
        ec = NLMSEchoCanceller(ref)
        frame = (np.arange(480) % 200).astype(np.int16)
        assert np.array_equal(ec.process(frame), frame)

    def test_passthrough_when_reference_is_stale(self):
        ref = EchoReference()
        ref.push(np.ones(480, dtype=np.int16) * 1000, 16000)
        ec = NLMSEchoCanceller(ref, stale_after=0.01)
        time.sleep(0.05)
        frame = np.ones(480, dtype=np.int16) * 500
        assert np.array_equal(ec.process(frame), frame)

    def test_user_speech_during_playback_is_not_cancelled(self):
        """Double-talk: a mic far louder than the reference must survive."""
        rng = np.random.default_rng(3)
        ref = EchoReference()
        ec = NLMSEchoCanceller(ref, filter_len=128)
        quiet_ref = rng.normal(0, 100, 480)
        loud_user = rng.normal(0, 8000, 480)
        for _ in range(30):
            ref.push(quiet_ref.astype(np.int16), 16000)
            out = ec.process(loud_user.astype(np.int16))
        assert echo_return_loss_db(loud_user, out) < 3
        assert float(np.abs(ec.w).max()) < 1.0       # didn't learn to cancel the user

    def test_reference_resamples_to_mic_rate(self):
        ref = EchoReference(samplerate=16000)
        ref.push(np.zeros(22050, dtype=np.int16), source_rate=22050)
        assert 15900 <= ref.tail(16000).size <= 16100
        assert ref._buf.size == pytest.approx(16000, abs=2)

    def test_reference_accepts_bytes_and_caps_length(self):
        ref = EchoReference(samplerate=16000, max_seconds=0.5)
        ref.push((np.ones(16000, dtype=np.int16) * 7).tobytes(), 16000)
        assert ref._buf.size == 8000

    def test_tail_pads_when_short(self):
        ref = EchoReference()
        ref.push(np.ones(10, dtype=np.int16), 16000)
        t = ref.tail(30)
        assert t.size == 30 and t[:20].sum() == 0 and t[20:].sum() == 10

    def test_output_is_clipped_int16(self):
        ref = EchoReference()
        ec = NLMSEchoCanceller(ref)
        ref.push(np.ones(480, dtype=np.int16), 16000)
        out = ec.process(np.full(480, 32767, dtype=np.int16))
        assert out.dtype == np.int16

    def test_barge_in_runs_detection_on_cleaned_audio(self):
        from core.barge_in import BargeInListener

        class _Shout:
            def process(self, frame):
                return np.zeros_like(frame)         # "removes everything"

        loud = np.full((480, 1), 5000, dtype=np.int16)
        b = BargeInListener(min_rms=300, echo_canceller=_Shout())
        # the canceller zeroes the frame, so the RMS gate sees silence
        assert b._frame_rms(b._clean(loud)) == 0.0
        assert np.array_equal(loud, np.full((480, 1), 5000, dtype=np.int16))  # raw intact

    def test_barge_in_ignores_a_crashing_canceller(self):
        from core.barge_in import BargeInListener

        class _Broken:
            def process(self, frame):
                raise RuntimeError("boom")

        data = np.full((480, 1), 123, dtype=np.int16)
        assert np.array_equal(BargeInListener(echo_canceller=_Broken())._clean(data), data)


# ===========================================================================
# #176 wake-word sensitivity
# ===========================================================================

@pytest.fixture
def detector(monkeypatch):
    import core.wake_word as ww
    monkeypatch.setattr(ww.os.path, "exists", lambda p: True)
    monkeypatch.setattr(ww.vosk, "Model", MagicMock(), raising=False)
    monkeypatch.setattr(ww.vosk, "KaldiRecognizer", MagicMock(), raising=False)
    return ww.WakeWordDetector


class TestWakeWordSensitivity:
    def test_normal_is_exact_match_only(self, detector):
        d = detector()
        assert d.sensitivity == "normal"
        assert d._match_window("hey hestia".split(), {}) is not None
        assert d._match_window("hello hesta".split(), {}) is None

    def test_quiet_accepts_near_misses_but_not_lookalikes(self, detector):
        d = detector(sensitivity="quiet")
        assert d._match_window("hello hesta here".split(), {}) is not None
        assert d._match_window("hey hestiya".split(), {}) is not None
        for word in ("history lesson", "estonia is nice", "pasta tonight", "hey asia"):
            assert d._match_window(word.split(), {}) is None, word

    def test_noisy_rejects_long_utterances(self, detector):
        d = detector(sensitivity="noisy")
        assert d._match_window("so the thing is hestia was there you know".split(), {}) is None
        assert d._match_window("hey hestia".split(), {}) is not None

    def test_noisy_gates_on_word_confidence(self, detector):
        d = detector(sensitivity="noisy")
        low = {"result": [{"word": "hey", "conf": 0.9}, {"word": "hestia", "conf": 0.4}]}
        high = {"result": [{"word": "hey", "conf": 0.9}, {"word": "hestia", "conf": 0.95}]}
        assert d._match_window("hey hestia".split(), low) is None
        assert d._match_window("hey hestia".split(), high) is not None

    def test_noisy_without_word_results_does_not_block(self, detector):
        d = detector(sensitivity="noisy")
        assert d._match_window("hey hestia".split(), {}) is not None

    def test_set_sensitivity_at_runtime_and_aliases(self, detector):
        d = detector()
        assert d.set_sensitivity("noisy") == "noisy"
        assert d.set_sensitivity("HIGH") == "quiet"
        assert d.set_sensitivity("bogus") == "normal"

    def test_unknown_sensitivity_warns(self, detector, capsys):
        d = detector(sensitivity="bogus")
        assert d.sensitivity == "normal"
        assert "Unknown sensitivity" in capsys.readouterr().err

    def test_short_wake_words_are_never_fuzzy_matched(self, detector):
        d = detector(wake_words=["hey"], sensitivity="quiet")
        assert d._match_window("hex".split(), {}) is None


# ===========================================================================
# Hestia integration: commands, DND, voices, fallback, build_io
# ===========================================================================

def make_hestia(barge_in_enabled=True):
    h = object.__new__(main.Hestia)
    h.mnemosyne = MagicMock()
    h.mnemosyne.get_recent.return_value = []
    h.nlu = MagicMock()
    h.nlu.understand.return_value = {"intent": "chat", "entities": {}, "response": ""}
    h.orchestrator = MagicMock()
    h.orchestrator.try_stream_chat.return_value = None
    h.orchestrator.dispatch.return_value = "a plain response"
    h.tts = MagicMock()
    h.wake_detector = MagicMock()
    h.stt = MagicMock()
    h.barge_in = MagicMock()
    h.barge_in.consume_triggered.return_value = False
    h.barge_in.consume_captured_audio.return_value = None
    h._barge_in_enabled = barge_in_enabled
    return h


class TestRepeatThat:
    def test_repeat_says_the_last_reply_without_calling_nlu(self):
        h = make_hestia()
        h.process_text("tell me something")
        h.nlu.understand.reset_mock()
        h.tts.speak.reset_mock()
        out = h.process_text("repeat that")
        assert out == "a plain response"
        h.tts.speak.assert_called_once_with("a plain response")
        h.nlu.understand.assert_not_called()

    def test_repeat_before_anything_said(self):
        h = make_hestia()
        assert h.process_text("repeat that") == "I haven't said anything yet."

    def test_repeating_does_not_overwrite_what_is_repeated(self):
        h = make_hestia()
        h.process_text("hello there")
        h.process_text("repeat that")
        assert h.process_text("say that again") == "a plain response"

    def test_voice_turn_repeat_speaks_with_barge_in(self):
        h = make_hestia()
        h.process_voice_turn("tell me something")
        h.tts.speak.reset_mock()
        assert h.process_voice_turn("repeat that") == "a plain response"
        h.tts.speak.assert_called_once_with("a plain response")
        h.barge_in.start.assert_called()


class TestDoNotDisturb:
    def _speak_handler(self, h):
        """Register Hestia's real bus handlers and return its 'speak' one."""
        from core.event_bus import bus
        before = set(bus.listeners_for("speak"))
        h.config = {}
        h._init_event_bus()
        new = [fn for fn in bus.listeners_for("speak") if fn not in before]
        assert new, "speak handler was not registered"
        handler = new[-1]
        # Don't leak this handler into other tests.
        self._cleanup = lambda: bus.off("speak", handler)
        return handler

    @pytest.fixture(autouse=True)
    def _drop_handlers(self):
        self._cleanup = None
        yield
        if self._cleanup:
            self._cleanup()

    def test_voice_commands_toggle_dnd(self):
        h = make_hestia()
        assert "Do not disturb is on." in h.process_text("do not disturb")
        assert h.voice_state.dnd_active()
        assert "Do not disturb is on" in h.process_text("is do not disturb on")
        assert h.process_text("resume notifications") == "Notifications are back on."
        assert not h.voice_state.dnd_active()

    def test_timed_dnd_reply(self):
        h = make_hestia()
        out = h.process_text("do not disturb for 90 minutes")
        assert "1 hour 30 minutes" in out
        assert h.voice_state.dnd_remaining_minutes() == pytest.approx(90, abs=0.5)

    def test_unreadable_duration_is_reported(self):
        h = make_hestia()
        out = h.process_text("mute notifications for the afternoon")
        assert "didn't understand" in out and h.voice_state.dnd_active()

    def test_notifications_held_then_read_on_resume(self):
        h = make_hestia()
        speak = self._speak_handler(h)
        h.process_text("do not disturb")
        h.tts.speak.reset_mock()
        speak({"text": "Drink water"})
        speak({"text": "Stand up"})
        h.tts.speak.assert_not_called()
        assert h.voice_state.held_count() == 2
        out = h.process_text("resume notifications")
        assert "2 came in" in out and "Drink water" in out and "Stand up" in out
        assert h.voice_state.held_count() == 0

    def test_notification_speaks_normally_when_dnd_off(self):
        h = make_hestia()
        speak = self._speak_handler(h)
        speak({"text": "Take out the trash"})
        h.tts.speak.assert_called_once_with("Take out the trash")
        assert h._last_spoken == "Take out the trash"

    def test_expired_timed_dnd_prefixes_digest_to_next_notification(self):
        h = make_hestia()
        speak = self._speak_handler(h)
        h.voice_state.hold("Old reminder")
        speak({"text": "New reminder"})
        spoken = h.tts.speak.call_args[0][0]
        assert spoken.endswith("New reminder") and "Old reminder" in spoken

    def test_idle_voice_loop_reads_leftovers(self):
        h = make_hestia()
        h.voice_state.hold("Leftover")
        h._speak_held_notifications()
        assert "Leftover" in h.tts.speak.call_args[0][0]

    def test_sensitivity_command(self):
        h = make_hestia()
        h.wake_detector.set_sensitivity.return_value = "noisy"
        assert h.process_text("I'm in a noisy room") == "Wake word sensitivity set to noisy."
        h.wake_detector.set_sensitivity.assert_called_once_with("noisy")

    def test_sensitivity_command_without_wake_detector(self):
        h = make_hestia()
        h.wake_detector = None
        assert "isn't running" in h.process_text("I'm in a quiet room")


class TestPerModuleVoices:
    def _h(self, mapping, voices=None, has=True):
        h = make_hestia()
        h.config = {"tts": {"voices": voices or {"calm": {}, "brisk": {}},
                            "voice_by_module": mapping}}
        h.tts.has_voice.return_value = has
        return h

    def test_module_mapping_and_default(self):
        h = self._h({"apollo": "calm", "default": "brisk"})
        assert h._voice_profile_for_module("apollo") == "calm"
        assert h._voice_profile_for_module("pluto") == "brisk"
        assert h._voice_profile_for_module(None) == "brisk"

    def test_no_mapping_means_base_voice(self):
        h = make_hestia()
        h.config = {"tts": {}}
        assert h._voice_profile_for_module("apollo") is None

    def test_profile_that_does_not_exist_is_ignored(self):
        h = self._h({"apollo": "ghost"}, has=False)
        assert h._voice_profile_for_module("apollo") is None

    def test_intent_resolves_to_module_voice(self):
        h = self._h({"apollo": "calm"})
        assert h._voice_profile_for_intent("apollo_log_sleep") == "calm"

    def test_reply_is_spoken_in_the_modules_voice(self):
        h = self._h({"apollo": "calm"})
        h.nlu.understand.return_value = {"intent": "apollo_log_sleep", "entities": {}, "response": ""}
        h.process_text("log my sleep")
        h.tts.speak.assert_called_with("a plain response", voice="calm")

    def test_default_path_calls_speak_with_text_only(self):
        h = make_hestia()
        h.process_text("hello")
        h.tts.speak.assert_called_once_with("a plain response")

    def test_streamed_reply_uses_voice(self):
        h = self._h({"default": "calm"})
        h.orchestrator.try_stream_chat.return_value = iter(["Hi. ", "There."])
        h.process_voice_turn("hello")
        assert h.tts.speak_stream.call_args.kwargs == {"voice": "calm"}


class TestVoiceLoopStatesAndFallback:
    def test_states_progress_through_a_turn(self):
        h = make_hestia()
        seen = []
        orig = h.voice_state.set_state
        h.voice_state.set_state = lambda s, detail=None: (seen.append(s), orig(s, detail))
        h.wake_detector.listen_for_wake_word.return_value = True
        h.stt.listen_once.side_effect = ["what time is it", "exit"]
        h._shutdown = MagicMock()
        h.run_voice_loop()
        assert seen[0] == "waiting_for_wake"
        assert "listening" in seen and "thinking" in seen and "speaking" in seen
        assert seen[-1] == "inactive"

    def test_missing_stt_falls_back_to_typing(self, monkeypatch, capsys):
        h = make_hestia()
        h.stt = None
        h._voice_io_errors = {"stt": "model not found"}
        h._shutdown = MagicMock()
        inputs = iter(["hello", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        h.run_voice_loop()
        h.orchestrator.dispatch.assert_called_once()
        assert "model not found" in capsys.readouterr().err
        h._shutdown.assert_called_once()

    def test_missing_wake_word_falls_back_to_typing(self, monkeypatch):
        h = make_hestia()
        h.wake_detector = None
        h._shutdown = MagicMock()
        monkeypatch.setattr("builtins.input", lambda _p="": "exit")
        h.run_voice_loop()
        assert h.voice_state.state == "typed" or h.voice_state.state == "typed"
        h._shutdown.assert_called_once()

    def test_repeated_mic_failures_fall_back_to_typing(self, monkeypatch):
        h = make_hestia()
        h.wake_detector.listen_for_wake_word.side_effect = OSError("no input device")
        h._shutdown = MagicMock()
        monkeypatch.setattr(main.time, "sleep", lambda s: None)
        inputs = iter(["exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        h.run_voice_loop()
        assert h.wake_detector.listen_for_wake_word.call_count == main._MAX_VOICE_FAILURES
        h._shutdown.assert_called_once()

    def test_one_transient_failure_does_not_end_voice_mode(self, monkeypatch):
        h = make_hestia()
        h.wake_detector.listen_for_wake_word.side_effect = [OSError("glitch"), True]
        h.stt.listen_once.return_value = "exit"
        h._shutdown = MagicMock()
        monkeypatch.setattr(main.time, "sleep", lambda s: None)
        h.run_voice_loop()
        assert h.wake_detector.listen_for_wake_word.call_count == 2

    def test_shutdown_tolerates_missing_barge_in(self):
        h = make_hestia()
        h.barge_in = None
        h._barge_in_enabled = False
        h.process_voice_turn("hello")      # speaking path must not touch barge_in


class _Boom:
    def __init__(self, *a, **kw):
        raise RuntimeError("component exploded")


class TestBuildIoDegrades:
    def _builder(self, monkeypatch, tmp_path, **broken):
        monkeypatch.chdir(tmp_path)            # no stray data/mic_calibration.json
        for name, repl in {
            "HestiaSTT": MagicMock(), "HestiaTTS": MagicMock(),
            "WakeWordDetector": MagicMock(), "BargeInListener": MagicMock(),
        }.items():
            monkeypatch.setattr(main, name, broken.get(name, repl))
        b = object.__new__(main._ComponentBuilder) \
            if hasattr(main, "_ComponentBuilder") else None
        return b

    def _find_builder_class(self):
        for name, obj in vars(main).items():
            if isinstance(obj, type) and hasattr(obj, "build_io") and obj is not main.Hestia:
                return obj
        pytest.skip("builder class not found")

    def _run(self, monkeypatch, tmp_path, config=None, **broken):
        self._builder(monkeypatch, tmp_path, **broken)
        cls = self._find_builder_class()
        b = object.__new__(cls)
        b.config = config or {}
        return b, b.build_io()

    def test_everything_builds(self, monkeypatch, tmp_path):
        b, (stt, tts, wake, barge) = self._run(monkeypatch, tmp_path)
        assert stt and tts and wake and barge and b.io_errors == {}

    def test_stt_failure_is_isolated(self, monkeypatch, tmp_path):
        b, (stt, tts, wake, barge) = self._run(monkeypatch, tmp_path, HestiaSTT=_Boom)
        assert stt is None and wake is not None and barge is not None
        assert "component exploded" in b.io_errors["stt"]

    def test_tts_failure_gives_null_tts(self, monkeypatch, tmp_path):
        b, (stt, tts, wake, barge) = self._run(monkeypatch, tmp_path, HestiaTTS=_Boom)
        assert isinstance(tts, NullTTS) and "tts" in b.io_errors

    def test_wake_and_barge_failures_are_isolated(self, monkeypatch, tmp_path):
        b, (stt, tts, wake, barge) = self._run(
            monkeypatch, tmp_path, WakeWordDetector=_Boom, BargeInListener=_Boom)
        assert wake is None and barge is None and stt is not None
        assert set(b.io_errors) == {"wake_word", "barge_in"}

    def test_calibration_fills_unset_values_but_config_wins(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "load_calibration", lambda **kw: {
            "min_rms": 777.0, "vad_aggressiveness": 3, "wake_sensitivity": "noisy"})
        monkeypatch.setattr(main, "current_input_device_name", lambda: "Mic")
        stt_cls, wake_cls, barge_cls = MagicMock(), MagicMock(), MagicMock()
        self._builder(monkeypatch, tmp_path)
        for n, c in (("HestiaSTT", stt_cls), ("WakeWordDetector", wake_cls),
                     ("BargeInListener", barge_cls)):
            monkeypatch.setattr(main, n, c)
        cls = self._find_builder_class()
        b = object.__new__(cls)
        b.config = {"barge_in": {"min_rms": 500}}
        b.build_io()
        assert barge_cls.call_args.kwargs["min_rms"] == 500          # config wins
        assert barge_cls.call_args.kwargs["vad_aggressiveness"] == 3  # calibrated
        assert wake_cls.call_args.kwargs["sensitivity"] == "noisy"
        assert stt_cls.call_args.kwargs["vad_aggressiveness"] == 3

    def test_calibration_can_be_disabled(self, monkeypatch, tmp_path):
        called = []
        monkeypatch.setattr(main, "load_calibration", lambda **kw: called.append(1))
        b, _ = self._run(monkeypatch, tmp_path,
                         config={"barge_in": {"use_calibration": False}})
        assert not called

    def test_echo_cancel_with_pyttsx3_is_left_off(self, monkeypatch, tmp_path):
        tts_cls = MagicMock()
        tts_cls.return_value.engine = "pyttsx3"
        barge_cls = MagicMock()
        self._builder(monkeypatch, tmp_path)
        monkeypatch.setattr(main, "HestiaTTS", tts_cls)
        monkeypatch.setattr(main, "BargeInListener", barge_cls)
        cls = self._find_builder_class()
        b = object.__new__(cls)
        b.config = {"tts": {"engine": "pyttsx3"},
                    "barge_in": {"echo_cancel": {"enabled": True}}}
        b.build_io()
        assert barge_cls.call_args.kwargs["echo_canceller"] is None
        assert tts_cls.call_args.kwargs["echo_reference"] is None

    def test_echo_cancel_with_piper_is_wired(self, monkeypatch, tmp_path):
        tts_cls = MagicMock()
        tts_cls.return_value.engine = "piper"
        barge_cls = MagicMock()
        self._builder(monkeypatch, tmp_path)
        monkeypatch.setattr(main, "HestiaTTS", tts_cls)
        monkeypatch.setattr(main, "BargeInListener", barge_cls)
        cls = self._find_builder_class()
        b = object.__new__(cls)
        b.config = {"tts": {"engine": "piper"},
                    "barge_in": {"echo_cancel": {"enabled": True, "delay_ms": 20}}}
        b.build_io()
        assert isinstance(barge_cls.call_args.kwargs["echo_canceller"], NLMSEchoCanceller)
        assert isinstance(tts_cls.call_args.kwargs["echo_reference"], EchoReference)


class TestCalibrateMicCli:
    def test_flag_parses_and_runs_without_booting_hestia(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(main, "run_calibration",
                            lambda tts=None: mc.recommend_settings([60.0] * 50, device="M"))
        monkeypatch.setattr(main, "Hestia", MagicMock(side_effect=AssertionError("booted")))
        assert main.main(["--calibrate-mic", "--config", str(tmp_path / "none.yaml")]) == 0
        assert (tmp_path / "data" / "mic_calibration.json").exists()

    def test_calibration_failure_returns_nonzero(self, monkeypatch, tmp_path):
        def boom(tts=None):
            raise RuntimeError("no mic")
        monkeypatch.setattr(main, "run_calibration", boom)
        assert main.main(["--calibrate-mic", "--config", str(tmp_path / "none.yaml")]) == 1


# ===========================================================================
# #178 web UI endpoints
# ===========================================================================

class TestWebVoiceEndpoints:
    def _client(self, vs):
        from web_ui import HestiaWebUI
        return HestiaWebUI(memory=MagicMock(spec=["get_stats"]), voice_state=vs).app.test_client()

    def test_state_without_voice_loop_is_inactive(self):
        j = self._client(None).get("/api/voice/state").get_json()
        assert j["active"] is False and j["mic_open"] is False and j["state"] == "inactive"

    def test_state_reflects_voice_loop(self):
        vs = VoiceState()
        vs.set_state(STATE_WAKE)
        j = self._client(vs).get("/api/voice/state").get_json()
        assert j["state"] == "waiting_for_wake" and j["mic_open"] is True and j["active"] is True

    def test_dnd_toggle_endpoint(self):
        vs = VoiceState()
        c = self._client(vs)
        j = c.post("/api/voice/dnd", json={"on": True, "minutes": 15}).get_json()
        assert j["dnd"] is True and 14 < j["dnd_remaining_minutes"] <= 15
        assert c.post("/api/voice/dnd", json={"on": False}).get_json()["dnd"] is False

    @pytest.mark.parametrize("body", [{}, {"on": "yes"}, {"on": True, "minutes": -5},
                                      {"on": True, "minutes": 99999}, {"on": True, "minutes": True}])
    def test_dnd_rejects_bad_input(self, body):
        assert self._client(VoiceState()).post("/api/voice/dnd", json=body).status_code == 400

    def test_dnd_without_voice_state_is_503(self):
        assert self._client(None).post("/api/voice/dnd", json={"on": True}).status_code == 503


# ===========================================================================
# Config validation
# ===========================================================================

class TestVoiceConfigValidation:
    def _validate(self, extra):
        from core import config_validation as cv
        return cv.validate_config(extra)

    def test_shipped_example_still_valid(self):
        import yaml
        from core import config_validation as cv
        with open("config/laptop_config.example.yaml", encoding="utf-8") as fh:
            rep = cv.validate_config(yaml.safe_load(fh))
        assert not rep.errors

    def test_bad_sensitivity_warns(self):
        rep = self._validate({"wake_word": {"sensitivity": "loud"}})
        assert any("sensitivity" in w for w in rep.warnings)

    def test_vad_out_of_range_errors(self):
        rep = self._validate({"stt": {"vad_aggressiveness": 7}})
        assert any("0-3" in e for e in rep.errors)

    def test_undefined_voice_profile_warns(self):
        rep = self._validate({"tts": {"voices": {"calm": {}},
                                      "voice_by_module": {"apollo": "ghost"}}})
        assert any("ghost" in w for w in rep.warnings)
