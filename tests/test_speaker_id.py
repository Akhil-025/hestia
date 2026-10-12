"""Backlog #177: speaker identification (synthetic voices only, never real people)."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import speaker_id as S  # noqa: E402

SR = 16000
OWNER_FORMANTS = [(700, 150), (1200, 200), (2600, 300)]
OTHER_FORMANTS = [(450, 120), (2000, 250), (3200, 300)]


def voice(f0, formants, seed, dur=2.0, jitter=0.03):
    """A crude vowel-like signal: a harmonic series shaped by formants, with a
    syllable envelope. Different f0/formants stand in for different people."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * dur)) / SR
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t + rng.uniform(0, 6))) ** 2
    f = f0 * (1 + jitter * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    sig = np.zeros_like(t)
    for h in range(1, 40):
        fh = h * f0
        if fh > 7000:
            break
        gain = sum(np.exp(-((fh - fm) / bw) ** 2) for fm, bw in formants)
        sig += gain / h ** 0.5 * np.sin(h * ph)
    sig = sig * env + 0.002 * rng.standard_normal(len(t))
    return (sig / np.abs(sig).max() * 0.5).astype(np.float32)


def owner_clip(i):
    return voice(120 + [0, 3, -2, 4, -3, 2, -1][i % 7], OWNER_FORMANTS, i)


@pytest.fixture
def gate(tmp_path):
    store = S.ProfileStore(str(tmp_path / "vp.json"))
    prof = S.build_profile("me", [S.voiceprint(owner_clip(i)) for i in range(5)])
    store.add(prof)
    return S.SpeakerID(store, module_for_intent=lambda i: {"chat": "core", "pluto_log_expense": "pluto",
                                                           "recommend_movie": "dionysus",
                                                           "metis_polish": "metis"}.get(i))


# ── features ─────────────────────────────────────────────────────────────────

def test_silence_noise_and_garbage_give_no_voiceprint():
    assert S.voiceprint(np.zeros(SR * 2)) is None
    assert S.voiceprint(np.random.default_rng(0).standard_normal(SR * 2) * 0.01) is None
    assert S.voiceprint(np.array([])) is None
    assert S.voiceprint(np.full(SR, np.nan)) is None
    assert S.voiceprint(owner_clip(0)[: SR // 10]) is None          # far too short
    assert S.voiceprint(owner_clip(0), sr=8000) is None             # wrong sample rate


def test_voiceprint_is_stable_for_one_voice_and_differs_between_voices():
    a1, a2 = S.voiceprint(owner_clip(10)), S.voiceprint(owner_clip(11))
    b = S.voiceprint(voice(210, OTHER_FORMANTS, 99))
    assert a1.shape == a2.shape == b.shape == (38,)
    assert np.linalg.norm(a1 - a2) < 0.2 * np.linalg.norm(a1 - b)


# ── identification ───────────────────────────────────────────────────────────

def test_owner_is_recognised_others_are_guests(gate):
    ident = gate.identify(owner_clip(20))
    assert ident.is_owner and ident.name == "me" and ident.reason == "matched"
    for f0, fm, seed in [(210, OTHER_FORMANTS, 1), (125, OTHER_FORMANTS, 2), (180, OWNER_FORMANTS, 3)]:
        g = gate.identify(voice(f0, fm, seed))
        assert not g.is_owner and g.name is None and g.reason == "voice not recognised"


def test_every_doubt_is_a_guest(gate, tmp_path):
    assert gate.identify(np.zeros(SR * 2)).reason == "too little speech to tell"
    assert gate.identify(None).role == S.GUEST
    empty = S.SpeakerID(S.ProfileStore(str(tmp_path / "none.json")))
    assert empty.identify(owner_clip(0)).reason == "no voice enrolled"
    (tmp_path / "bad.json").write_text("{broken")
    assert S.SpeakerID(S.ProfileStore(str(tmp_path / "bad.json"))).identify(owner_clip(0)).role == S.GUEST


def test_stricter_margin_refuses_more(gate):
    clip = owner_clip(30)
    gate.margin = 0.01
    gate.min_floor = 0.0
    assert not gate.identify(clip).is_owner


def test_non_owner_profile_never_gets_owner_rights(tmp_path):
    store = S.ProfileStore(str(tmp_path / "vp.json"))
    store.add(S.build_profile("guest-friend", [S.voiceprint(owner_clip(i)) for i in range(4)], role="friend"))
    ident = S.SpeakerID(store).identify(owner_clip(9))
    assert ident.name == "guest-friend" and not ident.is_owner


# ── policy ───────────────────────────────────────────────────────────────────

def test_guest_policy_defaults_fail_closed(gate):
    guest = S.Identity(None, S.GUEST, "x")
    owner = S.Identity("me", S.OWNER, "matched")
    assert gate.refusal_for(owner, "pluto_log_expense") is None
    assert gate.refusal_for(guest, "pluto_log_expense")                 # finance: refused
    assert gate.refusal_for(guest, "chat")                              # free chat reads memory: refused
    assert gate.refusal_for(guest, "totally_unknown_intent")            # unregistered: refused
    assert gate.refusal_for(guest, "recommend_movie") is None           # entertainment: allowed
    assert gate.refusal_for(guest, "metis_polish") is None
    assert gate.may_repeat(owner) and not gate.may_repeat(guest)


def test_guest_policy_from_config_and_registry_failure_refuses():
    pol = S.GuestPolicy.from_config({"allow_modules": ["chronos"], "deny_intents": ["set_timer"],
                                     "allow_chat": True, "allow_repeat": True})
    assert pol.allows("chat", "core") and pol.allows("x", "chronos") and not pol.allows("set_timer", "chronos")
    assert not pol.allows("x", "dionysus")

    def boom(_):
        raise RuntimeError("registry down")
    g = S.SpeakerID(S.ProfileStore("unused"), module_for_intent=boom)
    assert g.refusal_for(S.Identity(None, S.GUEST, "x"), "recommend_movie")


def test_from_config_off_by_default_and_reads_guest_section(tmp_path):
    assert S.SpeakerID.from_config(None) is None
    assert S.SpeakerID.from_config({"enabled": False}) is None
    g = S.SpeakerID.from_config({"enabled": True, "profiles_path": str(tmp_path / "p.json"),
                                 "margin": 1.5, "guest": {"allow_modules": ["chronos"]}})
    assert g.margin == 1.5 and g.guest_policy.allow_modules == frozenset({"chronos"})


# ── storage and enrolment ────────────────────────────────────────────────────

def test_store_keeps_numbers_not_audio_and_can_forget(tmp_path):
    store = S.ProfileStore(str(tmp_path / "d" / "vp.json"))
    store.add(S.build_profile("Me", [S.voiceprint(owner_clip(i)) for i in range(4)]))
    data = json.loads((tmp_path / "d" / "vp.json").read_text())
    p = data["profiles"]["Me"]
    assert set(p) == {"role", "centroid", "spread", "samples", "created"} and len(p["centroid"]) == 38
    assert len(store.load()) == 1
    store.add(S.build_profile("me", [S.voiceprint(owner_clip(i)) for i in range(4)]))   # same name: replaced
    assert len(store.load()) == 1
    assert store.remove("ME") and not store.remove("ME") and store.load() == []


def test_build_profile_needs_three_samples():
    with pytest.raises(ValueError):
        S.build_profile("x", [S.voiceprint(owner_clip(0))] * 2)


def test_enrolment_flow_with_fake_microphone(tmp_path):
    store = S.ProfileStore(str(tmp_path / "vp.json"))
    clips = iter([owner_clip(i) for i in range(5)])
    said = []
    prof = S.enroll_interactive(store, "me", record=lambda: next(clips), ask=lambda _p: "", say=said.append)
    assert prof and prof.samples == 5 and store.load()[0].name == "me"
    assert any("no audio kept" in s for s in said)


def test_enrolment_retries_silence_then_gives_up_without_saving(tmp_path):
    store = S.ProfileStore(str(tmp_path / "vp.json"))
    prof = S.enroll_interactive(store, "me", record=lambda: np.zeros(SR * 2), ask=lambda _p: "", say=lambda _s: None)
    assert prof is None and store.load() == []


# ── wired into the voice turn ────────────────────────────────────────────────

def hestia(gate, identity, intent="chat"):
    import main
    h = object.__new__(main.Hestia)
    h.mnemosyne = MagicMock()
    h.mnemosyne.get_recent.return_value = ["private earlier turn"]
    h.nlu = MagicMock()
    h.nlu.understand.return_value = {"intent": intent, "entities": {}, "response": ""}
    h.orchestrator = MagicMock()
    h.orchestrator.try_stream_chat.return_value = None
    h.orchestrator.dispatch.return_value = "the answer"
    h.tts, h.wake_detector, h.stt, h.barge_in = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    h._barge_in_enabled = False
    h.speaker_id = gate
    h._speaker_identity = identity
    h._last_spoken = "my salary is secret"
    return h


@pytest.fixture(autouse=True)
def _bus(monkeypatch):
    import main
    monkeypatch.setattr(main, "bus", MagicMock())


def test_voice_turn_refuses_a_guest_before_dispatch_and_gives_no_memory(gate):
    h = hestia(gate, S.Identity(None, S.GUEST, "voice not recognised"), "pluto_log_expense")
    out = h.process_voice_turn("log an expense of five hundred")
    assert "don't recognise your voice" in out
    h.orchestrator.dispatch.assert_not_called()
    h.orchestrator.try_stream_chat.assert_not_called()
    h.mnemosyne.get_recent.assert_not_called()                 # guest NLU gets no memory context
    assert h.nlu.understand.call_args.args[1] == []
    h.tts.speak.assert_called()                                 # the refusal is spoken


def test_voice_turn_lets_a_guest_use_allowed_modules(gate):
    h = hestia(gate, S.Identity(None, S.GUEST, "x"), "recommend_movie")
    assert h.process_voice_turn("suggest a movie") == "the answer"
    h.orchestrator.dispatch.assert_called_once()


def test_voice_turn_owner_is_unaffected(gate):
    h = hestia(gate, S.Identity("me", S.OWNER, "matched"), "pluto_log_expense")
    assert h.process_voice_turn("log an expense of five hundred") == "the answer"
    h.mnemosyne.get_recent.assert_called_once()


def test_feature_off_means_no_behaviour_change():
    h = hestia(None, None, "pluto_log_expense")
    assert h.process_voice_turn("log an expense") == "the answer"


def test_guest_cannot_make_hestia_repeat_a_private_answer(gate):
    h = hestia(gate, S.Identity(None, S.GUEST, "x"))
    out = h._try_local_command("repeat that", voice_turn=True)
    assert out == "I only repeat things for the owner." and "salary" not in out
    owner = hestia(gate, S.Identity("me", S.OWNER, "matched"))
    assert owner._try_local_command("repeat that", voice_turn=True) == "my salary is secret"
    # a typed or web request is never gated by whoever last spoke aloud
    assert h._try_local_command("repeat that", voice_turn=False) == "my salary is secret"


def test_identify_speaker_uses_last_audio_and_fails_closed(gate):
    h = hestia(gate, None)
    h.stt.samplerate = SR
    h.stt.take_last_audio.return_value = owner_clip(40)
    h._identify_speaker()
    assert h._speaker_identity.is_owner
    h.stt.take_last_audio.return_value = None
    h._identify_speaker()
    assert not h._speaker_identity.is_owner and h._speaker_identity.reason == "no audio captured"
    h.stt.take_last_audio.side_effect = RuntimeError("no mic")
    h._identify_speaker()
    assert not h._speaker_identity.is_owner
    h.speaker_id = None
    h._identify_speaker()
    assert h._speaker_identity is None


def test_stt_keeps_audio_only_until_collected():
    from core.stt import HestiaSTT
    stt = object.__new__(HestiaSTT)
    stt.samplerate = SR
    stt.noise_filter = MagicMock()
    stt.noise_filter.filter.side_effect = lambda a, sr: a
    stt._transcribe = lambda a: "hello"
    clip = owner_clip(0)
    assert stt.transcribe_audio(clip) == "hello"
    assert stt.take_last_audio() is clip
    assert stt.take_last_audio() is None


def test_config_validation_and_example_config_accept_speaker_id():
    import yaml
    from core.config_validation import validate_config
    cfg = yaml.safe_load((ROOT / "config" / "laptop_config.example.yaml").read_text(encoding="utf-8"))
    assert cfg["speaker_id"]["enabled"] is False and cfg["security"]["encrypt_at_rest"] is False
    rep = validate_config({"speaker_id": {"enabled": True, "guest": {"allow_chat": False}}})
    assert not [m for m in rep.errors + rep.warnings if "speaker_id" in m]
    assert any("speaker_id.enabled" in e for e in validate_config({"speaker_id": {"enabled": "yes"}}).errors)
