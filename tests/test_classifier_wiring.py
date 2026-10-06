# tests/test_classifier_wiring.py
"""
Backlog #4: the trained intent classifier replacing the fallback tiers.

Covers the training-data sources (core/classifier_data.py), the service's
retraining, Hecate's two integration points (primary / assist), the NLU's
fallback when the LLM can't answer, and the weekly heartbeat retrain.

Fakes stand in for the classifier wherever the test is about *routing*; one
test trains a real model on a handful of examples to prove the data path works
end to end.
"""
import json
import os
import sys
from datetime import date, datetime, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import classifier_data as cd
from core.intent_classifier import ClassifierService, Example, Prediction, normalise
from modules.hecate.engine import HecateEngine
from modules.hecate.intent_registry import ALL_INTENTS, INTENT_MODULE_MAP


# ---------------------------------------------------------------- helpers

class FakeClassifier:
    """Quacks like ClassifierService."""

    def __init__(self, intent=None, prob=0.8, mode="primary", enabled=True):
        self.intent, self.prob, self.mode, self.enabled = intent, prob, mode, enabled
        self.calls = []

    def classify(self, text):
        self.calls.append(text)
        if self.intent is None:
            return None
        return Prediction(self.intent, self.prob, 0.5)


def _intent_in(module):
    return next(i for i, m in sorted(INTENT_MODULE_MAP.items()) if m == module and i != "chat")


def _decide(hecate, query, intent="chat", conf=0.5, active=None):
    active = active or sorted(set(INTENT_MODULE_MAP.values()))
    return hecate.decide(query, {"intent": intent, "confidence": conf}, active)


# ---------------------------------------------------------------- data sources

def test_prompt_examples_come_from_the_prompt_file():
    ex = cd.prompt_examples()
    assert len(ex) > 100
    assert all(e.intent in ALL_INTENTS and e.source == "prompt" for e in ex)


def test_alias_examples_cover_every_registered_alias():
    ex = cd.alias_examples()
    assert len(ex) > 100
    assert any(e.text == "log my sleep" and e.intent == "apollo_track_sleep" for e in ex)


def test_prompt_examples_missing_file_is_empty(tmp_path):
    assert cd.prompt_examples(tmp_path / "nope.txt") == []


def test_add_label_rejects_unknown_intent(tmp_path):
    p = tmp_path / "labels.jsonl"
    assert cd.add_label("hello there", "not_an_intent", p) is False
    assert not p.exists()


def test_add_label_round_trips_with_weight(tmp_path):
    p = tmp_path / "d" / "labels.jsonl"
    assert cd.add_label("log my sleep tonight", "apollo_track_sleep", p)
    ex = cd.label_examples(p)
    assert len(ex) == 1 and ex[0].weight == cd.LABEL_WEIGHT and ex[0].source == "label"


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_log_examples_keep_only_confident_llm_unflagged_records(tmp_path):
    routing, feedback = tmp_path / "r.jsonl", tmp_path / "f.jsonl"
    _write_jsonl(routing, [
        {"query": "log my sleep", "intent": "apollo_track_sleep", "confidence": 0.95, "source": "nlu"},
        {"query": "low conf one", "intent": "apollo_track_sleep", "confidence": 0.4, "source": "nlu"},
        {"query": "from the cache", "intent": "apollo_track_sleep", "confidence": 0.99, "source": "cache"},
        {"query": "from the classifier", "intent": "apollo_track_sleep", "confidence": 0.99, "source": "classifier"},
        {"query": "unknown intent", "intent": "made_up", "confidence": 0.99, "source": "nlu"},
        {"query": "just chat", "intent": "chat", "confidence": 0.99, "source": "nlu"},
        {"query": "flagged wrong", "intent": "apollo_track_sleep", "confidence": 0.99, "source": "nlu"},
    ])
    _write_jsonl(feedback, [{"query": "Flagged  wrong", "intent": "apollo_track_sleep"}])
    got = cd.log_examples(routing, feedback)
    assert [e.text for e in got] == ["log my sleep"]


def test_golden_prompts_are_held_out_of_training(tmp_path):
    golden = tmp_path / "g.md"
    golden.write_text('## 1. x\n- "Secret golden prompt"\n', encoding="utf-8")
    labels = tmp_path / "l.jsonl"
    cd.add_label("secret golden prompt", "apollo_track_sleep", labels)
    cd.add_label("a different phrase", "apollo_track_sleep", labels)
    ex = cd.collect_examples(
        prompt_path=tmp_path / "none", alias_path=tmp_path / "none", labels_path=labels,
        routing_log=tmp_path / "none", feedback_log=tmp_path / "none", golden_path=golden)
    assert [e.text for e in ex] == ["a different phrase"]


def test_label_wins_over_a_weaker_source_for_the_same_text(tmp_path):
    labels, aliases = tmp_path / "l.jsonl", tmp_path / "a.yaml"
    cd.add_label("log my sleep", "apollo_log_mood", labels)
    aliases.write_text('apollo_track_sleep:\n  - "log my sleep"\n', encoding="utf-8")
    ex = cd.collect_examples(
        prompt_path=tmp_path / "none", alias_path=aliases, labels_path=labels,
        routing_log=tmp_path / "none", feedback_log=tmp_path / "none",
        golden_path=tmp_path / "none", hold_out_golden=False)
    assert len(ex) == 1 and ex[0].intent == "apollo_track_sleep" or ex[0].source == "label"
    assert ex[0].source == "label"


def test_real_collection_is_substantial_and_valid():
    ex = cd.collect_examples()
    assert len(ex) > 500
    assert all(e.intent in ALL_INTENTS for e in ex)


# ---------------------------------------------------------------- service

def _tiny_examples():
    return [
        Example("log my sleep", "apollo_track_sleep"), Example("i slept for seven hours", "apollo_track_sleep"),
        Example("track my sleep", "apollo_track_sleep"), Example("log my workout", "apollo_log_workout"),
        Example("i worked out today", "apollo_log_workout"), Example("record my workout", "apollo_log_workout"),
    ]


def test_service_trains_saves_and_classifies(tmp_path):
    svc = ClassifierService(model_path=tmp_path / "m.npz", examples_fn=_tiny_examples,
                            valid_intents=ALL_INTENTS)
    res = svc.train()
    assert res["ok"] and svc.ready and (tmp_path / "m.npz").exists()
    pred = svc.classify("log my sleep please")
    assert pred is not None and pred.intent == "apollo_track_sleep"
    # a fresh service loads the saved file instead of retraining
    again = ClassifierService(model_path=tmp_path / "m.npz")
    assert again.load() and again.classify("record my workout").intent == "apollo_log_workout"


def test_service_declines_gibberish(tmp_path):
    svc = ClassifierService(model_path=tmp_path / "m.npz", examples_fn=_tiny_examples)
    svc.train()
    assert svc.classify("zzzzqqqq") is None


def test_retrain_if_needed_skips_until_enough_new_examples(tmp_path):
    data = _tiny_examples()
    svc = ClassifierService(model_path=tmp_path / "m.npz", examples_fn=lambda: list(data))
    svc.train()
    assert "skipped" in svc.retrain_if_needed(min_new=3)
    data.extend(Example(f"extra sleep phrase {i}", "apollo_track_sleep") for i in range(4))
    res = svc.retrain_if_needed(min_new=3)
    assert res.get("ok") and "skipped" not in res
    assert svc.status()["examples"] == 10


def test_retrain_when_not_ready_trains(tmp_path):
    svc = ClassifierService(model_path=tmp_path / "m.npz", examples_fn=_tiny_examples)
    assert svc.retrain_if_needed()["ok"] and svc.ready


def test_retrain_off_is_a_noop(tmp_path):
    svc = ClassifierService(model_path=tmp_path / "m.npz", mode="off", examples_fn=_tiny_examples)
    assert "skipped" in svc.retrain_if_needed()
    assert not svc.ready


def test_failed_retrain_keeps_previous_model(tmp_path):
    state = {"ok": True}

    def src():
        if not state["ok"]:
            raise RuntimeError("log unreadable")
        return _tiny_examples()
    svc = ClassifierService(model_path=tmp_path / "m.npz", examples_fn=src)
    svc.train()
    state["ok"] = False
    assert svc.train()["ok"] is False
    assert svc.ready and svc.classify("log my sleep") is not None


# ---------------------------------------------------------------- golden scoring

class _Case:
    def __init__(self, prompt, kind, expected, forbidden=None):
        self.prompt, self.kind, self.expected, self.forbidden = prompt, kind, expected, forbidden


class _ByText:
    def __init__(self, table): self.table = table
    def classify(self, text):
        i = self.table.get(text)
        return Prediction(i, 0.9, 0.5) if i else None


def test_evaluate_on_golden_counts_coverage_and_precision():
    sleep = "apollo_track_sleep"
    svc = _ByText({"a": sleep, "b": "apollo_log_workout", "c": "set_reminder"})
    cases = [
        _Case("a", "module", "apollo"),                 # right (module level)
        _Case("b", "module", "mnemosyne"),              # wrong
        _Case("c", "intent", "set_reminder"),           # right (intent level)
        _Case("d", "intent", "set_reminder"),           # declined
    ]
    ev = cd.evaluate_on_golden(svc, cases)
    assert (ev["cases"], ev["answered"], ev["right"]) == (4, 3, 2)
    assert ev["coverage"] == 0.75 and ev["precision"] == round(2 / 3, 3)
    assert ev["wrong"][0]["prompt"] == "b"


def test_evaluate_on_golden_forbidden_intent_is_wrong():
    svc = _ByText({"a": "set_reminder"})
    ev = cd.evaluate_on_golden(svc, [_Case("a", "intent", "set_reminder", forbidden="set_reminder")])
    assert ev["right"] == 0


def test_real_model_beats_chance_on_held_out_golden_prompts(tmp_path):
    svc = ClassifierService(model_path=tmp_path / "m.npz",
                            examples_fn=cd.make_examples_fn(), valid_intents=ALL_INTENTS)
    assert svc.train()["ok"]
    ev = cd.evaluate_on_golden(svc)
    assert ev["cases"] > 50
    # Measured at ~58% answered / ~95% right when it answers; assert generous floors.
    assert ev["coverage"] > 0.3
    assert ev["precision"] > 0.8


# ---------------------------------------------------------------- Hecate integration

def test_primary_mode_overrides_a_chat_guess_before_text_triggers():
    target = "apollo_track_sleep"
    h = HecateEngine(FakeClassifier(target, 0.8, "primary"))
    d = _decide(h, "put me down for eight hours last night", intent="chat", conf=0.5)
    assert d["primary"] == "apollo"
    assert d["intent"] == "track_sleep"
    assert any("classifier (primary)" in c for c in d["checked"])


def test_primary_mode_does_not_override_a_confident_registered_intent():
    clf = FakeClassifier("apollo_track_sleep", 0.9, "primary")
    h = HecateEngine(clf)
    d = _decide(h, "set a reminder", intent="set_reminder", conf=0.95)
    assert d["primary"] == INTENT_MODULE_MAP["set_reminder"]
    assert clf.calls == []


def test_primary_mode_consulted_when_nlu_registered_but_unsure():
    # 0.5 clears the 0.45 clarify floor but is under the classifier's "unsure" bar.
    clf = FakeClassifier("apollo_track_sleep", 0.8, "primary")
    h = HecateEngine(clf)
    d = _decide(h, "sleep thing", intent="set_reminder", conf=0.5)
    assert d["primary"] == "apollo"


def test_primary_mode_declining_falls_through_to_text_triggers():
    h = HecateEngine(FakeClassifier(None, mode="primary"))
    d = _decide(h, "search my documents for turbulence", intent="chat", conf=0.5)
    assert d["primary"] == "athena"            # the Tier-2 trigger still works


def test_assist_mode_waits_until_every_tier_has_declined():
    clf = FakeClassifier("apollo_track_sleep", 0.8, "assist")
    h = HecateEngine(clf)
    # text trigger wins, so the classifier is never asked
    d1 = _decide(h, "search my documents for turbulence", intent="chat")
    assert d1["primary"] == "athena" and clf.calls == []
    # nothing else matches -> the classifier is the last resort
    d2 = _decide(h, "put me down for eight hours last night", intent="chat")
    assert d2["primary"] == "apollo" and d2["intent"] == "track_sleep"
    assert any("classifier (assist)" in c for c in d2["checked"])


def test_off_or_missing_classifier_changes_nothing():
    baseline = _decide(HecateEngine(), "tell me a joke", intent="chat")
    off = _decide(HecateEngine(FakeClassifier("apollo_track_sleep", mode="primary", enabled=False)),
                  "tell me a joke", intent="chat")
    assert off["primary"] == baseline["primary"] == "core"


def test_classifier_suggesting_an_inactive_module_is_ignored():
    h = HecateEngine(FakeClassifier("apollo_track_sleep", mode="primary"))
    d = _decide(h, "put me down for eight hours", intent="chat", active=["core", "chronos"])
    assert d["primary"] == "core"
    assert any("can't be dispatched" in c for c in d["checked"])


def test_classifier_exception_never_breaks_routing():
    class Boom(FakeClassifier):
        def classify(self, text): raise RuntimeError("model exploded")
    d = _decide(HecateEngine(Boom("x", mode="primary")), "hello", intent="chat")
    assert d["primary"] == "core"


def test_attach_classifier_can_be_set_and_cleared():
    h = HecateEngine()
    h.attach_classifier(FakeClassifier("apollo_track_sleep", mode="primary"))
    assert _decide(h, "put me down for eight hours", intent="chat")["primary"] == "apollo"
    h.attach_classifier(None)
    assert _decide(h, "put me down for eight hours", intent="chat")["primary"] == "core"


# ---------------------------------------------------------------- NLU fallback

def _bare_nlu():
    from core.nlu import HestiaNLU
    nlu = HestiaNLU.__new__(HestiaNLU)
    nlu._classifier = None
    return nlu


def test_nlu_classifier_fallback_result_shape():
    nlu = _bare_nlu()
    nlu.attach_classifier(FakeClassifier("apollo_track_sleep", 0.7))
    out = nlu._classifier_fallback("log my sleep")
    assert out["intent"] == "apollo_track_sleep" and out["entities"] == {}
    assert out["source"] == "classifier" and 0.5 <= out["confidence"] <= 0.9


def test_nlu_classifier_fallback_none_when_off_or_declining():
    nlu = _bare_nlu()
    assert nlu._classifier_fallback("x") is None
    nlu.attach_classifier(FakeClassifier(None))
    assert nlu._classifier_fallback("x") is None
    nlu.attach_classifier(FakeClassifier("apollo_track_sleep", enabled=False))
    assert nlu._classifier_fallback("x") is None


def test_nlu_uses_classifier_when_ollama_is_unreachable():
    from core.nlu import HestiaNLU
    nlu = HestiaNLU(prompt_path="config/nlu_prompt.txt", alias_path=None, cache_ttl_seconds=0)
    nlu._health_check = lambda: False
    nlu.attach_classifier(FakeClassifier("apollo_track_sleep", 0.7))
    out = nlu.understand("put me down for eight hours")
    assert out["intent"] == "apollo_track_sleep" and out["source"] == "classifier"


def test_nlu_without_classifier_still_gives_the_old_unreachable_reply():
    from core.nlu import HestiaNLU
    nlu = HestiaNLU(prompt_path="config/nlu_prompt.txt", alias_path=None, cache_ttl_seconds=0)
    nlu._health_check = lambda: False
    out = nlu.understand("put me down for eight hours")
    assert out["intent"] == "chat" and out["confidence"] == 0.0


# ---------------------------------------------------------------- heartbeat retrain

class _Stamp(datetime):
    hour_now = 3

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 10, 5, cls.hour_now, 0, 0)


def _heartbeat(clf):
    from core.heartbeat import HestiaHeartbeat
    return HestiaHeartbeat(classifier=clf)


class _RecordingService:
    enabled = True
    def __init__(self): self.calls = 0
    def retrain_if_needed(self):
        self.calls += 1
        return {"ok": True}


def test_heartbeat_retrains_off_peak_once_a_week(monkeypatch):
    import core.heartbeat as hb
    monkeypatch.setattr(hb, "datetime", _Stamp)
    svc = _RecordingService()
    beat = _heartbeat(svc)
    beat._maybe_retrain_classifier()
    beat._maybe_retrain_classifier()                 # same week: skipped
    assert svc.calls == 1
    beat._last_classifier_retrain_date = date.today() - timedelta(days=8)
    beat._maybe_retrain_classifier()
    assert svc.calls == 2


def test_heartbeat_does_not_retrain_in_the_daytime(monkeypatch):
    import core.heartbeat as hb
    _Stamp.hour_now = 14
    monkeypatch.setattr(hb, "datetime", _Stamp)
    svc = _RecordingService()
    _heartbeat(svc)._maybe_retrain_classifier()
    _Stamp.hour_now = 3
    assert svc.calls == 0


def test_heartbeat_without_classifier_is_unaffected():
    _heartbeat(None)._maybe_retrain_classifier()


def test_heartbeat_swallows_retrain_errors(monkeypatch):
    import core.heartbeat as hb
    monkeypatch.setattr(hb, "datetime", _Stamp)

    class Bad(_RecordingService):
        def retrain_if_needed(self): raise RuntimeError("boom")
    _heartbeat(Bad())._maybe_retrain_classifier()


# ---------------------------------------------------------------- config

def test_config_validation_accepts_the_new_blocks():
    from core.config_validation import validate_config
    rep = validate_config({
        "ollama": {"model": "m"}, "database": {"path": "x.db"},
        "classifier": {"mode": "assist", "min_probability": 0.3},
        "shadow": {"enabled": False, "rules": []},
        "processes": {"roles": ["core", "jobs"]},
    })
    assert rep.ok and not any("classifier" in w or "shadow" in w or "processes" in w for w in rep.warnings)


def test_config_validation_flags_a_wrong_type():
    from core.config_validation import validate_config
    rep = validate_config({"ollama": {"model": "m"}, "database": {"path": "x.db"},
                           "classifier": {"mode": 3}})
    assert not rep.ok
