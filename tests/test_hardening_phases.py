"""Tests for Hardening Plan phases 1, 3, 4, 5 (no Ollama required)."""
import json
import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.nlu import HestiaNLU
from core.nlu_retrieval import ExampleRetriever
from core.entity_specs import apply_spec, requires_entities
from core.route_guards import apply_guards
from modules.hecate.intent_registry import (
    DESTRUCTIVE, READ, WRITE, EXTERNAL, ORCHESTRATOR_CONFIRMED, risk_for, ALL_INTENTS)

_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ------------------------------------------------------------- phase 1
def _nlu(**kw):
    kw.setdefault("prompt_path", os.path.join(_ROOT, "config", "nlu_prompt.txt"))
    kw.setdefault("alias_path", None)
    kw.setdefault("cache_ttl_seconds", 0)
    kw.setdefault("examples_path", os.path.join(_ROOT, "config", "nlu_examples.yaml"))
    return HestiaNLU(**kw)


def test_examples_file_is_in_sync_with_prompt():
    import yaml
    data = yaml.safe_load(open(os.path.join(_ROOT, "config", "nlu_examples.yaml"), encoding="utf-8"))
    assert len(data["examples"]) >= 380
    assert all(e["intent"] in ALL_INTENTS for e in data["examples"])


def test_retrieval_prompt_is_small_and_static_first():
    n = _nlu(prompt_mode="retrieval", include_response=False)
    legacy = _nlu()
    p = n._build_prompt("write me a short poem about the ocean", None)
    assert len(p) < 12000 < len(legacy._build_prompt("x", None))
    assert p.startswith("You are Hestia's intent classifier")
    assert p.rstrip().endswith("JSON:") and "User: write me a short poem" in p
    assert "response" not in p.split("Valid intents")[0]


def test_retrieval_finds_relevant_example():
    r = ExampleRetriever(os.path.join(_ROOT, "config", "nlu_examples.yaml"))
    assert r.top("write me a poem about autumn", 4)[0]["intent"].startswith("orpheus_")


def test_schema_without_response_and_narrowed_enum():
    s = HestiaNLU._build_schema(frozenset({"chat", "get_notes"}), include_response=False)
    assert "response" not in s["properties"] and "response" not in s["required"]
    assert s["properties"]["intent"]["enum"] == ["chat", "get_notes"]
    n = _nlu(prompt_mode="retrieval", candidate_k=8)
    cands = n._candidate_enum("write me a short poem about the ocean")
    assert "orpheus_write_poem" in cands and "chat" in cands and len(cands) < 40


def test_missing_response_is_empty_not_raw_json():
    n = _nlu()
    parsed, ok = n._parse_response('{"intent": "chat", "entities": {}, "confidence": 0.9}')
    assert ok and parsed["response"] == ""


def test_single_attempt_budget_no_sleep(monkeypatch):
    n = _nlu(max_attempts=1, time_budget_s=2.0)
    calls = {"n": 0}
    monkeypatch.setattr(n, "_health_check", lambda: True)
    monkeypatch.setattr(n, "_call_llm", lambda p: (calls.__setitem__("n", calls["n"] + 1), None)[1])
    import time as _t
    slept = []
    monkeypatch.setattr(_t, "sleep", lambda s: slept.append(s))
    out = n.understand("hello there friend")
    assert calls["n"] == 1 and slept == [] and out["intent"] == "chat"


def test_llm_path_applies_guards_and_specs(monkeypatch):
    n = _nlu(prompt_mode="retrieval", include_response=False)
    monkeypatch.setattr(n, "_health_check", lambda: True)
    monkeypatch.setattr(n, "_call_llm", lambda p: json.dumps(
        {"intent": "dionysus_recommend_movie", "entities": {"topic": ["ocean"]}, "confidence": 0.8}))
    out = n.understand("Write me a short poem about the ocean")
    assert out["intent"] == "orpheus_write_poem"
    monkeypatch.setattr(n, "_call_llm", lambda p: json.dumps(
        {"intent": "take_note", "entities": {"content": "What notes do I have?"}, "confidence": 0.9}))
    assert n.understand("Which notes do I have saved")["intent"] == "get_notes"


def test_classifier_stage1_threshold():
    class P:  # minimal prediction/classifier stand-ins
        def __init__(s, i, p): s.intent, s.probability = i, p
    class C:
        enabled = True
        def __init__(s, i, p): s.pred = P(i, p)
        def classify(s, t): return s.pred
    n = _nlu(classifier_first_threshold=0.9)
    n.attach_classifier(C("get_time", 0.95))
    assert n._classifier_stage1("whats the time")["source"] == "classifier_stage1"
    n.attach_classifier(C("get_time", 0.5))
    assert n._classifier_stage1("x") is None
    n.attach_classifier(C("take_note", 0.99))       # needs an entity -> LLM
    assert n._classifier_stage1("x") is None


# ------------------------------------------------------------- phase 3
def test_risk_classes():
    assert risk_for("get_notes") == READ and risk_for("take_note") == WRITE
    assert risk_for("remove_habit") == DESTRUCTIVE and risk_for("send_email") == EXTERNAL
    assert ORCHESTRATOR_CONFIRMED <= ALL_INTENTS
    assert all(risk_for(i) == DESTRUCTIVE for i in ORCHESTRATOR_CONFIRMED)


def test_entity_specs():
    assert apply_spec("add_habit", {"habit": ["read"]}) == ({"name": "read"}, [])
    assert apply_spec("pluto_log_expense", {"amount": "Rs 500"})[0]["amount"] == 500
    assert apply_spec("pluto_log_expense", {"amount": "5k"})[0]["amount"] == "5k"
    assert apply_spec("take_note", {})[1] == ["content"] and requires_entities("take_note")
    assert not requires_entities("get_time")
    for bad in (None, 3, [], "x"):
        apply_spec("take_note", bad)


def test_route_guards():
    assert apply_guards("athena_search", {}, "search the web for latest iPhone price")[0] == "hephaestus_search_web"
    assert apply_guards("iris_find_person", {"person": "my dog"}, "pictures of my dog")[0] == "iris_search"
    assert apply_guards("iris_find_person", {"person": "Priya"}, "photos of Priya")[0] == "iris_find_person"
    assert apply_guards("dionysus_recommend_movie", {}, "recommend a movie")[0] == "dionysus_recommend_movie"


def _orch(*mods):
    from modules.hecate.engine import HecateEngine
    from modules.hestia.orchestrator import HestiaOrchestrator
    o = HestiaOrchestrator()
    o.register_hecate(HecateEngine())
    for m in mods:
        o.register(m)
    return o


def test_destructive_intent_needs_confirmation():
    from modules.base import BaseModule
    seen = []

    class Art(BaseModule):
        def can_handle(self, intent): return intent in ("remove_habit",)
        def handle(self, intent, entities, context):
            seen.append(dict(entities)); return {"response": "Removed.", "data": {}, "confidence": 0.9}
        def get_context(self): return {}
    Art.name = "artemis"
    o = _orch(Art())
    ask = o.dispatch("remove my habit reading", {"intent": "remove_habit", "entities": {"name": "reading"}, "confidence": 0.9})
    assert "Say yes" in ask and seen == []
    assert "Removed" in o.dispatch("yes", {"intent": "chat", "entities": {}, "confidence": 0.9})
    assert seen and seen[0]["_confirmed"] is True
    o.dispatch("remove my habit reading", {"intent": "remove_habit", "entities": {"name": "reading"}, "confidence": 0.9})
    o.dispatch("no", {"intent": "chat", "entities": {}, "confidence": 0.9})
    assert len(seen) == 1                       # declined: nothing more ran


def test_boundary_wrong_types_never_raise():
    from modules.base import BaseModule

    class Ares(BaseModule):
        def can_handle(self, intent): return True
        def handle(self, intent, entities, context):
            return {"response": entities["topic"].lower(), "data": {}, "confidence": 0.9}
        def get_context(self): return {}
    Ares.name = "ares"
    o = _orch(Ares())
    out = o.dispatch("summarize my recent work and highlight gaps",
                     {"intent": "ares_competitive_analysis", "entities": {"topic": ["gaps"]}, "confidence": 0.9})
    assert out == "gaps"


# ------------------------------------------------------------- phase 4 / 5
sys.path.insert(0, os.path.dirname(__file__))
from test_hermes_backlog import Agent, make, TZ  # noqa: E402


def test_calendar_create_from_combined_phrase_and_verified():
    a = Agent()
    r = make(a).handle("create_event", {"title": "Gym", "time": "tomorrow at 5pm"}, {})
    assert "Done" in r["response"] and a.created[0]["start"].hour == 17
    assert a.created[0]["start"].date() == (datetime.now(TZ) + timedelta(days=1)).date()


def test_unverified_create_is_reported_honestly():
    a = Agent()
    a.create_event = lambda *x, **k: True          # backend says ok, nothing stored
    r = make(a).handle("create_event", {"title": "Gym", "date": "tomorrow", "time": "5pm"}, {})
    assert "couldn't find it" in r["response"]


def test_make_it_6_instead_moves_last_event():
    a = Agent()
    h = make(a)
    h.handle("create_event", {"title": "Gym", "date": "tomorrow", "time": "5pm"}, {})
    r = h.handle("update_last_event", {"time": "6"}, {})
    assert "moved" in r["response"], r
    assert [c["start"].hour for c in a.created] == [17, 18]       # keeps pm
    assert [e.title for e in a.events].count('Gym') >= 1


def test_update_without_recent_event_asks():
    r = make(Agent()).handle("update_last_event", {"time": "6"}, {})
    assert "recent event" in r["response"]


def test_nlu_response_never_displayed():
    src = open(os.path.join(_ROOT, "modules", "hestia", "orchestrator.py"), encoding="utf-8").read()
    assert 'nlu_result.get("response")' not in src and 'nlu_result["response"]' not in src


def test_web_search_strips_command_words():
    from modules.hephaestus.engine import HephaestusEngine
    seen = []

    class B:
        def search_web(self, q): seen.append(q); return "result"
    e = HephaestusEngine.__new__(HephaestusEngine)
    e._browser = B()
    e._search_web({"raw_query": "search the web for latest iPhone price"})
    assert seen == ["latest iPhone price"]


# ------------------------------------------------------------- phase 6
def test_golden_eval_deterministic_layer_meets_thresholds():
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    import eval_golden
    import yaml
    cases = [c for c in yaml.safe_load(open(os.path.join(_ROOT, "tests", "golden", "eval_set.yaml"),
                                            encoding="utf-8")) if c["layer"] == "deterministic"]
    assert len(cases) >= 150
    s = eval_golden.summarise(eval_golden.run(cases, eval_golden.build_nlu(False), False))
    assert s["false_write_rate"] == 0.0 and s["accuracy"] >= 0.97, s


def test_stage_timings_and_budget():
    from core.observability import StageTimings
    t = StageTimings()
    for ms in (10, 20, 5000):
        t.record("nlu", ms)
    s = t.summary()["nlu"]
    assert s["n"] == 3 and s["over_budget"] is True and "OVER BUDGET" in t.format()


def test_clarification_only_when_ambiguous():
    from core.clarify import clarification_for as c
    assert "web, your documents" in c("search taxes", "chat", 0.3)
    assert c("search the web for taxes", "chat", 0.3) is None
    assert c("search taxes", "athena_search", 0.9) is None


def test_credentials_file_has_no_live_secret():
    import json
    d = json.load(open(os.path.join(_ROOT, "config", "google_credentials.json")))
    assert d["installed"]["client_secret"].startswith("REDACTED")


# ------------------------------------------- fixes from the first live run
def test_iso_datetime_in_one_field_is_split():
    from core.temporal import split_date_time
    assert split_date_time("", "2026-10-13T17:00:00") == ("2026-10-13", "17:00")
    assert split_date_time("2026-10-13T17:00:00+05:30", "") == ("2026-10-13", "17:00")


def test_hermes_creates_event_from_iso_time_entity():
    a = Agent()
    r = make(a).handle("create_event", {"title": "gym", "time": "2026-10-13T17:00:00"}, {})
    assert "Done" in r["response"]
    assert (a.created[0]["start"].month, a.created[0]["start"].day, a.created[0]["start"].hour) == (10, 13, 17)


def test_do_you_know_is_not_a_filler():
    import re
    src = open(os.path.join(_ROOT, "main.py"), encoding="utf-8").read()
    m = re.search(r'_FILLER_RE = re\.compile\(\s*(r".*?"),\s*re\.IGNORECASE\)', src, re.S)
    f = re.compile(eval(m.group(1)), re.I)
    assert f.sub("", "what do you know about me?").strip() == "what do you know about me?"
    assert f.sub("", "i was, you know, tired").strip() == "i was, tired"


def test_salvage_state_fact_and_no_cross_subject_conflict():
    from modules.mnemosyne.keys import salvage_fact
    assert salvage_fact("make a note that i am vegetarian") == ("user_is_vegetarian", "yes")
    from modules.mnemosyne.engine import MnemosyneEngine
    class DB:
        def get_all_facts(self, limit=1000):
            return [{"key": "user_name", "value": "Akhil"}]
    e = MnemosyneEngine.__new__(MnemosyneEngine)
    e.db = DB()
    assert e.check_for_contradiction("sister_name", "Priya") is None


def test_restaurant_without_details_asks():
    from modules.dionysus.engine import DionysusEngine
    e = DionysusEngine.__new__(DionysusEngine)
    e._browser = object()
    r = e._find_restaurant({})
    assert r["data"]["needs_clarification"] and "what are you in the mood for" in r["response"].lower()


def test_confirmation_prompt_names_the_target():
    from modules.base import BaseModule

    class Art(BaseModule):
        def can_handle(self, intent): return True
        def handle(self, intent, entities, context): return {"response": "x", "data": {}, "confidence": 0.9}
        def get_context(self): return {}
    Art.name = "artemis"
    o = _orch(Art())
    out = o.dispatch("remove my habit drink water",
                     {"intent": "remove_habit", "entities": {"name": "drink water"}, "confidence": 0.9})
    assert "drink water" in out


# ------------------------------------------- fixes from the second live run
def test_hinglish_aliases():
    from core.intent_aliases import IntentAliasResolver
    r = IntentAliasResolver(path=os.path.join(_ROOT, "config", "intent_aliases.yaml"))
    assert r.resolve("mera naam kya hai") == "get_user_info"
    assert r.resolve("mere notes dikhao") == "get_notes"


def test_poem_and_brainstorm_get_clean_topic():
    from core.slot_extract import extract_slots
    assert extract_slots("Write me a short poem about the ocean") == {
        "intent": "orpheus_write_poem", "entities": {"topic": "the ocean"}}
    assert extract_slots("Brainstorm ideas for a birthday gift")["entities"]["topic"] == "a birthday gift"
    assert extract_slots("Revise my poem") is None


def test_bare_search_asks_before_any_model_call(monkeypatch):
    n = _nlu()
    monkeypatch.setattr(n, "_call_llm", lambda p: (_ for _ in ()).throw(AssertionError("LLM called")))
    out = n.understand("search taxes")
    assert "web, your documents" in out["clarify"]
    from core.clarify import ambiguous_search_question as a
    assert a("find a good pizza place") is None and a("search the web for taxes") is None


def test_orchestrator_returns_clarify_text():
    o = _orch()
    out = o.dispatch("search taxes", {"intent": "chat", "entities": {}, "confidence": 0.97,
                                      "clarify": "Where should I look?"})
    assert out == "Where should I look?"


def test_yes_no_skips_model_only_when_pending(monkeypatch):
    n = _nlu()
    state = {"pending": False}
    n.set_pending_probe(lambda t: state["pending"] and t.strip().lower() in ("no", "yes"))
    monkeypatch.setattr(n, "_health_check", lambda: False)
    assert n.understand("no").get("source") != "pending_reply"
    state["pending"] = True
    assert n.understand("no")["source"] == "pending_reply"


def test_is_pending_reply_on_orchestrator():
    from modules.base import BaseModule

    class Art(BaseModule):
        def can_handle(self, intent): return True
        def handle(self, intent, entities, context): return {"response": "ok", "data": {}, "confidence": 0.9}
        def get_context(self): return {}
    Art.name = "artemis"
    o = _orch(Art())
    assert not o.is_pending_reply("no")
    o.dispatch("remove my habit x", {"intent": "remove_habit", "entities": {}, "confidence": 0.9})
    assert o.is_pending_reply("no") and not o.is_pending_reply("what time is it")


def test_confirmation_without_subject_quotes_what_you_said():
    from modules.base import BaseModule

    class Art(BaseModule):
        def can_handle(self, intent): return True
        def handle(self, intent, entities, context): return {"response": "x", "data": {}, "confidence": 0.9}
        def get_context(self): return {}
    Art.name = "artemis"
    out = _orch(Art()).dispatch("remove my habit drink water",
                                {"intent": "remove_habit", "entities": {}, "confidence": 0.9})
    assert "drink water" in out


def test_state_fact_reply_wording():
    from modules.mnemosyne.engine import MnemosyneEngine
    e = MnemosyneEngine.__new__(MnemosyneEngine)
    class DB:
        def get_all_facts(self, limit=1000): return []
    e.db = DB()
    e.learn = lambda k, v, confidence=None: {"deduplicated": False, "matched_key": None}
    e.check_for_contradiction = lambda k, v: None
    r = e._handle_learn_fact({"raw_query": "make a note that i am vegetarian"})
    assert "you're vegetarian" in r["response"]


# ------------------------------------------- fixes from the third live run
def test_create_event_slots():
    from core.slot_extract import extract_slots as e
    assert e("Create an event tomorrow at 5pm called Gym") == {
        "intent": "create_event", "entities": {"title": "Gym", "time": "tomorrow at 5pm"}}
    assert e("add a meeting called standup tomorrow at 9am")["entities"]["title"] == "standup"
    assert e("create an event called party") is None        # no when -> let the model ask


def test_greetings_thanks_and_brief_aliases():
    from core.intent_aliases import IntentAliasResolver
    r = IntentAliasResolver(path=os.path.join(_ROOT, "config", "intent_aliases.yaml"))
    assert r.resolve("shukriya") == "chat" and r.resolve("Good evening Hestia") == "chat"
    assert r.resolve("thanks, set a timer") is None          # exact-match only
    assert r.resolve("give me my morning brief") == "get_agenda"
    assert r.resolve("mera naam batao") == "get_user_info"


def test_retrieval_has_chat_food_and_calendar_examples():
    r = ExampleRetriever(os.path.join(_ROOT, "config", "nlu_examples.yaml"))
    assert r.top("I'm tired", 3)[0]["intent"] == "chat"
    assert r.top("I am hungry", 2)[0]["intent"] == "dionysus_find_restaurant"
    assert "create_event" in [x["intent"] for x in r.top("create an event tomorrow at 5pm called Gym", 3)]


def test_summarize_command_alone_asks_for_text():
    from modules.metis.engine import MetisEngine
    e = MetisEngine.__new__(MetisEngine)
    r = e._summarize_text({"raw_query": "summarize my recent work and highlight gaps"})
    assert "What text should I summarise" in r["response"]


def test_user_info_hides_generated_artefacts_and_understands_naam():
    from modules.hestia.core_module import CoreModule
    class DB:
        def get_all_facts(self, limit=10):
            return [{"key": "orpheus_poem_x", "value": "a long poem " * 50},
                    {"key": "sister_name", "value": "Priya"}]
        def get_fact(self, k): return "Akhil" if k == "user_name" else None
    class Mem: db = DB()
    m = CoreModule.__new__(CoreModule)
    m._memory = Mem()
    out = m._get_user_info({}, "mera naam kya hai")
    assert out["response"] == "Akhil"
    listing = m._get_user_info({}, "what do you know about me")["response"]
    assert "orpheus" not in listing and "sister name: Priya" in listing


def test_empty_web_search_message_is_honest():
    from modules.hephaestus.engine import HephaestusEngine
    class B:
        def search_web(self, q): return f"No results found for '{q}'."
    e = HephaestusEngine.__new__(HephaestusEngine)
    e._browser = B()
    r = e._search_web({"query": "iphone price"})
    assert "blocked" in r["response"]


# ------------------------------------------- fixes from the fourth live run
def test_move_uses_the_new_events_id_not_a_stale_same_title_one():
    """A leftover 'gym' at the same time used to be remembered as the new event."""
    from test_hermes_backlog import _event, _tomorrow_at
    a = Agent()
    deleted = []
    a.delete_event = lambda eid: (deleted.append(eid), True)[1]
    stale = _event("gym", _tomorrow_at(18))
    object.__setattr__(stale, "event_id", "STALE")
    a.events.append(stale)                      # already on the calendar at 18:00
    # fake ids are unique per created event, like Google's
    n = {"i": 0}
    orig = a.create_event
    def create(title, start_dt, end_dt=None, **k):
        orig(title, start_dt, end_dt, **k)
        n["i"] += 1
        object.__setattr__(a.events[-1], "event_id", f"NEW{n['i']}")
        return True
    a.create_event = create
    h = make(a)
    h.handle("create_event", {"title": "gym", "date": "tomorrow", "time": "5pm", "_confirmed": True}, {})
    h.handle("update_last_event", {"time": "6"}, {})
    assert deleted == ["NEW1"]                  # removed what *we* created, never STALE


def test_multi_intent_with_clarify_half_is_not_split():
    src = open(os.path.join(_ROOT, "main.py"), encoding="utf-8").read()
    assert '"clarify_intent" in intents' in src


def test_iris_refuses_to_load_clip_when_ram_is_low(monkeypatch):
    import types
    from modules.iris import embeddings
    fake = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=1 * 1024 ** 3))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    e = embeddings.ClipEmbedder.__new__(embeddings.ClipEmbedder)
    e._model = None
    e._load_failed = False
    e._model_name = "x"
    assert e._ensure_loaded() is False and e._load_failed is False   # retry allowed later
