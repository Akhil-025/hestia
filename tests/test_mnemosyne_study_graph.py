# tests/test_mnemosyne_study_graph.py
"""
Tests for backlog #33 (SM-2 spaced repetition + morning-brief surfacing),
#31 (knowledge graph) and #32 (graph data for the web UI).

Pure logic (sm2_update, grade_recall, extraction) is tested directly; the
conversational flows go through a real MnemosyneEngine.handle() exactly as
the orchestrator calls it, including the slot-fill contract the review
session relies on.
"""
import os
import shutil
import sys
import tempfile
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_mnemosyne import make_engine  # noqa: E402
from modules.mnemosyne.spaced_repetition import (  # noqa: E402
    StudyStore, grade_recall, sm2_update,
)
from modules.mnemosyne.knowledge_graph import (  # noqa: E402
    KnowledgeGraph, describe_connections, describe_path, extract_cooccurrence,
    extract_entities_heuristic, extract_from_fact, normalise_name,
    parse_llm_extraction, extract_with_llm,
)


@pytest.fixture
def engine():
    tmp = tempfile.mkdtemp()
    eng, _ = make_engine(tmp)
    yield eng
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def store(tmp_path):
    return StudyStore(str(tmp_path / "s.db"))


@pytest.fixture
def kg(tmp_path):
    return KnowledgeGraph(str(tmp_path / "k.db"))


# ---------------------------------------------------------------------------
# SM-2
# ---------------------------------------------------------------------------

def test_sm2_first_two_intervals_are_1_then_6_days():
    a = sm2_update(2.5, 0, 0, 4)
    assert (a["interval_days"], a["reps"]) == (1, 1)
    b = sm2_update(a["ease"], a["interval_days"], a["reps"], 4)
    assert (b["interval_days"], b["reps"]) == (6, 2)


def test_sm2_later_intervals_multiply_by_ease():
    r = sm2_update(2.5, 6, 2, 5)
    assert r["interval_days"] == round(6 * r["ease"])
    assert r["reps"] == 3


def test_sm2_lapse_resets_reps_and_returns_tomorrow():
    r = sm2_update(2.5, 30, 5, 1)
    assert r["lapsed"] and r["reps"] == 0 and r["interval_days"] == 1


def test_sm2_ease_never_drops_below_floor():
    ease = 2.5
    for _ in range(20):
        ease = sm2_update(ease, 1, 0, 0)["ease"]
    assert ease == pytest.approx(1.3)


def test_sm2_perfect_recall_raises_ease_and_hard_recall_lowers_it():
    assert sm2_update(2.5, 6, 2, 5)["ease"] > 2.5
    assert sm2_update(2.5, 6, 2, 3)["ease"] < 2.5


def test_sm2_clamps_out_of_range_quality():
    assert sm2_update(2.5, 0, 0, 99)["reps"] == 1
    assert sm2_update(2.5, 0, 0, -4)["lapsed"]


@pytest.mark.parametrize("answer,expected,minimum,maximum", [
    ("Priya", "Priya", 5, 5),
    ("it is priya", "Priya", 4, 4),
    ("priyaa", "Priya", 4, 4),
    ("idk", "Priya", 1, 1),
    ("Rohan", "Priya", 1, 1),
    ("", "Priya", 1, 1),
])
def test_grade_recall(answer, expected, minimum, maximum):
    assert minimum <= grade_recall(answer, expected) <= maximum


# ---------------------------------------------------------------------------
# StudyStore
# ---------------------------------------------------------------------------

def test_new_card_is_due_immediately(store):
    today = date(2026, 10, 1)
    assert store.add_card("entropy", today=today) is True
    assert [c["fact_key"] for c in store.due_cards(today)] == ["entropy"]


def test_adding_the_same_card_twice_is_a_noop(store):
    store.add_card("entropy")
    assert store.add_card("entropy") is False
    assert store.stats()["total"] == 1


def test_review_pushes_due_date_out_and_removes_from_due_list(store):
    today = date(2026, 10, 1)
    store.add_card("entropy", today=today)
    card = store.review("entropy", 4, today=today)
    assert card["due_date"] == (today + timedelta(days=1)).isoformat()
    assert store.due_cards(today) == []
    assert len(store.due_cards(today + timedelta(days=1))) == 1


def test_lapse_increments_lapse_count_and_resets_reps(store):
    today = date(2026, 10, 1)
    store.add_card("entropy", today=today)
    store.review("entropy", 5, today=today)
    card = store.review("entropy", 1, today=today)
    assert card["lapses"] == 1 and card["reps"] == 0


def test_review_of_unknown_card_returns_none(store):
    assert store.review("nope", 4) is None


def test_most_overdue_card_comes_first(store):
    d = date(2026, 10, 10)
    store.add_card("newer", today=date(2026, 10, 9))
    store.add_card("older", today=date(2026, 10, 1))
    assert [c["fact_key"] for c in store.due_cards(d)] == ["older", "newer"]


def test_add_card_requires_a_key(store):
    with pytest.raises(ValueError):
        store.add_card("")


# ---------------------------------------------------------------------------
# Study flow through the engine (slot-fill contract)
# ---------------------------------------------------------------------------

def _answer(engine, intent, reply, slot_entities):
    """Do what the orchestrator does with a slot-fill: re-dispatch with the reply."""
    ents = dict(slot_entities)
    ents["answer"] = reply
    ents.setdefault("raw_query", reply)
    return engine.handle(intent, ents, {})


def test_add_study_fact_stores_fact_and_creates_card(engine):
    r = engine.handle("add_study_fact", {"key": "Entropy", "value": "a measure of disorder"}, {})
    assert "Added 1" in r["response"]
    assert engine.db.get_fact("entropy") == "a measure of disorder"
    assert engine.study_store.get_card("entropy") is not None


def test_add_study_fact_parses_raw_query_when_no_entities(engine):
    r = engine.handle("add_study_fact", {},
                      {"raw_query": "add to my study material: entropy is a measure of disorder"})
    assert "Added 1" in r["response"]
    assert engine.db.get_fact("entropy") == "a measure of disorder"


def test_add_study_fact_tags_an_existing_fact(engine):
    engine.learn("carnot_cycle", "a reversible engine cycle")
    r = engine.handle("add_study_fact", {"topic": "carnot"}, {})
    assert "Added 1" in r["response"]


def test_add_study_fact_with_nothing_matching_asks_instead_of_inventing(engine):
    r = engine.handle("add_study_fact", {"topic": "zzz"}, {})
    assert "don't have a matching fact" in r["response"]
    assert engine.study_store.stats()["total"] == 0


def test_review_session_end_to_end(engine):
    engine.handle("add_study_fact", {"key": "entropy", "value": "disorder"}, {})
    engine.handle("add_study_fact", {"key": "enthalpy", "value": "heat content"}, {})
    first = engine.handle("review_study", {}, {})
    assert first["data"]["needs_clarification"] and first["data"]["missing_slot"] == "answer"
    assert "Card 1 of 2" in first["response"]

    key1 = engine._study_session["queue"][0]
    right = engine.db.get_fact(key1)
    second = _answer(engine, "review_study", right, first["data"]["slot_entities"])
    assert "Correct!" in second["response"] and "Card 2 of 2" in second["response"]

    final = _answer(engine, "review_study", "no idea", second["data"]["slot_entities"])
    assert "Not quite" in final["response"]
    assert "recalled 1 of 2" in final["response"]
    assert engine._study_session is None
    assert engine.study_store.due_cards() == []        # both rescheduled


def test_review_with_nothing_due_says_so(engine):
    engine.handle("add_study_fact", {"key": "entropy", "value": "disorder"}, {})
    engine.study_store.review("entropy", 4)
    r = engine.handle("review_study", {}, {})
    assert "Nothing due" in r["response"]


def test_review_with_no_cards_explains_how_to_start(engine):
    assert "haven't added any study material" in engine.handle("review_study", {}, {})["response"]


def test_review_skips_cards_whose_fact_was_forgotten(engine):
    engine.handle("add_study_fact", {"key": "entropy", "value": "disorder"}, {})
    engine.db.delete_fact("entropy")       # bypass forget() to leave an orphan card
    r = engine.handle("review_study", {}, {})
    assert "Review finished" in r["response"]
    assert engine.study_store.get_card("entropy") is None


def test_forgetting_a_fact_removes_its_study_card_and_graph_edges(engine):
    engine.handle("add_study_fact", {"key": "entropy", "value": "disorder"}, {})
    engine.forget("entropy")
    assert engine.study_store.get_card("entropy") is None
    assert engine.knowledge_graph.find_entity("disorder") is None


def test_quit_ends_a_review(engine):
    engine.handle("add_study_fact", {"key": "entropy", "value": "disorder"}, {})
    first = engine.handle("review_study", {}, {})
    r = _answer(engine, "review_study", "quit", first["data"]["slot_entities"])
    assert "Review finished" in r["response"]
    assert engine._study_session is None


def test_answer_with_no_session_is_handled_gracefully(engine):
    r = engine.handle("review_study", {"answer": "x"}, {})
    assert "No review in progress" in r["response"]


def test_study_brief_is_empty_when_nothing_due_and_counts_when_due(engine):
    assert engine.get_study_brief() == ""
    engine.handle("add_study_fact", {"key": "entropy", "value": "disorder"}, {})
    assert "1 study card due" in engine.get_study_brief()
    engine.handle("add_study_fact", {"key": "enthalpy", "value": "heat"}, {})
    assert "2 study cards due" in engine.get_study_brief()
    assert engine.get_study_brief(today=date.today() - timedelta(days=3)) == ""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_fact_becomes_user_relation_with_key_as_label():
    ents, rels = extract_from_fact("sister_name", "Priya")
    assert ("Priya", "thing") in ents
    assert rels == [("You", "sister name", "Priya")]


def test_long_fact_values_are_mined_for_proper_nouns_not_stored_whole():
    ents, rels = extract_from_fact(
        "bio", "I studied at Indian Institute of Technology Bombay under Professor Rao for many years")
    names = [e[0] for e in ents]
    assert "Indian Institute of Technology Bombay" in names and "Professor Rao" in names
    assert all(r[0] == "You" for r in rels)


def test_wikilinks_tags_and_proper_nouns_are_extracted():
    found = dict(extract_entities_heuristic("Review [[Entropy]] and #gate-prep with Professor Rao."))
    assert found["Entropy"] == "concept" and found["gate prep"] == "tag" and "Professor Rao" in found


def test_and_does_not_fuse_two_proper_nouns():
    names = [n for n, _ in extract_entities_heuristic("Professor Rao and Carnot disagreed")]
    assert "Professor Rao" in names and "Carnot" in names


def test_cooccurrence_links_only_within_a_sentence():
    ents, rels = extract_cooccurrence("Priya met Rohan. Meera lives in Pune.")
    pairs = {frozenset((a, b)) for a, _, b in rels}
    assert frozenset(("Priya", "Rohan")) in pairs
    assert frozenset(("Priya", "Meera")) not in pairs


def test_llm_output_is_validated_piecemeal():
    ents, rels = parse_llm_extraction(
        '{"entities":[{"name":"Rao","type":"person"},{"bad":1},{"name":"X","type":"nonsense"}],'
        '"relations":[{"subject":"Rao","relation":"teaches","object":"Thermo"},'
        '{"subject":"A","relation":"","object":"B"},{"subject":"Q","relation":"is","object":"Q"}]}')
    assert ("Rao", "person") in ents and ("X", "thing") in ents
    assert rels == [("Rao", "teaches", "Thermo")]
    assert ("Thermo", "thing") in ents              # relation endpoints are added


def test_llm_garbage_yields_nothing_and_never_raises():
    assert parse_llm_extraction("not json") == ([], [])
    assert parse_llm_extraction('[1,2]') == ([], [])


def test_extract_with_llm_falls_back_to_heuristic_on_failure():
    class Boom:
        def generate(self, *a, **k):
            raise RuntimeError("down")
    ents, _ = extract_with_llm(Boom(), "Priya met Rohan in Pune.")
    assert ("Priya", "thing") in ents


# ---------------------------------------------------------------------------
# Graph store
# ---------------------------------------------------------------------------

def _seed(kg):
    e, r = extract_from_fact("sister_name", "Priya")
    kg.add_extraction("fact:sister_name", e, r)
    e, r = extract_cooccurrence("Priya studies at IIT Bombay.")
    kg.add_extraction("note:a", e, r)


def test_add_extraction_is_idempotent(kg):
    _seed(kg)
    before = kg.stats()
    _seed(kg)
    assert kg.stats() == before


def test_find_entity_is_case_and_phrase_tolerant(kg):
    _seed(kg)
    assert kg.find_entity("PRIYA")["name"] == "Priya"
    assert kg.find_entity("my sister priya")["name"] == "Priya"
    assert kg.find_entity("priy")["name"] == "Priya"
    assert kg.find_entity("quantum zebra") is None


def test_neighbors_report_direction_and_relation(kg):
    _seed(kg)
    priya = kg.find_entity("priya")
    by_name = {n["name"]: n for n in kg.neighbors(priya["id"])}
    assert by_name["You"]["direction"] == "in" and by_name["You"]["relation"] == "sister name"
    assert by_name["IIT Bombay"]["direction"] == "both"


def test_path_between_entities_and_none_when_disconnected(kg):
    _seed(kg)
    you, iit = kg.find_entity("you"), kg.find_entity("iit bombay")
    path = kg.find_path(you["id"], iit["id"])
    assert [h["to"] for h in path] == ["Priya", "IIT Bombay"]
    kg.add_extraction("x", [("Island", "thing")], [])
    assert kg.find_path(you["id"], kg.find_entity("island")["id"]) is None
    assert kg.find_path(you["id"], you["id"]) == []


def test_remove_source_deletes_only_what_nothing_else_supports(kg):
    _seed(kg)
    kg.remove_source("fact:sister_name")
    assert kg.find_entity("priya") is not None          # still mentioned by note:a
    assert kg.find_entity("you") is None                # only that fact mentioned You
    kg.remove_source("note:a")
    assert kg.stats() == {"entities": 0, "edges": 0}


def test_graph_data_caps_nodes_and_drops_dangling_links(kg):
    for i in range(10):
        kg.add_extraction(f"s{i}", [("Hub", "thing"), (f"Leaf{i}", "thing")],
                          [("Hub", "links to", f"Leaf{i}")])
    data = kg.graph_data(max_nodes=4)
    ids = {n["id"] for n in data["nodes"]}
    assert len(data["nodes"]) == 4
    assert all(l["source"] in ids and l["target"] in ids for l in data["links"])
    assert data["nodes"][0]["name"] == "Hub"            # best connected first


def test_describe_helpers():
    e = {"name": "Priya"}
    assert "nothing is connected" in describe_connections(e, [])
    assert "can't find a connection" in describe_path(None)
    assert "same thing" in describe_path([])


def test_normalise_name():
    assert normalise_name("  Indian_Institute  of  Technology! ") == "indian institute of technology"


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

def test_learning_a_fact_updates_the_graph(engine):
    engine.learn("sister_name", "Priya")
    r = engine.handle("graph_connections", {"entity": "Priya"}, {})
    assert "sister name" in r["response"] and "You" in r["response"]


def test_graph_connections_reads_the_topic_from_the_raw_query(engine):
    engine.learn("sister_name", "Priya")
    r = engine.handle("graph_connections", {}, {"raw_query": "what connects to Priya?"})
    assert "sister name" in r["response"]


def test_graph_path_question(engine):
    engine.learn("sister_name", "Priya")
    engine.learn("university", "IIT Bombay")
    r = engine.handle("graph_connections", {"entity": "Priya", "to": "IIT Bombay"}, {})
    assert r["data"]["path"] and "Connection:" in r["response"]


def test_updating_a_fact_replaces_its_old_graph_edge(engine):
    engine.learn("sister_name", "Priya")
    engine.learn("sister_name", "Meera")
    assert engine.knowledge_graph.find_entity("priya") is None
    assert engine.knowledge_graph.find_entity("meera") is not None


def test_unknown_entity_is_reported_not_invented(engine):
    r = engine.handle("graph_connections", {"entity": "Zanzibar"}, {})
    assert "don't have anything called Zanzibar" in r["response"]


def test_empty_graph_message(engine):
    assert "graph is empty" in engine.handle("graph_connections", {}, {})["response"]


def test_other_modules_generated_facts_are_kept_out_of_the_graph(engine):
    engine.learn("ares_swot_plan", "Strengths: Alice Bob Carol are great")
    assert engine.knowledge_graph.stats()["entities"] == 0


def test_rebuild_graph_is_incremental(engine):
    engine.db.set_fact("hobby", "Chess")                 # bypasses learn(): not in graph yet
    first = engine.handle("graph_connections", {}, {"raw_query": "rebuild the knowledge graph"})
    assert "Graph updated from 1 fact" in first["response"]
    second = engine.handle("graph_connections", {}, {"raw_query": "rebuild the knowledge graph"})
    assert "from 0 fact" in second["response"]


def test_graph_data_for_the_web_ui(engine):
    engine.learn("sister_name", "Priya")
    data = engine.get_graph_data()
    assert {n["name"] for n in data["nodes"]} == {"You", "Priya"}
    assert data["links"][0]["relation"] == "sister name"
