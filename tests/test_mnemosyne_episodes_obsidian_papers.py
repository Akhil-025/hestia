# tests/test_mnemosyne_episodes_obsidian_papers.py
"""
Tests for backlog #43 (episodic clustering), #39 (Obsidian vault sync) and
#47 (arXiv monitoring).

Nothing here touches the network or a real vault: the paper monitor is fed
a saved Atom feed through its injectable fetch function, and the Obsidian
tests build a throwaway vault in tmp_path. The embedding-dependent branch
of clustering is exercised with hand-made vectors, so it is deterministic.
"""
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_mnemosyne import make_engine  # noqa: E402
from modules.mnemosyne.episodes import (  # noqa: E402
    DENSE, SPARSE, ClusterConfig, EpisodeState, EpisodeStore, assign_interactions,
    parse_timestamp, tokenize,
)
from modules.mnemosyne.knowledge_graph import KnowledgeGraph  # noqa: E402
from modules.mnemosyne.obsidian import (  # noqa: E402
    ObsidianVault, chunk_note, extract_tags, extract_wikilinks, parse_note,
    plain_text, safe_filename,
)
from modules.mnemosyne.paper_monitor import (  # noqa: E402
    PaperMonitor, build_arxiv_url, digest_markdown, fallback_digest,
    normalise_arxiv_id, parse_arxiv_feed,
)
from modules.mnemosyne.schema import init_db  # noqa: E402


@pytest.fixture
def engine():
    tmp = tempfile.mkdtemp()
    eng, _ = make_engine(tmp)
    yield eng
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "m.db")
    init_db(path)
    return path


def _log(db, rows):
    conn = sqlite3.connect(db)
    for ts, user, resp, intent in rows:
        conn.execute(
            "INSERT INTO interaction_log (user_text, hestia_response, intent, pushed_at) VALUES (?,?,?,?)",
            (user, resp, intent, ts))
    conn.commit()
    conn.close()


_CONVO = [
    ("2026-09-20 10:00:00", "explain the carnot cycle efficiency", "The Carnot cycle...", "chat"),
    ("2026-09-20 10:02:00", "and the entropy change in the carnot cycle", "Entropy...", "chat"),
    ("2026-09-20 10:03:00", "thanks", "You're welcome", "chat"),
    ("2026-09-20 10:04:00", "how much would that cost", "It depends", "chat"),
    ("2026-09-20 15:00:00", "book a flight to Delhi on Friday", "Checking", "hephaestus_check_flight"),
    ("2026-09-20 15:05:00", "any cheaper flight to Delhi", "Looking", "hephaestus_check_flight"),
    ("2026-09-21 09:00:00", "more carnot cycle questions for the gate exam", "Sure", "chat"),
]


# ===========================================================================
# #43 Episodes
# ===========================================================================

def test_tokenize_drops_stopwords_and_plural_s():
    assert tokenize("Please tell me about the engines") == ["engine"]


def test_related_messages_form_one_episode_and_topics_stay_apart(db):
    _log(db, _CONVO)
    store = EpisodeStore(db)
    assert store.cluster_new()["processed"] == 7
    labels = [e["label"] for e in store.list_episodes()]
    assert any("carnot" in l for l in labels) and any("flight" in l for l in labels)
    carnot = store.find_episodes("carnot cycle")[0]
    flight = store.find_episodes("flight delhi")[0]
    assert carnot["id"] != flight["id"]
    assert carnot["size"] >= 4 and flight["size"] == 2


def test_followup_and_filler_join_the_conversation_they_answer(db):
    _log(db, _CONVO)
    store = EpisodeStore(db)
    store.cluster_new()
    carnot = store.find_episodes("carnot")[0]
    queries = [m["query"] for m in store.get_members(carnot["id"])]
    assert "thanks" in queries and "how much would that cost" in queries


def test_lonely_filler_never_becomes_an_episode(db):
    _log(db, [("2026-09-20 10:00:00", "yes", "ok", "chat")])
    store = EpisodeStore(db)
    store.cluster_new()
    assert store.stats()["episodes"] == 0


def test_a_topic_resumed_next_day_rejoins_its_episode(db):
    _log(db, _CONVO)
    store = EpisodeStore(db)
    store.cluster_new()
    carnot = store.find_episodes("carnot")[0]
    assert carnot["end"].startswith("2026-09-21")


def test_clustering_is_incremental_and_idempotent(db):
    _log(db, _CONVO)
    store = EpisodeStore(db)
    store.cluster_new()
    assert store.cluster_new()["processed"] == 0
    before = store.stats()
    _log(db, [("2026-09-21 09:10:00", "carnot engine efficiency again", "ok", "chat")])
    r = store.cluster_new()
    assert r["processed"] == 1 and r["new_episodes"] == 0
    assert store.stats()["episodes"] == before["episodes"]


def test_episodes_expire_from_joinability_after_max_age(db):
    _log(db, [("2026-09-01 10:00:00", "explain the carnot cycle", "x", "chat"),
              ("2026-09-20 10:00:00", "explain the carnot cycle again", "x", "chat")])
    store = EpisodeStore(db)
    store.cluster_new()
    assert store.stats()["episodes"] == 2


def test_find_episodes_ignores_single_message_noise_and_unknown_terms(db):
    _log(db, _CONVO)
    store = EpisodeStore(db)
    store.cluster_new()
    assert store.find_episodes("quantum chromodynamics") == []
    assert store.find_episodes("") == []


def test_dense_vectors_cluster_by_similarity_not_words():
    eps = []
    items = [
        {"id": 1, "query": "alpha topic discussion", "pushed_at": "2026-09-20 10:00:00"},
        {"id": 2, "query": "another wording entirely", "pushed_at": "2026-09-20 11:30:00"},
        {"id": 3, "query": "completely different subject", "pushed_at": "2026-09-20 11:40:00"},
    ]
    out = assign_interactions(eps, items, vectors=[[1, 0], [0.95, 0.05], [0, 1]])
    assert out[1] is out[2] and out[3] is not out[1]
    assert all(e.kind == DENSE for e in eps)


def test_dense_and_sparse_episodes_are_never_compared():
    old = EpisodeState(kind=DENSE, centroid=[1.0, 0.0],
                       start=parse_timestamp("2026-09-20 10:00:00"),
                       end=parse_timestamp("2026-09-20 10:00:00"), size=1)
    eps = [old]
    assign_interactions(eps, [{"id": 9, "query": "alpha topic discussion",
                               "pushed_at": "2026-09-20 10:01:00"}])
    assert len(eps) == 2 and eps[1].kind == SPARSE


def test_embedding_failure_falls_back_to_bag_of_words(db):
    _log(db, _CONVO)
    store = EpisodeStore(db)

    def boom(texts):
        raise RuntimeError("model missing")

    assert store.cluster_new(embed_fn=boom)["processed"] == 7
    assert store.stats()["episodes"] >= 2


def test_parse_timestamp_handles_sqlite_and_iso_formats():
    assert parse_timestamp("2026-09-20 10:00:00") == datetime(2026, 9, 20, 10, 0)
    assert parse_timestamp("2026-09-20T10:00:00+00:00") == datetime(2026, 9, 20, 10, 0)
    assert parse_timestamp("2026-09-20T15:30:00+05:30") == datetime(2026, 9, 20, 10, 0)


def test_recall_episode_intent_answers_about_a_topic(engine):
    for ts, u, r, i in _CONVO:
        engine.db.push_interaction(u, r, i)
    conn = sqlite3.connect(engine.config.db_path)      # give the rows our fixed timestamps
    for n, (ts, *_rest) in enumerate(_CONVO, start=1):
        conn.execute("UPDATE interaction_log SET pushed_at=? WHERE id=?", (ts, n))
    conn.commit()
    conn.close()
    r = engine.handle("recall_episode", {"topic": "carnot cycle"}, {})
    assert "About" in r["response"] and "carnot" in r["response"].lower()
    assert "explain the carnot cycle efficiency" in r["response"]


def test_recall_episode_unknown_topic_is_honest(engine):
    r = engine.handle("recall_episode", {"topic": "quantum chromodynamics"}, {})
    assert "can't find a past stretch" in r["response"] or "haven't grouped" in r["response"]


# ===========================================================================
# #39 Obsidian
# ===========================================================================

def test_frontmatter_forms_are_parsed():
    meta, body = parse_note("---\ntitle: T\ntags: [a, b]\naliases:\n  - x\n  - y\n---\nBody")
    assert meta == {"title": "T", "tags": ["a", "b"], "aliases": ["x", "y"]} and body == "Body"


def test_note_without_or_with_broken_frontmatter_is_all_body():
    assert parse_note("just text") == ({}, "just text")
    assert parse_note("---\nunterminated\ntext")[0] == {}


def test_wikilink_variants_resolve_to_their_target_and_skip_code():
    links = extract_wikilinks("[[A|alias]] [[B#Heading]] ![[C]] `[[no]]`\n```\n[[also no]]\n```\n[[a]]")
    assert links == ["A", "B", "C"]


def test_tags_come_from_frontmatter_and_body_without_headings_or_code():
    tags = extract_tags({"tags": "x, y"}, "# Heading\ntext #z and `#nope`")
    assert tags == ["x", "y", "z"]


def test_plain_text_renders_what_a_reader_sees():
    assert plain_text("see [[Note|the note]] and [[Other]]") == "see the note and Other"


def test_chunks_follow_headings_and_carry_context_and_links():
    body = "# Laws\nThe [[Entropy|entropy]] rises.\n\n## Second\nHeat flows hot to cold. See [[Carnot]]."
    chunks = chunk_note("Thermo", body)
    assert [c["heading"] for c in chunks] == ["Laws", "Laws > Second"]
    assert chunks[0]["text"].startswith("Thermo › Laws\n")
    assert "[[" not in chunks[0]["text"] and chunks[0]["links"] == ["Entropy"]
    assert chunks[1]["links"] == ["Carnot"]


def test_code_fence_is_kept_whole_and_its_links_are_not_extracted():
    chunks = chunk_note("T", "intro\n\n```\nline1\n\nline2 [[x]]\n```\n")
    joined = "\n".join(c["text"] for c in chunks)
    assert "line1\n\nline2" in joined
    assert all(c["links"] == [] for c in chunks)


def test_long_paragraph_splits_but_never_inside_a_wikilink():
    sentence = "Intro words " * 20 + "[[A Very Long Link Name With Spaces]] " + "trailing words. " * 40
    chunks = chunk_note("T", sentence, max_chars=200)
    assert len(chunks) > 1
    for c in chunks:
        assert c["text"].count("[[") == 0
    assert any("A Very Long Link Name With Spaces" in c["text"] for c in chunks)


def test_safe_filename_strips_path_and_wiki_hazards():
    assert safe_filename("a/b:c*?") == "a b c"
    assert safe_filename("../../x") == "x"
    assert safe_filename("###") == "Untitled"
    assert "/" not in safe_filename("x" * 200 + "/y")


@pytest.fixture
def vault(tmp_path, db):
    root = tmp_path / "vault"
    (root / "sub").mkdir(parents=True)
    (root / ".obsidian").mkdir()
    (root / "Hestia").mkdir()
    (root / "Thermo.md").write_text(
        "---\ntitle: Thermodynamics\ntags: [physics]\n---\n# Laws\nSee [[Carnot Cycle]]. #exam\n", encoding="utf-8")
    (root / "sub" / "Carnot Cycle.md").write_text("Reversible. Back to [[Thermodynamics]].", encoding="utf-8")
    (root / ".obsidian" / "hidden.md").write_text("hidden", encoding="utf-8")
    (root / "Hestia" / "generated.md").write_text("old output", encoding="utf-8")
    return ObsidianVault(str(root), db), root


def test_scan_skips_hidden_tool_and_writeback_folders(vault):
    v, _ = vault
    assert [v._rel(p) for p in v.scan()] == ["Thermo.md", "sub/Carnot Cycle.md"]


def test_sync_adds_updates_removes_and_skips_unchanged(vault, db):
    v, root = vault
    added, deleted = [], []
    kg = KnowledgeGraph(db)
    run = lambda: v.sync(added.extend, deleted.extend, kg)

    s = run()
    assert (s["added"], s["updated"], s["removed"], s["errors"]) == (2, 0, 0, 0)
    assert {a["metadata"]["type"] for a in added} == {"note"}
    assert all(a["id"].startswith("obsidian:") for a in added)

    assert run()["unchanged"] == 2

    (root / "sub" / "Carnot Cycle.md").write_text("Changed. See [[Entropy]].", encoding="utf-8")
    (root / "Thermo.md").unlink()
    s = run()
    assert (s["updated"], s["removed"]) == (1, 1)
    assert "obsidian:Thermo.md:0" in deleted
    assert kg.find_entity("Thermodynamics") is None      # its only sources were deleted/rewritten
    assert kg.find_entity("Entropy") is not None         # the rewritten note's new link


def test_sync_builds_graph_links_and_tags(vault, db):
    v, _ = vault
    kg = KnowledgeGraph(db)
    v.sync(graph=kg)
    thermo = kg.find_entity("Thermodynamics")
    rels = {(n["relation"], n["name"]) for n in kg.neighbors(thermo["id"])}
    assert ("links to", "Carnot Cycle") in rels and ("tagged", "physics") in rels


def test_one_unreadable_note_does_not_abort_the_sync(vault, monkeypatch):
    v, _ = vault
    real = Path.read_text

    def flaky(self, *a, **k):
        if self.name == "Thermo.md":
            raise OSError("locked")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky)
    s = v.sync()
    assert s["errors"] == 1 and s["added"] == 1


def test_symlink_pointing_outside_the_vault_is_not_ingested(vault, tmp_path):
    v, root = vault
    outside = tmp_path / "secret.md"
    outside.write_text("private", encoding="utf-8")
    try:
        (root / "leak.md").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert "leak.md" not in [v._rel(p) for p in v.scan()]


def test_writeback_creates_new_notes_and_never_overwrites(vault):
    v, root = vault
    p1 = v.write_note("Weekly digest: W40", "body", tags=["digest"], links=["Carnot Cycle"])
    p2 = v.write_note("Weekly digest: W40", "second")
    assert p1.parent == root / "Hestia" and p1.name == "Weekly digest W40.md"
    assert p2.name == "Weekly digest W40 2.md"
    text = p1.read_text(encoding="utf-8")
    assert text.startswith("---\ngenerated_by: hestia") and "tags: [digest]" in text
    assert "- [[Carnot Cycle]]" in text
    assert (root / "Thermo.md").read_text(encoding="utf-8").startswith("---\ntitle: Thermodynamics")


def test_writeback_cannot_escape_the_hestia_folder(vault):
    v, root = vault
    p = v.write_note("../../evil", "x")
    assert p.parent == root / "Hestia" and p.name == "evil.md"


def test_writeback_does_not_create_a_missing_vault(db, tmp_path):
    v = ObsidianVault(str(tmp_path / "nope"), db)
    with pytest.raises(FileNotFoundError):
        v.write_note("t", "b")
    assert not (tmp_path / "nope").exists()


def test_generated_notes_are_not_reingested(vault):
    v, _ = vault
    v.write_note("Fresh output", "x")
    assert "Hestia/Fresh output.md" not in [v._rel(p) for p in v.scan()]


def test_engine_obsidian_is_off_by_default(engine):
    assert engine.obsidian is None
    assert "off" in engine.handle("obsidian_sync", {}, {})["response"]
    assert engine.write_obsidian_note("t", "b") is None


def test_engine_syncs_and_recalls_vault_notes(engine, tmp_path):
    root = tmp_path / "v"
    root.mkdir()
    (root / "Entropy.md").write_text("Entropy measures disorder.", encoding="utf-8")
    engine.configure_extensions({"obsidian": {"enabled": True, "vault_path": str(root)}})
    added = []
    engine._vector_add_chunks = added.extend
    r = engine.handle("obsidian_sync", {}, {})
    assert "1 new" in r["response"] and added[0]["metadata"]["title"] == "Entropy"
    assert "already up to date" in engine.handle("obsidian_sync", {}, {})["response"]


def test_writeback_requires_its_own_opt_in(engine, tmp_path):
    root = tmp_path / "v"
    root.mkdir()
    engine.configure_extensions({"obsidian": {"enabled": True, "vault_path": str(root)}})
    assert engine.write_obsidian_note("t", "b") is None
    engine.configure_extensions({"obsidian": {"enabled": True, "vault_path": str(root), "writeback": True}})
    assert engine.write_obsidian_note("t", "b").endswith("t.md")


# ===========================================================================
# #47 Papers
# ===========================================================================

_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2609.01234v2</id>
    <published>2026-09-27T09:00:00Z</published>
    <title>Attention   Is
   Still All You Need</title>
    <summary>  We study transformers. They work on time series.  Results are strong.  </summary>
    <author><name>A. Author</name></author><author><name>B. Author</name></author>
    <link href="http://arxiv.org/abs/2609.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2609.01234v2" rel="related" type="application/pdf"/>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/hep-th/9901001v1</id>
    <published>2020-01-01T00:00:00Z</published>
    <title>Old paper</title><summary>Old abstract.</summary>
  </entry>
  <entry><id>http://arxiv.org/api/errors#x</id><title>Error</title><summary>bad</summary></entry>
  <entry><title>No id</title></entry>
</feed>"""

_NOW = datetime(2026, 9, 30, tzinfo=timezone.utc)


def test_feed_parsing_normalises_ids_and_whitespace_and_skips_bad_entries():
    papers = parse_arxiv_feed(_FEED)
    assert [p.arxiv_id for p in papers] == ["2609.01234", "hep-th/9901001"]
    p = papers[0]
    assert p.title == "Attention Is Still All You Need"
    assert p.authors == ["A. Author", "B. Author"] and p.categories == ["cs.LG"]
    assert p.pdf_link.endswith("2609.01234v2") and p.link.endswith("abs/2609.01234v2")


def test_malformed_xml_yields_no_papers():
    assert parse_arxiv_feed("<<<not xml") == []


def test_normalise_arxiv_id():
    assert normalise_arxiv_id("http://arxiv.org/abs/2401.01234v3") == "2401.01234"
    assert normalise_arxiv_id("http://arxiv.org/abs/hep-th/9901001v1") == "hep-th/9901001"


def test_url_building_quotes_phrases_and_passes_field_syntax_through():
    assert "all%3A%22graph+neural+networks%22" in build_arxiv_url("graph neural networks")
    assert "search_query=all%3Atransformers" in build_arxiv_url("transformers")
    assert "search_query=cat%3Acs.LG" in build_arxiv_url("cat:cs.LG")
    assert "max_results=50" in build_arxiv_url("x", 999) and "max_results=1&" in build_arxiv_url("x", 0) + "&"
    assert "sortBy=submittedDate" in build_arxiv_url("x")


def test_fallback_digest_takes_leading_sentences():
    assert fallback_digest("One. Two. Three.") == "One. Two."


@pytest.fixture
def monitor(db, tmp_path):
    calls = []
    m = PaperMonitor(db, str(tmp_path / "docs" / "arxiv"), llm=None,
                     fetch_fn=lambda url: (calls.append(url), _FEED)[1], sleep_fn=lambda s: None)
    m.calls = calls
    return m


def test_interests_are_deduplicated_case_insensitively(monitor):
    assert monitor.add_interest("Transformers") is True
    assert monitor.add_interest("transformers") is False
    assert monitor.remove_interest("TRANSFORMERS") is True


def test_check_writes_recent_unseen_papers_for_athena_and_skips_old_ones(monitor, tmp_path):
    monitor.add_interest("transformers")
    result = monitor.check(now=_NOW)
    assert [p.arxiv_id for p in result["new"]] == ["2609.01234"]     # the 2020 paper is outside the window
    f = tmp_path / "docs" / "arxiv" / "arxiv_2609.01234.md"
    text = f.read_text(encoding="utf-8")
    assert "# Attention Is Still All You Need" in text and "Matched interest: transformers" in text
    assert "## Abstract" in text


def test_second_check_finds_nothing_new(monitor):
    monitor.add_interest("transformers")
    monitor.check(now=_NOW)
    assert monitor.check(now=_NOW)["new"] == []
    assert monitor.seen_count() == 1


def test_a_paper_matching_two_interests_is_taken_once(monitor):
    monitor.add_interest("transformers")
    monitor.add_interest("time series")
    assert len(monitor.check(now=_NOW)["new"]) == 1


def test_requests_are_spaced_out_between_interests(db, tmp_path):
    sleeps = []
    m = PaperMonitor(db, str(tmp_path / "d"), fetch_fn=lambda u: _FEED, sleep_fn=sleeps.append)
    for q in ("a", "b", "c"):
        m.add_interest(q)
    m.check(now=_NOW)
    assert sleeps == [3.0, 3.0]


def test_network_failure_on_one_interest_does_not_stop_the_others(db, tmp_path):
    def fetch(url):
        if "bad" in url:
            raise ConnectionError("down")
        return _FEED

    m = PaperMonitor(db, str(tmp_path / "d"), fetch_fn=fetch, sleep_fn=lambda s: None)
    m.add_interest("bad topic")
    m.add_interest("good")
    r = m.check(now=_NOW)
    assert r["errors"] == 1 and len(r["new"]) == 1


def test_paper_is_retried_if_its_file_could_not_be_written(db, tmp_path, monkeypatch):
    m = PaperMonitor(db, str(tmp_path / "d"), fetch_fn=lambda u: _FEED, sleep_fn=lambda s: None)
    m.add_interest("transformers")
    real = Path.write_text
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    r = m.check(now=_NOW)
    assert r["errors"] == 1 and r["new"] == [] and m.seen_count() == 0
    monkeypatch.setattr(Path, "write_text", real)
    assert len(m.check(now=_NOW)["new"]) == 1


def test_llm_summary_is_used_and_falls_back_when_it_fails(db, tmp_path):
    class Good:
        def generate(self, prompt, fmt=None):
            return "A tidy two sentence summary. It works."

    class Bad:
        def generate(self, *a, **k):
            raise RuntimeError("ollama down")

    m = PaperMonitor(db, str(tmp_path / "g"), llm=Good(), fetch_fn=lambda u: _FEED, sleep_fn=lambda s: None)
    m.add_interest("x")
    assert m.check(now=_NOW)["new"][0].digest.startswith("A tidy")
    m2 = PaperMonitor(str(tmp_path / "other.db"), str(tmp_path / "b"), llm=Bad(), fetch_fn=lambda u: _FEED, sleep_fn=lambda s: None)
    m2.add_interest("x")
    assert m2.check(now=_NOW)["new"][0].digest == "We study transformers. They work on time series."


def test_max_new_per_query_is_respected(db, tmp_path):
    entries = "".join(
        f"<entry><id>http://arxiv.org/abs/2609.0{i:04d}v1</id><published>2026-09-28T00:00:00Z</published>"
        f"<title>P{i}</title><summary>S.</summary></entry>" for i in range(10))
    feed = f'<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'
    m = PaperMonitor(db, str(tmp_path / "d"), fetch_fn=lambda u: feed, sleep_fn=lambda s: None,
                     max_new_per_query=3)
    m.add_interest("x")
    assert len(m.check(now=_NOW)["new"]) == 3


def test_digest_markdown_lists_each_paper(monitor):
    monitor.add_interest("transformers")
    papers = monitor.check(now=_NOW)["new"]
    md = digest_markdown(papers, _NOW)
    assert "1 new paper(s)" in md and "arXiv:2609.01234" in md


# -- through the engine -----------------------------------------------------

def test_papers_are_off_by_default(engine):
    assert "off" in engine.handle("watch_papers", {}, {"raw_query": "any new papers"})["response"]


def _papers_engine(engine, tmp_path):
    engine.configure_extensions({"papers": {"enabled": True, "docs_dir": str(tmp_path / "docs")}})
    engine.paper_monitor.fetch_fn = lambda u: _FEED
    engine.paper_monitor.sleep_fn = lambda s: None
    return engine


def test_watch_list_and_remove_via_the_intent(engine, tmp_path):
    e = _papers_engine(engine, tmp_path)
    assert "Now watching arXiv for graph neural networks" in e.handle(
        "watch_papers", {}, {"raw_query": "watch arxiv for graph neural networks"})["response"]
    assert "Watching: graph neural networks" in e.handle(
        "watch_papers", {"action": "list"}, {})["response"]
    assert "Stopped watching" in e.handle(
        "watch_papers", {"action": "remove", "topic": "graph neural networks"}, {})["response"]


def test_check_via_intent_reports_new_papers_and_triggers_athena_ingest(engine, tmp_path):
    e = _papers_engine(engine, tmp_path)
    ingested = []

    class _Athena:
        def _ingest(self, data_dir=None):
            ingested.append(True)

    e.attach_athena(_Athena())
    e.paper_monitor.add_interest("transformers")
    e.paper_monitor.lookback_days = 100_000           # keep this test independent of today's date
    r = e.handle("watch_papers", {"action": "check"}, {})
    assert "2 new paper(s)" in r["response"] and "Attention Is Still All You Need" in r["response"]
    assert ingested == [True]
    assert "No new papers" in e.handle("watch_papers", {"action": "check"}, {})["response"]


def test_background_jobs_are_cadenced_and_isolated(engine, tmp_path):
    e = _papers_engine(engine, tmp_path)
    e.paper_monitor.add_interest("transformers")
    e.paper_monitor.lookback_days = 100_000
    first = e.run_background_jobs(now=1_000_000.0)
    assert "episodes" in first and first["papers"] == 2
    assert e.run_background_jobs(now=1_000_100.0) == {}                 # nothing is due yet
    e.cluster_episodes = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    later = e.run_background_jobs(now=1_000_000.0 + 3700)               # episodes job raises...
    assert "episodes" not in later                                      # ...and nothing else breaks
