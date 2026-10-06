# tests/test_ieee_source.py
"""Backlog #47, IEEE half: the IEEE Xplore source, the multi-source monitor,
the daily call ceiling, de-duplication across sources, and scripts/check_ieee.py.

Nothing here touches the network. The IEEE JSON below is hand-written from the
documented response shape, which is exactly why scripts/check_ieee.py exists:
it is the check against the real thing.
"""
import json
import logging
import os
import sys
import types
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.mnemosyne.ieee_source import (
    DEFAULT_DAILY_LIMIT, IeeeSource, MAX_DAILY_LIMIT, parse_ieee_date, parse_ieee_response,
)
from modules.mnemosyne.paper_monitor import (
    ArxivSource, Paper, PaperMonitor, digest_markdown, normalise_title, paper_markdown,
)

KEY = "SECRETKEY123456"
NOW = datetime(2026, 10, 6, tzinfo=timezone.utc)
EMPTY_FEED = "<feed xmlns='http://www.w3.org/2005/Atom'/>"

ARTICLE = {
    "doi": "10.1109/TEST.2026.1234567",
    "title": "Graph Neural Networks for Power Grid Fault Localisation",
    "abstract": "We localise faults. It works well. Results are strong.",
    "article_number": "1234567",
    "authors": {"authors": [{"full_name": "A. Author", "author_order": 1},
                            {"full_name": "B. Author", "author_order": 2}]},
    "html_url": "https://ieeexplore.ieee.org/document/1234567/",
    "pdf_url": "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1234567",
    "publication_title": "IEEE Transactions on Power Systems",
    "publication_year": 2026,
    "publication_date": "October 2026",
    "index_terms": {"ieee_terms": {"terms": ["Graph neural networks", "Fault location"]},
                    "author_terms": {"terms": ["power grid"]}},
}


def _resp(*articles):
    return json.dumps({"total_records": len(articles), "total_searched": 999, "articles": list(articles)})


def _art(**over):
    a = dict(ARTICLE)
    a.update(over)
    return a


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2610.00001v1</id>
    <published>2026-10-03T09:00:00Z</published>
    <title>Graph Neural Networks for Power Grid Fault Localisation</title>
    <summary>Preprint version of the same work.</summary>
    <author><name>A. Author</name></author>
    <link href="http://arxiv.org/abs/2610.00001v1" rel="alternate" type="text/html"/>
    <arxiv:doi>10.1109/TEST.2026.1234567</arxiv:doi>
  </entry>
</feed>"""


class _Router:
    """fetch_fn: arXiv URLs get the feed, IEEE URLs get the JSON, both counted."""

    def __init__(self, ieee_body=None, arxiv_body=ARXIV_FEED):
        self.ieee_body, self.arxiv_body = ieee_body, arxiv_body
        self.urls = []
        self.raise_for_ieee = None

    def __call__(self, url):
        self.urls.append(url)
        if "ieeexploreapi" in url:
            if self.raise_for_ieee:
                raise self.raise_for_ieee
            return self.ieee_body if self.ieee_body is not None else _resp()
        return self.arxiv_body

    def count(self, which):
        return sum(1 for u in self.urls if which in u)


def _monitor(tmp_path, router, sources=None, **kw):
    sleeps = []
    m = PaperMonitor(str(tmp_path / "p.db"), str(tmp_path / "docs"), fetch_fn=router,
                     sleep_fn=sleeps.append, sources=sources or [ArxivSource(), IeeeSource(KEY, **kw)])
    m.sleeps = sleeps
    return m


class _HTTPError(Exception):
    def __init__(self, status, msg="boom"):
        super().__init__(msg)
        self.response = types.SimpleNamespace(status_code=status)


# ----------------------------------------------------------------- dates

def test_date_parsing_precisions():
    assert parse_ieee_date("12 March 2024") == ("2024-03-12", "day")
    assert parse_ieee_date("March 2024") == ("2024-03-01", "month")
    assert parse_ieee_date("Mar.-Apr. 2024") == ("2024-04-01", "month")      # last month of a range
    assert parse_ieee_date("Sept. 2022") == ("2022-09-01", "month")
    assert parse_ieee_date("2024") == ("2024-12-31", "year")
    assert parse_ieee_date("", 2023) == ("2023-12-31", "year")
    assert parse_ieee_date("", None) == ("", "day")
    assert parse_ieee_date("Marketing 2020") == ("2020-12-31", "year")        # not March


def test_impossible_day_falls_back_to_month():
    assert parse_ieee_date("31 February 2024") == ("2024-02-01", "month")


# ----------------------------------------------------------------- parsing

def test_parse_a_full_article():
    p = parse_ieee_response(_resp(ARTICLE))[0]
    assert p.source == "ieee" and p.paper_id == "ieee:1234567" and p.arxiv_id == ""
    assert p.title.startswith("Graph Neural Networks") and p.authors == ["A. Author", "B. Author"]
    assert p.doi == "10.1109/test.2026.1234567" and p.venue == "IEEE Transactions on Power Systems"
    assert p.published == "2026-10-01" and p.date_precision == "month"
    assert p.link.endswith("1234567/") and "arnumber" in p.pdf_link
    assert p.categories == ["Graph neural networks", "Fault location", "power grid"]


def test_malformed_and_error_payloads_yield_nothing():
    assert parse_ieee_response("<<<not json") == []
    assert parse_ieee_response("[1,2]") == []
    assert parse_ieee_response(json.dumps({"error": "bad key"})) == []
    assert parse_ieee_response(json.dumps({"articles": "nope"})) == []
    assert parse_ieee_response(None) == []


def test_bad_articles_are_skipped_not_fatal():
    body = _resp(_art(article_number=""), _art(title=""), "junk", _art(article_number="9", title="Good paper title here"))
    assert [p.external_id for p in parse_ieee_response(body)] == ["9"]


def test_missing_optional_fields_use_safe_defaults():
    body = json.dumps({"articles": [{"article_number": "5", "title": "Minimal"}]})
    p = parse_ieee_response(body)[0]
    assert p.summary == "" and p.authors == [] and p.doi == "" and p.published == ""
    assert p.link == "https://ieeexplore.ieee.org/document/5"


def test_authors_may_be_a_bare_list():
    body = _resp(_art(authors=[{"full_name": "X Y"}, "Z W", {"nope": 1}]))
    assert parse_ieee_response(body)[0].authors == ["X Y", "Z W"]


# ----------------------------------------------------------------- source object

def test_constructor_needs_a_key_and_clamps_the_ceiling():
    with pytest.raises(ValueError):
        IeeeSource("  ")
    assert IeeeSource("k").daily_limit == DEFAULT_DAILY_LIMIT
    assert IeeeSource("k", daily_limit=10_000).daily_limit == MAX_DAILY_LIMIT == 200
    assert IeeeSource("k", daily_limit=0).daily_limit == 1


def test_url_building_maps_fields_and_clamps_records():
    s = IeeeSource(KEY)
    assert "querytext=graph+networks" in s.build_url("graph networks", 5)
    assert "article_title=graph" in s.build_url("ti:graph", 5)
    assert "abstract=pruning" in s.build_url("abs:pruning", 5)
    assert "author=Hinton" in s.build_url("au:Hinton", 5)
    assert "querytext=transformers" in s.build_url("all:transformers", 5)
    assert "max_records=25" in s.build_url("x", 999) and "max_records=1&" in s.build_url("x", 0)
    assert f"apikey={KEY}" in s.build_url("x", 5)


def test_arxiv_only_syntax_is_not_sent_to_ieee():
    s = IeeeSource(KEY)
    for q in ("cat:cs.LG", "co:something", "jr:Nature", "rn:123"):
        assert s.supports(q) is False
    assert s.supports("graph networks") and s.supports("au:Hinton")


def test_key_never_appears_in_repr_or_redacted_text():
    s = IeeeSource(KEY)
    assert KEY not in repr(s)
    assert KEY not in s.redact(f"HTTPError for url: https://x/?apikey={KEY}&format=json")


# ----------------------------------------------------------------- monitor: basics

def test_ieee_paper_is_written_for_athena_and_marked_seen(tmp_path):
    r = _Router(_resp(ARTICLE), arxiv_body=EMPTY_FEED)
    m = _monitor(tmp_path, r)
    m.add_interest("power grid")
    res = m.check(now=NOW)
    assert [p.paper_id for p in res["new"]] == ["ieee:1234567"]
    assert res["by_source"] == {"arxiv": 0, "ieee": 1} and res["errors"] == 0
    f = tmp_path / "docs" / "ieee_1234567.md"
    text = f.read_text(encoding="utf-8")
    assert text.startswith("# Graph Neural Networks") and "- IEEE: 1234567" in text and "- DOI: 10.1109" in text
    assert "- Venue: IEEE Transactions on Power Systems" in text and "## Abstract" in text
    assert m.recent_papers()[0]["source"] == "ieee"
    assert m.check(now=NOW)["new"] == []                      # seen: not queued twice


def test_digest_markdown_labels_each_source():
    a = Paper(arxiv_id="2610.1", title="A", summary="", link="http://a", matched_query="q")
    i = Paper(arxiv_id="", title="B", summary="", link="http://b", matched_query="q",
              source="ieee", external_id="77")
    md = digest_markdown([a, i], NOW)
    assert "[arXiv:2610.1](http://a)" in md and "[IEEE:77](http://b)" in md


def test_arxiv_markdown_is_unchanged_for_athena_bibliography():
    md = paper_markdown(Paper(arxiv_id="2610.00001", title="T", summary="S", published="2026-10-03"))
    assert "- arXiv: 2610.00001" in md and "DOI" not in md


def test_default_monitor_is_arxiv_only(tmp_path):
    m = PaperMonitor(str(tmp_path / "p.db"), str(tmp_path / "d"))
    assert [s.name for s in m.sources] == ["arxiv"]


# ----------------------------------------------------------------- cross-source dedup

def test_same_work_from_both_sources_is_queued_once_by_doi(tmp_path):
    r = _Router(_resp(ARTICLE))
    m = _monitor(tmp_path, r)
    m.add_interest("power grid")
    res = m.check(now=NOW)
    assert len(res["new"]) == 1 and res["new"][0].source == "arxiv"          # arXiv ran first
    assert not (tmp_path / "docs" / "ieee_1234567.md").exists()
    assert m.check(now=NOW)["new"] == []


def test_dedup_by_title_when_there_is_no_doi(tmp_path):
    feed_no_doi = ARXIV_FEED.replace("<arxiv:doi>10.1109/TEST.2026.1234567</arxiv:doi>", "")
    r = _Router(_resp(_art(doi="")), arxiv_body=feed_no_doi)
    m = _monitor(tmp_path, r)
    m.add_interest("power grid")
    assert len(m.check(now=NOW)["new"]) == 1


def test_short_titles_are_not_merged(tmp_path):
    feed = ARXIV_FEED.replace("Graph Neural Networks for Power Grid Fault Localisation", "Survey").replace(
        "<arxiv:doi>10.1109/TEST.2026.1234567</arxiv:doi>", "")
    r = _Router(_resp(_art(title="Survey", doi="")), arxiv_body=feed)
    m = _monitor(tmp_path, r)
    m.add_interest("power grid")
    assert len(m.check(now=NOW)["new"]) == 2


def test_dedup_works_against_papers_stored_in_an_earlier_run(tmp_path):
    r = _Router(_resp(), arxiv_body=ARXIV_FEED)
    m = _monitor(tmp_path, r)
    m.add_interest("power grid")
    assert len(m.check(now=NOW)["new"]) == 1                 # arXiv copy stored
    r.ieee_body = _resp(ARTICLE)
    assert m.check(now=NOW)["new"] == []                     # IEEE copy recognised next time


# ----------------------------------------------------------------- recency

def test_month_precision_keeps_a_paper_from_this_month(tmp_path):
    late = datetime(2026, 10, 28, tzinfo=timezone.utc)       # the 1st is 27 days old, past the 14-day cutoff
    r = _Router(_resp(ARTICLE), arxiv_body=EMPTY_FEED)
    m = _monitor(tmp_path, r)
    m.add_interest("power grid")
    assert len(m.check(now=late)["new"]) == 1


def test_old_month_and_old_year_are_dropped(tmp_path):
    old = _resp(_art(article_number="1", publication_date="March 2020", publication_year=2020, doi="", title="Old paper one title"),
                _art(article_number="2", publication_date="", publication_year=2019, doi="", title="Older paper two title"))
    r = _Router(old, arxiv_body=EMPTY_FEED)
    m = _monitor(tmp_path, r)
    m.add_interest("power grid")
    assert m.check(now=NOW)["new"] == []


# ----------------------------------------------------------------- daily ceiling

def test_each_ieee_call_is_counted_arxiv_is_not(tmp_path):
    r = _Router(_resp())
    m = _monitor(tmp_path, r)
    for q in ("a", "b", "c"):
        m.add_interest(q)
    m.check(now=NOW)
    assert m.usage_today("ieee", NOW) == 3 and m.usage_today("arxiv", NOW) == 0


def test_check_stops_at_the_ceiling_and_reports_it(tmp_path):
    r = _Router(_resp())
    m = _monitor(tmp_path, r, daily_limit=2)
    for q in ("a", "b", "c", "d"):
        m.add_interest(q)
    res = m.check(now=NOW)
    assert r.count("ieeexploreapi") == 2 and res["quota_skipped"] == ["ieee"]
    assert r.count("arxiv.org") == 4                         # arXiv carries on


def test_the_ceiling_persists_across_checks_and_restarts(tmp_path):
    r = _Router(_resp())
    m = _monitor(tmp_path, r, daily_limit=2)
    m.add_interest("a"); m.add_interest("b")
    m.check(now=NOW)
    again = _monitor(tmp_path, r, daily_limit=2)             # a fresh object on the same database
    again.check(now=NOW)
    assert r.count("ieeexploreapi") == 2


def test_the_counter_resets_on_a_new_day(tmp_path):
    r = _Router(_resp())
    m = _monitor(tmp_path, r, daily_limit=1)
    m.add_interest("a")
    m.check(now=NOW)
    m.check(now=NOW)
    assert r.count("ieeexploreapi") == 1
    m.check(now=datetime(2026, 10, 7, tzinfo=timezone.utc))
    assert r.count("ieeexploreapi") == 2


def test_arxiv_only_interests_do_not_spend_ieee_calls(tmp_path):
    r = _Router(_resp())
    m = _monitor(tmp_path, r)
    m.add_interest("cat:cs.LG")
    m.check(now=NOW)
    assert r.count("ieeexploreapi") == 0 and m.usage_today("ieee", NOW) == 0


def test_http_429_marks_the_quota_used_for_the_day(tmp_path):
    r = _Router(_resp())
    r.raise_for_ieee = _HTTPError(429)
    m = _monitor(tmp_path, r, daily_limit=50)
    m.add_interest("a"); m.add_interest("b")
    res = m.check(now=NOW)
    assert r.count("ieeexploreapi") == 1 and res["quota_skipped"] == ["ieee"] and res["errors"] == 1
    assert m.usage_today("ieee", NOW) == 50
    r.raise_for_ieee = None
    m.check(now=NOW)
    assert r.count("ieeexploreapi") == 1                     # still stopped later the same day


def test_http_401_skips_ieee_for_the_rest_of_the_check_only(tmp_path):
    r = _Router(_resp())
    r.raise_for_ieee = _HTTPError(401)
    m = _monitor(tmp_path, r)
    m.add_interest("a"); m.add_interest("b")
    res = m.check(now=NOW)
    assert r.count("ieeexploreapi") == 1 and res["quota_skipped"] == []


def test_an_ieee_failure_does_not_stop_arxiv(tmp_path):
    r = _Router(_resp(), arxiv_body=ARXIV_FEED)
    r.raise_for_ieee = ConnectionError("down")
    m = _monitor(tmp_path, r)
    m.add_interest("power grid")
    res = m.check(now=NOW)
    assert res["errors"] == 1 and [p.source for p in res["new"]] == ["arxiv"]


# ----------------------------------------------------------------- secrets and pacing

class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage() + (" " + str(record.exc_info[1]) if record.exc_info else ""))


def test_the_key_never_reaches_the_logs_when_a_request_fails(tmp_path):
    r = _Router(_resp())
    r.raise_for_ieee = _HTTPError(500, f"500 Server Error for url: https://x/?apikey={KEY}&q=1")
    m = _monitor(tmp_path, r)
    m.add_interest("a")
    cap = _Capture()
    logging.getLogger().addHandler(cap)
    logging.getLogger().setLevel(logging.DEBUG)
    try:
        m.check(now=NOW)
    finally:
        logging.getLogger().removeHandler(cap)
    assert cap.lines and not any(KEY in line for line in cap.lines)


def test_ieee_calls_are_paced_independently_of_arxiv(tmp_path):
    r = _Router(_resp())
    m = _monitor(tmp_path, r)
    for q in ("a", "b", "c"):
        m.add_interest(q)
    m.check(now=NOW)
    assert sorted(m.sleeps) == [0.25, 0.25, 3.0, 3.0]


# ----------------------------------------------------------------- migration

_OLD_SCHEMA = """
    CREATE TABLE paper_interests (id INTEGER PRIMARY KEY, query TEXT NOT NULL UNIQUE COLLATE NOCASE,
                                  created_at TEXT, last_checked TEXT);
    CREATE TABLE papers_seen (arxiv_id TEXT PRIMARY KEY, title TEXT, query TEXT, published TEXT,
                              first_seen TEXT, digest TEXT);
"""


def test_an_existing_database_from_before_this_change_is_migrated(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(_OLD_SCHEMA + """
        INSERT INTO papers_seen VALUES ('2401.00001', 'An Older Paper About Graph Learning', 'q', '2024-01-01', 't', 'd');
    """)
    conn.commit(); conn.close()
    m = PaperMonitor(str(db), str(tmp_path / "d"))
    assert m._is_seen("2401.00001")
    row = m._conn.execute("SELECT source, norm_title FROM papers_seen").fetchone()
    assert row["source"] == "arxiv" and row["norm_title"] == "an older paper about graph learning"
    PaperMonitor(str(db), str(tmp_path / "d"))               # opening again is harmless


def test_migrated_titles_take_part_in_dedup(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(_OLD_SCHEMA)
    conn.execute("INSERT INTO papers_seen VALUES ('2401.1', ?, 'q', '2026-10-01', 't', 'd')",
                 ("Graph Neural Networks for Power Grid Fault Localisation",))
    conn.commit(); conn.close()
    m = PaperMonitor(str(db), str(tmp_path / "d"), fetch_fn=_Router(_resp(ARTICLE)), sleep_fn=lambda s: None,
                     sources=[IeeeSource(KEY)])
    m.add_interest("power grid")
    assert m.check(now=NOW)["new"] == []


def test_normalise_title():
    assert normalise_title("  Graph-Neural   Networks: A Survey! ") == "graph neural networks a survey"


# ----------------------------------------------------------------- extension wiring

def _paper_sources():
    from modules.mnemosyne.extensions import MnemosyneExtensionsMixin
    return MnemosyneExtensionsMixin._paper_sources


def test_sources_default_to_arxiv():
    out = _paper_sources()({}, ArxivSource)
    assert [s.name for s in out] == ["arxiv"]


def test_ieee_without_a_key_is_skipped_and_arxiv_survives(monkeypatch):
    monkeypatch.setenv("IEEE_API_KEY", "")
    out = _paper_sources()({"sources": ["arxiv", "ieee"]}, ArxivSource)
    assert [s.name for s in out] == ["arxiv"]


def test_ieee_with_a_key_is_added_with_the_configured_ceiling(monkeypatch):
    monkeypatch.setenv("IEEE_API_KEY", KEY)
    out = _paper_sources()({"sources": ["arxiv", "ieee"], "ieee": {"daily_limit": 40}}, ArxivSource)
    assert [s.name for s in out] == ["arxiv", "ieee"] and out[1].daily_limit == 40


def test_unknown_source_is_ignored_and_empty_falls_back_to_arxiv():
    assert [s.name for s in _paper_sources()({"sources": ["bogus"]}, ArxivSource)] == ["arxiv"]
    assert [s.name for s in _paper_sources()({"sources": "arxiv"}, ArxivSource)] == ["arxiv"]


# ----------------------------------------------------------------- scripts/check_ieee.py

def _run_check(argv=None, fetch=None, key=KEY, monkeypatch=None):
    from scripts import check_ieee
    monkeypatch.setenv("IEEE_API_KEY", key)
    lines = []
    code = check_ieee.main(argv or [], fetch=fetch, out=lines.append)
    return code, "\n".join(lines)


def test_check_script_needs_a_key(monkeypatch):
    code, out = _run_check(key="", monkeypatch=monkeypatch)
    assert code == 2 and "IEEE_API_KEY is not set" in out


def test_check_script_success_reports_fields_and_papers(monkeypatch):
    code, out = _run_check(fetch=lambda u: (200, _resp(ARTICLE)), monkeypatch=monkeypatch)
    assert code == 0 and "Parsed 1 paper(s)" in out and "parser matches the live format" in out
    assert KEY not in out


def test_check_script_flags_missing_fields(monkeypatch):
    art = {"article_number": "1", "title": "Only these two fields"}
    code, out = _run_check(fetch=lambda u: (200, json.dumps({"articles": [art]})), monkeypatch=monkeypatch)
    assert code == 0 and "MISSING abstract" in out and "send me the saved fixture" in out


def test_check_script_auth_and_quota_failures_exit_nonzero(monkeypatch):
    for status in (401, 403, 429, 500):
        code, out = _run_check(fetch=lambda u, s=status: (s, "denied " + KEY), monkeypatch=monkeypatch)
        assert code == 1 and KEY not in out


def test_check_script_handles_non_json_and_wrong_shape(monkeypatch):
    assert _run_check(fetch=lambda u: (200, "<html>"), monkeypatch=monkeypatch)[0] == 1
    code, out = _run_check(fetch=lambda u: (200, json.dumps({"hello": 1})), monkeypatch=monkeypatch)
    assert code == 1 and "hello" in out


def test_check_script_network_error_is_reported_without_the_key(monkeypatch):
    def boom(url):
        raise ConnectionError(f"failed for {url}")
    code, out = _run_check(fetch=boom, monkeypatch=monkeypatch)
    assert code == 1 and KEY not in out and "Request failed" in out


def test_check_script_saves_the_raw_fixture(tmp_path, monkeypatch):
    path = tmp_path / "fx.json"
    body = _resp(ARTICLE)
    code, out = _run_check(["--save-fixture", str(path)], fetch=lambda u: (200, body), monkeypatch=monkeypatch)
    assert code == 0 and path.read_text(encoding="utf-8") == body and KEY not in path.read_text(encoding="utf-8")
