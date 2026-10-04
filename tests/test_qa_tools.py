# tests/test_qa_tools.py
"""
Tests for the QA tooling itself (backlog #207, #211, #212, #214, #215):
scripts/coverage_history.py, mutation_check.py, smoke_test.py,
voice_latency.py and replay_queries.py. A quality gate nobody checks can fail
silently, so each one's logic is pinned here with no model, no real coverage
run and no booted application.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts import coverage_history as ch
from scripts import mutation_check as mc
from scripts import replay_queries as rq
from scripts import smoke_test as st
from scripts import voice_latency as vl


# ======================================================================= #207
class TestCoverageHistory:
    def test_summary_totals_and_worst_files(self):
        s = ch.summarise({"a.py": (100, 50), "b.py": (100, 0), "empty.py": (0, 0)})
        assert (s.statements, s.covered, s.files, s.percent) == (200, 150, 2, 75.0)
        assert s.worst[0] == ("a.py", 50.0, 50)

    def test_missing_cannot_exceed_statements(self):
        assert ch.summarise({"a.py": (10, 99)}).percent == 0.0

    def test_no_files_is_not_a_division_error(self):
        assert ch.summarise({}).percent == 100.0

    def test_history_round_trip_and_header_written_once(self, tmp_path):
        path = str(tmp_path / "sub" / "h.csv")
        s = ch.summarise({"a.py": (10, 1)})
        when = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
        ch.append_history(path, s, commit="abc", when=when)
        ch.append_history(path, ch.summarise({"a.py": (10, 0)}), tests_ok=False)
        rows = ch.read_history(path)
        assert [r["percent"] for r in rows] == ["90.00", "100.00"]
        assert rows[0]["date"] == "2026-01-02 03:04" and rows[0]["commit"] == "abc"
        assert rows[1]["tests_ok"] == "0"
        assert open(path).read().count("date,commit") == 1

    def test_unreadable_history_is_empty(self, tmp_path):
        assert ch.read_history(str(tmp_path / "nope.csv")) == []

    def test_previous_percent_skips_garbage(self):
        assert ch.previous_percent([{"percent": "80.0"}, {"percent": "x"}]) == 80.0
        assert ch.previous_percent([]) is None

    @pytest.mark.parametrize("prev,now,expect", [
        (None, 50.0, "First recorded"),
        (80.0, 80.0, "Unchanged"),
        (80.0, 82.5, "Up 2.50 points"),
        (80.0, 79.5, "Down 0.50 points"),
    ])
    def test_describe_change(self, prev, now, expect):
        assert expect in ch.describe_change(prev, now)

    def test_a_big_drop_is_called_out_and_a_small_one_is_not(self):
        assert "check what lost coverage" in ch.describe_change(80.0, 78.0)
        assert "check what lost coverage" not in ch.describe_change(80.0, 79.5)

    def test_split_args(self):
        opts, rest = ch.split_cov_args(["-x", "--coverage", "tests/a.py", "--no-record"])
        assert opts == {"coverage": True, "record": False, "fail_under": None}
        assert rest == ["-x", "tests/a.py"]

    def test_fail_under_implies_coverage_in_both_spellings(self):
        for argv in (["--cov-fail-under", "70"], ["--cov-fail-under=70"]):
            opts, rest = ch.split_cov_args(argv)
            assert opts["coverage"] and opts["fail_under"] == 70.0 and rest == []

    def test_no_flags_means_no_coverage(self):
        opts, rest = ch.split_cov_args(["-k", "x"])
        assert not opts["coverage"] and rest == ["-k", "x"]

    def test_source_scan_finds_unimported_files_and_skips_noise(self, tmp_path):
        for rel in ("core/a.py", "modules/m/b.py", "modules/m/__pycache__/c.py",
                    "modules/_stubs/d.py", "main.py", "other/e.py"):
            f = tmp_path / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x = 1\n")
        found = {os.path.relpath(p, tmp_path).replace(os.sep, "/") for p in ch.iter_source_files(str(tmp_path))}
        assert found == {"core/a.py", "modules/m/b.py", "main.py"}

    def _fake_coverage(self, per_file):
        class Data:
            def measured_files(self_inner): return list(per_file)

        class Cov:
            def __init__(self_inner, **kw): self_inner.kw = kw
            def start(self_inner): pass
            def stop(self_inner): pass
            def save(self_inner): pass
            def get_data(self_inner): return Data()
            def analysis2(self_inner, f):
                s, m = per_file[f]
                return f, list(range(s)), [], list(range(m)), ""

        class Mod:
            Coverage = Cov
        return Mod

    def test_driver_reports_records_and_enforces_the_floor(self, tmp_path):
        root = tmp_path
        (root / "core").mkdir()
        f = str(root / "core" / "a.py")
        Path(f).write_text("x=1\n")
        out: list[str] = []
        hist = str(root / "h.csv")
        code = ch.run_with_coverage(
            ["-q"], pytest_main=lambda a: 0, coverage_module=self._fake_coverage({f: (10, 5)}),
            root=str(root), history_path=hist, fail_under=60.0, printer=out.append)
        text = "\n".join(out)
        assert code == 2 and "50.00%" in text and "below the required 60%" in text
        assert len(ch.read_history(hist)) == 1

    def test_driver_keeps_pytests_failure_code_and_honours_no_record(self, tmp_path):
        out: list[str] = []
        code = ch.run_with_coverage(
            [], pytest_main=lambda a: 1, coverage_module=self._fake_coverage({}),
            root=str(tmp_path), history_path=str(tmp_path / "h.csv"), record=False, printer=out.append)
        assert code == 1 and not (tmp_path / "h.csv").exists()


# ======================================================================= #214
SAMPLE = '''
"""Doc."""
import logging
logger = logging.getLogger(__name__)

def f(x, y):
    """Docstring with 3 numbers 1 2."""
    logger.info("count %d", 5)
    if x > 10 and y:
        return x + 1
    return not y   # pragma: no mutate
'''


class TestMutationCheck:
    def test_lists_expected_mutants_and_skips_docstrings_logs_and_pragmas(self):
        desc = [m.description for m in mc.list_mutants(SAMPLE)]
        assert "Gt -> LtE" in desc and "And -> Or" in desc and "Add -> Sub" in desc
        assert "negated `if` condition" in desc and "10 -> 11" in desc
        assert not any("removed `not`" in d for d in desc)           # pragma line
        assert all(m.lineno not in (2, 7, 8) for m in mc.list_mutants(SAMPLE))

    def test_each_mutant_changes_exactly_the_described_thing(self):
        for m in mc.list_mutants(SAMPLE):
            out = mc.apply_mutant(SAMPLE, m.index)
            assert out != mc.ast.unparse(mc.ast.parse(SAMPLE))
            compile(out, "<t>", "exec")

    def test_out_of_range_index_raises(self):
        with pytest.raises(IndexError):
            mc.apply_mutant(SAMPLE, 999)

    def _target(self, tmp_path, body):
        p = tmp_path / "t.py"
        p.write_text(body, encoding="utf-8", newline="")
        return str(p)

    def test_a_test_that_pins_the_behaviour_kills_the_mutant(self, tmp_path):
        path = self._target(tmp_path, "def f(x):\n    return x > 10\n")

        def runner():
            ns = {}
            exec(open(path).read(), ns)
            return ns["f"](11) is True and ns["f"](10) is False
        r = mc.run_mutants(path, runner)
        assert r.baseline_ok and r.killed == r.tested and r.survivors == []

    def test_a_test_that_pins_nothing_leaves_survivors(self, tmp_path):
        path = self._target(tmp_path, "def f(x):\n    return x > 10\n")
        r = mc.run_mutants(path, lambda: True)
        assert r.survivors and r.killed == 0 and r.score == 0.0
        assert "Survivors" in mc.format_result(path, r)

    def test_failing_baseline_is_reported_not_scored(self, tmp_path):
        path = self._target(tmp_path, "def f(x):\n    return x\n")
        r = mc.run_mutants(path, lambda: False)
        assert not r.baseline_ok and "already fail" in mc.format_result(path, r)

    def test_original_is_restored_byte_for_byte_including_crlf(self, tmp_path):
        body = "def f(x):\r\n    return x > 10\r\n"
        path = self._target(tmp_path, body)
        mc.run_mutants(path, lambda: True)
        assert open(path, "rb").read() == body.encode()
        assert not os.path.exists(path + mc.BACKUP_SUFFIX)

    def test_original_is_restored_even_when_the_runner_explodes(self, tmp_path):
        body = "def f(x):\n    return x > 10\n"
        path = self._target(tmp_path, body)
        calls = {"n": 0}

        def runner():
            calls["n"] += 1
            if calls["n"] > 1:
                raise KeyboardInterrupt
            return True
        with pytest.raises(KeyboardInterrupt):
            mc.run_mutants(path, runner)
        assert open(path).read() == body and not os.path.exists(path + mc.BACKUP_SUFFIX)

    def test_leftover_backup_from_a_killed_run_is_restored_first(self, tmp_path):
        path = self._target(tmp_path, "BROKEN MUTANT")
        Path(path + mc.BACKUP_SUFFIX).write_text("def f():\n    return 1\n")
        assert mc.recover_backup(path) is True
        assert open(path).read() == "def f():\n    return 1\n"
        assert mc.recover_backup(path) is False

    def test_max_samples_deterministically(self, tmp_path):
        path = self._target(tmp_path, SAMPLE)
        a = mc.run_mutants(path, lambda: True, max_mutants=3, seed=1)
        b = mc.run_mutants(path, lambda: True, max_mutants=3, seed=1)
        assert a.tested == 3 and [m.index for m, _ in a.survivors] == [m.index for m, _ in b.survivors]

    def test_stale_bytecode_cannot_make_a_mutant_look_unchanged(self, tmp_path):
        """Regression: without dropping the .pyc, same-size same-second mutants
        reran the previous version (1.0 -> 2.0 is the same file size)."""
        path = self._target(tmp_path, "X = 1.0\n")
        results = []

        def runner():
            import importlib
            import importlib.util
            spec = importlib.util.spec_from_file_location("mut_probe_mod", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            results.append(mod.X)
            return mod.X == 1.0
        r = mc.run_mutants(path, runner)
        assert r.killed >= 1 and 2.0 in results

    def test_cli_list_does_not_run_tests(self, tmp_path, capsys):
        path = self._target(tmp_path, "def f(x):\n    return x > 1\n")
        assert mc.main([path, "--list"]) == 0
        assert "Gt -> LtE" in capsys.readouterr().out

    def test_cli_missing_file(self, capsys):
        assert mc.main(["/no/such/file.py"]) == 2


# ======================================================================= #212
class TestSmokeTest:
    @pytest.mark.parametrize("reply", [None, "", "   ", "Sorry, something went wrong.",
                                       "My backend isn't responding right now.",
                                       "Traceback (most recent call last):\n  File", "KeyError: 'x'"])
    def test_error_looking_replies_are_failures(self, reply):
        assert st.looks_like_error(reply)

    @pytest.mark.parametrize("reply", ["It is 3:45 PM.", "Hello! How can I help?", "144"])
    def test_ordinary_replies_pass(self, reply):
        assert st.looks_like_error(reply) is None

    def test_ten_distinct_read_only_canonical_queries(self):
        assert len(st.CANONICAL_QUERIES) == 10 == len(set(st.CANONICAL_QUERIES))
        verbs = ("log ", "save ", "send ", "delete ", "remind me to", "add ", "set ")
        assert not [q for q in st.CANONICAL_QUERIES if q.lower().startswith(verbs)]

    def test_slow_reply_is_a_failure(self):
        r = st.evaluate("q", "fine", seconds=12.0, max_seconds=10.0)
        assert not r.ok and "over the 10s limit" in r.problem

    def test_run_queries_survives_an_exception_and_keeps_going(self):
        def ask(q):
            if q == "bad":
                raise RuntimeError("boom")
            return "fine"
        res = st.run_queries(ask, ["good", "bad", "good2"])
        assert [r.ok for r in res] == [True, False, True]
        assert "RuntimeError: boom" in res[1].problem

    def test_exit_code_requires_every_query_to_pass_and_at_least_one_to_run(self):
        ok = st.run_queries(lambda q: "fine", ["a", "b"])
        assert st.exit_code(ok) == 0
        assert st.exit_code(st.run_queries(lambda q: "", ["a"])) == 1
        assert st.exit_code([]) == 1

    def test_report_says_do_not_deploy_on_failure(self):
        res = st.run_queries(lambda q: "", ["a"])
        assert "Do not deploy" in st.format_report(res)
        assert "Do not deploy" not in st.format_report(st.run_queries(lambda q: "ok", ["a"]))

    @pytest.mark.parametrize("decision,bad", [
        ({"primary": "apollo", "primary_can_handle": True}, False),
        ({"primary": "apollo"}, False),
        ({"primary": "apollo", "primary_can_handle": False, "dispatch_intent": "x"}, True),
        ({"primary": ""}, True), ({"error": "NLU failed"}, True), (None, True),
    ])
    def test_routing_problem(self, decision, bad):
        assert bool(st.routing_problem(decision)) is bad

    def test_routing_only_never_calls_a_handler(self):
        res = st.run_routing_only(lambda q: {"primary": "core", "intent": "chat",
                                             "primary_can_handle": True}, ["a", "b"])
        assert all(r.ok for r in res) and res[0].reply == "chat -> core"

    def test_query_file_ignores_comments_and_blanks(self, tmp_path):
        f = tmp_path / "q.txt"
        f.write_text("# header\nwhat time is it\n\n  hello  \n", encoding="utf-8")
        assert st.load_queries(str(f)) == ["what time is it", "hello"]


# ======================================================================= #211
class TestVoiceLatency:
    def test_percentile(self):
        assert vl.percentile([], 95) == 0.0 and vl.percentile([7], 95) == 7.0
        assert vl.percentile([1, 2, 3, 4], 50) == 2.5
        assert vl.percentile(list(range(1, 101)), 95) == pytest.approx(95.05)
        assert vl.percentile([5, 1, 3], 0) == 1 and vl.percentile([5, 1, 3], 100) == 5

    def test_time_stage_discards_the_warm_up_call(self):
        calls = []
        ticks = iter([0.0, 0.5, 1.0, 2.0])
        out = vl.time_stage(lambda: calls.append(1), 2, clock=lambda: next(ticks))
        assert len(calls) == 3 and out == [500.0, 1000.0]

    def test_over_budget_beats_regression(self):
        v = vl.judge(vl.StageStats("nlu", [4000.0] * 5), {"nlu": 3000}, {"nlu": 1000})
        assert v.status == "over budget"

    def test_regression_against_the_baseline_has_a_tolerance(self):
        within = vl.judge(vl.StageStats("nlu", [1200.0] * 5), {"nlu": 3000}, {"nlu": 1000}, 0.25)
        over = vl.judge(vl.StageStats("nlu", [1300.0] * 5), {"nlu": 3000}, {"nlu": 1000}, 0.25)
        assert within.status == "ok" and over.status == "regressed"

    def test_uses_p95_not_the_average(self):
        samples = [100.0] * 18 + [5000.0] * 2          # mean 590 ms, p95 5000 ms
        assert vl.judge(vl.StageStats("nlu", samples), {"nlu": 1000}, {}).status == "over budget"

    def test_a_skipped_stage_is_never_reported_as_passing(self):
        v = vl.judge(vl.StageStats("stt", skipped="no --wav given"), {"stt": 1}, {})
        assert v.status == "skipped" and "wav" in v.note

    def test_exit_codes(self):
        ok = vl.Verdict("a", 1, 2, None, "ok")
        skipped = vl.Verdict("b", None, 2, None, "skipped")
        assert vl.exit_code([ok, skipped]) == 0
        assert vl.exit_code([skipped]) == 2 and vl.exit_code([]) == 2
        assert vl.exit_code([ok, vl.Verdict("c", 9, 2, None, "over budget")]) == 1
        assert vl.exit_code([ok, vl.Verdict("c", 9, 2, 1, "regressed")]) == 1

    def test_number_files_tolerate_junk(self, tmp_path):
        p = tmp_path / "b.json"
        p.write_text(json.dumps({"nlu": 5, "tts": "x", "stt": -1, "round_trip": True, "extra": 2.5}))
        got = vl.load_json_numbers(p, {"nlu": 1, "tts": 9, "stt": 8, "round_trip": 7})
        assert got == {"nlu": 5.0, "tts": 9, "stt": 8, "round_trip": 7, "extra": 2.5}
        p.write_text("not json")
        assert vl.load_json_numbers(p, {"a": 1}) == {"a": 1}
        assert vl.load_json_numbers(tmp_path / "missing.json") == {}

    def test_write_json_is_atomic_and_creates_directories(self, tmp_path):
        p = tmp_path / "d" / "x.json"
        vl.write_json(p, {"b": 1, "a": 2})
        assert json.loads(p.read_text()) == {"a": 2, "b": 1} and not list(p.parent.glob("*.tmp"))

    def test_default_budget_covers_every_stage_and_the_round_trip(self):
        assert set(vl.DEFAULT_BUDGET_MS) == set(vl.STAGES) | {"round_trip"}

    def test_report_lists_every_stage(self):
        s = [vl.StageStats("nlu", [100.0, 200.0]), vl.StageStats("stt", skipped="none")]
        v = [vl.judge(x, {"nlu": 1000}, {}) for x in s]
        text = vl.format_report(s, v)
        assert "nlu" in text and "stt" in text and "skipped" in text


# ======================================================================= #215
class TestReplay:
    def test_anonymise_scrubs_identifiers_but_keeps_what_classification_needs(self):
        out = rq.anonymise("email priya@example.com about 2500 rupees, call +91 98765 43210 "
                           "see https://x.io/a?b=1 at 192.168.1.5 in C:\\Users\\Rohan\\docs", names=["Priya"])
        assert "<email>" in out and "<url>" in out and "<ip>" in out and "<number>" in out
        assert "C:\\Users\\<user>\\docs" in out and "2500 rupees" in out
        assert "priya@" not in out and "98765" not in out and "Rohan" not in out

    def test_names_match_whole_words_case_insensitively(self):
        assert rq.anonymise("tell ROHAN and Rohanda", names=["Rohan"]) == "tell <name> and Rohanda"

    def test_short_numbers_and_times_survive(self):
        assert rq.anonymise("remind me at 10:30 to pay 450") == "remind me at 10:30 to pay 450"

    def test_extract_filters_dedupes_and_never_keeps_shortcut_or_chat_rows(self):
        recs = [
            {"query": "log 70 kg", "intent": "apollo_log_weight", "confidence": 0.9, "source": "nlu"},
            {"query": "LOG  70 kg", "intent": "apollo_log_weight", "confidence": 0.9, "source": "nlu"},
            {"query": "what time", "intent": "get_time", "confidence": 0.99, "source": "alias"},
            {"query": "hi", "intent": "chat", "confidence": 0.9, "source": "nlu"},
            {"query": "unsure", "intent": "read_email", "confidence": 0.2, "source": "nlu"},
            {"query": "old", "intent": "no_such_intent", "confidence": 0.9, "source": "nlu"},
            {"query": "", "intent": "read_email"}, {"intent": "read_email"}, "junk",
            {"query": "mail Priya", "intent": "read_email", "confidence": "high"},
        ]
        rows = rq.extract_rows(recs, min_confidence=0.5, valid_intents={"apollo_log_weight", "read_email"})
        # the duplicate that is kept is the newest one
        assert [r.query for r in rows] == ["LOG  70 kg"] and rows[0].verified is False

    def test_extract_anonymises_and_keeps_the_newest_when_limited(self):
        recs = [{"query": f"mail Priya number {i}", "intent": "read_email", "confidence": 0.9} for i in range(5)]
        rows = rq.extract_rows(recs, names=["Priya"], limit=2)
        assert [r.query for r in rows] == ["mail <name> number 3", "mail <name> number 4"]

    def test_corpus_round_trip_and_bad_lines_are_skipped(self, tmp_path):
        p = tmp_path / "c.jsonl"
        rq.write_corpus(p, [rq.Row("héllo wörld", "chat", True), rq.Row("b", "read_email")])
        with open(p, "a", encoding="utf-8") as fh:
            fh.write("not json\n{\"query\": \"\", \"expected\": \"x\"}\n\n")
        assert rq.load_corpus(p) == [rq.Row("héllo wörld", "chat", True), rq.Row("b", "read_email", False)]

    def test_replay_turns_exceptions_into_outcomes(self):
        def classify(q):
            if q == "boom":
                raise ValueError("x")
            return "read_email"
        out = rq.replay([rq.Row("a", "read_email"), rq.Row("boom", "read_email")], classify)
        assert out[0].correct and out[1].got == "<error: ValueError>" and not out[1].correct

    def test_accuracy_and_verified_only(self):
        o = [rq.Outcome("a", "x", "x", True), rq.Outcome("b", "x", "y", False)]
        assert rq.accuracy(o) == 0.5 and rq.accuracy(o, verified_only=True) == 1.0
        assert rq.accuracy([], True) is None

    def test_compare_finds_regressions_fixes_changes_and_new(self):
        before = [rq.Outcome("reg", "x", "x", True), rq.Outcome("fix", "x", "y", True),
                  rq.Outcome("chg", "x", "y", False), rq.Outcome("same", "x", "x", True)]
        now = [rq.Outcome("reg", "x", "z", True), rq.Outcome("fix", "x", "x", True),
               rq.Outcome("chg", "x", "w", False), rq.Outcome("same", "x", "x", True),
               rq.Outcome("brand new", "x", "x", True)]
        c = rq.compare(now, before)
        assert [(o.query, was) for o, was in c.regressions] == [("reg", "x")]
        assert [o.query for o, _ in c.fixes] == ["fix"] and [o.query for o, _ in c.changed] == ["chg"]
        assert c.new == 1 and not c.clean

    def test_identical_runs_are_clean(self):
        o = [rq.Outcome("a", "x", "x", True)]
        assert rq.compare(o, o).clean

    def test_outcomes_json_round_trip_and_garbage(self):
        o = [rq.Outcome("a", "x", "y", True)]
        assert rq.outcomes_from_json(rq.outcomes_to_json(o)) == o
        assert rq.outcomes_from_json("nope") == [] and rq.outcomes_from_json('[{"query": 1}]') == []

    def test_report_flags_regressions_and_explains_unverified_rows(self):
        now = [rq.Outcome("reg", "x", "z", True), rq.Outcome("u", "x", "x", False)]
        before = [rq.Outcome("reg", "x", "x", True), rq.Outcome("u", "x", "x", False)]
        text = rq.format_report(now, rq.compare(now, before))
        assert "REGRESSED: 'reg': x -> z" in text and "1 regressed" in text
        assert "measures change" in rq.format_report(now)

    def test_cli_anonymise(self, capsys):
        assert rq.main(["anonymise", "mail a@b.co", "--name", "x"]) == 0
        assert capsys.readouterr().out.strip() == "mail <email>"

    def test_extract_merge_keeps_hand_verified_labels(self, tmp_path, monkeypatch):
        log, out = tmp_path / "r.jsonl", tmp_path / "c.jsonl"
        log.write_text(json.dumps({"query": "any mail", "intent": "read_email", "confidence": 0.9}) + "\n")
        rq.write_corpus(out, [rq.Row("any mail", "send_email", True), rq.Row("kept", "read_email", True)])
        assert rq.main(["extract", "--log", str(log), "--out", str(out)]) == 0
        rows = {r.query: r for r in rq.load_corpus(out)}
        assert rows["any mail"] == rq.Row("any mail", "send_email", True) and "kept" in rows

    def test_extract_with_no_log_fails_loudly(self, tmp_path):
        assert rq.main(["extract", "--log", str(tmp_path / "none.jsonl"), "--out", str(tmp_path / "o")]) == 2
