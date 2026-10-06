"""
Backlog #102 (voice-saved forms), #104 (breadboard check), #107 (browser pool),
#110 (multi-language scan + optional review), #119 (Google Fit / Health Connect
import). All synthetic: no real site, browser, phone export or model is used.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.browser_agent import HestiaBrowserAgent, form_fields_from_raw
from modules.apollo.engine import ApolloEngine, _parse_steps_file
from modules.hephaestus.engine import HephaestusEngine
from modules.hephaestus.repo_summary import build_review_prompt, format_summary, summarize_repo
from modules.iris.iris_engine import IrisEngine, parse_circuit_reply


# ------------------------------------------------------------------ #107
class _Page:
    def __init__(self):
        self.closed = False

    def is_closed(self):
        return self.closed

    def close(self):
        self.closed = True


def _pool(monkeypatch, **kw):
    browsers = []

    def launch(**_k):
        b = MagicMock()
        b.is_connected.return_value = True
        b.new_context.return_value.new_page.side_effect = lambda: _Page()
        browsers.append(b)
        return b

    pw = MagicMock()
    pw.chromium.launch.side_effect = launch
    fake = MagicMock()
    fake.sync_playwright.return_value.__enter__.return_value = pw
    monkeypatch.setitem(sys.modules, "playwright", MagicMock())
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake)
    return HestiaBrowserAgent(**kw), browsers


class TestBrowserPool:
    def test_default_is_one_browser_even_under_concurrent_pages(self, monkeypatch):
        a, browsers = _pool(monkeypatch)
        pages = [a._new_page() for _ in range(3)]
        assert len(browsers) == 1 and a.pool_stats()["pages_open"] == 3
        [p.close() for p in pages]

    def test_sequential_work_stays_on_one_browser(self, monkeypatch):
        a, browsers = _pool(monkeypatch, pool_size=3)
        for _ in range(4):
            a._new_page().close()
        assert len(browsers) == 1

    def test_grows_under_concurrent_work_up_to_the_limit(self, monkeypatch):
        a, browsers = _pool(monkeypatch, pool_size=2)
        pages = [a._new_page() for _ in range(4)]
        assert len(browsers) == 2 and a.pool_stats()["browsers_open"] == 2
        [p.close() for p in pages]
        assert a.pool_stats()["pages_open"] == 0

    @pytest.mark.parametrize("asked,expected", [(0, 1), (-3, 1), ("x", 1), (50, 8), (4, 4)])
    def test_pool_size_is_clamped(self, asked, expected):
        assert HestiaBrowserAgent(pool_size=asked).pool_size == expected

    def test_dead_browser_is_replaced(self, monkeypatch):
        a, browsers = _pool(monkeypatch, pool_size=2)
        a._new_page().close()
        browsers[0].is_connected.return_value = False
        a._new_page().close()
        assert len(browsers) == 2 and a.pool_stats()["browsers_open"] == 1

    def test_close_closes_every_browser(self, monkeypatch):
        a, browsers = _pool(monkeypatch, pool_size=2)
        pages = [a._new_page() for _ in range(2)]
        a.close()
        assert all(b.close.called for b in browsers) and a.pool_stats()["browsers_open"] == 0

    def test_idle_timeout_closes_the_whole_pool(self, monkeypatch):
        t = {"now": 0.0}
        a, browsers = _pool(monkeypatch, pool_size=2, idle_timeout_seconds=60, clock=lambda: t["now"])
        a._new_page()
        t["now"] = 500
        assert a.close_if_idle() is True and browsers[0].close.called

    def test_failed_extra_launch_reuses_an_open_browser(self, monkeypatch):
        a, browsers = _pool(monkeypatch, pool_size=2)
        a._new_page()
        a._playwright.chromium.launch.side_effect = RuntimeError("no memory")
        assert a._new_page() is not None and len(browsers) == 1


# ------------------------------------------------------------------ #102
RAW = {
    "fields": [
        {"id": "fn", "type": "text", "label": "Full name"},
        {"name": "email", "type": "email", "label": "Email address"},
        {"id": "pw", "type": "password", "label": ""},
        {"name": "card_number", "type": "text", "label": "Card number"},
        {"id": "country", "type": "select", "label": "Country"},
        {"id": "agree", "type": "checkbox"},
        {"id": "gone", "type": "text", "hidden": True},
        {"id": "off", "type": "text", "disabled": True},
        {"name": 'we"ird', "type": "text"},
        {"id": "fn", "type": "text", "label": "dup"},
        {"name": "email2", "type": "text", "label": "Email address"},
    ],
    "submits": [{"tag": "button", "id": "go", "type": "submit"}],
}


class TestFormDiscovery:
    def test_only_safe_fillable_fields_are_offered(self):
        r = form_fields_from_raw(RAW)
        assert [f["selector"] for f in r["fields"]] == ["#fn", '[name="email"]', '[name="email2"]']
        assert r["skipped_sensitive"] == 2 and r["submit_selector"] == "#go"

    def test_keys_are_unique_and_speakable(self):
        keys = [f["key"] for f in form_fields_from_raw(RAW)["fields"]]
        assert keys == ["full_name", "email_address", "email_address_2"]

    @pytest.mark.parametrize("bad", [None, [], "x", {"fields": "no"}, {"fields": [1, None]}])
    def test_garbage_does_not_raise(self, bad):
        assert form_fields_from_raw(bad)["fields"] == []

    def test_agent_returns_none_when_page_fails(self, monkeypatch):
        a = HestiaBrowserAgent()
        monkeypatch.setattr(a, "_new_page", lambda: None)
        assert a.discover_form_fields("example.com") is None
        page = MagicMock()
        page.goto.side_effect = Exception("dns")
        monkeypatch.setattr(a, "_new_page", lambda: page)
        assert a.discover_form_fields("example.com") is None
        page.close.assert_called()


class _FormBrowser:
    def __init__(self, found):
        self.found, self.fill_calls = found, []

    def discover_form_fields(self, url):
        return self.found

    def fill_form(self, url, fields, submit=None):
        self.fill_calls.append((url, fields, submit))
        return "Form filled but not submitted — no submit button specified."


FOUND = form_fields_from_raw(RAW)


def _eng(tmp_path=None, found=FOUND, forms=None):
    path = str(tmp_path / "forms.json") if tmp_path else None
    return HephaestusEngine(_FormBrowser(found), forms=forms, forms_store_path=path)


def _save(e, **ents):
    base = {"action": "save", "form": "gate", "url": "apply.example.org/gate"}
    return e.handle("fill_form", {**base, **ents}, {})


class TestSaveFormByVoice:
    def test_save_then_fill_asks_for_each_value_and_never_submits_by_default(self, tmp_path):
        e = _eng(tmp_path)
        r = _save(e)
        assert r["confidence"] > 0 and r["data"]["asked"] and not r["data"]["submit"]
        assert "left out 2 sensitive" in r["response"]
        ask = e.handle("fill_form", {"form": "gate"}, {})
        assert ask["data"]["needs_clarification"] and "full_name" in ask["response"]
        vals = {"full_name": "Asha", "email_address": "a@x.org", "email_address_2": "b@x.org"}
        first = e.handle("fill_form", {"form": "gate", "values": vals}, {})
        assert first["needs_confirmation"] and "Asha" not in first["response"]
        e.handle("fill_form", {**first["confirm_entities"], "_confirmed": True}, {})
        url, fields, submit = e._browser.fill_calls[0]
        assert fields["#fn"] == "Asha" and submit is None

    def test_given_values_become_fixed_and_submit_is_opt_in(self, tmp_path):
        e = _eng(tmp_path)
        r = _save(e, values={"full name": "Asha Rao"}, submit=True)
        assert r["data"]["fixed"] == ["full_name"] and r["data"]["submit"] is True
        assert e._forms["gate"]["submit_selector"] == "#go"

    def test_saved_forms_survive_a_restart_and_can_be_forgotten(self, tmp_path):
        _save(_eng(tmp_path))
        e2 = _eng(tmp_path)
        assert "gate" in e2.handle("fill_form", {"action": "list"}, {})["response"]
        assert "Forgot" in e2.handle("fill_form", {"action": "forget", "form": "gate"}, {})["response"]
        assert "gate" not in _eng(tmp_path)._forms

    def test_config_forms_cannot_be_overwritten_or_forgotten_by_voice(self, tmp_path):
        cfg = {"Gate": {"url": "https://apply.example.org/g", "fields": {"#a": "1"}}}
        e = _eng(tmp_path, forms=cfg)
        assert "config" in _save(e)["response"]
        assert "config" in e.handle("fill_form", {"action": "forget", "form": "gate"}, {})["response"]

    @pytest.mark.parametrize("url", ["http://localhost/x", "ftp://a.org/x", "http://192.168.1.1/"])
    def test_unsafe_addresses_are_refused_before_the_browser_is_used(self, tmp_path, url):
        e = _eng(tmp_path)
        e._browser.discover_form_fields = MagicMock()
        r = _save(e, url=url)
        assert r["confidence"] == 0.0 and not e._browser.discover_form_fields.called

    def test_no_fields_unreadable_page_and_missing_inputs(self, tmp_path):
        assert _save(_eng(tmp_path, found=None))["confidence"] == 0.0
        only_pw = form_fields_from_raw({"fields": [{"id": "p", "type": "password"}]})
        assert "sensitive" in _save(_eng(tmp_path, found=only_pw))["response"]
        assert _save(_eng(), form="")["data"]["needs_clarification"]
        assert _save(_eng(), url="")["data"]["needs_clarification"]

    def test_limit_and_unwritable_store(self, tmp_path):
        e = _eng(tmp_path)
        for i in range(20):
            assert _save(e, form=f"f{i}")["confidence"] > 0
        assert "only keep" in _save(e, form="one-too-many")["response"]
        bad = HephaestusEngine(_FormBrowser(FOUND), forms_store_path=str(tmp_path / "f.json" / "x" / "y"))
        (tmp_path / "f.json").write_text("not a dir")
        r = _save(bad)
        assert r["data"]["persisted"] is False and "forgotten when I restart" in r["response"]

    def test_corrupt_store_file_does_not_stop_startup(self, tmp_path):
        (tmp_path / "forms.json").write_text("{nope")
        assert _eng(tmp_path)._forms == {}


# ------------------------------------------------------------------ #110
def _w(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _body(n):
    return "\n".join(f"  x{i} = {i};" for i in range(n))


class TestMultiLanguageScan:
    def test_js_ts_go_rust_findings_and_dependencies(self, tmp_path):
        _w(tmp_path / "src/a.js", f"function big(a) {{\n{_body(90)}\n}}\ntry {{ f(); }} catch (e) {{}}\neval('1');\n")
        _w(tmp_path / "src/b.ts", "let a: any = 1; // @ts-ignore\n")
        _w(tmp_path / "srv/main.go", f"package main\nfunc H() {{\n{_body(85)}\n panic(\"x\")\n}}\n")
        _w(tmp_path / "lib/l.rs", "fn f() { let x = g().unwrap(); unsafe { h() }; }\n")
        _w(tmp_path / "requirements.txt", "flask==3.0\nrequests\n# c\n")
        _w(tmp_path / "package.json", json.dumps({"dependencies": {"a": "1"}, "devDependencies": {"b": "1", "c": "2"}}))
        s = summarize_repo(str(tmp_path))
        assert s["empty_catches"] == 1 and s["dynamic_exec"] == 1 and s["ts_any"] == 1
        assert s["ts_suppressions"] == 1 and s["go_panics"] == 1
        assert s["rust_unwraps"] == 1 and s["rust_unsafe"] == 1
        names = " ".join(w for _, w in s["long_functions"])
        assert "a.js" in names and "main.go" in names
        assert s["dependencies"]["requirements.txt"] == {"packages": 2, "unpinned": 1}
        assert s["dependencies"]["package.json"]["dev_packages"] == 2
        text = format_summary(s)
        assert "no version pinned" in text and "empty catch" in text

    def test_braces_in_strings_and_comments_do_not_confuse_it(self, tmp_path):
        _w(tmp_path / "a.js", '// function x() {\nconst s = "}}}";\nfunction ok() { return 1; }\n')
        s = summarize_repo(str(tmp_path))
        assert not s["long_functions"] and not s["unbalanced"]

    def test_unbalanced_file_is_reported_not_guessed(self, tmp_path):
        _w(tmp_path / "bad.js", "function f() { if (x) {\n")
        s = summarize_repo(str(tmp_path))
        assert s["unbalanced"] == ["bad.js"] and "unbalanced" in format_summary(s)

    def test_python_extras_and_old_summary_dicts_still_format(self, tmp_path):
        _w(tmp_path / "m.py", "def f(a=[]):\n    return eval('1')\n")
        s = summarize_repo(str(tmp_path))
        assert s["mutable_defaults"] == 1 and s["dynamic_exec"] == 1
        for k in ("dynamic_exec", "mutable_defaults", "dependencies", "unbalanced"):
            s.pop(k)
        assert "code files" in format_summary(s)


class _LLM:
    def __init__(self, reply="1. Split main()."):
        self.reply, self.prompts = reply, []

    def generate(self, prompt, fmt=None, options=None):
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class TestRepoReview:
    def _repo(self, tmp_path):
        _w(tmp_path / "main.py", "API_KEY = 'sk-123'\nprint('hi')\n" + "x = 1\n" * 10)
        return tmp_path

    def test_prompt_is_bounded_redacted_and_confined(self, tmp_path):
        repo = self._repo(tmp_path)
        s = summarize_repo(str(repo))
        s["entry_points"] = ["main.py", "../../etc/passwd"]
        p = build_review_prompt(s, max_chars=1500)
        assert len(p) <= 1500 and "sk-123" not in p and "--- main.py" in p and "--- ../../etc/passwd" not in p

    def test_review_is_off_unless_enabled_and_asked_for(self, tmp_path):
        repo = str(self._repo(tmp_path))
        llm = _LLM()
        e = HephaestusEngine(MagicMock(), llm=llm)
        r = e.handle("summarize_repo", {"path": repo, "review": True}, {})
        assert "turned off" in r["response"] and not llm.prompts
        e = HephaestusEngine(MagicMock(), llm=llm, repo_review={"enabled": True})
        assert not llm.prompts or e.handle("summarize_repo", {"path": repo}, {}) and not llm.prompts

    def test_enabled_review_returns_model_text_labelled_as_a_suggestion(self, tmp_path):
        repo = str(self._repo(tmp_path))
        llm = _LLM()
        e = HephaestusEngine(MagicMock(), llm=llm, repo_review={"enabled": True})
        r = e.handle("summarize_repo", {"path": repo, "review": True}, {})
        assert "Split main()" in r["response"] and "not a measurement" in r["response"]
        assert "sk-123" not in llm.prompts[0]

    @pytest.mark.parametrize("llm,expect", [
        (None, "isn't available"), (_LLM(RuntimeError("x")), "couldn't get a review"), (_LLM("  "), "no review")])
    def test_every_failure_still_returns_the_scan(self, tmp_path, llm, expect):
        e = HephaestusEngine(MagicMock(), llm=llm, repo_review={"enabled": True})
        r = e.handle("summarize_repo", {"path": str(self._repo(tmp_path)), "review": True}, {})
        assert expect in r["response"] and "code files" in r["response"] and r["confidence"] > 0


# ------------------------------------------------------------------ #104
class TestCircuitCheck:
    def test_parse_ok_problems_cannot_tell_and_unreadable(self):
        assert parse_circuit_reply("VERDICT: LOOKS_OK\nFINDINGS:\n- none") == {"verdict": "LOOKS_OK", "findings": []}
        r = parse_circuit_reply("VERDICT: POSSIBLE_PROBLEMS\nFINDINGS:\n- LED reversed, top left\n2. wire loose")
        assert r["verdict"] == "POSSIBLE_PROBLEMS" and r["findings"] == ["LED reversed, top left", "wire loose"]
        assert parse_circuit_reply("VERDICT: CANNOT_TELL\nFINDINGS:\n- too dark")["findings"] == ["too dark"]
        assert parse_circuit_reply("it works great!")["verdict"] == "UNREADABLE"
        assert parse_circuit_reply("")["verdict"] == "UNREADABLE"

    def test_contradictory_ok_with_findings_is_downgraded(self):
        r = parse_circuit_reply("VERDICT: LOOKS_OK\nFINDINGS:\n- cap reversed")
        assert r["verdict"] == "POSSIBLE_PROBLEMS"

    def _engine(self, tmp_path, reply, ftype="image", exists=True):
        eng = IrisEngine.__new__(IrisEngine)
        img = tmp_path / "bb.jpg"
        if exists:
            from PIL import Image
            Image.new("RGB", (40, 40), "red").save(img)
        rec = {"id": 7, "file_path": str(img), "file_type": ftype}
        eng.db = MagicMock()
        eng.db.get_all_files.return_value = [{"id": 1, "file_type": "video", "file_path": "v"}, rec]
        eng.db.get_file.side_effect = lambda i: rec if i == 7 else None
        eng.analyser = MagicMock()
        eng.analyser._send_to_ollama.return_value = reply
        return eng

    def test_newest_image_is_checked_and_reply_says_visual_only(self, tmp_path):
        eng = self._engine(tmp_path, "VERDICT: POSSIBLE_PROBLEMS\nFINDINGS:\n- LED reversed")
        r = eng.handle("iris_check_circuit", {}, {})
        assert "LED reversed" in r["response"] and "visual check only" in r["response"]
        assert eng.analyser._send_to_ollama.call_args[0][1].startswith("You are helping a hobbyist")

    def test_ok_never_claims_the_circuit_works(self, tmp_path):
        r = self._engine(tmp_path, "VERDICT: LOOKS_OK\nFINDINGS:\n- none").handle("check_circuit", {"file_id": 7}, {})
        assert "obviously wrong" in r["response"] and "works" not in r["response"].lower().replace("visual", "")

    @pytest.mark.parametrize("kw,needle", [
        ({"reply": ""}, "didn't answer"), ({"reply": "x", "ftype": "video"}, "isn't a photo"),
        ({"reply": "x", "exists": False}, "missing on disk")])
    def test_errors_are_spoken(self, tmp_path, kw, needle):
        r = self._engine(tmp_path, **kw).handle("check_circuit", {"file_id": 7}, {})
        assert needle in r["response"] and r["confidence"] < 0.5

    def test_unknown_photo_and_no_analyser(self, tmp_path):
        eng = self._engine(tmp_path, "x")
        assert "couldn't find" in eng.handle("check_circuit", {"file_id": 99}, {})["response"]
        eng.analyser = None
        assert "not configured" in eng.handle("check_circuit", {}, {})["response"]


# ------------------------------------------------------------------ #119
class TestStepExports:
    def test_google_fit_daily_metrics_csv(self, tmp_path):
        f = tmp_path / "Daily activity metrics.csv"
        f.write_text("Date,Move Minutes count,Step count\n2026-09-01,30,8123\n2026-09-02,10,\n2026-09-03,5,4000\n")
        assert _parse_steps_file(f) == ({"2026-09-01": 8123, "2026-09-03": 4000}, 0)

    def test_google_fit_per_day_file_takes_its_day_from_the_name(self, tmp_path):
        f = tmp_path / "2026-09-01.csv"
        f.write_text("Start time,End time,Step count\n06:00:00.000+05:30,06:15:00.000+05:30,100\n"
                     "07:00:00.000+05:30,07:15:00.000+05:30,250\n")
        assert _parse_steps_file(f)[0] == {"2026-09-01": 350}

    def test_google_fit_json_datapoints_use_the_local_day(self, tmp_path):
        from zoneinfo import ZoneInfo
        ns = int(datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc).timestamp() * 1e9)  # 00:30 on the 2nd in IST
        f = tmp_path / "all.json"
        f.write_text(json.dumps({"Data Points": [
            {"dataTypeName": "com.google.step_count.delta", "startTimeNanos": str(ns), "value": [{"intVal": 500}]},
            {"dataTypeName": "com.google.heart_rate.bpm", "startTimeNanos": str(ns), "value": [{"fpVal": 70}]}]}))
        assert _parse_steps_file(f, ZoneInfo("Asia/Kolkata"))[0] == {"2026-09-02": 500}
        assert _parse_steps_file(f)[0] == {"2026-09-01": 500}

    def _hc_db(self, path, with_apps=True):
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE steps_record_table (row_id INTEGER, start_time INTEGER, end_time INTEGER, "
                    "start_zone_offset INTEGER, count INTEGER, app_info_id INTEGER)")
        t = int(datetime(2026, 9, 1, 20, 0, tzinfo=timezone.utc).timestamp() * 1000)  # 01:30 IST on the 2nd
        con.executemany("INSERT INTO steps_record_table VALUES (?,?,?,?,?,?)", [
            (1, t, t + 1000, 19800, 1000, 1), (2, t + 5000, t + 6000, 19800, 500, 1),
            (3, t, t + 1000, 19800, 1200, 2),   # a watch counting the same walk
            (4, t, t + 1000, 19800, -5, 1)])
        con.execute("CREATE TABLE steps_series_record_table (x)")
        con.commit(); con.close()

    def test_health_connect_db_uses_recorded_zone_and_does_not_double_count(self, tmp_path):
        f = tmp_path / "hc.db"
        self._hc_db(f)
        days, skipped = _parse_steps_file(f)
        assert days == {"2026-09-02": 1500} and skipped == 1

    def test_zip_with_db_and_csv_members_keeps_the_larger_duplicate_day(self, tmp_path):
        self._hc_db(tmp_path / "hc.db")
        z = tmp_path / "export.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.write(tmp_path / "hc.db", "Health Connect/hc.db")
            zf.writestr("Takeout/Fit/Daily activity metrics.csv", "Date,Step count\n2026-09-02,1400\n2026-09-05,900\n")
            zf.writestr("../evil.csv", "Date,Step count\n2026-09-09,1\n")
            zf.writestr("notes.txt", "ignore me")
        days, _ = _parse_steps_file(z)
        assert days["2026-09-02"] == 1500 and days["2026-09-05"] == 900
        assert not (tmp_path.parent / "evil.csv").exists()

    def test_unrecognised_db_and_corrupt_zip(self, tmp_path):
        f = tmp_path / "other.db"
        sqlite3.connect(f).executescript("CREATE TABLE t (a)")
        assert _parse_steps_file(f) == ({}, 0)
        z = tmp_path / "bad.zip"
        z.write_bytes(b"not a zip")
        with pytest.raises(Exception):
            _parse_steps_file(z)

    def test_engine_imports_a_zip_applies_the_timezone_and_reports_a_bad_file(self, tmp_path):
        folder = tmp_path / "imp"
        folder.mkdir()
        self._hc_db(folder / "hc.db")
        (folder / "bad.zip").write_bytes(b"nope")
        e = ApolloEngine(ollama_cfg={}, db_path=tmp_path / "a.db", llm=MagicMock(),
                         config={"import_dir": str(folder), "timezone": "Asia/Kolkata"})
        r = e.handle("import_steps", {}, {})
        assert r["data"]["inserted"] == 1 and r["data"]["skipped_files"] == 1
        assert e.db.get_steps(4000)[0]["steps"] == 1500
        assert e.handle("import_steps", {}, {})["data"]["unchanged"] == 1
