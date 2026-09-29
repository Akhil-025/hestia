# tests/test_writing_workflow.py
"""
Tests for the Metis / Orpheus writing-workflow backlog items:

  #164  writing session (Orpheus draft -> Metis critique/polish)
  #165  style-profile learning
  #166  word-count / readability targets on shorten / expand
  #167  Orpheus version history
  #168  export to .md / .txt
  #169  plagiarism check surfaces sources
  #270  optional polish pass on Orpheus output

Run with:  pytest tests/test_writing_workflow.py -v
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import text_export
from modules.metis.analysis import (
    count_syllables, count_words, distinctive_phrases, parse_length_target,
    text_metrics,
)
from modules.metis.engine import MetisEngine
from modules.orpheus.db import OrpheusDB
from modules.orpheus.engine import OrpheusEngine


class FakeLLM:
    """Returns queued responses in order (last one repeats)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt, fmt=None):
        self.prompts.append(prompt)
        i = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[i]


def orpheus(tmp_path, *responses, **kw):
    return OrpheusEngine(ollama_cfg={}, db_path=tmp_path / "o.db",
                         llm=FakeLLM(*responses), export_dir=tmp_path / "exp", **kw)


def metis(tmp_path, *responses, **kw):
    return MetisEngine(ollama_cfg={}, db_path=tmp_path / "m.db",
                       llm=FakeLLM(*responses), export_dir=tmp_path / "exp", **kw)


def words(n, w="word"):
    return " ".join([w] * n)


SAMPLE = ("I never really trusted mornings. They arrive too loud, too bright, "
          "and far too sure of themselves. I'd rather ease into the day; "
          "it's kinder that way, isn't it? Coffee first. Opinions later. "
          "Nobody should have to be clever before nine, and I refuse to try.")


# ===========================================================================
# #167 — Orpheus version history
# ===========================================================================

class TestVersionHistory:
    def test_new_creation_records_version_one(self, tmp_path):
        o = orpheus(tmp_path, "roses are red")
        r = o.handle("write_poem", {"topic": "roses"}, {})
        cid = r["data"]["creation_id"]
        versions = o.db.get_versions(cid)
        assert [v["version"] for v in versions] == [1]
        assert versions[0]["content"] == "roses are red"

    def test_revise_adds_version_and_keeps_original(self, tmp_path):
        o = orpheus(tmp_path, "original text", "revised text")
        cid = o.handle("write_poem", {"topic": "x"}, {})["data"]["creation_id"]
        r = o.handle("revise_creation",
                     {"creation_id": cid, "instruction": "make it warmer"}, {})
        assert r["data"]["version"] == 2
        assert o.db.get(cid)["content"] == "revised text"
        assert o.db.get_version(cid, 1)["content"] == "original text"

    def test_revise_uses_latest_when_no_id(self, tmp_path):
        o = orpheus(tmp_path, "one", "two")
        o.handle("write_poem", {"topic": "x"}, {})
        r = o.handle("revise_creation", {"type": "poem", "instruction": "shorter"}, {})
        assert r["data"]["version"] == 2

    def test_revise_without_instruction_asks(self, tmp_path):
        o = orpheus(tmp_path, "one")
        cid = o.handle("write_poem", {"topic": "x"}, {})["data"]["creation_id"]
        r = o.handle("revise_creation", {"creation_id": cid}, {})
        assert r["data"].get("needs_clarification") is True

    def test_revise_unknown_id_is_graceful(self, tmp_path):
        o = orpheus(tmp_path, "x")
        r = o.handle("revise_creation", {"creation_id": 99, "instruction": "y"}, {})
        assert r["data"]["found"] is False

    def test_revise_llm_failure_changes_nothing(self, tmp_path):
        o = orpheus(tmp_path, "one", "")
        cid = o.handle("write_poem", {"topic": "x"}, {})["data"]["creation_id"]
        r = o.handle("revise_creation", {"creation_id": cid, "instruction": "y"}, {})
        assert r["confidence"] == 0.0
        assert len(o.db.get_versions(cid)) == 1

    def test_revise_refuses_overlong_piece_instead_of_truncating(self, tmp_path):
        o = orpheus(tmp_path, words(13_000))
        cid = o.handle("write_story", {"topic": "x"}, {})["data"]["creation_id"]
        llm = o._llm_instance
        before = len(llm.prompts)
        r = o.handle("revise_creation", {"creation_id": cid, "instruction": "y"}, {})
        assert r["confidence"] == 0.0
        assert len(llm.prompts) == before          # never even called the model

    def test_list_versions_and_show_one(self, tmp_path):
        o = orpheus(tmp_path, "first", "second")
        cid = o.handle("write_poem", {"topic": "x"}, {})["data"]["creation_id"]
        o.handle("revise_creation", {"creation_id": cid, "instruction": "y"}, {})
        listing = o.handle("get_versions", {"creation_id": cid}, {})
        assert [v["version"] for v in listing["data"]["versions"]] == [1, 2]
        one = o.handle("get_versions", {"creation_id": cid, "version": "v1"}, {})
        assert one["data"]["content"] == "first"
        missing = o.handle("get_versions", {"creation_id": cid, "version": 9}, {})
        assert missing["data"]["found"] is False

    def test_restore_appends_copy_and_loses_nothing(self, tmp_path):
        o = orpheus(tmp_path, "first", "second")
        cid = o.handle("write_poem", {"topic": "x"}, {})["data"]["creation_id"]
        o.handle("revise_creation", {"creation_id": cid, "instruction": "y"}, {})
        r = o.handle("restore_version", {"creation_id": cid, "version": 1}, {})
        assert r["data"]["version"] == 3 and r["data"]["restored"] is True
        assert o.db.get(cid)["content"] == "first"
        assert [v["content"] for v in o.db.get_versions(cid)] == ["first", "second", "first"]

    def test_restore_current_text_is_a_noop(self, tmp_path):
        o = orpheus(tmp_path, "first")
        cid = o.handle("write_poem", {"topic": "x"}, {})["data"]["creation_id"]
        r = o.handle("restore_version", {"creation_id": cid, "version": 1}, {})
        assert r["data"]["restored"] is False
        assert len(o.db.get_versions(cid)) == 1

    def test_restore_needs_a_version(self, tmp_path):
        o = orpheus(tmp_path, "first")
        cid = o.handle("write_poem", {"topic": "x"}, {})["data"]["creation_id"]
        r = o.handle("restore_version", {"creation_id": cid}, {})
        assert r["data"].get("needs_clarification") is True

    def test_legacy_database_is_upgraded_in_place(self, tmp_path):
        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE creations (id INTEGER PRIMARY KEY AUTOINCREMENT,
              type TEXT NOT NULL, title TEXT, content TEXT NOT NULL,
              metadata TEXT, logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            INSERT INTO creations (type, title, content) VALUES ('poem','Old','ancient verse');
        """)
        conn.commit(); conn.close()
        db = OrpheusDB(str(path))
        assert db.get_version(1, 1)["content"] == "ancient verse"
        assert db.add_version(1, "new verse") == 2
        # Re-opening must not create a duplicate v1.
        assert len(OrpheusDB(str(path)).get_versions(1)) == 2

    def test_add_version_unknown_creation_raises(self, tmp_path):
        with pytest.raises(KeyError):
            OrpheusDB(str(tmp_path / "x.db")).add_version(5, "x")

    def test_recall_listing_shows_id_and_version_tag(self, tmp_path):
        o = orpheus(tmp_path, "one", "two")
        cid = o.handle("write_poem", {"topic": "x"}, {})["data"]["creation_id"]
        o.handle("revise_creation", {"creation_id": cid, "instruction": "y"}, {})
        r = o.handle("get_creations", {}, {})
        assert f"#{cid}" in r["response"] and " v2" in r["response"]


# ===========================================================================
# #168 — export
# ===========================================================================

class TestExportHelpers:
    def test_filename_cannot_escape_directory(self, tmp_path):
        stem = text_export.safe_stem("../../etc/passwd", "x")
        p = text_export.write_export(tmp_path / "out", stem, "md", "hi")
        assert p.parent == tmp_path / "out"

    def test_windows_style_path_is_reduced(self):
        assert "\\" not in text_export.safe_stem("C:\\Windows\\evil.txt", "x")
        assert text_export.safe_stem("..", "fallback") == "fallback"

    def test_never_overwrites(self, tmp_path):
        a = text_export.write_export(tmp_path, "poem", "md", "1")
        b = text_export.write_export(tmp_path, "poem", "md", "2")
        assert a != b and a.read_text() == "1" and b.read_text() == "2"

    def test_unknown_format_falls_back_to_markdown(self):
        assert text_export.normalise_format("docx") == "md"
        assert text_export.normalise_format("TXT") == "txt"

    def test_markdown_keeps_poem_line_breaks(self):
        doc = text_export.render_document("T", "a\nb\n\nc", "md", preserve_lines=True)
        assert "a  \nb" in doc


class TestOrpheusExport:
    def test_export_markdown(self, tmp_path):
        o = orpheus(tmp_path, "line one\nline two")
        cid = o.handle("write_poem", {"topic": "sea"}, {})["data"]["creation_id"]
        r = o.handle("export_creation", {"creation_id": cid, "format": "markdown"}, {})
        text = open(r["data"]["path"], encoding="utf-8").read()
        assert r["data"]["format"] == "md"
        assert text.startswith("# ") and "line one" in text

    def test_export_txt_with_history(self, tmp_path):
        o = orpheus(tmp_path, "one", "two")
        cid = o.handle("write_poem", {"topic": "x"}, {})["data"]["creation_id"]
        o.handle("revise_creation", {"creation_id": cid, "instruction": "y"}, {})
        r = o.handle("export_creation",
                     {"creation_id": cid, "format": "txt", "include_history": "yes"}, {})
        text = open(r["data"]["path"], encoding="utf-8").read()
        assert "Version history" in text and "v1" in text

    def test_export_specific_version(self, tmp_path):
        o = orpheus(tmp_path, "first", "second")
        cid = o.handle("write_poem", {"topic": "x"}, {})["data"]["creation_id"]
        o.handle("revise_creation", {"creation_id": cid, "instruction": "y"}, {})
        r = o.handle("export_creation", {"creation_id": cid, "version": 1}, {})
        assert "first" in open(r["data"]["path"], encoding="utf-8").read()

    def test_hostile_filename_stays_inside_export_dir(self, tmp_path):
        o = orpheus(tmp_path, "x y z")
        o.handle("write_poem", {"topic": "x"}, {})
        r = o.handle("export_creation", {"filename": "../../oops"}, {})
        assert os.path.dirname(r["data"]["path"]) == str(tmp_path / "exp")

    def test_nothing_to_export(self, tmp_path):
        o = orpheus(tmp_path)
        assert o.handle("export_creation", {}, {})["data"]["found"] is False


class TestMetisExport:
    def test_export_session_includes_original_draft(self, tmp_path):
        o = orpheus(tmp_path, "the draft poem here")
        m = metis(tmp_path,
                  json.dumps({"verdict": "needs_work", "issues": ["flat ending"]}),
                  "the better poem here")
        m.attach_orpheus(o)
        m.handle("writing_session", {"kind": "poem", "topic": "sea"}, {})
        r = m.handle("export_session", {"format": "md"}, {})
        text = open(r["data"]["path"], encoding="utf-8").read()
        assert "the better poem here" in text
        assert "Original draft" in text and "the draft poem here" in text
        assert "flat ending" in text

    def test_export_with_nothing_saved(self, tmp_path):
        assert metis(tmp_path).handle("export_session", {}, {})["data"]["found"] is False


# ===========================================================================
# #270 — polish pass toggle on Orpheus
# ===========================================================================

class FakePolisher:
    def __init__(self, result):
        self.result, self.calls = result, []

    def polish_pass(self, text, kind="prose", **kw):
        self.calls.append((text, kind))
        return self.result


class TestOrpheusPolishToggle:
    def test_off_by_default_never_calls_metis(self, tmp_path):
        o = orpheus(tmp_path, "draft")
        pol = FakePolisher({"ok": True, "changed": True, "polished": "better", "issues": []})
        o.attach_metis(pol)
        r = o.handle("write_poem", {"topic": "x"}, {})
        assert pol.calls == [] and r["data"]["poem"] == "draft"

    def test_per_request_polish_saves_draft_as_v1(self, tmp_path):
        o = orpheus(tmp_path, "draft")
        pol = FakePolisher({"ok": True, "changed": True, "polished": "better", "issues": ["x"]})
        o.attach_metis(pol)
        r = o.handle("write_poem", {"topic": "x", "polish": "yes"}, {})
        cid = r["data"]["creation_id"]
        assert pol.calls == [("draft", "creative")]
        assert r["data"]["poem"] == "better"
        assert o.db.get_version(cid, 1)["content"] == "draft"
        assert o.db.get(cid)["content"] == "better"
        assert "Polished by Metis" in r["response"]

    def test_config_default_on_and_request_can_opt_out(self, tmp_path):
        o = orpheus(tmp_path, "draft", polish_default=True)
        pol = FakePolisher({"ok": True, "changed": False, "polished": "draft", "issues": []})
        o.attach_metis(pol)
        o.handle("write_story", {"topic": "x"}, {})
        assert len(pol.calls) == 1
        o.handle("write_story", {"topic": "x", "polish": "false"}, {})
        assert len(pol.calls) == 1

    def test_no_metis_attached_keeps_draft_and_says_so(self, tmp_path):
        o = orpheus(tmp_path, "draft")
        r = o.handle("generate_lyrics", {"topic": "x", "polish": True}, {})
        assert r["data"]["lyrics"] == "draft"
        assert "skipped" in r["response"]

    def test_metis_crash_keeps_draft(self, tmp_path):
        class Boom:
            def polish_pass(self, *a, **k):
                raise RuntimeError("boom")
        o = orpheus(tmp_path, "draft")
        o.attach_metis(Boom())
        r = o.handle("write_poem", {"topic": "x", "polish": True}, {})
        assert r["data"]["poem"] == "draft" and r["confidence"] > 0


# ===========================================================================
# #164 / #270 — Metis polish pass and writing session
# ===========================================================================

def critique(*issues):
    return json.dumps({"verdict": "needs_work" if issues else "good",
                       "issues": list(issues)})


class TestPolishPass:
    def test_clean_text_makes_one_call(self, tmp_path):
        m = metis(tmp_path, critique())
        res = m.polish_pass("a fine and complete little sentence for testing here")
        assert res["ok"] and not res["changed"] and res["verdict"] == "good"
        assert len(m._llm_instance.prompts) == 1

    def test_issues_trigger_exactly_one_revision(self, tmp_path):
        m = metis(tmp_path, critique("cliché opening"), "a fresh replacement text")
        res = m.polish_pass("a stale old text that needs some work today")
        assert res["changed"] and res["polished"] == "a fresh replacement text"
        assert len(m._llm_instance.prompts) == 2       # not an iterative loop

    def test_verdict_needs_work_without_issues_is_treated_as_good(self, tmp_path):
        m = metis(tmp_path, json.dumps({"verdict": "needs_work", "issues": []}))
        assert m.polish_pass("some text")["verdict"] == "good"

    def test_creative_prompt_protects_form_and_skips_style_profile(self, tmp_path):
        m = metis(tmp_path, critique("flat"), "new")
        m.db.save_style_profile(json.dumps({"voice_summary": "curt", "traits": ["curt"]}), 1)
        m.polish_pass("roses are red\nviolets are blue", kind="creative")
        revise_prompt = m._llm_instance.prompts[1]
        assert "line breaks" in revise_prompt and "curt" not in revise_prompt

    def test_runaway_revision_is_rejected(self, tmp_path):
        m = metis(tmp_path, critique("flat"), "tiny")
        res = m.polish_pass(words(50))
        assert not res["ok"] and res["polished"] == words(50)
        assert "drifted" in res["reason"]

    def test_critique_failure_is_graceful(self, tmp_path):
        res = metis(tmp_path, "not json").polish_pass("hello there friend")
        assert res["ok"] is False and res["polished"] == "hello there friend"

    def test_empty_input(self, tmp_path):
        assert metis(tmp_path).polish_pass("   ")["ok"] is False

    def test_polish_text_intent(self, tmp_path):
        m = metis(tmp_path, critique("wordy"), "tight")
        r = m.handle("polish_text", {"text": "a rather long and wordy sentence here"}, {})
        assert r["data"]["changed"] and "tight" in r["response"]
        assert m.handle("polish_text", {}, {})["data"].get("needs_clarification")


class TestWritingSession:
    def _pair(self, tmp_path, o_resp, m_resps):
        o = orpheus(tmp_path, o_resp)
        m = metis(tmp_path, *m_resps)
        m.attach_orpheus(o)
        return o, m

    def test_full_session_keeps_draft_as_v1_and_polish_as_v2(self, tmp_path):
        o, m = self._pair(tmp_path, "draft poem text",
                          [critique("weak ending"), "polished poem text"])
        r = m.handle("writing_session", {"kind": "poem", "topic": "sea"}, {})
        cid = r["data"]["creation_id"]
        assert r["data"]["final"] == "polished poem text" and r["data"]["polished"]
        assert [v["content"] for v in o.db.get_versions(cid)] == \
            ["draft poem text", "polished poem text"]
        assert "weak ending" in r["response"]

    def test_clean_draft_is_left_alone(self, tmp_path):
        o, m = self._pair(tmp_path, "already lovely", [critique()])
        r = m.handle("writing_session", {"kind": "story", "topic": "x"}, {})
        assert r["data"]["final"] == "already lovely"
        assert len(o.db.get_versions(r["data"]["creation_id"])) == 1

    def test_critique_only_mode(self, tmp_path):
        o, m = self._pair(tmp_path, "draft text", [critique("flat")])
        r = m.handle("writing_session", {"kind": "poem", "topic": "x", "polish": "no"}, {})
        assert r["data"]["final"] == "draft text" and r["data"]["issues"] == ["flat"]

    def test_kind_inferred_from_raw_query(self, tmp_path):
        _, m = self._pair(tmp_path, "song text", [critique()])
        r = m.handle("writing_session",
                     {"topic": "home", "raw_query": "start a writing session for a song"}, {})
        assert r["data"]["kind"] == "lyrics"

    def test_missing_pieces_ask_for_them(self, tmp_path):
        _, m = self._pair(tmp_path, "x", [critique()])
        r = m.handle("writing_session", {}, {})
        assert r["data"].get("needs_clarification") is True
        assert "poem, a story, or song lyrics" in r["response"]

    def test_without_orpheus_fails_gracefully(self, tmp_path):
        r = metis(tmp_path).handle("writing_session", {"kind": "poem", "topic": "x"}, {})
        assert r["confidence"] == 0.0

    def test_orpheus_failure_is_reported(self, tmp_path):
        _, m = self._pair(tmp_path, "", [critique()])
        r = m.handle("writing_session", {"kind": "poem", "topic": "x"}, {})
        assert r["confidence"] == 0.0

    def test_session_does_not_double_polish(self, tmp_path):
        o = orpheus(tmp_path, "draft", polish_default=True)
        m = metis(tmp_path, critique())
        m.attach_orpheus(o)
        o.attach_metis(m)
        m.handle("writing_session", {"kind": "poem", "topic": "x"}, {})
        # one critique call only: Orpheus's own polish was suppressed.
        assert len(m._llm_instance.prompts) == 1


# ===========================================================================
# #165 — style profile
# ===========================================================================

NOTES = json.dumps({"voice_summary": "Dry and wry.", "signature_traits": ["short asides"],
                    "avoid": ["corporate jargon"]})


class TestStyleProfile:
    def test_learn_builds_profile_and_reports_it(self, tmp_path):
        m = metis(tmp_path, NOTES)
        r = m.handle("learn_style", {"text": SAMPLE}, {})
        assert r["data"]["samples"] == 1
        assert "Dry and wry." in r["response"]
        assert m._load_profile()["voice_summary"] == "Dry and wry."

    def test_too_short_a_sample_is_refused(self, tmp_path):
        r = metis(tmp_path, NOTES).handle("learn_style", {"text": "too short"}, {})
        assert r["data"]["needs_more"] is True

    def test_no_sample_asks_for_one(self, tmp_path):
        assert metis(tmp_path).handle("learn_style", {}, {})["data"].get("needs_clarification")

    def test_llm_failure_still_gives_measured_profile(self, tmp_path):
        m = metis(tmp_path, "garbage")
        r = m.handle("learn_style", {"text": SAMPLE}, {})
        assert r["data"]["summary_available"] is False
        assert r["data"]["profile"]["traits"]          # measurements survive

    def test_second_sample_accumulates(self, tmp_path):
        m = metis(tmp_path, NOTES)
        m.handle("learn_style", {"text": SAMPLE}, {})
        r = m.handle("learn_style", {"text": SAMPLE + " Again."}, {})
        assert r["data"]["samples"] == 2

    def test_profile_reaches_rewrite_prompt(self, tmp_path):
        m = metis(tmp_path, NOTES)
        m.handle("learn_style", {"text": SAMPLE}, {})
        m._llm_instance = FakeLLM("rewritten")
        r = m.handle("rewrite_text", {"text": "hello", "goal": "clarity"}, {})
        assert "Dry and wry." in m._llm_instance.prompts[0]
        assert r["data"]["style_applied"] is True

    @pytest.mark.parametrize("intent,resp", [
        ("correct_text", json.dumps({"corrected": "ok", "changes": []})),
        ("improve_clarity", json.dumps({"revised": "ok", "notes": []})),
        ("suggest_style", json.dumps({"suggestions": [], "revised": "ok"})),
        ("expand_text", "ok"), ("shorten_text", "ok"), ("draft_content", "ok"),
    ])
    def test_profile_reaches_every_voice_aware_prompt(self, tmp_path, intent, resp):
        m = metis(tmp_path, NOTES)
        m.handle("learn_style", {"text": SAMPLE}, {})
        m._llm_instance = FakeLLM(resp)
        m.handle(intent, {"text": "hello world", "brief": "a note"}, {})
        assert "Dry and wry." in m._llm_instance.prompts[0]

    def test_use_style_false_opts_out(self, tmp_path):
        m = metis(tmp_path, NOTES)
        m.handle("learn_style", {"text": SAMPLE}, {})
        m._llm_instance = FakeLLM("rewritten")
        r = m.handle("rewrite_text", {"text": "hello", "use_style": "false"}, {})
        assert "Dry and wry." not in m._llm_instance.prompts[0]
        assert r["data"]["style_applied"] is False

    def test_no_profile_means_unchanged_prompts(self, tmp_path):
        m = metis(tmp_path, "rewritten")
        m.handle("rewrite_text", {"text": "hello"}, {})
        assert "personal writing voice" not in m._llm_instance.prompts[0]

    def test_show_and_clear(self, tmp_path):
        m = metis(tmp_path, NOTES)
        assert m.handle("show_style_profile", {}, {})["data"]["has_profile"] is False
        assert m.handle("clear_style_profile", {}, {})["data"]["removed"] == 0
        m.handle("learn_style", {"text": SAMPLE}, {})
        assert m.handle("show_style_profile", {}, {})["data"]["has_profile"] is True
        assert m.handle("clear_style_profile", {}, {})["data"]["removed"] == 1
        assert m._load_profile() is None
        assert m.db.get_style_samples() == []

    def test_corrupt_stored_profile_is_ignored(self, tmp_path):
        m = metis(tmp_path, "rewritten")
        m.db.save_style_profile("{not json", 1)
        assert m.handle("rewrite_text", {"text": "hello"}, {})["confidence"] > 0


# ===========================================================================
# #166 — length / readability targets
# ===========================================================================

class TestLengthTargets:
    def test_parse_variants(self):
        t, err = parse_length_target({"max_words": "under 120 words"})
        assert t.max_words == 120 and not err
        t, _ = parse_length_target({"target_words": 100, "grade": "8"})
        assert t.target_words == 100 and t.grade == 8.0
        assert parse_length_target({}) == (None, "")

    def test_unparseable_number_asks_instead_of_guessing(self):
        t, err = parse_length_target({"target_words": "abc"})
        assert t is None and err

    def test_met_first_time_is_one_call(self, tmp_path):
        m = metis(tmp_path, words(95))
        r = m.handle("shorten_text", {"text": words(300), "target_words": 100}, {})
        assert r["data"]["target_met"] is True and r["data"]["attempts"] == 1
        assert len(m._llm_instance.prompts) == 1
        assert "target met" in r["response"]

    def test_miss_triggers_one_retry_naming_the_problem(self, tmp_path):
        m = metis(tmp_path, words(250), words(98))
        r = m.handle("shorten_text", {"text": words(300), "target_words": 100}, {})
        assert r["data"]["target_met"] is True and r["data"]["attempts"] == 2
        assert "must be at most" in m._llm_instance.prompts[1] or \
               "words" in m._llm_instance.prompts[1]

    def test_unreachable_target_is_reported_not_faked(self, tmp_path):
        m = metis(tmp_path, words(250), words(240))
        r = m.handle("shorten_text", {"text": words(300), "max_words": 50}, {})
        assert r["data"]["target_met"] is False
        assert "couldn't quite reach" in r["response"]
        assert count_words(r["data"]["shortened"]) == 240     # never truncated

    def test_retry_keeps_the_closer_attempt(self, tmp_path):
        m = metis(tmp_path, words(120), words(300))
        r = m.handle("shorten_text", {"text": words(400), "max_words": 100}, {})
        assert count_words(r["data"]["shortened"]) == 120

    def test_expand_min_words(self, tmp_path):
        m = metis(tmp_path, words(210))
        r = m.handle("expand_text", {"text": words(20), "min_words": 200}, {})
        assert r["data"]["target_met"] is True

    def test_no_target_behaves_as_before(self, tmp_path):
        m = metis(tmp_path, "short result")
        r = m.handle("shorten_text", {"text": "something long enough"}, {})
        assert "target_met" not in r["data"] and r["response"] == "short result"
        assert len(m._llm_instance.prompts) == 1

    def test_bad_number_asks_a_question(self, tmp_path):
        r = metis(tmp_path).handle("shorten_text", {"text": "x y z", "max_words": "lots"}, {})
        assert r["data"].get("needs_clarification") is True

    def test_summarise_and_draft_accept_targets(self, tmp_path):
        m = metis(tmp_path, words(40))
        assert m.handle("summarize_text", {"text": words(200), "target_words": 40}, {})["data"]["target_met"]
        m = metis(tmp_path, words(40))
        assert m.handle("draft_content", {"brief": "a note", "target_words": 40}, {})["data"]["target_met"]

    def test_readability_report_includes_measured_metrics(self, tmp_path):
        m = metis(tmp_path, json.dumps({"reading_ease": "easy", "sentence_variety": "good",
                                         "issues": [], "suggestions": []}))
        r = m.handle("readability_report", {"text": "The cat sat. The dog ran fast."}, {})
        assert r["data"]["metrics"]["words"] == 7 and "Measured:" in r["response"]

    def test_metrics_sanity(self):
        assert count_syllables("created") == 2 and count_syllables("walked") == 1
        assert text_metrics("")["words"] == 0


# ===========================================================================
# #169 — plagiarism sources
# ===========================================================================

ESSAY = ("Cassandra Whitmore-Hughes founded the Halcyon Institute in 1987 to study "
         "migratory patterns of the Arctic tern across the northern hemisphere. "
         "Its longitudinal surveys remain unmatched in scope and rigour.")


class TestPlagiarismSources:
    def test_no_search_returns_honest_fallback_with_manual_queries(self, tmp_path):
        r = metis(tmp_path).handle("check_plagiarism", {"text": ESSAY}, {})
        assert r["data"]["supported"] is False and r["data"]["queries"]
        assert "originality score" in r["response"]

    def test_sources_are_reported_with_urls(self, tmp_path):
        m = metis(tmp_path)
        m.attach_web_search(lambda q, max_results=3:
                            [{"title": "Halcyon history", "url": "https://example.org/h"}])
        r = m.handle("check_plagiarism", {"text": ESSAY}, {})
        assert r["data"]["supported"] is True
        assert r["data"]["sources"][0]["url"] == "https://example.org/h"
        assert "https://example.org/h" in r["response"]

    def test_fetch_confirms_or_discards_leads(self, tmp_path):
        m = metis(tmp_path)

        def search(q, max_results=3):
            return [{"title": "T", "url": "https://real.example/a"},
                    {"title": "U", "url": "https://noise.example/b"}]

        def fetch(url):
            return ESSAY if "real" in url else "completely unrelated page about cooking"

        m.attach_web_search(search, fetch)
        r = m.handle("check_plagiarism", {"text": ESSAY}, {})
        urls = {s["url"] for s in r["data"]["sources"]}
        assert urls == {"https://real.example/a"}
        assert all(s["status"] == "confirmed" for s in r["data"]["sources"])

    def test_no_hits_does_not_overclaim(self, tmp_path):
        m = metis(tmp_path)
        m.attach_web_search(lambda q, max_results=3: [])
        r = m.handle("check_plagiarism", {"text": ESSAY}, {})
        assert r["data"]["phrases_matched"] == 0
        assert "isn't working" in r["response"] or "unique" in r["response"]

    def test_search_that_always_raises_falls_back(self, tmp_path):
        m = metis(tmp_path)

        def boom(*a, **k):
            raise RuntimeError("offline")

        m.attach_web_search(boom)
        r = m.handle("check_plagiarism", {"text": ESSAY}, {})
        assert r["data"]["supported"] is False and "didn't work" in r["response"]

    def test_distinctive_phrases_skip_boilerplate(self):
        assert distinctive_phrases("the and of to a in is it", k=3) == []
        assert distinctive_phrases(ESSAY, k=2)
