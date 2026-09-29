# tests/test_nlu_reload.py
"""
Tests for HestiaNLU.reload_prompt (backlog #15) — the hot-reload path for
config/nlu_prompt.txt.

Uses real temp files rather than mocking _load_prompt, since the point is
to verify the file is actually re-read from disk.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.nlu import HestiaNLU

_MINIMAL_PROMPT = "You are Hestia. Valid intents: chat, take_note\n---\n"


def make_nlu(tmp_path, prompt_text=_MINIMAL_PROMPT):
    path = tmp_path / "nlu_prompt.txt"
    path.write_text(prompt_text, encoding="utf-8")
    return HestiaNLU(prompt_path=str(path), alias_path=None), path


def test_reload_prompt_picks_up_a_real_change(tmp_path):
    nlu, path = make_nlu(tmp_path)
    path.write_text(_MINIMAL_PROMPT + "\nEXTRA LINE\n", encoding="utf-8")
    changed = nlu.reload_prompt()
    assert changed is True
    assert "EXTRA LINE" in nlu.system_prompt


def test_reload_with_no_change_returns_false(tmp_path):
    nlu, path = make_nlu(tmp_path)
    assert nlu.reload_prompt() is False


def test_reload_invalidates_the_classification_cache(tmp_path):
    nlu, path = make_nlu(tmp_path)
    # Seed a cache entry directly (bypassing a real Ollama call).
    nlu._cache.put("hello", {
        "intent": "chat", "entities": {}, "response": "", "confidence": 0.9,
    })
    assert nlu._cache.get("hello") is not None

    path.write_text(_MINIMAL_PROMPT + "\nchanged\n", encoding="utf-8")
    nlu.reload_prompt()

    assert nlu._cache.get("hello") is None


def test_reload_with_no_change_does_not_touch_the_cache(tmp_path):
    nlu, path = make_nlu(tmp_path)
    nlu._cache.put("hello", {
        "intent": "chat", "entities": {}, "response": "", "confidence": 0.9,
    })
    nlu.reload_prompt()  # file unchanged
    assert nlu._cache.get("hello") is not None


def test_reload_rebuilds_the_schema_object(tmp_path):
    nlu, path = make_nlu(tmp_path)
    original_schema = nlu._schema
    path.write_text(_MINIMAL_PROMPT + "\nx\n", encoding="utf-8")
    nlu.reload_prompt()
    assert nlu._schema is not original_schema


def test_reload_accepts_an_explicit_path_override(tmp_path):
    nlu, original_path = make_nlu(tmp_path)
    other_path = tmp_path / "other_prompt.txt"
    other_path.write_text("A totally different prompt.\n", encoding="utf-8")

    nlu.reload_prompt(str(other_path))

    assert "totally different" in nlu.system_prompt


def test_reload_from_a_deleted_file_falls_back_gracefully(tmp_path):
    nlu, path = make_nlu(tmp_path)
    path.unlink()
    # _load_prompt already has a built-in fallback for a read failure at
    # startup; reload_prompt must not raise either.
    changed = nlu.reload_prompt()
    assert isinstance(changed, bool)
    assert nlu.system_prompt  # never left empty


def test_prompt_path_is_recorded_for_the_hot_reload_watcher(tmp_path):
    nlu, path = make_nlu(tmp_path)
    assert nlu._prompt_path == str(path)
