"""
tests/test_telegram_main_hooks.py

The main.py side of the Telegram bot (backlog #191-#196): the hooks Hestia
hands the bot, and HestiaBuilder.build_telegram_bot's config handling.

Hestia.__init__ boots the whole app, so (as in test_main.py) these build an
instance with object.__new__ and wire only what each method touches. The bot
class itself is covered by test_telegram_bot.py.
"""
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main


def _hestia():
    h = object.__new__(main.Hestia)
    h.mnemosyne = MagicMock()
    h.mnemosyne.get_recent.return_value = []
    h.nlu = MagicMock()
    h.nlu.understand.return_value = {"intent": "get_time", "entities": {}, "response": ""}
    h.orchestrator = MagicMock()
    h.orchestrator._pending = None
    h.chronos = MagicMock()
    h.iris = MagicMock()
    h.athena = MagicMock()
    return h


# ---------------------------------------------------------------------------
# _classify_for_telegram (#194)
# ---------------------------------------------------------------------------

class TestClassifyForTelegram:
    def test_simple_message_returns_its_intent(self):
        h = _hestia()
        assert h._classify_for_telegram("what time is it") == ["get_time"]

    def test_empty_message_has_no_intents(self):
        assert _hestia()._classify_for_telegram("   ") == []

    def test_compound_message_classifies_the_whole_and_each_half(self):
        h = _hestia()
        h.nlu.understand.side_effect = lambda text, ctx: {
            "intent": "delete_notes" if "delete" in text else "get_time"
        }
        intents = h._classify_for_telegram("what time is it and delete my notes")
        # The whole sentence plus both halves: nothing can hide in the second clause.
        assert len(intents) == 3
        assert "delete_notes" in intents

    def test_voice_controls_are_reported_as_an_unregistered_intent(self):
        """"do not disturb" acts on the owner's machine before NLU ever runs, so
        it must not look like an allowed intent to a restricted chat."""
        h = _hestia()
        intents = h._classify_for_telegram("do not disturb for an hour")
        assert intents == ["voice_control"]
        assert main.module_for_intent("voice_control") is None
        h.nlu.understand.assert_not_called()

    def test_classification_uses_recent_context(self):
        h = _hestia()
        h.mnemosyne.get_recent.return_value = [{"q": "x"}]
        h._classify_for_telegram("what time is it")
        assert h.nlu.understand.call_args.args[1] == [{"q": "x"}]

    def test_nlu_errors_propagate_so_the_bot_can_fail_closed(self):
        h = _hestia()
        h.nlu.understand.side_effect = RuntimeError("ollama down")
        with pytest.raises(RuntimeError):
            h._classify_for_telegram("what time is it")


# ---------------------------------------------------------------------------
# _has_pending_confirmation (#191)
# ---------------------------------------------------------------------------

def test_pending_confirmation_follows_the_orchestrator():
    h = _hestia()
    assert h._has_pending_confirmation() is False
    h.orchestrator._pending = object()
    assert h._has_pending_confirmation() is True


# ---------------------------------------------------------------------------
# _telegram_snooze (#191)
# ---------------------------------------------------------------------------

class TestTelegramSnooze:
    def test_calls_chronos_directly_with_the_duration(self):
        h = _hestia()
        h.chronos.handle.return_value = {"response": "Snoozed 'call mum' for 10 minutes."}
        assert h._telegram_snooze(10) == "Snoozed 'call mum' for 10 minutes."
        intent, entities, _ctx = h.chronos.handle.call_args.args
        assert intent == "snooze_reminder"
        assert entities["duration"] == "10 minutes"

    def test_empty_response_gets_a_fallback(self):
        h = _hestia()
        h.chronos.handle.return_value = {"response": ""}
        assert "couldn't snooze" in h._telegram_snooze(10)

    def test_the_real_chronos_parses_the_duration_and_hint(self):
        """The entities the button sends must be understood by Chronos's own parsers."""
        from modules.chronos.recurrence import parse_duration, strip_duration
        from modules.chronos.engine import _command_hint
        from datetime import timedelta
        assert parse_duration("60 minutes") == timedelta(minutes=60)
        raw = "snooze 60 minutes"
        assert not _command_hint(strip_duration(raw), None)   # no hint: most recent fired reminder


# ---------------------------------------------------------------------------
# _telegram_ingest_photo (#192)
# ---------------------------------------------------------------------------

class TestTelegramIngestPhoto:
    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        h = _hestia()
        monkeypatch.setattr(h, "_telegram_inbox", lambda name: tmp_path / name, raising=False)
        src = tmp_path / "incoming.jpg"
        src.write_bytes(b"jpeg")
        return h, src, tmp_path

    def test_ingested_photo_is_kept_and_iris_scans_only_its_folder(self, setup):
        h, src, tmp_path = setup
        h.iris.ingest.return_value = {"ingested": 1, "duplicates_skipped": 0, "errors": 0}
        assert h._telegram_ingest_photo(str(src), "telegram_abc.jpg") == "Saved to your photo library."
        folder = Path(h.iris.ingest.call_args.kwargs["source_dir"])
        assert folder.parent == tmp_path / "photos"
        assert (folder / "telegram_abc.jpg").read_bytes() == b"jpeg"

    def test_duplicate_is_reported_and_not_kept(self, setup):
        h, src, tmp_path = setup
        h.iris.ingest.return_value = {"ingested": 0, "duplicates_skipped": 1, "errors": 0}
        assert "already in your library" in h._telegram_ingest_photo(str(src), "a.jpg")
        assert not Path(h.iris.ingest.call_args.kwargs["source_dir"]).exists()

    def test_quota_is_reported(self, setup):
        h, src, _ = setup
        h.iris.ingest.return_value = {"exceeds_quota": True, "ingested": 0}
        assert "quota" in h._telegram_ingest_photo(str(src), "a.jpg")

    def test_failure_is_reported_and_not_kept(self, setup):
        h, src, _ = setup
        h.iris.ingest.return_value = {"ingested": 0, "duplicates_skipped": 0, "errors": 1}
        assert "couldn't add" in h._telegram_ingest_photo(str(src), "a.jpg")
        assert not Path(h.iris.ingest.call_args.kwargs["source_dir"]).exists()

    def test_iris_exception_cleans_up_and_propagates(self, setup):
        h, src, _ = setup
        h.iris.ingest.side_effect = RuntimeError("clip failed")
        with pytest.raises(RuntimeError):
            h._telegram_ingest_photo(str(src), "a.jpg")
        assert not Path(h.iris.ingest.call_args.kwargs["source_dir"]).exists()

    def test_iris_disabled(self, setup):
        h, src, _ = setup
        h.iris = None
        assert "Iris is off" in h._telegram_ingest_photo(str(src), "a.jpg")


# ---------------------------------------------------------------------------
# _telegram_ingest_document (#192)
# ---------------------------------------------------------------------------

class TestTelegramIngestDocument:
    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        # Importing the real modules.athena.config drags in chromadb via the
        # package __init__, which this sandbox doesn't have; only get_config's
        # data_dir is used here.
        fake = types.ModuleType("modules.athena.config")
        fake.get_config = lambda: MagicMock(data_dir=tmp_path / "docs")
        monkeypatch.setitem(sys.modules, "modules.athena.config", fake)
        h = _hestia()
        src = tmp_path / "in.pdf"
        src.write_bytes(b"%PDF")
        return h, src, tmp_path

    def test_pdf_is_copied_under_a_telegram_subject_and_ingested(self, setup):
        h, src, tmp_path = setup
        h.athena.rag.ingest_file.return_value = (12, "new")
        reply = h._telegram_ingest_document(str(src), "paper.pdf")
        info = h.athena.rag.ingest_file.call_args.args[0]
        assert info["subject"] == "Telegram" and info["file_name"] == "paper.pdf"
        assert Path(info["full_path"]) == tmp_path / "docs" / "Telegram" / "paper.pdf"
        assert Path(info["full_path"]).read_bytes() == b"%PDF"
        assert reply.startswith("Added paper.pdf") and "12 chunks" in reply

    def test_updated_unchanged_and_failed_statuses(self, setup):
        h, src, _ = setup
        h.athena.rag.ingest_file.return_value = (5, "updated")
        assert h._telegram_ingest_document(str(src), "p.pdf").startswith("Updated p.pdf")
        h.athena.rag.ingest_file.return_value = (0, "unchanged")
        assert "already in your documents" in h._telegram_ingest_document(str(src), "p.pdf")
        h.athena.rag.ingest_file.return_value = (0, "failed")
        assert "couldn't read" in h._telegram_ingest_document(str(src), "p.pdf")

    def test_athena_disabled(self, setup):
        h, src, _ = setup
        h.athena = None
        assert "Athena is off" in h._telegram_ingest_document(str(src), "p.pdf")

    def test_the_real_file_info_shape_matches_what_athena_expects(self, setup):
        """ingest_file reads full_path/file_name/subject/module (see get_supported_files)."""
        h, src, _ = setup
        h.athena.rag.ingest_file.return_value = (1, "new")
        h._telegram_ingest_document(str(src), "p.pdf")
        info = h.athena.rag.ingest_file.call_args.args[0]
        assert {"full_path", "file_name", "subject", "module", "relative_path"} <= set(info)


# ---------------------------------------------------------------------------
# HestiaBuilder.build_telegram_bot
# ---------------------------------------------------------------------------

class TestBuildTelegramBot:
    @pytest.fixture(autouse=True)
    def _bot_class(self, monkeypatch):
        import core.telegram_bot as tb
        self.created = []

        class FakeBot:
            def __init__(inner, **kwargs):
                inner.kwargs = kwargs
                inner.started = False
                inner.bus = None
                self.created.append(inner)

            def start(inner):
                inner.started = True

            def attach_event_bus(inner, bus):
                inner.bus = bus

        monkeypatch.setattr(tb, "HestiaTelegramBot", FakeBot)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")

    def _build(self, tg_cfg, **hooks):
        builder = main.HestiaBuilder({"telegram": tg_cfg})
        return builder.build_telegram_bot(MagicMock(), None, None, **hooks)

    def test_disabled_builds_nothing(self):
        assert self._build({"enabled": False}) is None
        assert self.created == []

    def test_missing_token_builds_nothing(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
        assert self._build({"enabled": True}) is None

    def test_roles_policies_and_hooks_are_passed_through(self):
        hooks = {k: MagicMock() for k in (
            "classify_fn", "pending_fn", "snooze_fn", "ingest_photo_fn",
            "ingest_document_fn", "should_push_fn",
        )}
        cfg = {
            "enabled": True,
            "allowed_chat_ids": [1],
            "roles": {1: "owner", 2: "family"},
            "role_policies": {"family": {"allow_modules": ["core"]}},
            "snooze_minutes": [5, 30],
        }
        bot = self._build(cfg, **hooks)
        assert bot.started
        for name, fn in hooks.items():
            assert bot.kwargs[name] is fn
        assert bot.kwargs["roles"] == {1: "owner", 2: "family"}
        assert bot.kwargs["role_policies"] == {"family": {"allow_modules": ["core"]}}
        assert bot.kwargs["snooze_minutes"] == [5, 30]

    def test_defaults_match_the_old_behaviour(self):
        bot = self._build({"enabled": True})
        assert bot.kwargs["roles"] is None
        assert bot.kwargs["role_policies"] is None
        assert bot.kwargs["snooze_minutes"] == (10, 60)
        assert bot.bus is None            # push is off unless asked for

    def test_push_notifications_attach_the_event_bus(self, monkeypatch):
        fake_bus = MagicMock()
        monkeypatch.setattr(main, "bus", fake_bus)
        bot = self._build({"enabled": True, "push_notifications": True})
        assert bot.bus is fake_bus

    def test_startup_failure_is_swallowed(self, monkeypatch):
        import core.telegram_bot as tb
        monkeypatch.setattr(tb, "HestiaTelegramBot", MagicMock(side_effect=RuntimeError("bad token")))
        assert self._build({"enabled": True}) is None


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def _issues(self, telegram):
        from core.config_validation import validate_config
        result = validate_config({"telegram": telegram})
        return [str(i) for i in (result if isinstance(result, (list, tuple)) else getattr(result, "errors", result))]

    def test_valid_telegram_block_is_accepted(self):
        issues = self._issues({
            "enabled": True, "allowed_chat_ids": [1], "roles": {1: "owner"},
            "role_policies": {"owner": {}}, "push_notifications": True, "snooze_minutes": [10],
        })
        assert not [i for i in issues if "telegram" in i]

    @pytest.mark.parametrize("key, bad", [
        ("roles", ["owner"]),
        ("role_policies", "family"),
        ("push_notifications", "yes please"),
        ("snooze_minutes", 10),
        ("allowed_chat_ids", "123"),
    ])
    def test_wrong_types_are_reported(self, key, bad):
        issues = self._issues({"enabled": True, key: bad})
        assert any(f"telegram.{key}" in i for i in issues)
