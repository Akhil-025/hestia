"""
tests/test_telegram_bot.py

Covers core/telegram_bot.py's HestiaTelegramBot.

python-telegram-bot (telegram / telegram.ext) is faked in conftest.py, so
constructing a bot never touches the real Telegram API. Async handlers are
driven directly with asyncio.run() and hand-built Update/Message mocks.
"""
import asyncio
import subprocess
import sys
import types
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import core.telegram_bot as bot_module


def _make_bot(process_fn=None, allowed_chat_ids=None, stt=None, memory=None, **kwargs):
    return bot_module.HestiaTelegramBot(
        token="fake-token",
        process_fn=process_fn or MagicMock(return_value="a reply"),
        allowed_chat_ids=allowed_chat_ids,
        stt=stt,
        memory=memory,
        **kwargs,
    )


def _make_update(chat_id=1, text=None, has_message=True):
    update = MagicMock()
    update.effective_chat.id = chat_id
    if has_message:
        message = MagicMock()
        message.text = text
        message.reply_text = AsyncMock()
        update.message = message
        update.effective_message = message
    else:
        update.message = None
        update.effective_message = MagicMock()
    return update


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# __init__ / _is_allowed
# ---------------------------------------------------------------------------

class TestInitAndAllowlist:
    def test_registers_all_handlers(self):
        # start, help, text, voice, location, photo, document, callback
        bot = _make_bot()
        assert bot._app.add_handler.call_count == 8

    def test_allows_everyone_when_no_allowlist(self):
        bot = _make_bot(allowed_chat_ids=None)
        assert bot._is_allowed(12345) is True

    def test_allows_everyone_when_empty_allowlist(self):
        bot = _make_bot(allowed_chat_ids=[])
        assert bot._is_allowed(12345) is True

    def test_allows_chat_id_in_allowlist(self):
        bot = _make_bot(allowed_chat_ids=[1, 2, 3])
        assert bot._is_allowed(2) is True

    def test_blocks_chat_id_not_in_allowlist(self):
        bot = _make_bot(allowed_chat_ids=[1, 2, 3])
        assert bot._is_allowed(999) is False


# ---------------------------------------------------------------------------
# _handle_start
# ---------------------------------------------------------------------------

class TestHandleStart:
    def test_replies_with_greeting_when_allowed(self):
        bot = _make_bot()
        update = _make_update(chat_id=1)
        _run(bot._handle_start(update, MagicMock()))
        update.message.reply_text.assert_called_once()
        assert "Hestia" in update.message.reply_text.call_args[0][0]

    def test_replies_unauthorised_when_blocked(self):
        bot = _make_bot(allowed_chat_ids=[42])
        update = _make_update(chat_id=1)
        _run(bot._handle_start(update, MagicMock()))
        update.message.reply_text.assert_called_once_with("Unauthorised.")


# ---------------------------------------------------------------------------
# _handle_text
# ---------------------------------------------------------------------------

class TestHandleText:
    def test_calls_process_fn_and_replies(self):
        process_fn = MagicMock(return_value="Sure, done!")
        bot = _make_bot(process_fn=process_fn)
        update = _make_update(chat_id=1, text="turn on the lights")

        _run(bot._handle_text(update, MagicMock()))

        process_fn.assert_called_once_with("turn on the lights")
        update.message.reply_text.assert_called_once_with("Sure, done!")

    def test_strips_whitespace_before_processing(self):
        process_fn = MagicMock(return_value="ok")
        bot = _make_bot(process_fn=process_fn)
        update = _make_update(chat_id=1, text="  hello  ")

        _run(bot._handle_text(update, MagicMock()))

        process_fn.assert_called_once_with("hello")

    def test_ignores_empty_text(self):
        process_fn = MagicMock()
        bot = _make_bot(process_fn=process_fn)
        update = _make_update(chat_id=1, text="   ")

        _run(bot._handle_text(update, MagicMock()))

        process_fn.assert_not_called()
        update.message.reply_text.assert_not_called()

    def test_does_not_reply_when_process_fn_returns_falsy(self):
        process_fn = MagicMock(return_value="")
        bot = _make_bot(process_fn=process_fn)
        update = _make_update(chat_id=1, text="hi")

        _run(bot._handle_text(update, MagicMock()))

        update.message.reply_text.assert_not_called()

    def test_blocked_chat_id_is_ignored(self):
        process_fn = MagicMock()
        bot = _make_bot(process_fn=process_fn, allowed_chat_ids=[42])
        update = _make_update(chat_id=1, text="hi")

        _run(bot._handle_text(update, MagicMock()))

        process_fn.assert_not_called()


# ---------------------------------------------------------------------------
# _handle_location
# ---------------------------------------------------------------------------

class TestHandleLocation:
    def _update_with_location(self, chat_id=1, lat=1.23, lon=4.56, is_live_ping=False):
        update = MagicMock()
        update.effective_chat.id = chat_id
        message = MagicMock()
        loc = MagicMock()
        loc.latitude = lat
        loc.longitude = lon
        message.location = loc
        message.reply_text = AsyncMock()
        update.effective_message = message
        # Initial share: update.message is set. Live-location ping after the
        # first one: update.message is None (only effective_message is set).
        update.message = None if is_live_ping else message
        return update, message

    def test_saves_location_and_confirms_on_initial_share(self):
        memory = MagicMock()
        bot = _make_bot(memory=memory)
        update, message = self._update_with_location()

        _run(bot._handle_location(update, MagicMock()))

        memory.set_device_location.assert_called_once_with(1.23, 4.56, source="telegram")
        message.reply_text.assert_called_once_with("Got it — location saved.")

    def test_saves_location_silently_on_live_ping(self):
        memory = MagicMock()
        bot = _make_bot(memory=memory)
        update, message = self._update_with_location(is_live_ping=True)

        _run(bot._handle_location(update, MagicMock()))

        memory.set_device_location.assert_called_once_with(1.23, 4.56, source="telegram")
        message.reply_text.assert_not_called()

    def test_replies_gracefully_when_no_memory_configured(self):
        bot = _make_bot(memory=None)
        update, message = self._update_with_location()

        _run(bot._handle_location(update, MagicMock()))

        message.reply_text.assert_called_once_with(
            "Got your location, but I'm not able to save it right now."
        )

    def test_replies_error_when_save_raises_on_initial_share(self):
        memory = MagicMock()
        memory.set_device_location.side_effect = Exception("db down")
        bot = _make_bot(memory=memory)
        update, message = self._update_with_location()

        _run(bot._handle_location(update, MagicMock()))

        message.reply_text.assert_called_once_with("I couldn't save that location.")

    def test_silently_swallows_save_error_on_live_ping(self):
        memory = MagicMock()
        memory.set_device_location.side_effect = Exception("db down")
        bot = _make_bot(memory=memory)
        update, message = self._update_with_location(is_live_ping=True)

        _run(bot._handle_location(update, MagicMock()))

        message.reply_text.assert_not_called()

    def test_ignores_blocked_chat_id(self):
        memory = MagicMock()
        bot = _make_bot(memory=memory, allowed_chat_ids=[42])
        update, message = self._update_with_location(chat_id=1)

        _run(bot._handle_location(update, MagicMock()))

        memory.set_device_location.assert_not_called()

    def test_no_op_when_message_has_no_location(self):
        memory = MagicMock()
        bot = _make_bot(memory=memory)
        update = MagicMock()
        update.effective_chat.id = 1
        update.effective_message.location = None

        _run(bot._handle_location(update, MagicMock()))

        memory.set_device_location.assert_not_called()


# ---------------------------------------------------------------------------
# _handle_voice
# ---------------------------------------------------------------------------

class TestHandleVoice:
    def _update_with_voice(self, chat_id=1):
        update = MagicMock()
        update.effective_chat.id = chat_id
        message = MagicMock()
        message.reply_text = AsyncMock()
        voice_file = MagicMock()
        voice_file.download_to_drive = AsyncMock()
        message.voice.get_file = AsyncMock(return_value=voice_file)
        update.message = message
        return update, message

    def test_replies_unsupported_when_no_stt_configured(self):
        bot = _make_bot(stt=None)
        update, message = self._update_with_voice()

        _run(bot._handle_voice(update, MagicMock()))

        message.reply_text.assert_called_once_with("Voice notes aren't supported in this mode.")

    def test_full_pipeline_transcribes_and_replies(self, monkeypatch, tmp_path):
        stt = MagicMock()
        stt._transcribe.return_value = "turn off the lights"
        process_fn = MagicMock(return_value="Done.")
        bot = _make_bot(stt=stt, process_fn=process_fn)
        update, message = self._update_with_voice()

        monkeypatch.setattr(
            bot_module.tempfile,
            "NamedTemporaryFile",
            lambda suffix=None, delete=None: _FakeTempFile(tmp_path / "voice.ogg"),
        )
        monkeypatch.setattr(subprocess, "run", MagicMock())
        monkeypatch.setattr(bot_module.os, "unlink", MagicMock())

        wav_frames = (np.array([100, -100, 200], dtype=np.int16)).tobytes()
        fake_wave = MagicMock()
        fake_wave.__enter__.return_value.readframes.return_value = wav_frames
        monkeypatch.setattr(wave, "open", MagicMock(return_value=fake_wave))

        _run(bot._handle_voice(update, MagicMock()))

        stt._transcribe.assert_called_once()
        process_fn.assert_called_once_with("turn off the lights")
        calls = [c.args[0] for c in message.reply_text.await_args_list]
        assert "Heard: turn off the lights" in calls
        assert "Done." in calls

    def test_short_transcription_reports_could_not_understand(self, monkeypatch, tmp_path):
        stt = MagicMock()
        stt._transcribe.return_value = "a"
        bot = _make_bot(stt=stt)
        update, message = self._update_with_voice()

        monkeypatch.setattr(
            bot_module.tempfile,
            "NamedTemporaryFile",
            lambda suffix=None, delete=None: _FakeTempFile(tmp_path / "voice.ogg"),
        )
        monkeypatch.setattr(subprocess, "run", MagicMock())
        monkeypatch.setattr(bot_module.os, "unlink", MagicMock())
        fake_wave = MagicMock()
        fake_wave.__enter__.return_value.readframes.return_value = b"\x00\x00"
        monkeypatch.setattr(wave, "open", MagicMock(return_value=fake_wave))

        _run(bot._handle_voice(update, MagicMock()))

        message.reply_text.assert_called_once_with("I couldn't make out that voice note.")

    def test_ffmpeg_failure_reports_friendly_message(self, monkeypatch, tmp_path):
        stt = MagicMock()
        bot = _make_bot(stt=stt)
        update, message = self._update_with_voice()

        monkeypatch.setattr(
            bot_module.tempfile,
            "NamedTemporaryFile",
            lambda suffix=None, delete=None: _FakeTempFile(tmp_path / "voice.ogg"),
        )
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(side_effect=subprocess.CalledProcessError(1, "ffmpeg")),
        )

        _run(bot._handle_voice(update, MagicMock()))

        message.reply_text.assert_called_once_with(
            "I couldn't process that audio file. Is ffmpeg installed?"
        )

    def test_generic_exception_after_ffmpeg_stage_reports_friendly_message(self, monkeypatch, tmp_path):
        """Covers the generic `except Exception` fallback for a failure that
        happens *after* `import subprocess` has already run inside the
        handler (e.g. wave-file parsing blows up). See
        test_early_exception_before_subprocess_import_is_a_real_bug below
        for the case where the failure happens *before* that import."""
        stt = MagicMock()
        bot = _make_bot(stt=stt)
        update, message = self._update_with_voice()

        monkeypatch.setattr(
            bot_module.tempfile,
            "NamedTemporaryFile",
            lambda suffix=None, delete=None: _FakeTempFile(tmp_path / "voice.ogg"),
        )
        monkeypatch.setattr(subprocess, "run", MagicMock())
        monkeypatch.setattr(wave, "open", MagicMock(side_effect=Exception("corrupt wav")))

        _run(bot._handle_voice(update, MagicMock()))

        message.reply_text.assert_called_once_with(
            "Something went wrong processing that voice note."
        )

    def test_early_exception_before_ffmpeg_stage_reports_friendly_message(self):
        """Regression for a bug this file used to document: `import subprocess`
        lived inside the try block, so a failure in get_file()/download (a
        network blip) made `except subprocess.CalledProcessError` itself raise
        UnboundLocalError. The imports are module-level now, so the handler
        falls through to the friendly generic message."""
        stt = MagicMock()
        bot = _make_bot(stt=stt)
        update, message = self._update_with_voice()
        message.voice.get_file = AsyncMock(side_effect=Exception("network blip"))

        _run(bot._handle_voice(update, MagicMock()))

        message.reply_text.assert_called_once_with(
            "Something went wrong processing that voice note."
        )

    def test_temp_files_are_removed_even_when_ffmpeg_fails(self, monkeypatch, tmp_path):
        stt = MagicMock()
        bot = _make_bot(stt=stt)
        update, message = self._update_with_voice()
        ogg = tmp_path / "voice.ogg"
        wav = tmp_path / "voice.wav"
        ogg.write_bytes(b"x")
        wav.write_bytes(b"x")
        monkeypatch.setattr(
            bot_module.tempfile, "NamedTemporaryFile",
            lambda suffix=None, delete=None: _FakeTempFile(ogg),
        )
        monkeypatch.setattr(
            subprocess, "run",
            MagicMock(side_effect=subprocess.CalledProcessError(1, "ffmpeg")),
        )

        _run(bot._handle_voice(update, MagicMock()))

        assert not ogg.exists()
        assert not wav.exists()

    def test_transcript_goes_through_the_role_check(self, monkeypatch, tmp_path):
        """A voice note from a restricted chat is classified like typed text."""
        stt = MagicMock()
        stt._transcribe.return_value = "delete all my notes"
        process_fn = MagicMock(return_value="Deleted.")
        bot = _make_bot(
            stt=stt, process_fn=process_fn,
            roles={7: "family"},
            role_policies={"family": {"allow_modules": ["chronos"]}},
            classify_fn=lambda t: "delete_notes",
        )
        update, message = self._update_with_voice(chat_id=7)
        monkeypatch.setattr(
            bot_module.tempfile, "NamedTemporaryFile",
            lambda suffix=None, delete=None: _FakeTempFile(tmp_path / "voice.ogg"),
        )
        monkeypatch.setattr(subprocess, "run", MagicMock())
        fake_wave = MagicMock()
        fake_wave.__enter__.return_value.readframes.return_value = b"\x00\x00"
        monkeypatch.setattr(wave, "open", MagicMock(return_value=fake_wave))

        _run(bot._handle_voice(update, MagicMock()))

        process_fn.assert_not_called()
        assert message.reply_text.await_args_list[-1].args[0] == bot_module._DENIED

    def test_ignores_blocked_chat_id(self):
        stt = MagicMock()
        bot = _make_bot(stt=stt, allowed_chat_ids=[42])
        update, message = self._update_with_voice(chat_id=1)

        _run(bot._handle_voice(update, MagicMock()))

        message.reply_text.assert_not_called()


class _FakeTempFile:
    """Stand-in for tempfile.NamedTemporaryFile(delete=False) used as a
    context manager that exposes a `.name` path."""

    def __init__(self, path):
        self.name = str(path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# start() / stop()
# ---------------------------------------------------------------------------

class TestStartStop:
    def test_start_launches_background_daemon_thread(self, monkeypatch):
        bot = _make_bot()
        started = {}

        class _FakeThread:
            def __init__(self, target, daemon, name):
                started["target"] = target
                started["daemon"] = daemon

            def start(self):
                started["started"] = True

        monkeypatch.setattr(bot_module.threading, "Thread", _FakeThread)
        bot.start()
        assert started["daemon"] is True
        assert started["started"] is True

    def test_stop_calls_stop_running_when_app_running(self):
        bot = _make_bot()
        bot._app.running = True
        bot.stop()
        bot._app.stop_running.assert_called_once()

    def test_stop_is_noop_when_app_not_running(self):
        bot = _make_bot()
        bot._app.running = False
        bot.stop()
        bot._app.stop_running.assert_not_called()


# ===========================================================================
# Backlog section 18 (#191-#196)
# ===========================================================================

import threading
import time
from pathlib import Path

from core.telegram_bot import (
    RolePolicy,
    build_help_text,
    looks_like_reminder,
    _split_message,
)


def _buttons(markup):
    """[(text, callback_data), ...] flattened from an inline keyboard."""
    return [(b.text, b.callback_data) for row in markup.inline_keyboard for b in row]


def _callback_update(data, chat_id=1):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.send_action = AsyncMock()
    query = MagicMock()
    query.data = data
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    query.message.reply_text = AsyncMock()
    update.callback_query = query
    return update, query


# ---------------------------------------------------------------------------
# #191 inline buttons
# ---------------------------------------------------------------------------

class TestConfirmButtons:
    def test_confirmation_prompt_gets_confirm_and_cancel_buttons(self):
        process_fn = MagicMock(return_value='Send "hi" to Sam? Say yes to send it.')
        bot = _make_bot(process_fn=process_fn)
        update = _make_update(text="email sam hi")

        _run(bot._handle_text(update, MagicMock()))

        markup = update.message.reply_text.call_args.kwargs["reply_markup"]
        assert _buttons(markup) == [("✅ Confirm", "cf:yes"), ("❌ Cancel", "cf:no")]

    def test_plain_reply_has_no_buttons(self):
        bot = _make_bot(process_fn=MagicMock(return_value="It's 9:41."))
        update = _make_update(text="what time is it")

        _run(bot._handle_text(update, MagicMock()))

        update.message.reply_text.assert_called_once_with("It's 9:41.")

    def test_pending_fn_decides_when_available(self):
        """With pending_fn the buttons track real orchestrator state, not wording."""
        state = {"pending": False}

        def process(text):
            state["pending"] = text == "forget my name"
            return "Okay."          # no "say yes" wording at all

        bot = _make_bot(process_fn=process)
        bot.pending_fn = lambda: state["pending"]
        update = _make_update(text="forget my name")
        _run(bot._handle_text(update, MagicMock()))
        assert "reply_markup" in update.message.reply_text.call_args.kwargs

        update2 = _make_update(text="what time is it")
        _run(bot._handle_text(update2, MagicMock()))
        update2.message.reply_text.assert_called_once_with("Okay.")

    def test_pending_fn_ignores_wording_when_nothing_is_pending(self):
        bot = _make_bot(process_fn=MagicMock(return_value="Say yes to anything."))
        bot.pending_fn = lambda: False
        update = _make_update(text="x")
        _run(bot._handle_text(update, MagicMock()))
        update.message.reply_text.assert_called_once_with("Say yes to anything.")

    def test_confirm_tap_sends_yes_through_process_fn_and_clears_buttons(self):
        process_fn = MagicMock(return_value="Sent.")
        bot = _make_bot(process_fn=process_fn)
        update, query = _callback_update("cf:yes")

        _run(bot._handle_callback(update, MagicMock()))

        process_fn.assert_called_once_with("yes")
        query.answer.assert_awaited_once()
        query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)
        query.message.reply_text.assert_awaited_once_with("Sent.")

    def test_cancel_tap_sends_no(self):
        process_fn = MagicMock(return_value="Okay, I won't send it.")
        bot = _make_bot(process_fn=process_fn)
        update, query = _callback_update("cf:no")
        _run(bot._handle_callback(update, MagicMock()))
        process_fn.assert_called_once_with("no")

    def test_callback_from_blocked_chat_does_nothing(self):
        process_fn = MagicMock()
        bot = _make_bot(process_fn=process_fn, allowed_chat_ids=[42])
        update, query = _callback_update("cf:yes", chat_id=1)
        _run(bot._handle_callback(update, MagicMock()))
        process_fn.assert_not_called()
        query.message.reply_text.assert_not_called()

    def test_garbage_callback_data_is_ignored(self):
        process_fn = MagicMock()
        bot = _make_bot(process_fn=process_fn)
        for data in ("", "cf:maybe", "sz:abc", "zz:1", "nonsense"):
            update, query = _callback_update(data)
            _run(bot._handle_callback(update, MagicMock()))
        process_fn.assert_not_called()

    def test_callback_without_a_message_is_ignored(self):
        bot = _make_bot()
        update, query = _callback_update("cf:yes")
        query.message = None
        _run(bot._handle_callback(update, MagicMock()))   # must not raise

    def test_done_tap_only_clears_buttons(self):
        process_fn = MagicMock()
        bot = _make_bot(process_fn=process_fn)
        update, query = _callback_update("ok:0")
        _run(bot._handle_callback(update, MagicMock()))
        process_fn.assert_not_called()
        query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)

    def test_failure_clearing_buttons_does_not_stop_the_action(self):
        process_fn = MagicMock(return_value="Sent.")
        bot = _make_bot(process_fn=process_fn)
        update, query = _callback_update("cf:yes")
        query.edit_message_reply_markup = AsyncMock(side_effect=Exception("message is not modified"))
        _run(bot._handle_callback(update, MagicMock()))
        process_fn.assert_called_once_with("yes")


class TestSnoozeButtons:
    def test_snooze_tap_uses_snooze_fn(self):
        snooze_fn = MagicMock(return_value="Snoozed for 10 minutes.")
        process_fn = MagicMock()
        bot = _make_bot(process_fn=process_fn)
        bot.snooze_fn = snooze_fn
        update, query = _callback_update("sz:10")

        _run(bot._handle_callback(update, MagicMock()))

        snooze_fn.assert_called_once_with(10)
        process_fn.assert_not_called()
        query.message.reply_text.assert_awaited_once_with("Snoozed for 10 minutes.")
        query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)

    def test_snooze_falls_back_to_process_fn(self):
        process_fn = MagicMock(return_value="Snoozed.")
        bot = _make_bot(process_fn=process_fn)
        update, query = _callback_update("sz:60")
        _run(bot._handle_callback(update, MagicMock()))
        process_fn.assert_called_once_with("snooze for 60 minutes")

    def test_snooze_failure_reports_friendly_message(self):
        bot = _make_bot()
        bot.snooze_fn = MagicMock(side_effect=RuntimeError("db locked"))
        update, query = _callback_update("sz:10")
        _run(bot._handle_callback(update, MagicMock()))
        query.message.reply_text.assert_awaited_once_with("I couldn't snooze that reminder.")

    def test_restricted_role_needs_chronos_to_snooze(self):
        bot = _make_bot(
            roles={7: "kid"},
            role_policies={"kid": {"allow_modules": ["dionysus"]}},
        )
        bot.snooze_fn = MagicMock(return_value="Snoozed.")
        update, query = _callback_update("sz:10", chat_id=7)
        _run(bot._handle_callback(update, MagicMock()))
        bot.snooze_fn.assert_not_called()
        query.message.reply_text.assert_awaited_once_with(bot_module._DENIED)


class TestPush:
    def _post(self, monkeypatch, ok=True):
        import requests
        post = MagicMock(return_value=MagicMock(ok=ok))
        monkeypatch.setattr(requests, "post", post)
        return post

    def test_reminder_push_has_snooze_and_done_buttons(self, monkeypatch):
        post = self._post(monkeypatch)
        bot = _make_bot(allowed_chat_ids=[5])

        sent = bot.push("Reminder: call the dentist")

        assert sent == 1
        payload = post.call_args.kwargs["json"]
        assert payload["chat_id"] == 5
        keyboard = payload["reply_markup"]["inline_keyboard"][0]
        assert [b["callback_data"] for b in keyboard] == ["sz:10", "sz:60", "ok:0"]
        assert "bot" + "fake-token" in post.call_args.args[0]

    def test_non_reminder_push_has_no_buttons(self, monkeypatch):
        post = self._post(monkeypatch)
        bot = _make_bot(allowed_chat_ids=[5])
        bot.push("Good morning. It's sunny.")
        assert "reply_markup" not in post.call_args.kwargs["json"]

    def test_reminder_flag_overrides_detection(self, monkeypatch):
        post = self._post(monkeypatch)
        bot = _make_bot(allowed_chat_ids=[5])
        bot.push("Time to stretch", reminder=True)
        assert "reply_markup" in post.call_args.kwargs["json"]

    def test_custom_snooze_lengths(self, monkeypatch):
        post = self._post(monkeypatch)
        bot = _make_bot(allowed_chat_ids=[5])
        bot.snooze_minutes = (5, 120)
        bot.push("Reminder: x")
        keyboard = post.call_args.kwargs["json"]["reply_markup"]["inline_keyboard"][0]
        assert [b["text"] for b in keyboard] == ["⏰ 5m", "⏰ 2h", "✔️ Done"]

    def test_only_unrestricted_chats_are_pushed_to(self, monkeypatch):
        post = self._post(monkeypatch)
        bot = _make_bot(
            allowed_chat_ids=[1],
            roles={2: "family"},
            role_policies={"family": {"allow_modules": ["core"]}},
        )
        bot.push("Reminder: private thing")
        assert [c.kwargs["json"]["chat_id"] for c in post.call_args_list] == [1]

    def test_no_targets_when_allowlist_is_open(self, monkeypatch):
        post = self._post(monkeypatch)
        assert _make_bot(allowed_chat_ids=None).push("hello") == 0
        post.assert_not_called()

    def test_dnd_suppresses_push(self, monkeypatch):
        post = self._post(monkeypatch)
        bot = _make_bot(allowed_chat_ids=[1])
        bot.should_push_fn = lambda: False
        assert bot.push("Reminder: x") == 0
        post.assert_not_called()

    def test_failed_send_is_counted_and_never_logs_the_token(self, monkeypatch, caplog):
        import requests
        monkeypatch.setattr(requests, "post", MagicMock(side_effect=requests.ConnectionError("boom fake-token")))
        bot = _make_bot(allowed_chat_ids=[1])
        with caplog.at_level("ERROR"):
            assert bot.push("hello") == 0
        assert "fake-token" not in caplog.text

    def test_http_error_counts_as_not_sent(self, monkeypatch):
        self._post(monkeypatch, ok=False)
        assert _make_bot(allowed_chat_ids=[1]).push("hello") == 0

    def test_empty_text_is_not_sent(self, monkeypatch):
        post = self._post(monkeypatch)
        assert _make_bot(allowed_chat_ids=[1]).push("") == 0
        post.assert_not_called()

    def test_long_text_is_split_with_buttons_on_the_last_part(self, monkeypatch):
        post = self._post(monkeypatch)
        bot = _make_bot(allowed_chat_ids=[1])
        bot.push("Reminder: " + ("word " * 2000))
        calls = [c.kwargs["json"] for c in post.call_args_list]
        assert len(calls) >= 2
        assert all(len(c["text"]) <= 4096 for c in calls)
        assert "reply_markup" not in calls[0]
        assert "reply_markup" in calls[-1]

    def test_event_bus_speak_events_are_mirrored(self, monkeypatch):
        post = self._post(monkeypatch)
        bot = _make_bot(allowed_chat_ids=[1])
        bus = MagicMock()
        bot.attach_event_bus(bus)
        event, callback = bus.on.call_args.args
        assert event == "speak"
        callback({"text": "Reminder: stand up"})
        assert post.call_args.kwargs["json"]["text"] == "Reminder: stand up"

    def test_speak_mirror_swallows_errors(self, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "post", MagicMock(side_effect=RuntimeError("x")))
        bot = _make_bot(allowed_chat_ids=[1])
        bus = MagicMock()
        bot.attach_event_bus(bus)
        bus.on.call_args.args[1]({"text": "hello"})    # must not raise

    @pytest.mark.parametrize("text, expected", [
        ("Reminder: call mum", True),
        ("You've arrived at the shop. Reminder: buy milk", True),
        ("You missed a reminder from 9am: call mum.", False),
        ("Good morning!", False),
        ("", False),
    ])
    def test_looks_like_reminder(self, text, expected):
        assert looks_like_reminder(text) is expected


# ---------------------------------------------------------------------------
# Confirmations belong to the chat that raised them
# ---------------------------------------------------------------------------

class TestConfirmationOwnership:
    def _bot(self, pending, **kw):
        bot = _make_bot(**kw)
        bot.pending_fn = lambda: pending["on"]
        return bot

    def test_another_chat_cannot_answer_a_confirmation(self):
        pending = {"on": False}
        calls = []

        def process(text):
            calls.append(text)
            if text == "delete my notes":
                pending["on"] = True
                return "Delete everything? Say yes to confirm."
            return "Deleted."

        bot = self._bot(pending, process_fn=process, allowed_chat_ids=[1, 2])
        _run(bot._handle_text(_make_update(chat_id=1, text="delete my notes"), MagicMock()))
        intruder = _make_update(chat_id=2, text="yes")
        _run(bot._handle_text(intruder, MagicMock()))

        assert calls == ["delete my notes"]
        intruder.message.reply_text.assert_called_once_with(bot_module._FOREIGN_CONFIRMATION)

    def test_the_originating_chat_can_answer(self):
        pending = {"on": False}

        def process(text):
            if text == "delete my notes":
                pending["on"] = True
                return "Sure? Say yes."
            pending["on"] = False
            return "Deleted."

        bot = self._bot(pending, process_fn=process, allowed_chat_ids=[1, 2])
        _run(bot._handle_text(_make_update(chat_id=1, text="delete my notes"), MagicMock()))
        update, query = _callback_update("cf:yes", chat_id=1)
        _run(bot._handle_callback(update, MagicMock()))
        query.message.reply_text.assert_awaited_once_with("Deleted.")

    def test_a_confirm_tap_from_another_chat_is_refused(self):
        pending = {"on": False}

        def process(text):
            pending["on"] = True
            return "Sure? Say yes."

        bot = self._bot(pending, process_fn=process, allowed_chat_ids=[1, 2])
        _run(bot._handle_text(_make_update(chat_id=1, text="x"), MagicMock()))
        process_fn = MagicMock()
        bot.process_fn = process_fn
        update, query = _callback_update("cf:yes", chat_id=2)
        _run(bot._handle_callback(update, MagicMock()))
        process_fn.assert_not_called()
        query.message.reply_text.assert_awaited_once_with(bot_module._FOREIGN_CONFIRMATION)

    def test_restricted_chat_cannot_answer_a_confirmation_raised_elsewhere(self):
        """Raised from the desktop (owner is None), answered from a restricted chat."""
        process_fn = MagicMock(return_value="Sent.")
        bot = _make_bot(
            process_fn=process_fn,
            roles={7: "family"},
            role_policies={"family": {"allow_modules": ["core"]}},
            classify_fn=lambda t: "chat",
        )
        bot.pending_fn = lambda: True
        update = _make_update(chat_id=7, text="yes")
        _run(bot._handle_text(update, MagicMock()))
        process_fn.assert_not_called()
        update.message.reply_text.assert_called_once_with(bot_module._FOREIGN_CONFIRMATION)

    def test_owner_may_answer_a_confirmation_raised_elsewhere(self):
        process_fn = MagicMock(return_value="Sent.")
        bot = _make_bot(process_fn=process_fn)
        bot.pending_fn = lambda: True
        update = _make_update(chat_id=1, text="yes")
        _run(bot._handle_text(update, MagicMock()))
        process_fn.assert_called_once_with("yes")

    def test_a_broken_pending_fn_is_treated_as_nothing_pending(self):
        process_fn = MagicMock(return_value="ok")
        bot = _make_bot(process_fn=process_fn)
        bot.pending_fn = MagicMock(side_effect=RuntimeError("x"))
        _run(bot._handle_text(_make_update(text="hello"), MagicMock()))
        process_fn.assert_called_once_with("hello")


# ---------------------------------------------------------------------------
# #192 photo / PDF handling
# ---------------------------------------------------------------------------

def _file_update(chat_id=1, *, photo=None, document=None, caption=None):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.send_action = AsyncMock()
    msg = MagicMock()
    msg.reply_text = AsyncMock()
    msg.caption = caption
    msg.photo = photo
    msg.document = document
    update.message = msg
    update.effective_message = msg
    return update, msg


def _tg_file(content=b"data"):
    f = MagicMock()

    async def _download(path):
        Path(path).write_bytes(content)

    f.download_to_drive = AsyncMock(side_effect=_download)
    return f


def _photo_size(unique="abc", size=1000):
    p = MagicMock()
    p.file_unique_id = unique
    p.file_size = size
    p.get_file = AsyncMock(return_value=_tg_file(b"jpegbytes"))
    return p


def _document(name, mime, size=1000):
    d = MagicMock()
    d.file_name = name
    d.mime_type = mime
    d.file_size = size
    d.get_file = AsyncMock(return_value=_tg_file(b"%PDF-1.4"))
    return d


class TestPhotoHandling:
    def test_photo_is_downloaded_and_handed_to_the_iris_hook(self):
        seen = {}

        def hook(path, filename):
            seen["bytes"] = Path(path).read_bytes()
            seen["filename"] = filename
            seen["dir"] = str(Path(path).parent)
            return "Saved to your photo library."

        bot = _make_bot()
        bot.ingest_photo_fn = hook
        update, msg = _file_update(photo=[_photo_size("small"), _photo_size("big")])

        _run(bot._handle_photo(update, MagicMock()))

        assert seen["bytes"] == b"jpegbytes"
        assert seen["filename"] == "telegram_big.jpg"      # largest size wins
        msg.reply_text.assert_awaited_once_with("Saved to your photo library.")
        assert not Path(seen["dir"]).exists()               # temp dir cleaned up

    def test_the_largest_photo_size_is_the_one_downloaded(self):
        small, big = _photo_size("small"), _photo_size("big")
        bot = _make_bot()
        bot.ingest_photo_fn = lambda p, f: "ok"
        update, msg = _file_update(photo=[small, big])
        _run(bot._handle_photo(update, MagicMock()))
        big.get_file.assert_awaited_once()
        small.get_file.assert_not_called()

    def test_no_hook_means_a_clear_message(self):
        bot = _make_bot()
        update, msg = _file_update(photo=[_photo_size()])
        _run(bot._handle_photo(update, MagicMock()))
        msg.reply_text.assert_awaited_once_with("I can't file photos in this setup.")

    def test_hook_failure_reports_friendly_message_and_cleans_up(self):
        dirs = []

        def hook(path, filename):
            dirs.append(Path(path).parent)
            raise RuntimeError("iris exploded")

        bot = _make_bot()
        bot.ingest_photo_fn = hook
        update, msg = _file_update(photo=[_photo_size()])
        _run(bot._handle_photo(update, MagicMock()))
        msg.reply_text.assert_awaited_once_with("Something went wrong filing that photo.")
        assert not dirs[0].exists()

    def test_download_failure_reports_friendly_message(self):
        photo = _photo_size()
        photo.get_file = AsyncMock(side_effect=Exception("network blip"))
        bot = _make_bot()
        bot.ingest_photo_fn = MagicMock()
        update, msg = _file_update(photo=[photo])
        _run(bot._handle_photo(update, MagicMock()))
        bot.ingest_photo_fn.assert_not_called()
        msg.reply_text.assert_awaited_once_with("Something went wrong filing that photo.")

    def test_oversized_photo_is_declined_without_downloading(self):
        photo = _photo_size(size=50 * 1024 * 1024)
        bot = _make_bot()
        bot.ingest_photo_fn = MagicMock()
        update, msg = _file_update(photo=[photo])
        _run(bot._handle_photo(update, MagicMock()))
        photo.get_file.assert_not_called()
        assert "too large" in msg.reply_text.call_args.args[0]

    def test_blocked_chat_is_ignored(self):
        bot = _make_bot(allowed_chat_ids=[42])
        bot.ingest_photo_fn = MagicMock()
        update, msg = _file_update(chat_id=1, photo=[_photo_size()])
        _run(bot._handle_photo(update, MagicMock()))
        bot.ingest_photo_fn.assert_not_called()
        msg.reply_text.assert_not_called()

    def test_empty_photo_list_is_ignored(self):
        bot = _make_bot()
        bot.ingest_photo_fn = MagicMock()
        update, msg = _file_update(photo=[])
        _run(bot._handle_photo(update, MagicMock()))
        bot.ingest_photo_fn.assert_not_called()

    def test_restricted_role_without_iris_is_refused(self):
        bot = _make_bot(roles={7: "family"}, role_policies={"family": {"allow_modules": ["chronos"]}})
        bot.ingest_photo_fn = MagicMock()
        update, msg = _file_update(chat_id=7, photo=[_photo_size()])
        _run(bot._handle_photo(update, MagicMock()))
        bot.ingest_photo_fn.assert_not_called()
        msg.reply_text.assert_awaited_once_with(bot_module._DENIED)

    def test_restricted_role_with_iris_is_allowed(self):
        bot = _make_bot(roles={7: "family"}, role_policies={"family": {"allow_modules": ["iris"]}})
        bot.ingest_photo_fn = MagicMock(return_value="Saved.")
        update, msg = _file_update(chat_id=7, photo=[_photo_size()])
        _run(bot._handle_photo(update, MagicMock()))
        bot.ingest_photo_fn.assert_called_once()


class TestDocumentHandling:
    def test_pdf_goes_to_the_athena_hook_with_its_own_filename(self):
        seen = {}

        def hook(path, filename):
            seen["name"] = Path(path).name
            seen["filename"] = filename
            seen["bytes"] = Path(path).read_bytes()
            return "Added paper.pdf in your documents."

        bot = _make_bot()
        bot.ingest_document_fn = hook
        bot.ingest_photo_fn = MagicMock()
        update, msg = _file_update(document=_document("paper.pdf", "application/pdf"))

        _run(bot._handle_document(update, MagicMock()))

        assert seen == {"name": "paper.pdf", "filename": "paper.pdf", "bytes": b"%PDF-1.4"}
        bot.ingest_photo_fn.assert_not_called()
        msg.reply_text.assert_awaited_once_with("Added paper.pdf in your documents.")

    def test_pdf_detected_by_mime_gets_a_pdf_extension(self):
        seen = {}
        bot = _make_bot()
        bot.ingest_document_fn = lambda p, f: seen.setdefault("f", f) and "ok"
        update, msg = _file_update(document=_document("scan", "application/pdf"))
        _run(bot._handle_document(update, MagicMock()))
        assert seen["f"] == "scan.pdf"

    def test_image_sent_as_a_file_goes_to_iris(self):
        bot = _make_bot()
        bot.ingest_photo_fn = MagicMock(return_value="Saved.")
        bot.ingest_document_fn = MagicMock()
        update, msg = _file_update(document=_document("IMG_1.png", "image/png"))
        _run(bot._handle_document(update, MagicMock()))
        bot.ingest_photo_fn.assert_called_once()
        bot.ingest_document_fn.assert_not_called()

    def test_unsupported_file_type_is_declined(self):
        bot = _make_bot()
        bot.ingest_document_fn = MagicMock()
        bot.ingest_photo_fn = MagicMock()
        update, msg = _file_update(document=_document("setup.exe", "application/octet-stream"))
        _run(bot._handle_document(update, MagicMock()))
        bot.ingest_document_fn.assert_not_called()
        bot.ingest_photo_fn.assert_not_called()
        msg.reply_text.assert_awaited_once_with("I can only file photos, images and PDFs.")

    def test_hostile_filename_cannot_escape_the_temp_dir(self):
        seen = {}

        def hook(path, filename):
            seen["path"] = path
            seen["filename"] = filename
            return "ok"

        bot = _make_bot()
        bot.ingest_document_fn = hook
        update, msg = _file_update(document=_document("../../etc/evil.pdf", "application/pdf"))
        _run(bot._handle_document(update, MagicMock()))
        assert seen["filename"] == "evil.pdf"
        assert ".." not in seen["path"]

    def test_oversized_document_is_declined(self):
        doc = _document("big.pdf", "application/pdf", size=99 * 1024 * 1024)
        bot = _make_bot()
        bot.ingest_document_fn = MagicMock()
        update, msg = _file_update(document=doc)
        _run(bot._handle_document(update, MagicMock()))
        doc.get_file.assert_not_called()
        assert "too large" in msg.reply_text.call_args.args[0]

    def test_restricted_role_without_athena_is_refused(self):
        bot = _make_bot(roles={7: "family"}, role_policies={"family": {"allow_modules": ["iris"]}})
        bot.ingest_document_fn = MagicMock()
        update, msg = _file_update(chat_id=7, document=_document("a.pdf", "application/pdf"))
        _run(bot._handle_document(update, MagicMock()))
        bot.ingest_document_fn.assert_not_called()
        msg.reply_text.assert_awaited_once_with(bot_module._DENIED)

    def test_no_document_is_ignored(self):
        bot = _make_bot()
        update, msg = _file_update(document=None)
        _run(bot._handle_document(update, MagicMock()))
        msg.reply_text.assert_not_called()


# ---------------------------------------------------------------------------
# #193 /help
# ---------------------------------------------------------------------------

class TestHelp:
    def test_help_lists_every_registered_module_except_chat(self):
        from modules.hecate.intent_registry import INTENT_MODULE_MAP
        text = build_help_text()
        for module in set(INTENT_MODULE_MAP.values()):
            label = "General" if module == "core" else module.capitalize()
            assert f"{label}:" in text, module
        assert len(text) <= 4096

    def test_help_is_derived_from_the_registry(self, monkeypatch):
        import modules.hecate.intent_registry as reg
        monkeypatch.setattr(reg, "INTENT_MODULE_MAP", {"chat": "core", "set_reminder": "chronos", "zap_lights": "hearth"})
        text = build_help_text()
        assert "set reminder" in text
        assert "zap lights" in text
        assert "Hearth:" in text

    def test_help_hides_what_a_role_cannot_use(self):
        policy = RolePolicy.from_config({"allow_modules": ["chronos"], "deny_intents": ["cancel_reminder"]})
        text = build_help_text(policy)
        assert "Chronos:" in text
        assert "Pluto:" not in text
        assert "cancel reminder" not in text
        assert "set reminder" in text

    def test_help_for_one_section_lists_everything_in_it(self):
        from modules.hecate.intent_registry import INTENT_MODULE_MAP
        n = sum(1 for k, v in INTENT_MODULE_MAP.items() if v == "chronos")
        text = build_help_text(module="Chronos")
        assert f"{n} things" in text
        assert "snooze reminder" in text

    def test_general_is_an_alias_for_core(self):
        assert "General" in build_help_text(module="general")

    def test_unknown_section_lists_the_valid_ones(self):
        text = build_help_text(module="nonsense")
        assert "nonsense" in text and "chronos" in text

    def test_help_for_a_role_with_nothing_allowed(self):
        assert "nothing" in build_help_text(RolePolicy(allow_modules=frozenset())).lower()

    def test_handler_replies_and_passes_the_section_argument(self):
        bot = _make_bot()
        update = _make_update(text="/help chronos")
        context = MagicMock()
        context.args = ["chronos"]
        _run(bot._handle_help(update, context))
        assert "Chronos" in update.message.reply_text.call_args.args[0]
        assert "things you can ask" in update.message.reply_text.call_args.args[0]

    def test_handler_without_arguments_gives_the_overview(self):
        bot = _make_bot()
        update = _make_update(text="/help")
        context = MagicMock()
        context.args = []
        _run(bot._handle_help(update, context))
        assert "/help <section>" in update.message.reply_text.call_args.args[0]

    def test_handler_is_role_filtered(self):
        bot = _make_bot(roles={7: "family"}, role_policies={"family": {"allow_modules": ["chronos"]}})
        update = _make_update(chat_id=7, text="/help")
        context = MagicMock()
        context.args = []
        _run(bot._handle_help(update, context))
        reply = update.message.reply_text.call_args.args[0]
        assert "Chronos:" in reply and "Pluto:" not in reply

    def test_blocked_chat_is_unauthorised(self):
        bot = _make_bot(allowed_chat_ids=[42])
        update = _make_update(chat_id=1, text="/help")
        _run(bot._handle_help(update, MagicMock()))
        update.message.reply_text.assert_called_once_with("Unauthorised.")


# ---------------------------------------------------------------------------
# #194 roles
# ---------------------------------------------------------------------------

_FAMILY = {"family": {"allow_modules": ["core", "chronos"], "deny_intents": ["delete_notes"]}}


class TestRolePolicy:
    def test_wildcard_and_empty_mean_any_module(self):
        assert RolePolicy.from_config({"allow_modules": ["*"]}).allows_module("pluto")
        assert RolePolicy.from_config({}).allows_module("pluto")

    def test_allow_list_is_exclusive(self):
        p = RolePolicy.from_config({"allow_modules": ["Core", "chronos"]})
        assert p.allows_module("core") and p.allows_module("CHRONOS")
        assert not p.allows_module("pluto")

    def test_deny_module_beats_allow(self):
        p = RolePolicy.from_config({"allow_modules": ["*"], "deny_modules": ["pluto"]})
        assert not p.allows_module("pluto")
        assert p.allows_module("apollo")

    def test_deny_intent_beats_allowed_module(self):
        p = RolePolicy.from_config(_FAMILY["family"])
        assert not p.allows("delete_notes", "core")
        assert p.allows("get_time", "chronos")

    def test_unregistered_module_is_never_allowed_for_an_intent(self):
        assert not RolePolicy.from_config({}).allows("mystery", None)

    def test_a_single_string_is_accepted_as_a_list(self):
        assert RolePolicy.from_config({"allow_modules": "chronos"}).allows_module("chronos")


class TestRoles:
    def _family_bot(self, intents, process_fn=None):
        return _make_bot(
            process_fn=process_fn or MagicMock(return_value="done"),
            roles={7: "family"},
            role_policies=_FAMILY,
            classify_fn=lambda text: intents,
        )

    def test_chats_without_a_role_are_owners_and_unrestricted(self):
        bot = _make_bot(allowed_chat_ids=[1])
        assert bot.role_for(1) == "owner"
        assert bot._policy_for(1) is None

    def test_owner_never_calls_classify(self):
        classify = MagicMock()
        process_fn = MagicMock(return_value="ok")
        bot = _make_bot(process_fn=process_fn, classify_fn=classify)
        _run(bot._handle_text(_make_update(text="anything"), MagicMock()))
        classify.assert_not_called()
        process_fn.assert_called_once()

    def test_allowed_intent_goes_through(self):
        process_fn = MagicMock(return_value="It's 9.")
        bot = self._family_bot("get_time", process_fn)
        update = _make_update(chat_id=7, text="what time is it")
        _run(bot._handle_text(update, MagicMock()))
        process_fn.assert_called_once_with("what time is it")
        update.message.reply_text.assert_called_once_with("It's 9.")

    def test_disallowed_module_is_refused_and_nothing_runs(self):
        process_fn = MagicMock()
        bot = self._family_bot("pluto_get_portfolio", process_fn)
        update = _make_update(chat_id=7, text="how is my portfolio")
        _run(bot._handle_text(update, MagicMock()))
        process_fn.assert_not_called()
        update.message.reply_text.assert_called_once_with(bot_module._DENIED)

    def test_denied_intent_in_an_allowed_module_is_refused(self):
        process_fn = MagicMock()
        bot = self._family_bot("delete_notes", process_fn)
        update = _make_update(chat_id=7, text="delete my notes")
        _run(bot._handle_text(update, MagicMock()))
        process_fn.assert_not_called()

    def test_every_intent_of_a_compound_query_must_be_allowed(self):
        process_fn = MagicMock()
        bot = self._family_bot(["get_time", "delete_notes"], process_fn)
        update = _make_update(chat_id=7, text="what time is it and delete my notes")
        _run(bot._handle_text(update, MagicMock()))
        process_fn.assert_not_called()

    def test_all_intents_allowed_in_a_compound_query_goes_through(self):
        process_fn = MagicMock(return_value="ok")
        bot = self._family_bot(["get_time", "get_weather"], process_fn)
        _run(bot._handle_text(_make_update(chat_id=7, text="time and weather"), MagicMock()))
        process_fn.assert_called_once()

    def test_unregistered_intent_fails_closed(self):
        process_fn = MagicMock()
        bot = self._family_bot("totally_made_up", process_fn)
        _run(bot._handle_text(_make_update(chat_id=7, text="x"), MagicMock()))
        process_fn.assert_not_called()

    def test_no_intents_returned_fails_closed(self):
        process_fn = MagicMock()
        bot = self._family_bot([], process_fn)
        _run(bot._handle_text(_make_update(chat_id=7, text="x"), MagicMock()))
        process_fn.assert_not_called()

    def test_classifier_error_fails_closed(self):
        process_fn = MagicMock()
        bot = _make_bot(
            process_fn=process_fn, roles={7: "family"}, role_policies=_FAMILY,
            classify_fn=MagicMock(side_effect=RuntimeError("ollama down")),
        )
        update = _make_update(chat_id=7, text="x")
        _run(bot._handle_text(update, MagicMock()))
        process_fn.assert_not_called()
        update.message.reply_text.assert_called_once_with(bot_module._DENIED)

    def test_missing_classifier_fails_closed_for_restricted_roles(self):
        process_fn = MagicMock()
        bot = _make_bot(process_fn=process_fn, roles={7: "family"}, role_policies=_FAMILY)
        _run(bot._handle_text(_make_update(chat_id=7, text="x"), MagicMock()))
        process_fn.assert_not_called()

    def test_role_without_a_policy_is_refused_everything(self):
        process_fn = MagicMock()
        bot = _make_bot(
            process_fn=process_fn, roles={7: "guest"}, role_policies={},
            classify_fn=lambda t: "get_time",
        )
        _run(bot._handle_text(_make_update(chat_id=7, text="time?"), MagicMock()))
        process_fn.assert_not_called()

    def test_owner_policy_can_still_be_restricted_explicitly(self):
        bot = _make_bot(
            roles={1: "owner"}, role_policies={"owner": {"deny_modules": ["pluto"]}},
            classify_fn=lambda t: "pluto_get_portfolio",
        )
        assert bot._policy_for(1) is not None

    def test_roles_grant_access_without_being_in_allowed_chat_ids(self):
        bot = _make_bot(allowed_chat_ids=[1], roles={7: "family"}, role_policies=_FAMILY)
        assert bot._is_allowed(7) and bot._is_allowed(1)
        assert not bot._is_allowed(99)

    def test_roles_alone_close_the_allowlist(self):
        bot = _make_bot(allowed_chat_ids=None, roles={7: "family"}, role_policies=_FAMILY)
        assert bot._is_allowed(7)
        assert not bot._is_allowed(8)

    def test_role_keys_may_be_strings_as_they_come_from_yaml_or_json(self):
        bot = _make_bot(roles={"7": "Family", "oops": "x"}, role_policies=_FAMILY)
        assert bot.role_for(7) == "family"
        assert 7 in bot.roles and len(bot.roles) == 1

    def test_restricted_chat_cannot_overwrite_the_device_location(self):
        memory = MagicMock()
        bot = _make_bot(memory=memory, roles={7: "family"}, role_policies=_FAMILY)
        update = _make_update(chat_id=7)
        update.effective_message.location.latitude = 1.0
        update.effective_message.location.longitude = 2.0
        _run(bot._handle_location(update, MagicMock()))
        memory.set_device_location.assert_not_called()

    def test_registry_failure_fails_closed(self, monkeypatch):
        monkeypatch.setattr(bot_module, "_module_for_intent", lambda intent: None)
        process_fn = MagicMock()
        bot = self._family_bot("get_time", process_fn)
        _run(bot._handle_text(_make_update(chat_id=7, text="time?"), MagicMock()))
        process_fn.assert_not_called()

    def test_policies_are_applied_through_the_real_registry(self):
        """No stubbing: get_time is Chronos in the real INTENT_MODULE_MAP."""
        process_fn = MagicMock(return_value="9:41")
        bot = self._family_bot("get_time", process_fn)
        _run(bot._handle_text(_make_update(chat_id=7, text="time"), MagicMock()))
        process_fn.assert_called_once()


# ---------------------------------------------------------------------------
# #195 typing indicator / not blocking the event loop
# ---------------------------------------------------------------------------

class TestTypingIndicator:
    def test_typing_is_sent_while_a_slow_request_runs(self):
        def slow(text):
            time.sleep(0.25)
            return "done"

        bot = _make_bot(process_fn=slow)
        bot.typing_interval = 0.05
        update = _make_update(text="run the backtest")
        update.effective_chat.send_action = AsyncMock()

        _run(bot._handle_text(update, MagicMock()))

        assert update.effective_chat.send_action.await_count >= 3
        assert update.effective_chat.send_action.await_args.args[0] == "typing"
        update.message.reply_text.assert_called_once_with("done")

    def test_typing_stops_once_the_request_finishes(self):
        bot = _make_bot(process_fn=MagicMock(return_value="ok"))
        bot.typing_interval = 0.02
        update = _make_update(text="hi")
        update.effective_chat.send_action = AsyncMock()

        async def scenario():
            await bot._handle_text(update, MagicMock())
            count = update.effective_chat.send_action.await_count
            await asyncio.sleep(0.15)
            return count, update.effective_chat.send_action.await_count

        before, after = _run(scenario())
        assert before == after

    def test_typing_failure_does_not_break_the_request(self):
        bot = _make_bot(process_fn=MagicMock(return_value="ok"))
        update = _make_update(text="hi")
        update.effective_chat.send_action = AsyncMock(side_effect=Exception("flood control"))
        _run(bot._handle_text(update, MagicMock()))
        update.message.reply_text.assert_called_once_with("ok")

    def test_the_event_loop_stays_free_while_a_request_runs(self):
        """process_fn used to run on the loop, freezing every other chat."""
        release = threading.Event()

        def blocking(text):
            release.wait(2)
            return "done"

        bot = _make_bot(process_fn=blocking)
        update = _make_update(text="slow")

        async def scenario():
            task = asyncio.ensure_future(bot._handle_text(update, MagicMock()))
            await asyncio.sleep(0.05)
            still_running = not task.done()
            ticked = False

            async def tick():
                nonlocal ticked
                ticked = True

            await tick()          # the loop is responsive
            release.set()
            await task
            return still_running, ticked

        assert _run(scenario()) == (True, True)

    def test_process_fn_calls_never_overlap(self):
        active = {"n": 0, "max": 0}
        lock = threading.Lock()

        def process(text):
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            time.sleep(0.05)
            with lock:
                active["n"] -= 1
            return "ok"

        bot = _make_bot(process_fn=process)

        async def scenario():
            await asyncio.gather(*[
                bot._handle_text(_make_update(text=f"q{i}"), MagicMock()) for i in range(4)
            ])

        _run(scenario())
        assert active["max"] == 1


# ---------------------------------------------------------------------------
# Reply splitting
# ---------------------------------------------------------------------------

class TestMessageSplitting:
    def test_short_text_is_one_chunk(self):
        assert _split_message("hello") == ["hello"]

    def test_long_text_is_split_within_the_limit_and_loses_nothing(self):
        text = "\n".join(f"line {i} " + "x" * 40 for i in range(300))
        chunks = _split_message(text)
        assert len(chunks) > 1
        assert all(len(c) <= 4096 for c in chunks)
        assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")

    def test_text_with_no_break_points_is_hard_split(self):
        chunks = _split_message("x" * 10000)
        assert [len(c) for c in chunks] == [4096, 4096, 1808]

    def test_a_long_reply_is_sent_in_parts_with_buttons_on_the_last(self):
        bot = _make_bot(process_fn=MagicMock(return_value=("word " * 2000) + " Say yes to continue."))
        update = _make_update(text="go")
        _run(bot._handle_text(update, MagicMock()))
        calls = update.message.reply_text.call_args_list
        assert len(calls) >= 2
        assert "reply_markup" not in calls[0].kwargs
        assert "reply_markup" in calls[-1].kwargs
