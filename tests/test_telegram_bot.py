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


def _make_bot(process_fn=None, allowed_chat_ids=None, stt=None, memory=None):
    return bot_module.HestiaTelegramBot(
        token="fake-token",
        process_fn=process_fn or MagicMock(return_value="a reply"),
        allowed_chat_ids=allowed_chat_ids,
        stt=stt,
        memory=memory,
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
    def test_registers_four_handlers(self):
        bot = _make_bot()
        assert bot._app.add_handler.call_count == 4

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

    def test_early_exception_before_subprocess_import_is_a_real_bug(self, monkeypatch):
        """DOCUMENTS A REAL BUG in core/telegram_bot.py, does not assert
        desired behavior.

        `import subprocess` and `import wave` happen *inside* the try
        block, after the `get_file()`/`download_to_drive()` Telegram API
        calls. If either of those calls raises (e.g. a transient network
        error), `subprocess` was never bound in the function's local
        scope, so `except subprocess.CalledProcessError:` itself raises
        UnboundLocalError instead of matching — the handler crashes
        instead of falling through to the friendly generic-error message
        the user would otherwise see for any other post-import failure.

        Fix: move `import subprocess` and `import wave` to the top of the
        file (or above the try block) so the except clause can always
        resolve the name.
        """
        stt = MagicMock()
        bot = _make_bot(stt=stt)
        update, message = self._update_with_voice()
        message.voice.get_file = AsyncMock(side_effect=Exception("network blip"))

        with pytest.raises(UnboundLocalError):
            _run(bot._handle_voice(update, MagicMock()))

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
