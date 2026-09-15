"""
tests/conftest.py

Installs lightweight fake modules into sys.modules for third-party packages
that are hardware-bound (mic/speaker), require native binaries, or hit the
network at *import time* — vosk, webrtcvad, faster_whisper, pyttsx3,
noisereduce, and python-telegram-bot.

This runs before any test module imports code from core/, so
core/wake_word.py, core/stt.py, core/tts.py, core/noise_filter.py, and
core/telegram_bot.py can be imported normally in a CI/sandbox environment
that doesn't have Vosk models, a sound card, or the telegram package
installed.

Individual test files further customize the relevant fake attributes
(e.g. monkeypatching `vosk.Model` to raise) per-test; this file only
guarantees the imports succeed with sane default Mocks.

google-api-python-client / google-auth-oauthlib are deliberately NOT
stubbed here: core/google_agent.py defers those imports to inside
authenticate(), so each test that exercises authenticate() installs its
own scoped fakes (see test_google_agent.py) rather than polluting every
test in the suite.
"""
import sys
import types
from unittest.mock import MagicMock


def _install_fake_module(name: str, **attrs) -> types.ModuleType:
    """Create (or fetch) a fake module under `name`, set attrs, register it."""
    mod = sys.modules.get(name) or types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


# ---------------------------------------------------------------------------
# vosk (core/wake_word.py)
# ---------------------------------------------------------------------------
_install_fake_module(
    "vosk",
    Model=MagicMock(name="vosk.Model"),
    KaldiRecognizer=MagicMock(name="vosk.KaldiRecognizer"),
)

# ---------------------------------------------------------------------------
# webrtcvad (core/stt.py)
# ---------------------------------------------------------------------------
_install_fake_module("webrtcvad", Vad=MagicMock(name="webrtcvad.Vad"))

# ---------------------------------------------------------------------------
# faster_whisper (core/stt.py)
# ---------------------------------------------------------------------------
_install_fake_module("faster_whisper", WhisperModel=MagicMock(name="faster_whisper.WhisperModel"))

# ---------------------------------------------------------------------------
# noisereduce (core/noise_filter.py)
# ---------------------------------------------------------------------------
_install_fake_module("noisereduce", reduce_noise=MagicMock(name="noisereduce.reduce_noise"))

# ---------------------------------------------------------------------------
# pyttsx3 (core/tts.py)
# ---------------------------------------------------------------------------
_install_fake_module("pyttsx3", init=MagicMock(name="pyttsx3.init"))

# ---------------------------------------------------------------------------
# python-telegram-bot (core/telegram_bot.py)
# ---------------------------------------------------------------------------
_telegram = _install_fake_module("telegram", Update=MagicMock(name="telegram.Update"))


class _FakeApplicationBuilder:
    """Chained builder mimicking telegram.ext.ApplicationBuilder's fluent API."""

    def __init__(self):
        self._token = None

    def token(self, token):
        self._token = token
        return self

    def build(self):
        app = MagicMock(name="telegram.ext.Application")
        app.running = False
        app.add_handler = MagicMock()
        return app


_install_fake_module(
    "telegram.ext",
    ApplicationBuilder=_FakeApplicationBuilder,
    CommandHandler=MagicMock(name="telegram.ext.CommandHandler"),
    MessageHandler=MagicMock(name="telegram.ext.MessageHandler"),
    ContextTypes=types.SimpleNamespace(DEFAULT_TYPE=MagicMock(name="ContextTypes.DEFAULT_TYPE")),
    filters=types.SimpleNamespace(
        TEXT=MagicMock(name="filters.TEXT"),
        COMMAND=MagicMock(name="filters.COMMAND"),
        VOICE=MagicMock(name="filters.VOICE"),
        LOCATION=MagicMock(name="filters.LOCATION"),
    ),
)
# Note: `filters.TEXT & ~filters.COMMAND` is evaluated at HestiaTelegramBot.__init__
# time in telegram_bot.py. MagicMock implements the bitwise dunders
# (__and__, __invert__, etc.) out of the box, so this "just works" without
# any extra configuration here.
