"""
main.py

Hestia: personal AI assistant — entry point and wiring layer.

Responsibilities
----------------
- Load configuration.
- Initialise all subsystems in dependency order.
- Wire the event bus.
- Expose process_text() as the single query entry point.
- Provide voice and CLI run-loops.

Nothing in this file contains business logic; every decision is delegated
to the appropriate module.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Environment – must happen before any ML library import
# ---------------------------------------------------------------------------

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_NOISY_LOGGERS = (
    "huggingface_hub", "transformers", "sentence_transformers",
    "torch", "urllib3", "httpx", "httpcore", "asyncio", "werkzeug",
)

def _configure_logging() -> logging.Logger:
    warnings.filterwarnings("ignore")
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("hestia")
    logger.setLevel(logging.INFO)
    return logger

logger = _configure_logging()

# ---------------------------------------------------------------------------
# Third-party + internal imports (after env / logging setup)
# ---------------------------------------------------------------------------

import yaml

from core.browser_agent import HestiaBrowserAgent
from core.event_bus import bus
from core.heartbeat import HestiaHeartbeat
from core.llm import HestiaLLM
from core.nlu import HestiaNLU
from core.ollama_manager import OllamaManager
from core.stt import HestiaSTT
from core.tts import HestiaTTS
from core.wake_word import WakeWordDetector
from modules.apollo import ApolloEngine
from modules.ares import AresEngine
from modules.artemis import ArtemisEngine
from modules.chronos.engine import ChronosEngine
from modules.dionysus import DionysusEngine
from modules.hecate import HecateEngine
from modules.hephaestus.engine import HephaestusEngine
from modules.hermes.engine import HermesEngine
from modules.hestia.core_module import CoreModule
from modules.hestia.orchestrator import HestiaOrchestrator
from modules.mnemosyne.engine import MnemosyneEngine
from modules.orpheus import OrpheusEngine
from modules.pluto import PlutoEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = Path("config/laptop_config.yaml")
_EXIT_WORDS: frozenset[str] = frozenset({"bye", "exit", "stop", "shutdown"})
_FILLER_RE = re.compile(r"\b(uh|um|you know)\b\s*", re.IGNORECASE)
_OLLAMA_STARTUP_DELAY = 2          # seconds after ensure_running()
_STT_MAX_DURATION = 10             # seconds per utterance
_WAKE_WORD_TIMEOUT = 30            # seconds per listen cycle
_MIN_VOICE_INPUT_LEN = 2           # discard utterances shorter than this
_RECENT_CONTEXT_TURNS = 5


# ---------------------------------------------------------------------------
# HestiaBuilder
# ---------------------------------------------------------------------------

class HestiaBuilder:
    """
    Constructs Hestia's subsystems from configuration.

    Each ``build_*`` method is a factory that takes its dependencies as
    explicit arguments (not a ``Hestia`` instance) and returns a fully
    constructed object. That makes every subsystem independently
    constructible and testable in isolation — e.g.
    ``HestiaBuilder(cfg).build_llm(manager)`` can be unit-tested, or a
    single subsystem swapped for a mock, without booting the rest of the
    app.

    ``Hestia.__init__`` is left owning only *wiring*: deciding the
    dependency order, holding the resulting references, and connecting
    already-built subsystems together (e.g. the event bus). Construction
    (this class) and wiring/startup (``Hestia``) are deliberately kept
    separate.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.ollama_cfg: dict[str, Any] = config.get("ollama", {})
        self.google_cfg: dict[str, Any] = config.get("google", {})
        self.sync_cfg: dict[str, Any] = config.get("sync", {})

    # -- Core inference stack -------------------------------------------------

    def build_ollama_manager(self) -> OllamaManager:
        manager = OllamaManager(
            host=self.ollama_cfg.get("host", "localhost"),
            port=self.ollama_cfg.get("port", 11434),
        )
        manager.ensure_running()
        time.sleep(_OLLAMA_STARTUP_DELAY)
        logger.info("Ollama running at %s:%s.", manager.host, manager.port)
        return manager

    def build_llm(self, ollama_manager: OllamaManager) -> HestiaLLM:
        return HestiaLLM(
            ollama_manager.host,
            ollama_manager.port,
            self.ollama_cfg.get("model", "mistral"),
        )

    def build_nlu(self) -> HestiaNLU:
        return HestiaNLU(
            model=self.ollama_cfg.get("model", "mistral"),
            host=self.ollama_cfg.get("host", "127.0.0.1"),
            port=self.ollama_cfg.get("port", 11434),
            prompt_path=self.config.get("nlu", {}).get("prompt_path"),
        )

    def build_mnemosyne(self, llm: HestiaLLM) -> MnemosyneEngine:
        """
        Mnemosyne is the single mandatory memory store.

        Every other module that needs memory receives a reference to this
        instance; no other memory object is created.
        """
        mnemosyne = MnemosyneEngine(llm)
        logger.info("Mnemosyne engine initialised.")
        return mnemosyne

    # -- Optional, feature-flagged modules ------------------------------------

    def build_optional_modules(self, llm: HestiaLLM) -> dict[str, Any]:
        """
        Build feature-flagged modules (Athena, Iris, Google, browser).

        Each is ``None`` in the returned dict when disabled or when its own
        initialisation fails, so callers can guard with ``if modules["x"]``.
        """
        modules: dict[str, Any] = {
            "athena": None,
            "iris": None,
            "google_agent": None,
            "browser_agent": None,
        }

        if self.config.get("athena", {}).get("enabled", False):
            try:
                from modules.athena.engine import AthenaEngine
                modules["athena"] = AthenaEngine(llm)
                logger.info("Athena enabled.")
            except Exception:
                logger.exception("Athena failed to initialise; disabling.")

        if self.config.get("iris", {}).get("enabled", False):
            try:
                from modules.iris import IrisEngine
                modules["iris"] = IrisEngine(llm)
                logger.info("Iris enabled.")
            except Exception:
                logger.exception("Iris failed to initialise; disabling.")

        if self.google_cfg.get("enabled", False):
            try:
                from core.google_agent import HestiaGoogleAgent
                agent = HestiaGoogleAgent(
                    credentials_path=self.google_cfg.get("credentials_path"),
                    token_path=self.google_cfg.get("token_path"),
                )
                agent.authenticate()
                modules["google_agent"] = agent
                logger.info("Google agent authenticated.")
            except Exception:
                logger.exception("Google agent failed to initialise; disabling.")

        try:
            modules["browser_agent"] = HestiaBrowserAgent()
        except Exception:
            logger.exception("Browser agent failed to initialise; disabling.")

        return modules

    # -- Orchestrator + module registration -----------------------------------

    def build_orchestrator(
        self, mnemosyne: MnemosyneEngine, optional_modules: dict[str, Any]
    ) -> tuple[HestiaOrchestrator, ApolloEngine]:
        """
        Build the orchestrator and register every module in priority order.

        Mandatory modules are registered unconditionally; optional ones are
        skipped when their subsystem is ``None``. Returns the orchestrator
        together with the ``ApolloEngine`` instance, since ``Hestia`` keeps
        a direct reference to Apollo for the web UI's mood endpoint.
        """
        athena       = optional_modules.get("athena")
        iris         = optional_modules.get("iris")
        google_agent = optional_modules.get("google_agent")
        browser_agent = optional_modules.get("browser_agent")

        orchestrator = HestiaOrchestrator(ollama_cfg=self.ollama_cfg)
        orchestrator.register_hecate(HecateEngine())

        # Core – always first so chat fallback is always available
        orchestrator.register(
            CoreModule(memory=mnemosyne, ollama_cfg=self.ollama_cfg)
        )

        # Memory
        orchestrator.register(mnemosyne)

        # Optional knowledge modules
        for mod in (athena, iris):
            if mod is not None:
                orchestrator.register(mod)

        # Time / calendar / communication
        orchestrator.register(ChronosEngine(memory=mnemosyne))
        orchestrator.register(ArtemisEngine())

        if google_agent:
            orchestrator.register(HermesEngine(google_agent))

        orchestrator.register(HephaestusEngine(browser_agent))

        # Specialist modules
        apollo = ApolloEngine(ollama_cfg=self.ollama_cfg)
        orchestrator.register(apollo)
        orchestrator.register(
            AresEngine(memory=mnemosyne, ollama_cfg=self.ollama_cfg)
        )
        orchestrator.register(
            OrpheusEngine(ollama_cfg=self.ollama_cfg, memory=mnemosyne)
        )
        orchestrator.register(
            DionysusEngine(
                ollama_cfg=self.ollama_cfg,
                browser_agent=browser_agent,
                memory=mnemosyne,
            )
        )
        orchestrator.register(PlutoEngine(ollama_cfg=self.ollama_cfg))

        logger.info(
            "Orchestrator ready (%d module(s) registered).",
            len(orchestrator.registered_modules),
        )
        return orchestrator, apollo

    # -- I/O --------------------------------------------------------------

    def build_io(self) -> tuple[HestiaSTT, HestiaTTS, WakeWordDetector]:
        stt_cfg = self.config.get("stt", {})
        tts_cfg = self.config.get("tts", {})
        wake_cfg = self.config.get("wake_word", {})

        stt = HestiaSTT(
            model_size=stt_cfg.get("model_size", "base.en"),
            device=stt_cfg.get("device", "cuda"),
            compute_type=stt_cfg.get("compute_type", "int8"),
            samplerate=stt_cfg.get("samplerate", 16000),
            noise_filter=stt_cfg.get("noise_filter", True),
            silence_frames=stt_cfg.get("silence_frames", 33),
        )
        tts = HestiaTTS(
            engine=tts_cfg.get("engine", "pyttsx3"),
            rate=tts_cfg.get("rate", 175),
            volume=tts_cfg.get("volume", 1.0),
            piper_model_path=tts_cfg.get("piper_model_path"),
        )
        wake_detector = WakeWordDetector(
            model_path=wake_cfg.get("model_path", "models/vosk-model-small-en-us-0.15"),
            wake_words=wake_cfg.get("wake_words"),
        )
        return stt, tts, wake_detector

    # -- Heartbeat / web UI / sync API ------------------------------------

    def build_heartbeat(self, mnemosyne: MnemosyneEngine) -> HestiaHeartbeat:
        return HestiaHeartbeat(interval=1800, mnemosyne=mnemosyne)

    def build_web_ui(
        self, mnemosyne: MnemosyneEngine, process_fn, apollo: Optional[ApolloEngine]
    ) -> Optional[Any]:
        try:
            from web_ui import HestiaWebUI
            web_ui = HestiaWebUI(
                memory=mnemosyne,
                process_fn=process_fn,
                apollo=apollo,
            )
            web_ui.start()
            logger.info("Web UI started.")
            return web_ui
        except Exception:
            logger.exception("Web UI failed to start; continuing without it.")
            return None

    def build_telegram_bot(self, process_fn, stt: Optional[HestiaSTT]) -> Optional[Any]:
        """
        Start the Telegram bot in-process, using the already-initialised
        Hestia stack.

        Previously this was only reachable via ``python -m core.telegram_bot``,
        but that module has no ``__main__`` entry point — running it standalone
        does nothing (imports the class, then exits). Wiring it here, the same
        way web UI and the sync API are wired, is what actually starts it, and
        gives it access to a fully-initialised ``process_fn`` (STT, memory,
        orchestrator, etc.) instead of requiring a second, separate init path.

        Note: ``config/laptop_config.yaml`` writes the token as
        ``${TELEGRAM_BOT_TOKEN}`` but ``_load_config`` uses plain
        ``yaml.safe_load`` with no env-var interpolation, so that placeholder
        is never resolved from the YAML. The token is read directly from the
        environment instead, same as Dionysus reads its API keys.
        """
        telegram_cfg = self.config.get("telegram", {})
        if not telegram_cfg.get("enabled", False):
            return None

        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not token:
            logger.warning(
                "Telegram enabled in config but TELEGRAM_BOT_TOKEN is not set "
                "in the environment; skipping Telegram bot startup."
            )
            return None

        try:
            from core.telegram_bot import HestiaTelegramBot
            bot = HestiaTelegramBot(
                token=token,
                process_fn=process_fn,
                allowed_chat_ids=telegram_cfg.get("allowed_chat_ids"),
                stt=stt,
            )
            bot.start()
            logger.info("Telegram bot started.")
            return bot
        except Exception:
            logger.exception("Telegram bot failed to start; continuing without it.")
            return None

    def start_sync_api(self, mnemosyne: MnemosyneEngine) -> None:
        if not self.sync_cfg.get("enabled", False):
            return
        try:
            import uvicorn
            from api import app as sync_app

            sync_app.state.memory = mnemosyne
            host = self.sync_cfg.get("host", "127.0.0.1")
            port = int(self.sync_cfg.get("port", 5001))

            def _run() -> None:
                uvicorn.run(sync_app, host=host, port=port, log_level="warning")

            threading.Thread(target=_run, daemon=True, name="SyncAPI").start()
            logger.info("Sync API running at http://%s:%d", host, port)
        except Exception:
            logger.exception("Sync API failed to start; continuing without it.")


# ---------------------------------------------------------------------------
# Hestia
# ---------------------------------------------------------------------------

class Hestia:
    """
    Top-level wiring class.

    Delegates all subsystem construction to ``HestiaBuilder`` and owns only
    the dependency order, the resulting references, and connecting
    already-built subsystems together (the event bus). Exposes
    ``process_text`` as the single synchronous query entry point.
    """

    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG) -> None:
        logger.info("Initialising Hestia…")
        self.config = _load_config(Path(config_path))
        builder = HestiaBuilder(self.config)

        # Derived config sections (read-only after __init__)
        self._ollama_cfg: dict[str, Any] = builder.ollama_cfg
        self._google_cfg: dict[str, Any] = builder.google_cfg
        self._sync_cfg: dict[str, Any] = builder.sync_cfg

        # -- Construction (via builder), in dependency order --------------
        self.ollama_manager = builder.build_ollama_manager()

        self.llm = builder.build_llm(self.ollama_manager)
        self.nlu = builder.build_nlu()

        self.mnemosyne = builder.build_mnemosyne(self.llm)
        self.nlu.set_memory(self.mnemosyne)

        optional_modules = builder.build_optional_modules(self.llm)
        self.athena        = optional_modules["athena"]
        self.iris           = optional_modules["iris"]
        self.google_agent   = optional_modules["google_agent"]
        self.browser_agent: Optional[HestiaBrowserAgent] = optional_modules["browser_agent"]

        self.orchestrator, self.apollo = builder.build_orchestrator(
            self.mnemosyne, optional_modules
        )

        self.stt, self.tts, self.wake_detector = builder.build_io()

        # -- Wiring: connect already-built subsystems together -------------
        self._init_event_bus()

        self.heartbeat = builder.build_heartbeat(self.mnemosyne)
        self.heartbeat.start()
        logger.info("Heartbeat started (interval=1800 s).")

        self.web_ui = builder.build_web_ui(self.mnemosyne, self.process_text, self.apollo)

        self.telegram_bot = builder.build_telegram_bot(self.process_text, self.stt)

        builder.start_sync_api(self.mnemosyne)

        logger.info("Hestia is ready.")

    # ------------------------------------------------------------------
    # Event-bus wiring (connects already-built subsystems; not construction)
    # ------------------------------------------------------------------

    def _init_event_bus(self) -> None:
        """
        Register all event-bus listeners.

        Listeners are registered with descriptive lambdas or named wrappers
        so that bus.listeners_for() returns meaningful names in diagnostics.
        """
        # Persist every interaction to memory
        mn = self.mnemosyne

        def _on_interaction(data: dict) -> None:
            try:
                mn.push(data["query"], data["response"], data["intent"])
            except Exception:
                logger.exception("interaction_logged handler failed.")

        bus.on("interaction_logged", _on_interaction)

        # TTS output
        def _on_speak(data: dict) -> None:
            try:
                self.tts.speak(data.get("text", ""))
            except Exception:
                logger.exception("speak handler failed.")

        bus.on("speak", _on_speak)

        # Morning brief
        bus.on(
            "morning_brief_requested",
            lambda _: self.process_text("give me my morning brief"),
        )

        # Summarisation trigger
        bus.on("mnemosyne_summarise", lambda _: mn.trigger_summarise())

        # Unrecognised HEARTBEAT.md task — surface it instead of letting it
        # silently vanish. This isn't an error (the task text may just be
        # awaiting a handler in heartbeat.py's _evaluate_task), so it's
        # logged at warning level rather than raised.
        def _on_heartbeat_unhandled_task(data: dict) -> None:
            logger.warning(
                "Heartbeat: no handler for HEARTBEAT.md task %r — "
                "add a case to HestiaHeartbeat._evaluate_task().",
                data.get("task", ""),
            )

        bus.on("heartbeat_unhandled_task", _on_heartbeat_unhandled_task)

        logger.info("Event bus wired.")

    # ------------------------------------------------------------------
    # Core query entry point
    # ------------------------------------------------------------------

    def process_text(self, text: str) -> str:
        """
        Accept a raw text query, route it through the pipeline, and return
        a plain-string response.

        Steps
        -----
        1. Sanitise and clean the input.
        2. Fetch recent context from memory.
        3. Run NLU.
        4. Dispatch via orchestrator.
        5. Post-process (unwrap JSON if the LLM leaked a dict).
        6. Speak the response and emit interaction event.

        Never raises; errors produce a safe fallback string.
        """
        cleaned = _clean_input(text)
        if not cleaned:
            return ""

        logger.info("You: %s", cleaned)

        try:
            context = self.mnemosyne.get_recent(_RECENT_CONTEXT_TURNS)
            nlu_result = self.nlu.understand(cleaned, context)
        except Exception:
            logger.exception("NLU failed for input=%r.", cleaned[:80])
            nlu_result = {"intent": "chat", "entities": {}, "response": ""}

        try:
            response = self.orchestrator.dispatch(cleaned, nlu_result)
        except Exception:
            logger.exception("Orchestrator dispatch failed.")
            response = "I'm sorry, something went wrong."

        response = _postprocess(response)

        logger.info("Hestia: %s", response)

        try:
            self.tts.speak(response)
        except Exception:
            logger.exception("TTS failed.")

        try:
            self.wake_detector.flush_audio_queue()
        except Exception:
            logger.debug("flush_audio_queue() failed; ignoring.")

        bus.emit_sync(
            "interaction_logged",
            {
                "query": cleaned,
                "response": response,
                "intent": nlu_result.get("intent", "chat"),
            },
        )

        return response

    # ------------------------------------------------------------------
    # Run-loops
    # ------------------------------------------------------------------

    def run_voice_loop(self) -> None:
        """
        Alternate between wake-word detection and STT, feeding into
        ``process_text``.
        """
        logger.info("Voice loop started — listening for wake word.")
        try:
            while True:
                if not self.wake_detector.listen_for_wake_word(
                    timeout=_WAKE_WORD_TIMEOUT
                ):
                    continue

                self.tts.speak("Yes?")

                text = self.stt.listen_once(max_duration=_STT_MAX_DURATION)
                if not text or len(text.strip()) < _MIN_VOICE_INPUT_LEN:
                    self.tts.speak("I didn't catch that.")
                    continue

                if text.lower().strip() in _EXIT_WORDS:
                    self.tts.speak("Goodbye.")
                    break

                self.process_text(text)

        except KeyboardInterrupt:
            logger.info("Voice loop interrupted by user.")
        finally:
            self._shutdown()

    def run_cli_loop(self) -> None:
        """Accept text queries from stdin."""
        logger.info("CLI loop started. Type 'exit' to quit.")
        try:
            while True:
                try:
                    user_input = input("> ").strip()
                except EOFError:
                    break
                if not user_input:
                    continue
                if user_input.lower() in _EXIT_WORDS:
                    break
                self.process_text(user_input)
        except KeyboardInterrupt:
            logger.info("CLI loop interrupted by user.")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Gracefully stop background services."""
        logger.info("Shutting down Hestia…")
        try:
            self.heartbeat.stop()
        except Exception:
            logger.debug("heartbeat.stop() raised; ignoring.")

        try:
            bus.shutdown()  # graceful executor shutdown
        except Exception:
            logger.debug("bus.shutdown() raised; ignoring.")

        try:
            bus.clear()
        except Exception:
            logger.debug("bus.clear() raised; ignoring.")

        logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict[str, Any]:
    """Load and return the YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}. "
            "Copy config/laptop_config.example.yaml and edit it."
        )
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file {path} must be a YAML mapping.")
    return cfg


def _clean_input(text: str) -> str:
    """
    Sanitise raw user input.

    - Strip surrounding whitespace.
    - Lower-case.
    - Remove common speech fillers (uh, um, you know).
    """
    stripped = text.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()
    return _FILLER_RE.sub("", lowered).strip()


def _postprocess(response: str) -> str:
    """
    Unwrap a JSON-encoded response string if the LLM leaked a dict.

    Returns the original string unchanged when it is not valid JSON or
    when the parsed object does not contain a ``response`` key.
    """
    if not response:
        return "Done."

    stripped = response.strip()
    if not stripped.startswith("{"):
        return stripped

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "response" in parsed:
            return str(parsed["response"])
    except (json.JSONDecodeError, ValueError):
        pass

    return stripped


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hestia personal AI assistant")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Run in voice mode (wake word + STT).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    hestia = Hestia(config_path=args.config)

    if args.voice:
        hestia.run_voice_loop()
    else:
        hestia.run_cli_loop()