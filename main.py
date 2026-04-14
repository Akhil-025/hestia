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
# Hestia
# ---------------------------------------------------------------------------

class Hestia:
    """
    Top-level wiring class.

    Initialises every subsystem exactly once, in dependency order, and
    exposes ``process_text`` as the single synchronous query entry point.
    """

    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG) -> None:
        logger.info("Initialising Hestia…")
        self.config = _load_config(Path(config_path))

        # Derived config sections (read-only after __init__)
        self._ollama_cfg: dict[str, Any] = self.config.get("ollama", {})
        self._google_cfg: dict[str, Any] = self.config.get("google", {})
        self._sync_cfg: dict[str, Any] = self.config.get("sync", {})

        # Subsystem attributes – declared here for IDE / type-checker visibility
        self.mnemosyne: MnemosyneEngine
        self.orchestrator: HestiaOrchestrator
        self.stt: HestiaSTT
        self.tts: HestiaTTS
        self.wake_detector: WakeWordDetector
        self.heartbeat: HestiaHeartbeat

        self._init_ollama()
        self._init_llm()
        self._init_mnemosyne()
        self._init_optional_modules()
        self._init_orchestrator()
        self._init_io()
        self._init_event_bus()
        self._init_heartbeat()
        self._init_web_ui()
        self._init_sync_api()

        logger.info("Hestia is ready.")

    # ------------------------------------------------------------------
    # Initialisation helpers (private, each owns one concern)
    # ------------------------------------------------------------------

    def _init_ollama(self) -> None:
        self.ollama_manager = OllamaManager(
            host=self._ollama_cfg.get("host", "localhost"),
            port=self._ollama_cfg.get("port", 11434),
        )
        self.ollama_manager.ensure_running()
        time.sleep(_OLLAMA_STARTUP_DELAY)
        logger.info(
            "Ollama running at %s:%s.",
            self.ollama_manager.host,
            self.ollama_manager.port,
        )

    def _init_llm(self) -> None:
        self.llm = HestiaLLM(
            self.ollama_manager.host,
            self.ollama_manager.port,
            self._ollama_cfg.get("model", "mistral"),
        )
        self.nlu = HestiaNLU(
            model=self._ollama_cfg.get("model", "mistral"),
            host=self._ollama_cfg.get("host", "127.0.0.1"),
            port=self._ollama_cfg.get("port", 11434),
            prompt_path=self.config.get("nlu", {}).get("prompt_path"),
        )

    def _init_mnemosyne(self) -> None:
        """
        Mnemosyne is the single mandatory memory store.

        Every other module that needs memory receives a reference to this
        instance; no other memory object is created.
        """
        self.mnemosyne = MnemosyneEngine(self.llm)
        self.nlu.set_memory(self.mnemosyne)
        logger.info("Mnemosyne engine initialised.")

    def _init_optional_modules(self) -> None:
        """
        Initialise feature-flagged modules (Athena, Iris, Google, browser).

        Each module is set to ``None`` when disabled so downstream code can
        guard with ``if self.X``.
        """
        # Athena (RAG knowledge base)
        self.athena = None
        if self.config.get("athena", {}).get("enabled", False):
            try:
                from modules.athena.engine import AthenaEngine
                self.athena = AthenaEngine(self.llm)
                logger.info("Athena enabled.")
            except Exception:
                logger.exception("Athena failed to initialise; disabling.")

        # Iris
        self.iris = None
        if self.config.get("iris", {}).get("enabled", False):
            try:
                from modules.iris import IrisEngine
                self.iris = IrisEngine(self.llm)
                logger.info("Iris enabled.")
            except Exception:
                logger.exception("Iris failed to initialise; disabling.")

        # Google (Gmail + Calendar)
        self.google_agent = None
        if self._google_cfg.get("enabled", False):
            try:
                from core.google_agent import HestiaGoogleAgent
                agent = HestiaGoogleAgent(
                    credentials_path=self._google_cfg.get("credentials_path"),
                    token_path=self._google_cfg.get("token_path"),
                )
                agent.authenticate()
                self.google_agent = agent
                logger.info("Google agent authenticated.")
            except Exception:
                logger.exception("Google agent failed to initialise; disabling.")

        # Browser
        self.browser_agent: Optional[HestiaBrowserAgent] = None
        try:
            self.browser_agent = HestiaBrowserAgent()
        except Exception:
            logger.exception("Browser agent failed to initialise; disabling.")

    def _init_orchestrator(self) -> None:
        """
        Build the orchestrator and register every module in priority order.

        Mandatory modules are registered unconditionally; optional ones are
        skipped when their subsystem is ``None``.
        """
        self.orchestrator = HestiaOrchestrator()
        self.orchestrator.register_hecate(HecateEngine())

        # Core – always first so chat fallback is always available
        self.orchestrator.register(
            CoreModule(memory=self.mnemosyne, ollama_cfg=self._ollama_cfg)
        )

        # Memory
        self.orchestrator.register(self.mnemosyne)

        # Optional knowledge modules
        for mod in (self.athena, self.iris):
            if mod is not None:
                self.orchestrator.register(mod)

        # Time / calendar / communication
        self.orchestrator.register(ChronosEngine(memory=self.mnemosyne))
        self.orchestrator.register(ArtemisEngine())

        if self.google_agent:
            self.orchestrator.register(HermesEngine(self.google_agent))

        self.orchestrator.register(HephaestusEngine(self.browser_agent))

        # Specialist modules
        self.orchestrator.register(ApolloEngine(ollama_cfg=self._ollama_cfg))
        self.orchestrator.register(
            AresEngine(memory=self.mnemosyne, ollama_cfg=self._ollama_cfg)
        )
        self.orchestrator.register(
            OrpheusEngine(ollama_cfg=self._ollama_cfg, memory=self.mnemosyne)
        )
        self.orchestrator.register(
            DionysusEngine(
                ollama_cfg=self._ollama_cfg,
                browser_agent=self.browser_agent,
                memory=self.mnemosyne,
            )
        )
        self.orchestrator.register(PlutoEngine(ollama_cfg=self._ollama_cfg))

        logger.info(
            "Orchestrator ready (%d module(s) registered).",
            len(self.orchestrator.registered_modules),
        )

    def _init_io(self) -> None:
        self.stt = HestiaSTT()
        self.tts = HestiaTTS()
        self.wake_detector = WakeWordDetector()

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

        logger.info("Event bus wired.")

    def _init_heartbeat(self) -> None:
        self.heartbeat = HestiaHeartbeat(interval=1800, mnemosyne=self.mnemosyne)
        self.heartbeat.start()
        logger.info("Heartbeat started (interval=1800 s).")

    def _init_web_ui(self) -> None:
        try:
            from web_ui import HestiaWebUI
            self.web_ui = HestiaWebUI(
                memory=self.mnemosyne,
                process_fn=self.process_text,
            )
            self.web_ui.start()
            logger.info("Web UI started.")
        except Exception:
            logger.exception("Web UI failed to start; continuing without it.")
            self.web_ui = None

    def _init_sync_api(self) -> None:
        if not self._sync_cfg.get("enabled", False):
            return
        try:
            import uvicorn
            from api import app as sync_app

            sync_app.state.memory = self.mnemosyne
            host = self._sync_cfg.get("host", "127.0.0.1")
            port = int(self._sync_cfg.get("port", 5001))

            def _run() -> None:
                uvicorn.run(sync_app, host=host, port=port, log_level="warning")

            threading.Thread(target=_run, daemon=True, name="SyncAPI").start()
            logger.info("Sync API running at http://%s:%d", host, port)
        except Exception:
            logger.exception("Sync API failed to start; continuing without it.")

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