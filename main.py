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

from core.barge_in import BargeInListener
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
from modules.metis import MetisEngine
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
        # NLU classification and general reasoning are different workloads
        # sharing one model/one Ollama instance today: classification is a
        # small schema-constrained JSON call that runs on EVERY query (see
        # core/nlu.py), often with up to 3 retries, while reasoning (chat,
        # RAG synthesis, Pluto's multi-step ReAct loop) is much heavier
        # generation — see core/ollama_client.py's new latency logging for
        # how to tell which one is actually the bottleneck before deciding
        # this is worth acting on.
        #
        # HestiaNLU already accepts an independent model/host/port (see its
        # `providers` handling); this was never wired up to config, so NLU
        # was silently forced onto whatever `ollama.model` was. `nlu.model`
        # (and optionally `nlu.host`/`nlu.port`, for pointing NLU at an
        # entirely separate Ollama instance) let classification run on a
        # smaller/faster model independently of reasoning. All three are
        # optional and fall back to the `ollama:` block, so existing
        # configs behave exactly as before until set.
        nlu_cfg = self.config.get("nlu", {})
        return HestiaNLU(
            model=nlu_cfg.get("model") or self.ollama_cfg.get("model", "mistral"),
            host=nlu_cfg.get("host") or self.ollama_cfg.get("host", "127.0.0.1"),
            port=nlu_cfg.get("port") or self.ollama_cfg.get("port", 11434),
            prompt_path=nlu_cfg.get("prompt_path"),
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
                    # Previously omitted, so HestiaGoogleAgent silently
                    # defaulted to "UTC" regardless of the chronos.timezone
                    # config value — see HermesEngine registration below.
                    timezone=self.google_cfg.get(
                        "timezone",
                        self.config.get("chronos", {}).get("timezone", "Asia/Kolkata"),
                    ),
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

        # Core – always first so chat fallback is always available.
        # Same config key ChronosEngine/HermesEngine use below — without
        # this, CoreModule's get_user_info()/get_system_info() fall back to
        # server-local time instead of the user's configured timezone,
        # disagreeing with Chronos for the exact date/time questions the
        # NLU sometimes misroutes to Core (see core_module.py).
        orchestrator.register(
            CoreModule(
                memory=mnemosyne,
                ollama_cfg=self.ollama_cfg,
                timezone_name=self.config.get("chronos", {}).get("timezone", "Asia/Kolkata"),
            )
        )

        # Memory
        orchestrator.register(mnemosyne)

        # Optional knowledge modules
        for mod in (athena, iris):
            if mod is not None:
                orchestrator.register(mod)

        # Time / calendar / communication
        chronos = ChronosEngine(
            memory=mnemosyne,
            local_tz=self.config.get("chronos", {}).get("timezone", "Asia/Kolkata"),
        )
        orchestrator.register(chronos)
        artemis = ArtemisEngine(ollama_cfg=self.ollama_cfg)
        orchestrator.register(artemis)

        if google_agent:
            # Same config key ChronosEngine uses above — without this,
            # HermesEngine defaults to UTC and every created event lands
            # offset by the difference between UTC and the user's real
            # timezone (e.g. "3pm" becomes "8:30pm" for Asia/Kolkata).
            hermes_tz = self.config.get("chronos", {}).get("timezone", "Asia/Kolkata")
            orchestrator.register(HermesEngine(google_agent, timezone_name=hermes_tz))

        orchestrator.register(
            HephaestusEngine(
                browser_agent,
                app_map=self.config.get("hephaestus", {}).get("app_map"),
            )
        )

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
            MetisEngine(ollama_cfg=self.ollama_cfg, memory=mnemosyne)
        )
        orchestrator.register(
            DionysusEngine(
                ollama_cfg=self.ollama_cfg,
                browser_agent=browser_agent,
                memory=mnemosyne,
            )
        )
        pluto = PlutoEngine(ollama_cfg=self.ollama_cfg)
        orchestrator.register(pluto)

        logger.info(
            "Orchestrator ready (%d module(s) registered).",
            len(orchestrator.registered_modules),
        )
        return orchestrator, apollo, pluto, artemis, chronos

    # -- I/O --------------------------------------------------------------

    def build_io(
        self,
    ) -> tuple[HestiaSTT, HestiaTTS, WakeWordDetector, BargeInListener]:
        stt_cfg = self.config.get("stt", {})
        tts_cfg = self.config.get("tts", {})
        wake_cfg = self.config.get("wake_word", {})
        barge_in_cfg = self.config.get("barge_in", {})

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
        # Barge-in is opt-out (default true): it only ever runs while
        # Hestia is speaking (see Hestia._speak_streaming /
        # _speak_with_barge_in), so leaving it enabled costs nothing when
        # the user never interrupts, and lets them the moment they do.
        barge_in = BargeInListener(
            samplerate=barge_in_cfg.get("samplerate", stt_cfg.get("samplerate", 16000)),
            vad_aggressiveness=barge_in_cfg.get("vad_aggressiveness", 2),
            speech_frames_to_trigger=barge_in_cfg.get("speech_frames_to_trigger", 3),
            # See core/barge_in.py's docstring: without real acoustic echo
            # cancellation, min_rms is the only lever for cutting down
            # false self-interruptions from Hestia's own voice bleeding
            # into the mic on shared speaker/mic hardware (e.g. a
            # laptop). Raise barge_in.min_rms in config if she's
            # interrupting herself; lower it (or use a headset, the real
            # fix) if real interruptions go unnoticed.
            min_rms=barge_in_cfg.get("min_rms", 300.0),
            pre_roll_frames=barge_in_cfg.get("pre_roll_frames", 10),
            post_trigger_silence_frames=barge_in_cfg.get(
                "post_trigger_silence_frames", stt_cfg.get("silence_frames", 33) - 8
            ),
            max_capture_seconds=barge_in_cfg.get("max_capture_seconds", 12.0),
        )
        return stt, tts, wake_detector, barge_in

    # -- Heartbeat / web UI / sync API ------------------------------------

    def build_heartbeat(self, mnemosyne: MnemosyneEngine) -> HestiaHeartbeat:
        return HestiaHeartbeat(interval=1800, mnemosyne=mnemosyne)

    def build_web_ui(
        self,
        mnemosyne: MnemosyneEngine,
        process_fn,
        apollo: Optional[ApolloEngine],
        pluto: Optional[Any] = None,
        artemis: Optional[Any] = None,
        chronos: Optional[Any] = None,
        athena: Optional[Any] = None,
        skill_loader: Optional[Any] = None,
        stt: Optional[HestiaSTT] = None,
        tts: Optional[HestiaTTS] = None,
    ) -> Optional[Any]:
        try:
            from web_ui import HestiaWebUI
            web_ui = HestiaWebUI(
                memory=mnemosyne,
                process_fn=process_fn,
                apollo=apollo,
                pluto=pluto,
                artemis=artemis,
                chronos=chronos,
                athena=athena,
                skill_loader=skill_loader,
                stt=stt,
                tts=tts,
            )
            web_ui.start()
            logger.info("Web UI started.")
            return web_ui
        except Exception:
            logger.exception("Web UI failed to start; continuing without it.")
            return None

    def build_telegram_bot(
        self, process_fn, stt: Optional[HestiaSTT], memory: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Start the Telegram bot in-process, using the already-initialised
        Hestia stack.

        Previously this was only reachable via ``python -m core.telegram_bot``,
        but that module has no ``__main__`` entry point — running it standalone
        does nothing (imports the class, then exits). Wiring it here, the same
        way web UI and the sync API are wired, is what actually starts it, and
        gives it access to a fully-initialised ``process_fn`` (STT, memory,
        orchestrator, etc.) instead of requiring a second, separate init path.

        Note: the bot token is read directly from the ``TELEGRAM_BOT_TOKEN``
        environment variable, same as Dionysus reads its API keys —
        ``config/laptop_config.yaml`` deliberately has no ``token:`` key.
        (An earlier version of that file wrote one as
        ``${TELEGRAM_BOT_TOKEN}``, but ``_load_config`` uses plain
        ``yaml.safe_load`` with no env-var interpolation, so that
        placeholder was never actually resolved — it just looked like it
        worked. Removed rather than "fixed" via interpolation, since
        secrets belong in the environment regardless.)
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
                memory=memory,
            )
            bot.start()
            logger.info("Telegram bot started.")
            return bot
        except Exception:
            logger.exception("Telegram bot failed to start; continuing without it.")
            return None

    # Hosts that are only reachable from this machine. Anything else means
    # /sync/* is potentially reachable by other devices/users. Mirrors the
    # same constant/contract in web_ui.py's _register_auth_guard.
    _LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

    def start_sync_api(self, mnemosyne: MnemosyneEngine) -> None:
        if not self.sync_cfg.get("enabled", False):
            return

        host = self.sync_cfg.get("host", "127.0.0.1")
        port = int(self.sync_cfg.get("port", 5001))

        # Shared secret for /sync/*, same convention as TELEGRAM_BOT_TOKEN:
        # read from the environment, never from the YAML, so it can't end
        # up committed or zipped up alongside the rest of the config.
        api_key = os.getenv("HESTIA_SYNC_API_KEY", "").strip() or None

        if not api_key and host not in self._LOOPBACK_HOSTS:
            logger.error(
                "Refusing to start Sync API: host=%r is not loopback-only "
                "and HESTIA_SYNC_API_KEY is not set. Set that environment "
                "variable, or bind sync.host to 127.0.0.1 for strictly-"
                "local use.", host,
            )
            return

        if not api_key:
            logger.warning(
                "Sync API starting without HESTIA_SYNC_API_KEY — /sync/* "
                "is unauthenticated. This is only safe because host=%r is "
                "loopback-only.", host,
            )

        try:
            import uvicorn
            from api import create_app

            sync_app = create_app(api_key=api_key)
            sync_app.state.memory = mnemosyne

            def _run() -> None:
                uvicorn.run(sync_app, host=host, port=port, log_level="warning")

            threading.Thread(target=_run, daemon=True, name="SyncAPI").start()
            logger.info(
                "Sync API running at http://%s:%d%s",
                host, port, " (authenticated)" if api_key else "",
            )
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

        # Lowest-priority location source (see MnemosyneEngine.
        # ensure_device_location_via_ip docstring for the full priority
        # order). Backgrounded so a slow/unreachable IP-geolocation API
        # never delays CLI startup; it's a no-op once GPS/Telegram has
        # already provided a location.
        threading.Thread(
            target=self.mnemosyne.ensure_device_location_via_ip,
            daemon=True,
            name="IPLocationFallback",
        ).start()

        optional_modules = builder.build_optional_modules(self.llm)
        self.athena        = optional_modules["athena"]
        self.iris           = optional_modules["iris"]
        self.google_agent   = optional_modules["google_agent"]
        self.browser_agent: Optional[HestiaBrowserAgent] = optional_modules["browser_agent"]

        self.orchestrator, self.apollo, self.pluto, self.artemis, self.chronos = (
            builder.build_orchestrator(self.mnemosyne, optional_modules)
        )

        self.stt, self.tts, self.wake_detector, self.barge_in = builder.build_io()
        self._barge_in_enabled: bool = self.config.get("barge_in", {}).get("enabled", True)

        # -- Wiring: connect already-built subsystems together -------------
        self._init_event_bus()

        self.heartbeat = builder.build_heartbeat(self.mnemosyne)
        self.heartbeat.start()
        logger.info("Heartbeat started (interval=1800 s).")

        self.web_ui = builder.build_web_ui(
            self.mnemosyne,
            self.process_text,
            self.apollo,
            pluto=self.pluto,
            artemis=self.artemis,
            chronos=self.chronos,
            athena=self.athena,
            stt=self.stt,
            tts=self.tts,
        )

        self.telegram_bot = builder.build_telegram_bot(self.process_text, self.stt, self.mnemosyne)

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
    # Voice-only query entry point (streaming + barge-in)
    # ------------------------------------------------------------------

    def process_voice_turn(self, text: str) -> str:
        """
        Voice-loop counterpart to process_text(): same NLU → dispatch →
        speak → log pipeline, but for the common "just chatting" case it
        streams the LLM's reply straight into TTS sentence-by-sentence
        (see HestiaOrchestrator.try_stream_chat) instead of waiting for
        the whole response, and lets the user interrupt Hestia mid-reply
        by talking over her (see core/barge_in.py).

        Only used by run_voice_loop() — process_text() remains the
        blocking, single-string-in-single-string-out entry point used by
        the web UI, Telegram bot, and heartbeat, none of which have
        anywhere to send partial output as it streams in.

        Never raises; errors produce a safe fallback string, same
        contract as process_text().
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

        stream = None
        try:
            stream = self.orchestrator.try_stream_chat(cleaned, nlu_result)
        except Exception:
            logger.exception(
                "try_stream_chat() failed for input=%r; falling back to "
                "blocking dispatch().", cleaned[:80],
            )
            stream = None

        if stream is not None:
            response = self._speak_streaming(stream)
        else:
            try:
                response = self.orchestrator.dispatch(cleaned, nlu_result)
            except Exception:
                logger.exception("Orchestrator dispatch failed.")
                response = "I'm sorry, something went wrong."

            response = _postprocess(response)
            logger.info("Hestia: %s", response)
            self._speak_with_barge_in(response)

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

    def _speak_streaming(self, chunks) -> str:
        """
        Feed a generator of streamed text chunks into HestiaTTS.speak_stream
        while the barge-in listener is armed, and return the full
        concatenated response once playback finishes (or is cut short by
        an interruption).
        """
        parts: list[str] = []

        def _tap():
            for chunk in chunks:
                parts.append(chunk)
                yield chunk

        self._with_barge_in(lambda: self.tts.speak_stream(_tap()))

        response = "".join(parts).strip()
        if not response:
            response = "Done."
        logger.info("Hestia: %s", response)
        return response

    def _speak_with_barge_in(self, response: str) -> None:
        """Speak a single finished response with the barge-in listener
        armed, so even non-streamed replies can be interrupted."""
        self._with_barge_in(lambda: self.tts.speak(response))

    def _with_barge_in(self, speak_fn) -> None:
        """
        Run *speak_fn* (a zero-arg callable that queues something on
        self.tts) with the barge-in listener armed for its duration, then
        wait for playback to actually finish before disarming — arming
        only around a single turn, rather than for the whole voice loop,
        is what keeps this from ever conflicting with WakeWordDetector or
        HestiaSTT owning the microphone (see core/barge_in.py's
        docstring).
        """
        if not self._barge_in_enabled:
            speak_fn()
            self.tts.wait_until_done()
            return

        self.barge_in.reset()
        self.barge_in.start(on_barge_in=self.tts.stop)
        try:
            speak_fn()
            self.tts.wait_until_done()
        finally:
            self.barge_in.stop()

    # ------------------------------------------------------------------
    # Run-loops
    # ------------------------------------------------------------------

    def run_voice_loop(self) -> None:
        """
        Alternate between wake-word detection and STT, feeding into
        ``process_voice_turn``.

        Normally each cycle starts by waiting for the wake word again.
        But if the user barged in — started talking over Hestia's reply
        before it finished — the next cycle skips straight back to
        listening instead: they're already mid-sentence, so making them
        say "Hestia" again first would be exactly the walkie-talkie
        feeling barge-in exists to avoid.

        When BargeInListener captured the user's follow-up itself (see
        core/barge_in.py — it keeps recording, on the same mic stream,
        from the moment it noticed the interruption), that recording is
        used directly instead of calling stt.listen_once() again: a
        second, freshly-opened microphone stream would only start
        capturing *after* the barge-in listener's stream had already
        closed, losing whatever the user said in that gap. Falling back
        to stt.listen_once() (skip_wake_word) only happens if barge-in
        fired but for some reason didn't end up with usable audio (e.g.
        it hit max_capture_seconds with nothing but noise).
        """
        logger.info("Voice loop started — listening for wake word.")
        try:
            skip_wake_word = False
            pending_audio = None
            while True:
                if pending_audio is not None:
                    text = self.stt.transcribe_audio(pending_audio)
                    pending_audio = None
                elif skip_wake_word:
                    skip_wake_word = False
                    text = self.stt.listen_once(max_duration=_STT_MAX_DURATION)
                else:
                    if not self.wake_detector.listen_for_wake_word(
                        timeout=_WAKE_WORD_TIMEOUT
                    ):
                        continue

                    self.tts.speak("Yes?")
                    self.tts.wait_until_done()
                    text = self.stt.listen_once(max_duration=_STT_MAX_DURATION)

                if not text or len(text.strip()) < _MIN_VOICE_INPUT_LEN:
                    self.tts.speak("I didn't catch that.")
                    self.tts.wait_until_done()
                    continue

                if text.lower().strip() in _EXIT_WORDS:
                    self.tts.speak("Goodbye.")
                    self.tts.wait_until_done()
                    break

                self.process_voice_turn(text)

                if self._barge_in_enabled and self.barge_in.consume_triggered():
                    captured = self.barge_in.consume_captured_audio()
                    if captured is not None and len(captured) > 0:
                        pending_audio = captured
                    else:
                        skip_wake_word = True

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
            # force=True: a mid-capture barge-in listener would otherwise
            # wait for its own natural end-of-utterance (up to
            # max_capture_seconds) before stop() returns — fine during
            # normal turn-taking, but shutdown (e.g. Ctrl-C) should never
            # be held up by that.
            self.barge_in.stop(force=True)
        except Exception:
            logger.debug("barge_in.stop() raised; ignoring.")

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