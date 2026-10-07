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
import shutil
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

import signal  # noqa: E402

from core.observability import (  # noqa: E402  (must precede _configure_logging)
    Diagnostics,
    RequestIdFilter,
    Timer,
    current_request_id,
    new_request_id,
)


def _configure_logging(verbosity: str = "normal") -> logging.Logger:
    """
    Configure logging once, at import time, and again if --verbose/--quiet
    is passed (backlog #275).

    `verbosity` is one of:
      "quiet"   WARNING on Hestia's own logger  — daily use; only problems
      "normal"  INFO                            — the previous behaviour
      "verbose" DEBUG everywhere, including the routing/dispatch traces
                Hecate and the orchestrator already emit at debug level

    Every record also carries the in-flight request id (backlog #17) via
    RequestIdFilter, so one query's full trace across core/, modules/ and
    api.py can be grepped with a single `grep req=<id>`. The filter is
    attached to the root handler rather than to individual loggers so
    module-level `logging.getLogger(__name__)` calls anywhere in the tree
    pick it up without changes.
    """
    warnings.filterwarnings("ignore")
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(
            logging.DEBUG if verbosity == "verbose" else logging.ERROR
        )
    logging.basicConfig(
        level=logging.DEBUG if verbosity == "verbose" else logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  req=%(request_id)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    request_id_filter = RequestIdFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(request_id_filter)

    logger = logging.getLogger("hestia")
    logger.setLevel(
        {"quiet": logging.WARNING, "verbose": logging.DEBUG}.get(
            verbosity, logging.INFO
        )
    )
    return logger

logger = _configure_logging()

# ---------------------------------------------------------------------------
# Third-party + internal imports (after env / logging setup)
# ---------------------------------------------------------------------------

import yaml

from core.barge_in import BargeInListener
from core.echo_cancel import EchoReference, NLMSEchoCanceller
from core.mic_calibration import (
    current_input_device_name,
    format_report,
    load_calibration,
    run_calibration,
    save_calibration,
)
from core.voice_commands import (
    DND_OFF,
    DND_ON,
    DND_STATUS,
    REPEAT,
    SET_SENSITIVITY,
    VoiceCommand,
    parse_voice_command,
)
from core.voice_state import (
    STATE_INACTIVE,
    STATE_LISTENING,
    STATE_SPEAKING,
    STATE_THINKING,
    STATE_TYPED,
    STATE_WAKE,
    VoiceState,
)
from core.config_validation import ConfigError, validate_config, validate_or_raise
from core.module_loader import discover_skills
from core.query_splitter import candidate_segments
from core.intent_chains import apply_chain, detect_chain_reference
from core.hot_reload import FileWatcher
from core.browser_agent import HestiaBrowserAgent
from core.event_bus import bus
from core.heartbeat import HestiaHeartbeat
from core.consensus import ConsensusEngine
from core.conference import Conference, ollama_synthesizer
from core.whatif import WhatIfEngine
from core.db_maintenance import DBMaintenance
from core.intent_classifier import ClassifierService
from core.classifier_data import make_examples_fn
from core.shadow import ShadowRecorder, ShadowRules
from core.event_queue import EventBridge, EventQueue
from core.process_split import (
    CoreQueryServer, RoleProfile, SAY_TOPIC, Supervisor, VoiceFrontend,
    profile_for, queue_path,
)
from core.llm import HestiaLLM
from core.todoist_agent import TodoistAgent
from core.travel_time import TravelTimeEstimator
from core.nlu import HestiaNLU
from core.ollama_manager import OllamaManager
from core.stt import HestiaSTT
from core.tts import HestiaTTS, NullTTS
from core.wake_word import WakeWordDetector
from modules.apollo import ApolloEngine
from modules.ares import AresEngine
from modules.artemis import ArtemisEngine
from modules.chronos.engine import ChronosEngine
from modules.dionysus import DionysusEngine
from modules.hecate import HecateEngine
from modules.hecate.intent_registry import (
    MODULE_PREFIXES,
    module_for_intent,
    registry_info,
    strip_module_prefix,
)
from modules.hephaestus.engine import HephaestusEngine
from modules.hephaestus.monitor import MonitorStore
from modules.hephaestus.scrapers import ScraperRegistry
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
_MAX_VOICE_FAILURES = 3            # consecutive mic/STT errors before falling back to typing


def _format_minutes(minutes: float) -> str:
    """Spoken form of a duration: 30 -> "30 minutes", 90 -> "1 hour 30 minutes"."""
    total = int(round(minutes))
    hours, mins = divmod(total, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if mins or not hours:
        parts.append(f"{mins} minute{'s' if mins != 1 else ''}")
    return " ".join(parts)
_RECENT_CONTEXT_TURNS = 5


# ---------------------------------------------------------------------------
# HestiaBuilder
# ---------------------------------------------------------------------------

class _Skipped(Exception):
    """Internal: a component this process role doesn't build."""


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

    @staticmethod
    def _classifier_kwargs(cfg: dict, mode: str) -> dict:
        """``ClassifierService`` keyword arguments from the ``classifier:`` config
        block (backlog #25). Thresholds left out of the config stay ``None`` so the
        loaded model's own (calibrated) ones apply. An unknown backend falls back
        to tfidf; the transformer backend reads a model DIRECTORY, so a leftover
        ``.npz`` path from the tfidf setup is replaced by the default one."""
        backend = str(cfg.get("backend") or "tfidf").lower()
        if backend not in ("tfidf", "embedding", "ensemble", "transformer"):
            logger.warning("classifier.backend %r is not tfidf/embedding/ensemble/transformer; using tfidf.",
                           cfg.get("backend"))
            backend = "tfidf"
        model_path = cfg.get("model_path", "data/intent_classifier.npz")
        if backend == "transformer" and str(model_path).endswith(".npz"):
            model_path = "data/intent_distilbert"
        mp, mm = cfg.get("min_probability"), cfg.get("min_margin")
        return dict(
            model_path=model_path, mode=mode, backend=backend,
            min_prob=None if mp is None else float(mp),
            min_margin=None if mm is None else float(mm),
            examples_fn=make_examples_fn(augment=bool(cfg.get("augment", False)),
                                         augment_target=int(cfg.get("augment_target", 6))),
            embedding_model=cfg.get("embedding_model"),
            device=str(cfg.get("device", "cpu")),
        )

    def build_classifier(self) -> Optional[ClassifierService]:
        """The trained intent classifier (backlog #4), or None when off.

        Off unless ``classifier.mode`` is ``assist`` or ``primary``: it changes
        routing, so it is opt-in. Training runs on a background thread the
        first time, so startup is never delayed.
        """
        cfg = self.config.get("classifier") or {}
        raw_mode = cfg.get("mode", "off")
        # YAML 1.1 reads a bare `off` as boolean False; treat that as "off".
        mode = "off" if raw_mode in (False, None) else str(raw_mode).lower()
        if mode not in ("assist", "primary"):
            if mode != "off":
                logger.warning("classifier.mode %r is not off/assist/primary; classifier left off.", raw_mode)
            return None
        from modules.hecate.intent_registry import ALL_INTENTS
        svc = ClassifierService(valid_intents=ALL_INTENTS, **self._classifier_kwargs(cfg, mode))
        svc.ensure_ready(background=True)
        return svc

    def build_shadow(self) -> Optional[ShadowRecorder]:
        """Shadow-mode recorder (backlog #16), or None when no rules are enabled."""
        rules = ShadowRules.from_config(self.config.get("shadow"))
        for bad in rules.rejected:
            logger.warning("Shadow rule rejected: %s", bad)
        if not rules.enabled or not rules.rules:
            return None
        return ShadowRecorder(rules)

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
        # Optional features (Obsidian sync, arXiv monitoring, episode
        # embeddings) live under `mnemosyne:` in laptop_config.yaml and are
        # all off by default — see MnemosyneEngine.configure_extensions.
        configure = getattr(mnemosyne, "configure_extensions", None)
        if callable(configure):
            try:
                configure(self.config.get("mnemosyne") or {})
            except Exception:
                logger.exception("Mnemosyne extension config failed; using defaults.")
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
            # Separate headless agent for scheduled page monitors (#101); see
            # HephaestusEngine's docstring for why it isn't shared.
            "monitor_browser": None,
            # Handed through so HermesEngine can word email drafts (#93).
            "llm": llm,
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
                    # Opt-in (#99): also requests the Gmail modify scope so
                    # inbox-zero can archive mail after a "yes". Off by default;
                    # turning it on needs one re-authorisation.
                    allow_mailbox_changes=bool(
                        (self.config.get("hermes", {}) or {}).get("allow_mailbox_changes", False)
                    ),
                )
                agent.authenticate()
                modules["google_agent"] = agent
                logger.info("Google agent authenticated.")
            except Exception:
                logger.exception("Google agent failed to initialise; disabling.")

        # `browser:` options. `enabled` and `headless` were documented in the
        # example config but never read; they are now (#108). `--headed`
        # overrides headless from the command line.
        browser_cfg = self.config.get("browser", {}) or {}
        if browser_cfg.get("enabled", True):
            try:
                modules["browser_agent"] = HestiaBrowserAgent(
                    headless=bool(browser_cfg.get("headless", True)),
                    screenshot_dir=browser_cfg.get("screenshot_dir") or None,
                    slow_mo_ms=browser_cfg.get("slow_mo_ms", 0),
                    idle_timeout_seconds=browser_cfg.get("idle_timeout_seconds", 0),
                    pool_size=browser_cfg.get("pool_size", 1),
                )
                if ((self.config.get("hephaestus", {}) or {}).get("monitors") or {}).get("enabled", True):
                    modules["monitor_browser"] = HestiaBrowserAgent(
                        headless=True,
                        screenshot_dir=browser_cfg.get("screenshot_dir") or None,
                    )
            except Exception:
                logger.exception("Browser agent failed to initialise; disabling.")

        return modules

    # -- Orchestrator + module registration -----------------------------------

    def build_orchestrator(
        self,
        mnemosyne: MnemosyneEngine,
        optional_modules: dict[str, Any],
        diagnostics: Any = None,
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

        orchestrator = HestiaOrchestrator(
            ollama_cfg=self.ollama_cfg,
            # Backlog #12. Configurable via `hecate.session_ttl_seconds`
            # in laptop_config.yaml; defaults to 30 minutes if unset.
            session_ttl_seconds=float(
                self.config.get("hecate", {}).get("session_ttl_seconds", 1800)
            ),
        )
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
                # Backs the modules_status / explain_routing /
                # report_mistake intents (backlog #3, #8, #259).
                diagnostics=diagnostics,
            )
        )

        # Memory
        orchestrator.register(mnemosyne)

        # Optional knowledge modules
        for mod in (athena, iris):
            if mod is not None:
                orchestrator.register(mod)

        # Time / calendar / communication
        # Chronos options (backlog #81-#90). Every key is optional; the
        # defaults reproduce the pre-#81 behaviour plus the always-safe
        # extras (recurring / snooze / ICS), so an old config keeps working.
        chronos_cfg = self.config.get("chronos", {}) or {}
        chronos = ChronosEngine(
            memory=mnemosyne,
            local_tz=chronos_cfg.get("timezone", "Asia/Kolkata"),
            skip_public_holidays=bool(chronos_cfg.get("skip_public_holidays", False)),
            holiday_country=chronos_cfg.get("holiday_country"),
            exports_dir=chronos_cfg.get("exports_dir"),
            default_snooze_minutes=chronos_cfg.get("default_snooze_minutes", 10),
            proactive_weather=bool(chronos_cfg.get("proactive_weather", False)),
        )
        orchestrator.register(chronos)
        artemis_cfg = self.config.get("artemis") or {}
        artemis = ArtemisEngine(ollama_cfg=self.ollama_cfg,
                                habit_grace_days=int(artemis_cfg.get("habit_grace_days", 0) or 0),
                                timezone_name=artemis_cfg.get("timezone")
                                or chronos_cfg.get("timezone", "Asia/Kolkata"),
                                nudges=artemis_cfg.get("nudges") or {})
        orchestrator.register(artemis)

        hermes = None
        # Todoist (#91) is independent of Google: a token alone is enough to
        # get Hermes' task intents.
        todoist_cfg = self.config.get("todoist", {}) or {}
        todoist_agent = None
        if todoist_cfg.get("enabled", False):
            todoist_agent = TodoistAgent(
                token=todoist_cfg.get("api_token"),
                timeout=float(todoist_cfg.get("timeout_seconds", 10)),
            )
            if not todoist_agent.is_ready():
                logger.warning(
                    "todoist.enabled is true but no token is set (todoist.api_token "
                    "or TODOIST_API_TOKEN); Todoist intents will say it isn't connected."
                )
        if google_agent or todoist_agent is not None:
            # Same config key ChronosEngine uses above — without this,
            # HermesEngine defaults to UTC and every created event lands
            # offset by the difference between UTC and the user's real
            # timezone (e.g. "3pm" becomes "8:30pm" for Asia/Kolkata).
            hermes_tz = chronos_cfg.get("timezone", "Asia/Kolkata")
            # Hermes options (backlog #92-#100). Every key is optional; see
            # config/laptop_config.example.yaml.
            hermes_cfg = self.config.get("hermes", {}) or {}
            work_hours = hermes_cfg.get("work_hours")
            # Driving-time lookups for "back-to-back" checks (#95). The default
            # provider "flat" never calls out; osrm/google send event
            # locations to that service.
            travel_cfg = hermes_cfg.get("travel") or {}
            travel = TravelTimeEstimator(
                provider=str(travel_cfg.get("provider", "flat")),
                google_api_key=str(travel_cfg.get("google_api_key") or ""),
            )
            hermes = HermesEngine(
                google_agent,
                timezone_name=hermes_tz,
                llm=optional_modules.get("llm"),
                contacts=hermes_cfg.get("contacts") or {},
                vip_senders=hermes_cfg.get("vip_senders") or [],
                work_hours=tuple(work_hours) if work_hours else None,
                buffer_minutes=hermes_cfg.get("buffer_minutes", 10),
                travel_minutes=hermes_cfg.get("travel_minutes", 30),
                travel=travel,
                todoist=todoist_agent,
                allow_mailbox_changes=bool(hermes_cfg.get("allow_mailbox_changes", False)),
                digest_time=hermes_cfg.get("digest_time"),
                state_path=hermes_cfg.get("state_path", "data/hermes_state.json"),
            )
            orchestrator.register(hermes)

        # Hephaestus options (backlog #101-#110). Every key is optional; see
        # config/laptop_config.example.yaml.
        heph_cfg = self.config.get("hephaestus", {}) or {}
        mon_cfg = heph_cfg.get("monitors", {}) or {}
        monitor_store = None
        if browser_agent is not None and mon_cfg.get("enabled", True):
            try:
                db_path = str(mon_cfg.get("db_path", "data/hephaestus.db"))
                if db_path != ":memory:":
                    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
                monitor_store = MonitorStore(
                    db_path, max_monitors=int(mon_cfg.get("max_monitors", 20)),
                )
            except Exception:
                logger.exception("Hephaestus monitor store failed to open; page watching disabled.")
        scrapers = ScraperRegistry()
        scrapers.load_config(heph_cfg.get("scrapers"))
        scrapers.load_plugins(heph_cfg.get("scraper_dir"))
        orchestrator.register(
            HephaestusEngine(
                browser_agent,
                app_map=heph_cfg.get("app_map"),
                monitor_store=monitor_store,
                monitor_browser=optional_modules.get("monitor_browser"),
                scrapers=scrapers,
                forms=heph_cfg.get("forms") or {},
                forms_store_path=heph_cfg.get("forms_store_path", "data/hephaestus_forms.json"),
                llm=optional_modules.get("llm"),
                repo_review=heph_cfg.get("repo_review") or {},
                repo_roots=heph_cfg.get("repo_roots") or [],
                min_host_interval=heph_cfg.get("min_request_interval_seconds", 1.0),
                quiet_hours=mon_cfg.get("quiet_hours", (22, 7)),
            )
        )

        # Specialist modules
        # Apollo's `apollo:` block (units, hydration, weekly summary, step
        # import folder, ...). Its timezone defaults to Chronos's so "today"
        # means the same day everywhere.
        apollo_cfg = dict(self.config.get("apollo") or {})
        apollo_cfg.setdefault(
            "timezone",
            (self.config.get("chronos") or {}).get("timezone", "Asia/Kolkata"),
        )
        apollo = ApolloEngine(ollama_cfg=self.ollama_cfg, config=apollo_cfg)
        orchestrator.register(apollo)
        # Ares keeps tracked decisions, outcomes and playbooks (#154, #155,
        # #157) in its own SQLite file; `ares:` in laptop_config.yaml is optional.
        ares_cfg = self.config.get("ares") or {}
        ares_db_path = ares_cfg.get("db_path") or str(
            Path(__file__).resolve().parent / "data" / "ares" / "ares.db"
        )
        os.makedirs(os.path.dirname(os.path.abspath(ares_db_path)), exist_ok=True)
        orchestrator.register(
            AresEngine(
                memory=mnemosyne,
                ollama_cfg=self.ollama_cfg,
                db_path=ares_db_path,
                auto_review_days=ares_cfg.get("auto_review_days"),
            )
        )
        # Writing pair (backlog #164, #168, #169, #270). Orpheus and Metis
        # are wired to each other so a writing session can chain
        # draft -> critique/polish, and Orpheus can request Metis's optional
        # polish pass. Options live under `writing:` in laptop_config.yaml.
        writing_cfg = self.config.get("writing", {}) or {}
        orpheus = OrpheusEngine(
            ollama_cfg=self.ollama_cfg, memory=mnemosyne,
            export_dir=writing_cfg.get("export_dir"),
            polish_default=bool(writing_cfg.get("polish_pass", False)),
        )
        metis = MetisEngine(
            ollama_cfg=self.ollama_cfg, memory=mnemosyne,
            export_dir=writing_cfg.get("export_dir"),
        )
        orpheus.attach_metis(metis)
        metis.attach_orpheus(orpheus)
        # Plagiarism spot-check needs a live search. Read-only, no
        # confirmation needed; without a browser Metis falls back to
        # handing you the passages to search yourself.
        if (
            browser_agent is not None
            and bool(writing_cfg.get("plagiarism_web_check", True))
            and hasattr(browser_agent, "search_web_results")
        ):
            metis.attach_web_search(
                browser_agent.search_web_results,
                getattr(browser_agent, "get_page_text", None),
            )
        orchestrator.register(orpheus)
        orchestrator.register(metis)
        # Backlog #149: mood-aware picks, only when the user gave no mood.
        dionysus_mood_aware = bool(
            (self.config.get("dionysus") or {}).get("mood_aware", True)
        )
        # Backlog #147: dismissals stop counting after this many days
        # (0 = never expire).
        dionysus_expire_days = (self.config.get("dionysus") or {}).get(
            "dismiss_expire_days", 180
        )
        dionysus = DionysusEngine(
            ollama_cfg=self.ollama_cfg,
            browser_agent=browser_agent,
            memory=mnemosyne,
            mood_aware=dionysus_mood_aware,
            dismiss_expire_days=dionysus_expire_days,
        )
        _attach_apollo = getattr(dionysus, "attach_apollo", None)
        if dionysus_mood_aware and callable(_attach_apollo):
            _attach_apollo(apollo)
        orchestrator.register(dionysus)
        # Chronos is registered before these exist, so hand them over now:
        # the "what's on my plate" timeline (#86) reads Hermes + Artemis, and
        # weather-triggered suggestions (#88) can offer Dionysus's indoor plans.
        chronos.attach_sources(hermes=hermes, artemis=artemis, dionysus=dionysus)
        # Backlog #152: recharge routines are repeating Chronos reminders.
        _attach_chronos = getattr(dionysus, "attach_chronos", None)
        if callable(_attach_chronos):
            _attach_chronos(chronos)
        pluto = PlutoEngine(ollama_cfg=self.ollama_cfg)
        orchestrator.register(pluto)
        # Cross-module reads for Apollo (#126 habit/mood, #161 burnout
        # signals). Read-only: Apollo never writes to Artemis or Pluto.
        for _hook, _target in (("attach_artemis", artemis), ("attach_pluto", pluto)):
            _fn = getattr(apollo, _hook, None)
            if callable(_fn):
                _fn(_target)
        # Backlog #159: surface disagreements between Apollo ("rest") and
        # Artemis ("push"). Append-only; `consensus.enabled: false` kills it.
        consensus_cfg = self.config.get("consensus") or {}
        consensus_engine = None
        if consensus_cfg.get("enabled", True) and hasattr(orchestrator, "attach_consensus"):
            consensus_engine = ConsensusEngine(
                apollo=apollo,
                artemis=artemis,
                intents=consensus_cfg.get("intents") or None,
            )
            orchestrator.attach_consensus(consensus_engine)
        # Backlog #158 / #160: cross-module reasoning, convened here because
        # this is where every module is in hand. Both are read-only, and both
        # have a kill switch. Hecate decides who sits at a conference; this
        # only supplies the means to ask them.
        conference_cfg = self.config.get("conference") or {}
        if conference_cfg.get("enabled", True) and hasattr(orchestrator, "attach_conference"):
            orchestrator.attach_conference(
                Conference(
                    orchestrator.call_module,
                    synthesize=(
                        ollama_synthesizer(self.ollama_cfg)
                        if conference_cfg.get("llm_summary", True) else None
                    ),
                    consensus=consensus_engine,
                )
            )
        whatif_cfg = self.config.get("whatif") or {}
        if whatif_cfg.get("enabled", True) and hasattr(orchestrator, "attach_whatif"):
            orchestrator.attach_whatif(
                WhatIfEngine(pluto=pluto, artemis=artemis, apollo=apollo)
            )

        # Drop-in skills (backlog #9): single-file BaseModule subclasses
        # under `skills.path`, auto-discovered and registered here rather
        # than requiring an edit to this method for every new one. See
        # core/module_loader.py for the file convention. Every built-in
        # module above is already registered by this point, so a skill
        # cannot accidentally shadow one — discover_skills() checks
        # registered_modules and skips any name collision.
        skills_cfg = self.config.get("skills", {})
        if skills_cfg.get("enabled", True):
            for skill in discover_skills(
                skills_cfg.get("path", "skills"),
                ollama_cfg=self.ollama_cfg,
                memory=mnemosyne,
                skip_names=orchestrator.registered_modules,
            ):
                orchestrator.register(skill)

        logger.info(
            "Orchestrator ready (%d module(s) registered).",
            len(orchestrator.registered_modules),
        )
        return orchestrator, apollo, pluto, artemis, chronos

    # -- I/O --------------------------------------------------------------

    def build_io(
        self,
        only: Optional[frozenset] = None,
    ) -> tuple[
        Optional[HestiaSTT],
        "HestiaTTS | NullTTS",
        Optional[WakeWordDetector],
        Optional[BargeInListener],
    ]:
        """Build the voice I/O components, degrading instead of crashing.

        Each component is constructed independently (backlog #179): a missing
        Vosk model, a Whisper load failure or an absent audio stack costs you
        *that* feature, not the whole assistant. A component that fails comes
        back as ``None`` (``NullTTS`` for speech) and the reason is recorded in
        ``self.io_errors`` so the voice loop can fall back to typed input and
        say why. Hestia itself still starts, text/web/Telegram keep working.
        """
        stt_cfg = self.config.get("stt", {})
        tts_cfg = self.config.get("tts", {})
        wake_cfg = self.config.get("wake_word", {})
        barge_in_cfg = self.config.get("barge_in", {})
        self.io_errors: dict[str, str] = {}
        # Process roles (#20) build only some components; None builds all.
        want = (lambda name: only is None or name in only)

        # Saved per-device calibration (backlog #174): fills in only the
        # settings the config leaves unset, and is ignored if it was made
        # on a different input device.
        calib: Optional[dict] = None
        if barge_in_cfg.get("use_calibration", True):
            try:
                calib = load_calibration(device=current_input_device_name())
            except Exception:
                calib = None
        if calib:
            logger.info(
                "Using saved mic calibration (min_rms=%g, vad=%d, wake=%s). "
                "Settings in your config take precedence.",
                calib["min_rms"], calib["vad_aggressiveness"], calib["wake_sensitivity"],
            )
        calib_vad = calib["vad_aggressiveness"] if calib else 2

        stt: Optional[HestiaSTT] = None
        try:
            if not want("stt"):
                raise _Skipped()
            stt = HestiaSTT(
                model_size=stt_cfg.get("model_size", "base.en"),
                device=stt_cfg.get("device", "cuda"),
                compute_type=stt_cfg.get("compute_type", "int8"),
                samplerate=stt_cfg.get("samplerate", 16000),
                noise_filter=stt_cfg.get("noise_filter", True),
                silence_frames=stt_cfg.get("silence_frames", 33),
                vad_aggressiveness=stt_cfg.get("vad_aggressiveness", calib_vad),
            )
        except _Skipped:
            pass
        except Exception as exc:
            self.io_errors["stt"] = str(exc) or exc.__class__.__name__
            logger.warning("Speech-to-text unavailable: %s", self.io_errors["stt"])

        # Echo cancellation needs a playback reference, which only Piper
        # exposes (backlog #175). Off unless explicitly enabled.
        ec_cfg = barge_in_cfg.get("echo_cancel") or {}
        echo_reference = None
        if ec_cfg.get("enabled", False):
            if tts_cfg.get("engine", "pyttsx3") == "piper":
                echo_reference = EchoReference()
            else:
                logger.warning(
                    "barge_in.echo_cancel.enabled needs tts.engine: piper "
                    "(pyttsx3 plays through the OS, so there is no playback "
                    "reference to cancel); leaving echo cancellation off."
                )

        try:
            if not want("tts"):
                raise _Skipped()
            tts = HestiaTTS(
                engine=tts_cfg.get("engine", "pyttsx3"),
                rate=tts_cfg.get("rate", 175),
                volume=tts_cfg.get("volume", 1.0),
                piper_model_path=tts_cfg.get("piper_model_path"),
                voices=tts_cfg.get("voices"),
                echo_reference=echo_reference,
            )
        except _Skipped:
            tts = NullTTS()
        except Exception as exc:
            tts = NullTTS()
            self.io_errors["tts"] = str(exc) or exc.__class__.__name__
            logger.warning("Text-to-speech unavailable: %s", self.io_errors["tts"])

        if echo_reference is not None and getattr(tts, "engine", None) != "piper":
            logger.warning(
                "Piper isn't the active TTS engine (model missing?); "
                "echo cancellation is off."
            )
            echo_reference = None

        wake_detector: Optional[WakeWordDetector] = None
        try:
            if not want("wake"):
                raise _Skipped()
            wake_detector = WakeWordDetector(
                model_path=wake_cfg.get("model_path", "models/vosk-model-small-en-us-0.15"),
                wake_words=wake_cfg.get("wake_words"),
                sensitivity=wake_cfg.get(
                    "sensitivity", calib["wake_sensitivity"] if calib else "normal"
                ),
            )
        except _Skipped:
            pass
        except Exception as exc:
            self.io_errors["wake_word"] = str(exc) or exc.__class__.__name__
            logger.warning("Wake-word detection unavailable: %s", self.io_errors["wake_word"])

        # Barge-in is opt-out (default true): it only ever runs while
        # Hestia is speaking (see Hestia._speak_streaming /
        # _speak_with_barge_in), so leaving it enabled costs nothing when
        # the user never interrupts, and lets them the moment they do.
        barge_in: Optional[BargeInListener] = None
        try:
            if not want("barge"):
                raise _Skipped()
            canceller = None
            if echo_reference is not None:
                canceller = NLMSEchoCanceller(
                    echo_reference,
                    filter_len=int(ec_cfg.get("filter_len", 1024)),
                    mu=float(ec_cfg.get("mu", 0.4)),
                    delay_ms=float(ec_cfg.get("delay_ms", 0.0)),
                )
            barge_in = BargeInListener(
                samplerate=barge_in_cfg.get("samplerate", stt_cfg.get("samplerate", 16000)),
                vad_aggressiveness=barge_in_cfg.get("vad_aggressiveness", calib_vad),
                speech_frames_to_trigger=barge_in_cfg.get("speech_frames_to_trigger", 3),
                # See core/barge_in.py's docstring: min_rms is the cheap
                # lever against false self-interruptions from Hestia's own
                # voice bleeding into the mic on shared speaker/mic
                # hardware (e.g. a laptop). `python main.py --calibrate-mic`
                # measures a value for this device; an explicit value here
                # always wins over the saved calibration. Raise it if she's
                # interrupting herself; lower it (or use a headset, the
                # real fix) if real interruptions go unnoticed.
                min_rms=barge_in_cfg.get("min_rms", calib["min_rms"] if calib else 300.0),
                pre_roll_frames=barge_in_cfg.get("pre_roll_frames", 10),
                post_trigger_silence_frames=barge_in_cfg.get(
                    "post_trigger_silence_frames", stt_cfg.get("silence_frames", 33) - 8
                ),
                max_capture_seconds=barge_in_cfg.get("max_capture_seconds", 12.0),
                echo_canceller=canceller,
            )
        except _Skipped:
            pass
        except Exception as exc:
            self.io_errors["barge_in"] = str(exc) or exc.__class__.__name__
            logger.warning("Barge-in unavailable: %s", self.io_errors["barge_in"])

        return stt, tts, wake_detector, barge_in

    # -- Heartbeat / web UI / sync API ------------------------------------

    def build_heartbeat(
        self, mnemosyne: MnemosyneEngine, diagnostics: Any = None,
        apollo: Any = None, artemis: Any = None, hephaestus: Any = None,
        pluto: Any = None, classifier: Any = None, hermes: Any = None,
    ) -> HestiaHeartbeat:
        # diagnostics powers the nightly low-confidence review (backlog
        # #6); optional, so a heartbeat built without one just never runs
        # that job, same as every other diagnostics-gated feature.
        maintenance = None
        maint_cfg = self.config.get("maintenance") or {}
        if maint_cfg.get("enabled", True):
            maintenance = DBMaintenance(
                paths=maint_cfg.get("extra_paths") or None,
                free_ratio_threshold=float(maint_cfg.get("free_ratio_threshold", 0.2)),
                min_interval_days=int(maint_cfg.get("min_interval_days", 7)),
            )
        return HestiaHeartbeat(
            interval=1800, mnemosyne=mnemosyne, diagnostics=diagnostics,
            apollo=apollo, maintenance=maintenance, artemis=artemis,
            hephaestus=hephaestus, pluto=pluto, classifier=classifier,
            hermes=hermes,
        )

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
        voice_state: Optional[Any] = None,
        iris: Optional[Any] = None,
        process_traced_fn=None,
        config_path: Optional[Path] = None,
    ) -> Optional[Any]:
        try:
            from web_ui import HestiaWebUI

            # `webui:` in laptop_config.yaml. host/port were validated but never
            # passed on, so the page always bound to 127.0.0.1:5000. Secrets
            # prefer the environment so they need not live in the YAML.
            wcfg = self.config.get("webui") or {}
            web_ui = HestiaWebUI(
                memory=mnemosyne,
                process_fn=process_fn,
                host=str(wcfg.get("host") or "127.0.0.1"),
                port=int(wcfg.get("port") or 5000),
                api_key=os.environ.get("HESTIA_WEB_API_KEY") or wcfg.get("api_key") or None,
                password=os.environ.get("HESTIA_WEB_PASSWORD") or wcfg.get("password") or None,
                password_hash=wcfg.get("password_hash") or None,
                secret_key=os.environ.get("HESTIA_WEB_SECRET") or wcfg.get("secret_key") or None,
                session_hours=float(wcfg.get("session_hours") or 12),
                cookie_secure=bool(wcfg.get("cookie_secure", False)),
                allowed_origins=wcfg.get("allowed_origins") or None,
                apollo=apollo,
                pluto=pluto,
                artemis=artemis,
                chronos=chronos,
                athena=athena,
                iris=iris,
                skill_loader=skill_loader,
                stt=stt,
                tts=tts,
                voice_state=voice_state,
                process_traced_fn=process_traced_fn,
                config_path=config_path,
            )
            web_ui.start()
            logger.info("Web UI started.")
            return web_ui
        except Exception:
            logger.exception("Web UI failed to start; continuing without it.")
            return None

    def build_telegram_bot(
        self,
        process_fn,
        stt: Optional[HestiaSTT],
        memory: Optional[Any] = None,
        *,
        classify_fn=None,
        pending_fn=None,
        snooze_fn=None,
        ingest_photo_fn=None,
        ingest_document_fn=None,
        should_push_fn=None,
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

        The keyword hooks are what turn the bot from a text relay into the
        full Telegram front-end (backlog #191-#196); each is optional and the
        bot degrades to its plain behaviour without it. Config under
        ``telegram:`` (all optional, see laptop_config.example.yaml):
        ``roles`` / ``role_policies`` (per-chat intent scoping, #194),
        ``push_notifications`` (mirror reminders and other proactive
        announcements to the owner chats, with Snooze/Done buttons, #191) and
        ``snooze_minutes``.
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
                roles=telegram_cfg.get("roles"),
                role_policies=telegram_cfg.get("role_policies"),
                classify_fn=classify_fn,
                pending_fn=pending_fn,
                snooze_fn=snooze_fn,
                ingest_photo_fn=ingest_photo_fn,
                ingest_document_fn=ingest_document_fn,
                should_push_fn=should_push_fn,
                snooze_minutes=telegram_cfg.get("snooze_minutes") or (10, 60),
            )
            if telegram_cfg.get("push_notifications", False):
                bot.attach_event_bus(bus)
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

    def start_sync_api(
        self,
        mnemosyne: MnemosyneEngine,
        diagnostics: Any = None,
        nlu: Any = None,
    ) -> None:
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

            # diagnostics/nlu power /health/modules (backlog #19); both
            # are optional there, so passing None degrades the endpoint to
            # status="unknown" rather than breaking the sync API.
            sync_app = create_app(api_key=api_key, diagnostics=diagnostics, nlu=nlu)
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

    # Last thing Hestia said in reply to you or announced proactively, for
    # "repeat that" (backlog #171). Deliberately NOT updated by "Yes?" /
    # "I didn't catch that." / local-command acknowledgements.
    _last_spoken: str = ""
    # Routing record of the query this thread handled last (see
    # process_text_traced). Thread-local so concurrent web/Telegram/voice
    # queries cannot read each other's.
    _trace_local = threading.local()
    _voice_io_errors: dict = {}

    @property
    def voice_state(self) -> VoiceState:
        """Shared listening-state + do-not-disturb holder (see
        core/voice_state.py). Created on first use so the object also exists
        on instances built without running __init__."""
        vs = self.__dict__.get("_voice_state")
        if vs is None:
            vs = VoiceState()
            self.__dict__["_voice_state"] = vs
        return vs

    # Class-level defaults so a bare/stand-in instance behaves as the original
    # single process (backlog #20).
    role: str = "all"
    profile: Optional[RoleProfile] = None
    _queue: Optional[EventQueue] = None

    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG, headed: bool = False,
                 role: str = "all") -> None:
        logger.info("Initialising Hestia…")
        # Process role (backlog #20). "all" is the original single process; "core"
        # and "jobs" build only their half (see core/process_split.py).
        self.role: str = role
        self.profile: RoleProfile = profile_for(role)
        self._queue: Optional[EventQueue] = None
        self._bridge: Optional[EventBridge] = None
        self._query_server: Optional[CoreQueryServer] = None
        self._config_path = Path(config_path)
        self.config = _load_config(self._config_path)
        if headed:
            # --headed (#108): show the browser and slow it down so you can
            # follow what it's doing. An explicit slow_mo_ms in the config wins.
            browser_cfg = self.config.setdefault("browser", {})
            browser_cfg["headless"] = False
            browser_cfg.setdefault("slow_mo_ms", 250)
        builder = HestiaBuilder(self.config)

        # Derived config sections (read-only after __init__)
        self._ollama_cfg: dict[str, Any] = builder.ollama_cfg
        self._google_cfg: dict[str, Any] = builder.google_cfg
        self._sync_cfg: dict[str, Any] = builder.sync_cfg

        # -- Construction (via builder), in dependency order --------------
        self.ollama_manager = builder.build_ollama_manager()

        self.llm = builder.build_llm(self.ollama_manager)
        self.nlu = builder.build_nlu()
        # Trained intent classifier (backlog #4): off unless classifier.mode is set.
        self.classifier = builder.build_classifier()
        self.nlu.attach_classifier(self.classifier)
        if role != "all":
            self._queue = EventQueue(queue_path(self.config), origin=role)

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

        # Observability first: the routing log has to be ready before the
        # very first query, and Diagnostics is injected into CoreModule
        # during orchestrator construction below. The orchestrator itself
        # is bound afterwards (chicken-and-egg: modules_status needs the
        # orchestrator, the orchestrator's CoreModule needs Diagnostics).
        self.diagnostics = Diagnostics()

        optional_modules = builder.build_optional_modules(self.llm)
        self.athena        = optional_modules["athena"]
        self.iris           = optional_modules["iris"]
        self.google_agent   = optional_modules["google_agent"]
        self.browser_agent: Optional[HestiaBrowserAgent] = optional_modules["browser_agent"]

        # Quizzes (#34) draw their questions from Athena's documents, and the
        # arXiv monitor (#47) hands new papers to Athena's ingest.
        attach_athena = getattr(self.mnemosyne, "attach_athena", None)
        if self.athena is not None and callable(attach_athena):
            attach_athena(self.athena)

        self.orchestrator, self.apollo, self.pluto, self.artemis, self.chronos = (
            builder.build_orchestrator(
                self.mnemosyne, optional_modules, diagnostics=self.diagnostics
            )
        )
        self.diagnostics.bind_orchestrator(self.orchestrator)
        self.orchestrator.attach_classifier(self.classifier)          # backlog #4
        self.orchestrator.attach_event_bus(bus)                       # backlog #10
        self.shadow = builder.build_shadow()                          # backlog #16
        if self.shadow is not None:
            self.orchestrator.attach_shadow(self.shadow)
            logger.info("Shadow mode on for %d rule(s).", len(self.shadow.rules.rules))

        prof = self.profile
        io_parts = frozenset(
            name for name, on in (("stt", prof.stt), ("tts", prof.tts),
                                  ("wake", prof.wake_word), ("barge", prof.barge_in)) if on)
        self.stt, self.tts, self.wake_detector, self.barge_in = builder.build_io(only=io_parts)
        self._voice_io_errors = dict(getattr(builder, "io_errors", {}) or {})
        # Barge-in needs a working listener; if it failed to build there is
        # simply nothing to arm (see build_io).
        self._barge_in_enabled: bool = bool(
            self.config.get("barge_in", {}).get("enabled", True)
            and self.barge_in is not None
        )

        # -- Wiring: connect already-built subsystems together -------------
        self._init_event_bus()

        self.hephaestus = getattr(self.orchestrator, "_modules", {}).get("hephaestus")
        self.hermes = getattr(self.orchestrator, "_modules", {}).get("hermes")
        self.heartbeat = builder.build_heartbeat(
            self.mnemosyne, diagnostics=self.diagnostics, apollo=self.apollo,
            artemis=self.artemis, hephaestus=self.hephaestus, pluto=self.pluto,
            classifier=self.classifier, hermes=self.hermes,
        )
        # Chronos owns reminder delivery (recurring, snooze, location and the
        # missed-reminder catch-up on startup - backlog #81-#89). When its
        # scheduler is running the heartbeat must not also fire one-shot
        # reminders, or each would be announced twice.
        chronos_cfg = self.config.get("chronos", {}) or {}
        if chronos_cfg.get("scheduler_enabled", True) and prof.chronos_scheduler:
            try:
                started = self.chronos.start_scheduler(
                    chronos_cfg.get("scheduler_interval_seconds")
                )
            except Exception:
                logger.exception("Chronos scheduler failed to start; heartbeat keeps reminders.")
                started = False
            if started:
                self.heartbeat.handle_reminders = False
                logger.info("Chronos scheduler started; heartbeat reminders disabled.")
        if prof.heartbeat:
            self.heartbeat.start()
            logger.info("Heartbeat started (interval=1800 s).")
        else:
            logger.info("Heartbeat not started in the %s process.", role)

        self.web_ui = None if not prof.web_ui else builder.build_web_ui(
            self.mnemosyne,
            self.process_text,
            self.apollo,
            pluto=self.pluto,
            artemis=self.artemis,
            chronos=self.chronos,
            athena=self.athena,
            stt=self.stt,
            # A NullTTS has nothing to synthesise with; hand the web UI None
            # so /api/tts reports "not available" instead of failing per call.
            tts=self.tts if getattr(self.tts, "available", True) else None,
            voice_state=self.voice_state,
            iris=self.iris,
            process_traced_fn=self.process_text_traced,
            config_path=self._config_path,
        )

        self.telegram_bot = None if not prof.telegram else builder.build_telegram_bot(
            self.process_text, self.stt, self.mnemosyne,
            classify_fn=self._classify_for_telegram,
            pending_fn=self._has_pending_confirmation,
            snooze_fn=self._telegram_snooze,
            ingest_photo_fn=self._telegram_ingest_photo,
            ingest_document_fn=self._telegram_ingest_document,
            should_push_fn=lambda: not self.voice_state.dnd_active(),
        )

        if prof.sync_api:
            builder.start_sync_api(
                self.mnemosyne, diagnostics=self.diagnostics, nlu=self.nlu
            )

        # Cross-process wiring (backlog #20): bridge chosen bus topics to the
        # local queue, and let the voice process ask this one questions.
        if self._queue is not None:
            self._bridge = EventBridge(
                bus, self._queue, consumer=f"bridge.{role}",
                export=prof.export_topics, imports=prof.import_topics)
            self._bridge.start()
            if prof.serve_queries:
                self._query_server = CoreQueryServer(self._queue, self.process_text)
                self._query_server.start()
            logger.info("Process role %r connected to the queue at %s.", role, self._queue.path)

        self._start_hot_reload_watchers()

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

        # TTS output (proactive notifications: reminders, nudges, the
        # morning brief...). Honours do-not-disturb (backlog #173): while it
        # is on, announcements are held and read out together afterwards.
        def _on_speak(data: dict) -> None:
            try:
                text = data.get("text", "") or ""
                vs = self.voice_state
                if vs.dnd_active():
                    vs.hold(text)
                    logger.info(
                        "Do-not-disturb: held a notification (%d waiting).",
                        vs.held_count(),
                    )
                    return
                # A timed DND that ran out since the last notification:
                # read what piled up first, in the same utterance (speak()
                # cancels earlier speech, so two calls would clip the digest).
                digest = self._held_notifications_digest()
                if digest:
                    text = f"{digest} {text}".strip()
                self._last_spoken = text
                voice = data.get("voice") or self._voice_profile_for_module(data.get("module"))
                self._speak(text, voice)
            except Exception:
                logger.exception("speak handler failed.")

        if self.role != "jobs":
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
    # Voice helpers: per-module voices, local commands, do-not-disturb
    # ------------------------------------------------------------------

    def _speak(self, text: str, voice: Optional[str] = None) -> None:
        """tts.speak(), adding ``voice=`` only when a profile was chosen so
        the default call stays exactly ``speak(text)``.

        In split mode (#20) there is no speaker here: the jobs process hands
        the text to core as a ``speak`` notification (core applies
        do-not-disturb), and core queues it for the voice process."""
        if self.role == "jobs":
            bus.emit("speak", {"text": text, "voice": voice} if voice else {"text": text})
            return
        if self.profile is not None and self.profile.speak_via_queue and self._queue is not None:
            self._queue.publish(SAY_TOPIC, {"text": text, "voice": voice})
            return
        if voice:
            self.tts.speak(text, voice=voice)
        else:
            self.tts.speak(text)

    def _voice_profile_for_module(self, module: Optional[str]) -> Optional[str]:
        """Voice profile configured for *module* (backlog #170), else the
        ``default`` mapping, else None (= the base voice). Configured via
        ``tts.voices`` (the profiles) and ``tts.voice_by_module`` (who uses
        which). A mapping that names a profile that doesn't exist is ignored
        rather than failing."""
        tts_cfg = (getattr(self, "config", None) or {}).get("tts") or {}
        mapping = tts_cfg.get("voice_by_module") or {}
        if not mapping or not tts_cfg.get("voices"):
            return None
        profile = mapping.get(module) if module else None
        if not profile:
            profile = mapping.get("default")
        if not profile:
            return None
        has_voice = getattr(self.tts, "has_voice", None)
        if callable(has_voice) and not has_voice(profile):
            return None
        return profile

    def _voice_profile_for_intent(self, intent: Optional[str]) -> Optional[str]:
        """Resolve an NLU intent to its owning module, then to a voice."""
        if not intent:
            return self._voice_profile_for_module(None)
        module = module_for_intent(intent) or next(
            (p.rstrip("_") for p in MODULE_PREFIXES if intent.startswith(p)), None
        )
        return self._voice_profile_for_module(module)

    def _held_notifications_digest(self) -> str:
        """Collect notifications held during do-not-disturb into one
        utterance ("" if DND is still on or nothing was held)."""
        vs = self.voice_state
        if vs.dnd_active():
            return ""
        held = vs.release_held()
        if not held:
            return ""
        n = len(held)
        return (
            f"While notifications were muted, {n} came in. " + " ".join(held)
        )

    # ------------------------------------------------------------------
    # Telegram hooks (backlog #191-#196). The bot owns the transport; these
    # give it the bits of Hestia it needs without it importing main.
    # ------------------------------------------------------------------

    def _classify_for_telegram(self, text: str) -> list[str]:
        """Every intent *text* could trigger, so a restricted chat's role can be
        enforced before anything runs (#194). Mirrors process_text: the whole
        message and, for a compound query, each half are classified, and all of
        them must be allowed. Assistant-voice controls (do-not-disturb, "repeat
        that") run before NLU and act on the owner's machine, so they are
        reported as an unregistered intent and a restricted chat is refused."""
        cleaned = _clean_input(text)
        if not cleaned:
            return []
        if parse_voice_command(cleaned) is not None:
            return ["voice_control"]
        context = self.mnemosyne.get_recent(_RECENT_CONTEXT_TURNS)
        candidates = [cleaned]
        segments = candidate_segments(cleaned)
        if len(segments) == 2:
            candidates.extend(segments)
        return [self.nlu.understand(c, context).get("intent", "chat") for c in candidates]

    def _has_pending_confirmation(self) -> bool:
        """True while the orchestrator is waiting on a yes/no (drives the
        Confirm/Cancel buttons and keeps confirmations with the chat that
        raised them)."""
        return getattr(self.orchestrator, "_pending", None) is not None

    def _telegram_snooze(self, minutes: int) -> str:
        """Snooze button: snooze the most recently fired reminder, directly
        through Chronos rather than re-parsing "snooze for N minutes" with NLU."""
        result = self.chronos.handle(
            "snooze_reminder",
            {"duration": f"{minutes} minutes", "raw_query": f"snooze {minutes} minutes"},
            {},
        )
        return (result or {}).get("response") or "I couldn't snooze that reminder."

    def _telegram_inbox(self, name: str) -> Path:
        path = Path(__file__).resolve().parent / "data" / "telegram_inbox" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _telegram_ingest_photo(self, path: str, filename: str) -> str:
        """A photo (or image sent as a file) from Telegram → Iris. Each upload
        gets its own folder under data/telegram_inbox/ and Iris ingests just
        that folder, so nothing is added to the library folder Iris scans and
        the result describes this one file. Kept only if Iris stored it."""
        if self.iris is None:
            return "Photo filing isn't available right now (Iris is off)."
        folder = self._telegram_inbox("photos") / f"{int(time.time() * 1000)}"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, folder / Path(filename).name)
        try:
            stats = self.iris.ingest(source_dir=str(folder)) or {}
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        if stats.get("ingested"):
            return "Saved to your photo library."
        shutil.rmtree(folder, ignore_errors=True)
        if stats.get("exceeds_quota"):
            return "Your photo library is over its storage quota, so I didn't add that."
        if stats.get("duplicates_skipped"):
            return "That photo is already in your library."
        return "I couldn't add that to your photo library."

    def _telegram_ingest_document(self, path: str, filename: str) -> str:
        """A PDF from Telegram → Athena's document index under a "Telegram" subject."""
        if self.athena is None:
            return "Document filing isn't available right now (Athena is off)."
        from modules.athena.config import get_config as get_athena_config

        folder = Path(str(get_athena_config().data_dir)) / "Telegram"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / Path(filename).name
        shutil.copyfile(path, target)
        chunks, status = self.athena.rag.ingest_file({
            "full_path": str(target),
            "file_name": target.name,
            "subject": "Telegram",
            "module": "General",
            "relative_path": os.path.join("Telegram", target.name),
        })
        if status == "failed":
            return f"I couldn't read {target.name}."
        if status == "unchanged":
            return f"{target.name} is already in your documents."
        verb = "Updated" if status == "updated" else "Added"
        return f"{verb} {target.name} in your documents ({chunks} chunks). You can ask me about it now."

    def _try_local_command(self, cleaned: str, voice_turn: bool) -> Optional[str]:
        """Handle assistant-voice control phrases ("repeat that", "do not
        disturb", "I'm in a noisy room") before NLU. Returns the spoken
        reply, or None if *cleaned* is an ordinary query."""
        cmd = parse_voice_command(cleaned)
        if cmd is None:
            return None

        logger.info("You: %s", cleaned)
        try:
            response = self._run_voice_command(cmd)
        except Exception:
            logger.exception("Voice command %r failed.", cmd.name)
            response = "Sorry, that didn't work."
        logger.info("Hestia: %s", response)

        try:
            if voice_turn:
                self._speak_with_barge_in(response)
            else:
                self._speak(response)
        except Exception:
            logger.exception("TTS failed.")

        if voice_turn:
            try:
                self.wake_detector.flush_audio_queue()
            except Exception:
                logger.debug("flush_audio_queue() failed; ignoring.")
        return response

    def _run_voice_command(self, cmd: VoiceCommand) -> str:
        """Carry out *cmd* and return the sentence to say back."""
        vs = self.voice_state

        if cmd.name == REPEAT:
            return self._last_spoken or "I haven't said anything yet."

        if cmd.name == DND_ON:
            vs.set_dnd(True, cmd.minutes)
            if cmd.unparsed:
                return (
                    f"I didn't understand '{cmd.unparsed}' as a length of time, "
                    "so do not disturb is on until you say resume notifications."
                )
            if cmd.minutes:
                return (
                    f"Do not disturb is on for {_format_minutes(cmd.minutes)}. "
                    "I'll hold reminders and nudges until then."
                )
            return (
                "Do not disturb is on. I'll hold reminders and nudges until "
                "you say resume notifications."
            )

        if cmd.name == DND_OFF:
            was_on = vs.dnd_active()
            vs.set_dnd(False)
            digest = self._held_notifications_digest()
            if digest:
                return f"Notifications are back on. {digest}"
            return "Notifications are back on." if was_on else "Notifications weren't muted."

        if cmd.name == DND_STATUS:
            if not vs.dnd_active():
                return "Do not disturb is off."
            held = vs.held_count()
            remaining = vs.dnd_remaining_minutes()
            until = (
                f" for another {_format_minutes(max(remaining, 1))}"
                if remaining is not None else ""
            )
            waiting = f" {held} notification{'s' if held != 1 else ''} waiting." if held else ""
            return f"Do not disturb is on{until}.{waiting}"

        if cmd.name == SET_SENSITIVITY:
            detector = getattr(self, "wake_detector", None)
            if detector is None:
                return (
                    "Wake word detection isn't running, so there's no "
                    "sensitivity to change."
                )
            applied = detector.set_sensitivity(cmd.level)
            return f"Wake word sensitivity set to {applied}."

        return ""

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
        self._trace_local.trace = None
        cleaned = _clean_input(text)
        if not cleaned:
            return ""

        # "repeat that" / "do not disturb" / "I'm in a noisy room": controls
        # for the voice pipeline itself, answered before NLU.
        local = self._try_local_command(cleaned, voice_turn=False)
        if local is not None:
            self._trace_local.trace = {
                "module": "core", "intent": "local_command", "confidence": 1.0,
                "source": "local_command",
                "reason": "A voice-control command, answered before language understanding.",
            }
            return local

        # One id per query, attached to a contextvar and stamped onto every
        # log record emitted while this query is in flight (backlog #17).
        new_request_id()
        logger.info("You: %s", cleaned)

        with Timer() as turn_timer:
            multi = self._try_multi_intent(cleaned)
            if multi is not None:
                response, nlu_result = multi
            else:
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

        # Every classification + routing decision, one JSON line each
        # (backlog #5). Best-effort: observability must never be the
        # reason a query fails. Skipped for a committed multi-intent split
        # — _try_multi_intent already logged one record per segment, which
        # is the accurate picture; a third combined record here would just
        # misattribute two different routing decisions to one module.
        trace = None
        if nlu_result.get("source") != "multi_intent":
            try:
                trace = self._log_routing(cleaned, nlu_result, turn_timer.ms)
            except Exception:
                logger.debug("Routing log failed; continuing.")
        else:
            trace = {
                "module": "multiple", "intent": "multi_intent",
                "confidence": nlu_result.get("confidence"), "source": "multi_intent",
                "reason": "Split into two separate requests, each routed on its own.",
                "latency_ms": round(turn_timer.ms, 1),
            }
        self._trace_local.trace = trace

        logger.info("Hestia: %s", response)

        self._last_spoken = response
        try:
            self._speak(response, self._voice_profile_for_intent(nlu_result.get("intent")))
        except Exception:
            logger.exception("TTS failed.")

        try:
            if self.wake_detector is not None:
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

    def process_text_traced(self, text: str) -> tuple[str, Optional[dict]]:
        """``process_text`` plus how the query was routed (backlog #189).

        Returns ``(response, trace)``. *trace* is the same record written to
        the routing log: module, intent, confidence, source, reason, latency
        and the tiers Hecate checked. It is None for an empty query. The web
        UI shows it under "Why this answer?".
        """
        response = self.process_text(text)
        return response, getattr(self._trace_local, "trace", None)

    # ------------------------------------------------------------------
    # Multi-intent queries (backlog #13)
    # ------------------------------------------------------------------

    def _try_multi_intent(self, cleaned: str) -> Optional[tuple[str, dict]]:
        """
        Attempt to split *cleaned* into two independent requests and
        dispatch each separately — "log my workout and tell me the
        weather" should do both, not whichever one the NLU happened to
        pick for the whole sentence.

        Returns ``(combined_response, representative_nlu_result)`` if a
        split was committed to, or ``None`` to fall through to the normal
        single-query path in ``process_text``.

        A split is committed to only when BOTH candidate segments
        independently classify — via a real ``self.nlu.understand()``
        call each, not a guess from ``core.query_splitter``'s string
        heuristic alone — to two DIFFERENT, concrete (non-``chat``),
        REGISTERED intents. That's deliberately a much higher bar than
        "the text contains the word 'and'": a wrong split candidate costs
        one discarded NLU call, never a wrong action, because nothing is
        dispatched until both halves already look like two genuine,
        distinct requests. "log my workout and how I felt" (one request,
        two clauses) fails this bar — the second half has no concrete
        intent of its own — and falls through to single-query handling,
        same as before this feature existed.
        """
        segments = candidate_segments(cleaned)
        if len(segments) != 2:
            return None

        results: list[dict] = []
        for segment in segments:
            try:
                context = self.mnemosyne.get_recent(_RECENT_CONTEXT_TURNS)
                results.append(self.nlu.understand(segment, context))
            except Exception:
                logger.exception(
                    "Multi-intent candidate classification failed for "
                    "segment %r; falling back to single-query handling.",
                    segment[:80],
                )
                return None

        intents = [r.get("intent", "chat") for r in results]
        if (
            intents[0] == "chat"
            or intents[1] == "chat"
            or intents[0] == intents[1]
            or module_for_intent(intents[0]) is None
            or module_for_intent(intents[1]) is None
        ):
            return None

        logger.info(
            "Multi-intent split: %r -> [%r (%s), %r (%s)]",
            cleaned[:80], segments[0][:40], intents[0], segments[1][:40], intents[1],
        )

        responses: list[str] = []
        for i, (segment, result) in enumerate(zip(segments, results)):
            # Intent chaining (backlog #24): "summarize this and add IT to
            # my reading list" — the second segment refers back to the
            # first segment's result rather than carrying its own
            # content. Detected purely from the second segment's own
            # text (see core/intent_chains.py); only applies from segment
            # 0 into segment 1, since a two-segment split has nowhere
            # else for a reference to point.
            if i == 1 and responses:
                chain_key = detect_chain_reference(segment, intents[1])
                if chain_key is not None:
                    result = dict(result)
                    result["entities"] = apply_chain(
                        result.get("entities", {}), chain_key, responses[0]
                    )
                    logger.info(
                        "Chained segment 2 (%r) off segment 1's result via "
                        "entity %r.", intents[1], chain_key,
                    )

            try:
                raw = self.orchestrator.dispatch(segment, result)
            except Exception:
                logger.exception(
                    "Multi-intent dispatch failed for segment %r.", segment[:80]
                )
                raw = "something went wrong with that part"
            piece = _postprocess(raw).strip()
            # Each segment is logged individually here (its own accurate
            # module/reason/confidence) rather than once at the end for
            # the combined pair — see process_text's guard on
            # nlu_result["source"] == "multi_intent", which skips a
            # second, misattributing log entry for the pair as a whole.
            try:
                self._log_routing(segment, result, 0.0)
            except Exception:
                logger.debug("Routing log failed for multi-intent segment; continuing.")
            responses.append(piece)

        combined = " Also, ".join(
            r if r.rstrip().endswith((".", "!", "?")) else f"{r}."
            for r in responses if r
        )
        representative = {
            "intent": "multi_intent",
            "entities": {},
            "response": "",
            "confidence": min(
                float(r.get("confidence", 0.0) or 0.0) for r in results
            ),
            "source": "multi_intent",
        }
        return combined or "Done.", representative

    # ------------------------------------------------------------------
    # Hot reload (backlog #15)
    # ------------------------------------------------------------------

    def _start_hot_reload_watchers(self) -> None:
        """
        Start the two file watchers described in core/hot_reload.py's
        module docstring: a full reload for the NLU prompt, and a
        detect-and-validate-only watcher for the main config.

        Best-effort like the rest of observability: a watcher that fails
        to start (e.g. the prompt file was deleted between startup and
        here) is logged and skipped, never fatal — hot reload is a
        development convenience, not something a query's correctness
        depends on.
        """
        self._hot_reload_watchers: list[FileWatcher] = []

        prompt_path = getattr(self.nlu, "_prompt_path", None)
        if prompt_path:
            try:
                watcher = FileWatcher(prompt_path, self.nlu.reload_prompt)
                watcher.start()
                self._hot_reload_watchers.append(watcher)
                logger.info("Watching %s for hot reload.", prompt_path)
            except Exception:
                logger.exception("Could not start prompt hot-reload watcher.")

        try:
            self._config_snapshot = dict(self.config)
            watcher = FileWatcher(self._config_path, self._on_config_file_changed)
            watcher.start()
            self._hot_reload_watchers.append(watcher)
            logger.info("Watching %s for changes.", self._config_path)
        except Exception:
            logger.exception("Could not start config change watcher.")

    def _on_config_file_changed(self) -> None:
        """
        Detect-and-report handler for `laptop_config.yaml` (backlog #15).

        Deliberately does NOT re-wire any subsystem — see
        core/hot_reload.py's module docstring for why most config keys
        can't be safely hot-applied. What this DOES do: re-validate the
        edited file exactly as startup would (so a typo is caught in
        seconds, not at the next restart), and log which top-level
        sections changed, so tuning a value and wondering "did that even
        take" has an immediate, honest answer — "no, and here's why" or
        "yes, but not until you restart".
        """
        try:
            with self._config_path.open("r", encoding="utf-8") as fh:
                new_config = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(
                "%s changed but could not be read: %s", self._config_path, exc
            )
            return

        if not isinstance(new_config, dict):
            logger.warning(
                "%s changed but is no longer a YAML mapping; ignoring.",
                self._config_path,
            )
            return

        report = validate_config(new_config)
        for warning in report.warnings:
            logger.warning("Config: %s", warning)
        if not report.ok:
            logger.error(
                "%s changed but is now invalid — the running process is "
                "UNAFFECTED (it keeps the config it started with), but a "
                "restart right now would fail:\n%s",
                self._config_path, report.format(),
            )
            return

        old_keys = set(self._config_snapshot)
        new_keys = set(new_config)
        changed = sorted(
            k for k in old_keys & new_keys
            if self._config_snapshot.get(k) != new_config.get(k)
        )
        added = sorted(new_keys - old_keys)
        removed = sorted(old_keys - new_keys)
        self._config_snapshot = dict(new_config)

        if not (changed or added or removed):
            return  # touched but not actually different (e.g. a re-save)

        parts = []
        if changed:
            parts.append(f"changed: {', '.join(changed)}")
        if added:
            parts.append(f"added: {', '.join(added)}")
        if removed:
            parts.append(f"removed: {', '.join(removed)}")
        logger.info(
            "%s changed and is valid (%s). Most settings take effect on "
            "the next restart — nothing was hot-applied.",
            self._config_path, "; ".join(parts),
        )

    def _stop_hot_reload_watchers(self) -> None:
        for watcher in getattr(self, "_hot_reload_watchers", []):
            try:
                watcher.stop()
            except Exception:
                logger.debug("Hot-reload watcher stop() raised; ignoring.")

    # ------------------------------------------------------------------
    # Observability helpers (backlog #1, #3, #5)
    # ------------------------------------------------------------------

    def _log_routing(self, query: str, nlu_result: dict, latency_ms: float) -> dict:
        """
        Record one classification + routing decision.

        Reads the module actually chosen from the orchestrator's
        last_decision rather than re-running Hecate, so the log reflects
        what really happened (including fallback tiers and the
        can_handle()-mismatch recovery path) instead of a second, possibly
        different, routing guess.
        """
        decision = self.orchestrator.last_decision or {}
        return self.diagnostics.record_classification(
            query=query,
            intent=nlu_result.get("intent", "chat"),
            confidence=nlu_result.get("confidence", 0.0),
            module=decision.get("primary", "unknown"),
            reason=decision.get("reason", ""),
            latency_ms=latency_ms,
            source=nlu_result.get("source", "nlu"),
            checked=decision.get("checked"),
        )

    def resolve_only(self, text: str) -> dict:
        """
        Run the full routing pipeline WITHOUT executing the handler
        (backlog #1 — this is what `--dry-run` calls).

        NLU classification and Hecate's decision both run for real, so the
        answer is the genuine routing decision, not a re-implementation of
        it. What is skipped is `handle()`: no side effects, nothing written
        to any module's database, no LLM generation, no TTS. The resolved
        intent is also NOT recorded to the routing log, since a dry run
        isn't a real interaction and would otherwise pollute the very
        dataset the log exists to build.
        """
        cleaned = _clean_input(text)
        if not cleaned:
            return {"error": "empty query"}

        new_request_id()
        with Timer() as timer:
            try:
                context = self.mnemosyne.get_recent(_RECENT_CONTEXT_TURNS)
                nlu_result = self.nlu.understand(cleaned, context)
            except Exception as exc:
                logger.exception("Dry run: NLU failed.")
                return {"error": f"NLU failed: {exc}"}

        intent = nlu_result.get("intent", "chat")
        active = list(self.orchestrator.registered_modules)
        try:
            decision = self.orchestrator._route(cleaned, nlu_result)
        except Exception as exc:  # pragma: no cover - _route never raises
            return {"error": f"routing failed: {exc}"}

        primary = decision.get("primary", "core")
        dispatch_intent = decision.get("intent") or strip_module_prefix(intent)
        target = self.orchestrator._modules.get(primary)
        can_handle = None
        if target is not None:
            try:
                can_handle = bool(target.can_handle(dispatch_intent))
            except Exception:
                can_handle = None

        return {
            "query": cleaned,
            "request_id": current_request_id(),
            "intent": intent,
            "entities": nlu_result.get("entities", {}),
            "confidence": float(nlu_result.get("confidence", 0.0) or 0.0),
            "nlu_source": nlu_result.get("source", "nlu"),
            "nlu_latency_ms": round(timer.ms, 1),
            "registry_module": module_for_intent(intent),
            "primary": primary,
            "secondary": decision.get("secondary", []),
            "reason": decision.get("reason", ""),
            "dispatch_intent": dispatch_intent,
            "primary_can_handle": can_handle,
            "synthesize": bool(decision.get("synthesize", False)),
            "active_modules": active,
            "executed": False,
        }

    @staticmethod
    def format_dry_run(result: dict) -> str:
        """Render resolve_only() output as an aligned, readable block."""
        if "error" in result:
            return f"dry-run failed: {result['error']}"

        order = [
            ("query", "query"),
            ("request_id", "request id"),
            ("intent", "NLU intent"),
            ("confidence", "confidence"),
            ("nlu_source", "classified by"),
            ("nlu_latency_ms", "NLU latency (ms)"),
            ("entities", "entities"),
            ("registry_module", "registry says"),
            ("primary", "routed to"),
            ("dispatch_intent", "dispatched as"),
            ("primary_can_handle", "target can_handle"),
            ("secondary", "secondary"),
            ("synthesize", "synthesise"),
            ("reason", "Hecate reason"),
        ]
        width = max(len(label) for _, label in order)
        lines = ["", "--- dry run (no handler executed) ---"]
        for key, label in order:
            lines.append(f"  {label.rjust(width)} : {result.get(key)}")

        # A registry/routing disagreement is the single most useful thing
        # a dry run can surface, so call it out rather than leaving it to
        # be spotted by eye in the two adjacent lines above.
        registry_module = result.get("registry_module")
        if registry_module and registry_module != result.get("primary"):
            lines.append(
                f"  NOTE: registry maps this intent to {registry_module!r} but "
                f"Hecate chose {result.get('primary')!r} — check the tier in "
                f"'Hecate reason' above."
            )
        if result.get("primary_can_handle") is False:
            lines.append(
                "  NOTE: the target module's can_handle() rejects this intent, "
                "so a real run would fall back to another module (usually core "
                "chat)."
            )
        lines.append("-------------------------------------")
        return "\n".join(lines)

    def install_signal_handlers(self) -> None:
        """
        Shut down cleanly on SIGTERM/SIGINT (backlog #18).

        Without this, a `kill` (or a systemd stop, or the OS closing the
        process) skipped _shutdown() entirely: the barge-in listener kept
        the microphone open, the heartbeat thread was killed mid-write, and
        the event bus's executor never drained. Registering here rather
        than at import time keeps `import main` side-effect-free for the
        tests.

        Only installs when called from the main thread — Python only
        permits signal registration there, and Hestia is also imported by
        the web UI and test suite from worker threads.
        """
        if threading.current_thread() is not threading.main_thread():
            logger.debug("Not the main thread; skipping signal handlers.")
            return

        def _handle(signum, _frame) -> None:
            name = signal.Signals(signum).name
            logger.info("Received %s — shutting down gracefully.", name)
            self._shutdown()
            raise SystemExit(0)

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handle)
            except (ValueError, OSError, AttributeError):
                # Not available on this platform (SIGTERM on some Windows
                # builds) — the finally: block in the run loops still runs.
                logger.debug("Could not install handler for %s.", sig)

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

        local = self._try_local_command(cleaned, voice_turn=True)
        if local is not None:
            return local

        logger.info("You: %s", cleaned)
        self.voice_state.set_state(STATE_THINKING)

        try:
            context = self.mnemosyne.get_recent(_RECENT_CONTEXT_TURNS)
            nlu_result = self.nlu.understand(cleaned, context)
        except Exception:
            logger.exception("NLU failed for input=%r.", cleaned[:80])
            nlu_result = {"intent": "chat", "entities": {}, "response": ""}

        voice = self._voice_profile_for_intent(nlu_result.get("intent"))

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
            response = self._speak_streaming(stream, voice)
        else:
            try:
                response = self.orchestrator.dispatch(cleaned, nlu_result)
            except Exception:
                logger.exception("Orchestrator dispatch failed.")
                response = "I'm sorry, something went wrong."

            response = _postprocess(response)
            logger.info("Hestia: %s", response)
            self._speak_with_barge_in(response, voice)

        self._last_spoken = response

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

    def _speak_streaming(self, chunks, voice: Optional[str] = None) -> str:
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

        if voice:
            self._with_barge_in(lambda: self.tts.speak_stream(_tap(), voice=voice))
        else:
            self._with_barge_in(lambda: self.tts.speak_stream(_tap()))

        response = "".join(parts).strip()
        if not response:
            response = "Done."
        logger.info("Hestia: %s", response)
        return response

    def _speak_with_barge_in(self, response: str, voice: Optional[str] = None) -> None:
        """Speak a single finished response with the barge-in listener
        armed, so even non-streamed replies can be interrupted."""
        self._with_barge_in(lambda: self._speak(response, voice))

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
        vs = self.voice_state
        vs.set_state(STATE_SPEAKING)

        if not self._barge_in_enabled or self.barge_in is None:
            speak_fn()
            self.tts.wait_until_done()
            return

        self.barge_in.reset()
        self.barge_in.start(on_barge_in=self.tts.stop)
        vs.set_barge_in_armed(True)
        try:
            speak_fn()
            self.tts.wait_until_done()
        finally:
            vs.set_barge_in_armed(False)
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
        if self.stt is None or self.wake_detector is None:
            missing = [k for k in ("stt", "wake_word") if k in self._voice_io_errors]
            detail = "; ".join(f"{k}: {self._voice_io_errors[k]}" for k in missing)
            try:
                self._voice_fallback_to_typing(
                    "Voice input isn't available" + (f" ({detail})" if detail else "")
                )
            finally:
                self._shutdown()
            return

        vs = self.voice_state
        logger.info("Voice loop started — listening for wake word.")
        failures = 0
        try:
            skip_wake_word = False
            pending_audio = None
            while True:
                try:
                    if pending_audio is not None:
                        vs.set_state(STATE_THINKING)
                        text = self.stt.transcribe_audio(pending_audio)
                        pending_audio = None
                    elif skip_wake_word:
                        skip_wake_word = False
                        vs.set_state(STATE_LISTENING)
                        text = self.stt.listen_once(max_duration=_STT_MAX_DURATION)
                    else:
                        vs.set_state(STATE_WAKE)
                        if not self.wake_detector.listen_for_wake_word(
                            timeout=_WAKE_WORD_TIMEOUT
                        ):
                            self._speak_held_notifications()
                            failures = 0
                            continue

                        vs.set_state(STATE_SPEAKING)
                        self.tts.speak("Yes?")
                        self.tts.wait_until_done()
                        vs.set_state(STATE_LISTENING)
                        text = self.stt.listen_once(max_duration=_STT_MAX_DURATION)
                    failures = 0
                except Exception as exc:
                    # Mic unplugged, PortAudio error, model crash... One bad
                    # cycle shouldn't end voice mode, but a mic that keeps
                    # failing means voice mode can't work: drop to typing
                    # instead of spinning (backlog #179).
                    failures += 1
                    logger.warning(
                        "Voice input error (%d/%d): %s",
                        failures, _MAX_VOICE_FAILURES, exc,
                    )
                    pending_audio = None
                    skip_wake_word = False
                    if failures >= _MAX_VOICE_FAILURES:
                        self._voice_fallback_to_typing(
                            f"The microphone keeps failing ({exc})"
                        )
                        break
                    time.sleep(0.5)
                    continue

                if not text or len(text.strip()) < _MIN_VOICE_INPUT_LEN:
                    vs.set_state(STATE_SPEAKING)
                    self.tts.speak("I didn't catch that.")
                    self.tts.wait_until_done()
                    continue

                if text.lower().strip() in _EXIT_WORDS:
                    vs.set_state(STATE_SPEAKING)
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
            vs.set_state(STATE_INACTIVE)
            self._shutdown()

    def _speak_held_notifications(self) -> None:
        """Read out notifications held during a do-not-disturb that has since
        run out (a timed DND ends silently, so the voice loop checks for
        leftovers each idle cycle)."""
        try:
            digest = self._held_notifications_digest()
            if digest:
                self._last_spoken = digest
                self.tts.speak(digest)
                self.tts.wait_until_done()
        except Exception:
            logger.exception("Could not read held notifications.")

    def _voice_fallback_to_typing(self, reason: str) -> None:
        """Voice mode can't run: say so and carry on with typed input
        (backlog #179). Replies are still spoken if a speech engine works."""
        self.voice_state.set_state(STATE_TYPED, detail=reason)
        logger.warning("%s — falling back to typed input.", reason)
        print(
            f"\n[voice] {reason}.\n"
            "[voice] Continuing with typed input; type 'exit' to quit.\n",
            file=sys.stderr,
        )
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in _EXIT_WORDS:
                break
            self.process_text(user_input)

    def run_headless(self) -> None:
        """Run without a terminal or microphone (the core and jobs roles, #20):
        stay alive until SIGTERM/SIGINT, which ``install_signal_handlers`` turns
        into a clean shutdown."""
        logger.info("Running headless as the %s process; waiting for work.", self.role)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Headless run interrupted.")
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
            self._stop_hot_reload_watchers()
        except Exception:
            logger.debug("_stop_hot_reload_watchers() raised; ignoring.")

        try:
            # force=True: a mid-capture barge-in listener would otherwise
            # wait for its own natural end-of-utterance (up to
            # max_capture_seconds) before stop() returns — fine during
            # normal turn-taking, but shutdown (e.g. Ctrl-C) should never
            # be held up by that.
            if self.barge_in is not None:
                self.barge_in.stop(force=True)
        except Exception:
            logger.debug("barge_in.stop() raised; ignoring.")

        try:
            self.heartbeat.stop()
        except Exception:
            logger.debug("heartbeat.stop() raised; ignoring.")

        for part in (getattr(self, "_query_server", None), getattr(self, "_bridge", None)):
            try:
                if part is not None:
                    part.stop()
            except Exception:
                logger.debug("%r.stop() raised; ignoring.", part)

        # Close the browsers (and the monitor worker thread) so no Chromium
        # process outlives Hestia.
        try:
            hephaestus = getattr(self, "hephaestus", None)
            if hephaestus is not None and callable(getattr(hephaestus, "close", None)):
                hephaestus.close()
        except Exception:
            logger.debug("hephaestus.close() raised; ignoring.")
        try:
            browser_agent = getattr(self, "browser_agent", None)
            if browser_agent is not None:
                browser_agent.close()
        except Exception:
            logger.debug("browser_agent.close() raised; ignoring.")

        try:
            stop_sched = getattr(self.chronos, "stop_scheduler", None)
            if callable(stop_sched):
                stop_sched()
        except Exception:
            logger.debug("chronos.stop_scheduler() raised; ignoring.")

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

    # Fail fast with every problem named at once (backlog #14). Without
    # this, a wrong type or a missing required key surfaced much later,
    # deep inside whichever module happened to read it first, in a form
    # that never mentioned the config key responsible. Warnings (unknown
    # top-level sections — usually typos) are logged, not fatal.
    report = validate_or_raise(cfg, source=str(path))
    for warning in report.warnings:
        logger.warning("Config: %s", warning)
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

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Takes `argv` so the parser itself is unit-testable without
    monkeypatching sys.argv (tests/test_main_cli.py).
    """
    parser = argparse.ArgumentParser(
        description="Hestia personal AI assistant",
        epilog=(
            "Examples:\n"
            "  python main.py                              # interactive CLI\n"
            "  python main.py --voice                      # wake word + STT\n"
            '  python main.py --dry-run "log my sleep"     # show routing, run nothing\n'
            "  python main.py --check-config               # validate config and exit\n"
            "  python main.py --calibrate-mic              # tune barge-in / VAD for this mic\n"
            "  python main.py --headed                     # watch the browser automation work\n"
            "  python main.py --quiet                      # warnings only\n"
            "  python main.py --verbose                    # full debug trace\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    # Backlog #1. Nargs="+" so the query needn't be quoted, though quoting
    # is still clearer for anything containing shell metacharacters.
    parser.add_argument(
        "--dry-run",
        nargs="+",
        metavar="QUERY",
        help=(
            "Classify and route QUERY, print the resolved routing decision, "
            "and exit without executing the handler (no side effects)."
        ),
    )
    # Backlog #14. Useful in CI and after editing the YAML by hand.
    parser.add_argument(
        "--check-config",
        action="store_true",
        help=(
            "Validate the configuration file and exit. Exit code 0 if valid, "
            "1 if not. Does not start any subsystem."
        ),
    )
    # Backlog #174.
    parser.add_argument(
        "--calibrate-mic",
        action="store_true",
        help=(
            "Measure this microphone/speaker pair (quiet room, your voice, "
            "Hestia's own echo) and save recommended barge-in / VAD / "
            "wake-word settings to data/mic_calibration.json, then exit."
        ),
    )
    # Backlog #108.
    parser.add_argument(
        "--headed",
        action="store_true",
        help=(
            "Show the browser window (and slow it slightly) so you can watch "
            "what browser automation is doing. Page monitors stay headless."
        ),
    )
    # Backlog #20: process roles.
    parser.add_argument(
        "--role",
        choices=["all", "core", "voice", "jobs", "supervisor"],
        default="all",
        help=(
            "Which process to run. 'all' (default) is the original single "
            "process. 'supervisor' starts and restarts core + jobs + voice; "
            "the others run one part on its own (see core/process_split.py)."
        ),
    )
    # Backlog #4: the trained intent classifier.
    parser.add_argument(
        "--train-classifier",
        action="store_true",
        help=(
            "Train the intent classifier from the prompt examples, aliases, "
            "your labels and confident log lines, score it on the held-out "
            "golden prompts, save it, and exit."
        ),
    )
    parser.add_argument(
        "--label",
        nargs=2,
        metavar=("QUERY", "INTENT"),
        help="Record that QUERY should be classified as INTENT (used by --train-classifier), then exit.",
    )
    # Backlog #16: shadow mode.
    parser.add_argument(
        "--shadow-report",
        action="store_true",
        help="Print how each shadow-mode candidate handler compares with the current one, then exit.",
    )
    # Backlog #275.
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Debug-level logging, including Hecate's routing traces.",
    )
    verbosity.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Warnings and errors only — the sane default for daily use.",
    )
    return parser.parse_args(argv)


def _verbosity_from_args(args: argparse.Namespace) -> str:
    """Map the mutually exclusive --verbose/--quiet flags onto a level name."""
    if getattr(args, "verbose", False):
        return "verbose"
    if getattr(args, "quiet", False):
        return "quiet"
    return "normal"


def _run_check_config(path: str) -> int:
    """
    Implement --check-config: validate and report, without booting Hestia.

    Deliberately does NOT use validate_or_raise: the point here is to show
    the user every problem at once in a readable form, not to raise on the
    first one.
    """
    config_path = Path(path)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        print(f"{config_path} is not valid YAML:\n{exc}", file=sys.stderr)
        return 1

    report = validate_config(cfg)
    print(f"{config_path}:")
    print(report.format())
    registry = registry_info()
    print(
        f"Intent registry v{registry['version']} "
        f"({registry['intent_count']} intents across "
        f"{registry['module_count']} modules, "
        f"fingerprint {registry['fingerprint']})."
    )
    return 0 if report.ok else 1


def _run_calibrate_mic(path: str) -> int:
    """Implement --calibrate-mic. Does not boot Hestia: it needs only the
    microphone and, if one can be started, the configured TTS engine (for the
    echo measurement)."""
    tts = None
    config_path = Path(path)
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            tts_cfg = cfg.get("tts") or {}
            tts = HestiaTTS(
                engine=tts_cfg.get("engine", "pyttsx3"),
                rate=tts_cfg.get("rate", 175),
                volume=tts_cfg.get("volume", 1.0),
                piper_model_path=tts_cfg.get("piper_model_path"),
            )
        except Exception as exc:
            print(f"(Echo test skipped — TTS unavailable: {exc})", file=sys.stderr)
            tts = None
    try:
        result = run_calibration(tts=tts)
    except (KeyboardInterrupt, EOFError):
        print("\nCalibration cancelled.")
        return 1
    except Exception as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1
    print()
    print(format_report(result))
    saved = save_calibration(result)
    print(f"\nSaved to {saved}. Hestia uses it automatically for any setting "
          "your config doesn't pin.")
    return 0


def _run_train_classifier(path: str) -> int:
    """``--train-classifier``: train, score on the held-out golden prompts, save."""
    from core.classifier_data import evaluate_on_golden
    from modules.hecate.intent_registry import ALL_INTENTS
    try:
        cfg = _load_config(Path(path))
    except Exception as exc:
        print(f"Could not read config: {exc}", file=sys.stderr)
        return 1
    ccfg = cfg.get("classifier") or {}
    kwargs = HestiaBuilder._classifier_kwargs(ccfg, "assist")
    if kwargs["backend"] == "transformer":
        print("classifier.backend is 'transformer': that model is made by "
              "scripts/finetune_classifier.py, not by --train-classifier.", file=sys.stderr)
        return 1
    svc = ClassifierService(valid_intents=ALL_INTENTS, **kwargs)
    print(f"Backend: {kwargs['backend']}"
          f"{' + augmentation' if ccfg.get('augment') else ''}")
    result = svc.train()
    if not result.get("ok"):
        print(f"Training failed: {result.get('error')}", file=sys.stderr)
        return 1
    st = result["stats"]
    print(f"Trained on {st.examples} examples across {st.classes} intents "
          f"in {st.seconds:.1f}s; sources: {dict(st.sources)}")
    if st.calibration:
        c = st.calibration
        print(f"Calibrated on cross-validation: answer when probability >= {c.get('min_prob')} and "
              f"margin >= {c.get('min_margin')} (target precision {c.get('target_precision'):.0%}, "
              f"{'met' if c.get('met_target') else 'NOT met'})")
    print(f"Saved to {svc.model_path}")
    ev = evaluate_on_golden(svc)
    print(f"Held-out golden prompts: answered {ev['answered']}/{ev['cases']} "
          f"({ev['coverage']:.0%}), right {ev['right']}/{ev['answered']} "
          f"({ev['precision']:.0%} of the ones it answered)")
    for w in ev["wrong"][:10]:
        print(f"  wrong: {w['prompt']!r} expected {w['expected']!r}, got {w['got']!r}")
    print("It declines the rest, which then follow the normal routing tiers.")
    return 0


def _run_label(query: str, intent: str) -> int:
    from core.classifier_data import add_label
    if add_label(query, intent):
        print(f"Recorded: {query!r} -> {intent}. Run --train-classifier to use it.")
        return 0
    print(f"{intent!r} is not a registered intent (or the query is empty); nothing recorded.",
          file=sys.stderr)
    return 1


def _run_shadow_report(path: str) -> int:
    try:
        cfg = _load_config(Path(path))
    except Exception as exc:
        print(f"Could not read config: {exc}", file=sys.stderr)
        return 1
    rules = ShadowRules.from_config(cfg.get("shadow"))
    print(ShadowRecorder(rules).summary())
    return 0


def _run_voice_role(path: str) -> int:
    """``--role voice``: just the microphone/speaker side (core/process_split.py)."""
    try:
        cfg = _load_config(Path(path))
    except Exception as exc:
        print(f"Could not read config: {exc}", file=sys.stderr)
        return 1
    builder = HestiaBuilder(cfg)
    stt, tts, wake, barge = builder.build_io()
    queue = EventQueue(queue_path(cfg), origin="voice")
    frontend = VoiceFrontend(stt, tts, wake, barge, queue)

    def _term(signum, _frame):
        frontend.stop()
        raise SystemExit(0)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _term)
        except (ValueError, OSError, AttributeError):
            pass
    try:
        frontend.run()
    except RuntimeError as exc:
        print(f"Voice process cannot run: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_supervisor(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(Path(args.config))
    except Exception as exc:
        print(f"Could not read config: {exc}", file=sys.stderr)
        return 1
    roles = (cfg.get("processes") or {}).get("roles") or ["core", "jobs", "voice"]
    base = [sys.executable, str(Path(__file__).resolve()), "--config", args.config]
    if getattr(args, "verbose", False):
        base.append("--verbose")
    elif getattr(args, "quiet", False):
        base.append("--quiet")
    sup = Supervisor(roles=roles, base_command=base)
    logger.info("Supervisor starting roles: %s", ", ".join(roles))
    return sup.run_forever()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    global logger
    logger = _configure_logging(_verbosity_from_args(args))

    # --check-config must not construct Hestia — it's the thing you run
    # precisely when Hestia won't start.
    if args.check_config:
        return _run_check_config(args.config)

    # Likewise: calibration only needs the mic, not a booted assistant.
    if args.calibrate_mic:
        return _run_calibrate_mic(args.config)

    if args.label:
        return _run_label(*args.label)
    if args.train_classifier:
        return _run_train_classifier(args.config)
    if args.shadow_report:
        return _run_shadow_report(args.config)
    if args.role == "supervisor":
        return _run_supervisor(args)
    if args.role == "voice":
        return _run_voice_role(args.config)

    try:
        hestia = Hestia(config_path=args.config, headed=args.headed, role=args.role)
    except ConfigError as exc:
        # Already fully formatted by the validator; a traceback here would
        # bury the one thing the user needs to read.
        print(str(exc), file=sys.stderr)
        return 1

    hestia.install_signal_handlers()

    if args.dry_run:
        query = " ".join(args.dry_run)
        print(hestia.format_dry_run(hestia.resolve_only(query)))
        hestia._shutdown()
        return 0

    if args.role in ("core", "jobs"):
        hestia.run_headless()
    elif args.voice:
        hestia.run_voice_loop()
    else:
        hestia.run_cli_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
