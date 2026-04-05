# main.py

import sys
import re
import time
import logging
import warnings
import os
import json
from dotenv import load_dotenv
load_dotenv()

# ── Silence noisy output ─────────────────────────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

for _noisy in (
    "huggingface_hub", "transformers", "sentence_transformers",
    "torch", "urllib3", "httpx", "httpcore", "asyncio", "werkzeug",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("hestia").setLevel(logging.INFO)
log = logging.getLogger("hestia")

# ── Imports ──────────────────────────────────────────────────────────
import yaml
from core.stt import HestiaSTT
from core.tts import HestiaTTS
from core.nlu import HestiaNLU
from core.google_agent import HestiaGoogleAgent
from core.wake_word import WakeWordDetector
from core.ollama_manager import OllamaManager
from core.browser_agent import HestiaBrowserAgent
from core.ollama_client import generate
from modules.hestia.core_module import CoreModule
from core.heartbeat import HestiaHeartbeat
from modules.mnemosyne.engine import MnemosyneEngine
from modules.hecate import HecateEngine
from modules.artemis import ArtemisEngine
from core.event_bus import bus
from modules.hestia.orchestrator import HestiaOrchestrator
from modules.chronos.engine import ChronosEngine
from modules.hermes.engine import HermesEngine
from modules.hephaestus.engine import HephaestusEngine
from modules.apollo import ApolloEngine
from modules.ares import AresEngine
from modules.orpheus import OrpheusEngine
from modules.dionysus import DionysusEngine
from modules.pluto import PlutoEngine
from core.llm import HestiaLLM

# ── Constants ────────────────────────────────────────────────────────
EXIT_WORDS = {"bye", "exit", "stop", "shutdown"}

_FILLER_RE = re.compile(r'\b(uh|um|you know)\b\s*', re.IGNORECASE)


# ── Main ────────────────────────────────────────────────────────────
class Hestia:

    def __init__(self, config_path="config/laptop_config.yaml"):
        print("Initializing Hestia...")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.mnemosyne_engine = None

        # Memory + Actions
        db_path = self.config.get("database", {}).get("path", "data/hestia.db")
        self.memory = None  # temporary placeholder

        # Ollama
        ollama_cfg = self.config.get("ollama", {})
        self.ollama_cfg = ollama_cfg

        self.ollama_manager = OllamaManager(
            host=ollama_cfg.get("host", "localhost"),
            port=ollama_cfg.get("port", 11434),
        )
        self.ollama_manager.ensure_running()
        time.sleep(2)

        # NLU
        self.nlu = HestiaNLU(
            model=ollama_cfg.get("model", "mistral"),
            host=ollama_cfg.get("host", "127.0.0.1"),
            port=ollama_cfg.get("port", 11434),
            prompt_path=self.config.get("nlu", {}).get("prompt_path"),
        )

        # ── LLM (ALWAYS INITIALIZED) ─────────────────────────────
        self.llm = HestiaLLM(
            self.ollama_manager.host,
            self.ollama_manager.port,
            ollama_cfg.get("model", "mistral"),
        )

        # Athena (optional)
        self.athena = None
        if self.config.get("athena", {}).get("enabled", False):
            from modules.athena.engine import AthenaEngine
            self.athena = AthenaEngine(self.llm)

        # ── MNEMOSYNE (MANDATORY) ─────────────────────────────
        self.mnemosyne = MnemosyneEngine(self.llm)
        self.mnemosyne_engine = self.mnemosyne
        self.memory = self.mnemosyne

        self.nlu.set_memory(self.mnemosyne)

        mn = self.mnemosyne
        bus.on("interaction_logged", lambda data: mn.push(
            data["query"],
            data["response"],
            data["intent"]
        ))

        if not self.mnemosyne_engine:
            raise RuntimeError(
                "Mnemosyne must be enabled — it is the single memory store."
            )

        self.iris = None
        if self.config.get("iris", {}).get("enabled", False):
            from modules.iris import IrisEngine
            self.iris = IrisEngine(self.llm)

        self.artemis = ArtemisEngine()
        self.hecate = HecateEngine()

        # Browser + Google
        self.browser_agent = HestiaBrowserAgent()

        google_cfg = self.config.get("google", {})
        self.google_agent = None
        if google_cfg.get("enabled", False):
            self.google_agent = HestiaGoogleAgent(
                credentials_path=google_cfg.get("credentials_path"),
                token_path=google_cfg.get("token_path"),
            )

        # ── ORCHESTRATOR (CORE CHANGE) ───────────────────────────────
        self.orchestrator = HestiaOrchestrator()
        self.orchestrator.register_hecate(self.hecate)

        self.core_module = CoreModule(memory=self.mnemosyne_engine, ollama_cfg=self.ollama_cfg)
        self.orchestrator.register(self.core_module)


        if self.athena: self.orchestrator.register(self.athena)
        if self.mnemosyne: self.orchestrator.register(self.mnemosyne)
        if self.iris: self.orchestrator.register(self.iris)
        self.orchestrator.register(self.artemis)

        self.chronos = ChronosEngine(memory=self.mnemosyne_engine)
        self.orchestrator.register(self.chronos)

        if self.google_agent:
            self.hermes = HermesEngine(self.google_agent)
            self.orchestrator.register(self.hermes)

        self.hephaestus = HephaestusEngine(self.browser_agent)
        self.orchestrator.register(self.hephaestus)

        self.orchestrator.register(ApolloEngine())
        self.orchestrator.register(AresEngine())
        self.orchestrator.register(OrpheusEngine())
        self.orchestrator.register(DionysusEngine())
        self.orchestrator.register(PlutoEngine())

        # STT + TTS
        self.stt = HestiaSTT()
        self.tts = HestiaTTS()

        # ── EVENT BUS WIRING ─────────────────────────────

        # Speak handler (CRITICAL)
        bus.on("speak", lambda data: self.tts.speak(data.get("text", "")))

        # Morning brief handler
        bus.on("morning_brief_requested", lambda _: self.process_text("give me my morning brief"))

        # Mnemosyne summariser trigger
        bus.on("mnemosyne_summarise", lambda _: self.mnemosyne.trigger_summarise())

        self.wake_detector = WakeWordDetector()

        # ── HEARTBEAT ─────────────────────────────
        self.heartbeat = HestiaHeartbeat(interval=1800, mnemosyne=self.mnemosyne)
        self.heartbeat.start()


        from web_ui import HestiaWebUI

        self.web_ui = HestiaWebUI(
            memory=self.mnemosyne,
            process_fn=self.process_text,   # CRITICAL
        )

        self.web_ui.start()

        # ── SYNC API (optional, port 5001) ───────────────────────
        sync_cfg = self.config.get("sync", {})
        if sync_cfg.get("enabled", False):
            from api import app as sync_app
            import uvicorn, threading
            sync_app.state.memory = self.mnemosyne
            def _run_sync():
                uvicorn.run(sync_app, host="127.0.0.1", port=5001, log_level="warning")
            threading.Thread(target=_run_sync, daemon=True, name="SyncAPI").start()
            print("[Hestia] Sync API running at http://127.0.0.1:5001")

        print("Hestia is ready.")

    # ── PROCESS TEXT (FULLY REPLACED CORE LOGIC) ─────────────────────
    def process_text(self, text: str) -> str:
        if not text.strip():
            return ""

        cleaned = _FILLER_RE.sub('', text.lower().strip()).strip()
        print(f"You: {cleaned}")

        context = self.mnemosyne.get_recent(5)
        nlu_result = self.nlu.understand(cleaned, context)

        # ── SINGLE ENTRY POINT ───────────────────────────────
        response = self.orchestrator.dispatch(cleaned, nlu_result)

        # REMOVE NLU RESPONSE FALLBACK FOR NON-CHAT
        if not response:
            response = "Done."

        # Post-processing
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and "response" in parsed:
                response = parsed["response"]
        except:
            pass

        print(f"Hestia: {response}")
        self.tts.speak(response)
        self.wake_detector.flush_audio_queue()

        bus.emit("interaction_logged", {
            "query": cleaned,
            "response": response,
            "intent": nlu_result.get("intent", "chat")
        })

        return response
    
    def run_voice_loop(self):
        """Alternate between wake word detection and STT, feeding into process_text."""
        print("[Voice] Listening for wake word ('hey hestia')…")
        try:
            while True:
                detected = self.wake_detector.listen_for_wake_word(timeout=30)
                if not detected:
                    continue

                self.tts.speak("Yes?")
                text = self.stt.listen_once(max_duration=10)

                if not text or len(text.strip()) < 2:
                    self.tts.speak("I didn't catch that.")
                    continue

                if text.lower().strip() in EXIT_WORDS:
                    self.tts.speak("Goodbye.")
                    break

                self.process_text(text)

        finally:
            if hasattr(self, "heartbeat"):
                self.heartbeat.stop()

    # ── RUNNERS ─────────────────────────────────────────────────────
    def run_cli_loop(self):
        try:
            while True:
                user_input = input("> ")
                if user_input.lower() in EXIT_WORDS:
                    break
                self.process_text(user_input)
        finally:
            if hasattr(self, "heartbeat"):
                self.heartbeat.stop()

# ── Entry ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hestia personal AI")
    parser.add_argument("--voice", action="store_true", help="Run in voice mode")
    args = parser.parse_args()

    hestia = Hestia()
    if args.voice:
        hestia.run_voice_loop()
    else:
        hestia.run_cli_loop()