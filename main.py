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

# ── Constants ────────────────────────────────────────────────────────
CORE_INTENTS = {
    "get_time", "get_date", "get_weather",
    "take_note", "get_history", "get_user_info",
    "save_name", "get_system_info",
}

EXIT_WORDS = {"bye", "exit", "stop", "shutdown"}

_FILLER_RE = re.compile(r'\b(uh|um|you know)\b\s*', re.IGNORECASE)

CHAT_SYSTEM_PROMPT = (
    "You are Hestia. Answer concisely in 1-2 sentences.\n\n"
    "Question: {query}"
)

# ── LLM Wrapper ─────────────────────────────────────────────────────
class HestiaLLM:
    def __init__(self, host, port, model):
        self.host = host
        self.port = port
        self.model = model

    def generate(self, prompt: str) -> str:
        return generate(prompt, model=self.model, host=self.host, port=self.port)

# ── Main ────────────────────────────────────────────────────────────
class Hestia:

    def __init__(self, config_path="config/laptop_config.yaml"):
        print("Initializing Hestia...")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

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

        # Modules
        self.athena = None
        if self.config.get("athena", {}).get("enabled", False):
            from modules.athena.engine import AthenaEngine
            self.llm = HestiaLLM(
                self.ollama_manager.host,
                self.ollama_manager.port,
                ollama_cfg.get("model", "mistral"),
            )
            self.athena = AthenaEngine(self.llm)

        self.mnemosyne = None
        if self.config.get("mnemosyne", {}).get("enabled", False):
            self.mnemosyne = MnemosyneEngine(self.llm if self.athena else self.nlu)

        if self.mnemosyne:
            mn = self.mnemosyne
            bus.on("interaction_logged", lambda data: mn.push(
                data["query"],
                data["response"],
                data["intent"]
            ))
        if self.mnemosyne:
            self.memory = self.mnemosyne  # TEMP alias (to be removed)
            self.mnemosyne_engine = self.mnemosyne

        self.iris = None
        if self.config.get("iris", {}).get("enabled", False):
            from modules.iris import IrisEngine
            self.iris = IrisEngine(self.nlu)

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

        self.wake_detector = WakeWordDetector()

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

        # Chat fallback if empty
        if not response:
            response = nlu_result.get("response", "") or "I'm not sure about that."

        # Post-processing
        try:
            json.loads(response)
            response = "I've handled that for you."
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

    # ── RUNNERS ─────────────────────────────────────────────────────
    def run_cli_loop(self):
        while True:
            user_input = input("> ")
            if user_input.lower() in EXIT_WORDS:
                break
            self.process_text(user_input)

# ── Entry ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    hestia = Hestia()
    hestia.run_cli_loop()