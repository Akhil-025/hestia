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
from core.actions import HestiaActions
from core.memory import HestiaMemory
from core.google_agent import HestiaGoogleAgent
from core.wake_word import WakeWordDetector
from core.ollama_manager import OllamaManager
from core.browser_agent import HestiaBrowserAgent
from core.ollama_client import generate

from skills.base import SkillLoader
from skills import browser_tasks

# Modules
from modules.hecate import HecateEngine
from modules.artemis import ArtemisEngine

# NEW
from modules.hestia.orchestrator import HestiaOrchestrator
from modules.chronos.engine import ChronosEngine
from modules.hermes.engine import HermesEngine
from modules.hephaestus.engine import HephaestusEngine

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
        self.memory = HestiaMemory(db_path)
        self.actions = HestiaActions(self.memory)

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
            from modules.mnemosyne.engine import MnemosyneEngine
            self.mnemosyne = MnemosyneEngine(self.nlu)

        self.iris = None
        if self.config.get("iris", {}).get("enabled", False):
            from modules.iris import IrisEngine
            self.iris = IrisEngine(self.nlu)

        self.artemis = ArtemisEngine()
        self.hecate = HecateEngine()

        # Browser + Google
        self.browser_agent = HestiaBrowserAgent()
        self.actions.set_browser_agent(self.browser_agent)

        google_cfg = self.config.get("google", {})
        self.google_agent = None
        if google_cfg.get("enabled", False):
            self.google_agent = HestiaGoogleAgent(
                credentials_path=google_cfg.get("credentials_path"),
                token_path=google_cfg.get("token_path"),
            )
            if self.google_agent.authenticate():
                self.actions.set_google_agent(self.google_agent)

        # ── ORCHESTRATOR (CORE CHANGE) ───────────────────────────────
        self.orchestrator = HestiaOrchestrator()
        self.orchestrator.register_hecate(self.hecate)

        # Register modules
        self.orchestrator.register(self.hecate)
        if self.athena: self.orchestrator.register(self.athena)
        if self.mnemosyne: self.orchestrator.register(self.mnemosyne)
        if self.iris: self.orchestrator.register(self.iris)
        self.orchestrator.register(self.artemis)

        # New modules
        self.chronos = ChronosEngine(memory=self.memory)
        self.orchestrator.register(self.chronos)

        if self.google_agent:
            self.hermes = HermesEngine(self.google_agent)
            self.orchestrator.register(self.hermes)

        self.hephaestus = HephaestusEngine(self.browser_agent)
        self.orchestrator.register(self.hephaestus)

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

        # Fast intent
        intent, entities = self.fast_intent(cleaned)

        if intent == "__handled__":
            return ""

        # NLU fallback
        if not intent:
            context = self.memory.get_recent(5)
            nlu_result = self.nlu.understand(cleaned, context)
        else:
            nlu_result = {
                "intent": intent,
                "entities": entities or {},
                "confidence": 0.9,
                "response": "",
            }

        # ── SINGLE ENTRY POINT ───────────────────────────────
        response = self.orchestrator.dispatch(cleaned, nlu_result)

        # Chat fallback if empty
        if not response and nlu_result.get("intent") == "chat":
            response = generate(
                CHAT_SYSTEM_PROMPT.format(query=cleaned),
                model=self.ollama_cfg.get("model", "mistral"),
                host=self.ollama_manager.host,
                port=self.ollama_manager.port,
            )

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

        self.memory.add_interaction(cleaned, response, nlu_result.get("intent", "chat"))

        if self.mnemosyne:
            try:
                self.mnemosyne.push(cleaned, response, nlu_result.get("intent", "chat"))
            except:
                pass

        return response

    # ── FAST INTENT (UNCHANGED) ─────────────────────────────────────
    def fast_intent(self, text: str):
        t = text.lower()

        if "time" in t:
            return "get_time", {}
        if "date" in t:
            return "get_date", {}
        if "weather" in t:
            return "get_weather", {}

        return None, None

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