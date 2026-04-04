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

# ── Silence noisy third-party output BEFORE any imports that trigger them ────
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

for _noisy in (
    "huggingface_hub", "transformers", "sentence_transformers",
    "torch", "urllib3", "httpx", "httpcore", "asyncio", "werkzeug",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("hestia").setLevel(logging.INFO)

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
from skills import browser_tasks
from web_ui import HestiaWebUI
from core.ollama_client import generate
from core.telegram_bot import HestiaTelegramBot
from skills.base import SkillLoader
from modules.hecate import HecateEngine

log = logging.getLogger("hestia")

# ── Constants ─────────────────────────────────────────────────────────────────

CORE_INTENTS = {
    "get_time", "get_date", "get_weather",
    "take_note", "get_history", "get_user_info",
    "save_name", "get_system_info",
}

EXIT_WORDS = {"bye", "exit", "stop", "shutdown"}

_ATHENA_TRIGGERS = [
    "from my notes", "in my documents", "from my files",
    "according to my notes", "what does my", "explain from",
    "in my notes", "from my docs",
]

_MNEMOSYNE_TRIGGERS = [
    "do you remember", "what do you know about me",
    "what are my goals", "remind me", "what did we talk about",
    "what have i told you", "my goals", "forget that",
]

_IRIS_TRIGGERS = [
    "in my photos", "in my pictures", "in my images", "in my videos",
    "in my media", "in my gallery", "from my photos", "from my pictures",
    "from my images", "from my videos", "from my media", "from my gallery",
    "search my photos", "search my pictures", "search my images",
    "search my videos", "search my media", "search my gallery",
    "find photo", "find image", "find video", "find picture",
    "find media file", "find media", "ingest media", "ingest photos",
    "ingest images", "ingest videos", "ingest gallery",
    "analyse my photos", "analyze my photos", "describe my photos",
]

_FILLER_RE = re.compile(r'\b(uh|um|you know)\b\s*', re.IGNORECASE)
_NOTE_RE   = re.compile(
    r'^(take a note|note down|remember this|jot this)\s*[:\-]?\s*',
    re.IGNORECASE,
)

CHAT_SYSTEM_PROMPT = (
    "You are Hestia, a warm and witty personal assistant. "
    "Answer the following question concisely and accurately in 1-2 sentences. "
    "Do not use poetic language or metaphors. Just answer directly.\n\n"
    "Question: {query}"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def match_trigger(text: str, triggers: list) -> bool:
    """Word-boundary-aware trigger matching to avoid false positives."""
    return any(re.search(r"\b" + re.escape(t) + r"\b", text) for t in triggers)


# ── LLM wrapper ───────────────────────────────────────────────────────────────

class HestiaLLM:
    def __init__(self, host, port, model):
        self.host  = host
        self.port  = port
        self.model = model

    def generate(self, prompt: str) -> str:
        return generate(prompt, model=self.model, host=self.host, port=self.port)


# ── Main orchestrator ─────────────────────────────────────────────────────────

class Hestia:
    """Main orchestrator: voice/CLI, STT, NLU, actions, memory."""

    def __init__(self, config_path: str = "config/laptop_config.yaml"):
        print("Initializing Hestia...")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")

        # Memory (must be first)
        db_path      = self.config.get("database", {}).get("path", "data/hestia.db")
        self.memory  = HestiaMemory(db_path)
        self.actions = HestiaActions(self.memory)

        # Ollama
        ollama_cfg          = self.config.get("ollama", {})
        self.ollama_cfg     = ollama_cfg
        self.ollama_manager = OllamaManager(
            host=ollama_cfg.get("host", "localhost"),
            port=ollama_cfg.get("port", 11434),
        )
        if not self.ollama_manager.ensure_running():
            print("WARNING: Ollama backend not available.")
        time.sleep(2)

        # NLU
        llm_cfg       = self.config.get("llm", {})
        llm_providers = llm_cfg.get("providers", None)
        self.nlu = HestiaNLU(
            model=ollama_cfg.get("model", "mistral"),
            host=ollama_cfg.get("host", "127.0.0.1"),
            port=ollama_cfg.get("port", 11434),
            prompt_path=self.config.get("nlu", {}).get("prompt_path", "config/nlu_prompt.txt"),
            providers=llm_providers,
        )

        # Athena
        self.athena = None
        athena_cfg  = self.config.get("athena", {})
        if athena_cfg.get("enabled", False):
            try:
                from modules.athena.engine import AthenaEngine
                self.llm    = HestiaLLM(
                    host=self.ollama_manager.host,
                    port=self.ollama_manager.port,
                    model=ollama_cfg.get("model", "mistral"),
                )
                self.athena = AthenaEngine(hestia_llm=self.llm)
                log.info("[Athena] Knowledge engine loaded.")
            except Exception as e:
                log.warning("[Athena] Failed to load: %s", e)

        # Mnemosyne
        self.mnemosyne = None
        mnemosyne_cfg  = self.config.get("mnemosyne", {})
        if mnemosyne_cfg.get("enabled", False):
            try:
                from modules.mnemosyne.engine import MnemosyneEngine
                self.mnemosyne = MnemosyneEngine(hestia_llm=self.nlu)
                log.info("[Mnemosyne] Memory engine loaded.")
            except Exception as e:
                log.warning("[Mnemosyne] Failed to load: %s", e)

        # Iris
        self.iris = None
        iris_cfg  = self.config.get("iris", {})
        if iris_cfg.get("enabled", False):
            try:
                from modules.iris import IrisEngine
                self.iris = IrisEngine(hestia_llm=self.nlu)
                log.info("[Iris] Media engine loaded.")
            except Exception as e:
                log.warning("[Iris] Failed to load: %s", e)

        # Hecate
        self.hecate = HecateEngine()
        log.info("[Hecate] Decision engine ready.")

        # Active modules list — computed once at init, never changes at runtime
        self.active_modules = ["core"]
        if self.athena:    self.active_modules.append("athena")
        if self.mnemosyne: self.active_modules.append("mnemosyne")
        if self.iris:      self.active_modules.append("iris")

        # STT
        stt_cfg  = self.config.get("stt", {})
        self.stt = HestiaSTT(
            model_size=stt_cfg.get("model_size", "base.en"),
            device=stt_cfg.get("device", "cuda"),
            compute_type=stt_cfg.get("compute_type", "float16"),
            samplerate=stt_cfg.get("samplerate", 16000),
            noise_filter=stt_cfg.get("noise_filter", True),
        )

        # TTS
        tts_cfg = self.config.get("tts", {})
        self.nlu.set_memory(self.memory)
        self.tts = HestiaTTS(
            engine=tts_cfg.get("engine", "pyttsx3"),
            rate=tts_cfg.get("rate", 180),
            volume=tts_cfg.get("volume", 1.0),
            piper_model_path=tts_cfg.get("piper_model_path", "models/piper/en_US-lessac-medium.onnx"),
        )

        # Wake word
        wake_cfg           = self.config.get("wake_word", {})
        self.wake_detector = WakeWordDetector(
            model_path=wake_cfg.get("model_path", "models/vosk-model-small-en-us-0.15"),
            wake_words=wake_cfg.get("wake_words", None),
        )

        # Skills
        skills_dir        = self.config.get("skills", {}).get("path", "skills")
        self.skill_loader = SkillLoader(skills_dir=skills_dir)
        skill_count       = self.skill_loader.load_all()
        if skill_count > 0:
            skill_block = self.skill_loader.inject_into_nlu_prompt(
                self.config.get("nlu", {}).get("prompt_path", "config/nlu_prompt.txt")
            )
            if skill_block:
                self.nlu.system_prompt += skill_block
            log.info("[Skills] %d skill(s) loaded.", skill_count)

        # Browser
        self.browser_agent = HestiaBrowserAgent()
        self.actions.set_browser_agent(self.browser_agent)
        browser_tasks.set_browser_agent(self.browser_agent)

        # Google
        google_cfg = self.config.get("google", {})
        if google_cfg.get("enabled", False):
            self.google_agent = HestiaGoogleAgent(
                credentials_path=google_cfg.get("credentials_path", "config/google_credentials.json"),
                token_path=google_cfg.get("token_path", "data/google_token.json"),
            )
            if self.google_agent.authenticate():
                self.actions.set_google_agent(self.google_agent)
                log.info("[Google] Gmail and Calendar connected.")
            else:
                log.warning("[Google] Authentication failed — Gmail/Calendar disabled.")
                self.google_agent = None
        else:
            self.google_agent = None

        # Web UI
        webui_cfg = self.config.get("webui", {})
        if webui_cfg.get("enabled", False):
            self.web_ui = HestiaWebUI(
                memory=self.memory,
                skill_loader=self.skill_loader,
                process_fn=self.process_text,
                host=webui_cfg.get("host", "127.0.0.1"),
                port=webui_cfg.get("port", 5000),
            )
            self.web_ui.start()
        else:
            self.web_ui = None

        # Telegram
        telegram_cfg = self.config.get("telegram", {})
        if telegram_cfg.get("enabled", False):
            env_token    = os.getenv("TELEGRAM_BOT_TOKEN")
            config_token = telegram_cfg.get("token", "")
            token        = env_token if env_token else config_token
            allowed      = telegram_cfg.get("allowed_chat_ids", [])

            if not token:
                raise ValueError("TELEGRAM_BOT_TOKEN not set")
            if token == "YOUR_BOT_TOKEN_HERE":
                raise ValueError("Invalid placeholder token")

            self.telegram_bot = HestiaTelegramBot(
                token=token,
                process_fn=self.process_text,
                allowed_chat_ids=allowed if allowed else None,
                stt=self.stt,
            )
            self.telegram_bot.start()
            log.info("[Telegram] Bot started.")
        else:
            self.telegram_bot = None

        print("Hestia is ready.")
        user_name = self.memory.get_preference("user_name", "")
        greeting  = f"Welcome back, {user_name}." if user_name else "Hey there, I'm Hestia."
        self.tts.speak(greeting)
        self.wake_detector.flush_audio_queue()

    # ── Core query processor ──────────────────────────────────────────────────

    def process_text(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        cleaned = _FILLER_RE.sub('', text.lower().strip()).strip()
        print(f"You: {cleaned}")

        # ── Step 1: Fast rule-based intent ───────────────────────────────────
        intent, entities = self.fast_intent(cleaned)

        if intent == "__handled__":
            return ""

        # ── Step 2: NLU fallback if no fast intent ────────────────────────────
        if not intent:
            context    = self.memory.get_recent(5)
            nlu_result = self.nlu.understand(cleaned, context)
            intent     = nlu_result.get("intent", "chat")
            entities   = nlu_result.get("entities", {})
        else:
            nlu_result = {"intent": intent, "entities": entities or {}, "confidence": 0.9, "response": ""}

        # ── Step 3: Routing — priority order ──────────────────────────────────
        # Hard triggers always win over Hecate
        if match_trigger(cleaned, _ATHENA_TRIGGERS) and self.athena:
            primary    = "athena"
            secondary  = []
            confidence = 1.0

        elif match_trigger(cleaned, _MNEMOSYNE_TRIGGERS) and self.mnemosyne:
            primary    = "mnemosyne"
            secondary  = []
            confidence = 1.0

        elif match_trigger(cleaned, _IRIS_TRIGGERS) and self.iris:
            primary    = "iris"
            secondary  = []
            confidence = 1.0

        elif intent in CORE_INTENTS:
            primary    = "core"
            secondary  = []
            confidence = 1.0

        else:
            try:
                decision   = self.hecate.decide(cleaned, nlu_result, self.active_modules)
            except Exception as e:
                log.debug("[Hecate] Failure: %s", e)
                decision   = {"primary": "core", "secondary": [], "confidence": 0.0}

            primary    = decision.get("primary", "core")
            secondary  = decision.get("secondary", [])
            confidence = decision.get("confidence", 0.0)

            if confidence < 0.4:
                primary = "core"

        # ── Step 4: Execute routed handler ────────────────────────────────────
        final_response = ""

        if primary == "athena" and self.athena:
            try:
                final_response = self.athena.query(cleaned)
            except Exception as e:
                log.debug("[Athena] Query error: %s", e)
                final_response = "I had trouble searching your documents."

        elif primary == "mnemosyne" and self.mnemosyne:
            try:
                final_response = self.mnemosyne.remember(cleaned)
            except Exception as e:
                log.debug("[Mnemosyne] Remember error: %s", e)
                final_response = "I had trouble accessing my memory."

        elif primary == "iris" and self.iris:
            try:
                if any(w in cleaned for w in ["organize", "index", "sort", "scan", "catalog"]):
                    result         = self.iris.ingest()
                    final_response = f"Done. Indexed {result.get('ingested', 0)} new files."
                elif any(w in cleaned for w in ["analyse", "analyze", "describe"]):
                    final_response = self.iris.analyse(limit=20)
                elif any(w in cleaned for w in ["ingest", "import", "add"]):
                    stats          = self.iris.ingest()
                    final_response = (
                        f"Media ingestion complete. "
                        f"{stats.get('ingested', 0)} files processed, "
                        f"{stats.get('duplicates_skipped', 0)} duplicates skipped."
                    )
                else:
                    result         = self.iris.search(cleaned)
                    final_response = result if result else "I couldn't find any relevant media."
            except Exception as e:
                log.debug("[Iris] Error: %s", e)
                final_response = "I had trouble accessing your media files."

        else:
            # Core / actions path
            action_response = self.actions.execute(intent, entities, raw_query=cleaned)

            # Optional secondary enrichment from Mnemosyne
            if "mnemosyne" in secondary and self.mnemosyne and confidence > 0.6:
                try:
                    memory_result = self.mnemosyne.remember(cleaned)
                    if memory_result and "don't have any relevant" not in memory_result:
                        action_response = (action_response or "") + "\n" + memory_result
                except Exception as e:
                    log.debug("[Mnemosyne] Recall error: %s", e)

            # Chat fallback via Ollama
            if intent == "chat" and not action_response:
                try:
                    action_response = generate(
                        CHAT_SYSTEM_PROMPT.format(query=cleaned),
                        model=self.ollama_cfg.get("model", "mistral"),
                        host=self.ollama_manager.host,
                        port=self.ollama_manager.port,
                    )
                except Exception as e:
                    log.debug("[Chat] Ollama error: %s", e)
                    action_response = nlu_result.get("response", "") or "I'm not sure about that."

            final_response = action_response or nlu_result.get("response", "") or "I'm not sure about that."

        # ── Step 5: Post-processing ───────────────────────────────────────────
        if intent == "save_name":
            name = self.memory.get_preference("user_name", "")
            if name:
                final_response = f"{name}, what a lovely name. I won't forget it."

        # Guard: if response is raw JSON, replace with neutral acknowledgement
        try:
            json.loads(final_response)
            final_response = "I've handled that for you."
        except (json.JSONDecodeError, TypeError):
            pass

        print(f"Hestia: {final_response}")
        self.tts.speak(final_response)
        self.wake_detector.flush_audio_queue()

        # Single memory write — no duplicate logging across athena/mnemosyne/iris paths
        self.memory.add_interaction(cleaned, final_response, intent)

        if self.mnemosyne:
            try:
                self.mnemosyne.push(cleaned, final_response, intent)
            except Exception as e:
                log.debug("[Mnemosyne] Push error: %s", e)

        return final_response

    # ── Fast rule-based intent detection ─────────────────────────────────────

    def fast_intent(self, text: str):
        t      = text.lower().strip()
        recent = self.memory.get_recent(1)

        if recent and recent[0].get("intent") == "get_time" and "tomorrow" in t:
            return "get_date", {}

        time_match    = any(p in t for p in ["what time", "current time", "tell me the time", "what's the time"])
        date_match    = any(p in t for p in ["what date", "today's date", "what day", "what is the date", "what is today"])
        weather_match = "weather" in t

        if "calendar" in t or "schedule" in t or "events" in t:
            return "list_events", {"days": 7}

        if "email" in t or "mail" in t or "inbox" in t:
            return "read_email", {"count": 5}

        # Compound: time + weather answered together, marked as handled
        if time_match and weather_match:
            time_resp    = self.actions.execute("get_time", {}, raw_query=t)
            weather_resp = self.actions.execute("get_weather", {}, raw_query=t)
            combined     = f"{time_resp} {weather_resp}"
            print(f"Hestia: {combined}")
            self.tts.speak(combined)
            self.wake_detector.flush_audio_queue()
            self.memory.add_interaction(t, combined, "compound")
            return "__handled__", {}

        if time_match:    return "get_time", {}
        if date_match:    return "get_date", {}
        if weather_match: return "get_weather", {}

        if "my name is" in t:
            parts = t.split("my name is")
            if len(parts) > 1:
                name = parts[-1].strip().split()[0]
                return "save_name", {"name": name}

        if "call me" in t:
            parts = t.split("call me")
            if len(parts) > 1:
                name = parts[-1].strip().split()[0]
                return "save_name", {"name": name}

        if "what is my name" in t or "what's my name" in t:
            return "get_user_info", {"key": "user_name"}

        if "who am i" in t:
            return "get_user_info", {"key": "user_name"}

        if "history" in t or "what did we talk" in t:
            return "get_history", {"limit": 5}

        if "system info" in t or "system information" in t:
            return "get_system_info", {}

        if any(p in t for p in ["note", "remember this", "jot this"]):
            cleaned = _NOTE_RE.sub('', t).strip()
            return "take_note", {"text": cleaned}

        if any(t.startswith(p) for p in ["what is ", "what are ", "how does ", "how do ", "explain ", "tell me about "]):
            return "chat", {}

        if t in {"how are you", "how are you?", "hey", "hello", "hi"}:
            return "chat", {}

        return None, None

    # ── Run loops ─────────────────────────────────────────────────────────────

    def run_voice_loop(self) -> None:
        self.tts.speak("I'll be here. Just say Hey Hestia when you need me.")
        self.wake_detector.flush_audio_queue()
        try:
            while True:
                detected = self.wake_detector.listen_for_wake_word()
                if not detected:
                    continue
                self.tts.speak("Hmm?")
                self.wake_detector.flush_audio_queue()
                text = self.stt.listen_once()
                if not text or len(text.strip()) < 2:
                    self.tts.speak("Say that again?")
                    continue
                # Exact-word check — avoids "don't stop the music" false triggers
                if text.lower().strip() in EXIT_WORDS:
                    self.tts.speak("Okay, resting now.")
                    break
                self.process_text(text)
                last_activity = time.time()
                while True:
                    follow_up = self.stt.listen_once(max_duration=4)
                    if follow_up and len(follow_up.strip()) > 2:
                        self.process_text(follow_up)
                        last_activity = time.time()
                    elif time.time() - last_activity > 6:
                        break
        except KeyboardInterrupt:
            print()
            self.tts.speak("Take care. I'll be here when you return.")

    def run_cli_loop(self) -> None:
        print("--- CLI Mode (type 'quit') ---")
        try:
            while True:
                user_input = input("> ")
                if user_input.lower().strip() in EXIT_WORDS | {"quit", "goodbye", "good bye"}:
                    break
                self.process_text(user_input)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.tts.speak("Okay, resting now.")
        self.tts.wait_until_done()
        if hasattr(self, "heartbeat") and self.heartbeat:
            self.heartbeat.stop()
        if hasattr(self, "telegram_bot") and self.telegram_bot:
            try:
                self.telegram_bot.stop()
            except RuntimeError:
                pass
        if hasattr(self, "browser_agent") and self.browser_agent:
            self.browser_agent.close()
        sys.exit(0)


if __name__ == "__main__":
    hestia = Hestia()
    if "--cli" in sys.argv:
        hestia.run_cli_loop()
    else:
        hestia.run_voice_loop()