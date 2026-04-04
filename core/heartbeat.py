
import threading
import time
import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from core.memory_summariser import MemorySummariser
from core.memory import HestiaMemory
from core.tts import HestiaTTS

class HestiaHeartbeat:
    """
    Periodically checks and processes tasks from HEARTBEAT.md.
    """
    def __init__(self, memory: HestiaMemory, tts: HestiaTTS, interval: int = 1800, summariser: "Optional[MemorySummariser]" = None):
        """
        Initialize the heartbeat with memory, tts, interval in seconds, and optional summariser.
        """
        self.memory = memory
        self.tts = tts
        self.interval = interval
        self.summariser = summariser
        self._running = False
        self._thread = threading.Thread(target=self._tick, daemon=True)

    def start(self) -> None:
        """
        Start the heartbeat thread.
        """
        self._running = True
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        """
        Stop the heartbeat thread.
        """
        self._running = False

    def _tick(self) -> None:
        """
        Loop, sleeping for interval seconds, and run heartbeat tasks.
        """
        while self._running:
            self._run_heartbeat()
            time.sleep(self.interval)

    def _run_heartbeat(self) -> None:
        """
        Read HEARTBEAT.md and process pending tasks.
        """
        try:
            root = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(root, os.pardir))
            heartbeat_path = os.path.join(project_root, "HEARTBEAT.md")
            if not os.path.exists(heartbeat_path):
                return
            with open(heartbeat_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("- [ ] "):
                    task = line[6:].strip()
                    self._evaluate_task(task)
        except Exception:
            pass  # Silently ignore errors

    def _evaluate_task(self, task: str) -> None:
        """
        Evaluate a single task string and act if it matches known patterns.
        """
        task_lower = task.lower()
        now = datetime.now()
        # Nightly summary (run between 0:00 and 5:00)
        if "nightly summary" in task_lower:
            current_hour = datetime.now().hour
            if 0 <= current_hour <= 5:
                if self.summariser:
                    result = self.summariser.run()
                    print(f"[Heartbeat] Nightly summary done: {result}", file=sys.stderr)
                return
        if "morning brief" in task_lower and 7 <= now.hour <= 9:
            self.tts.speak("Good morning! Here is your morning brief.")
            self._morning_brief()
        elif "reminder:" in task_lower:
            idx = task_lower.find("reminder:")
            reminder_text = task[idx + 9:].strip()
            if reminder_text:
                self.tts.speak(reminder_text)
        # Otherwise, skip (future: send to NLU)

    def _morning_brief(self) -> None:
        """
        Speak the current date, time, last 3 preferences, top user facts, and recent mood.
        """
        now = datetime.now()
        date_str = now.strftime("Today is %A, %B %d, %Y. The time is %I:%M %p.")
        self.tts.speak(date_str)
        prefs = self.memory.get_all_preferences()
        if prefs:
            last_prefs = list(prefs)[-3:]
            for pref in last_prefs:
                self.tts.speak(str(pref))

        # Inject top user facts into brief
        facts_str = self.memory.get_top_facts_for_context(limit=3)
        if facts_str:
            self.tts.speak("Here is what I know about you: " + facts_str.replace("Known about user: ", ""))

        # Mood check — if recent mood was negative or stressed, acknowledge it
        recent_moods = self.memory.get_recent_moods(days=3)
        if recent_moods:
            latest = recent_moods[0]["valence"]
            if latest in ("negative", "stressed"):
                self.tts.speak("You seemed a bit stressed recently. I hope today is better.")
