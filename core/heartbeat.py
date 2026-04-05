# core/heartbeat.py

import threading
import time
import os
import sys
from datetime import datetime
from core.event_bus import bus

class HestiaHeartbeat:
    """
    Periodically checks and processes tasks from HEARTBEAT.md
    using an event-driven architecture.
    """

    def __init__(self, interval: int = 1800, mnemosyne=None):
        self.interval = interval
        self.mnemosyne = mnemosyne
        self._running = False
        self._thread = threading.Thread(target=self._tick, daemon=True)

    def start(self) -> None:
        self._running = True
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _tick(self) -> None:
        while self._running:
            self._run_heartbeat()
            time.sleep(self.interval)

    def _run_heartbeat(self) -> None:
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
            pass  # intentionally silent

    def _evaluate_task(self, task: str) -> None:
        task_lower = task.lower()
        now = datetime.now()

        # Nightly summary
        if "nightly summary" in task_lower:
            if 0 <= now.hour <= 5:
                if self.mnemosyne and self.mnemosyne.summariser:
                    bus.emit("mnemosyne_summarise", {})
                    print("[Heartbeat] Nightly summary done", file=sys.stderr)
                return

        # Morning brief trigger
        if "morning brief" in task_lower and 7 <= now.hour <= 9:
            self._morning_brief()

        # Reminder handling
        elif "reminder:" in task_lower:
            idx = task_lower.find("reminder:")
            reminder_text = task[idx + 9:].strip()
            if reminder_text:
                bus.emit("speak", {"text": reminder_text})

        # Future: send unknown tasks to NLU
        else:
            bus.emit("heartbeat_unhandled_task", {"task": task})

    def _morning_brief(self) -> None:
        """
        Emit events instead of directly accessing memory or TTS.
        """
        now = datetime.now()

        date_str = now.strftime(
            "Today is %A, %B %d, %Y. The time is %I:%M %p."
        )

        # Speak intro + time
        bus.emit("speak", {"text": "Good morning! Here is your morning brief."})
        bus.emit("speak", {"text": date_str})

        # Delegate full brief construction
        bus.emit("morning_brief_requested", {})