# core/heartbeat.py 

import threading
import time
import os
import sys
from datetime import datetime, date
from core.event_bus import bus

class HestiaHeartbeat:
    def __init__(self, interval: int = 1800, mnemosyne=None):
        self.interval = interval
        self.mnemosyne = mnemosyne
        self._running = False
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._last_brief_date = None          # tracks date of last morning brief
        self._reminder_last_fired: dict = {}  # task_text -> timestamp

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
        if self.mnemosyne:
            reminders = self.mnemosyne.get_due_reminders()
            for rid, text in reminders:
                bus.emit("speak", {"text": f"Reminder: {text}"})
                self.mnemosyne.mark_reminder_done(rid)

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
            pass

    def _evaluate_task(self, task: str) -> None:
        task_lower = task.lower()
        now = datetime.now()

        if "nightly summary" in task_lower:
            if 0 <= now.hour <= 5:
                if self.mnemosyne and self.mnemosyne.summariser:
                    bus.emit("mnemosyne_summarise", {})
            return

        if "morning brief" in task_lower:
            if 7 <= now.hour <= 9:
                today = date.today()
                if self._last_brief_date != today:
                    self._last_brief_date = today
                    self._morning_brief()
            return

        if "reminder:" in task_lower:
            idx = task_lower.find("reminder:")
            reminder_text = task[idx + 9:].strip()
            if not reminder_text:
                return

            # Cooldown: don't re-fire the same reminder within 4 hours
            last = self._reminder_last_fired.get(reminder_text, 0)
            cooldown = 4 * 3600  # 4 hours in seconds
            if time.time() - last < cooldown:
                return

            self._reminder_last_fired[reminder_text] = time.time()
            bus.emit("speak", {"text": reminder_text})
            return

        bus.emit("heartbeat_unhandled_task", {"task": task})

    def _morning_brief(self) -> None:
        now = datetime.now()
        date_str = now.strftime("Today is %A, %B %d, %Y. The time is %I:%M %p.")
        bus.emit("speak", {"text": "Good morning! Here is your morning brief."})
        bus.emit("speak", {"text": date_str})
        bus.emit("morning_brief_requested", {})