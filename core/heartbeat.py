# core/heartbeat.py 

import threading
import time
import os
import sys
from datetime import datetime, date
from core.event_bus import bus

import logging

logger = logging.getLogger(__name__)

class HestiaHeartbeat:
    def __init__(self, interval: int = 1800, mnemosyne=None, diagnostics=None):
        self.interval = interval
        self.mnemosyne = mnemosyne
        # Backlog #6. Optional: a heartbeat built without one (as in the
        # existing tests) simply never runs the review job, same as any
        # other feature gated on an injected collaborator elsewhere in
        # this codebase (see CoreModule's diagnostics param).
        self.diagnostics = diagnostics
        self._running = False
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._last_brief_date = None          # tracks date of last morning brief
        self._last_review_date = None         # tracks date of last low-confidence review
        self._last_weekly_review_date = None  # tracks date of last per-intent accuracy review
        self._reminder_last_fired: dict = {}  # task_text -> timestamp
        # Set to False by main.py once ChronosEngine's own scheduler has taken
        # over reminder delivery (backlog #81-#89), so a one-shot reminder is
        # never announced by both.
        self.handle_reminders = True

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
        if self.mnemosyne and self.handle_reminders:
            reminders = self.mnemosyne.get_due_reminders()
            for rid, text in reminders:
                bus.emit("speak", {"text": f"Reminder: {text}"})
                self.mnemosyne.mark_reminder_done(rid)

        # Backlog #6: once a day, review the last day's low-confidence
        # classifications and queue them for manual labelling. Checked
        # unconditionally here (like the reminder check above) rather than
        # only when a matching line exists in HEARTBEAT.md — this is a
        # standing observability feature, not a user-authored task, and
        # shouldn't depend on remembering to add a line for it.
        self._maybe_run_low_confidence_review()
        self._maybe_run_weekly_accuracy_review()

        try:
            root = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(root, os.pardir))
            heartbeat_path = os.path.join(project_root, "HEARTBEAT.md")

            if not os.path.exists(heartbeat_path):
                return

            with open(heartbeat_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # NOTE: `- [ ] ` here does not mean "one-time to-do" — every
            # matching line is a *recurring* condition that gets
            # re-evaluated on every tick, and this code intentionally never
            # rewrites the file to check a box off. Each task's own
            # in-memory state (see `_last_brief_date`, `_reminder_last_fired`
            # below) is what controls how often it actually fires. See
            # HEARTBEAT.md for the same note aimed at anyone editing tasks.
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

    def _maybe_run_low_confidence_review(self) -> None:
        """
        Run the nightly low-confidence review once per calendar day
        (backlog #6), during the same off-peak window as the nightly
        summary above (0-5am) so it doesn't compete with active daytime
        use for whatever's making the classification calls.

        Mirrors `_last_brief_date`'s date-tracking exactly: an in-memory
        "have I already run today" flag, not a HEARTBEAT.md checkbox —
        see the note above `_run_heartbeat` for why tasks track their own
        firing state instead of the file being rewritten.
        """
        if self.diagnostics is None:
            return

        now = datetime.now()
        if not (0 <= now.hour <= 5):
            return

        today = date.today()
        if self._last_review_date == today:
            return
        self._last_review_date = today

        try:
            added = self.diagnostics.write_review_queue()
        except Exception:
            logger.exception("Low-confidence review job failed.")
            return

        if added:
            logger.info(
                "Nightly review: %d new low-confidence classification(s) "
                "queued for manual labelling.", added,
            )
            bus.emit(
                "low_confidence_review_ready",
                {"count": added, "summary": self.diagnostics.review_queue_summary()},
            )
        else:
            logger.debug("Nightly review: nothing new to queue.")

    def _maybe_run_weekly_accuracy_review(self) -> None:
        """
        Run the per-intent accuracy review once every 7 days (backlog
        #30), in the same off-peak window as the other nightly jobs.

        Unlike `_last_brief_date`/`_last_review_date`'s "already ran
        today" check, this tracks a 7-day gap rather than a calendar day —
        a weekly job that instead fired once per calendar week would run
        twice in quick succession across a week boundary (Sunday night,
        then Monday morning) and then not again for six days, which is a
        worse cadence than a straightforward rolling week.
        """
        if self.diagnostics is None:
            return

        now = datetime.now()
        if not (0 <= now.hour <= 5):
            return

        today = date.today()
        if (
            self._last_weekly_review_date is not None
            and (today - self._last_weekly_review_date).days < 7
        ):
            return
        self._last_weekly_review_date = today

        try:
            summary = self.diagnostics.weekly_accuracy_summary()
            worst = self.diagnostics.worst_performing_intents()
        except Exception:
            logger.exception("Weekly accuracy review job failed.")
            return

        logger.info("Weekly accuracy review:\n%s", summary)
        flagged = [w for w in worst if w[1]["flagged_wrong"] > 0]
        if flagged:
            bus.emit(
                "weekly_accuracy_review_ready",
                {"summary": summary, "worst": flagged},
            )