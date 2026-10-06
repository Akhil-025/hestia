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
    def __init__(self, interval: int = 1800, mnemosyne=None, diagnostics=None,
                 apollo=None, maintenance=None, artemis=None, hephaestus=None,
                 pluto=None, classifier=None, jobs_only=False, hermes=None):
        self.interval = interval
        # Backlog #92: HermesEngine gives a once-a-day email digest. Optional,
        # like artemis: a heartbeat built without it never runs the check.
        self.hermes = hermes
        # Backlog #4: the trained intent classifier is retrained weekly from
        # new labels and confident log lines. Optional, like the rest.
        self.classifier = classifier
        self._last_classifier_retrain_date = None
        self.mnemosyne = mnemosyne
        # Backlog #6. Optional: a heartbeat built without one (as in the
        # existing tests) simply never runs the review job, same as any
        # other feature gated on an injected collaborator elsewhere in
        # this codebase (see CoreModule's diagnostics param).
        self.diagnostics = diagnostics
        # Backlog #115/#118/#120/#161: ApolloEngine supplies hydration,
        # weekly-summary, goal-pace and burnout check-ins. Optional, like
        # diagnostics: a heartbeat built without it simply skips them.
        self.apollo = apollo
        # Backlog #129: ArtemisEngine supplies smart habit nudges. Optional, like apollo.
        self.artemis = artemis
        # Backlog #101: HephaestusEngine re-checks watched web pages. Optional,
        # like artemis: a heartbeat built without it never runs the check.
        self.hephaestus = hephaestus
        # Backlog #134: PlutoEngine raises budget alerts (80% / over). Optional, like artemis.
        self.pluto = pluto
        # Backlog #233: core.db_maintenance.DBMaintenance (optional).
        self.maintenance = maintenance
        self._last_maintenance_date = None
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
        self._maybe_retrain_classifier()
        self._maybe_run_apollo_checkins()
        self._maybe_run_artemis_checkins()
        self._maybe_run_db_maintenance()
        self._maybe_run_mnemosyne_jobs()
        self._maybe_run_hephaestus_checks()
        self._maybe_run_pluto_budget_alerts()
        self._maybe_run_pluto_price_alerts()
        self._maybe_run_hermes_digest()

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

    # Apollo hooks, in the order they're checked. Each returns text to speak
    # or None; the cadence, quiet hours, caps and sent-markers all live in
    # ApolloEngine (persisted in its DB), so this loop stays trivial.
    _APOLLO_HOOKS = (
        "check_hydration_nudge",
        "check_weekly_summary",
        "check_burnout",
        "check_goal_pace_reminder",
    )

    def _maybe_run_artemis_checkins(self) -> None:
        hook = getattr(self.artemis, "check_habit_nudges", None)
        if not callable(hook):
            return
        try:
            text = hook()
        except Exception:
            logging.getLogger(__name__).exception("Artemis habit nudge check failed.")
            return
        if isinstance(text, str) and text.strip():
            bus.emit("speak", {"text": text})

    def _maybe_run_pluto_budget_alerts(self) -> None:
        """Backlog #134: speak a budget alert when a category newly crosses 80% or its limit.

        PlutoEngine remembers what it has already announced (per month and
        category) and holds alerts during its quiet hours, so this is safe on every tick.
        """
        hook = getattr(self.pluto, "check_budget_alerts", None)
        if not callable(hook):
            return
        try:
            text = hook()
        except Exception:
            logging.getLogger(__name__).exception("Pluto budget alert check failed.")
            return
        if isinstance(text, str) and text.strip():
            bus.emit("speak", {"text": text})

    def _maybe_run_pluto_price_alerts(self) -> None:
        """Backlog #132: speak a price-move or headline alert for held/watched stocks.

        Opt-in and self-throttled inside PlutoEngine (about every 30 minutes,
        quiet hours respected), so this is safe on every tick.
        """
        hook = getattr(self.pluto, "check_price_alerts", None)
        if not callable(hook):
            return
        try:
            text = hook()
        except Exception:
            logging.getLogger(__name__).exception("Pluto price alert check failed.")
            return
        if isinstance(text, str) and text.strip():
            bus.emit("speak", {"text": text})

    def _maybe_run_hermes_digest(self) -> None:
        """Backlog #92: speak the morning email digest once a day.

        HermesEngine keeps the schedule (digest_time, once per day, a window
        so a late laptop start doesn't read a "morning" digest at night), so
        this is safe to call on every tick. Off unless hermes.digest_time is set.
        """
        hook = getattr(self.hermes, "check_email_digest", None)
        if not callable(hook):
            return
        try:
            text = hook()
        except Exception:
            logging.getLogger(__name__).exception("Hermes email digest check failed.")
            return
        if isinstance(text, str) and text.strip():
            bus.emit("speak", {"text": text})

    def _maybe_run_hephaestus_checks(self) -> None:
        """Backlog #101: re-check watched web pages and speak any alerts.

        Each watch has its own interval and HephaestusEngine decides what is
        due, so this is safe to call on every tick. Alerts raised in the
        engine's quiet hours are held there and come out on a later tick.
        """
        hook = getattr(self.hephaestus, "check_web_monitors", None)
        if not callable(hook):
            return
        try:
            text = hook()
        except Exception:
            logging.getLogger(__name__).exception("Hephaestus page check failed.")
            return
        if isinstance(text, str) and text.strip():
            bus.emit("speak", {"text": text})

    def _maybe_run_apollo_checkins(self) -> None:
        if self.apollo is None:
            return
        log = logging.getLogger(__name__)
        for name in self._APOLLO_HOOKS:
            hook = getattr(self.apollo, name, None)
            if not callable(hook):
                continue
            try:
                text = hook()
            except Exception:
                log.exception("Apollo check-in %s failed.", name)
                continue
            if isinstance(text, str) and text.strip():
                bus.emit("speak", {"text": text})

    def _maybe_run_mnemosyne_jobs(self) -> None:
        """Backlog #39/#43/#47: episode clustering, Obsidian sync, paper check.

        The cadence of each job (hourly / 30 min / daily) lives in
        MnemosyneEngine.run_background_jobs, so this is safe to call on
        every tick. A heartbeat built with a bare mock (as in the existing
        tests) or an older engine without the method simply skips it.
        """
        job = getattr(self.mnemosyne, "run_background_jobs", None) if self.mnemosyne else None
        if not callable(job):
            return
        try:
            job()
        except Exception:
            logger.exception("Mnemosyne background jobs failed.")

    def _maybe_run_db_maintenance(self) -> None:
        """Backlog #233: SQLite housekeeping in the 0-5am off-peak window.

        Runs at most once per day here (DBMaintenance additionally limits
        each database to once a week). If a database was busy the day is not
        marked done, so the next tick inside the window retries.
        """
        if self.maintenance is None:
            return
        now = datetime.now()
        if not (0 <= now.hour < 5):
            return
        if self._last_maintenance_date == now.date():
            return
        log = logging.getLogger(__name__)
        try:
            results = self.maintenance.run()
        except Exception:
            log.exception("DB maintenance failed.")
            return
        if not self.maintenance.needs_retry(results):
            self._last_maintenance_date = now.date()

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
        # Backlog #33: spaced-repetition cards due today. Spoken before the
        # full brief is generated so it is never lost if that step fails.
        study = getattr(self.mnemosyne, "get_study_brief", None) if self.mnemosyne else None
        if callable(study):
            try:
                line = study()
            except Exception:
                logger.exception("Study brief failed.")
                line = ""
            if isinstance(line, str) and line.strip():
                bus.emit("speak", {"text": line})
        # Backlog #36: facts the decay job flagged as stale.
        stale = getattr(self.mnemosyne, "get_stale_brief", None) if self.mnemosyne else None
        if callable(stale):
            try:
                line = stale()
            except Exception:
                logger.exception("Stale-facts brief failed.")
                line = ""
            if isinstance(line, str) and line.strip():
                bus.emit("speak", {"text": line})
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

    def _maybe_retrain_classifier(self) -> None:
        """Retrain the intent classifier at most once every 7 days, off-peak
        (backlog #4). Skipped quietly when there is no classifier or too
        little new data; a failed run leaves the previous model serving."""
        if self.classifier is None or not getattr(self.classifier, "enabled", False):
            return
        now = datetime.now()
        if not (0 <= now.hour <= 5):
            return
        today = date.today()
        last = self._last_classifier_retrain_date
        if last is not None and (today - last).days < 7:
            return
        self._last_classifier_retrain_date = today
        try:
            result = self.classifier.retrain_if_needed()
        except Exception:
            logger.exception("Classifier retrain job failed.")
            return
        logger.info("Classifier retrain check: %s", result.get("skipped") or result)

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