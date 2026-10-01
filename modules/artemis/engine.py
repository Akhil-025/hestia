# artemis/engine.py
"""
modules/artemis/engine.py

ArtemisEngine: JSON-backed habit and goal tracker, plus AI-generated
motivation and productivity coaching.

Design notes
------------
- State reads/writes go through ArtemisTracker; TrackerError (corrupted or
  temporarily-unwritable state) is caught once in handle() so it degrades
  to a clear response instead of bubbling up to the orchestrator.
- Per-intent handlers catch HabitNotFoundError/GoalNotFoundError/ValueError
  locally, same as before — only the "this isn't the caller's fault" class
  of failure is handled centrally.
- LLM calls are isolated in a single helper (_llm) that raises a typed
  LLMError on failure; get_motivation and productivity_summary catch it
  and fall back to a static response, same isolated-call-with-static-
  fallback pattern Apollo uses, so Artemis never breaks if Ollama's down.
- Every public method conforms to the BaseModule response contract:
  {response: str, data: dict, confidence: float}.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

from core.event_bus import bus
from core.ollama_client import generate
from core.free_apis import (
    FreeAPIError,
    is_public_holiday as _fa_is_public_holiday,
    suggest_activity as _fa_suggest_activity,
)
from modules.base import BaseModule
from . import extras
from .extras import BADGES
from .templates import GOAL_TEMPLATES, find_template, template_names
from .tracker import (
    ArtemisTracker,
    Goal,
    GoalNotFoundError,
    Habit,
    HabitNotFoundError,
    MAX_GRACE_DAYS,
    TrackerError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "mistral"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 11434


@dataclass(frozen=True)
class OllamaConfig:
    model: str = _DEFAULT_MODEL
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "OllamaConfig":
        return cls(
            model=str(cfg.get("model", _DEFAULT_MODEL)),
            host=str(cfg.get("host", _DEFAULT_HOST)),
            port=int(cfg.get("port", _DEFAULT_PORT)),
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ArtemisError(Exception):
    """Base exception for Artemis-specific (non-tracker) failures."""


class LLMError(ArtemisError):
    """Raised when the LLM returns an empty or invalid response."""


# ---------------------------------------------------------------------------
# Milestones
# ---------------------------------------------------------------------------

# Streak lengths worth calling out on complete_habit, beyond the
# "new best streak" callout itself.
_MILESTONE_DAYS: tuple[int, ...] = (7, 14, 30, 60, 100, 365)

# ---------------------------------------------------------------------------
# At-risk goals
# ---------------------------------------------------------------------------

_AT_RISK_DAYS_THRESHOLD = 7
_AT_RISK_PROGRESS_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_MOTIVATION_PROMPT = """\
You are Artemis, an encouraging productivity coach.
Here is the user's current habit and goal data:

HABITS:
{habit_detail}

GOALS:
{goal_detail}

AT-RISK / OVERDUE GOALS:
{at_risk_detail}

Write a short (2-3 sentence) motivational pep talk grounded in this real
data. Reference something specific — a streak, a goal, a milestone. Be
warm and direct. No bullet points."""

_COACHING_PROMPT = """\
You are Artemis, a productivity coach.
Here is the user's productivity data:

HABITS ({habit_count} tracked, avg streak {avg_streak} day(s)):
{habit_detail}

GOALS ({goal_count} active):
{goal_detail}

AT-RISK / OVERDUE GOALS:
{at_risk_detail}

Write a short (2-3 sentence) coaching paragraph: what's going well, what
needs attention, and one concrete next action. Be direct and warm. No
bullet points."""


class ArtemisEngine(BaseModule):
    name = "artemis"

    _INTENTS: frozenset[str] = frozenset({
        "add_habit", "complete_habit", "list_habits", "remove_habit",
        "add_goal", "update_goal", "list_goals", "get_goals",
        "remove_goal", "abandon_goal", "get_at_risk_goals",
        "productivity_summary", "get_motivation",
        "suggest_activity",
        # backlog #123, #128, #125
        "set_habit_grace", "pause_habit", "resume_habit", "weekly_habit_review",
        # backlog #122, #124, #127, #130
        "start_focus", "stop_focus", "focus_stats",
        "decompose_goal", "complete_milestone",
        "list_goal_templates", "add_goal_from_template", "list_badges",
    })

    def __init__(
        self,
        ollama_cfg: Optional[dict[str, Any]] = None,
        tracker: Optional[ArtemisTracker] = None,
        llm: Optional[Any] = None,
        habit_grace_days: int = 0,
        timezone_name: str = "UTC",
        nudges: Optional[dict] = None,
    ) -> None:
        # habit_grace_days (config: artemis.habit_grace_days) is the default grace period
        # for habits that don't set their own (#123). Ignored when a tracker is passed in.
        self.tracker = tracker or ArtemisTracker(default_grace_days=habit_grace_days, timezone_name=timezone_name)
        # Smart nudges (#129): config artemis.nudges.{enabled, lateness_minutes}.
        nudges = nudges or {}
        self._nudges_enabled = bool(nudges.get("enabled", True))
        self._nudge_lateness = max(0, int(nudges.get("lateness_minutes", 60)))
        self._focus_timer: Optional[threading.Timer] = None
        self._cfg = OllamaConfig.from_dict(ollama_cfg or {})
        self._llm_instance = llm  # HestiaLLM | None — preferred path

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        # State reads/writes go through ArtemisTracker, which can raise
        # TrackerError (e.g. StateCorruptedError, or an OSError wrapped on
        # write failure) for problems that aren't the caller's fault, on
        # top of the HabitNotFoundError/GoalNotFoundError/ValueError cases
        # handled per-intent below. Catch it here so a corrupted or
        # temporarily-unwritable state file degrades to a clear response
        # instead of bubbling up to the orchestrator's generic error.
        try:
            result = self._handle(intent, entities, context)
            if intent in _BADGE_INTENTS:
                self._announce_badges(result)
            return result
        except TrackerError:
            logger.exception("ArtemisTracker failure while handling intent %r.", intent)
            return {
                "response": "I couldn't access your habit/goal data right now. Please try again shortly.",
                "data": {},
                "confidence": 0.3,
            }

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _handle(self, intent: str, entities: dict, context: dict) -> dict:
        if intent == "add_habit":
            return self._add_habit(entities)
        if intent == "complete_habit":
            return self._complete_habit(entities)
        if intent == "list_habits":
            return self._list_habits()
        if intent == "remove_habit":
            return self._remove_habit(entities)
        if intent == "add_goal":
            return self._add_goal(entities)
        if intent == "update_goal":
            return self._update_goal(entities)
        if intent in ("list_goals", "get_goals"):
            return self._list_goals()
        if intent == "remove_goal":
            return self._remove_goal(entities)
        if intent == "abandon_goal":
            return self._abandon_goal(entities)
        if intent == "get_at_risk_goals":
            return self._at_risk_goals()
        if intent == "productivity_summary":
            return self._productivity_summary()
        if intent == "get_motivation":
            return self._motivation()
        if intent == "suggest_activity":
            return self._suggest_activity(entities)
        if intent == "set_habit_grace":
            return self._set_habit_grace(entities, context)
        if intent == "pause_habit":
            return self._pause_habit(entities, context)
        if intent == "resume_habit":
            return self._resume_habit(entities, context)
        if intent == "weekly_habit_review":
            return self._weekly_habit_review()
        if intent == "start_focus":
            return self._start_focus(entities, context)
        if intent == "stop_focus":
            return self._stop_focus()
        if intent == "focus_stats":
            return self._focus_stats()
        if intent == "decompose_goal":
            return self._decompose_goal(entities, context)
        if intent == "complete_milestone":
            return self._complete_milestone(entities, context)
        if intent == "list_goal_templates":
            return self._list_goal_templates()
        if intent == "add_goal_from_template":
            return self._add_goal_from_template(entities, context)
        if intent == "list_badges":
            return self._list_badges()
        return {"response": "Artemis can't handle that request.", "data": {}, "confidence": 0.5}

    # ------------------------------------------------------------------
    # Habits
    # ------------------------------------------------------------------

    def _add_habit(self, entities: dict) -> dict:
        name = entities.get("name", "").strip()
        if not name:
            return _ok("Please specify a habit name.")
        self.tracker.add_habit(name)
        return _ok(f"Added habit '{name}'.")

    def _complete_habit(self, entities: dict) -> dict:
        name = entities.get("name", "").strip()
        if not name:
            return _ok("Please specify a habit name.")
        try:
            result = self.tracker.complete_habit(name)
        except HabitNotFoundError:
            return _ok(
                f"You haven't added '{name}' as a habit yet. "
                f"Say 'add habit {name}' first.",
                confidence=0.6,
            )

        streak = result["streak"]
        if result["already_done"]:
            response = f"'{name}' is already marked complete for today. Current streak: {streak} days."
        else:
            response = f"Marked '{name}' complete. Current streak: {streak} days."
            if result.get("resumed"):
                response = f"Welcome back: '{name}' is no longer paused. " + response
            if result.get("grace_used"):
                n = result["grace_used"]
                response += f" You missed {n} day{'s' if n != 1 else ''}, but your grace period kept the streak going."
            callout = _milestone_callout(streak, result["is_new_best"])
            if callout:
                response += f" {callout}"
        return _ok(response, data=result)

    def _remove_habit(self, entities: dict) -> dict:
        name = entities.get("name", "").strip()
        if not name:
            return _ok("Please specify a habit name.")
        try:
            self.tracker.remove_habit(name)
        except HabitNotFoundError:
            return _ok(f"You don't have a habit called '{name}'.", confidence=0.6)
        return _ok(f"Removed habit '{name}'.")

    def _list_habits(self) -> dict:
        habits = self.tracker.get_habits()
        today = date.today()
        if not habits:
            return _ok("You have no habits tracked.")
        default_grace = self.tracker.default_grace_days
        parts = [
            f"{h} ({v.streak}🔥, {v.consistency_pct(today):.0f}% consistent"
            + (", paused" if v.paused_on(today) else "")
            + (f", {v._grace(default_grace)}-day grace" if v._grace(default_grace) else "")
            + ")"
            for h, v in habits.items()
        ]
        response = f"You have {len(habits)} habits: " + ", ".join(parts)
        data = {
            name: {**v.to_dict(), "consistency_pct": v.consistency_pct(today)}
            for name, v in habits.items()
        }
        return _ok(response, data=data)

    def _announce_badges(self, result: dict) -> None:
        """Append any badge earned by this action to its reply (#127). Never fails the action."""
        try:
            new = self.tracker.award_badges()
        except Exception:
            logger.exception("Badge check failed.")
            return
        if new:
            result["response"] = f"{result['response']} {_badge_text(new)}"
            result.setdefault("data", {})["new_badges"] = new

    # ------------------------------------------------------------------
    # Focus sessions / Pomodoro (#122)
    # ------------------------------------------------------------------

    def _now(self) -> datetime:
        return self.tracker.now()

    def _start_focus(self, entities: dict, context: dict) -> dict:
        raw = str(entities.get("raw_query") or context.get("raw_query") or "")
        minutes = _parse_count(entities.get("minutes", entities.get("duration")), raw, unit="min(?:ute)?")
        if minutes is None and re.search(r"\bhour\b", raw, re.I):
            minutes = 60
        minutes = minutes or extras.DEFAULT_FOCUS_MINUTES
        task = str(entities.get("task") or entities.get("name") or "").strip()
        if not task:
            m = re.search(r"\b(?:on|for working on|to work on)\s+(?!\d)(.+?)(?:\s+for\s+\d.*)?$", raw, re.I)
            task = m.group(1).strip() if m else ""
        now = self._now()

        active = self.tracker.active_focus(now)
        if active:
            left = active["remaining"]
            if left is not None and left > 0:
                return _ok(f"A focus session is already running: {_fmt_minutes(left)} left.",
                           data={"active": active}, confidence=0.7)
            self._finish_focus(now)          # it ran out while nobody was listening; log it first
        try:
            session = self.tracker.start_focus(minutes, task, now)
        except ValueError as exc:
            return _ok(str(exc), confidence=0.5)
        self._schedule_focus_timer(session["start"], minutes)
        on = f" on {task}" if task else ""
        return _ok(f"Focus session started{on}: {minutes} minutes. I'll tell you when it's time for a break.",
                   data={"session": session})

    def _finish_focus(self, now: datetime) -> Optional[dict]:
        self._cancel_focus_timer()
        return self.tracker.stop_focus(now)

    def _stop_focus(self) -> dict:
        record = self._finish_focus(self._now())
        if record is None:
            return _ok("No focus session is running.", confidence=0.6)
        if record["completed"]:
            return _ok(f"Session done: {record['minutes']} focused minutes logged. " + self._break_advice(),
                       data={"session": record})
        return _ok(f"Stopped early. {record['minutes']} focused minute{'s' if record['minutes'] != 1 else ''} logged.",
                   data={"session": record})

    def _break_advice(self) -> str:
        n = self.tracker.completed_focus_today(self._now())
        if n and n % 4 == 0:
            return f"That's {n} today, so take a longer break (about 15 minutes)."
        return f"Take a {extras.DEFAULT_BREAK_MINUTES}-minute break."

    def _focus_stats(self) -> dict:
        now = self._now()
        stats = self.tracker.focus_stats(now)
        active = self.tracker.active_focus(now)
        if not stats["total_sessions"] and not active:
            return _ok("No focus sessions yet. Say 'start a focus session' to begin one.", data=stats, confidence=0.7)
        parts = []
        if active and active["remaining"] is not None and active["remaining"] > 0:
            parts.append(f"A session is running: {_fmt_minutes(active['remaining'])} left.")
        parts.append(
            f"Last 7 days: {_fmt_minutes(stats['minutes'])} of focus across {stats['sessions']} "
            f"session{'s' if stats['sessions'] != 1 else ''} ({stats['completed']} finished)."
        )
        if stats["tasks"]:
            parts.append("Worked on: " + ", ".join(stats["tasks"]) + ".")
        parts.append(f"All time: {_fmt_minutes(stats['total_minutes'])}.")
        return _ok(" ".join(parts), data=stats)

    def _schedule_focus_timer(self, start_iso: str, minutes: int) -> None:
        self._cancel_focus_timer()
        timer = threading.Timer(minutes * 60, self._on_focus_timer, args=(start_iso,))
        timer.daemon = True
        self._focus_timer = timer
        timer.start()

    def _cancel_focus_timer(self) -> None:
        timer, self._focus_timer = self._focus_timer, None
        if timer is not None:
            timer.cancel()

    def _on_focus_timer(self, start_iso: str) -> None:
        """Timer thread: close the session it was set for and announce the break."""
        try:
            active = self.tracker.active_focus(self._now())
            if not active or active.get("start") != start_iso:
                return                      # stopped, or replaced by a newer session
            record = self.tracker.stop_focus(self._now())
            self._focus_timer = None
            if record:
                text = f"Focus session finished: {record['minutes']} minutes. " + self._break_advice()
                new = self.tracker.award_badges()
                if new:
                    text += " " + _badge_text(new)
                bus.emit("speak", {"text": text})
        except Exception:
            logger.exception("Focus timer callback failed.")

    # ------------------------------------------------------------------
    # Goal milestones (#124) and templates (#130)
    # ------------------------------------------------------------------

    def _pick_goal(self, entities: dict, context: dict, need_milestones: bool = False) -> tuple[Optional[str], Optional[dict]]:
        goals = {n: g for n, g in self.tracker.get_goals().items() if g.status == "active"}
        if need_milestones:
            goals = {n: g for n, g in goals.items() if g.milestones}
        if not goals:
            return None, _ok("You have no goals with steps yet. Try 'break down <goal> into steps'."
                             if need_milestones else "You have no active goals yet.", confidence=0.6)
        asked = str(entities.get("name") or entities.get("goal") or "").strip()
        raw = str(entities.get("raw_query") or context.get("raw_query") or "")
        name = _find_habit(goals, asked) or _find_habit(goals, raw)
        if name:
            return name, None
        if len(goals) == 1 and not asked:
            return next(iter(goals)), None
        if asked:
            return None, _ok(f"You don't have an active goal called '{asked}'.", confidence=0.6)
        return None, _ok("Which goal? You have: " + ", ".join(goals) + ".", confidence=0.5)

    def _decompose_goal(self, entities: dict, context: dict) -> dict:
        asked = str(entities.get("name") or entities.get("goal") or "").strip()
        raw = str(entities.get("raw_query") or context.get("raw_query") or "")
        goals = self.tracker.get_goals()
        name = _find_habit(goals, asked) or _find_habit(goals, raw) or asked
        if not name:
            return _ok("Which goal should I break down?", confidence=0.5)
        try:
            steps = _parse_steps(self._llm(_DECOMPOSE_PROMPT.format(goal=name)))
        except LLMError:
            steps = []
        if len(steps) < 3:
            return _ok(f"I couldn't break '{name}' down just now (the language model gave no usable steps). "
                       "Try again in a moment, or start from a template: 'show goal templates'.", confidence=0.4)
        try:
            if name not in goals:
                self.tracker.add_goal(name)
            self.tracker.set_goal_milestones(name, steps)
        except ValueError as exc:
            return _ok(str(exc), confidence=0.5)
        listing = "; ".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        return _ok(f"Here's a plan for '{name}': {listing}. Say 'finished step 1 of {name}' as you go.",
                   data={"goal": name, "milestones": steps})

    def _complete_milestone(self, entities: dict, context: dict) -> dict:
        name, reply = self._pick_goal(entities, context, need_milestones=True)
        if reply:
            return reply
        raw = str(entities.get("raw_query") or context.get("raw_query") or "")
        ref = entities.get("milestone") or entities.get("step")
        if ref is None:
            m = re.search(r"\b(?:step|milestone)\s*#?\s*(\d+)\b", raw, re.I)
            ref = m.group(1) if m else None
        if ref is None:
            goal = self.tracker.get_goal(name)
            nxt = next((m["title"] for m in goal.milestones if not m["done"]), None)
            if nxt is None:
                return _ok(f"Every step of '{name}' is already done.", confidence=0.6)
            ref = nxt                                   # "next step" when none is named
        result = self.tracker.complete_goal_milestone(name, ref)
        if result is None:
            return _ok(f"I couldn't find a step matching '{ref}' in '{name}'.", confidence=0.5)
        msg = f"Ticked '{result['title']}' ({result['done']}/{result['total']}, {int(result['progress'] * 100)}%)."
        if result["goal_completed"]:
            msg += f" '{name}' is complete. Well done!"
        return _ok(msg, data=result)

    def _list_goal_templates(self) -> dict:
        lines = [f"{t['title']} ({len(t['milestones'])} steps, ~{t['days']} days)" for t in GOAL_TEMPLATES.values()]
        return _ok("Goal templates: " + "; ".join(lines) + ". Say 'start the Run a 5K template' to use one.",
                   data={"templates": list(GOAL_TEMPLATES)})

    def _add_goal_from_template(self, entities: dict, context: dict) -> dict:
        raw = str(entities.get("raw_query") or context.get("raw_query") or "")
        key = find_template(str(entities.get("template") or "")) or find_template(str(entities.get("name") or "")) \
            or find_template(raw)
        if key is None:
            return _ok("Which template? Options: " + ", ".join(template_names()) + ".", confidence=0.5)
        t = GOAL_TEMPLATES[key]
        name = str(entities.get("goal_name") or t["title"]).strip()
        if name in self.tracker.get_goals():
            return _ok(f"You already have a goal called '{name}'.", confidence=0.6)
        due = (self._now().date() + timedelta(days=t["days"])).isoformat()
        try:
            self.tracker.add_goal(name, due_date=due, priority=t["priority"], milestones=t["milestones"])
        except ValueError as exc:
            return _ok(str(exc), confidence=0.5)
        steps = "; ".join(f"{i}. {s}" for i, s in enumerate(t["milestones"], 1))
        return _ok(f"Added '{name}' (due {due}, {t['priority']} priority) with {len(t['milestones'])} steps: {steps}.",
                   data={"goal": name, "template": key, "due_date": due})

    # ------------------------------------------------------------------
    # Badges (#127)
    # ------------------------------------------------------------------

    def _list_badges(self) -> dict:
        self.tracker.award_badges()
        earned = self.tracker.earned_badges()
        if not earned:
            first = ", ".join(f"{BADGES[b][0]} ({BADGES[b][1]})" for b in list(BADGES)[:3])
            return _ok(f"No badges yet. First ones to aim for: {first}.", data={"earned": {}})
        got = ", ".join(f"{BADGES[b][0]} ({d})" for b, d in earned.items() if b in BADGES)
        locked = [BADGES[b] for b in BADGES if b not in earned][:3]
        more = " Next: " + ", ".join(f"{lbl} ({desc})" for lbl, desc in locked) + "." if locked else ""
        return _ok(f"Badges earned: {got}.{more}", data={"earned": earned})

    # ------------------------------------------------------------------
    # Smart nudges (#129)
    # ------------------------------------------------------------------

    def check_habit_nudges(self, now: Optional[datetime] = None) -> Optional[str]:
        """
        Heartbeat hook. A gentle reminder for one habit that's usually logged by now and hasn't
        been today, at most one per habit per day. None when nothing is due or nudges are off.
        """
        if not self._nudges_enabled:
            return None
        due = self.tracker.habits_due_for_nudge(now, self._nudge_lateness)
        if not due:
            return None
        pick = max(due, key=lambda d: d["streak"])        # most at stake first
        self.tracker.mark_nudged(pick["name"])
        at = extras.fmt_minute(pick["typical_minute"])
        stake = f" Your {pick['streak']}-day streak is on the line." if pick["streak"] >= 3 else ""
        return f"Gentle nudge: you usually log '{pick['name']}' by around {at}, and it's not done yet today.{stake}"

    # ------------------------------------------------------------------
    # Grace periods (#123), pauses (#128), weekly review (#125)
    # ------------------------------------------------------------------

    def _pick_habit(self, entities: dict, context: dict) -> tuple[Optional[str], Optional[dict]]:
        """(habit name, None) or (None, a ready-made reply) when it can't tell which habit."""
        habits = self.tracker.get_habits()
        if not habits:
            return None, _ok("You have no habits tracked yet.", confidence=0.6)
        asked = str(entities.get("name") or entities.get("habit") or "").strip()
        raw = str(entities.get("raw_query") or context.get("raw_query") or "")
        name = _find_habit(habits, asked) or _find_habit(habits, raw)
        if name:
            return name, None
        if asked:
            return None, _ok(f"You don't have a habit called '{asked}'.", confidence=0.6)
        return None, _ok("Which habit? You have: " + ", ".join(habits) + ".", confidence=0.5)

    def _set_habit_grace(self, entities: dict, context: dict) -> dict:
        name, reply = self._pick_habit(entities, context)
        if reply:
            return reply
        raw = str(entities.get("raw_query") or context.get("raw_query") or "")
        days = _parse_count(entities.get("days", entities.get("grace_days")), raw, unit="day")
        if days is None and re.search(r"\b(?:no|off|strict|remove|disable|without)\b", raw, re.I):
            days = 0
        if days is None:
            return _ok(f"How many missed days should '{name}' survive? (0 to {MAX_GRACE_DAYS})", confidence=0.5)
        try:
            self.tracker.set_habit_grace(name, days)
        except ValueError as exc:
            return _ok(str(exc), confidence=0.5)
        if days == 0:
            return _ok(f"'{name}' is strict again: any missed day resets its streak.", data={"grace_days": 0})
        return _ok(
            f"'{name}' now keeps its streak through up to {days} missed day{'s' if days != 1 else ''} in a row.",
            data={"grace_days": days},
        )

    def _pause_habit(self, entities: dict, context: dict) -> dict:
        name, reply = self._pick_habit(entities, context)
        if reply:
            return reply
        raw = str(entities.get("raw_query") or context.get("raw_query") or "")
        days = _parse_count(entities.get("days"), raw, unit="day", allow_weeks=True)
        try:
            result = self.tracker.pause_habit(name, days)
        except ValueError as exc:
            return _ok(str(exc), confidence=0.5)
        if result["already_paused"]:
            return _ok(f"'{name}' is already paused.", data=result)
        streak = result["streak"]
        keep = f" Your {streak}-day streak is safe while it's paused." if streak else ""
        if result["until"]:
            return _ok(f"Paused '{name}' until {result['until']}.{keep}", data=result)
        return _ok(f"Paused '{name}' until you resume it. Say 'resume {name}' when you're back.{keep}", data=result)

    def _resume_habit(self, entities: dict, context: dict) -> dict:
        name, reply = self._pick_habit(entities, context)
        if reply:
            return reply
        result = self.tracker.resume_habit(name)
        if not result["was_paused"]:
            return _ok(f"'{name}' isn't paused.", data=result)
        return _ok(f"Resumed '{name}'. Its streak is {result['streak']} day{'s' if result['streak'] != 1 else ''}.", data=result)

    def _weekly_habit_review(self) -> dict:
        review = self.tracker.weekly_review()
        rows = review["habits"]
        if not rows:
            return _ok("You have no habits tracked.", data=review)
        if review["overall_pct"] is None:
            return _ok("I don't have enough habit history for a weekly review yet. Keep logging and ask again.", data=review)
        parts = [f"Over the last 7 days you were {review['overall_pct']}% consistent."]
        scored = [r for r in rows if r["pct"] is not None]
        parts.append("; ".join(
            f"{r['name']} {r['done']}/{r['possible']}" + (" (paused)" if r["paused"] else "") for r in scored[:6]) + ".")
        if review["weakest"]:
            weak = next(r for r in scored if r["name"] == review["weakest"])
            if weak["pct"] < 100:
                parts.append(f"Needs attention: {weak['name']}.")
        moves = [f"{r['name']} {'up' if r['pct'] > r['prev_pct'] else 'down'} from {r['prev_pct']}% to {r['pct']}%"
                 for r in scored if r["prev_pct"] is not None and abs(r["pct"] - r["prev_pct"]) >= 15]
        if moves:
            parts.append("Compared with the week before: " + ", ".join(moves[:3]) + ".")
        unknown = len(rows) - len(scored)
        if unknown:
            parts.append(f"{unknown} habit{'s' if unknown != 1 else ''} don't have enough history yet.")
        return _ok(" ".join(parts), data=review)

    # ------------------------------------------------------------------
    # Goals
    # ------------------------------------------------------------------

    def _add_goal(self, entities: dict) -> dict:
        name = entities.get("name", "").strip()
        if not name:
            return _ok("Please specify a goal name.")

        due_date = entities.get("due_date") or None
        priority = entities.get("priority") or None
        try:
            self.tracker.add_goal(name, due_date=due_date, priority=priority)
        except ValueError as exc:
            return _ok(str(exc), confidence=0.5)

        extras = [p for p in (f"due {due_date}" if due_date else None,
                               f"{priority} priority" if priority else None) if p]
        suffix = f" ({', '.join(extras)})" if extras else ""
        return _ok(f"Goal '{name}' added{suffix}.")

    def _update_goal(self, entities: dict) -> dict:
        name = entities.get("name", "").strip()
        progress = entities.get("progress")
        due_date = entities.get("due_date")
        priority = entities.get("priority")

        if not name:
            return _ok("Please specify a goal name and progress.")
        if progress is None and due_date is None and priority is None:
            return _ok("Please specify a goal name and progress.")

        updates: list[str] = []

        if progress is not None:
            try:
                progress = float(progress)
            except (TypeError, ValueError):
                return _ok("Please provide a numeric progress value.")
            fraction = progress / 100 if progress > 1 else progress
            try:
                self.tracker.update_goal(name, fraction)
            except GoalNotFoundError:
                return _ok(
                    f"You haven't added '{name}' as a goal yet. "
                    f"Say 'add goal {name}' first.",
                    confidence=0.6,
                )
            except ValueError:
                return _ok("Progress must be between 0 and 100 percent.", confidence=0.5)
            pct = int(max(0.0, min(1.0, fraction)) * 100)
            updates.append(f"now {pct}% complete")

        if due_date is not None or priority is not None:
            try:
                self.tracker.set_goal_metadata(name, due_date=due_date, priority=priority)
            except GoalNotFoundError:
                return _ok(
                    f"You haven't added '{name}' as a goal yet. "
                    f"Say 'add goal {name}' first.",
                    confidence=0.6,
                )
            except ValueError as exc:
                return _ok(str(exc), confidence=0.5)
            if due_date:
                updates.append(f"due {due_date}")
            if priority:
                updates.append(f"{priority} priority")

        return _ok(f"Goal '{name}' updated: " + ", ".join(updates) + ".")

    def _list_goals(self) -> dict:
        goals = self.tracker.get_goals()
        active = {g: v for g, v in goals.items() if v.status == "active"}
        if not active:
            response = "You have no active goals."
        else:
            parts = [f"{g} ({int(v.progress*100)}%)" for g, v in active.items()]
            response = "Active goals: " + ", ".join(parts)
        data = {name: v.to_dict() for name, v in goals.items()}
        return _ok(response, data=data)

    def _remove_goal(self, entities: dict) -> dict:
        name = entities.get("name", "").strip()
        if not name:
            return _ok("Please specify a goal name.")
        try:
            self.tracker.remove_goal(name)
        except GoalNotFoundError:
            return _ok(f"You don't have a goal called '{name}'.", confidence=0.6)
        return _ok(f"Removed goal '{name}'.")

    def _abandon_goal(self, entities: dict) -> dict:
        name = entities.get("name", "").strip()
        if not name:
            return _ok("Please specify a goal name.")
        try:
            self.tracker.set_goal_status(name, "abandoned")
        except GoalNotFoundError:
            return _ok(f"You don't have a goal called '{name}'.", confidence=0.6)
        return _ok(f"Marked goal '{name}' as abandoned.")

    def _at_risk_goals(self) -> dict:
        today = date.today()
        at_risk = self.tracker.get_at_risk_goals(
            days_threshold=_AT_RISK_DAYS_THRESHOLD,
            progress_threshold=_AT_RISK_PROGRESS_THRESHOLD,
            today=today,
        )
        if not at_risk:
            return _ok("No goals are at risk — nothing overdue or close to deadline with low progress.")

        parts = []
        for name, g in at_risk.items():
            days_left = g.days_until_due(today)
            when = "overdue" if days_left is not None and days_left < 0 else f"due in {days_left}d"
            parts.append(f"{name} ({when}, {int(g.progress*100)}%)")
        response = f"{len(at_risk)} goal(s) at risk: " + ", ".join(parts)
        data = {name: g.to_dict() for name, g in at_risk.items()}
        return _ok(response, data=data, confidence=0.9)

    # ------------------------------------------------------------------
    # Productivity summary / motivation (AI voice)
    # ------------------------------------------------------------------

    def _productivity_summary(self) -> dict:
        habits = self.tracker.get_habits()
        goals = self.tracker.get_goals()
        today = date.today()
        total_habits = len(habits)
        avg_streak = round(sum(h.streak for h in habits.values()) / total_habits, 1) if total_habits else 0.0
        goals_progress = {g: v.progress for g, v in goals.items()}
        insights = self.analyze()
        at_risk = self.tracker.get_at_risk_goals(
            days_threshold=_AT_RISK_DAYS_THRESHOLD,
            progress_threshold=_AT_RISK_PROGRESS_THRESHOLD,
            today=today,
        )
        overdue = {
            name: g for name, g in at_risk.items()
            if (g.days_until_due(today) or 0) < 0
        }

        response = (
            f"Avg streak: {insights['avg_streak']} days. "
            f"Strong: {', '.join(insights['strong_habits']) or 'none'}. "
            f"Needs focus: {', '.join(insights['weak_habits']) or 'none'}."
            f" Goals progress: " + ", ".join(f"{g} ({int(p*100)}%)" for g, p in goals_progress.items())
        )
        if at_risk:
            response += (
                f" ⚠️ {len(at_risk)} goal(s) at risk"
                f" ({len(overdue)} overdue): " + ", ".join(at_risk.keys()) + "."
            )

        active_goal_count = sum(1 for g in goals.values() if g.status == "active")
        habit_detail = _format_habits(habits, today)
        goal_detail = _format_goals(goals)
        at_risk_detail = _format_at_risk(at_risk, today)

        try:
            coaching = self._llm(
                _COACHING_PROMPT.format(
                    habit_count=total_habits,
                    avg_streak=insights["avg_streak"],
                    habit_detail=habit_detail,
                    goal_count=active_goal_count,
                    goal_detail=goal_detail,
                    at_risk_detail=at_risk_detail,
                )
            )
        except LLMError:
            logger.warning("_productivity_summary: LLM unavailable; using static coaching line.")
            coaching = _static_coaching(insights, at_risk)

        focus = self.tracker.focus_stats(self._now())
        if focus["minutes"]:
            response += (f" Focus this week: {_fmt_minutes(focus['minutes'])} "
                         f"over {focus['sessions']} session{'s' if focus['sessions'] != 1 else ''}.")
        response = f"{response}\n\n{coaching}"

        # Holiday-aware framing: a quiet day that happens to be a public
        # holiday shouldn't read as a habit failure. Best-effort only —
        # Nager.Date is free/keyless but this must never block the summary
        # if it's unreachable.
        holiday_name: Optional[str] = None
        try:
            holiday_name = _fa_is_public_holiday(today.isoformat())
        except FreeAPIError:
            logger.debug("_productivity_summary: holiday check unavailable.", exc_info=True)
        if holiday_name:
            response += f"\n\n(Today is {holiday_name} — factor that into today's numbers.)"

        data = {
            "avg_streak": avg_streak,
            "habits": {name: v.to_dict() for name, v in habits.items()},
            "goals": {name: v.to_dict() for name, v in goals.items()},
            "at_risk_goals": {name: g.to_dict() for name, g in at_risk.items()},
            "overdue_goals": list(overdue.keys()),
            "holiday_today": holiday_name,
            "focus": focus,
        }
        return _ok(response, data=data, confidence=0.8)

    def _suggest_activity(self, entities: dict) -> dict:
        """
        Suggest a concrete activity to break stagnation, via the Bored
        API (free, no key; falls back to a small static list if the
        remote service is unreachable — see core/free_apis.py). New
        intent: `suggest_activity`.
        """
        activity_type = (entities.get("type") or entities.get("category") or "").strip() or None
        suggestion = _fa_suggest_activity(activity_type)
        activity = suggestion.get("activity", "Take a short break and stretch.")
        kind = suggestion.get("type")
        response = f"Here's an idea: {activity}"
        if kind:
            response += f" ({kind})"
        return _ok(response, data={"suggestion": suggestion}, confidence=0.75)

    def _motivation(self) -> dict:
        habits = self.tracker.get_habits()
        goals = self.tracker.get_goals()
        today = date.today()
        at_risk = self.tracker.get_at_risk_goals(
            days_threshold=_AT_RISK_DAYS_THRESHOLD,
            progress_threshold=_AT_RISK_PROGRESS_THRESHOLD,
            today=today,
        )

        if not habits and not goals:
            return _ok(
                "You don't have any habits or goals tracked yet — add one and "
                "I'll help you stay on track.",
                confidence=0.7,
            )

        habit_detail = _format_habits(habits, today)
        goal_detail = _format_goals(goals)
        at_risk_detail = _format_at_risk(at_risk, today)

        try:
            pep_talk = self._llm(
                _MOTIVATION_PROMPT.format(
                    habit_detail=habit_detail,
                    goal_detail=goal_detail,
                    at_risk_detail=at_risk_detail,
                )
            )
        except LLMError:
            logger.warning("_motivation: LLM unavailable; using static fallback.")
            pep_talk = self.suggest_next_action()

        return _ok(pep_talk, data={"habits": list(habits.keys()), "goals": list(goals.keys())}, confidence=0.85)

    # ------------------------------------------------------------------
    # Private – LLM helper
    # ------------------------------------------------------------------

    def _llm(self, prompt: str) -> str:
        """
        Call the LLM and return a non-empty stripped response.

        Raises
        ------
        LLMError
            If the LLM returns an empty string.
        """
        if self._llm_instance is not None:
            result = self._llm_instance.generate(prompt)
        else:
            result = generate(
                prompt,
                model=self._cfg.model,
                host=self._cfg.host,
                port=self._cfg.port,
            )
        if not result or not result.strip():
            raise LLMError("LLM returned an empty response.")
        return result.strip()

    # ------------------------------------------------------------------
    # Cross-module context / analysis
    # ------------------------------------------------------------------

    def get_context(self) -> dict:
        """Expose current habit/goal state for Hecate and secondary enrichment."""
        try:
            habits = self.tracker.get_habits()
            goals = self.tracker.get_goals()
            return {
                "habit_count": len(habits),
                "active_goals": [k for k, v in goals.items() if v.status == "active"],
                "avg_streak": self.analyze().get("avg_streak", 0),
                "at_risk_goal_count": len(self.tracker.get_at_risk_goals(
                    days_threshold=_AT_RISK_DAYS_THRESHOLD,
                    progress_threshold=_AT_RISK_PROGRESS_THRESHOLD,
                )),
            }
        except Exception:
            return {}

    def analyze(self):
        habits = self.tracker.get_habits()
        goals = self.tracker.get_goals()

        insights = {
            "weak_habits": [],
            "strong_habits": [],
            "avg_streak": 0,
        }

        if habits:
            avg = sum(h.streak for h in habits.values()) / len(habits)
            insights["avg_streak"] = round(avg, 1)

            for name, h in habits.items():
                if h.streak < avg:
                    insights["weak_habits"].append(name)
                else:
                    insights["strong_habits"].append(name)

        insights["stalled_goals"] = [
            name for name, g in goals.items()
            if g.status == "active" and g.progress == 0
        ]

        return insights

    def suggest_next_action(self):
        insights = self.analyze()

        if insights["weak_habits"]:
            return f"You should focus on '{insights['weak_habits'][0]}' next."

        return "You're on track. Continue your habits."


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_BADGE_INTENTS = frozenset({"complete_habit", "update_goal", "stop_focus", "complete_milestone",
                           "add_goal_from_template"})

_DECOMPOSE_PROMPT = (
    "Break this goal into 4 to 7 concrete, ordered milestones a person can tick off. "
    "Each is a short imperative phrase under 12 words. "
    "Reply with ONLY a JSON array of strings, nothing else.\n\nGoal: {goal}"
)


def _parse_steps(text: str) -> list[str]:
    """Milestone titles from an LLM reply: a JSON array, else numbered/bulleted lines. 3 to 8 kept."""
    import json
    body = (text or "").strip()
    steps: list[str] = []
    m = re.search(r"\[.*\]", body, re.S)
    if m:
        try:
            steps = [str(s).strip() for s in json.loads(m.group(0)) if str(s).strip()]
        except (ValueError, TypeError):
            steps = []
    if not steps:
        for line in body.splitlines():
            line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip(" *`")
            if line and len(line) <= 120:
                steps.append(line)
    return steps[:8]


def _fmt_minutes(minutes: float) -> str:
    total = max(0, int(round(minutes)))
    if total < 60:
        return f"{total} min"
    h, m = divmod(total, 60)
    return f"{h}h {m:02d}m" if m else f"{h}h"


def _badge_text(ids: list[str]) -> str:
    names = ", ".join(f"{BADGES[b][0]} ({BADGES[b][1]})" for b in ids if b in BADGES)
    return f"\U0001F3C5 New badge{'s' if len(ids) != 1 else ''}: {names}!"


_WORD_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fortnight": 14}


def _find_habit(habits: dict, text: str) -> Optional[str]:
    """The habit *text* refers to: its name inside the text (longest wins), else one shared word."""
    low = (text or "").lower().strip()
    if not low:
        return None
    for name in sorted(habits, key=len, reverse=True):
        if name.lower() in low:
            return name
    skip = {"habit", "habits", "pause", "resume", "unpause", "grace", "period", "days", "week", "weeks",
            "give", "turn", "back", "this", "that", "with", "from", "have"}
    words = {_stem(w) for w in re.findall(r"[a-z]{3,}", low) if w not in skip}
    hits = [n for n in habits if words & {_stem(w) for w in re.findall(r"[a-z]{3,}", n.lower())}]
    return hits[0] if len(hits) == 1 else None


def _stem(word: str) -> str:
    """Crude stem so "running" matches "run" and "meditation" matches "meditate"."""
    for suffix in ("ation", "ing", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word[:-1] if len(word) > 3 and word[-1] == word[-2] else word


def _parse_count(value: Any, raw: str, unit: str, allow_weeks: bool = False) -> Optional[int]:
    """A number of days from an entity (3, "3", "3 days") or from the raw request ("for two weeks")."""
    if value is not None and not isinstance(value, bool):
        m = re.search(r"\d+", str(value))
        if m:
            return int(m.group(0))
    num = r"(\d+|" + "|".join(_WORD_NUMBERS) + ")"
    m = re.search(rf"\b{num}\s*[- ]?\s*(?:{unit}s?)\b", raw or "", re.I)
    if m:
        tok = m.group(1).lower()
        return int(tok) if tok.isdigit() else _WORD_NUMBERS[tok]
    if allow_weeks:
        m = re.search(rf"\b{num}\s*[- ]?\s*weeks?\b", raw or "", re.I)
        if m:
            tok = m.group(1).lower()
            return (int(tok) if tok.isdigit() else _WORD_NUMBERS[tok]) * 7
        if re.search(r"\bfortnight\b", raw or "", re.I):
            return 14
    return None


def _ok(response: str, data: Optional[dict[str, Any]] = None, confidence: float = 0.9) -> dict[str, Any]:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _milestone_callout(streak: int, is_new_best: bool) -> str:
    """
    Build a short callout string for complete_habit — a milestone-day
    marker, a new-best-streak marker, or both. Empty string if neither
    applies. "New best" is only surfaced past day 1, since every fresh
    habit trivially sets a "best" of 1.
    """
    parts = []
    if streak in _MILESTONE_DAYS:
        parts.append(f"🎉 {streak}-day milestone!")
    if is_new_best and streak > 1:
        parts.append("🔥 New best streak!")
    return " ".join(parts)


def _format_habits(habits: dict[str, Habit], today: date) -> str:
    if not habits:
        return "  None tracked"
    return "\n".join(
        f"  {name} — streak {h.streak}d (best {h.best_streak}d), "
        f"{h.total_completions} total, {h.consistency_pct(today):.0f}% consistent"
        for name, h in habits.items()
    )


def _format_goals(goals: dict[str, Goal]) -> str:
    active = {g: v for g, v in goals.items() if v.status == "active"}
    if not active:
        return "  None active"
    lines = []
    for name, g in active.items():
        extras = [p for p in (
            f"due {g.due_date}" if g.due_date else None,
            f"{g.priority} priority" if g.priority else None,
        ) if p]
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"  {name} — {int(g.progress*100)}%{suffix}")
    return "\n".join(lines)


def _format_at_risk(at_risk: dict[str, Goal], today: date) -> str:
    if not at_risk:
        return "  None"
    lines = []
    for name, g in at_risk.items():
        days_left = g.days_until_due(today)
        when = "overdue" if days_left is not None and days_left < 0 else f"due in {days_left}d"
        lines.append(f"  {name} — {when}, {int(g.progress*100)}%")
    return "\n".join(lines)


def _static_coaching(insights: dict, at_risk: dict[str, Goal]) -> str:
    """Deterministic fallback for the AI coaching paragraph when the LLM is unavailable."""
    bits = [f"Avg streak is {insights['avg_streak']} days."]
    if insights["strong_habits"]:
        bits.append(f"Keep up '{insights['strong_habits'][0]}'.")
    if at_risk:
        bits.append(f"Focus on '{next(iter(at_risk))}' — it's at risk of missing its deadline.")
    elif insights["weak_habits"]:
        bits.append(f"Give '{insights['weak_habits'][0]}' some attention next.")
    else:
        bits.append("Everything's on track — keep going.")
    return " ".join(bits)