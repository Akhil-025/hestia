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
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from core.ollama_client import generate
from core.free_apis import (
    FreeAPIError,
    is_public_holiday as _fa_is_public_holiday,
    suggest_activity as _fa_suggest_activity,
)
from modules.base import BaseModule
from .tracker import (
    ArtemisTracker,
    Goal,
    GoalNotFoundError,
    Habit,
    HabitNotFoundError,
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
    })

    def __init__(
        self,
        ollama_cfg: Optional[dict[str, Any]] = None,
        tracker: Optional[ArtemisTracker] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self.tracker = tracker or ArtemisTracker()
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
            return self._handle(intent, entities, context)
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
        parts = [
            f"{h} ({v.streak}🔥, {v.consistency_pct(today):.0f}% consistent)"
            for h, v in habits.items()
        ]
        response = f"You have {len(habits)} habits: " + ", ".join(parts)
        data = {
            name: {**v.to_dict(), "consistency_pct": v.consistency_pct(today)}
            for name, v in habits.items()
        }
        return _ok(response, data=data)

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