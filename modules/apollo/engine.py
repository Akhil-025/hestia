"""
modules/apollo/engine.py

ApolloEngine: personal health tracking module for workouts, sleep, mood,
weight, hydration, and health goals — plus AI-generated feedback and
health summaries.

Design notes
------------
- All LLM calls are isolated in a single helper that raises a typed
  exception on failure; handlers catch it and fall back to a static
  response so the user always gets an answer.
- DB calls are wrapped per-handler so a storage failure returns a clear
  error without crashing the orchestrator.
- Validation (duration, hours, mood, weight, water, goal targets) is
  handled by pure module-level helpers that are independently testable.
- Sleep quality thresholds and comment strings are module-level constants
  so they can be audited and adjusted without touching business logic.
- Weight is always persisted in kg (ApolloDB.log_weight expects kg); the
  engine converts lb -> kg on the way in and back to the user's stated
  unit on the way out, so unit handling lives in exactly one place.
- track_sleep/log_weight/log_water share one disambiguation pattern:
  the NLU maps both "I did X" (a log) and "how's my X?" (a query) to the
  same intent with no separate query intent, so each handler checks for a
  missing value + question-like phrasing before asking the user to
  repeat data they weren't offering in the first place.
- Every public method conforms to the BaseModule response contract:
  {response: str, data: dict, confidence: float}.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from core.ollama_client import generate
from modules.base import BaseModule
from .db import ApolloDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "apollo" / "apollo.db"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "mistral"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 11434

_DEFAULT_WORKOUT_TYPE = "general"
_DEFAULT_WORKOUT_DURATION = 30      # minutes
_MIN_WORKOUT_DURATION = 1
_MAX_WORKOUT_DURATION = 600

_MIN_SLEEP_HOURS = 0.5
_MAX_SLEEP_HOURS = 24.0
_LOW_SLEEP_THRESHOLD = 6.0
_GOOD_SLEEP_THRESHOLD = 8.0

_MOOD_HISTORY_WINDOW = 5
_MOOD_TREND_WINDOW = 3              # consecutive low moods → trend warning
_HEALTH_SUMMARY_DAYS = 7
_RECENT_MOOD_CONTEXT = 3

_SLEEP_COMMENTS: dict[str, str] = {
    "low":  "That's below the recommended 7-8 hours. Try to get to bed earlier tonight.",
    "ok":   "Decent sleep. Aim for 8 hours when you can.",
    "good": "Great sleep! Consistent rest like this makes a real difference.",
}

# The NLU maps both "I slept 7 hours" (a log) and "how did I sleep last
# night?" (a query) to the single `apollo_track_sleep` intent — there's no
# separate query intent. When no hours entity comes through, this pattern
# distinguishes the two so a genuine question gets an answer instead of
# being met with the same "how many hours?" prompt that ignores what was
# actually asked.
_SLEEP_QUERY_RE = re.compile(
    r"\bhow\b|\bwhat\b|\?|\bdid i\b", flags=re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Weight tracking
# ---------------------------------------------------------------------------

_KG_PER_LB = 0.45359237
_DEFAULT_WEIGHT_UNIT = "kg"
_MIN_WEIGHT_KG = 20.0
_MAX_WEIGHT_KG = 300.0
_WEIGHT_TREND_DAYS = 30

_WEIGHT_UNIT_ALIASES: dict[str, str] = {
    "kg": "kg", "kgs": "kg", "kilogram": "kg", "kilograms": "kg", "k": "kg",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb", "b": "lb",
}

# Same log-vs-query ambiguity as sleep (see _SLEEP_QUERY_RE above), applied
# to weight: "log my weight" logs, "what's my weight trend?" asks.
_WEIGHT_QUERY_RE = re.compile(
    r"\bhow\b|\bwhat\b|\?|\btrend\b|\bprogress\b", flags=re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Hydration tracking
# ---------------------------------------------------------------------------

_ML_PER_GLASS = 250
_ML_PER_OZ = 29.5735
_MIN_WATER_ML = 10
_MAX_WATER_ML = 5000          # per single log entry
_DEFAULT_WATER_GOAL_ML = 2000

_WATER_QUERY_RE = re.compile(
    r"\bhow\b|\bwhat\b|\?|\btoday\b", flags=re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Health goals
# ---------------------------------------------------------------------------

# Canonical goal types. Each maps to a metric ApolloEngine already tracks,
# so "progress" is always computable from existing data — no goal type is
# accepted that the engine can't later report on.
_GOAL_TYPES: frozenset[str] = frozenset(
    {"workout_frequency", "sleep_hours", "weight_target", "water_ml"}
)

_GOAL_TYPE_ALIASES: dict[str, str] = {
    "workout": "workout_frequency", "workouts": "workout_frequency",
    "exercise": "workout_frequency", "workout_frequency": "workout_frequency",
    "sleep": "sleep_hours", "sleep_hours": "sleep_hours",
    "weight": "weight_target", "weight_target": "weight_target",
    "water": "water_ml", "hydration": "water_ml", "water_ml": "water_ml",
}

_GOAL_LABELS: dict[str, str] = {
    "workout_frequency": "workouts/week",
    "sleep_hours": "hours of sleep/night",
    "weight_target": "kg target weight",
    "water_ml": "ml of water/day",
}

_MIN_GOAL_TARGET = 0.1
_MAX_GOAL_TARGET = 10_000.0

# ---------------------------------------------------------------------------
# Streaks
# ---------------------------------------------------------------------------

_STREAK_LOOKBACK_DAYS = 60

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_WORKOUT_FEEDBACK_PROMPT = """\
You are Apollo, a health coach.
The user just logged a workout:
  Type    : {type_}
  Duration: {duration} minutes
  Notes   : {notes}
  Workouts this week: {count}
  Current daily streak: {streak} day(s)

Give a short (2-3 sentence) motivating response. Mention their weekly count
or streak, whichever is more impressive. Be warm and specific. No bullet
points."""

_MOOD_RESPONSE_PROMPT = """\
You are Apollo, a compassionate health assistant.
The user logged their mood as: {mood}
Recent mood history (newest first): {history}

Write a 2-3 sentence empathetic response.
If there is a negative trend (3+ low moods), gently acknowledge it and suggest one small action.
If positive, celebrate it briefly.
Be human and warm. No bullet points."""

_HEALTH_SUMMARY_PROMPT = """\
You are Apollo, a personal health coach.
Here is the user's health data for the last {days} days:

WORKOUTS ({workout_count} sessions, current streak: {streak} day(s)):
{workout_detail}

SLEEP (avg {avg_sleep} hours/night):
{sleep_detail}

MOOD LOG:
{mood_detail}

WEIGHT:
{weight_detail}

HYDRATION (today: {water_today} ml):
{water_detail}

ACTIVE GOALS:
{goal_detail}

Write a health summary with:
1. What's going well
2. What needs attention
3. One specific recommendation for next week

Under 200 words. Be direct and actionable."""

_GOAL_PROGRESS_PROMPT = """\
You are Apollo, a personal health coach.
The user asked for progress on their health goals:
{progress_detail}

Write a 2-3 sentence response. Call out the goal they're closest to missing
or furthest ahead on, and give one concrete nudge. Be warm and direct.
No bullet points."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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

class ApolloError(Exception):
    """Base exception for ApolloEngine failures."""


class LLMError(ApolloError):
    """Raised when the LLM returns an empty or invalid response."""


class StorageError(ApolloError):
    """Raised when a database operation fails."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ApolloEngine(BaseModule):
    """
    Personal health tracking module.

    Tracks workouts, sleep, and mood; generates AI-powered summaries and
    feedback via a local Ollama LLM.

    Parameters
    ----------
    ollama_cfg:
        Dict with optional keys ``model``, ``host``, ``port``.
    db_path:
        Override the default SQLite database path (useful in tests).
    """

    name = "apollo"

    _INTENTS: frozenset[str] = frozenset(
        {
            "log_workout",
            "track_sleep",
            "log_mood",
            "log_health",
            "get_health_summary",
            "log_weight",
            "log_water",
            "set_health_goal",
            "get_goal_progress",
        }
    )

    def __init__(
        self,
        ollama_cfg: Optional[dict[str, Any]] = None,
        db_path: Optional[Path] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self._cfg = OllamaConfig.from_dict(ollama_cfg or {})
        self._llm_instance = llm  # HestiaLLM | None — preferred path
        resolved = (db_path or _DB_PATH).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.db = ApolloDB(str(resolved))
        logger.info(
            "ApolloEngine ready (model=%s, db=%s).", self._cfg.model, resolved
        )

    # ------------------------------------------------------------------
    # BaseModule interface
    # ------------------------------------------------------------------

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        """
        Dispatch an intent to the appropriate handler.

        Never raises; all errors produce a graceful response dict.
        """
        try:
            return self._dispatch(intent, entities)
        except Exception:
            logger.exception(
                "ApolloEngine.handle() raised for intent=%s.", intent
            )
            return _err("Something went wrong in the health module.")

    def get_context(self) -> dict:
        """Return a lightweight health context for NLU enrichment."""
        try:
            latest_weight = self.db.latest_weight()
            return {
                "apollo_workouts_this_week": self.db.workout_count(_HEALTH_SUMMARY_DAYS),
                "apollo_avg_sleep": self.db.avg_sleep(_HEALTH_SUMMARY_DAYS),
                "apollo_recent_moods": self.db.recent_moods(_RECENT_MOOD_CONTEXT),
                "apollo_workout_streak": _compute_streak(
                    self.db.workout_dates(_STREAK_LOOKBACK_DAYS)
                ),
                "apollo_latest_weight_kg": (
                    latest_weight["weight_kg"] if latest_weight else None
                ),
                "apollo_water_today_ml": self.db.water_today(),
                "apollo_active_goals": [g["goal_type"] for g in self.db.get_all_goals()],
            }
        except Exception:
            logger.exception("get_context() failed.")
            return {}

    # ------------------------------------------------------------------
    # Private – dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, intent: str, entities: dict) -> dict:
        if intent == "log_workout":
            return self._log_workout(entities)
        if intent in ("track_sleep", "log_health"):
            return self._track_sleep(entities)
        if intent == "log_mood":
            return self._log_mood(entities)
        if intent == "get_health_summary":
            return self._health_summary()
        if intent == "log_weight":
            return self._log_weight(entities)
        if intent == "log_water":
            return self._log_water(entities)
        if intent == "set_health_goal":
            return self._set_health_goal(entities)
        if intent == "get_goal_progress":
            return self._goal_progress()
        return _err(f"Unknown intent: {intent!r}")

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
    # Private – intent handlers
    # ------------------------------------------------------------------

    def _log_workout(self, entities: dict) -> dict:
        """Validate, persist, and generate motivating feedback for a workout."""
        type_: str = (
            entities.get("type")
            or entities.get("workout_type")
            or _DEFAULT_WORKOUT_TYPE
        ).strip()

        raw_duration = entities.get("duration") or entities.get("minutes")
        duration, err = _parse_duration(raw_duration)
        if err:
            return _clarify(err)

        notes: str = (
            entities.get("notes") or entities.get("raw_query") or ""
        ).strip()

        try:
            self.db.log_workout(type_, duration, notes)
            count = self.db.workout_count(_HEALTH_SUMMARY_DAYS)
            streak = _compute_streak(self.db.workout_dates(_STREAK_LOOKBACK_DAYS))
        except Exception:
            logger.exception("_log_workout: DB operation failed.")
            return _err("I couldn't save your workout right now.")

        try:
            feedback = self._llm(
                _WORKOUT_FEEDBACK_PROMPT.format(
                    type_=type_, duration=duration, notes=notes,
                    count=count, streak=streak,
                )
            )
        except LLMError:
            logger.warning("_log_workout: LLM unavailable; using static response.")
            streak_str = f" That's a {streak}-day streak!" if streak > 1 else ""
            feedback = (
                f"Workout logged — {duration} min of {type_}. "
                f"That's {count} session(s) this week!{streak_str}"
            )

        logger.info(
            "Workout logged: type=%s duration=%d count=%d streak=%d.",
            type_, duration, count, streak,
        )
        return _ok(
            feedback,
            data={
                "type": type_, "duration": duration,
                "weekly_count": count, "streak": streak,
            },
            confidence=0.95,
        )

    def _track_sleep(self, entities: dict) -> dict:
        """
        Validate and persist a sleep entry, or — when no hours were given
        and the phrasing reads as a question ("how did I sleep last
        night?") — report the most recently logged entry instead of
        demanding new data the user isn't offering.

        ``apollo_track_sleep`` is the intent the NLU maps *both* "log 7.5
        hours of sleep" and "how did I sleep last night" to (there's no
        separate query intent), so this handler has to disambiguate the
        two itself rather than always asking for hours to log.
        """
        raw_hours = entities.get("hours") or entities.get("duration")
        if raw_hours is None:
            raw_query: str = entities.get("raw_query") or ""
            if _SLEEP_QUERY_RE.search(raw_query):
                return self._report_last_sleep()
            return _clarify("How many hours did you sleep, and how was the quality?")

        hours, err = _parse_hours(raw_hours)
        if err:
            return _clarify(err)

        quality: str = (entities.get("quality") or "").strip()
        notes: str = (entities.get("notes") or entities.get("raw_query") or "").strip()

        try:
            self.db.log_sleep(hours, quality, notes)
            avg = self.db.avg_sleep(_HEALTH_SUMMARY_DAYS)
        except Exception:
            logger.exception("_track_sleep: DB operation failed.")
            return _err("I couldn't save your sleep data right now.")

        comment = _sleep_comment(hours)
        avg_str = f" Your {_HEALTH_SUMMARY_DAYS}-day average is {avg:.1f} hours." if avg else ""

        logger.info("Sleep logged: hours=%.1f quality=%r.", hours, quality)
        return _ok(
            f"Sleep logged — {hours} hours.{avg_str} {comment}",
            data={"hours": hours, "quality": quality, "avg_7d": avg},
            confidence=0.95,
        )

    def _report_last_sleep(self) -> dict:
        """Answer a "how did I sleep" query from the most recent log entry."""
        try:
            recent = self.db.get_sleep(_HEALTH_SUMMARY_DAYS)
        except Exception:
            logger.exception("_report_last_sleep: DB read failed.")
            return _err("I couldn't check your sleep log right now.")

        if not recent:
            return _clarify(
                "I don't have any sleep logged yet. How many hours did you "
                "sleep, and how was the quality?"
            )

        latest = recent[0]
        hours = latest.get("hours")
        quality = (latest.get("quality") or "").strip()
        logged_at = (latest.get("logged_at") or "")[:10]

        quality_str = f", quality: {quality}" if quality else ""
        comment = _sleep_comment(hours) if isinstance(hours, (int, float)) else ""

        return _ok(
            f"Your last logged sleep was {hours} hours on {logged_at}{quality_str}. "
            f"{comment}".strip(),
            data={"hours": hours, "quality": quality, "logged_at": logged_at},
            confidence=0.85,
        )

    def _log_weight(self, entities: dict) -> dict:
        """
        Validate and persist a weight entry, or — when no weight was given
        and the phrasing reads as a question ("what's my weight trend?") —
        report the trend instead. Same log-vs-query pattern as sleep.
        """
        raw_weight = entities.get("weight") or entities.get("value")
        if raw_weight is None:
            raw_query: str = entities.get("raw_query") or ""
            if _WEIGHT_QUERY_RE.search(raw_query):
                return self._report_weight_trend()
            return _clarify("What's your current weight, and in kg or lb?")

        unit = _normalize_weight_unit(entities.get("unit"))
        weight_kg, err = _parse_weight(raw_weight, unit)
        if err:
            return _clarify(err)

        notes: str = (entities.get("notes") or entities.get("raw_query") or "").strip()

        try:
            self.db.log_weight(weight_kg, notes)
            previous = self.db.previous_weight()
        except Exception:
            logger.exception("_log_weight: DB operation failed.")
            return _err("I couldn't save your weight right now.")

        display_weight = _kg_to_unit(weight_kg, unit)
        delta_str = ""
        if previous:
            delta_kg = weight_kg - previous["weight_kg"]
            delta_str = _weight_delta_comment(delta_kg, unit)

        logger.info("Weight logged: %.2f kg (unit=%s).", weight_kg, unit)
        return _ok(
            f"Weight logged — {display_weight:.1f} {unit}.{delta_str}",
            data={"weight_kg": weight_kg, "unit": unit},
            confidence=0.95,
        )

    def _report_weight_trend(self) -> dict:
        """Answer a "what's my weight trend" query from recent logs."""
        try:
            recent = self.db.get_weight(_WEIGHT_TREND_DAYS)
        except Exception:
            logger.exception("_report_weight_trend: DB read failed.")
            return _err("I couldn't check your weight log right now.")

        if not recent:
            return _clarify(
                "I don't have any weight logged yet. What's your current weight?"
            )

        latest = recent[0]["weight_kg"]
        logged_at = (recent[0].get("logged_at") or "")[:10]

        if len(recent) == 1:
            return _ok(
                f"Your last logged weight was {latest:.1f} kg on {logged_at}. "
                "Log a few more entries and I can show you a trend.",
                data={"latest_kg": latest, "logged_at": logged_at},
                confidence=0.85,
            )

        oldest = recent[-1]["weight_kg"]
        delta = latest - oldest
        trend = _trend_word(delta)
        return _ok(
            f"Your weight is {trend} {abs(delta):.1f} kg over the last "
            f"{_WEIGHT_TREND_DAYS} days — {oldest:.1f} kg to {latest:.1f} kg. "
            f"Most recent log: {logged_at}.",
            data={
                "latest_kg": latest, "oldest_kg": oldest,
                "delta_kg": round(delta, 2), "logged_at": logged_at,
            },
            confidence=0.9,
        )

    def _log_water(self, entities: dict) -> dict:
        """
        Validate and persist a hydration entry, or — when no amount was
        given and the phrasing reads as a question ("how much water have I
        had today?") — report today's total instead.
        """
        raw_amount = entities.get("amount") or entities.get("ml")
        glasses = entities.get("glasses")

        if raw_amount is None and glasses is None:
            raw_query: str = entities.get("raw_query") or ""
            if _WATER_QUERY_RE.search(raw_query):
                return self._report_water_today()
            return _clarify("How much water did you drink (ml or glasses)?")

        if raw_amount is None and glasses is not None:
            raw_amount = glasses
            unit_hint = "glasses"
        else:
            unit_hint = (entities.get("unit") or "ml").strip().lower()

        amount_ml, err = _parse_water(raw_amount, unit_hint)
        if err:
            return _clarify(err)

        try:
            self.db.log_water(amount_ml)
            total_today = self.db.water_today()
        except Exception:
            logger.exception("_log_water: DB operation failed.")
            return _err("I couldn't save your water intake right now.")

        goal_str = ""
        try:
            goal = self.db.get_goal("water_ml")
        except Exception:
            goal = None
        if goal:
            pct = min(100, round(100 * total_today / goal["target_value"]))
            goal_str = f" That's {pct}% of your {goal['target_value']:.0f} ml goal."

        logger.info("Water logged: %d ml (today total=%d).", amount_ml, total_today)
        return _ok(
            f"Water logged — {amount_ml} ml. Today's total: {total_today} ml.{goal_str}",
            data={"amount_ml": amount_ml, "total_today_ml": total_today},
            confidence=0.95,
        )

    def _report_water_today(self) -> dict:
        """Answer a "how much water have I had today" query."""
        try:
            total_today = self.db.water_today()
            goal = self.db.get_goal("water_ml")
        except Exception:
            logger.exception("_report_water_today: DB read failed.")
            return _err("I couldn't check your water log right now.")

        if total_today == 0:
            return _ok(
                "No water logged yet today. Let me know when you have some!",
                data={"total_today_ml": 0},
                confidence=0.9,
            )

        goal_str = ""
        if goal:
            remaining = goal["target_value"] - total_today
            goal_str = (
                f" You're {remaining:.0f} ml short of your {goal['target_value']:.0f} ml goal."
                if remaining > 0
                else " You've hit your daily goal!"
            )

        return _ok(
            f"You've had {total_today} ml of water today.{goal_str}",
            data={"total_today_ml": total_today},
            confidence=0.9,
        )

    def _set_health_goal(self, entities: dict) -> dict:
        """Validate and persist a target for one of the tracked health metrics."""
        raw_type = (
            entities.get("goal_type") or entities.get("metric")
            or entities.get("raw_query") or ""
        )
        goal_type = _normalize_goal_type(raw_type)
        if goal_type is None:
            valid = ", ".join(sorted(_GOAL_TYPE_ALIASES))
            return _clarify(
                f"What kind of goal? I can track: {valid}."
            )

        raw_target = entities.get("target") or entities.get("value")
        target, err = _parse_goal_target(raw_target)
        if err:
            return _clarify(err)

        try:
            self.db.set_goal(goal_type, target)
        except Exception:
            logger.exception("_set_health_goal: DB operation failed.")
            return _err("I couldn't save that goal right now.")

        label = _GOAL_LABELS[goal_type]
        logger.info("Health goal set: type=%s target=%.2f.", goal_type, target)
        return _ok(
            f"Goal set — {target:g} {label}. I'll track your progress toward this.",
            data={"goal_type": goal_type, "target": target},
            confidence=0.95,
        )

    def _goal_progress(self) -> dict:
        """Compare current metrics against every active health goal."""
        try:
            goals = self.db.get_all_goals()
        except Exception:
            logger.exception("_goal_progress: DB read failed.")
            return _err("I couldn't retrieve your goals right now.")

        if not goals:
            return _ok(
                "No health goals set yet. Try: \"set a goal of 4 workouts a "
                "week\" or \"set a sleep goal of 8 hours\".",
                data={"goals": []},
                confidence=0.9,
            )

        lines: list[str] = []
        progress_data: list[dict[str, Any]] = []
        for goal in goals:
            line, entry = self._format_goal_progress(goal)
            lines.append(line)
            progress_data.append(entry)

        progress_detail = "\n".join(f"  {ln}" for ln in lines)

        try:
            commentary = self._llm(
                _GOAL_PROGRESS_PROMPT.format(progress_detail=progress_detail)
            )
        except LLMError:
            logger.warning("_goal_progress: LLM unavailable; using static response.")
            commentary = "Keep going — consistency matters more than any single day."

        response = f"Goal Progress\n\n{progress_detail}\n\n{commentary}"
        return _ok(
            response,
            data={"goals": progress_data},
            confidence=0.9,
        )

    def _format_goal_progress(self, goal: dict) -> tuple[str, dict[str, Any]]:
        """Render one goal's progress line and return its structured data."""
        goal_type = goal["goal_type"]
        target = goal["target_value"]
        label = _GOAL_LABELS.get(goal_type, goal_type)

        current: Optional[float] = None
        try:
            if goal_type == "workout_frequency":
                current = float(self.db.workout_count(_HEALTH_SUMMARY_DAYS))
            elif goal_type == "sleep_hours":
                current = self.db.avg_sleep(_HEALTH_SUMMARY_DAYS)
            elif goal_type == "weight_target":
                latest = self.db.latest_weight()
                current = latest["weight_kg"] if latest else None
            elif goal_type == "water_ml":
                current = float(self.db.water_today())
        except Exception:
            logger.exception("_format_goal_progress: metric lookup failed for %s.", goal_type)

        if current is None:
            return f"{label}: target {target:g}, no data logged yet", {
                "goal_type": goal_type, "target": target, "current": None,
            }

        if goal_type == "weight_target":
            remaining = abs(current - target)
            direction = "to lose" if current > target else "to gain"
            line = f"{label}: currently {current:.1f} kg, {remaining:.1f} kg {direction}"
        else:
            pct = min(100, round(100 * current / target)) if target else 0
            line = f"{label}: {current:g} / {target:g} ({pct}%)"

        return line, {"goal_type": goal_type, "target": target, "current": current}

    def _log_mood(self, entities: dict) -> dict:
        """Persist a mood entry and return an empathetic AI response."""
        mood: str = (
            entities.get("mood") or entities.get("raw_query") or "neutral"
        ).strip()
        notes: str = (entities.get("notes") or "").strip()

        try:
            self.db.log_mood(mood, notes)
            history = self.db.recent_moods(_MOOD_HISTORY_WINDOW)
        except Exception:
            logger.exception("_log_mood: DB operation failed.")
            return _err("I couldn't save your mood right now.")

        history_str = ", ".join(history) if history else "none"

        try:
            response = self._llm(
                _MOOD_RESPONSE_PROMPT.format(mood=mood, history=history_str)
            )
        except LLMError:
            logger.warning("_log_mood: LLM unavailable; using static response.")
            response = f"Mood logged as {mood!r}. Thanks for checking in."

        logger.info("Mood logged: mood=%r.", mood)
        return _ok(
            response,
            data={"mood": mood, "recent_history": history},
            confidence=0.95,
        )

    def _health_summary(self) -> dict:
        """Compile a 7-day health report and generate an AI analysis."""
        try:
            workouts = self.db.get_workouts(_HEALTH_SUMMARY_DAYS)
            sleep = self.db.get_sleep(_HEALTH_SUMMARY_DAYS)
            moods = self.db.get_mood(_HEALTH_SUMMARY_DAYS)
            weight = self.db.get_weight(_HEALTH_SUMMARY_DAYS)
            water = self.db.get_water(_HEALTH_SUMMARY_DAYS)
            goals = self.db.get_all_goals()
            avg_sleep = self.db.avg_sleep(_HEALTH_SUMMARY_DAYS) or 0.0
            water_today = self.db.water_today()
            streak = _compute_streak(self.db.workout_dates(_STREAK_LOOKBACK_DAYS))
        except Exception:
            logger.exception("_health_summary: DB read failed.")
            return _err("I couldn't retrieve your health data right now.")

        if not workouts and not sleep and not moods and not weight and not water:
            return _ok(
                "No health data logged yet. "
                "Start by telling me about your workout, sleep, mood, weight, or water.",
                confidence=0.9,
            )

        workout_detail = _format_workouts(workouts)
        sleep_detail = _format_sleep(sleep)
        mood_detail = _format_moods(moods)
        weight_detail = _format_weight(weight)
        water_detail = _format_water(water)
        goal_detail = _format_goals(goals) if goals else "  None set"

        try:
            analysis = self._llm(
                _HEALTH_SUMMARY_PROMPT.format(
                    days=_HEALTH_SUMMARY_DAYS,
                    workout_count=len(workouts),
                    workout_detail=workout_detail,
                    streak=streak,
                    avg_sleep=f"{avg_sleep:.1f}",
                    sleep_detail=sleep_detail,
                    mood_detail=mood_detail,
                    weight_detail=weight_detail,
                    water_today=water_today,
                    water_detail=water_detail,
                    goal_detail=goal_detail,
                )
            )
        except LLMError:
            logger.warning("_health_summary: LLM unavailable; returning data-only summary.")
            analysis = "AI analysis unavailable — raw data shown above."

        response = (
            f"Health Summary — Last {_HEALTH_SUMMARY_DAYS} Days\n\n"
            f"WORKOUTS  : {len(workouts)} session(s), streak: {streak} day(s)\n"
            f"AVG SLEEP : {avg_sleep:.1f} h/night\n"
            f"MOOD LOGS : {len(moods)} entry/entries\n"
            f"WEIGHT    : {_weight_summary_line(weight)}\n"
            f"HYDRATION : {water_today} ml today\n\n"
            f"ANALYSIS\n{analysis}"
        )

        return _ok(
            response,
            data={
                "workout_count": len(workouts),
                "workout_streak": streak,
                "avg_sleep": avg_sleep,
                "mood_count": len(moods),
                "water_today_ml": water_today,
                "goal_count": len(goals),
            },
            confidence=0.95,
        )


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

_LEADING_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _extract_number(raw: Any) -> Optional[float]:
    """
    Pull the first number out of a string that may carry extra words the
    NLU left in (e.g. "30 min run", "7 hours"). ``float(str(raw))`` chokes
    on anything but a bare number, which is why these fields kept
    round-tripping back to the user as "I didn't catch that" even when the
    NLU had, in fact, caught it.
    """
    if raw is None:
        return None
    match = _LEADING_NUMBER_RE.search(str(raw))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_duration(raw: Any) -> tuple[int, Optional[str]]:
    """
    Parse and validate a workout duration.

    Returns ``(duration_int, None)`` on success or ``(0, error_message)``
    on failure.
    """
    if raw is None:
        return _DEFAULT_WORKOUT_DURATION, None
    value_f = _extract_number(raw)
    if value_f is None:
        return 0, "I didn't catch the workout duration. How many minutes?"
    value = int(value_f)
    if not (_MIN_WORKOUT_DURATION <= value <= _MAX_WORKOUT_DURATION):
        return 0, (
            f"Duration should be between {_MIN_WORKOUT_DURATION} and "
            f"{_MAX_WORKOUT_DURATION} minutes."
        )
    return value, None


def _parse_hours(raw: Any) -> tuple[float, Optional[str]]:
    """
    Parse and validate sleep hours.

    Returns ``(hours_float, None)`` on success or ``(0.0, error_message)``
    on failure.
    """
    value = _extract_number(raw)
    if value is None:
        return 0.0, "I didn't catch the sleep duration. How many hours?"
    if not (_MIN_SLEEP_HOURS <= value <= _MAX_SLEEP_HOURS):
        return 0.0, (
            f"Sleep hours should be between {_MIN_SLEEP_HOURS} and "
            f"{_MAX_SLEEP_HOURS}."
        )
    return value, None


def _sleep_comment(hours: float) -> str:
    """Return a contextual comment based on sleep duration."""
    if hours < _LOW_SLEEP_THRESHOLD:
        return _SLEEP_COMMENTS["low"]
    if hours >= _GOOD_SLEEP_THRESHOLD:
        return _SLEEP_COMMENTS["good"]
    return _SLEEP_COMMENTS["ok"]


def _normalize_weight_unit(raw: Any) -> str:
    """Map a free-form unit string onto 'kg' or 'lb', defaulting to kg."""
    if not raw:
        return _DEFAULT_WEIGHT_UNIT
    key = str(raw).strip().lower().rstrip(".")
    return _WEIGHT_UNIT_ALIASES.get(key, _DEFAULT_WEIGHT_UNIT)


def _parse_weight(raw: Any, unit: str) -> tuple[float, Optional[str]]:
    """
    Parse and validate a weight entry, converting to kg for storage.

    Returns ``(weight_kg, None)`` on success or ``(0.0, error_message)``
    on failure. Range validation is applied in kg regardless of the
    input unit, so the same physical limits apply no matter how the
    user phrases it.
    """
    try:
        value = float(str(raw))
    except (ValueError, TypeError):
        return 0.0, "I didn't catch the weight. What's the number?"
    weight_kg = value * _KG_PER_LB if unit == "lb" else value
    if not (_MIN_WEIGHT_KG <= weight_kg <= _MAX_WEIGHT_KG):
        lo = _MIN_WEIGHT_KG if unit == "kg" else round(_MIN_WEIGHT_KG / _KG_PER_LB)
        hi = _MAX_WEIGHT_KG if unit == "kg" else round(_MAX_WEIGHT_KG / _KG_PER_LB)
        return 0.0, f"That doesn't look right — weight should be between {lo} and {hi} {unit}."
    return weight_kg, None


def _kg_to_unit(weight_kg: float, unit: str) -> float:
    """Convert a stored kg weight back to the unit the user is using."""
    return weight_kg / _KG_PER_LB if unit == "lb" else weight_kg


def _weight_delta_comment(delta_kg: float, unit: str) -> str:
    """Short comment on the change since the previous weight log."""
    if abs(delta_kg) < 0.05:
        return " That's steady since your last log."
    delta_display = abs(_kg_to_unit(delta_kg, unit))
    direction = "down" if delta_kg < 0 else "up"
    return f" That's {direction} {delta_display:.1f} {unit} from your last log."


def _trend_word(delta: float) -> str:
    """Describe a signed delta as a direction word."""
    if delta < 0:
        return "down"
    if delta > 0:
        return "up"
    return "unchanged"


def _parse_water(raw: Any, unit_hint: str) -> tuple[int, Optional[str]]:
    """
    Parse and validate a hydration entry, converting to ml for storage.

    ``unit_hint`` may be "glasses", "oz", or "ml" (default). Returns
    ``(amount_ml, None)`` on success or ``(0, error_message)`` on failure.
    """
    try:
        value = float(str(raw))
    except (ValueError, TypeError):
        return 0, "I didn't catch the amount. How much water (ml or glasses)?"

    unit_hint = (unit_hint or "ml").strip().lower()
    if unit_hint.startswith("glass"):
        amount_ml = value * _ML_PER_GLASS
    elif unit_hint in ("oz", "ounce", "ounces"):
        amount_ml = value * _ML_PER_OZ
    else:
        amount_ml = value

    if not (_MIN_WATER_ML <= amount_ml <= _MAX_WATER_ML):
        return 0, f"That should be between {_MIN_WATER_ML} and {_MAX_WATER_ML} ml per entry."
    return round(amount_ml), None


def _normalize_goal_type(raw: str) -> Optional[str]:
    """Map free-form goal phrasing onto a canonical goal type, or None."""
    key = str(raw).strip().lower()
    if key in _GOAL_TYPES:
        return key
    for alias, canonical in _GOAL_TYPE_ALIASES.items():
        if alias in key:
            return canonical
    return None


def _parse_goal_target(raw: Any) -> tuple[float, Optional[str]]:
    """
    Parse and validate a goal target value.

    Returns ``(target, None)`` on success or ``(0.0, error_message)``
    on failure.
    """
    try:
        value = float(str(raw))
    except (ValueError, TypeError):
        return 0.0, "What target should I set for that goal?"
    if not (_MIN_GOAL_TARGET <= value <= _MAX_GOAL_TARGET):
        return 0.0, f"That target should be between {_MIN_GOAL_TARGET} and {_MAX_GOAL_TARGET:g}."
    return value, None


def _compute_streak(dates_desc: list[str]) -> int:
    """
    Count consecutive calendar days with an activity, working backward
    from today (or yesterday, so a streak isn't broken just because
    today hasn't been logged yet).

    ``dates_desc`` is a list of "YYYY-MM-DD" strings, newest first, as
    returned by ApolloDB.workout_dates(). Pure and independently
    testable — no DB or clock dependency beyond the date strings given.
    """
    if not dates_desc:
        return 0

    dates = {date.fromisoformat(d) for d in dates_desc}
    today = date.today()

    cursor = today if today in dates else today - timedelta(days=1)
    if cursor not in dates:
        return 0

    streak = 0
    while cursor in dates:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def _format_workouts(workouts: list[dict]) -> str:
    if not workouts:
        return "  None logged"
    return "\n".join(
        f"  {w.get('logged_at', '')[:10]} — "
        f"{w.get('type', 'unknown')} ({w.get('duration', '?')} min)"
        for w in workouts
    )


def _format_sleep(sleep: list[dict]) -> str:
    if not sleep:
        return "  None logged"
    return "\n".join(
        f"  {s.get('logged_at', '')[:10]} — "
        f"{s.get('hours', '?')}h  {s.get('quality') or ''}".rstrip()
        for s in sleep
    )


def _format_moods(moods: list[dict]) -> str:
    if not moods:
        return "  None logged"
    return "\n".join(
        f"  {m.get('logged_at', '')[:10]} — {m.get('mood', 'unknown')}"
        for m in moods
    )


def _format_weight(weight: list[dict]) -> str:
    if not weight:
        return "  None logged"
    return "\n".join(
        f"  {w.get('logged_at', '')[:10]} — {w.get('weight_kg', '?'):.1f} kg"
        for w in weight
    )


def _weight_summary_line(weight: list[dict]) -> str:
    """One-line weight summary for the fixed-format header of the health summary."""
    if not weight:
        return "no data logged"
    latest = weight[0]["weight_kg"]
    if len(weight) == 1:
        return f"{latest:.1f} kg"
    delta = latest - weight[-1]["weight_kg"]
    return f"{latest:.1f} kg ({_trend_word(delta)} {abs(delta):.1f} kg)"


def _format_water(water: list[dict]) -> str:
    if not water:
        return "  None logged"
    return "\n".join(
        f"  {w.get('logged_at', '')[:10]} — {w.get('amount_ml', '?')} ml"
        for w in water
    )


def _format_goals(goals: list[dict]) -> str:
    if not goals:
        return "  None set"
    return "\n".join(
        f"  {_GOAL_LABELS.get(g['goal_type'], g['goal_type'])}: target {g['target_value']:g}"
        for g in goals
    )


def _ok(
    response: str,
    data: Optional[dict[str, Any]] = None,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict[str, Any]:
    return {"response": response, "data": {}, "confidence": 0.0}


def _clarify(question: str) -> dict[str, Any]:
    return {
        "response": question,
        "data": {"needs_clarification": True},
        "confidence": 0.5,
    }