"""
modules/apollo/engine.py

ApolloEngine: personal health tracking module for workouts, sleep, mood,
and AI-generated health summaries.

Design notes
------------
- All LLM calls are isolated in a single helper that raises a typed
  exception on failure; handlers catch it and fall back to a static
  response so the user always gets an answer.
- DB calls are wrapped per-handler so a storage failure returns a clear
  error without crashing the orchestrator.
- Validation (duration, hours, mood) is handled by pure module-level
  helpers that are independently testable.
- Sleep quality thresholds and comment strings are module-level constants
  so they can be audited and adjusted without touching business logic.
- Every public method conforms to the BaseModule response contract:
  {response: str, data: dict, confidence: float}.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
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

Give a short (2-3 sentence) motivating response. Mention their weekly count.
Be warm and specific. No bullet points."""

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

WORKOUTS ({workout_count} sessions):
{workout_detail}

SLEEP (avg {avg_sleep} hours/night):
{sleep_detail}

MOOD LOG:
{mood_detail}

Write a health summary with:
1. What's going well
2. What needs attention
3. One specific recommendation for next week

Under 200 words. Be direct and actionable."""


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
        }
    )

    def __init__(
        self,
        ollama_cfg: Optional[dict[str, Any]] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        self._cfg = OllamaConfig.from_dict(ollama_cfg or {})
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
            return {
                "apollo_workouts_this_week": self.db.workout_count(_HEALTH_SUMMARY_DAYS),
                "apollo_avg_sleep": self.db.avg_sleep(_HEALTH_SUMMARY_DAYS),
                "apollo_recent_moods": self.db.recent_moods(_RECENT_MOOD_CONTEXT),
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
        except Exception:
            logger.exception("_log_workout: DB operation failed.")
            return _err("I couldn't save your workout right now.")

        try:
            feedback = self._llm(
                _WORKOUT_FEEDBACK_PROMPT.format(
                    type_=type_, duration=duration, notes=notes, count=count
                )
            )
        except LLMError:
            logger.warning("_log_workout: LLM unavailable; using static response.")
            feedback = f"Workout logged — {duration} min of {type_}. That's {count} session(s) this week!"

        logger.info("Workout logged: type=%s duration=%d count=%d.", type_, duration, count)
        return _ok(
            feedback,
            data={"type": type_, "duration": duration, "weekly_count": count},
            confidence=0.95,
        )

    def _track_sleep(self, entities: dict) -> dict:
        """Validate, persist, and comment on a sleep entry."""
        raw_hours = entities.get("hours") or entities.get("duration")
        if raw_hours is None:
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
            avg_sleep = self.db.avg_sleep(_HEALTH_SUMMARY_DAYS) or 0.0
        except Exception:
            logger.exception("_health_summary: DB read failed.")
            return _err("I couldn't retrieve your health data right now.")

        if not workouts and not sleep and not moods:
            return _ok(
                "No health data logged yet. "
                "Start by telling me about your workout, sleep, or mood.",
                confidence=0.9,
            )

        workout_detail = _format_workouts(workouts)
        sleep_detail = _format_sleep(sleep)
        mood_detail = _format_moods(moods)

        try:
            analysis = self._llm(
                _HEALTH_SUMMARY_PROMPT.format(
                    days=_HEALTH_SUMMARY_DAYS,
                    workout_count=len(workouts),
                    workout_detail=workout_detail,
                    avg_sleep=f"{avg_sleep:.1f}",
                    sleep_detail=sleep_detail,
                    mood_detail=mood_detail,
                )
            )
        except LLMError:
            logger.warning("_health_summary: LLM unavailable; returning data-only summary.")
            analysis = "AI analysis unavailable — raw data shown above."

        response = (
            f"Health Summary — Last {_HEALTH_SUMMARY_DAYS} Days\n\n"
            f"WORKOUTS  : {len(workouts)} session(s)\n"
            f"AVG SLEEP : {avg_sleep:.1f} h/night\n"
            f"MOOD LOGS : {len(moods)} entry/entries\n\n"
            f"ANALYSIS\n{analysis}"
        )

        return _ok(
            response,
            data={
                "workout_count": len(workouts),
                "avg_sleep": avg_sleep,
                "mood_count": len(moods),
            },
            confidence=0.95,
        )


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _parse_duration(raw: Any) -> tuple[int, Optional[str]]:
    """
    Parse and validate a workout duration.

    Returns ``(duration_int, None)`` on success or ``(0, error_message)``
    on failure.
    """
    if raw is None:
        return _DEFAULT_WORKOUT_DURATION, None
    try:
        value = int(float(str(raw)))
    except (ValueError, TypeError):
        return 0, "I didn't catch the workout duration. How many minutes?"
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
    try:
        value = float(str(raw))
    except (ValueError, TypeError):
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