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

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from core.ollama_client import generate
from core.free_apis import (
    FreeAPIError,
    food_lookup as _fa_food_lookup,
    exercise_lookup as _fa_exercise_lookup,
)
from modules.base import BaseModule
from .db import ApolloDB
from . import insights as _insights
from . import schedule as _schedule

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
    {"workout_frequency", "sleep_hours", "weight_target", "water_ml",
     "calories_kcal"}
)

_GOAL_TYPE_ALIASES: dict[str, str] = {
    "workout": "workout_frequency", "workouts": "workout_frequency",
    "exercise": "workout_frequency", "workout_frequency": "workout_frequency",
    "sleep": "sleep_hours", "sleep_hours": "sleep_hours",
    "weight": "weight_target", "weight_target": "weight_target",
    "water": "water_ml", "hydration": "water_ml", "water_ml": "water_ml",
    "calorie": "calories_kcal", "calories": "calories_kcal",
    "kcal": "calories_kcal", "calories_kcal": "calories_kcal",
}

_GOAL_LABELS: dict[str, str] = {
    "workout_frequency": "workouts/week",
    "sleep_hours": "hours of sleep/night",
    "weight_target": "kg target weight",
    "water_ml": "ml of water/day",
    "calories_kcal": "kcal/day",
}

# Apollo never sets a calorie target for the user, and refuses one below this
# floor when the user asks for it themselves.
_MIN_CALORIE_GOAL = 1200
_MAX_MEAL_KCAL = 5000
_FAST_WEIGHT_CHANGE_KG_WEEK = 1.0
_DEFAULT_MIN_SAMPLE = 4
_DEFAULT_STREAK_MIN = 2
_DEFAULT_NUDGE_THRESHOLD_ML = 400
_MAX_IMPORT_BYTES = 5 * 1024 * 1024

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
            "lookup_food",
            "suggest_exercise",
            # Backlog #111-#120, #126, #161
            "set_units",
            "get_sleep_quality",
            "get_correlations",
            "log_meal",
            "get_meal_summary",
            "hydration_status",
            "get_workout_streaks",
            "log_pain",
            "get_pain_trend",
            "get_weekly_summary",
            "import_steps",
            "get_goal_pace",
            "habit_mood_correlation",
            "burnout_check",
        }
    )

    def __init__(
        self,
        ollama_cfg: Optional[dict[str, Any]] = None,
        db_path: Optional[Path] = None,
        llm: Optional[Any] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self._config: dict[str, Any] = dict(config or {})
        self._tz = _insights.resolve_tz(self._config.get("timezone"))
        self._artemis: Optional[Any] = None
        self._pluto: Optional[Any] = None
        self._food_lookup = _fa_food_lookup  # injectable for tests
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
        if intent == "lookup_food":
            return self._lookup_food(entities)
        if intent == "suggest_exercise":
            return self._suggest_exercise(entities)
        handler = {
            "set_units": self._set_units,
            "get_sleep_quality": self._get_sleep_quality,
            "get_correlations": self._get_correlations,
            "log_meal": self._log_meal,
            "get_meal_summary": self._get_meal_summary,
            "hydration_status": self._hydration_status,
            "get_workout_streaks": self._get_workout_streaks,
            "log_pain": self._log_pain,
            "get_pain_trend": self._get_pain_trend,
            "get_weekly_summary": self._get_weekly_summary,
            "import_steps": self._import_steps,
            "get_goal_pace": self._get_goal_pace,
            "habit_mood_correlation": self._habit_mood_correlation,
            "burnout_check": self._burnout_check,
        }.get(intent)
        if handler is not None:
            return handler(entities)
        return _err(f"Unknown intent: {intent!r}")

    # ------------------------------------------------------------------
    # Wiring hooks (read-only views of sibling modules)
    # ------------------------------------------------------------------

    def attach_artemis(self, artemis: Any) -> None:
        """Give Apollo read-only access to Artemis habit history (#126, #161).

        Apollo only ever calls ``tracker.habit_history()`` / ``get_habits()``;
        it never mutates Artemis state.
        """
        self._artemis = artemis

    def attach_pluto(self, pluto: Any) -> None:
        """Give Apollo read-only access to Pluto's expenses (#161)."""
        self._pluto = pluto

    # ------------------------------------------------------------------
    # Config / time / units helpers
    # ------------------------------------------------------------------

    def _cfg_section(self, name: str) -> dict:
        value = self._config.get(name)
        return value if isinstance(value, dict) else {}

    def _now(self, now: Optional[datetime] = None) -> datetime:
        if now is None:
            return datetime.now(self._tz)
        if now.tzinfo is None:
            return now.replace(tzinfo=self._tz)
        return now.astimezone(self._tz)

    def _profile(self) -> dict:
        try:
            return self.db.get_profile()
        except Exception:
            return {}

    def _weight_unit(self) -> str:
        value = (
            self._profile().get("weight_unit")
            or self._cfg_section("units").get("weight")
            or "kg"
        )
        return value if value in ("kg", "lb") else "kg"

    def _water_unit(self) -> str:
        value = (
            self._profile().get("water_unit")
            or self._cfg_section("units").get("water")
            or "ml"
        )
        return value if value in ("ml", "oz") else "ml"

    def _min_sample(self, entities: Optional[dict] = None) -> int:
        raw = (entities or {}).get("min_sample") or self._config.get("min_sample")
        try:
            return max(2, int(raw))
        except (TypeError, ValueError):
            return _DEFAULT_MIN_SAMPLE

    def _days_of(self, rows: list[dict], start: date, end: date) -> list[tuple[date, dict]]:
        out = []
        for row in rows:
            d = _insights.local_date(row.get("logged_at"), self._tz)
            if d is not None and start <= d <= end:
                out.append((d, row))
        return out

    def _water_by_day(self, days: int) -> dict[date, float]:
        out: dict[date, float] = defaultdict(float)
        for row in self.db.get_water(days):
            d = _insights.local_date(row.get("logged_at"), self._tz)
            if d is not None:
                out[d] += float(row["amount_ml"])
        return out

    def _meals_on(self, day: date) -> list[dict]:
        return [
            r for d, r in self._days_of(self.db.get_meals(3), day, day)
        ]

    # ------------------------------------------------------------------
    # #113 Units
    # ------------------------------------------------------------------

    def _set_units(self, entities: dict) -> dict:
        blob = " ".join(
            str(entities.get(k) or "")
            for k in ("weight_unit", "water_unit", "unit", "system", "raw_query")
        ).lower()
        tokens = re.findall(r"[a-z]+", blob)
        weight = _explicit_weight_unit(entities.get("weight_unit"))
        water = _explicit_water_unit(entities.get("water_unit"))
        if "metric" in tokens:
            weight, water = weight or "kg", water or "ml"
        if "imperial" in tokens:
            weight, water = weight or "lb", water or "oz"
        weight = weight or _explicit_weight_unit(blob)
        water = water or _explicit_water_unit(blob)
        if not weight and not water:
            return _clarify(
                "Which units would you like — kg or lb for weight, and ml or oz for water?"
            )
        try:
            if weight:
                self.db.set_profile("weight_unit", weight)
            if water:
                self.db.set_profile("water_unit", water)
        except Exception:
            logger.exception("_set_units: DB operation failed.")
            return _err("I couldn't save your unit preference right now.")
        parts = []
        if weight:
            parts.append(f"weight in {weight}")
        if water:
            parts.append(f"water in {water}")
        return _ok(
            f"Units set — {' and '.join(parts)}. Everything is still stored as kg "
            "and ml underneath, so you can switch back without losing anything. "
            "If you type a unit in a message, that wins for that entry.",
            data={"weight_unit": self._weight_unit(), "water_unit": self._water_unit()},
            confidence=0.95,
        )

    # ------------------------------------------------------------------
    # #111 Sleep quality
    # ------------------------------------------------------------------

    def _sleep_times(self, entities: dict, raw_query: str) -> tuple[Optional[str], Optional[str]]:
        bed = _insights.parse_time_of_day(entities.get("bed_time"))
        wake = _insights.parse_time_of_day(entities.get("wake_time"))
        if not (bed and wake):
            window = _insights.parse_sleep_window(raw_query)
            if window:
                bed, wake = bed or window[0], wake or window[1]
        return bed, wake

    def _sleep_scores(self, rows_asc: list[dict]) -> list[dict[str, Any]]:
        """Score each night; consistency uses the night and up to 6 before it."""
        out = []
        for i, row in enumerate(rows_asc):
            window = rows_asc[max(0, i - 6): i + 1]
            score = _insights.sleep_quality_score(
                row.get("hours"), row.get("rating"), row.get("quality"),
                [r["bed_time"] for r in window if r.get("bed_time")],
                [r["wake_time"] for r in window if r.get("wake_time")],
            )
            out.append({**score, "row": row})
        return out

    def _score_latest(self, rows_asc: list[dict]) -> Optional[dict[str, Any]]:
        scored = self._sleep_scores(rows_asc)
        return scored[-1] if scored else None

    def _get_sleep_quality(self, entities: dict) -> dict:
        try:
            rows = list(reversed(self.db.get_sleep(14)))
        except Exception:
            logger.exception("_get_sleep_quality: DB read failed.")
            return _err("I couldn't check your sleep log right now.")
        if not rows:
            return _clarify(
                "I don't have any sleep logged yet. How many hours did you sleep last night?"
            )
        scored = self._sleep_scores(rows)
        last = scored[-1]
        recent = [s["score"] for s in scored[-7:] if s["score"] is not None]
        line = f"Last logged night: {last['row']['hours']:g} h"
        if last["score"] is not None:
            parts = ", ".join(f"{k} {v}" for k, v in last["components"].items())
            line += f", sleep quality score {last['score']}/100 ({parts})."
        else:
            line += "."
        if len(recent) >= 2:
            line += f" Average score over your last {len(recent)} nights: {sum(recent) / len(recent):.0f}/100."
        notes = []
        if "consistency" in last["missing"]:
            notes.append(
                "Consistency isn't counted yet — log bed and wake times (e.g. "
                "\"slept 11pm to 6:30am\") on at least 3 nights."
            )
        if "rating" in last["missing"]:
            notes.append("Add a 1-5 rating or quality word to include how it felt.")
        return _ok(
            " ".join([line] + notes),
            data={"score": last["score"], "components": last["components"],
                  "missing": last["missing"], "nights": len(rows)},
            confidence=0.9,
        )

    # ------------------------------------------------------------------
    # #112 Correlations (no LLM)
    # ------------------------------------------------------------------

    def _get_correlations(self, entities: dict, now: Optional[datetime] = None) -> dict:
        days = 90
        n_min = self._min_sample(entities)
        try:
            moods = self.db.get_mood(days)
            sleep = self.db.get_sleep(days)
            workouts = self.db.get_workouts(days)
        except Exception:
            logger.exception("_get_correlations: DB read failed.")
            return _err("I couldn't read your logs right now.")

        mood_days = _insights.daily_mood(moods, self._tz)
        if len(mood_days) < 2 * n_min:
            return _ok(
                f"I can't say anything reliable yet. I need mood logged on at "
                f"least {2 * n_min} different days to compare anything, and I "
                f"have {len(mood_days)} scorable day(s) in the last {days} days. "
                "Keep logging mood alongside sleep and workouts and ask again.",
                data={"status": "insufficient", "mood_days": len(mood_days)},
                confidence=0.9,
            )

        sleep_days: dict[date, float] = {}
        for row in sleep:
            d = _insights.local_date(row.get("logged_at"), self._tz)
            if d is not None:
                sleep_days[d] = float(row["hours"])
        workout_days = {
            d for d in (
                _insights.local_date(w.get("logged_at"), self._tz) for w in workouts
            ) if d is not None
        }

        by_sleep = _insights.mood_by_sleep(sleep_days, mood_days, min_n=n_min)
        by_workout = _insights.mood_by_workout(workout_days, mood_days, min_n=n_min)

        lines = [
            f"Patterns in your own logs (mood scored -2 to +2, last {days} days):"
        ]
        lines.append(
            self._compare_line(
                by_sleep, "days with under 6h of sleep", "days with 7h or more",
                "sleep vs mood", n_min,
            )
        )
        lines.append(
            self._compare_line(
                by_workout, "workout days", "days without a workout",
                "workouts vs mood", n_min,
            )
        )
        lines.append(
            "These are patterns in what you logged, not proof that one thing "
            "causes another."
        )
        return _ok(
            "\n".join(lines),
            data={"sleep": by_sleep, "workout": by_workout,
                  "mood_days": len(mood_days)},
            confidence=0.9,
        )

    @staticmethod
    def _compare_line(res: dict, name_a: str, name_b: str, title: str, n_min: int) -> str:
        na, nb = res["n_a"], res["n_b"]
        if res["status"] == "insufficient":
            return (
                f"- {title.capitalize()}: not enough data yet — I need at least "
                f"{n_min} mood-logged days in each group ({na} {name_a}, {nb} {name_b})."
            )
        a, b = res["mean_a"], res["mean_b"]
        if res["status"] == "no_difference":
            return (
                f"- {title.capitalize()}: no clear difference in your logs "
                f"(mood {a:+.1f} on {name_a}, n={na}; {b:+.1f} on {name_b}, n={nb})."
            )
        word = "higher" if a > b else "lower"
        return (
            f"- {title.capitalize()}: in your logs, mood averaged {a:+.1f} on "
            f"{name_a} (n={na}) vs {b:+.1f} on {name_b} (n={nb}) — {word} on the first."
        )

    # ------------------------------------------------------------------
    # #116 Workout streaks
    # ------------------------------------------------------------------

    def _get_workout_streaks(self, entities: dict, now: Optional[datetime] = None) -> dict:
        today = self._now(now).date()
        raw_min = entities.get("min_sessions") or self._config.get("streak_min_sessions")
        try:
            min_sessions = max(1, int(raw_min)) if raw_min is not None else _DEFAULT_STREAK_MIN
        except (TypeError, ValueError):
            min_sessions = _DEFAULT_STREAK_MIN
        try:
            workouts = self.db.get_workouts(400)
        except Exception:
            logger.exception("_get_workout_streaks: DB read failed.")
            return _err("I couldn't read your workouts right now.")
        if not workouts:
            return _ok("No workouts logged yet — log one and I'll start tracking streaks.",
                       data={"streaks": {}}, confidence=0.9)

        streaks = _insights.type_streaks(workouts, self._tz, today, min_sessions)
        wanted = (entities.get("type") or entities.get("exercise") or "").strip().lower()
        if wanted:
            streaks = {k: v for k, v in streaks.items() if wanted in k}
            if not streaks:
                return _ok(f"I don't see any '{wanted}' workouts logged.",
                           data={"streaks": {}}, confidence=0.85)

        ordered = sorted(streaks.items(), key=lambda kv: (-kv[1]["weeks"], -kv[1]["days"], kv[0]))
        lines = [
            f"Workout streaks (a week counts with {min_sessions}+ sessions of that type):"
        ]
        for kind, s in ordered[:8]:
            lines.append(
                f"- {kind}: {s['days']} day(s) in a row, {s['weeks']} week(s) in a row "
                f"({s['sessions']} sessions logged)"
            )
        return _ok("\n".join(lines),
                   data={"streaks": streaks, "min_sessions": min_sessions},
                   confidence=0.9)

    # ------------------------------------------------------------------
    # #114 Meals
    # ------------------------------------------------------------------

    def _log_meal(self, entities: dict, now: Optional[datetime] = None) -> dict:
        raw_query = (entities.get("raw_query") or "").strip()
        name = str(
            entities.get("food") or entities.get("meal") or entities.get("name")
            or entities.get("item") or ""
        ).strip()
        raw_kcal = entities.get("kcal") if entities.get("kcal") is not None else entities.get("calories")
        raw_grams = entities.get("grams") if entities.get("grams") is not None else entities.get("quantity")
        raw_protein = entities.get("protein")

        kcal = _extract_number(raw_kcal) if raw_kcal is not None else None
        protein = _extract_number(raw_protein) if raw_protein is not None else None
        grams = _extract_number(raw_grams) if raw_grams is not None else None

        if not name and kcal is not None and raw_query:
            name = raw_query[:60]
        if not name:
            return _clarify(
                "What did you eat? Add the calories if you know them, e.g. "
                "\"log a meal: dal and rice, 450 kcal\"."
            )
        if kcal is not None and not (0 <= kcal <= _MAX_MEAL_KCAL):
            return _clarify(
                f"{kcal:g} kcal doesn't look right for one entry. How many calories was it?"
            )

        estimated = False
        source = "manual"
        if kcal is None:
            per100 = self._lookup_per_100g(name)
            if per100 is None:
                return _clarify(
                    f"I couldn't find calories for '{name}'. Tell me the kcal "
                    "and I'll log it as you entered it."
                )
            assumed = grams is None
            grams = grams if grams is not None else 100.0
            kcal = per100["kcal"] * grams / 100.0
            if protein is None and per100.get("protein") is not None:
                protein = per100["protein"] * grams / 100.0
            estimated, source = True, "Open Food Facts"

        try:
            self.db.log_meal(
                name, kcal, protein_g=protein, grams=grams,
                estimated=estimated, source=source,
                notes=raw_query or None,
            )
            today_meals = self._meals_on(self._now(now).date())
            goal = self.db.get_goal("calories_kcal")
        except Exception:
            logger.exception("_log_meal: DB operation failed.")
            return _err("I couldn't save that meal right now.")

        if estimated:
            head = (
                f"Logged {name}: about {kcal:.0f} kcal — an estimate from Open Food "
                f"Facts for {grams:g} g"
                + (" (I assumed 100 g)" if assumed else "")
                + "; packaged-food data varies, so treat it as rough."
            )
        else:
            head = f"Logged {name}: {kcal:.0f} kcal (as you entered it)."
        total = sum(m.get("kcal") or 0 for m in today_meals)
        prot = sum(m.get("protein_g") or 0 for m in today_meals)
        tail = (
            f" Today so far: {total:.0f} kcal"
            + (f", {prot:.0f} g protein" if prot else "")
            + f" from {len(today_meals)} logged meal(s) — only what you've logged."
        )
        if goal:
            tail += f" Your goal is {goal['target_value']:g} kcal/day."
        return _ok(
            head + tail,
            data={"name": name, "kcal": round(kcal, 1), "protein_g": protein,
                  "estimated": estimated, "today_kcal": round(total, 1)},
            confidence=0.9,
        )

    def _lookup_per_100g(self, name: str) -> Optional[dict[str, Any]]:
        try:
            results = self._food_lookup(name, limit=3)
        except Exception:
            logger.warning("_log_meal: food lookup failed for %r.", name)
            return None
        for item in results or []:
            kcal = item.get("calories_kcal_100g")
            try:
                if kcal is not None and float(kcal) >= 0:
                    prot = item.get("protein_g_100g")
                    return {
                        "kcal": float(kcal),
                        "protein": float(prot) if prot is not None else None,
                    }
            except (TypeError, ValueError):
                continue
        return None

    def _get_meal_summary(self, entities: dict, now: Optional[datetime] = None) -> dict:
        today = self._now(now).date()
        try:
            meals = self.db.get_meals(8)
            goal = self.db.get_goal("calories_kcal")
        except Exception:
            logger.exception("_get_meal_summary: DB read failed.")
            return _err("I couldn't read your meal log right now.")
        rows = self._days_of(meals, today - timedelta(days=6), today)
        if not rows:
            return _ok("No meals logged in the last week. Try \"log a meal: oats, 300 kcal\".",
                       data={"days": 0}, confidence=0.9)

        by_day: dict[date, list[dict]] = defaultdict(list)
        for d, r in rows:
            by_day[d].append(r)
        lines = []
        todays = by_day.get(today, [])
        if todays:
            total = sum(m.get("kcal") or 0 for m in todays)
            prot = sum(m.get("protein_g") or 0 for m in todays)
            lines.append(
                f"Today: {total:.0f} kcal" + (f", {prot:.0f} g protein" if prot else "")
                + f" across {len(todays)} meal(s):"
            )
            for m in reversed(todays):
                tag = " (estimate)" if m.get("estimated") else ""
                lines.append(f"  - {m['name']}: {(m.get('kcal') or 0):.0f} kcal{tag}")
        else:
            lines.append("Nothing logged today yet.")
        totals = [sum(m.get("kcal") or 0 for m in v) for v in by_day.values()]
        lines.append(
            f"Average on the {len(totals)} day(s) you logged this week: "
            f"{sum(totals) / len(totals):.0f} kcal."
        )
        if goal:
            lines.append(f"Your goal is {goal['target_value']:g} kcal/day.")
        lines.append(
            "These are estimates and only cover what you logged, so gaps in "
            "logging mean gaps in the numbers."
        )
        return _ok("\n".join(lines),
                   data={"days_logged": len(totals),
                         "today_kcal": round(sum(m.get('kcal') or 0 for m in todays), 1)},
                   confidence=0.9)

    # ------------------------------------------------------------------
    # #115 Hydration pacing
    # ------------------------------------------------------------------

    def _hydration_snapshot(self, now: Optional[datetime] = None) -> dict[str, Any]:
        n = self._now(now)
        hc = self._cfg_section("hydration")
        total = self._water_by_day(3).get(n.date(), 0.0)
        goal_row = self.db.get_goal("water_ml")
        goal = float(goal_row["target_value"]) if goal_row else _schedule.DEFAULT_WATER_GOAL_ML
        wake = _schedule.parse_hhmm(hc.get("wake_start"), 7 * 60)
        sleep = _schedule.parse_hhmm(hc.get("wake_end"), 22 * 60)
        status = _schedule.hydration_status(total, goal, n.hour * 60 + n.minute, wake, sleep)
        status["has_goal"] = goal_row is not None
        return status

    def _hydration_status(self, entities: dict, now: Optional[datetime] = None) -> dict:
        try:
            st = self._hydration_snapshot(now)
        except Exception:
            logger.exception("_hydration_status failed.")
            return _err("I couldn't check your hydration right now.")
        unit = self._water_unit()
        threshold = float(self._cfg_section("hydration").get("threshold_ml", _DEFAULT_NUDGE_THRESHOLD_ML))
        text = (
            f"Hydration: {_fmt_water(st['total_ml'], unit)} of "
            f"{_fmt_water(st['goal_ml'], unit)} so far today."
        )
        if st["phase"] == "active":
            if st["behind_ml"] >= threshold:
                text += (
                    f" Pace for this time of day is about {_fmt_water(st['expected_ml'], unit)}, "
                    f"so you're roughly {_fmt_water(st['behind_ml'], unit)} behind — "
                    "a glass or two would catch you up."
                )
            elif st["ahead_ml"] > 0:
                text += " You're ahead of pace."
            else:
                text += " You're on pace."
        else:
            text += " (It's outside your waking-hours window, so no pacing applies.)"
        if not st["has_goal"]:
            text += (
                f" I'm using a default of {_fmt_water(st['goal_ml'], unit)}; "
                "set a water goal to change it."
            )
        return _ok(text, data={k: round(v, 1) if isinstance(v, float) else v
                               for k, v in st.items()}, confidence=0.9)

    def check_hydration_nudge(self, now: Optional[datetime] = None) -> Optional[str]:
        """Heartbeat hook: a nudge message if the user is behind pace, else None.

        Quiet hours, a per-day cap and a cooldown are enforced in
        ``schedule.nudge_decision``. Only fires for people who actually use
        water tracking (a water goal, or water logged in the past week).
        """
        hc = self._cfg_section("hydration")
        if hc.get("enabled") is False:
            return None
        try:
            n = self._now(now)
            if not (self.db.get_goal("water_ml") or self.db.get_water(7)):
                return None
            st = self._hydration_snapshot(n)
            try:
                state = json.loads(self.db.get_state("hydration_nudge") or "{}")
            except ValueError:
                state = {}
            send, _reason, new_state = _schedule.nudge_decision(
                st, state, n,
                threshold_ml=float(hc.get("threshold_ml", _DEFAULT_NUDGE_THRESHOLD_ML)),
                cooldown_min=int(hc.get("cooldown_minutes", 90)),
                daily_cap=int(hc.get("daily_cap", 4)),
            )
            if not send:
                return None
            self.db.set_state("hydration_nudge", json.dumps(new_state))
            unit = self._water_unit()
            return (
                f"Hydration check: you're at {_fmt_water(st['total_ml'], unit)} and pacing "
                f"toward {_fmt_water(st['goal_ml'], unit)} suggests about "
                f"{_fmt_water(st['expected_ml'], unit)} by now. A glass of water would help."
            )
        except Exception:
            logger.exception("check_hydration_nudge failed.")
            return None

    # ------------------------------------------------------------------
    # #118 Weekly summary
    # ------------------------------------------------------------------

    def _weekly_data(self, now: Optional[datetime] = None) -> dict[str, Any]:
        today = self._now(now).date()
        start = today - timedelta(days=6)
        unit = self._weight_unit()
        data: dict[str, Any] = {"weight_unit": unit}

        sleep_all = list(reversed(self.db.get_sleep(16)))
        scored = self._sleep_scores(sleep_all)
        in_win = [
            s for s in scored
            if (d := _insights.local_date(s["row"].get("logged_at"), self._tz)) is not None
            and start <= d <= today
        ]
        data["sleep_n"] = len(in_win)
        if in_win:
            data["sleep_avg"] = sum(s["row"]["hours"] for s in in_win) / len(in_win)
            vals = [s["score"] for s in in_win if s["score"] is not None]
            data["sleep_score_avg"] = sum(vals) / len(vals) if vals else None

        wk = self._days_of(self.db.get_workouts(9), start, today)
        data["workouts"] = len(wk)
        data["active_days"] = len({d for d, _ in wk})

        wt = sorted(self._days_of(self.db.get_weight(9), start, today), key=lambda x: x[1]["logged_at"])
        data["weight_n"] = len(wt)
        if wt:
            data["weight_first"] = wt[0][1]["weight_kg"]
            data["weight_last"] = wt[-1][1]["weight_kg"]

        goal_row = self.db.get_goal("water_ml")
        goal = float(goal_row["target_value"]) if goal_row else _schedule.DEFAULT_WATER_GOAL_ML
        by_day = {d: v for d, v in self._water_by_day(9).items() if start <= d <= today}
        data["water_goal_ml"] = goal
        data["water_days_logged"] = len(by_day)
        data["water_days_met"] = sum(1 for v in by_day.values() if v >= goal)

        steps = [
            r for r in self.db.get_steps(9)
            if start.isoformat() <= r["day"] <= today.isoformat()
        ]
        data["steps_days"] = len(steps)
        if steps:
            data["steps_avg"] = sum(r["steps"] for r in steps) / len(steps)

        moods = self._days_of(self.db.get_mood(9), start, today)
        data["mood_n"] = len(moods)
        if moods:
            common = Counter(str(r["mood"]).strip().lower() for _, r in moods).most_common(1)
            data["mood_label"] = common[0][0] if common else None
        return data

    def weekly_summary_text(self, now: Optional[datetime] = None) -> tuple[str, bool]:
        """(summary text, whether there was any data in the window)."""
        data = self._weekly_data(now)
        has_data = any(
            data.get(k) for k in
            ("sleep_n", "workouts", "weight_n", "water_days_logged", "steps_days", "mood_n")
        )
        return _insights.build_weekly_summary(data), has_data

    def _get_weekly_summary(self, entities: dict, now: Optional[datetime] = None) -> dict:
        try:
            text, has_data = self.weekly_summary_text(now)
        except Exception:
            logger.exception("_get_weekly_summary failed.")
            return _err("I couldn't build your weekly summary right now.")
        if not has_data:
            return _ok("Nothing logged in the last 7 days, so there's nothing to summarise yet.",
                       data={"has_data": False}, confidence=0.9)
        return _ok(text, data={"has_data": True}, confidence=0.9)

    def check_weekly_summary(self, now: Optional[datetime] = None) -> Optional[str]:
        """Heartbeat hook: the summary once per ISO week, else None.

        The sent-marker lives in the DB so it survives restarts (no double
        send after a reboot, no skipped week).
        """
        wc = self._cfg_section("weekly_summary")
        if wc.get("enabled") is False:
            return None
        try:
            n = self._now(now)
            last = self.db.get_state("weekly_summary_sent")
            if not _schedule.weekly_due(
                n, last, weekday=int(wc.get("weekday", 6)), hour=int(wc.get("hour", 18))
            ):
                return None
            text, has_data = self.weekly_summary_text(n)
            if not has_data:
                return None
            self.db.set_state("weekly_summary_sent", _schedule.iso_week_key(n.date()))
            return text
        except Exception:
            logger.exception("check_weekly_summary failed.")
            return None

    # ------------------------------------------------------------------
    # #120 Goal pace
    # ------------------------------------------------------------------

    def _parse_deadline(self, entities: dict) -> tuple[Optional[str], Optional[str]]:
        today = self._now().date()
        raw = str(entities.get("deadline") or "").strip()
        query = str(entities.get("raw_query") or "")
        found: Optional[date] = None
        for text in (raw, query):
            m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
            if m:
                try:
                    found = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    return None, "That deadline isn't a valid date. Try YYYY-MM-DD."
                break
            m = re.search(r"\b(?:in|within)\s+(\d+)\s*(day|week|month)s?\b", text, re.I)
            if m:
                n, unit = int(m.group(1)), m.group(2).lower()
                found = today + timedelta(days=n * {"day": 1, "week": 7, "month": 30}[unit])
                break
        if found is None:
            return None, None
        if found <= today:
            return None, "That deadline is already in the past. What date should I use?"
        return found.isoformat(), None

    def _weight_pace(self, now: Optional[datetime] = None) -> Optional[dict[str, Any]]:
        goal = self.db.get_goal("weight_target")
        latest = self.db.latest_weight()
        if not goal or not latest:
            return None
        today = self._now(now).date()
        points = []
        for row in self.db.get_weight(28):
            d = _insights.local_date(row.get("logged_at"), self._tz)
            if d is not None:
                points.append(((d - today).days, float(row["weight_kg"])))
        slope = None
        if len(points) >= 2 and (max(p[0] for p in points) - min(p[0] for p in points)) >= 7:
            slope = _insights.linear_slope(points)
        days_left = None
        if goal.get("deadline"):
            try:
                days_left = (date.fromisoformat(goal["deadline"]) - today).days
            except ValueError:
                days_left = None
        pace = _insights.goal_pace(
            latest["weight_kg"], goal["target_value"], slope,
            start=goal.get("start_value"), days_left=days_left,
        )
        pace["deadline"] = goal.get("deadline")
        pace["today"] = today
        return pace

    def _pace_text(self, pace: dict[str, Any]) -> str:
        unit = self._weight_unit()
        conv = (lambda kg: kg / _KG_PER_LB) if unit == "lb" else (lambda kg: kg)
        cur, tgt = conv(pace["current"]), conv(pace["target"])
        status = pace["status"]
        if status == "reached":
            return f"Weight goal: you're at your target ({tgt:.1f} {unit}; now {cur:.1f} {unit})."
        direction = "to lose" if pace["remaining"] < 0 else "to gain"
        text = (
            f"Weight goal: {cur:.1f} {unit} now, target {tgt:.1f} {unit} — "
            f"{abs(cur - tgt):.1f} {unit} {direction}."
        )
        if pace["progress_pct"] is not None:
            text += f" About {pace['progress_pct']:.0f}% of the way from where you started."
        rate = pace["rate_per_day"]
        if rate is not None:
            text += f" Recent trend: {conv(rate) * 7:+.2f} {unit}/week."
        if status == "no_trend":
            text += (
                " I don't have a trend yet — weigh in on a few different days "
                "across at least a week."
            )
        elif status == "moving_away":
            text += " Your recent trend is heading away from the target rather than toward it."
        elif status == "steady_no_deadline":
            eta = pace["today"] + timedelta(days=pace["eta_days"])
            text += f" At that pace you'd arrive in about {pace['eta_days']} days (around {eta.isoformat()})."
        elif status in ("on_pace", "behind"):
            eta = pace["today"] + timedelta(days=pace["eta_days"] or 0)
            if status == "on_pace":
                text += (
                    f" At that pace you'd arrive in about {pace['eta_days']} days "
                    f"(around {eta.isoformat()}), before your {pace['deadline']} deadline."
                )
            else:
                text += (
                    f" At that pace you'd arrive around {eta.isoformat()}, after your "
                    f"{pace['deadline']} deadline."
                    if pace["eta_days"] else
                    f" You're behind pace for your {pace['deadline']} deadline."
                )
        # Wellbeing guardrails: never cheer fast change, and flag a deadline
        # that would require it.
        weekly_kg = abs(rate) * 7 if rate is not None else 0.0
        req_kg = abs(pace["required_per_day"]) * 7 if pace.get("required_per_day") else 0.0
        if weekly_kg > _FAST_WEIGHT_CHANGE_KG_WEEK:
            text += (
                " That's a faster rate of change than is usually advised, so rather than "
                "pushing further it's worth checking in with a doctor or dietitian."
            )
        elif req_kg > _FAST_WEIGHT_CHANGE_KG_WEEK:
            text += (
                " Your deadline would need more than about 1 kg a week, which is faster "
                "than is generally recommended — consider moving the date."
            )
        return text

    def _get_goal_pace(self, entities: dict, now: Optional[datetime] = None) -> dict:
        try:
            pace = self._weight_pace(now)
            goals = [g for g in self.db.get_all_goals() if g["goal_type"] != "weight_target"]
        except Exception:
            logger.exception("_get_goal_pace failed.")
            return _err("I couldn't work out your goal pace right now.")
        if pace is None:
            if self.db.get_goal("weight_target"):
                return _clarify("You have a weight goal but no weight logged yet. What do you weigh right now?")
            return _ok(
                "I can work out pace for a weight goal. Try \"set a weight goal of "
                "68 kg by 2026-12-31\".",
                data={"pace": None}, confidence=0.85,
            )
        lines = [self._pace_text(pace)]
        if goals:
            lines.append("Other goals:")
            lines += [f"  {self._format_goal_progress(g)[0]}" for g in goals]
        data = {k: v for k, v in pace.items() if k != "today"}
        return _ok("\n".join(lines), data=data, confidence=0.9)

    def check_goal_pace_reminder(self, now: Optional[datetime] = None) -> Optional[str]:
        """Heartbeat hook. Off unless ``goal_reminders.every_days`` > 0.

        Off by default on purpose: nudges about weight are sensitive, so the
        user opts in and chooses the cadence.
        """
        gc = self._cfg_section("goal_reminders")
        try:
            every = int(gc.get("every_days", 0)) if gc.get("enabled", False) else 0
        except (TypeError, ValueError):
            every = 0
        if every <= 0:
            return None
        try:
            n = self._now(now)
            if not _schedule.every_n_days_due(
                n.date(), self.db.get_state("goal_pace_sent"), every
            ):
                return None
            pace = self._weight_pace(n)
            if pace is None or pace["status"] == "no_trend":
                return None
            self.db.set_state("goal_pace_sent", n.date().isoformat())
            return "Goal check-in: " + self._pace_text(pace)
        except Exception:
            logger.exception("check_goal_pace_reminder failed.")
            return None

    # ------------------------------------------------------------------
    # #117 Pain / injury
    # ------------------------------------------------------------------

    def _log_pain(self, entities: dict, now: Optional[datetime] = None) -> dict:
        raw_query = str(entities.get("raw_query") or "")
        notes = str(entities.get("notes") or raw_query).strip()
        area = str(
            entities.get("area") or entities.get("body_part")
            or entities.get("location") or _insights.guess_body_area(raw_query) or ""
        ).strip().lower()
        raw_sev = entities.get("severity")
        if raw_sev is None:
            raw_sev = entities.get("level") if entities.get("level") is not None else entities.get("intensity")
        if raw_sev is None:
            m = re.search(r"(\d{1,2})\s*(?:/|out of)\s*10", raw_query)
            raw_sev = m.group(1) if m else None

        if not area:
            return _clarify("Which part of your body is it, and how bad is it from 0 to 10?")
        sev = _extract_number(raw_sev) if raw_sev is not None else None
        if sev is None:
            return _clarify(f"How bad is the {area} pain, from 0 (none) to 10 (worst)?")
        if not (0 <= sev <= 10):
            return _clarify("Please rate it from 0 (none) to 10 (worst).")
        sev_i = int(round(sev))

        flags = _insights.red_flags(f"{raw_query} {notes}")
        try:
            self.db.log_pain(area, sev_i, notes or None)
            rows = self.db.get_pain(60, area=area)
        except Exception:
            logger.exception("_log_pain: DB operation failed.")
            return _err("I couldn't save that right now.")

        today = self._now(now).date()
        summ = next(iter(_insights.pain_summary(rows, self._tz, today)), None)
        text = f"Logged {area} pain at {sev_i}/10."
        if summ and summ["trend"] != "not_enough_data":
            text += (
                f" Over the last week it averaged {summ['avg_recent']:g} vs "
                f"{summ['avg_prior']:g} the week before ({summ['trend']})."
            )
        else:
            text += " Keep logging for a week or two and I can show you a trend."
        advice = _insights.clinician_advice(summ) if summ else None
        if flags:
            text += (
                f" What you described ({', '.join(flags)}) can be a sign of something "
                "that needs prompt medical attention — please contact a doctor or "
                "urgent care, or emergency services if it's severe or getting worse fast."
            )
        elif advice:
            text += " " + advice
        text += " I can keep track of this, but I can't diagnose it."
        return _ok(text,
                   data={"area": area, "severity": sev_i, "red_flags": flags,
                         "clinician_advised": bool(flags or advice)},
                   confidence=0.9)

    def _get_pain_trend(self, entities: dict, now: Optional[datetime] = None) -> dict:
        try:
            rows = self.db.get_pain(60)
        except Exception:
            logger.exception("_get_pain_trend: DB read failed.")
            return _err("I couldn't read your pain log right now.")
        if not rows:
            return _ok("No pain logged yet. Try \"my left knee hurts, 4 out of 10\".",
                       data={"areas": []}, confidence=0.9)
        area = str(entities.get("area") or entities.get("body_part")
                   or _insights.guess_body_area(entities.get("raw_query")) or "").lower()
        today = self._now(now).date()
        summaries = _insights.pain_summary(rows, self._tz, today)
        if area:
            summaries = [s for s in summaries if area in s["area"]] or summaries
        lines = ["Pain log (last 60 days):"]
        advice_lines = []
        for s in summaries[:6]:
            if s["trend"] == "not_enough_data":
                trend = f"not enough data for a weekly trend yet ({s['n']} entr{'y' if s['n'] == 1 else 'ies'})"
            else:
                trend = (
                    f"last 7 days averaged {s['avg_recent']:g}, previous 7 days "
                    f"{s['avg_prior']:g} — {s['trend']}"
                )
            lines.append(f"- {s['area']}: latest {s['latest']:g}/10 on {s['latest_date']}; {trend}.")
            adv = _insights.clinician_advice(s)
            if adv:
                advice_lines.append(adv)
        lines += advice_lines
        lines.append("I track what you log; I can't diagnose anything.")
        return _ok("\n".join(lines), data={"areas": summaries}, confidence=0.9)

    # ------------------------------------------------------------------
    # #119 Step import (partial: layouts documented, real exports untested)
    # ------------------------------------------------------------------

    def _import_steps(self, entities: dict) -> dict:
        folder = self._config.get("import_dir")
        if not folder:
            return _ok(
                "Step import isn't set up. Point `apollo.import_dir` in your config at "
                "a folder holding your CSV/JSON step export, then ask again.",
                data={"configured": False}, confidence=0.9,
            )
        base = Path(str(folder)).expanduser().resolve()
        if not base.is_dir():
            return _ok(f"The import folder ({base}) doesn't exist.",
                       data={"configured": True, "found": False}, confidence=0.9)

        name = entities.get("file") or entities.get("filename") or entities.get("path")
        files: list[Path] = []
        if name:
            # Basename only, resolved inside the import folder: a path from a
            # chat message must never reach outside it.
            cand = (base / Path(str(name)).name).resolve()
            if cand.parent != base or not cand.is_file():
                return _ok(f"I can't find '{Path(str(name)).name}' in the import folder.",
                           data={"found": False}, confidence=0.9)
            files = [cand]
        else:
            for p in base.iterdir():
                try:
                    rp = p.resolve()
                    if rp.parent == base and rp.is_file() and rp.suffix.lower() in (".csv", ".json"):
                        files.append(rp)
                except OSError:
                    continue
            files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:20]
        if not files:
            return _ok("No .csv or .json files in the import folder.",
                       data={"files": 0}, confidence=0.9)

        totals = Counter()
        for path in files:
            try:
                if path.stat().st_size > _MAX_IMPORT_BYTES:
                    totals["skipped_files"] += 1
                    continue
                days, skipped = _parse_steps_file(path)
            except Exception:
                logger.exception("_import_steps: failed to parse %s", path.name)
                totals["skipped_files"] += 1
                continue
            totals["skipped_rows"] += skipped
            for day, steps in days.items():
                totals[self.db.upsert_steps(day, steps, source=path.name)] += 1
        text = (
            f"Imported steps from {len(files)} file(s): {totals['inserted']} new day(s), "
            f"{totals['updated']} updated, {totals['unchanged']} unchanged."
        )
        if totals["skipped_rows"]:
            text += f" Skipped {totals['skipped_rows']} row(s) with no readable date or step count."
        if totals["skipped_files"]:
            text += f" Couldn't read {totals['skipped_files']} file(s) (too large or unrecognised layout)."
        if not (totals["inserted"] or totals["updated"] or totals["unchanged"]):
            text += " Expected a date column and a steps column (see the CHANGELOG for the layout)."
        return _ok(text, data=dict(totals), confidence=0.85)

    # ------------------------------------------------------------------
    # #126 Habit vs mood (reads Artemis history)
    # ------------------------------------------------------------------

    def _artemis_history(self) -> Optional[dict[str, dict[str, Any]]]:
        tracker = getattr(self._artemis, "tracker", None)
        if tracker is None or not hasattr(tracker, "habit_history"):
            return None
        return tracker.habit_history()

    def _habit_mood_correlation(self, entities: dict, now: Optional[datetime] = None) -> dict:
        hist = self._artemis_history()
        if hist is None:
            return _ok("I can't see your habits from here, so I can't compare them with mood.",
                       data={"status": "no_artemis"}, confidence=0.8)
        if not hist:
            return _ok("You don't have any habits tracked yet.",
                       data={"status": "no_habits"}, confidence=0.9)
        if not any(v["dates"] for v in hist.values()):
            return _ok(
                "Not enough data yet — habit completion dates only started being "
                "recorded recently. Keep completing habits and logging mood, and "
                "I'll be able to compare them after a few weeks.",
                data={"status": "no_history"}, confidence=0.9,
            )
        n_min = self._min_sample(entities)
        # Artemis buckets days in UTC, so mood is bucketed in UTC here too:
        # both sides of the pairing must use the same calendar.
        today = self._now(now).astimezone(timezone.utc).date()
        try:
            mood_days = _insights.daily_mood(self.db.get_mood(120), timezone.utc)
        except Exception:
            logger.exception("_habit_mood_correlation: DB read failed.")
            return _err("I couldn't read your mood log right now.")

        lines = ["Habits vs mood, from your own logs (high-mood day = mood scored at or above +0.5, low = at or below -0.5):"]
        reported = 0
        best_hi = best_lo = 0
        for name, info in sorted(hist.items()):
            if not info["dates"]:
                continue
            since = date.fromisoformat(info["since"]) if info["since"] else None
            res = _insights.habit_rate_by_mood(
                {date.fromisoformat(d) for d in info["dates"]}, since, mood_days, today
            )
            best_hi, best_lo = max(best_hi, res["n_high"]), max(best_lo, res["n_low"])
            if res["n_high"] < n_min or res["n_low"] < n_min:
                continue
            reported += 1
            hi, lo = res["rate_high"] * 100, res["rate_low"] * 100
            if abs(hi - lo) < 25:
                lines.append(
                    f"- {name}: no clear difference ({hi:.0f}% on high-mood days, n={res['n_high']}; "
                    f"{lo:.0f}% on low-mood days, n={res['n_low']})."
                )
            else:
                lines.append(
                    f"- {name}: kept on {hi:.0f}% of high-mood days (n={res['n_high']}) vs "
                    f"{lo:.0f}% of low-mood days (n={res['n_low']})."
                )
        if not reported:
            return _ok(
                f"Not enough data yet. I need at least {n_min} high-mood and {n_min} low-mood "
                f"days since a habit's history began; the best case so far is {best_hi} and {best_lo}.",
                data={"status": "insufficient"}, confidence=0.9,
            )
        lines.append("It's a pattern in your data, not a cause — low mood can make habits harder, and the reverse.")
        return _ok("\n".join(lines), data={"status": "ok", "habits": reported}, confidence=0.9)

    # ------------------------------------------------------------------
    # #161 Burnout signals
    # ------------------------------------------------------------------

    def _pluto_db(self) -> Optional[Any]:
        pluto = self._pluto
        if pluto is None:
            return None
        for owner in (getattr(pluto, "pf_manager", None), pluto):
            for attr in ("db", "db_manager"):
                candidate = getattr(owner, attr, None) if owner is not None else None
                if candidate is not None and hasattr(candidate, "get_expenses"):
                    return candidate
        return None

    def burnout_assessment(self, now: Optional[datetime] = None) -> dict[str, Any]:
        """Combine sleep, mood, habit consistency and spending into signals.

        Returns ``{"level": low|watch|elevated|None, "signals": [...],
        "sources": [...], "n_sources": int}``. ``level`` is None (not enough
        data) unless at least two sources had data. Signals are things worth
        a look, never a diagnosis.
        """
        n = self._now(now)
        today = n.date()
        signals: list[dict[str, Any]] = []
        sources: list[str] = []

        # Sleep (Apollo)
        sleep = self._days_of(self.db.get_sleep(9), today - timedelta(days=6), today)
        if len(sleep) >= 3:
            sources.append("sleep")
            avg = sum(r["hours"] for _, r in sleep) / len(sleep)
            if avg < _LOW_SLEEP_THRESHOLD:
                signals.append({"source": "sleep", "weight": 1.0,
                                "text": f"sleep averaged {avg:.1f} h over {len(sleep)} nights this week"})
            scored = [s["score"] for s in self._sleep_scores(list(reversed(self.db.get_sleep(9))))
                      if s["score"] is not None]
            if len(scored) >= 3 and sum(scored[-7:]) / len(scored[-7:]) < 50 and avg >= _LOW_SLEEP_THRESHOLD:
                signals.append({"source": "sleep", "weight": 0.5,
                                "text": "sleep quality scores have been low even if duration looks okay"})

        # Mood (Apollo)
        mood_days = _insights.daily_mood(self.db.get_mood(9), self._tz)
        recent = [v for d, v in mood_days.items() if today - timedelta(days=6) <= d <= today]
        if len(recent) >= 3:
            sources.append("mood")
            m = sum(recent) / len(recent)
            if m <= -1.0:
                signals.append({"source": "mood", "weight": 1.5,
                                "text": f"logged mood has been low on most days (average {m:+.1f} on a -2 to +2 scale)"})
            elif m <= -0.5:
                signals.append({"source": "mood", "weight": 1.0,
                                "text": f"logged mood has leaned low this week (average {m:+.1f})"})

        # Habit consistency (Artemis)
        hist = self._artemis_history() or {}
        drops = []
        eligible = 0
        for name, info in hist.items():
            if not (info["dates"] and info["since"]):
                continue
            since = date.fromisoformat(info["since"])
            if (today - since).days < 14:
                continue
            eligible += 1
            days = {date.fromisoformat(d) for d in info["dates"]}
            last7 = sum(1 for i in range(7) if today - timedelta(days=i) in days)
            prev7 = sum(1 for i in range(7, 14) if today - timedelta(days=i) in days)
            if prev7 >= 3 and last7 <= prev7 / 2:
                drops.append(name)
        if eligible:
            sources.append("habits")
            if drops:
                signals.append({"source": "habits", "weight": 1.0,
                                "text": "habit completions dropped by half or more versus the week before ("
                                        + ", ".join(sorted(drops)[:3]) + ")"})

        # Spending (Pluto) — weak proxy
        pdb = self._pluto_db()
        if pdb is not None:
            try:
                spike = _insights.weekly_spend_spike(pdb.get_expenses(500), today, self._tz)
            except Exception:
                logger.exception("burnout_assessment: Pluto read failed.")
                spike = None
            if spike:
                sources.append("spending")
                if spike["spike"]:
                    signals.append({"source": "spending", "weight": 0.5,
                                    "text": f"spending this week is {spike['ratio']:.1f}x your recent weekly average "
                                            "(a weak signal — spending rises for plenty of ordinary reasons)"})

        level: Optional[str]
        total = sum(s["weight"] for s in signals)
        if len(sources) < 2:
            level = None
        elif total >= 2.5:
            level = "elevated"
        elif total >= 1.0:
            level = "watch"
        else:
            level = "low"
        return {"level": level, "signals": signals, "sources": sources,
                "n_sources": len(sources), "score": total}

    def _burnout_text(self, res: dict[str, Any], sustained: bool = False) -> str:
        if res["level"] is None:
            return (
                "I can't say much yet — I need data from at least two areas (sleep, mood, "
                "habits, spending) in the past week, and I only have "
                f"{res['n_sources']}. Keep logging and check again."
            )
        head = {
            "low": "Nothing stands out this week across "
                   + ", ".join(res["sources"]) + ".",
            "watch": "A couple of signals are worth a look this week:",
            "elevated": "Several signals are pointing the same way this week:",
        }[res["level"]]
        lines = [head]
        lines += [f"- {s['text']}" for s in res["signals"]]
        lines.append(
            "These are patterns in what you've logged, not a diagnosis, and they "
            "can have ordinary explanations."
        )
        if sustained:
            lines.append(
                "This has looked elevated for several weeks running. It could be worth "
                "talking it over with someone you trust or a health professional."
            )
        return "\n".join(lines)

    def _burnout_check(self, entities: dict, now: Optional[datetime] = None) -> dict:
        try:
            res = self.burnout_assessment(now)
        except Exception:
            logger.exception("_burnout_check failed.")
            return _err("I couldn't put that together right now.")
        return _ok(self._burnout_text(res), data=res, confidence=0.85)

    def check_burnout(self, now: Optional[datetime] = None) -> Optional[str]:
        """Heartbeat hook: weekly; only speaks up for watch/elevated."""
        bc = self._cfg_section("burnout")
        if bc.get("enabled") is False:
            return None
        try:
            n = self._now(now)
            if not _schedule.weekly_due(n, self.db.get_state("burnout_checked"),
                                        weekday=int(bc.get("weekday", 6)),
                                        hour=int(bc.get("hour", 19))):
                return None
            res = self.burnout_assessment(n)
            self.db.set_state("burnout_checked", _schedule.iso_week_key(n.date()))
            try:
                history = json.loads(self.db.get_state("burnout_history") or "[]")
            except ValueError:
                history = []
            history = (history + [res["level"]])[-4:]
            self.db.set_state("burnout_history", json.dumps(history))
            if res["level"] not in ("watch", "elevated"):
                return None
            sustained = len(history) >= 3 and all(h == "elevated" for h in history[-3:])
            return "Weekly check-in: " + self._burnout_text(res, sustained)
        except Exception:
            logger.exception("check_burnout failed.")
            return None

    # ------------------------------------------------------------------
    # #149 / #159 hooks: read-only mood and rest signals for other modules
    # ------------------------------------------------------------------

    def recent_mood_context(
        self, now: Optional[datetime] = None, hours: int = 48
    ) -> Optional[dict[str, Any]]:
        """Latest logged mood (within *hours*) and whether recent moods are low.

        Used by Dionysus so recommendations can reflect how the user has said
        they feel, only when they gave no mood of their own.
        """
        try:
            rows = self.db.get_mood(7)
        except Exception:
            return None
        if not rows:
            return None
        n = self._now(now)
        latest = rows[0]
        ts = _insights.parse_ts(latest.get("logged_at"))
        if ts is None or (n - ts.astimezone(self._tz)) > timedelta(hours=hours):
            return None
        scores = [s for s in (_insights.mood_score(r["mood"]) for r in rows[:5]) if s is not None]
        low_trend = len(scores) >= 3 and sum(1 for s in scores if s < 0) >= 3
        return {
            "mood": str(latest["mood"]).strip(),
            "score": _insights.mood_score(latest["mood"]),
            "low_trend": low_trend,
        }

    def rest_signals(self, now: Optional[datetime] = None) -> list[dict[str, Any]]:
        """Signals that say 'rest' (weighted), for the consensus layer (#159)."""
        n = self._now(now)
        today = n.date()
        out: list[dict[str, Any]] = []
        try:
            sleep = self._days_of(self.db.get_sleep(5), today - timedelta(days=2), today)
            if sleep:
                avg = sum(r["hours"] for _, r in sleep) / len(sleep)
                if avg < _LOW_SLEEP_THRESHOLD:
                    out.append({"direction": "rest", "weight": 1.0, "source": "apollo",
                                "reason": f"you've averaged {avg:.1f}h of sleep over the last few nights"})
            ctx = self.recent_mood_context(n)
            if ctx and ctx["score"] is not None:
                if ctx["score"] <= -1:
                    out.append({"direction": "rest", "weight": 1.0, "source": "apollo",
                                "reason": f"your latest logged mood was '{ctx['mood']}'"})
                elif ctx["low_trend"]:
                    out.append({"direction": "rest", "weight": 0.5, "source": "apollo",
                                "reason": "your last few logged moods have leaned low"})
        except Exception:
            logger.exception("rest_signals failed.")
        return out

    # ------------------------------------------------------------------
    # #183 Dashboard data + Moods tab rows
    # ------------------------------------------------------------------

    def mood_entries(self, days: int = 60) -> list[dict[str, Any]]:
        """Mood rows enriched with a score/valence and the field names the
        Moods tab reads (``timestamp``, ``valence``)."""
        out = []
        for row in self.db.get_mood(days):
            score = _insights.mood_score(row.get("mood"))
            out.append({**row, "timestamp": row.get("logged_at"),
                        "score": score, "valence": _insights.mood_valence(score)})
        return out

    def dashboard_data(self, days: int = 30, now: Optional[datetime] = None) -> dict[str, Any]:
        """All the series the Health tab charts, keyed by section."""
        days = max(7, min(int(days), 180))
        today = self._now(now).date()
        start = today - timedelta(days=days - 1)
        unit = self._weight_unit()

        sleep_asc = list(reversed(self.db.get_sleep(days + 7)))
        sleep = []
        for s in self._sleep_scores(sleep_asc):
            d = _insights.local_date(s["row"].get("logged_at"), self._tz)
            if d is not None and start <= d <= today:
                sleep.append({"date": d.isoformat(), "hours": s["row"]["hours"],
                              "score": s["score"], "bed_time": s["row"].get("bed_time"),
                              "wake_time": s["row"].get("wake_time")})

        weight_by_day: dict[date, float] = {}
        for row in sorted(self.db.get_weight(days), key=lambda r: r["logged_at"]):
            d = _insights.local_date(row["logged_at"], self._tz)
            if d is not None:
                weight_by_day[d] = row["weight_kg"]
        weight = [{"date": d.isoformat(), "value": round(_kg_to_unit(v, unit), 2)}
                  for d, v in sorted(weight_by_day.items())]

        goal_row = self.db.get_goal("water_ml")
        goal = float(goal_row["target_value"]) if goal_row else _schedule.DEFAULT_WATER_GOAL_ML
        water = [{"date": d.isoformat(), "ml": round(v), "goal_ml": goal}
                 for d, v in sorted(self._water_by_day(days).items()) if start <= d <= today]

        mood_days = _insights.daily_mood(self.db.get_mood(days), self._tz)
        mood = [{"date": d.isoformat(), "score": round(v, 2)}
                for d, v in sorted(mood_days.items()) if start <= d <= today]

        streaks = _insights.type_streaks(
            self.db.get_workouts(400), self._tz, today, _DEFAULT_STREAK_MIN)
        steps = [{"date": r["day"], "steps": r["steps"]}
                 for r in reversed(self.db.get_steps(days))]
        return {
            "days": days, "units": {"weight": unit, "water": self._water_unit()},
            "sleep": sleep, "weight": weight, "water": water, "mood": mood,
            "streaks": streaks, "steps": steps,
        }

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

        Optional bed/wake times ("slept 11pm to 6:30am") add a consistency
        component to the 0-100 sleep score (#111). When they're missing the
        reply says so instead of quietly scoring on less.
        """
        raw_query: str = entities.get("raw_query") or ""
        bed, wake = self._sleep_times(entities, raw_query)

        raw_hours = entities.get("hours") or entities.get("duration")
        if raw_hours is None and bed and wake:
            raw_hours = _insights.window_hours(bed, wake)
        if raw_hours is None:
            if _SLEEP_QUERY_RE.search(raw_query):
                return self._report_last_sleep()
            return _clarify(
                "How many hours did you sleep, and how was the quality? "
                "(You can also say e.g. \"11pm to 6:30am\".)"
            )

        hours, err = _parse_hours(raw_hours)
        if err:
            return _clarify(err)

        quality: str = (entities.get("quality") or "").strip()
        notes: str = (entities.get("notes") or raw_query).strip()
        rating = _parse_rating(entities.get("rating"))

        try:
            self.db.log_sleep(
                hours, quality, notes,
                bed_time=bed, wake_time=wake, rating=rating,
            )
            avg = self.db.avg_sleep(_HEALTH_SUMMARY_DAYS)
            recent_asc = list(reversed(self.db.get_sleep(_HEALTH_SUMMARY_DAYS)))
        except Exception:
            logger.exception("_track_sleep: DB operation failed.")
            return _err("I couldn't save your sleep data right now.")

        comment = _sleep_comment(hours)
        avg_str = f" Your {_HEALTH_SUMMARY_DAYS}-day average is {avg:.1f} hours." if avg else ""
        score = self._score_latest(recent_asc)

        score_str = ""
        if score and score["score"] is not None:
            parts = ", ".join(f"{k} {v}" for k, v in score["components"].items())
            score_str = f" Sleep quality score: {score['score']}/100 ({parts})."
            if not (bed and wake):
                score_str += (
                    " I didn't have bed and wake times for this one, so "
                    "consistency is only counted from nights where you gave them."
                )
            elif "consistency" in score["missing"]:
                score_str += (
                    " Consistency needs bed/wake times from at least 3 nights "
                    "in the past week."
                )

        logger.info("Sleep logged: hours=%.1f quality=%r.", hours, quality)
        return _ok(
            f"Sleep logged — {hours} hours.{avg_str} {comment}{score_str}",
            data={
                "hours": hours, "quality": quality, "avg_7d": avg,
                "bed_time": bed, "wake_time": wake, "rating": rating,
                "quality_score": score["score"] if score else None,
            },
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

        Units (#113): an explicit unit in the message wins; otherwise the
        profile's unit is used. Storage is always kg.
        """
        raw_weight = entities.get("weight") or entities.get("value")
        if raw_weight is None:
            raw_query: str = entities.get("raw_query") or ""
            if _WEIGHT_QUERY_RE.search(raw_query):
                return self._report_weight_trend()
            return _clarify(
                f"What's your current weight? (I'll assume {self._weight_unit()} "
                "unless you say otherwise.)"
            )

        unit = (
            _explicit_weight_unit(entities.get("unit"))
            or _explicit_weight_unit(entities.get("raw_query"))
            or self._weight_unit()
        )
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

        unit = self._weight_unit()
        latest = recent[0]["weight_kg"]
        logged_at = (recent[0].get("logged_at") or "")[:10]
        show = lambda kg: f"{_kg_to_unit(kg, unit):.1f} {unit}"  # noqa: E731

        if len(recent) == 1:
            return _ok(
                f"Your last logged weight was {show(latest)} on {logged_at}. "
                "Log a few more entries and I can show you a trend.",
                data={"latest_kg": latest, "logged_at": logged_at},
                confidence=0.85,
            )

        oldest = recent[-1]["weight_kg"]
        delta = latest - oldest
        trend = _trend_word(delta)
        return _ok(
            f"Your weight is {trend} {_kg_to_unit(abs(delta), unit):.1f} {unit} over the last "
            f"{_WEIGHT_TREND_DAYS} days — {show(oldest)} to {show(latest)}. "
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

        Units (#113): an explicit unit wins, otherwise the profile's water
        unit applies. Storage stays in ml.
        """
        raw_amount = entities.get("amount") or entities.get("ml")
        glasses = entities.get("glasses")

        if raw_amount is None and glasses is None:
            raw_query: str = entities.get("raw_query") or ""
            if _WATER_QUERY_RE.search(raw_query):
                return self._report_water_today()
            return _clarify("How much water did you drink (ml, oz or glasses)?")

        if raw_amount is None and glasses is not None:
            raw_amount = glasses
            unit_hint = "glasses"
        else:
            unit_hint = (
                (entities.get("unit") or "").strip().lower()
                or _explicit_water_unit(entities.get("raw_query"))
                or self._water_unit()
            )

        amount_ml, err = _parse_water(raw_amount, unit_hint)
        if err:
            return _clarify(err)

        unit = self._water_unit()
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
            goal_str = (
                f" That's {pct}% of your "
                f"{_fmt_water(goal['target_value'], unit)} goal."
            )

        logger.info("Water logged: %d ml (today total=%d).", amount_ml, total_today)
        return _ok(
            f"Water logged — {_fmt_water(amount_ml, unit)}. "
            f"Today's total: {_fmt_water(total_today, unit)}.{goal_str}",
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

        unit = self._water_unit()
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
                f" You're {_fmt_water(remaining, unit)} short of your "
                f"{_fmt_water(goal['target_value'], unit)} goal."
                if remaining > 0
                else " You've hit your daily goal!"
            )

        return _ok(
            f"You've had {_fmt_water(total_today, unit)} of water today.{goal_str}",
            data={"total_today_ml": total_today},
            confidence=0.9,
        )

    def _set_health_goal(self, entities: dict) -> dict:
        """Validate and persist a target for one of the tracked health metrics.

        Weight and water targets are read in the user's unit (explicit unit
        in the message, else the profile's) and stored in kg / ml. A weight
        goal remembers the weight at the time it was set (the baseline for
        goal-pace maths, #120) and can carry an optional deadline. Calorie
        goals are only ever set by the user, and never below a safe floor.
        """
        raw_query = entities.get("raw_query") or ""
        raw_type = (
            entities.get("goal_type") or entities.get("metric") or raw_query or ""
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

        deadline, derr = self._parse_deadline(entities)
        if derr:
            return _clarify(derr)

        stored = target
        shown = f"{target:g} {_GOAL_LABELS[goal_type]}"
        start_value: Optional[float] = None

        if goal_type == "calories_kcal":
            if target < _MIN_CALORIE_GOAL:
                return _ok(
                    f"I won't set a daily target below {_MIN_CALORIE_GOAL:,} kcal — "
                    "that's too low to be a safe thing to aim for without "
                    "professional guidance. If you're planning to change how "
                    "you eat, a doctor or registered dietitian can set a "
                    "target that fits you.",
                    data={"goal_type": goal_type, "refused": True},
                    confidence=0.9,
                )
        elif goal_type == "weight_target":
            unit = (
                _explicit_weight_unit(entities.get("unit"))
                or _explicit_weight_unit(raw_query)
                or self._weight_unit()
            )
            stored = target * _KG_PER_LB if unit == "lb" else target
            if not (_MIN_WEIGHT_KG <= stored <= _MAX_WEIGHT_KG):
                return _clarify(
                    "That target doesn't look right for a body weight. "
                    "What weight are you aiming for?"
                )
            shown = f"{target:g} {unit} target weight"
            try:
                latest = self.db.latest_weight()
                start_value = latest["weight_kg"] if latest else None
            except Exception:
                start_value = None
        elif goal_type == "water_ml":
            unit = (
                _explicit_water_unit(entities.get("unit"))
                or _explicit_water_unit(raw_query)
                or self._water_unit()
            )
            if unit == "oz":
                stored = target * _ML_PER_OZ
                shown = f"{target:g} oz (about {stored:.0f} ml) of water/day"

        try:
            self.db.set_goal(goal_type, stored, start_value=start_value,
                             deadline=deadline)
        except Exception:
            logger.exception("_set_health_goal: DB operation failed.")
            return _err("I couldn't save that goal right now.")

        deadline_str = f" Deadline: {deadline}." if deadline else ""
        logger.info("Health goal set: type=%s target=%.2f.", goal_type, stored)
        return _ok(
            f"Goal set — {shown}.{deadline_str} I'll track your progress toward this.",
            data={"goal_type": goal_type, "target": stored, "deadline": deadline},
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
            elif goal_type == "calories_kcal":
                meals = self._meals_on(self._now().date())
                current = float(sum(m.get("kcal") or 0 for m in meals)) if meals else None
        except Exception:
            logger.exception("_format_goal_progress: metric lookup failed for %s.", goal_type)

        if goal_type == "weight_target":
            unit = self._weight_unit()
            label = f"{unit} target weight"
            t_show = _kg_to_unit(target, unit)
        else:
            unit, t_show = "kg", target

        if current is None:
            return f"{label}: target {t_show:g}, no data logged yet", {
                "goal_type": goal_type, "target": target, "current": None,
            }

        if goal_type == "weight_target":
            c_show = _kg_to_unit(current, unit)
            remaining = abs(c_show - t_show)
            direction = "to lose" if current > target else "to gain"
            line = f"{label}: currently {c_show:.1f} {unit}, {remaining:.1f} {unit} {direction}"
        elif goal_type == "calories_kcal":
            line = (
                f"{label}: {current:g} logged today of your {target:g} goal "
                "(only counts what you've logged)"
            )
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

    def _lookup_food(self, entities: dict) -> dict:
        """
        Look up a food item's nutrition facts via Open Food Facts (free,
        no key). New intent: `lookup_food`. Purely informational — does
        not write to the DB (use `log_health`/`log_water` etc. for that).
        """
        query: str = (
            entities.get("food") or entities.get("item") or entities.get("raw_query") or ""
        ).strip()
        if not query:
            return _ok("What food would you like nutrition info for?", confidence=0.5)

        try:
            results = _fa_food_lookup(query, limit=3)
        except FreeAPIError:
            logger.exception("_lookup_food: Open Food Facts request failed.")
            return _err("I couldn't reach the food database right now — try again shortly.")

        if not results:
            return _ok(f"No nutrition data found for {query!r}.", confidence=0.6)

        lines = [f"Nutrition info for {query!r} (per 100g):"]
        for r in results:
            cal = r.get("calories_kcal_100g")
            protein = r.get("protein_g_100g")
            sugar = r.get("sugar_g_100g")
            bits = [r["name"]]
            if r.get("brand"):
                bits.append(f"({r['brand']})")
            lines.append("  " + " ".join(bits))
            detail = []
            if cal is not None:
                detail.append(f"{cal:g} kcal")
            if protein is not None:
                detail.append(f"{protein:g}g protein")
            if sugar is not None:
                detail.append(f"{sugar:g}g sugar")
            if detail:
                lines.append("    " + ", ".join(detail))

        return _ok("\n".join(lines), data={"query": query, "results": results}, confidence=0.85)

    def _suggest_exercise(self, entities: dict) -> dict:
        """
        Suggest exercises matching a muscle group / keyword via wger's
        public exercise database (free, no key). New intent:
        `suggest_exercise`. Complements `_log_workout`, which records
        what was already done.
        """
        query: str = (
            entities.get("muscle_group") or entities.get("query")
            or entities.get("raw_query") or "full body"
        ).strip()

        try:
            results = _fa_exercise_lookup(query, limit=5)
        except FreeAPIError:
            logger.exception("_suggest_exercise: wger request failed.")
            return _err("I couldn't reach the exercise database right now — try again shortly.")

        if not results:
            return _ok(f"No exercises found for {query!r}. Try a broader term like 'legs' or 'back'.",
                        confidence=0.6)

        lines = [f"Exercises for {query!r}:"]
        lines.extend(f"  - {r['name']}" for r in results if r.get("name"))

        return _ok("\n".join(lines), data={"query": query, "results": results}, confidence=0.85)


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


# ---------------------------------------------------------------------------
# Units, ratings and step-file parsing (module-level helpers)
# ---------------------------------------------------------------------------

_WEIGHT_TOKEN_UNITS: dict[str, str] = {
    "kg": "kg", "kgs": "kg", "kilo": "kg", "kilos": "kg",
    "kilogram": "kg", "kilograms": "kg",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
}
_WATER_TOKEN_UNITS: dict[str, str] = {
    "ml": "ml", "milliliter": "ml", "milliliters": "ml",
    "millilitre": "ml", "millilitres": "ml",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
}


def _explicit_unit(text: Any, table: dict[str, str]) -> Optional[str]:
    """The first recognised unit word in *text*, or None. Unlike the older
    ``_normalize_weight_unit`` this never defaults, so callers can tell an
    explicit unit (which wins) from silence (which falls back to the profile)."""
    for tok in re.findall(r"[a-z]+", str(text or "").lower()):
        if tok in table:
            return table[tok]
    return None


def _explicit_weight_unit(text: Any) -> Optional[str]:
    return _explicit_unit(text, _WEIGHT_TOKEN_UNITS)


def _explicit_water_unit(text: Any) -> Optional[str]:
    return _explicit_unit(text, _WATER_TOKEN_UNITS)


def _fmt_water(ml: float, unit: str) -> str:
    if unit == "oz":
        return f"{ml / _ML_PER_OZ:.0f} oz"
    return f"{ml:.0f} ml"


def _parse_rating(raw: Any) -> Optional[int]:
    num = _extract_number(raw) if raw is not None else None
    if num is None or not (1 <= num <= 5):
        return None
    return int(round(num))


def _norm_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(key).lower()).strip()


_STEP_DATE_KEYS = [_norm_key(k) for k in (
    "date", "day", "start_date", "startdate", "start time", "start_time",
    "timestamp", "datetime", "time", "end_date", "enddate",
)]
_STEP_COUNT_KEYS = [_norm_key(k) for k in (
    "steps", "step_count", "step count", "stepcount", "total steps",
    "steps (count)", "count", "value",
)]
_MAX_DAILY_STEPS = 200_000


def _pick_key(keys: list[str], candidates: list[str]) -> Optional[str]:
    for cand in candidates:
        if cand in keys:
            return cand
    return None


def _parse_day(value: Any) -> Optional[date]:
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()
                                           and len(value.strip()) >= 10):
        try:
            v = float(value)
            v = v / 1000.0 if v > 1e11 else v
            return datetime.fromtimestamp(v, timezone.utc).date()
        except (ValueError, OverflowError, OSError):
            return None
    text = str(value or "").strip()
    m = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    for fmt in ("%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _parse_step_value(value: Any) -> Optional[int]:
    try:
        num = float(str(value).replace(",", "").strip())
    except ValueError:
        return None
    if num != num or num < 0 or num > _MAX_DAILY_STEPS:
        return None
    return int(round(num))


def _json_records(obj: Any) -> list[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("steps", "data", "records", "days", "values", "items"):
            val = obj.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val
        if obj and all(
            _parse_day(k) is not None and isinstance(v, (int, float, str))
            for k, v in obj.items()
        ):
            return [{"date": k, "steps": v} for k, v in obj.items()]
    return []


def _parse_steps_file(path: Path) -> tuple[dict[str, int], int]:
    """Read a CSV/JSON step export -> ({YYYY-MM-DD: steps}, skipped_rows).

    Recognised layout: one record per row/object with a date column
    (date, day, start_date, timestamp, ...) and a steps column (steps,
    step_count, count, value, ...). Several records on one day are summed
    (per-record exports); a daily-total file has one per day so the sum is
    the total. Column names are matched case- and punctuation-insensitively.
    """
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() == ".json":
        records = _json_records(json.loads(text))
    else:
        try:
            dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        records = list(csv.DictReader(text.splitlines(), dialect=dialect))

    totals: dict[str, int] = defaultdict(int)
    skipped = 0
    for rec in records:
        norm = {_norm_key(k): v for k, v in rec.items() if k is not None}
        dk = _pick_key(list(norm), _STEP_DATE_KEYS)
        sk = _pick_key(list(norm), _STEP_COUNT_KEYS)
        day = _parse_day(norm.get(dk)) if dk else None
        steps = _parse_step_value(norm.get(sk)) if sk else None
        if day is None or steps is None:
            skipped += 1
            continue
        totals[day.isoformat()] += steps
    return {d: min(v, _MAX_DAILY_STEPS) for d, v in totals.items()}, skipped
