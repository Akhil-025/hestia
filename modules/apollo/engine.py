# modules/apollo/engine.py

import json
import logging
import os
from pathlib import Path
from typing import Optional

from core.ollama_client import generate
from modules.base import BaseModule
from .db import ApolloDB

log = logging.getLogger(__name__)

_DB_PATH = str(Path(__file__).parent.parent.parent / "data" / "apollo" / "apollo.db")

# ── Prompts ───────────────────────────────────────────────────────────────────

_WORKOUT_FEEDBACK_PROMPT = """You are Apollo, a health coach.
The user just logged a workout:
  Type    : {type_}
  Duration: {duration} minutes
  Notes   : {notes}
  Workouts this week: {count}

Give a short (2-3 sentence) motivating response. Mention their weekly count.
Be warm and specific. No bullet points."""

_MOOD_RESPONSE_PROMPT = """You are Apollo, a compassionate health assistant.
The user logged their mood as: {mood}
Recent mood history (newest first): {history}

Write a 2-3 sentence empathetic response.
If there is a negative trend (3+ low moods), gently acknowledge it and suggest one small action.
If positive, celebrate it briefly.
Be human and warm. No bullet points."""

_HEALTH_SUMMARY_PROMPT = """You are Apollo, a personal health coach.
Here is the user's health data for the last 7 days:

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


class ApolloEngine(BaseModule):
    name = "apollo"
    _INTENTS = {
        "log_workout",
        "track_sleep",
        "log_mood",
        "log_health",
        "get_health_summary",
    }

    def __init__(self, ollama_cfg: dict = None):
        self._ollama = ollama_cfg or {}
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        self.db = ApolloDB(_DB_PATH)

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        if intent == "log_workout":
            return self._log_workout(entities)
        if intent in ("track_sleep", "log_health"):
            return self._track_sleep(entities)
        if intent == "log_mood":
            return self._log_mood(entities)
        if intent == "get_health_summary":
            return self._health_summary()
        return {"response": "Unknown Apollo intent.", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        try:
            return {
                "apollo_workouts_this_week": self.db.workout_count(7),
                "apollo_avg_sleep":          self.db.avg_sleep(7),
                "apollo_recent_moods":       self.db.recent_moods(3),
            }
        except Exception:
            return {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ollama_text(self, prompt: str) -> str:
        return generate(
            prompt,
            model=self._ollama.get("model", "mistral"),
            host=self._ollama.get("host", "127.0.0.1"),
            port=self._ollama.get("port", 11434),
        )

    # ── log_workout ───────────────────────────────────────────────────────────

    def _log_workout(self, entities: dict) -> dict:
        type_    = entities.get("type") or entities.get("workout_type") or "general"
        duration = entities.get("duration") or entities.get("minutes") or 30
        notes    = entities.get("notes") or entities.get("raw_query", "")

        try:
            duration = int(duration)
        except (ValueError, TypeError):
            duration = 30

        self.db.log_workout(type_, duration, notes)
        count = self.db.workout_count(7)

        feedback = self._ollama_text(
            _WORKOUT_FEEDBACK_PROMPT.format(
                type_=type_,
                duration=duration,
                notes=notes,
                count=count,
            )
        )

        return {
            "response": feedback.strip() or f"Workout logged. That's {count} sessions this week.",
            "data": {"type": type_, "duration": duration, "weekly_count": count},
            "confidence": 0.95,
        }

    # ── track_sleep ───────────────────────────────────────────────────────────

    def _track_sleep(self, entities: dict) -> dict:
        hours   = entities.get("hours") or entities.get("duration")
        quality = entities.get("quality", "")
        notes   = entities.get("notes") or entities.get("raw_query", "")

        if not hours:
            return {
                "response": "How many hours did you sleep, and how was the quality?",
                "data": {},
                "confidence": 0.5,
            }

        try:
            hours = float(hours)
        except (ValueError, TypeError):
            return {"response": "I didn't catch the sleep duration.", "data": {}, "confidence": 0.4}

        self.db.log_sleep(hours, quality, notes)
        avg = self.db.avg_sleep(7)

        if hours < 6:
            comment = "That's below the recommended 7-8 hours. Try to get to bed earlier tonight."
        elif hours >= 8:
            comment = "Great sleep! Consistent rest like this makes a real difference."
        else:
            comment = "Decent sleep. Aim for 8 hours when you can."

        avg_str = f" Your 7-day average is {avg} hours." if avg else ""

        return {
            "response": f"Sleep logged — {hours} hours.{avg_str} {comment}",
            "data": {"hours": hours, "quality": quality, "avg_7d": avg},
            "confidence": 0.95,
        }

    # ── log_mood ──────────────────────────────────────────────────────────────

    def _log_mood(self, entities: dict) -> dict:
        mood  = entities.get("mood") or entities.get("raw_query", "neutral")
        notes = entities.get("notes", "")

        self.db.log_mood(mood, notes)
        history = self.db.recent_moods(5)

        response = self._ollama_text(
            _MOOD_RESPONSE_PROMPT.format(
                mood=mood,
                history=", ".join(history) if history else "none",
            )
        )

        return {
            "response": response.strip() or f"Mood logged as {mood}.",
            "data": {"mood": mood, "recent_history": history},
            "confidence": 0.95,
        }

    # ── get_health_summary ────────────────────────────────────────────────────

    def _health_summary(self) -> dict:
        workouts = self.db.get_workouts(7)
        sleep    = self.db.get_sleep(7)
        moods    = self.db.get_mood(7)
        avg_sleep= self.db.avg_sleep(7) or 0

        if not workouts and not sleep and not moods:
            return {
                "response": (
                    "No health data logged yet. "
                    "Start by telling me about your workout, sleep, or mood."
                ),
                "data": {},
                "confidence": 0.9,
            }

        # format detail blocks
        workout_detail = "\n".join(
            f"  {w['logged_at'][:10]} — {w['type']} ({w['duration']} min)"
            for w in workouts
        ) or "  None logged"

        sleep_detail = "\n".join(
            f"  {s['logged_at'][:10]} — {s['hours']}h  {s['quality'] or ''}"
            for s in sleep
        ) or "  None logged"

        mood_detail = "\n".join(
            f"  {m['logged_at'][:10]} — {m['mood']}"
            for m in moods
        ) or "  None logged"

        summary = self._ollama_text(
            _HEALTH_SUMMARY_PROMPT.format(
                workout_count=len(workouts),
                workout_detail=workout_detail,
                avg_sleep=avg_sleep,
                sleep_detail=sleep_detail,
                mood_detail=mood_detail,
            )
        )

        response = (
            f"Health Summary — Last 7 Days\n\n"
            f"WORKOUTS  : {len(workouts)} sessions\n"
            f"AVG SLEEP : {avg_sleep}h/night\n"
            f"MOOD LOGS : {len(moods)} entries\n\n"
            f"ANALYSIS\n{summary.strip()}"
        )

        return {
            "response": response,
            "data": {
                "workout_count": len(workouts),
                "avg_sleep": avg_sleep,
                "mood_count": len(moods),
            },
            "confidence": 0.95,
        }