import datetime
import re
import requests
import dateparser
from modules.base import BaseModule


class ChronosEngine(BaseModule):
    name = "chronos"

    _CITY_COORDS = {
        "mumbai":    (19.0760,  72.8777),
        "delhi":     (28.6139,  77.2090),
        "bangalore": (12.9716,  77.5946),
        "bengaluru": (12.9716,  77.5946),
        "chennai":   (13.0827,  80.2707),
        "kolkata":   (22.5726,  88.3639),
        "hyderabad": (17.3850,  78.4867),
        "pune":      (18.5204,  73.8567),
        "london":    (51.5074,  -0.1278),
        "new york":  (40.7128, -74.0060),
        "tokyo":     (35.6762, 139.6503),
    }

    def __init__(self, memory=None):
        self._memory = memory

    def can_handle(self, intent: str) -> bool:
        return intent in {"get_time", "get_date", "get_weather", "set_reminder"}

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        if intent == "get_time":
            return self._time()
        elif intent == "get_date":
            return self._date()
        elif intent == "get_weather":
            return self._weather(entities, context)
        elif intent == "set_reminder":
            return self._reminder(entities)
        return {"response": "Unknown time request.", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        now = datetime.datetime.now()
        return {
            "current_time": now.strftime("%I:%M %p"),
            "current_date": now.strftime("%A, %B %d, %Y"),
            "hour": now.hour,
            "day_of_week": now.strftime("%A"),
        }

    # ── BASIC FUNCTIONS ─────────────────────────────

    def _time(self) -> dict:
        now = datetime.datetime.now()
        return {
            "response": f"It's {now.strftime('%I:%M %p')}.",
            "data": {"time": now.isoformat()},
            "confidence": 1.0,
        }

    def _date(self) -> dict:
        now = datetime.datetime.now()
        return {
            "response": f"Today is {now.strftime('%A, %B %d, %Y')}.",
            "data": {"date": now.date().isoformat()},
            "confidence": 1.0,
        }

    def _weather(self, entities: dict, context: dict) -> dict:
        location = "Mumbai"
        if self._memory:
            location = self._memory.get_preference("location", "Mumbai")

        lat, lon = self._CITY_COORDS.get(location.lower(), (19.0760, 72.8777))

        try:
            resp = requests.get(
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}&current_weather=true&wind_speed_unit=kmh",
                timeout=10,
            )
            resp.raise_for_status()
            w = resp.json().get("current_weather", {})

            return {
                "response": f"Currently in {location}: {w.get('temperature', '?')}°C, wind {w.get('windspeed', '?')} km/h.",
                "data": w,
                "confidence": 0.95,
            }

        except Exception:
            return {
                "response": "I couldn't fetch the weather right now.",
                "data": {},
                "confidence": 0.0,
            }

    # ── FINAL REMINDER ENGINE ───────────────────────

    def _reminder(self, entities: dict) -> dict:
        raw = entities.get("raw_query", "")
        task = entities.get("task")
        date_str = entities.get("date")
        time_str = entities.get("time")

        # ── STEP 1: Extract task (single logic) ───────
        if not task or str(task).strip().lower() in {"none", "null", ""}:

            match = re.search(r"\bto (.+)", raw)
            if match:
                task = match.group(1).strip()
            else:
                cleaned = re.sub(
                    r"\b(remind me|at|in|on|tomorrow|today|tonight|morning|evening|night)\b.*",
                    "",
                    raw,
                    flags=re.IGNORECASE
                ).strip()
                task = cleaned if cleaned else "your task"

                # prevent broken tokens like "rem"
                if len(task) < 3:
                    task = "your task"

        if not task or task.strip() == "":
            task = "your task"

        # ── STEP 2: Parse time ───────────────────────
        parsed_time = dateparser.parse(
            raw,
            settings={
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": datetime.datetime.now()
            }
        )

        if not parsed_time and time_str:
            parsed_time = dateparser.parse(
                f"{date_str or ''} {time_str}",
                settings={"PREFER_DATES_FROM": "future"}
            )

        # manual fallback
        if not parsed_time and time_str:
            now = datetime.datetime.now()

            if "morning" in time_str:
                parsed_time = now.replace(hour=9, minute=0) + datetime.timedelta(days=1)
            elif "evening" in time_str:
                parsed_time = now.replace(hour=18, minute=0) + datetime.timedelta(days=1)
            elif "night" in time_str:
                parsed_time = now.replace(hour=21, minute=0)

        # ── STEP 3: Validate ─────────────────────────
        if not parsed_time:
            return {"response": "Couldn't understand the time.", "confidence": 0.6}

        due_iso = parsed_time.replace(tzinfo=None).isoformat()

        if self._memory:
            self._memory.add_reminder(task, due_iso)

        return {
            "response": f"Reminder set for {task}.",
            "confidence": 0.95
        }