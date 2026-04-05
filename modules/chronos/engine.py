# modules/chronos/engine.py

import datetime
import requests
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
            "hour":         now.hour,
            "day_of_week":  now.strftime("%A"),
        }

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _time(self) -> dict:
        now = datetime.datetime.now()
        return {
            "response":   f"It's {now.strftime('%I:%M %p')}.",
            "data":       {"time": now.isoformat()},
            "confidence": 1.0,
        }

    def _date(self) -> dict:
        now = datetime.datetime.now()
        return {
            "response":   f"Today is {now.strftime('%A, %B %d, %Y')}.",
            "data":       {"date": now.date().isoformat()},
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
            w    = resp.json().get("current_weather", {})
            temp = w.get("temperature", "?")
            wind = w.get("windspeed", "?")
            return {
                "response":   f"Currently in {location}: {temp}°C, wind {wind} km/h.",
                "data":       {"temperature": temp, "windspeed": wind, "location": location},
                "confidence": 0.95,
            }
        except Exception:
            return {"response": "I couldn't fetch the weather right now.", "data": {}, "confidence": 0.0}

    def _reminder(self, entities: dict) -> dict:
        task     = entities.get("task", "something")
        time_str = entities.get("time", "")
        date_str = entities.get("date", "today")
        return {
            "response":       f"Done. Reminder set: {task} on {date_str} {time_str}.",
            "data":           {"reminder": entities},
            "confidence":     0.95,
            "context_update": {"pending_reminder": entities},  # Hestia can persist this
        }