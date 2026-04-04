# modules/hermes/engine.py

import datetime
from modules.base import BaseModule


class HermesEngine(BaseModule):
    name = "hermes"

    _INTENTS = {"read_email", "send_email", "list_events", "create_event"}

    def __init__(self, google_agent=None):
        self._google = google_agent  # HestiaGoogleAgent, injected by orchestrator

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        if not self._google or not self._google.is_authenticated():
            return {"response": "Communication services are not connected.", "data": {}, "confidence": 0.0}

        if intent == "read_email":
            count  = int(entities.get("count", 5))
            emails = self._google.read_emails(max_results=count)
            return {
                "response":   self._google.format_emails_for_tts(emails),
                "data":       {"emails": emails},
                "confidence": 0.95,
            }

        elif intent == "send_email":
            to      = entities.get("to", "")
            subject = entities.get("subject", "Message from Hestia")
            body    = entities.get("body", entities.get("message", ""))
            if not to:
                return {"response": "Who should I send it to?", "data": {}, "confidence": 0.0}
            if not body:
                return {"response": "What should the email say?", "data": {}, "confidence": 0.0}
            success = self._google.send_email(to, subject, body)
            return {
                "response":   f"Email sent to {to}." if success else "I couldn't send that email.",
                "data":       {},
                "confidence": 0.9 if success else 0.0,
            }

        elif intent == "list_events":
            days   = int(entities.get("days", 7))
            events = self._google.list_events(days_ahead=days)
            return {
                "response":   self._google.format_events_for_tts(events),
                "data":       {"events": events},
                "confidence": 0.95,
            }

        elif intent == "create_event":
            return self._create_event(entities)

        return {"response": "I can't handle that communication request.", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        return {"hermes_connected": bool(self._google and self._google.is_authenticated())}

    def _create_event(self, entities: dict) -> dict:
        title    = entities.get("task") or entities.get("title") or entities.get("event", "")
        if not title:
            return {"response": "What should I call the event?", "data": {}, "confidence": 0.0}
        date_str = entities.get("date", "today")
        time_str = entities.get("time", "09:00")
        try:
            now = datetime.datetime.now()
            if date_str.lower() in ("today", ""):
                base_date = now.date()
            elif date_str.lower() == "tomorrow":
                base_date = (now + datetime.timedelta(days=1)).date()
            else:
                base_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            hour, minute = map(int, time_str.replace(":", " ").split()[:2])
            start_dt     = datetime.datetime.combine(base_date, datetime.time(hour, minute))
        except Exception:
            start_dt = datetime.datetime.now() + datetime.timedelta(hours=1)
        success = self._google.create_event(title=title, start_dt=start_dt)
        return {
            "response":   f"Done. {title} added to your calendar." if success else "I couldn't create that event.",
            "data":       {},
            "confidence": 0.9 if success else 0.0,
        }