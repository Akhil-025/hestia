
# core/actions.py

from core.browser_agent import HestiaBrowserAgent
import datetime as _datetime
import re
import requests
from core.memory import HestiaMemory
from core.event_bus import bus
from skills.base import SkillLoader



class HestiaActions:
    """Execute structured intents from NLU using memory and external APIs."""
    

    def __init__(self, memory: HestiaMemory):
        """Store memory reference and event bus."""
        self.memory = memory
        self._google = None
        self._browser = None
    def set_browser_agent(self, agent: "HestiaBrowserAgent") -> None:
        """Inject browser agent reference."""
        self._browser = agent

    def execute(self, intent: str, entities: dict, raw_query: str = "") -> str:

        _MODULE_OWNED = {
            "get_time", "get_date", "get_weather", "set_reminder",
            "read_email", "send_email", "list_events", "create_event",
            "browser_action", "check_flight", "search_web",
        }

        if intent in _MODULE_OWNED:
            import logging
            logging.getLogger("hestia").warning(
                "[Actions] Intent '%s' bypassed orchestrator — route via modules.", intent
            )
            return ""

        result = None

        if intent == "save_name":
            result = self._save_name(entities)

        elif intent == "take_note":
            result = self._take_note(entities, raw_query)

        elif intent == "get_notes":
            result = self._get_notes()

        elif intent == "get_history":
            result = self._get_history(entities)

        elif intent == "get_user_info":
            result = self._get_user_info(entities)

        elif intent == "set_preference":
            result = self._set_preference(entities)

        elif intent == "chat":
            result = ""

        else:
            result = SkillLoader.execute_skill(intent, entities, self.memory, raw_query)

        if result:
            bus.emit("action_done", {"intent": intent, "response": result})

        return result or "I'm not sure how to help with that yet."



    def _browser_action(self, entities: dict, raw_query) -> str:
        """Delegate to browser agent."""
        if not self._browser:
            return "Browser automation is not enabled."
        from skills.browser_tasks import execute as browser_execute
        return browser_execute(entities, self.memory, raw_query)

    def _check_flight(self, entities: dict) -> str:
        """Check flight status via browser agent."""
        if not self._browser:
            return "Browser automation is not enabled."
        flight = entities.get("flight_number", entities.get("flight", ""))
        if not flight:
            return "Which flight number should I check?"
        return self._browser.check_flight_status(flight)

    def set_google_agent(self, agent) -> None:
        """Inject Google agent reference after initialisation."""
        self._google = agent

    def _read_email(self, entities: dict) -> str:
        """Fetch and summarise unread emails."""
        if not self._google or not self._google.is_authenticated():
            return "Google is not connected. Please set up your credentials."
        count = int(entities.get("count", 5))
        emails = self._google.read_emails(max_results=count)
        return self._google.format_emails_for_tts(emails)

    def _send_email(self, entities: dict) -> str:
        """Send an email using entities: to, subject, body."""
        if not self._google or not self._google.is_authenticated():
            return "Google is not connected."
        to = entities.get("to", "")
        subject = entities.get("subject", "Message from Hestia")
        body = entities.get("body", entities.get("message", ""))
        if not to:
            return "Who should I send it to?"
        if not body:
            return "What should the email say?"
        success = self._google.send_email(to, subject, body)
        return f"Email sent to {to}." if success else "I couldn't send that email."

    def _list_events(self, entities: dict) -> str:
        """List upcoming calendar events."""
        if not self._google or not self._google.is_authenticated():
            return "Google Calendar is not connected."

        days = int(entities.get("days", 7))
        events = self._google.list_events(days_ahead=days)

        if not events:
            return "No events scheduled for the next few days."

        return self._google.format_events_for_tts(events)

    def _create_event(self, entities: dict) -> str:
        """Create a calendar event from NLU entities."""
        if not self._google or not self._google.is_authenticated():
            return "Google Calendar is not connected."
        title = entities.get("task") or entities.get("title") or entities.get("event", "")
        if not title:
            return "What should I call the event?"
        date_str = entities.get("date", "today")
        time_str = entities.get("time", "09:00")
        try:
            now = _datetime.datetime.now()
            if date_str.lower() in ("today", ""):
                base_date = now.date()
            elif date_str.lower() == "tomorrow":
                base_date = (now + _datetime.timedelta(days=1)).date()
            else:
                base_date = _datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            hour, minute = map(int, time_str.replace(":", " ").split()[:2])
            start_dt = _datetime.datetime.combine(base_date, _datetime.time(hour, minute))
        except Exception:
            start_dt = _datetime.datetime.now() + _datetime.timedelta(hours=1)
        success = self._google.create_event(title=title, start_dt=start_dt)
        if success:
            return f"Done. {title} added to your calendar."
        return "I couldn't create that event."



    def _save_name(self, entities: dict) -> str:
        """Save user's name to preferences."""
        name = entities.get("name", "").strip()
        if not name:
            return "I didn't catch your name, could you repeat it?"
        name = name.title()
        self.memory.set_preference("user_name", name)
        return f"Got it! I'll remember you as {name}."

    def _get_time(self) -> str:
        """Return current time."""
        now = _datetime.datetime.now()
        return f"It's {now.strftime('%I:%M %p')}."

    def _get_date(self) -> str:
        """Return current date."""
        now = _datetime.datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."

    def _get_weather(self, entities: dict) -> str:
        """Fetch current weather from Open-Meteo."""
        
        location = self.memory.get_preference("location", "Mumbai")
        CITY_COORDS = {
            "mumbai":   (19.0760, 72.8777),
            "delhi":    (28.6139, 77.2090),
            "bangalore": (12.9716, 77.5946),
            "bengaluru": (12.9716, 77.5946),
            "chennai":  (13.0827, 80.2707),
            "kolkata":  (22.5726, 88.3639),
            "hyderabad": (17.3850, 78.4867),
            "pune":     (18.5204, 73.8567),
            "london":   (51.5074, -0.1278),
            "new york": (40.7128, -74.0060),
            "tokyo":    (35.6762, 139.6503),
        }
        lat, lon = CITY_COORDS.get(location.lower(), (19.0760, 72.8777))

        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&wind_speed_unit=kmh"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            weather = data.get("current_weather", {})
            temp = weather.get("temperature", "?")
            windspeed = weather.get("windspeed", "?")
            return f"Currently in {location}: {temp}°C, wind {windspeed} km/h."
        except Exception:
            return "I couldn't fetch the weather right now."

    def _set_reminder(self, entities: dict) -> str:
        """Store reminder in memory as a special interaction."""
        task = entities.get("task", "something")
        time_str = entities.get("time", "")
        date_str = entities.get("date", "today")

        reminder_text = f"REMINDER: {task}"
        reminder_detail = f"{date_str} {time_str}".strip()
        self.memory.add_interaction(reminder_text, reminder_detail, "set_reminder")
        return f"Done. Reminder set: {task} on {date_str} {time_str}."

    def _take_note(self, entities: dict, raw_query: str = "") -> str:
        """Save a note to memory."""
        note_text = entities.get("content") or entities.get("text") or entities.get("task") or entities.get("note", "")

        if not note_text and raw_query:
            cleaned = re.sub(r'^(take a note|note down|jot down|remember|note)\s*[:\-]?\s*',
                             '', raw_query, flags=re.IGNORECASE).strip()
            note_text = cleaned

        if not note_text:
            return "What would you like me to note down?"

        self.memory.add_interaction(note_text, "Note saved.", "take_note")
        return "Note saved."

    def _get_notes(self) -> str:
        """Retrieve recent notes from memory."""
        notes = self.memory.get_by_intent("take_note", limit=10)
        if not notes:
            return "You don't have any notes saved yet."

        result = "Here are your recent notes:\n"
        for item in notes:
            result += f"- {item['query']}\n"
        return result.strip()

    def _get_history(self, entities: dict) -> str:
        """Retrieve recent conversation history, excluding notes and reminders."""
        limit = int(entities.get("limit", 5))
        recent = self.memory.get_recent_filtered(limit, exclude_intents=["take_note", "set_reminder"])
        if not recent:
            return "We haven't talked much yet."

        result = "Here's what we talked about:\n"
        for item in recent:
            result += f"- {item['query']}\n"
        return result.strip()

    def _get_user_info(self, entities: dict) -> str:
        """Return all stored user preferences."""
        prefs = self.memory.get_all_preferences()
        if not prefs:
            return "I don't know much about you yet."

        result = "Here's what I know about you:\n"
        for key, value in prefs.items():
            formatted_key = key.replace("_", " ").capitalize()
            result += f"- {formatted_key}: {value}\n"
        return result.strip()

    def _set_preference(self, entities: dict) -> str:
        """Store a user preference."""
        key = entities.get("key", "")
        value = entities.get("value", "")

        if not value:
            return "I didn't quite catch what you want me to remember."

        if not key or key == "preference":
            words = value.strip().split()
            key = "_".join(words[:3]).lower() if words else "preference"

        self.memory.set_preference(key, value)
        return "Got it, I'll remember that."