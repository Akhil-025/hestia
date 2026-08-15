# modules/hestia/core_module.py

import logging
from modules.base import BaseModule
from core.ollama_client import generate
import platform, datetime, re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)


def _safe_int(value, default: int) -> int:
    """Coerce *value* to int, falling back to *default* if that fails."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class CoreModule(BaseModule):
    name = "core"

    _INTENTS = {
        "save_name", "take_note", "get_notes", "delete_notes",
        "get_history", "set_preference",
        "get_system_info", "get_user_info", "chat"
    }

    def __init__(self, memory, ollama_cfg: dict, llm=None, timezone_name: str = "UTC"):
        self._memory = memory
        self._ollama = ollama_cfg
        self._llm_instance = llm  # HestiaLLM | None — preferred path
        # ChronosEngine and HermesEngine both resolve the user's configured
        # IANA timezone (main.py passes chronos.timezone to both) so "what
        # time is it" / calendar events land on local wall-clock time
        # instead of the server's own clock. CoreModule.get_user_info()
        # and get_system_info() need the same thing: they're the recovery
        # path the NLU falls into when it misclassifies date/time questions
        # (see _get_user_info below), so answering with server-local time
        # instead of the user's local time would silently disagree with
        # what Chronos would have said for the exact same question.
        try:
            self._tz = ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, KeyError):
            logger.warning(
                "Unrecognised timezone %r; falling back to UTC.", timezone_name
            )
            self._tz = ZoneInfo("UTC")

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        raw = entities.get("raw_query", "")

        if intent == "chat":
            return self._chat(raw)

        if intent == "get_system_info":
            return self._sys_info()

        if intent == "save_name":
            return self._save_name(entities)

        if intent == "take_note":
            return self._take_note(entities, raw)

        if intent == "get_notes":
            return self._get_notes()

        if intent == "delete_notes":
            return self._delete_notes()

        if intent == "get_history":
            return self._get_history(entities)

        if intent == "get_user_info":
            return self._get_user_info(entities, raw)

        if intent == "set_preference":
            return self._set_preference(entities)

        return {"response": "", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        return {}

    # ───── handlers ─────

    def _chat(self, query: str) -> dict:
        try:
            prompt = f"You are Hestia. Answer concisely in 1-2 sentences.\n\nQuestion: {query}"
            if self._llm_instance is not None:
                text = self._llm_instance.generate(prompt)
            else:
                text = generate(
                    prompt,
                    model=self._ollama.get("model", "mistral"),
                    host=self._ollama.get("host", "127.0.0.1"),
                    port=self._ollama.get("port", 11434),
                )
            return {"response": text, "data": {}, "confidence": 0.7}
        except Exception:
            logger.exception("CoreModule._chat() failed for query=%r", query[:80])
            return {"response": "I'm not sure about that.", "data": {}, "confidence": 0.3}

    def _sys_info(self) -> dict:
        return {
            "response": (
                f"Running on {platform.system()} {platform.release()}, "
                f"Python {platform.python_version()}, "
                f"time {datetime.datetime.now(self._tz).strftime('%I:%M %p')}."
            ),
            "data": {},
            "confidence": 1.0,
        }

    def _save_name(self, entities: dict) -> dict:
        # `entities.get("name", "")` only covers a *missing* key; the NLU
        # can also emit the key with an explicit null (entities={"name": None}),
        # and "".strip() on None raises AttributeError. `or ""` covers both,
        # matching the defensive pattern used throughout modules/hermes.
        name = (entities.get("name") or "").strip().title()
        if not name:
            return {"response": "I didn't catch your name.", "data": {}, "confidence": 0.0}

        self._memory.learn("user_name", name)
        return {"response": f"Got it! I'll remember you as {name}.", "data": {}, "confidence": 0.9}

    def _take_note(self, entities: dict, raw: str) -> dict:
        note = (
            entities.get("content")
            or entities.get("text")
            or entities.get("task")
            or ""
        )

        if not note and raw:
            note = re.sub(
                r'^(take a note|note down|jot down|remember|note)\s*[:\-]?\s*',
                '',
                raw,
                flags=re.IGNORECASE
            ).strip()

        if not note:
            return {"response": "What would you like me to note down?", "data": {}, "confidence": 0.0}

        # Do NOT call self._memory.learn() here.
        # The orchestrator's interaction_logged bus event persists this naturally
        # to interaction_log, which is where _get_notes reads from.
        #
        # The response text below (not just entities["data"]) is what actually
        # ends up in interaction_log.hestia_response, so it has to carry the
        # parsed note content — otherwise _get_notes has nothing but the raw
        # user_text (e.g. "take a note: buy toy", "> take note buy milk") to
        # show, which is why the notes list used to echo back raw commands
        # instead of clean note text.
        return {
            "response": f"Note saved: {note}",
            "data": {"note": note},
            "confidence": 0.95
        }

    # Matches the "Note saved: " prefix _take_note stores in hestia_response.
    _NOTE_PREFIX_RE = re.compile(r'^Note saved:\s*', re.IGNORECASE)

    def _get_notes(self) -> dict:
        rows = self._memory.db.get_by_intent("take_note", 10)
        notes = [{"query": r["query"], "response": r["response"], "intent": r["intent"]} for r in rows]

        if not notes:
            return {"response": "No notes saved yet.", "data": {}, "confidence": 0.9}

        def _content(n: dict) -> str:
            resp = n.get("response") or ""
            if self._NOTE_PREFIX_RE.match(resp):
                return self._NOTE_PREFIX_RE.sub('', resp).strip()
            # Older rows saved before this fix only have the raw query;
            # fall back to that rather than showing nothing.
            return n.get("query", "").strip()

        body = "Your notes:\n" + "\n".join(f"- {_content(n)}" for n in notes)

        return {"response": body, "data": {"notes": notes}, "confidence": 0.9}

    def _delete_notes(self) -> dict:
        try:
            deleted = self._memory.db.delete_by_intent("take_note")
        except Exception:
            logger.exception("_delete_notes: DB delete failed.")
            return {"response": "I couldn't delete your notes right now.", "data": {}, "confidence": 0.0}

        if not deleted:
            return {"response": "You don't have any notes to delete.", "data": {}, "confidence": 0.9}

        return {
            "response": f"Deleted {deleted} note(s).",
            "data": {"deleted": deleted},
            "confidence": 0.95,
        }

    def _get_history(self, entities: dict) -> dict:
        # int() raises on a non-numeric NLU extraction (e.g. "a few"); that
        # would otherwise surface to the user as the orchestrator's generic
        # "something went wrong" instead of just falling back to a sane
        # default, so coerce defensively rather than trusting the entity.
        limit = _safe_int(entities.get("limit"), default=5)

        recent = self._memory.db.get_recent_interactions_excluding(
            limit, ["take_note", "set_reminder"]
        )

        if not recent:
            return {"response": "We haven't talked much yet.", "data": {}, "confidence": 0.9}

        body = "Here's what we talked about:\n" + "\n".join(
            f"- {r['query']}" for r in recent
        )

        return {"response": body, "data": {"history": recent}, "confidence": 0.9}


    # Keys the NLU has been observed to send for date/time questions it
    # misclassifies as get_user_info instead of Chronos's get_time/get_date
    # (e.g. "what is todays date" → {"key": "current_date"}). Answered
    # directly here rather than claiming ignorance, since Hestia obviously
    # knows the date/time regardless of which intent name the NLU picked.
    _DATE_KEYS = frozenset({"current_date", "date", "today", "todays_date"})
    _TIME_KEYS = frozenset({"current_time", "time"})

    # Keys/phrasing the NLU has been observed to send (or fail to send) for
    # "what's my name?" — it often emits get_user_info with entities={}
    # rather than {"key": "user_name"}, even though _save_name() always
    # stores the fact under the fixed key "user_name". Without this alias
    # the fact is unreachable from that phrasing even though it was saved
    # correctly — same class of bug the date/time aliasing above already
    # covers, just for identity instead of date/time.
    _NAME_KEYS = frozenset({"name", "user_name", "my_name", "users_name"})
    _NAME_QUERY_RE = re.compile(r"\bname\b", re.IGNORECASE)

    def _get_user_info(self, entities: dict, raw: str = "") -> dict:
        key = (entities.get("key") or "").strip().lower()

        if key in self._DATE_KEYS:
            return {
                "response": f"Today's date is {datetime.datetime.now(self._tz).strftime('%A, %B %d, %Y')}.",
                "data": {"key": key},
                "confidence": 0.95,
            }
        if key in self._TIME_KEYS:
            return {
                "response": f"It's {datetime.datetime.now(self._tz).strftime('%I:%M %p')}.",
                "data": {"key": key},
                "confidence": 0.95,
            }

        if key in self._NAME_KEYS:
            key = "user_name"
        elif not key and raw and self._NAME_QUERY_RE.search(raw):
            # Last-resort recovery: NLU sent no key at all, but the raw
            # question is plainly asking about the user's name (e.g.
            # "what's my name?" / "do you know my name").
            key = "user_name"

        if key:
            try:
                value = self._memory.db.get_fact(key)
            except Exception:
                logger.exception("_get_user_info: get_fact(%r) failed.", key)
                value = None
            if value:
                return {"response": value, "data": {"key": key}, "confidence": 0.85}

        return {"response": "I don't have that information yet.", "data": {}, "confidence": 0.3}

    def _set_preference(self, entities: dict) -> dict:
        key = entities.get("key", "")
        value = entities.get("value", "")

        if not value:
            return {"response": "What should I remember?", "data": {}, "confidence": 0.0}

        if not key or key == "preference":
            # No deterministic key was extracted by NLU. Rather than fabricate one
            # from the value's leading words (which produced unpredictable, hard
            # to look up keys), ask the user to be specific.
            return {
                "response": "What should I call this preference? (e.g. 'set my location preference to Mumbai')",
                "data": {},
                "confidence": 0.3,
            }

        self._memory.learn(key, value)

        return {"response": "Got it, I'll remember that.", "data": {}, "confidence": 0.95}