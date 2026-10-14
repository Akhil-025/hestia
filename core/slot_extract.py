"""
core/slot_extract.py

Deterministic regex slot extraction for entity-bearing commands
(Hardening Plan, Phase 2.2). Runs before the LLM, like the existing
``save_name`` fast path. Aliases only set the *intent*; these patterns also
fill the entities the module needs, which is why "Log an expense: Rs500 for
groceries" used to re-ask for the amount.

``extract_slots(text)`` returns ``{"intent": ..., "entities": {...}}`` or
``None``. Question-form input (ends with '?' or starts with an interrogative)
never matches: a question must not become a write.
"""
from __future__ import annotations

import re
from typing import Optional

_INTERROGATIVE_RE = re.compile(
    r"^\s*(?:what|what's|whats|which|who|whom|whose|when|where|why|how|"
    r"do|does|did|can|could|is|are|was|were|will|would|should|have|has)\b",
    re.I,
)


def is_question(text: str) -> bool:
    t = (text or "").strip()
    return t.endswith("?") or bool(_INTERROGATIVE_RE.match(t))


_NOTE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:take\s+(?:a\s+)?note|make\s+(?:a\s+)?note|note\s+down|"
    r"jot\s+down|write\s+down)\s*(?:that|this|:|-)?\s*[:\-]?\s*(?P<c>\S.*?)\s*$",
    re.I | re.S,
)
_HABIT_RE = re.compile(
    r"^\s*(?:please\s+)?(?:add|create|start|begin|track)\s+(?:a\s+|new\s+|a\s+new\s+)?"
    r"habit\s*(?:of|called|named|to|:|-)?\s*[:\-]?\s*(?P<n>\S.*?)\s*[.!]*\s*$",
    re.I,
)
_CUR = r"(?:rs\.?|inr|rupees?|₹|\$|usd|eur|€|£)"
_EXPENSE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:(?:log|add|record|track)\s+(?:an?\s+)?expense|"
    r"i\s+(?:spent|paid)|spent|paid)\s*[:\-]?\s*(?:of\s+)?"
    rf"(?P<cur>{_CUR})?\s*(?P<amt>\d[\d,]*(?:\.\d+)?)\s*(?P<cur2>rs\.?|inr|rupees?)?"
    r"\s*(?:for|on|towards)\s+(?P<d>\S.*?)\s*[.!]*\s*$",
    re.I,
)


_POEM_RE = re.compile(
    r"^\s*(?:please\s+)?(?:write|compose|make)\s+(?:me\s+)?(?:an?\s+)?(?:[\w\-]+\s+){0,3}?"
    r"(?:poem|poetry|haiku|sonnet|limerick)\s+(?:about|on|of|for)\s+(?P<t>\S.*?)\s*[.!]*\s*$", re.I)
_BRAIN_RE = re.compile(
    r"^\s*(?:please\s+)?(?:help\s+me\s+)?brainstorm(?:\s+(?:some\s+)?ideas)?"
    r"(?:\s+(?:for|about|on))?\s+(?P<t>\S.*?)\s*[.!]*\s*$", re.I)
_EVENT_A_RE = re.compile(
    r"^\s*(?:please\s+)?(?:create|add|schedule|set\s+up|book|make)\s+(?:an?\s+)?"
    r"(?:event|meeting|appointment)\s+(?P<when>.+?)\s+(?:called|named|titled)\s+(?P<title>\S.*?)\s*[.!]*\s*$",
    re.I)
_EVENT_B_RE = re.compile(
    r"^\s*(?:please\s+)?(?:create|add|schedule|set\s+up|book|make)\s+(?:an?\s+)?"
    r"(?:event|meeting|appointment)\s+(?:called|named|titled)\s+(?P<title>.+?)\s+"
    r"(?P<when>(?:tomorrow|today|tonight|on|at|next|this)\b.+?)\s*[.!]*\s*$", re.I)
_UPDATE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:make it|change it to|move it to|push it to|"
    r"reschedule(?:\s+it)?\s+to)\s+(?P<t>\S.*?)\s*(?:instead)?\s*[.!]*\s*$", re.I)
_WHEN_RE = re.compile(
    r"\d|\b(?:noon|midnight|tomorrow|today|tonight|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)\b", re.I)


def extract_slots(text: str) -> Optional[dict]:
    if not text or is_question(text):
        return None

    # "make it 6 instead" -> change the event just created (Phase 5.2). Needs
    # a time/date-looking tail so "make it shorter" is NOT captured.
    m = _UPDATE_RE.match(text)
    if m and _WHEN_RE.search(m.group("t")):
        return {"intent": "update_last_event",
                "entities": {"time": re.sub(r"\s*instead\s*$", "", m.group("t"), flags=re.I)}}

    m = _NOTE_RE.match(text)
    # "make a note that I am vegetarian" is a durable fact about the user
    # (learn_fact, via the alias list), not a free-form note.
    if m and not re.search(r"\b(?:note|jot down|write down)\s+that\s+(?:i|my)\b", text, re.I):
        return {"intent": "take_note",
                "entities": {"content": m.group("c").strip()}}

    for rx in (_EVENT_A_RE, _EVENT_B_RE):
        m = rx.match(text)
        if m and _WHEN_RE.search(m.group("when")):
            return {"intent": "create_event",
                    "entities": {"title": m.group("title").strip(),
                                 "time": m.group("when").strip()}}
    m = _POEM_RE.match(text)
    if m:
        return {"intent": "orpheus_write_poem", "entities": {"topic": m.group("t").strip()}}
    m = _BRAIN_RE.match(text)
    if m:
        return {"intent": "orpheus_brainstorm", "entities": {"topic": m.group("t").strip()}}

    m = _HABIT_RE.match(text)
    if m:
        return {"intent": "add_habit",
                "entities": {"name": m.group("n").strip()}}

    m = _EXPENSE_RE.match(text)
    if m:
        amount = m.group("amt").replace(",", "")
        desc = re.sub(r"^(?:my|the|a|an)\s+", "", m.group("d").strip(),
                      flags=re.I)
        return {"intent": "pluto_log_expense",
                "entities": {"amount": amount, "description": desc}}
    return None


# ---------------------------------------------------------------------------
# Question-form guard (Hardening Plan, Phase 2.4)
# ---------------------------------------------------------------------------
# A write intent (take_note, learn_fact, ...) must never fire for a question
# such as "What notes do I have?" — that is a read. Remap to the read twin.
_READ_QUESTION_RE = re.compile(
    r"^\s*(?:what|what's|whats|which|who|whom|where|when|how many|how much|"
    r"do i have|have i|did i|show me|list)\b",
    re.I,
)

WRITE_TO_READ = {
    "take_note": "get_notes",
    "learn_fact": "get_user_info",
    "save_name": "get_user_info",
    "remove_habit": "list_habits",
    "delete_notes": "get_notes",
}


def guard_write_intent(intent: str, text: str) -> str:
    """Return the intent to use: the read twin if a write intent was chosen
    for a question-form utterance, else *intent* unchanged."""
    target = WRITE_TO_READ.get(intent)
    if target and _READ_QUESTION_RE.match(text or ""):
        return target
    return intent
