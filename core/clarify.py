"""
core/clarify.py  (Hardening Plan, Phase 6.5)

Targeted clarification instead of a generic "I didn't understand". Only fires
when the utterance is genuinely ambiguous between a small, known set of
intents, and names those options so a one-word reply resolves it.
"""
from __future__ import annotations

import re
from typing import Optional

_BARE_SEARCH_RE = re.compile(
    r"^\s*(?:please\s+)?(?:search|look\s+up)\s+(?!the web\b|online\b|"
    r"my\b|in my\b|on the web\b|photos?\b|pictures?\b)(?P<q>\S.+?)\s*[.!?]*\s*$", re.I)
_SOURCE_WORD_RE = re.compile(
    r"\b(web|internet|online|google|documents?|docs?|files?|papers?|photos?|"
    r"pictures?|images?|notes?|email|mail|calendar)\b", re.I)


def ambiguous_search_question(text: str) -> Optional[str]:
    """Question for a bare "search X" naming no source, else None. Deterministic,
    so it runs before the model (which guessed pluto_forecast_spending for
    "search taxes")."""
    m = _BARE_SEARCH_RE.match(text or "")
    if m and not _SOURCE_WORD_RE.search(text or ""):
        return (f"Where should I look for \"{m.group('q')}\" — the web, your "
                f"documents, or your photos?")
    return None


def clarification_for(text: str, intent: str, confidence: float) -> Optional[str]:
    """A clarifying question, or None. Called when the classifier fell back to
    chat / low confidence."""
    if intent != "chat" and confidence >= 0.5:
        return None
    m = _BARE_SEARCH_RE.match(text or "")
    if m and not _SOURCE_WORD_RE.search(text or ""):
        q = m.group("q")
        return (f"Where should I look for \"{q}\" — the web, your documents, "
                f"or your photos?")
    return None
