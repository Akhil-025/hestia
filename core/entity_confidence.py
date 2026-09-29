"""
core/entity_confidence.py

Per-entity confidence scores (backlog #23).

Overall classification confidence (the top-level `confidence` field
`core/nlu.py` already returns) answers "how sure is the model that this is
the right INTENT". It says nothing about whether any individual EXTRACTED
VALUE is trustworthy — a query can classify as `send_email` with 0.95
confidence while the extracted `to` field is empty, garbled, or a
hallucinated placeholder. Those are two different questions, and this
module answers the second one.

Design: heuristics, not a second LLM call
------------------------------------------
Asking the model to self-report a confidence number per field was
considered and rejected: small local models are unreliable at introspecting
their own uncertainty on demand (they tend to report high confidence
regardless), and it would mean a second round-trip — or restructuring the
JSON schema and every few-shot example — for a number that wouldn't be
trustworthy anyway.

Instead, each entity is scored against what can actually be VERIFIED about
its shape, keyed off the entity's name (a handful of recognised key
patterns covering dates/times, emails, and numeric amounts — the types
that have a checkable structure), with a conservative default heuristic
(length/non-emptiness) for anything else. This never touches the LLM
again and is deterministic — same input, same score, every time.

This module only SCORES; it doesn't act on the scores. Whether a low score
should trigger a clarifying question is a decision for the caller (the
existing slot-filling mechanism in modules/hermes/engine.py's _clarify(),
or a future consumer) — conflating "compute a number" with "decide what
to do about it" here would make this module's job harder to test and its
output harder to reuse for something other than re-prompting.
"""
from __future__ import annotations

import re
from typing import Any

# Key-name patterns (case-insensitive substring match against the entity
# key) mapped to which scorer applies. Checked in this order; first match
# wins, so a more specific pattern should precede a more general one if
# they could both match the same key (none currently do).
_DATE_TIME_KEY_HINTS = (
    "date", "time", "when", "due", "deadline", "schedule", "start", "end",
)
_EMAIL_KEY_HINTS = ("to", "recipient", "email", "cc", "bcc")
_AMOUNT_KEY_HINTS = (
    "amount", "price", "cost", "quantity", "hours", "weight", "count",
    "duration", "servings",
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Below this, a value is treated the same as if it were structurally
# unverifiable rather than confidently wrong — a single character or an
# empty string after stripping isn't "low confidence text", it's
# effectively missing.
_MIN_TEXT_LENGTH = 2


def _matches_any(key: str, hints: tuple[str, ...]) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in hints)


def _score_date_time(value: Any) -> float:
    """
    Confidence for a date/time-shaped entity, via dateparser if available.

    dateparser is an existing project dependency (core/chronos.py and
    Hermes already use it for the same purpose), so this doesn't add a
    new one. A successful parse is treated as reasonably trustworthy
    (0.85) rather than maximal (1.0) — dateparser can resolve genuinely
    ambiguous input ("Friday" with no reference date, "the 5th" with no
    month) to *some* datetime without any signal here that it had to
    guess, so this stays a notch below what a verified email or a
    directly-echoed number gets.
    """
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        import dateparser
        parsed = dateparser.parse(text)
    except Exception:
        parsed = None
    return 0.85 if parsed is not None else 0.3


def _score_email(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    return 0.95 if _EMAIL_RE.match(text) else 0.3


def _score_amount(value: Any, raw_query: str) -> float:
    """
    Confidence for a numeric entity: high if it's a real number AND that
    same number appears in the original query text (so it was extracted,
    not invented), lower if only one of those holds.
    """
    text = str(value).strip() if value is not None else ""
    if not text:
        return 0.0
    match = _NUMERIC_RE.fullmatch(text) or _NUMERIC_RE.fullmatch(text.replace(",", ""))
    if not match:
        return 0.3  # claims to be an amount but isn't a parseable number
    if text in raw_query or text.replace(".0", "") in raw_query:
        return 0.9
    return 0.6  # a real number, but not verifiably echoed from the input


def _score_generic_text(value: Any) -> float:
    text = str(value).strip() if value is not None else ""
    if len(text) < _MIN_TEXT_LENGTH:
        return 0.0 if not text else 0.3
    return 0.85


def score_entities(entities: dict[str, Any], raw_query: str = "") -> dict[str, float]:
    """
    Score every entity in *entities* against its key-name-inferred type.

    Returns a flat ``{key: confidence}`` dict, confidence in [0, 1].
    Never raises: an entity whose value can't be scored for any reason
    (an unexpected type, a scorer's own internal error) gets the
    conservative generic-text fallback rather than aborting the whole
    call, since one bad field must never hide the other entities'
    otherwise-valid scores.
    """
    scores: dict[str, float] = {}
    for key, value in (entities or {}).items():
        if key.startswith("_"):
            continue  # internal bookkeeping (e.g. "_confirmed"), not user data
        try:
            if _matches_any(key, _EMAIL_KEY_HINTS):
                scores[key] = _score_email(value)
            elif _matches_any(key, _DATE_TIME_KEY_HINTS):
                scores[key] = _score_date_time(value)
            elif _matches_any(key, _AMOUNT_KEY_HINTS):
                scores[key] = _score_amount(value, raw_query)
            else:
                scores[key] = _score_generic_text(value)
        except Exception:
            scores[key] = _score_generic_text(value)
    return scores


def lowest_confidence_entity(scores: dict[str, float]) -> tuple[str, float] | None:
    """
    The single lowest-scored entity, or None if *scores* is empty.

    A small convenience for a caller that wants "is there anything here
    worth double-checking" without writing its own min() over an
    empty-dict-safe comparison.
    """
    if not scores:
        return None
    key = min(scores, key=scores.get)
    return key, scores[key]
