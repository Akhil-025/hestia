"""
core/temporal.py

Temporal normaliser (Hardening Plan, Phase 3.4).

The NLU model regularly puts a whole phrase such as ``"tomorrow at 5pm"`` in
the ``time`` entity (or in ``date``), while consumers such as Hermes'
``_parse_datetime`` expect a *date word / ISO date* and a separate
``HH:MM`` / ``5pm`` time. This module splits those phrases into the two
fields consumers understand, and keeps Hermes' documented design of naive
local wall-clock times (no tzinfo is ever attached here).

Public API
----------
split_date_time(date_str, time_str, today=None) -> (date_str, time_str)
    Returns canonical strings: the date is ``"today"``, ``"tomorrow"`` or an
    ISO ``YYYY-MM-DD``; the time is ``"HH:MM"`` (24h). A part that could not
    be found is returned unchanged ('' stays '').  Never raises.

normalize_event_entities(entities, today=None) -> dict
    Returns a copy of *entities* with ``date`` / ``time`` normalised.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday"]

_TIME_RE = re.compile(
    r"(?:\bat\s+|\b@\s*)?"
    r"\b(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>[ap]\.?m\.?)(?!\w)"
    r"|(?:\bat\s+)?\b(?P<h24>[01]?\d|2[0-3]):(?P<m24>[0-5]\d)\b",
    re.IGNORECASE,
)
_NOON_RE = re.compile(r"\bnoon\b", re.I)
_MIDNIGHT_RE = re.compile(r"\bmidnight\b", re.I)
_ISO_DT_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?")
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
_NEXT_WD_RE = re.compile(
    r"\b(?:(next|this|on)\s+)?(" + "|".join(_WEEKDAYS) + r")\b", re.I)


def _extract_time(text: str) -> Tuple[Optional[str], str]:
    """Return (HH:MM or None, text with the time phrase removed)."""
    if _NOON_RE.search(text):
        return "12:00", _NOON_RE.sub(" ", text)
    if _MIDNIGHT_RE.search(text):
        return "00:00", _MIDNIGHT_RE.sub(" ", text)
    m = _TIME_RE.search(text)
    if not m:
        return None, text
    if m.group("h") is not None:
        hour = int(m.group("h"))
        minute = int(m.group("m") or 0)
        ap = m.group("ap").lower().replace(".", "")
        if not (1 <= hour <= 12 and 0 <= minute <= 59):
            return None, text
        if ap == "pm" and hour != 12:
            hour += 12
        elif ap == "am" and hour == 12:
            hour = 0
    else:
        hour, minute = int(m.group("h24")), int(m.group("m24"))
    rest = (text[:m.start()] + " " + text[m.end():])
    return f"{hour:02d}:{minute:02d}", rest


def _extract_date(text: str, today: date) -> Tuple[Optional[str], str]:
    """Return (canonical date or None, text with the date phrase removed)."""
    low = text.lower()
    m = _ISO_RE.search(low)
    if m:
        return m.group(0), (text[:m.start()] + " " + text[m.end():])
    m = _DMY_RE.search(low)
    if m:
        d, mo, y = (int(g) for g in m.groups())
        try:
            return date(y, mo, d).isoformat(), (text[:m.start()] + " " + text[m.end():])
        except ValueError:
            return None, text
    for word, canon in (("day after tomorrow", None), ("tomorrow", "tomorrow"),
                        ("tonight", "today"), ("today", "today")):
        i = low.find(word)
        if i != -1 and re.search(r"\b" + re.escape(word) + r"\b", low):
            rest = text[:i] + " " + text[i + len(word):]
            if canon is None:
                return (today + timedelta(days=2)).isoformat(), rest
            return canon, rest
    m = _NEXT_WD_RE.search(low)
    if m:
        target = _WEEKDAYS.index(m.group(2).lower())
        ahead = (target - today.weekday()) % 7
        if ahead == 0 or (m.group(1) or "").lower() == "next":
            ahead = ahead or 7
        return (today + timedelta(days=ahead)).isoformat(), \
            (text[:m.start()] + " " + text[m.end():])
    return None, text


def split_date_time(date_str: str = "", time_str: str = "",
                    today: Optional[date] = None) -> Tuple[str, str]:
    """Split combined phrases into canonical (date, time). Never raises."""
    try:
        today = today or datetime.now().date()
        d_in = str(date_str or "").strip()
        t_in = str(time_str or "").strip()
        combined = f"{d_in} {t_in}".strip()
        if not combined:
            return d_in, t_in

        # The model often returns a full ISO timestamp ("2026-10-13T17:00:00",
        # with or without seconds / timezone) in a single field. Take both
        # parts from it; the wall-clock time is kept as written (no tz shift).
        m = _ISO_DT_RE.search(combined)
        if m:
            return m.group(1), f"{m.group(2)}:{m.group(3)}"

        # time first (it is the most specific token), searching both fields
        t_val, _ = _extract_time(t_in)
        if t_val is None:
            t_val, _ = _extract_time(d_in)
        d_val, _ = _extract_date(d_in, today)
        if d_val is None:
            d_val, _ = _extract_date(t_in, today)

        return (d_val if d_val is not None else d_in,
                t_val if t_val is not None else t_in)
    except Exception:  # defensive: normaliser must never break a request
        return str(date_str or ""), str(time_str or "")


def normalize_event_entities(entities: dict,
                             today: Optional[date] = None) -> dict:
    """Return a copy of *entities* with date/time split and canonicalised."""
    out = dict(entities or {})
    if not (out.get("date") or out.get("time")):
        return out
    d, t = split_date_time(out.get("date") or "", out.get("time") or "", today)
    if d:
        out["date"] = d
    if t:
        out["time"] = t
    return out
