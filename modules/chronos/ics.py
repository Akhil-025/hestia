"""
modules/chronos/ics.py

Minimal, dependency-free iCalendar (RFC 5545) reader/writer for Chronos
reminders (backlog #90: "ICS export/import so Chronos reminders can
round-trip with any standard calendar app").

Scope, deliberately narrow
--------------------------
*Write*: one VEVENT per reminder, each with a display VALARM so a calendar
app actually alerts at the due time. Recurring reminders carry an RRULE.

*Read*: VEVENT and VTODO components, with the properties that matter for a
reminder - SUMMARY, DTSTART / DUE (UTC, TZID, floating and all-day forms),
RRULE, a relative VALARM TRIGGER (so "15 minutes before" becomes a reminder
15 minutes early), STATUS:CANCELLED (skipped). Constructs Chronos cannot
represent faithfully (EXDATE, RECURRENCE-ID overrides, nth-weekday-of-month
rules) are reported as warnings instead of being imported as something
subtly different.

This module knows nothing about the database: it converts between iCalendar
text and plain dicts / ``ParsedItem`` objects, and ``ChronosEngine`` does
the persistence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PRODID = "-//Hestia//Chronos Reminders//EN"
UID_DOMAIN = "hestia.local"
_UID_RE = re.compile(r"^hestia-reminder-(\d+)@" + re.escape(UID_DOMAIN) + r"$")

# RFC 5545 section 3.1: lines SHOULD be folded at 75 octets.
_FOLD_LIMIT = 75


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def escape_text(value: str) -> str:
    """Escape a TEXT value (RFC 5545 section 3.3.11)."""
    return (
        (value or "")
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )


def unescape_text(value: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            out.append("\n" if nxt in "nN" else nxt)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def fold_line(line: str) -> str:
    """Fold *line* at 75 **octets** (not characters) without ever splitting a
    multi-byte UTF-8 sequence, joining segments with CRLF + one space."""
    encoded = line.encode("utf-8")
    if len(encoded) <= _FOLD_LIMIT:
        return line
    parts: list[str] = []
    current = bytearray()
    limit = _FOLD_LIMIT
    for ch in line:
        b = ch.encode("utf-8")
        if len(current) + len(b) > limit:
            parts.append(current.decode("utf-8"))
            current = bytearray()
            limit = _FOLD_LIMIT - 1  # continuation lines start with a space
        current += b
    if current:
        parts.append(current.decode("utf-8"))
    return "\r\n ".join(parts)


def _fmt_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_ics(
    items: Iterable[dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    calendar_name: str = "Hestia reminders",
) -> str:
    """
    Serialise *items* to an iCalendar document (CRLF line endings).

    Each item is a plain dict with:
      ``uid``           str    stable identifier
      ``summary``       str    reminder text
      ``start``         datetime (aware) - the (next) due moment
      ``tzid``          str|None - IANA zone; when set the DTSTART is written
                        in that zone's local time (needed for recurring rules
                        to keep their wall-clock time across DST). Otherwise
                        DTSTART is written in UTC.
      ``rrule``         str|None - RRULE value
      ``description``   str|None
    """
    stamp = _fmt_utc(now or datetime.now(timezone.utc))
    lines: list[str] = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:{PRODID}",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{escape_text(calendar_name)}",
    ]
    for item in items:
        start: datetime = item["start"]
        summary = item.get("summary") or "Reminder"
        lines.append("BEGIN:VEVENT")
        lines.append(f"UID:{item['uid']}")
        lines.append(f"DTSTAMP:{stamp}")
        tzid = item.get("tzid")
        if tzid:
            local = start.astimezone(ZoneInfo(tzid))
            lines.append(f"DTSTART;TZID={tzid}:{local.strftime('%Y%m%dT%H%M%S')}")
        else:
            lines.append(f"DTSTART:{_fmt_utc(start)}")
        lines.append(f"SUMMARY:{escape_text(summary)}")
        if item.get("description"):
            lines.append(f"DESCRIPTION:{escape_text(item['description'])}")
        if item.get("rrule"):
            lines.append(f"RRULE:{item['rrule']}")
        if item.get("skip_holidays"):
            lines.append("X-HESTIA-SKIP-HOLIDAYS:TRUE")
        lines.extend([
            "BEGIN:VALARM",
            "ACTION:DISPLAY",
            f"DESCRIPTION:{escape_text(summary)}",
            "TRIGGER:PT0S",
            "END:VALARM",
            "END:VEVENT",
        ])
    lines.append("END:VCALENDAR")
    return "\r\n".join(fold_line(l) for l in lines) + "\r\n"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

@dataclass
class ParsedItem:
    """One importable VEVENT/VTODO, before it is turned into a reminder."""

    uid: Optional[str]
    summary: str
    start: "datetime | date"           # aware/naive datetime, or a date for all-day
    tzid: Optional[str] = None         # explicit TZID if the source gave one
    utc: bool = False                  # DTSTART was written with a trailing Z
    all_day: bool = False
    rrule: Optional[str] = None
    alarm_offset: Optional[timedelta] = None   # negative = before the start
    skip_holidays: bool = False
    kind: str = "VEVENT"
    warnings: list[str] = field(default_factory=list)


def unfold(text: str) -> list[str]:
    """Undo RFC 5545 line folding; tolerates CRLF, LF and stray CRs."""
    raw = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    for line in raw:
        if line[:1] in (" ", "\t") and out:
            out[-1] += line[1:]
        else:
            out.append(line)
    return [l for l in out if l.strip() != ""]


def _split_property(line: str) -> tuple[str, dict[str, str], str]:
    """``NAME;P=V;P2="v:x":value`` -> (NAME, {P: V, ...}, value). The first
    colon *outside double quotes* separates the parameters from the value."""
    in_quotes = False
    colon = -1
    for i, ch in enumerate(line):
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == ":" and not in_quotes:
            colon = i
            break
    if colon < 0:
        return line.strip().upper(), {}, ""
    head, value = line[:colon], line[colon + 1:]
    pieces: list[str] = []
    buf: list[str] = []
    in_q = False
    for ch in head:
        if ch == '"':
            in_q = not in_q
            buf.append(ch)
        elif ch == ";" and not in_q:
            pieces.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    pieces.append("".join(buf))
    name = pieces[0].strip().upper()
    params: dict[str, str] = {}
    for piece in pieces[1:]:
        if "=" in piece:
            k, v = piece.split("=", 1)
            params[k.strip().upper()] = v.strip().strip('"')
    return name, params, value


def parse_duration(value: str) -> Optional[timedelta]:
    """RFC 5545 DURATION, e.g. ``-PT15M``, ``PT0S``, ``-P1DT2H``, ``P1W``."""
    m = re.fullmatch(
        r"([+-])?P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?",
        value.strip().upper(),
    )
    if not m or not any(m.group(i) for i in range(2, 7)):
        if m and value.strip().upper() in ("PT0S", "P0D", "-PT0S"):
            return timedelta(0)
        return None
    sign = -1 if m.group(1) == "-" else 1
    weeks, days, hours, mins, secs = (int(m.group(i) or 0) for i in range(2, 7))
    return sign * timedelta(weeks=weeks, days=days, hours=hours, minutes=mins, seconds=secs)


def _parse_dt(
    value: str, params: dict[str, str]
) -> tuple["datetime | date", Optional[str], bool, bool, Optional[str]]:
    """-> (value, tzid, is_utc, is_all_day, warning). Raises ValueError if
    the value is unparseable."""
    value = value.strip()
    if params.get("VALUE", "").upper() == "DATE" or re.fullmatch(r"\d{8}", value):
        return datetime.strptime(value, "%Y%m%d").date(), None, False, True, None
    is_utc = value.endswith("Z")
    naive = datetime.strptime(value.rstrip("Z"), "%Y%m%dT%H%M%S")
    if is_utc:
        return naive.replace(tzinfo=timezone.utc), None, True, False, None
    tzid = params.get("TZID")
    if tzid:
        try:
            return naive.replace(tzinfo=ZoneInfo(tzid)), tzid, False, False, None
        except (ZoneInfoNotFoundError, KeyError, ValueError):
            # Unknown zone name (e.g. a Windows zone name from Outlook): keep
            # it floating and let the caller apply the user's own zone.
            return naive, None, False, False, (
                f"Unknown time zone {tzid!r}; treated as your local time."
            )
    return naive, None, False, False, None  # floating


def parse_ics(text: str) -> tuple[list[ParsedItem], list[str]]:
    """
    Parse *text* into importable items.

    Returns ``(items, notes)`` where *notes* are document-level problems
    (no calendar found, malformed components skipped). Per-item problems live
    on ``ParsedItem.warnings``. Never raises on malformed input.
    """
    notes: list[str] = []
    lines = unfold(text or "")
    if not any(l.strip().upper() == "BEGIN:VCALENDAR" for l in lines):
        return [], ["That doesn't look like an iCalendar (.ics) file."]

    items: list[ParsedItem] = []
    stack: list[str] = []
    cur: Optional[dict[str, Any]] = None

    def finish(component: dict[str, Any]) -> None:
        try:
            item = _component_to_item(component)
        except ValueError as exc:
            notes.append(f"Skipped an entry: {exc}")
            return
        if item is not None:
            items.append(item)

    for line in lines:
        name, params, value = _split_property(line)
        if name == "BEGIN":
            comp = value.strip().upper()
            stack.append(comp)
            if comp in ("VEVENT", "VTODO") and len(stack) == 2:
                cur = {"kind": comp, "props": [], "alarms": []}
            elif comp == "VALARM" and cur is not None:
                cur["alarms"].append([])
            continue
        if name == "END":
            comp = value.strip().upper()
            if stack and stack[-1] == comp:
                stack.pop()
            if comp in ("VEVENT", "VTODO") and cur is not None and cur["kind"] == comp:
                finish(cur)
                cur = None
            continue
        if cur is None:
            continue
        if stack and stack[-1] == "VALARM" and cur["alarms"]:
            cur["alarms"][-1].append((name, params, value))
        elif len(stack) == 2 and stack[-1] in ("VEVENT", "VTODO"):
            cur["props"].append((name, params, value))

    if not items and not notes:
        notes.append("The calendar contained no events or to-dos.")
    return items, notes


def _component_to_item(comp: dict[str, Any]) -> Optional[ParsedItem]:
    props: dict[str, tuple[dict[str, str], str]] = {}
    for name, params, value in comp["props"]:
        props.setdefault(name, (params, value))

    status = props.get("STATUS", ({}, ""))[1].strip().upper()
    if status == "CANCELLED":
        return None

    warnings: list[str] = []
    if "RECURRENCE-ID" in props:
        # One modified instance of a recurring event. Importing it would
        # create a phantom extra reminder alongside the series itself.
        return None

    summary = unescape_text(props.get("SUMMARY", ({}, ""))[1]).strip()
    if not summary:
        summary = unescape_text(props.get("DESCRIPTION", ({}, ""))[1]).strip()
    if not summary:
        raise ValueError("an event with no title")
    summary = summary.splitlines()[0][:300]

    kind = comp["kind"]
    key = "DUE" if kind == "VTODO" and "DUE" in props else "DTSTART"
    if key not in props:
        raise ValueError(f"{summary!r} has no start time")
    dt_params, dt_value = props[key]
    try:
        start, tzid, is_utc, all_day, tz_warning = _parse_dt(dt_value, dt_params)
    except ValueError as exc:
        raise ValueError(f"{summary!r} has an unreadable date ({dt_value!r})") from exc
    if tz_warning:
        warnings.append(tz_warning)

    rrule = props.get("RRULE", ({}, ""))[1].strip() or None
    if "EXDATE" in props:
        warnings.append("Skipped-date exceptions (EXDATE) aren't supported; they were ignored.")
    if "RDATE" in props:
        warnings.append("Extra dates (RDATE) aren't supported; they were ignored.")

    offset: Optional[timedelta] = None
    for alarm in comp["alarms"]:
        for name, params, value in alarm:
            if name != "TRIGGER":
                continue
            if params.get("VALUE", "").upper() == "DATE-TIME":
                continue  # absolute alarm time; not relative to the start
            if params.get("RELATED", "START").upper() == "END":
                continue
            parsed = parse_duration(value)
            if parsed is not None:
                # Prefer the earliest alert (most negative offset) so we
                # never remind *later* than the calendar app would have.
                offset = parsed if offset is None else min(offset, parsed)
    if offset is not None and offset > timedelta(0):
        offset = None  # an alarm AFTER the start isn't a lead time

    return ParsedItem(
        uid=props.get("UID", ({}, ""))[1].strip() or None,
        summary=summary,
        start=start,
        tzid=tzid,
        utc=is_utc,
        all_day=all_day,
        rrule=rrule,
        alarm_offset=offset,
        skip_holidays=props.get("X-HESTIA-SKIP-HOLIDAYS", ({}, ""))[1].strip().upper() == "TRUE",
        kind=kind,
        warnings=warnings,
    )


def own_reminder_id(uid: Optional[str]) -> Optional[int]:
    """If *uid* is one Chronos itself exported, the reminder id inside it."""
    if not uid:
        return None
    m = _UID_RE.match(uid.strip())
    return int(m.group(1)) if m else None


def make_uid(reminder_id: int) -> str:
    return f"hestia-reminder-{reminder_id}@{UID_DOMAIN}"