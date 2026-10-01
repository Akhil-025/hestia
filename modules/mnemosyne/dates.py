"""
modules/mnemosyne/dates.py

Free-text date phrases -> a local calendar range, for dated recall (#48) and
EXIF photo search (#74). Pure stdlib (no dateparser) so it behaves the same in
every environment. Returns None for anything that is not clearly a date, so a
caller can fall through to another interpretation instead of guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
_MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
                "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12})
_DAYS = {d: i for i, d in enumerate(
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DOW = "|".join(_DAYS)
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
_MIN_YEAR = 1900


@dataclass(frozen=True)
class DateRange:
    start: date          # inclusive
    end: date            # inclusive
    label: str           # human wording for replies
    phrase: str          # the matched text


def _n(s: str) -> int:
    return _NUM.get(s.lower()) or int(s)


def _mk(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _month_range(y: int, m: int, today: date) -> DateRange:
    start = date(y, m, 1)
    end = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    return start, min(end, today) if (y, m) == (today.year, today.month) else end


def resolve_range(text: str, today: Optional[date] = None) -> Optional[DateRange]:
    today = today or date.today()
    t = " ".join((text or "").lower().replace(",", " ").split())
    if not t:
        return None

    def done(s, e, label, phrase):
        return DateRange(s, e, label, phrase) if s and e and s <= e else None

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        d = _mk(int(m[1]), int(m[2]), int(m[3]))
        return done(d, d, d.isoformat() if d else "", m[0])

    if re.search(r"\bday before yesterday\b", t):
        d = today - timedelta(days=2)
        return done(d, d, "the day before yesterday", "day before yesterday")
    m = re.search(r"\byesterday\b", t)
    if m:
        d = today - timedelta(days=1)
        return done(d, d, "yesterday", m[0])
    m = re.search(r"\btoday\b", t)
    if m:
        return done(today, today, "today", m[0])

    m = re.search(r"\b(a|an|one|two|three|four|five|six|seven|\d{1,3}) days? ago\b", t)
    if m:
        d = today - timedelta(days=_n(m[1]))
        return done(d, d, m[0], m[0])
    m = re.search(r"\b(?:last|past) (a|an|one|two|three|four|five|six|seven|\d{1,3}) days\b", t)
    if m:
        n = _n(m[1])
        return done(today - timedelta(days=n - 1), today, f"the last {n} days", m[0])
    m = re.search(r"\b(?:last|past) (a|an|one|two|three|four|\d{1,2}) weeks\b", t)
    if m:
        n = _n(m[1])
        return done(today - timedelta(weeks=n), today, f"the last {n} weeks", m[0])

    monday = today - timedelta(days=today.weekday())
    m = re.search(r"\blast week\b", t)
    if m:
        return done(monday - timedelta(days=7), monday - timedelta(days=1), "last week", m[0])
    m = re.search(r"\bthis week\b", t)
    if m:
        return done(monday, today, "this week", m[0])
    m = re.search(r"\blast month\b", t)
    if m:
        first = today.replace(day=1)
        end = first - timedelta(days=1)
        return done(end.replace(day=1), end, end.strftime("%B %Y"), m[0])
    m = re.search(r"\bthis month\b", t)
    if m:
        return done(today.replace(day=1), today, "this month", m[0])
    m = re.search(r"\blast year\b", t)
    if m:
        y = today.year - 1
        return done(date(y, 1, 1), date(y, 12, 31), str(y), m[0])
    m = re.search(r"\bthis year\b", t)
    if m:
        return done(date(today.year, 1, 1), today, str(today.year), m[0])

    # "(last|on|this) tuesday" -> the most recent such weekday strictly before today
    m = re.search(rf"\b(?:(last|on|this) )?({_DOW})\b", t)
    if m and (m[1] or re.fullmatch(rf"(?:what.*)?{_DOW}", t) or True):
        if m[1] or not re.search(rf"\b(?:{_MON})\b", t):
            back = (today.weekday() - _DAYS[m[2]]) % 7 or 7
            d = today - timedelta(days=back)
            return done(d, d, f"{m[2].title()} ({d.isoformat()})", m[0])

    def year_ok(y: int) -> bool:
        return _MIN_YEAR <= y <= today.year

    # "3 march 2025" / "3rd of march"
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?: of)? ({_MON})(?: (\d{{4}}))?\b", t)
    if m:
        return _day_in_month(int(m[1]), _MONTHS[m[2]], m[3], today, m[0])
    # "march 3rd 2025"
    m = re.search(rf"\b({_MON}) (\d{{1,2}})(?:st|nd|rd|th)?(?: (\d{{4}}))?\b", t)
    if m and not re.match(r"\d{4}$", m[2]):
        return _day_in_month(int(m[2]), _MONTHS[m[1]], m[3], today, m[0])
    # "march 2024" / "in august"
    m = re.search(rf"\b(?:(in|of|during|from) )?({_MON}) ?(\d{{4}})?\b", t)
    if m and (m[1] or m[3]):
        mon = _MONTHS[m[2]]
        if m[3]:
            y = int(m[3])
            if not year_ok(y):
                return None
        else:
            y = today.year - (1 if mon > today.month else 0)
        s, e = _month_range(y, mon, today)
        return done(s, e, f"{date(y, mon, 1):%B %Y}", m[0])
    # a bare year
    m = re.search(r"\b(?:(in|from|during|of|the year) )?(\d{4})\b", t)
    if m and year_ok(int(m[2])) and (m[1] or t.strip() == m[2] or re.search(r"\b(?:taken|photos?|pictures?|shot)\b", t)):
        y = int(m[2])
        return done(date(y, 1, 1), min(date(y, 12, 31), today), str(y), m[0])
    return None


def _day_in_month(day: int, mon: int, year_s: Optional[str], today: date, phrase: str) -> Optional[DateRange]:
    if year_s:
        y = int(year_s)
        if not (_MIN_YEAR <= y <= today.year):
            return None
        d = _mk(y, mon, day)
    else:
        d = _mk(today.year, mon, day)
        if d and d > today:
            d = _mk(today.year - 1, mon, day)
    if d is None:
        return None
    return DateRange(d, d, d.isoformat(), phrase)


def resolve_day(text: str, today: Optional[date] = None) -> Optional[date]:
    """A single calendar day, or None when the phrase is absent or spans several days."""
    r = resolve_range(text, today)
    return r.start if r and r.start == r.end else None


def local_day_bounds_utc(start: date, end: date) -> tuple[str, str]:
    """
    [start 00:00 local, end+1 00:00 local) as 'YYYY-MM-DD HH:MM:SS' UTC strings,
    the format SQLite's CURRENT_TIMESTAMP writes - so a local "yesterday" that
    straddles UTC midnight is still selected correctly.
    """
    def conv(d: date) -> str:
        local = datetime.combine(d, time.min).astimezone()
        return local.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return conv(start), conv(end + timedelta(days=1))
