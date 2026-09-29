"""
modules/chronos/reminders.py

ReminderService: the scheduling brain behind Chronos reminders.

Everything that needs to *decide* something about a stored reminder lives
here, so ``ChronosEngine`` stays a thin intent router and the rules can be
tested with a fake clock and an in-memory database:

  #81 / #84  firing recurring reminders and advancing them to the next slot
  #82        location-triggered reminders (arm when away, fire on arrival)
  #83        snooze (postpone a waiting reminder, or re-arm one that fired)
  #85        per-reminder timezone (every wall-clock calculation uses the
             reminder's own zone, not the global one)
  #87        holiday-aware scheduling (user-marked days, optionally public
             holidays): skip a recurring occurrence, or roll a one-shot to
             the next non-holiday day
  #89        missed-reminder handling: anything overdue by more than a grace
             period is fired *once*, flagged as missed, and announced as a
             single digest instead of being silently dropped
  #90        ICS export / import glue around ``chronos.ics``

The service talks to a ``MnemosyneDB``-shaped store (see
``modules/mnemosyne/db.py``); it never touches SQL itself. It performs no
speaking or printing - ``tick()`` and ``on_location()`` *return* what fired
and the caller decides how to announce it.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import ics as ics_mod
from .recurrence import (
    DEFAULT_HOUR,
    Recurrence,
    from_rrule,
    next_after,
    to_rrule,
    to_zoneinfo,
)

logger = logging.getLogger(__name__)

# A reminder overdue by more than this when the scheduler sees it is "missed".
# Comfortably larger than the scheduler's poll interval so an ordinary tick
# is never mistaken for a missed reminder.
MISSED_GRACE = timedelta(minutes=15)
DEFAULT_SNOOZE = timedelta(minutes=10)
MIN_SNOOZE = timedelta(minutes=1)
MAX_SNOOZE = timedelta(days=7)
# "snooze it" refers to a reminder that fired within this window.
SNOOZE_WINDOW = timedelta(hours=12)
DEFAULT_RADIUS_M = 150.0
# A location reminder re-arms only once the device is this many radii away,
# so GPS jitter at the edge of the radius can't fire it spuriously.
ARM_DISTANCE_FACTOR = 1.5
_MAX_HOLIDAY_ROLL_DAYS = 60


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def parse_iso(value: Any, default_tz: Any = timezone.utc) -> Optional[datetime]:
    """Parse a stored ISO timestamp into an aware datetime (naive values are
    read in *default_tz*). Returns ``None`` for empty/garbage input."""
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=default_tz)
    return dt


def to_iso(dt: datetime) -> str:
    """ISO string with whole-second precision (what the store compares on)."""
    return dt.replace(microsecond=0).isoformat()


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def describe_when(dt: Optional[datetime], tz: Any, now: datetime) -> str:
    """'today at 9:00 AM', 'tomorrow at 7:00 AM', 'Friday at 6:00 PM',
    'Monday, October 12 at 9:00 AM'."""
    if dt is None:
        return "no set time"
    local = dt.astimezone(tz)
    clock = local.strftime("%I:%M %p").lstrip("0")
    delta = (local.date() - now.astimezone(tz).date()).days
    if delta == 0:
        return f"today at {clock}"
    if delta == 1:
        return f"tomorrow at {clock}"
    if delta == -1:
        return f"yesterday at {clock}"
    if 1 < delta < 7:
        return f"{local.strftime('%A')} at {clock}"
    return f"{local.strftime('%A, %B')} {local.day} at {clock}"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Fired:
    """One reminder occurrence that the scheduler just claimed."""

    id: int
    text: str
    fired_at: datetime
    tz: Any
    due: Optional[datetime] = None       # scheduled moment (None: location)
    missed: bool = False
    recurring: bool = False
    series_ended: bool = False
    place: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "due": self.due.isoformat() if self.due else None,
            "fired_at": self.fired_at.isoformat(),
            "missed": self.missed,
            "recurring": self.recurring,
            "series_ended": self.series_ended,
            "place": self.place,
        }


@dataclass
class ImportResult:
    created: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ReminderService:
    """
    Parameters
    ----------
    store:
        A ``MnemosyneDB``-shaped object (``add_reminder_full``,
        ``get_due_reminders_full``, ``claim_reminder`` ...).
    default_tz:
        tzinfo used for reminders that carry no timezone of their own.
    clock:
        Zero-arg callable returning the current aware datetime. Tests inject
        a fake; production uses UTC now.
    public_holiday_lookup:
        Optional ``callable(iso_date) -> Optional[str]`` returning a holiday
        name. When given, public holidays count as holidays for reminders
        flagged ``skip_holidays`` (user-marked days always count).
    """

    def __init__(
        self,
        store: Any,
        default_tz: Any = timezone.utc,
        *,
        clock: Optional[Callable[[], datetime]] = None,
        public_holiday_lookup: Optional[Callable[[str], Optional[str]]] = None,
        missed_grace: timedelta = MISSED_GRACE,
        default_snooze: timedelta = DEFAULT_SNOOZE,
    ) -> None:
        self.store = store
        self.default_tz = default_tz
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._public_lookup = public_holiday_lookup
        self.missed_grace = missed_grace
        self.default_snooze = default_snooze
        key = getattr(default_tz, "key", None)
        self.default_tz_name: str = key or "UTC"

    # -- basics -----------------------------------------------------------

    def now(self) -> datetime:
        return self._clock()

    def zone_for(self, row: dict[str, Any]) -> Any:
        return to_zoneinfo(row.get("tz"), self.default_tz)

    def tz_name_for(self, row: dict[str, Any]) -> str:
        name = row.get("tz")
        if name:
            try:
                ZoneInfo(name)
                return name
            except (ZoneInfoNotFoundError, KeyError, ValueError):
                pass
        return self.default_tz_name

    def _due_of(self, row: dict[str, Any]) -> Optional[datetime]:
        return parse_iso(row.get("due_time"), self.zone_for(row))

    @staticmethod
    def _rule_of(row: dict[str, Any]) -> Optional[Recurrence]:
        raw = row.get("recurrence")
        if not raw:
            return None
        rule = Recurrence.from_json(raw)
        if rule is None:
            logger.warning("Reminder %s has an unreadable recurrence; treating as one-shot.", row.get("id"))
        return rule

    # -- holidays (#87) ---------------------------------------------------

    def holiday_label(self, day: date) -> Optional[str]:
        """Why *day* counts as a holiday (a user label or public-holiday
        name), or ``None`` if it doesn't."""
        iso = day.isoformat()
        getter = getattr(self.store, "get_holiday", None)
        if getter is not None:
            try:
                label = getter(iso)
            except Exception:
                logger.exception("holiday lookup failed for %s", iso)
                label = None
            if label is not None:
                return label or "a day off"
        if self._public_lookup is not None:
            try:
                name = self._public_lookup(iso)
            except Exception:
                logger.warning("public-holiday lookup failed for %s", iso, exc_info=True)
                name = None
            if name:
                return str(name)
        return None

    def next_non_holiday(self, due: datetime, tz: Any) -> tuple[datetime, int]:
        """Roll *due* forward whole days (same wall-clock time) until it is
        not a holiday. Returns ``(new_due, days_rolled)``."""
        local = due.astimezone(tz)
        day = local.date()
        rolled = 0
        while rolled < _MAX_HOLIDAY_ROLL_DAYS and self.holiday_label(day):
            day += timedelta(days=1)
            rolled += 1
        if rolled == 0:
            return due, 0
        naive = datetime.combine(day, dtime(local.hour, local.minute))
        return naive.replace(tzinfo=tz).astimezone(timezone.utc).astimezone(tz), rolled

    # -- creation ---------------------------------------------------------

    def create(
        self,
        text: str,
        due: Optional[datetime],
        *,
        tz_name: Optional[str] = None,
        rule: Optional[Recurrence] = None,
        skip_holidays: bool = False,
        place: Optional[dict[str, Any]] = None,
        ics_uid: Optional[str] = None,
        snooze_of: Optional[int] = None,
        snooze_count: int = 0,
    ) -> int:
        """Persist a reminder. *place* is ``{label, lat, lon, radius_m, armed}``
        for a location reminder (which has no *due*)."""
        place = place or {}
        return self.store.add_reminder_full(
            text,
            to_iso(due) if due is not None else None,
            recurrence=rule.to_json() if rule else None,
            tz=tz_name,
            skip_holidays=skip_holidays,
            place_label=place.get("label"),
            place_lat=place.get("lat"),
            place_lon=place.get("lon"),
            radius_m=place.get("radius_m"),
            armed=bool(place.get("armed")),
            snooze_of=snooze_of,
            snooze_count=snooze_count,
            ics_uid=ics_uid,
        )

    # -- the scheduler tick (#81, #87, #89) -------------------------------

    def tick(self, now: Optional[datetime] = None) -> list[Fired]:
        """
        Fire every due time-based reminder exactly once and return them.

        Safe to call from several threads/processes: each reminder is claimed
        with a compare-and-swap in the store, and only the caller that wins
        the claim gets it back.
        """
        now = (now or self.now()).astimezone(timezone.utc)
        try:
            rows = self.store.get_due_reminders_full(now.isoformat())
        except Exception:
            logger.exception("tick(): could not read due reminders.")
            return []
        fired: list[Fired] = []
        for row in rows:
            try:
                item = self._process_due(row, now)
            except Exception:
                logger.exception("tick(): failed processing reminder %s.", row.get("id"))
                continue
            if item is not None:
                fired.append(item)
        return fired

    def _process_due(self, row: dict[str, Any], now: datetime) -> Optional[Fired]:
        due = self._due_of(row)
        if due is None:
            return None
        tz = self.zone_for(row)
        rule = self._rule_of(row)

        if row.get("skip_holidays") and self.holiday_label(due.astimezone(tz).date()):
            self._skip_for_holiday(row, due, tz, rule, now)
            return None

        missed = (now - due) > self.missed_grace
        fired_iso = to_iso(now)

        if rule is not None:
            fire_rule = rule.consume()
            nxt = None if fire_rule is None else next_after(fire_rule, max(now, due), tz)
            if nxt is None:
                new_recurrence = row["recurrence"]      # keep the record
            else:
                new_recurrence = fire_rule.to_json()    # type: ignore[union-attr]
            ok = self.store.advance_reminder(
                row["id"], row["due_time"],
                to_iso(nxt) if nxt is not None else None,
                new_recurrence, fired_iso, missed=missed,
            )
            ended = nxt is None
        else:
            ok = self.store.claim_reminder(row["id"], fired_iso, missed=missed)
            ended = True
        if not ok:
            return None
        return Fired(
            id=row["id"], text=row.get("text") or "", fired_at=now, tz=tz, due=due,
            missed=missed, recurring=rule is not None, series_ended=ended and rule is not None,
        )

    def _skip_for_holiday(
        self, row: dict[str, Any], due: datetime, tz: Any,
        rule: Optional[Recurrence], now: datetime,
    ) -> None:
        if rule is not None:
            # The skipped occurrence is not "used up": COUNT is left alone.
            nxt = next_after(rule, max(now, due), tz)
            self.store.advance_reminder(
                row["id"], row["due_time"],
                to_iso(nxt) if nxt is not None else None,
                row["recurrence"], None, skipped=1, fired=False,
            )
            logger.info("Reminder %s skipped a holiday occurrence.", row["id"])
            return
        new_due, rolled = self.next_non_holiday(due, tz)
        if rolled:
            self.store.reschedule_reminder(row["id"], to_iso(new_due), skipped=1)
            logger.info("Reminder %s rolled %d day(s) past a holiday.", row["id"], rolled)

    # -- location reminders (#82) -----------------------------------------

    def arm_if_outside(self, reminder_id: int, lat: float, lon: float) -> bool:
        """Arm a freshly created location reminder if the device is already
        away from the place (otherwise it must first leave, then return)."""
        row = self.store.get_reminder(reminder_id)
        if not row or row.get("place_lat") is None:
            return False
        radius = float(row.get("radius_m") or DEFAULT_RADIUS_M)
        if haversine_m(lat, lon, row["place_lat"], row["place_lon"]) > radius * ARM_DISTANCE_FACTOR:
            self.store.set_reminder_armed(reminder_id, True)
            return True
        return False

    def on_location(self, lat: float, lon: float, now: Optional[datetime] = None) -> list[Fired]:
        """Feed a device position in. Arms location reminders while the
        device is away and fires them when it comes within their radius."""
        now = (now or self.now()).astimezone(timezone.utc)
        fired: list[Fired] = []
        try:
            rows = self.store.get_location_reminders()
        except Exception:
            logger.exception("on_location(): could not read location reminders.")
            return []
        for row in rows:
            try:
                radius = float(row.get("radius_m") or DEFAULT_RADIUS_M)
                dist = haversine_m(lat, lon, row["place_lat"], row["place_lon"])
                if not row.get("armed"):
                    if dist > radius * ARM_DISTANCE_FACTOR:
                        self.store.set_reminder_armed(row["id"], True)
                    continue
                if dist <= radius and self.store.claim_location_reminder(row["id"], to_iso(now)):
                    fired.append(Fired(
                        id=row["id"], text=row.get("text") or "", fired_at=now,
                        tz=self.zone_for(row), place=row.get("place_label") or "the place",
                    ))
            except Exception:
                logger.exception("on_location(): failed for reminder %s.", row.get("id"))
        return fired

    # -- snooze (#83) -----------------------------------------------------

    def snooze(
        self,
        *,
        reminder_id: Optional[int] = None,
        text_hint: Optional[str] = None,
        duration: Optional[timedelta] = None,
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Snooze a reminder.

        * A reminder that already fired (the usual "snooze it"): a one-shot
          copy is scheduled *duration* from now, linked back to the original
          via ``snooze_of`` so a recurring series is left untouched.
        * A reminder that is still waiting: its due time is pushed back.

        Returns ``{"ok": bool, ...}``; on failure ``reason`` is one of
        ``none`` (nothing to snooze), ``cancelled``, ``location``.
        """
        now = (now or self.now()).astimezone(timezone.utc)
        duration = duration or self.default_snooze
        duration = max(MIN_SNOOZE, min(duration, MAX_SNOOZE))
        since = now - SNOOZE_WINDOW

        row: Optional[dict[str, Any]] = None
        fired = False
        if reminder_id is not None:
            row = self.store.get_reminder(reminder_id)
            fired = bool(row) and self._is_fired_instance(row, since)
        elif text_hint:
            for cand in self.store.find_reminders_by_text(text_hint, status=None):
                fa = parse_iso(cand.get("fired_at"))
                if fa is not None and fa >= since and cand.get("status") != "cancelled":
                    row, fired = cand, True
                    break
            if row is None:
                pending = self.store.find_reminders_by_text(text_hint, status="pending")
                row = pending[0] if pending else None
        else:
            row = self.store.most_recent_fired_reminder(since_iso=to_iso(since))
            fired = row is not None

        if row is None:
            return {"ok": False, "reason": "none"}
        if row.get("status") == "cancelled":
            return {"ok": False, "reason": "cancelled"}
        tz = self.zone_for(row)

        if not fired:
            due = self._due_of(row)
            if due is None:
                return {"ok": False, "reason": "location"}
            new_due = max(due, now) + duration
            ok = self.store.reschedule_reminder(row["id"], to_iso(new_due))
            return {
                "ok": bool(ok), "mode": "postponed", "id": row["id"], "text": row["text"],
                "due": new_due, "tz": tz, "duration": duration,
                "snooze_count": int(row.get("snooze_count") or 0),
            }

        new_due = now + duration
        root = row.get("snooze_of") or row["id"]
        count = int(row.get("snooze_count") or 0) + 1
        new_id = self.create(
            row["text"], new_due, tz_name=row.get("tz"), snooze_of=root, snooze_count=count,
        )
        return {
            "ok": True, "mode": "new", "id": new_id, "text": row["text"], "due": new_due,
            "tz": tz, "duration": duration, "snooze_count": count,
        }

    @staticmethod
    def _is_fired_instance(row: dict[str, Any], since: datetime) -> bool:
        if row.get("status") != "pending":
            return True
        fa = parse_iso(row.get("fired_at"))
        return bool(row.get("recurrence")) and fa is not None and fa >= since

    # -- listing / cancelling ---------------------------------------------

    def list_pending(self) -> list[dict[str, Any]]:
        return self.store.list_reminders("pending")

    def recently_missed(self, days: int = 7, now: Optional[datetime] = None) -> list[dict[str, Any]]:
        now = (now or self.now()).astimezone(timezone.utc)
        cutoff = now - timedelta(days=days)
        out = []
        for row in self.store.list_reminders(None):
            fa = parse_iso(row.get("fired_at"))
            if row.get("missed") and fa is not None and fa >= cutoff:
                out.append(row)
        out.sort(key=lambda r: r.get("fired_at") or "", reverse=True)
        return out

    def find_pending(self, text_hint: str) -> list[dict[str, Any]]:
        return self.store.find_reminders_by_text(text_hint, status="pending")

    def cancel(self, reminder_id: int) -> bool:
        return bool(self.store.set_reminder_status(reminder_id, "cancelled"))

    def describe_row(self, row: dict[str, Any], now: Optional[datetime] = None) -> str:
        """One line for a reminder, suitable for speech."""
        now = (now or self.now()).astimezone(timezone.utc)
        tz = self.zone_for(row)
        text = row.get("text") or "reminder"
        rule = self._rule_of(row)
        if row.get("place_lat") is not None and not row.get("due_time"):
            return f"{text} (when you get to {row.get('place_label') or 'the place'})"
        due = self._due_of(row)
        tz_note = f" ({self.tz_name_for(row)})" if row.get("tz") else ""
        if rule is not None:
            return f"{text} ({rule.describe(tz)}; next {describe_when(due, tz, now)}{tz_note})"
        extra = " (skips holidays)" if row.get("skip_holidays") else ""
        return f"{text} ({describe_when(due, tz, now)}{tz_note}){extra}"

    # -- announcements ----------------------------------------------------

    def format_fired(self, fired: list[Fired], now: Optional[datetime] = None) -> list[str]:
        """Turn fired reminders into spoken lines. On-time ones are announced
        individually; all missed ones are folded into a single digest."""
        now = (now or self.now()).astimezone(timezone.utc)
        lines: list[str] = []
        missed = [f for f in fired if f.missed]
        for f in fired:
            if f.missed:
                continue
            prefix = f"You've arrived at {f.place}. " if f.place else ""
            lines.append(f"{prefix}Reminder: {f.text}")
        if len(missed) == 1:
            f = missed[0]
            lines.append(f"You missed a reminder from {describe_when(f.due, f.tz, now)}: {f.text}.")
        elif missed:
            parts = [f"{f.text} ({describe_when(f.due, f.tz, now)})" for f in missed[:5]]
            more = f", and {len(missed) - 5} more" if len(missed) > 5 else ""
            lines.append(
                f"While you were away, you missed {len(missed)} reminders: "
                + "; ".join(parts) + more + "."
            )
        return lines

    # -- ICS (#90) --------------------------------------------------------

    def export_ics(self, now: Optional[datetime] = None) -> tuple[str, int, list[str]]:
        """Serialise pending reminders. Returns ``(ics_text, exported, notes)``;
        *notes* explain anything that could not be represented."""
        now = (now or self.now()).astimezone(timezone.utc)
        notes: list[str] = []
        items: list[dict[str, Any]] = []
        for row in self.store.list_reminders("pending"):
            text = row.get("text") or "Reminder"
            due = self._due_of(row)
            if due is None:
                notes.append(f"'{text}' is a location reminder and can't be exported.")
                continue
            tz = self.zone_for(row)
            rule = self._rule_of(row)
            rrule = None
            if rule is not None:
                rrule = to_rrule(rule, tz)
                if rrule is None:
                    notes.append(
                        f"'{text}' repeats on a schedule with no calendar equivalent; "
                        "only its next occurrence was exported."
                    )
            items.append({
                "uid": row.get("ics_uid") or ics_mod.make_uid(row["id"]),
                "summary": text,
                "start": due,
                "tzid": self.tz_name_for(row) if (rrule or row.get("tz")) else None,
                "rrule": rrule,
                "skip_holidays": bool(row.get("skip_holidays")),
            })
        return ics_mod.build_ics(items, now=now), len(items), notes

    def import_ics(self, text: str, now: Optional[datetime] = None) -> ImportResult:
        """Import VEVENT/VTODO items as reminders (see ``chronos.ics``)."""
        now = (now or self.now()).astimezone(timezone.utc)
        items, notes = ics_mod.parse_ics(text)
        result = ImportResult(warnings=list(notes))
        for it in items:
            try:
                self._import_item(it, now, result)
            except Exception as exc:  # one bad entry must not sink the file
                logger.exception("import_ics: failed on %r", it.summary)
                result.skipped.append(f"'{it.summary}': {exc}")
        return result

    def _import_item(self, it: "ics_mod.ParsedItem", now: datetime, result: ImportResult) -> None:
        name = it.summary
        for w in it.warnings:
            result.warnings.append(f"'{name}': {w}")

        tz = self.default_tz
        if it.all_day:
            start = datetime.combine(it.start, dtime(DEFAULT_HOUR, 0)).replace(tzinfo=tz)  # type: ignore[arg-type]
            result.warnings.append(f"'{name}' is an all-day entry; it will remind you at 9:00 AM.")
        elif it.start.tzinfo is not None:  # type: ignore[union-attr]
            start = it.start  # type: ignore[assignment]
        else:
            start = it.start.replace(tzinfo=tz)  # type: ignore[union-attr]

        rz = to_zoneinfo(it.tzid, tz)
        tz_name = it.tzid if it.tzid else None

        # Already have it? (our own export coming back, or a repeated import)
        uid = it.uid
        own = ics_mod.own_reminder_id(uid)
        if own is not None:
            existing = self.store.get_reminder(own)
            if existing and existing.get("status") == "pending":
                result.skipped.append(f"'{name}' is already in your reminders.")
                return
        if uid:
            existing = self.store.find_reminder_by_ics_uid(uid)
            if existing and existing.get("status") == "pending":
                result.skipped.append(f"'{name}' is already in your reminders.")
                return

        lead = it.alarm_offset or timedelta(0)
        fire0 = start + lead
        if it.rrule and lead and fire0.astimezone(rz).date() != start.astimezone(rz).date():
            result.warnings.append(
                f"'{name}': the alert is set more than a day before it starts and can't be "
                "applied to a repeating entry; reminding at the start time instead."
            )
            lead, fire0 = timedelta(0), start

        rule: Optional[Recurrence] = None
        if it.rrule:
            try:
                rule = from_rrule(it.rrule, fire0.astimezone(rz), rz)
            except ValueError as exc:
                result.warnings.append(
                    f"'{name}': can't represent its repeat pattern ({exc}); "
                    "imported as a single reminder."
                )

        if rule is not None:
            due = fire0 if fire0 > now else next_after(rule, now, rz)
            if due is None:
                result.skipped.append(f"'{name}' repeats, but its series has already finished.")
                return
        else:
            if fire0 <= now:
                result.skipped.append(f"'{name}' is in the past.")
                return
            due = fire0

        if self.store.reminder_exists(name, to_iso(due)):
            result.skipped.append(f"'{name}' is already in your reminders.")
            return

        rid = self.create(
            name, due, tz_name=tz_name, rule=rule,
            skip_holidays=it.skip_holidays, ics_uid=uid,
        )
        result.created.append({
            "id": rid, "text": name, "due": to_iso(due),
            "recurrence": rule.describe(rz) if rule else None,
        })