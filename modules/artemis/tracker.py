"""
artemis/tracker.py

Persistent tracker for habits and goals.
State is stored as JSON; all mutations are written atomically via a
temporary file so a crash mid-write never corrupts the state file.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from . import extras

try:  # stdlib on 3.9+, but tzdata may be missing on some Windows installs
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A relative default path is anchored to the Hestia project root rather
# than Path.resolve()'s implicit cwd — otherwise where the state file lands
# depends on whether the process was launched via `python main.py`,
# `python web_ui.py`, or the Telegram bot subprocess, each of which may have
# a different working directory. Same convention as modules/pluto/config.py
# (db_path) and modules/orpheus/engine.py (_DB_PATH).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _PROJECT_ROOT / "data" / "artemis_state.json"
_EMPTY_STATE: dict[str, Any] = {"habits": {}, "goals": {}}

_MAX_HABIT_NAME_LEN = 256
_MAX_GOAL_NAME_LEN = 256
_PROGRESS_MIN = 0.0
_PROGRESS_MAX = 1.0

# Per-day completion history (backlog #126/#161/#183). Capped so the JSON
# state file can't grow without bound; ~13 months is plenty for heatmaps and
# correlations. Oldest dates are dropped first.
HISTORY_CAP_DAYS = 400

# Habit grace periods (#123) and pauses (#128).
MAX_GRACE_DAYS = 7          # most missed days a streak may survive
_MAX_MILESTONES = 12        # per goal
_MAX_PAUSES = 24            # pause intervals remembered per habit; oldest dropped first

_VALID_PRIORITIES: frozenset[str] = frozenset({"low", "medium", "high"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TrackerError(Exception):
    """Base exception for ArtemisTracker failures."""


class HabitNotFoundError(TrackerError):
    """Raised when an operation targets a habit that does not exist."""


class GoalNotFoundError(TrackerError):
    """Raised when an operation targets a goal that does not exist."""


class StateCorruptedError(TrackerError):
    """Raised when the persisted state cannot be parsed or is structurally invalid."""


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class Habit:
    """In-memory representation of a single habit."""

    __slots__ = (
        "name", "streak", "last_done",
        "best_streak", "total_completions", "created_at",
        "history", "history_since", "grace_days", "pauses", "times",
    )

    def __init__(
        self,
        name: str,
        streak: int = 0,
        last_done: str = "",
        best_streak: int = 0,
        total_completions: int = 0,
        created_at: str = "",
        history: Optional[list[str]] = None,
        history_since: str = "",
        grace_days: Optional[int] = None,
        pauses: Optional[list] = None,
        times: Optional[list] = None,
    ) -> None:
        self.name = name
        self.streak = streak
        self.last_done = last_done  # ISO date string or ""
        # Guard against legacy/corrupt data where best_streak wasn't
        # persisted yet but the current streak already exceeds it.
        self.best_streak = max(best_streak, streak)
        self.total_completions = total_completions
        self.created_at = created_at or _utc_now()
        # ISO dates of completion, oldest first. Empty for habits that
        # pre-date history recording; ``history_since`` is the first day we
        # can vouch for (days before it are unknown, not "missed").
        self.history: list[str] = _clean_history(history or [])
        self.history_since = history_since
        # Missed days a streak survives; None means "use the tracker's default" (#123).
        self.grace_days: Optional[int] = _clean_grace(grace_days)
        # [[start_iso, end_iso_or_""], ...]: the habit is paused on start <= day < end;
        # "" means open-ended (#128).
        self.pauses: list[list[str]] = _clean_pauses(pauses or [])
        # Minute-of-day (local) of recent completions, for smart nudges (#129).
        self.times: list[int] = _clean_times(times or [])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "streak": self.streak,
            "last_done": self.last_done,
            "best_streak": self.best_streak,
            "total_completions": self.total_completions,
            "created_at": self.created_at,
        }
        # Only written once history exists, so untouched state files keep
        # their exact old shape.
        if self.history:
            out["history"] = list(self.history)
            out["history_since"] = self.history_since
        if self.grace_days is not None:
            out["grace_days"] = self.grace_days
        if self.pauses:
            out["pauses"] = [list(p) for p in self.pauses]
        if self.times:
            out["times"] = list(self.times)
        return out

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> "Habit":
        return cls(
            name=name,
            streak=int(raw.get("streak", 0)),
            last_done=str(raw.get("last_done", "")),
            best_streak=int(raw.get("best_streak", 0)),
            total_completions=int(raw.get("total_completions", 0)),
            created_at=str(raw.get("created_at", "")),
            history=raw.get("history") if isinstance(raw.get("history"), list) else None,
            history_since=str(raw.get("history_since", "") or ""),
            grace_days=raw.get("grace_days"),
            pauses=raw.get("pauses") if isinstance(raw.get("pauses"), list) else None,
            times=raw.get("times") if isinstance(raw.get("times"), list) else None,
        )

    # ------------------------------------------------------------------
    # Business logic
    # ------------------------------------------------------------------

    def _record_history(self, day_iso: str) -> None:
        if not self.history_since:
            self.history_since = day_iso
        if day_iso not in self.history:
            self.history.append(day_iso)
            self.history = _clean_history(self.history)

    # ------------------------------------------------------------------
    # Pauses (#128) and grace periods (#123)
    # ------------------------------------------------------------------

    def paused_on(self, day: date) -> bool:
        iso = day.isoformat()
        return any(start <= iso and (not end or iso < end) for start, end in self.pauses)

    def pause(self, today: date, days: Optional[int] = None) -> bool:
        """Pause from *today* (for *days* days, or until resumed). False if already paused today."""
        if self.paused_on(today):
            return False
        end = (today + timedelta(days=days)).isoformat() if days else ""
        self.pauses.append([today.isoformat(), end])
        self.pauses = _clean_pauses(self.pauses)
        return True

    def resume(self, today: date) -> bool:
        """End the pause covering *today*, so *today* counts as a normal day. False if not paused."""
        iso = today.isoformat()
        for p in self.pauses:
            if p[0] <= iso and (not p[1] or iso < p[1]):
                p[1] = iso
                self.pauses = _clean_pauses(self.pauses)      # a zero-length pause is dropped
                return True
        return False

    def missed_days(self, today: date) -> int:
        """Days strictly between the last completion and *today* that were neither done nor paused."""
        try:
            last = date.fromisoformat(self.last_done)
        except ValueError:
            return 0
        gap = (today - last).days - 1
        return sum(1 for i in range(1, gap + 1) if not self.paused_on(last + timedelta(days=i)))

    def streak_alive(self, today: date, default_grace: int = 0) -> bool:
        """Would completing the habit today continue the streak (rather than restart it)?"""
        if self.streak <= 0 or not self.last_done:
            return False
        if self.last_done >= today.isoformat():
            return True
        return self.missed_days(today) <= self._grace(default_grace)

    def streak_ends_if_skipped_today(self, today: date, default_grace: int = 0) -> bool:
        """An alive streak that today's skip would break: no grace left, not paused, not done today."""
        return (self.streak > 0 and self.last_done < today.isoformat() and not self.paused_on(today)
                and self.streak_alive(today, default_grace)
                and self.missed_days(today) + 1 > self._grace(default_grace))

    def _grace(self, default_grace: int) -> int:
        return self.grace_days if self.grace_days is not None else max(0, int(default_grace))

    def window_stats(self, start: date, end: date, today: date) -> tuple[int, int]:
        """
        (days done, days possible) over start..end inclusive. A day is "possible" only if the
        habit's history covers it and it wasn't paused; today is possible only once done, since
        the day isn't over. (0, 0) when the habit has no history that reaches the window.
        """
        if self.history_since:
            since = date.fromisoformat(self.history_since)
        elif self.total_completions == 0:
            try:
                since = date.fromisoformat(self.created_at[:10])
            except ValueError:
                return 0, 0
        else:
            return 0, 0                           # completed before history was recorded: unknown
        done_set = set(self.history)
        done = possible = 0
        d = max(start, since)
        while d <= end:
            iso = d.isoformat()
            if not self.paused_on(d) and (d != today or iso in done_set):
                possible += 1
                done += iso in done_set
            d += timedelta(days=1)
        return done, possible

    def complete(self, today: date, default_grace: int = 0, at_minute: Optional[int] = None) -> dict[str, Any]:
        """
        Mark the habit as completed for *today*, updating the streak,
        best streak, and completion count.

        Rules
        -----
        - Same day  → no-op (idempotent), reports the existing state.
        - Yesterday → extend streak.
        - Missed days → extend if they fit the habit's grace period (``grace_days``, else
          *default_grace*); days the habit was paused never count as missed (#123, #128).
        - Older / never done → reset streak to 1.
        - Completing a paused habit resumes it.

        Returns
        -------
        dict with ``already_done``, ``streak``, ``best_streak``,
        ``is_new_best``, and ``total_completions`` — enough for the
        caller to build a milestone callout without re-reading state.
        """
        today_iso = today.isoformat()
        yesterday_iso = (today - timedelta(days=1)).isoformat()

        if self.last_done == today_iso:
            logger.debug("Habit %r already completed today; no-op.", self.name)
            # Completed before history existed: backfill today so the
            # heatmap doesn't show a done habit as blank.
            self._record_history(today_iso)
            return {
                "already_done": True,
                "streak": self.streak,
                "best_streak": self.best_streak,
                "is_new_best": False,
                "total_completions": self.total_completions,
            }

        resumed = self.resume(today)
        grace_used = 0
        if self.last_done == yesterday_iso:
            self.streak += 1
        elif self.last_done and self.last_done < today_iso and self.streak_alive(today, default_grace):
            grace_used = self.missed_days(today)
            self.streak += 1
        else:
            self.streak = 1

        self.last_done = today_iso
        self._record_history(today_iso)
        if at_minute is not None and 0 <= at_minute < 1440:
            self.times = _clean_times(self.times + [at_minute])
        self.total_completions += 1
        is_new_best = self.streak > self.best_streak
        if is_new_best:
            self.best_streak = self.streak

        logger.debug(
            "Habit %r completed. streak=%d best_streak=%d total=%d last_done=%s",
            self.name, self.streak, self.best_streak,
            self.total_completions, self.last_done,
        )
        return {
            "already_done": False,
            "streak": self.streak,
            "best_streak": self.best_streak,
            "is_new_best": is_new_best,
            "total_completions": self.total_completions,
            "grace_used": grace_used,
            "resumed": resumed,
        }

    def consistency_pct(self, today: date) -> float:
        """
        Percentage of days since this habit was added that it's been
        completed. Unlike ``streak`` (which a single missed day resets
        to zero), this doesn't erase history on a break — it's a
        best-effort figure since ``created_at`` may be absent on habits
        persisted before this field existed, in which case 0.0 is
        returned rather than guessing.
        """
        if not self.created_at:
            return 0.0
        try:
            created = date.fromisoformat(self.created_at[:10])
        except ValueError:
            return 0.0
        days_tracked = max(1, (today - created).days + 1)
        return round(min(1.0, self.total_completions / days_tracked) * 100, 1)


class Goal:
    """In-memory representation of a single goal."""

    __slots__ = (
        "name", "progress", "status", "created_at", "updated_at",
        "due_date", "priority", "milestones",
    )

    _VALID_STATUSES = frozenset({"active", "completed", "abandoned"})

    def __init__(
        self,
        name: str,
        progress: float = 0.0,
        status: str = "active",
        created_at: str = "",
        updated_at: str = "",
        due_date: Optional[str] = None,
        priority: Optional[str] = None,
        milestones: Optional[list] = None,
    ) -> None:
        self.name = name
        self.progress = progress
        self.status = status
        self.created_at = created_at or _utc_now()
        self.updated_at = updated_at or _utc_now()
        self.due_date = due_date or None      # ISO date string ("YYYY-MM-DD") or None
        self.priority = priority or None      # one of _VALID_PRIORITIES or None
        # Sub-steps (#124/#130): [{"title", "done"}]. When present, progress follows them.
        self.milestones: list[dict[str, Any]] = _clean_milestones(milestones or [])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "progress": self.progress,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "due_date": self.due_date,
            "priority": self.priority,
        }
        if self.milestones:
            out["milestones"] = [dict(m) for m in self.milestones]
        return out

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> "Goal":
        return cls(
            milestones=raw.get("milestones") if isinstance(raw.get("milestones"), list) else None,
            name=name,
            progress=float(raw.get("progress", 0.0)),
            status=str(raw.get("status", "active")),
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
            due_date=raw.get("due_date") or None,
            priority=raw.get("priority") or None,
        )

    # ------------------------------------------------------------------
    # Business logic
    # ------------------------------------------------------------------

    def update_progress(self, progress: float) -> None:
        """Clamp *progress* to [0, 1] and update the goal."""
        self.progress = max(_PROGRESS_MIN, min(_PROGRESS_MAX, progress))
        self.updated_at = _utc_now()

        if self.progress >= _PROGRESS_MAX:
            self.status = "completed"
            logger.info("Goal %r marked as completed.", self.name)

    def set_milestones(self, titles: list[Any]) -> None:
        """Replace the milestones (keeping "done" for titles that survive) and re-derive progress."""
        done = {m["title"].lower() for m in self.milestones if m["done"]}
        fresh = _clean_milestones(titles)
        for m in fresh:
            if isinstance(m, dict) and m["title"].lower() in done:
                m["done"] = True
        self.milestones = fresh
        self._sync_progress()

    def complete_milestone(self, ref: Any) -> Optional[dict[str, Any]]:
        """Tick a milestone by 1-based number or title fragment; None if nothing matches."""
        idx = _match_milestone(self.milestones, ref)
        if idx is None:
            return None
        self.milestones[idx]["done"] = True
        self._sync_progress()
        return self.milestones[idx]

    def _sync_progress(self) -> None:
        if not self.milestones:
            return
        self.update_progress(sum(m["done"] for m in self.milestones) / len(self.milestones))

    def set_status(self, status: str) -> None:
        if status not in self._VALID_STATUSES:
            raise ValueError(
                f"Invalid status {status!r}. Valid: {self._VALID_STATUSES}"
            )
        self.status = status
        self.updated_at = _utc_now()

    def days_until_due(self, today: date) -> Optional[int]:
        """
        Days remaining until ``due_date`` (negative if overdue), or
        ``None`` if no due date is set.
        """
        if not self.due_date:
            return None
        try:
            due = date.fromisoformat(self.due_date)
        except ValueError:
            return None
        return (due - today).days


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_history(dates: list[Any]) -> list[str]:
    """Valid, unique ISO dates, oldest first, capped to HISTORY_CAP_DAYS."""
    valid: set[str] = set()
    for item in dates:
        try:
            valid.add(date.fromisoformat(str(item)[:10]).isoformat())
        except ValueError:
            continue
    return sorted(valid)[-HISTORY_CAP_DAYS:]


def _clean_times(values: list[Any]) -> list[int]:
    """Valid minute-of-day ints, newest last, capped."""
    out: list[int] = []
    for v in values:
        try:
            m = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= m < 1440:
            out.append(m)
    return out[-extras.TIMES_CAP:]


def _clean_milestones(items: list[Any]) -> list[dict[str, Any]]:
    """[{"title": str, "done": bool}], blank/duplicate titles dropped."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for it in items:
        title = str(it.get("title", "") if isinstance(it, dict) else it).strip()[:120]
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        out.append({"title": title, "done": bool(it.get("done")) if isinstance(it, dict) else False})
    return out[:_MAX_MILESTONES]


def _match_milestone(milestones: list[dict[str, Any]], ref: Any) -> Optional[int]:
    """Index of the first not-yet-done milestone matching *ref* (1-based number or title fragment)."""
    text = str(ref or "").strip().lower()
    if not text or not milestones:
        return None
    if text.isdigit():
        n = int(text) - 1
        return n if 0 <= n < len(milestones) else None
    for i, m in enumerate(milestones):
        if not m["done"] and text in m["title"].lower():
            return i
    for i, m in enumerate(milestones):
        if text in m["title"].lower():
            return i
    return None


def _clean_grace(value: Any) -> Optional[int]:
    """A grace period in whole days within 0..MAX_GRACE_DAYS, or None (use the default)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return max(0, min(MAX_GRACE_DAYS, int(value)))
    except (TypeError, ValueError):
        return None


def _clean_pauses(pauses: list[Any]) -> list[list[str]]:
    """Valid [start, end] pairs, oldest first; zero-length or inverted pauses are dropped."""
    out: list[list[str]] = []
    for p in pauses:
        try:
            start = date.fromisoformat(str(p[0])[:10]).isoformat()
            end = date.fromisoformat(str(p[1])[:10]).isoformat() if len(p) > 1 and p[1] else ""
        except (ValueError, TypeError, IndexError):
            continue
        if end and end <= start:
            continue
        out.append([start, end])
    return sorted(out)[-_MAX_PAUSES:]


def _load_tz(name: str):
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name or "UTC")
        except Exception:
            logger.warning("Unknown timezone %r; using UTC.", name)
    return timezone.utc


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _validate_name(name: str, max_len: int, label: str) -> None:
    if not name or not name.strip():
        raise ValueError(f"{label} name must be a non-empty string.")
    if len(name) > max_len:
        raise ValueError(f"{label} name exceeds maximum length of {max_len}.")


def _validate_due_date(due_date: Optional[str]) -> Optional[str]:
    """Return a normalised ISO date string, or None. Raises ValueError if unparsable."""
    if due_date is None:
        return None
    try:
        return date.fromisoformat(str(due_date)).isoformat()
    except ValueError as exc:
        raise ValueError(
            f"due_date must be an ISO date (YYYY-MM-DD), got {due_date!r}."
        ) from exc


def _validate_priority(priority: Optional[str]) -> Optional[str]:
    """Return a normalised priority string, or None. Raises ValueError if invalid."""
    if priority is None:
        return None
    key = str(priority).strip().lower()
    if key not in _VALID_PRIORITIES:
        raise ValueError(
            f"priority must be one of {sorted(_VALID_PRIORITIES)}, got {priority!r}."
        )
    return key


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ArtemisTracker:
    """
    Persistent, thread-safe tracker for habits and goals.

    All state is stored in a single JSON file. Writes are atomic: data is
    first written to a sibling temp file, then renamed over the target path.
    This ensures the state file is never left in a partial state after a
    crash or power loss.

    Parameters
    ----------
    path:
        Path to the JSON state file.
    timezone_name:
        IANA zone used to read the clock for completion times and focus sessions (#122, #129).
    default_grace_days:
        Missed days a habit's streak survives unless that habit sets its own (#123).
        0 (the default) keeps the strict "miss a day and it resets" behaviour.
    """

    def __init__(self, path: str | Path = _DEFAULT_PATH, default_grace_days: int = 0,
                 timezone_name: str = "UTC") -> None:
        self.default_grace_days = _clean_grace(default_grace_days) or 0
        self.tz = _load_tz(timezone_name)
        raw_path = Path(path)
        if raw_path.is_absolute():
            self._path = raw_path
        else:
            # Anchor relative paths (including any custom one a caller
            # passes in) to the project root instead of cwd — see the
            # note on _DEFAULT_PATH above.
            self._path = (_PROJECT_ROOT / raw_path).resolve()
        self._lock = threading.Lock()
        self._ensure_state_file()
        logger.info("ArtemisTracker initialised (path=%s)", self._path)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_state_file(self) -> None:
        """Create the state file and its parent directories if absent."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write_raw(_EMPTY_STATE)
            logger.debug("Created new state file at %s", self._path)

    # ------------------------------------------------------------------
    # Persistence (private)
    # ------------------------------------------------------------------

    def _read_raw(self) -> dict[str, Any]:
        """
        Read and parse the state file.

        Raises
        ------
        StateCorruptedError
            If the file cannot be parsed or is missing required top-level keys.
        """
        try:
            text = self._path.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StateCorruptedError(
                f"State file {self._path} contains invalid JSON: {exc}"
            ) from exc
        except OSError as exc:
            raise TrackerError(
                f"Cannot read state file {self._path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise StateCorruptedError("State file root must be a JSON object.")

        data.setdefault("habits", {})
        data.setdefault("goals", {})
        return data

    def _write_raw(self, data: dict[str, Any]) -> None:
        """
        Atomically write *data* to the state file.

        Writes to a temporary file in the same directory, then renames it
        over the target.  The rename is atomic on POSIX systems.
        """
        parent = self._path.parent
        try:
            fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
            except Exception:
                os.unlink(tmp_path)
                raise
            os.replace(tmp_path, self._path)
        except OSError as exc:
            raise TrackerError(
                f"Cannot write state file {self._path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Habits – public API
    # ------------------------------------------------------------------

    def add_habit(self, name: str) -> None:
        """
        Register a new habit.  Silently no-ops if the habit already exists.

        Raises
        ------
        ValueError
            If *name* is empty or too long.
        """
        _validate_name(name, _MAX_HABIT_NAME_LEN, "Habit")

        with self._lock:
            data = self._read_raw()
            if name in data["habits"]:
                logger.debug("Habit %r already exists; skipping.", name)
                return
            habit = Habit(name=name)
            data["habits"][name] = habit.to_dict()
            self._write_raw(data)

        logger.info("Habit added: %r", name)

    def now(self) -> datetime:
        """The current moment in the tracker's timezone."""
        return datetime.now(self.tz)

    def complete_habit(self, name: str, today: Optional[date] = None,
                       at: Optional[datetime] = None) -> dict[str, Any]:
        """
        Mark *name* as completed for today (or an injected *today* for testing).

        Returns
        -------
        The milestone dict from ``Habit.complete()`` — ``already_done``,
        ``streak``, ``best_streak``, ``is_new_best``, ``total_completions``.

        Raises
        ------
        ValueError
            If *name* is empty.
        HabitNotFoundError
            If the habit does not exist.
        """
        _validate_name(name, _MAX_HABIT_NAME_LEN, "Habit")
        effective_today = today or _today_utc()
        # Time of day is only recorded for a real, live completion (or an explicit *at*), so a
        # back-dated or test completion never skews the "usually done by" estimate.
        if at is None and today is None:
            at = self.now()
        at_minute = None
        if at is not None:
            local = at.astimezone(self.tz) if at.tzinfo else at
            at_minute = local.hour * 60 + local.minute

        with self._lock:
            data = self._read_raw()
            raw = data["habits"].get(name)
            if raw is None:
                raise HabitNotFoundError(
                    f"Habit {name!r} not found. Add it first with add_habit()."
                )
            habit = Habit.from_dict(name, raw)
            result = habit.complete(effective_today, self.default_grace_days, at_minute)
            data["habits"][name] = habit.to_dict()
            self._write_raw(data)

        return result

    # -- grace periods (#123), pauses (#128), weekly review (#125) ------

    def _mutate_habit(self, name: str, fn) -> Any:
        _validate_name(name, _MAX_HABIT_NAME_LEN, "Habit")
        with self._lock:
            data = self._read_raw()
            raw = data["habits"].get(name)
            if raw is None:
                raise HabitNotFoundError(f"Habit {name!r} not found.")
            habit = Habit.from_dict(name, raw)
            result = fn(habit)
            data["habits"][name] = habit.to_dict()
            self._write_raw(data)
        return result

    def set_habit_grace(self, name: str, days: Optional[int]) -> Optional[int]:
        """Set how many missed days *name*'s streak survives (0 = strict; None = use the default).
        Returns the value stored. Raises ValueError outside 0..MAX_GRACE_DAYS."""
        if days is not None and (isinstance(days, bool) or not 0 <= int(days) <= MAX_GRACE_DAYS):
            raise ValueError(f"Grace period must be between 0 and {MAX_GRACE_DAYS} days.")
        value = None if days is None else int(days)

        def apply(h: Habit) -> Optional[int]:
            h.grace_days = value
            return value
        return self._mutate_habit(name, apply)

    def pause_habit(self, name: str, days: Optional[int] = None, today: Optional[date] = None) -> dict[str, Any]:
        """Pause *name* from today for *days* days, or until it is resumed. Streak survives the break."""
        if days is not None and not 1 <= int(days) <= 365:
            raise ValueError("A pause must last between 1 and 365 days.")
        t = today or _today_utc()

        def apply(h: Habit) -> dict[str, Any]:
            changed = h.pause(t, int(days) if days else None)
            end = next((p[1] for p in h.pauses if p[0] <= t.isoformat() and (not p[1] or t.isoformat() < p[1])), "")
            return {"already_paused": not changed, "until": end or None, "streak": h.streak}
        return self._mutate_habit(name, apply)

    def resume_habit(self, name: str, today: Optional[date] = None) -> dict[str, Any]:
        t = today or _today_utc()

        def apply(h: Habit) -> dict[str, Any]:
            return {"was_paused": h.resume(t), "streak": h.streak}
        return self._mutate_habit(name, apply)

    def weekly_review(self, today: Optional[date] = None, days: int = 7) -> dict[str, Any]:
        """
        Consistency over the last *days* days against the *days* before (#125). Paused days and
        days before a habit's history begins are left out of the denominator rather than counted
        as misses; a habit with no usable history is listed with ``pct`` None.
        """
        t = today or _today_utc()
        start, prev_end = t - timedelta(days=days - 1), t - timedelta(days=days)
        prev_start = prev_end - timedelta(days=days - 1)
        rows, tot_done, tot_poss = [], 0, 0
        for name, h in self.get_habits().items():
            done, poss = h.window_stats(start, t, t)
            pdone, pposs = h.window_stats(prev_start, prev_end, t)
            rows.append({
                "name": name, "done": done, "possible": poss,
                "pct": round(done / poss * 100) if poss else None,
                "prev_pct": round(pdone / pposs * 100) if pposs >= 3 else None,
                "streak": h.streak, "paused": h.paused_on(t),
            })
            tot_done, tot_poss = tot_done + done, tot_poss + poss
        scored = [r for r in rows if r["pct"] is not None]
        return {
            "start": start.isoformat(), "end": t.isoformat(), "habits": rows,
            "overall_pct": round(tot_done / tot_poss * 100) if tot_poss else None,
            "strongest": max(scored, key=lambda r: (r["pct"], r["done"]))["name"] if scored else None,
            "weakest": min(scored, key=lambda r: (r["pct"], r["done"]))["name"] if len(scored) > 1 else None,
        }

    def remove_habit(self, name: str) -> None:
        """
        Delete a habit permanently.

        Raises
        ------
        HabitNotFoundError
            If the habit does not exist.
        """
        _validate_name(name, _MAX_HABIT_NAME_LEN, "Habit")

        with self._lock:
            data = self._read_raw()
            if name not in data["habits"]:
                raise HabitNotFoundError(f"Habit {name!r} not found.")
            del data["habits"][name]
            self._write_raw(data)

        logger.info("Habit removed: %r", name)

    def get_habits(self) -> dict[str, Habit]:
        """Return all habits as ``{name: Habit}``."""
        with self._lock:
            data = self._read_raw()
        return {
            name: Habit.from_dict(name, raw)
            for name, raw in data["habits"].items()
        }

    def habit_history(self) -> dict[str, dict[str, Any]]:
        """Read-only per-habit completion history for correlations/heatmaps.

        ``{name: {"dates": [iso, ...], "since": iso-or-"", "streak": int,
        "last_done": iso-or-""}}``. Habits with no recorded history are
        still listed (empty ``dates``) so callers can say "not enough data
        yet" rather than silently omitting them.
        """
        with self._lock:
            raw = self._read_raw()
        out: dict[str, dict[str, Any]] = {}
        for name, blob in raw.get("habits", {}).items():
            h = Habit.from_dict(name, blob)
            out[name] = {
                "dates": list(h.history), "since": h.history_since,
                "streak": h.streak, "last_done": h.last_done,
                "paused": h.paused_on(_today_utc()),
            }
        return out

    def get_habit(self, name: str) -> Habit:
        """
        Return a single habit by name.

        Raises
        ------
        HabitNotFoundError
            If the habit does not exist.
        """
        _validate_name(name, _MAX_HABIT_NAME_LEN, "Habit")
        with self._lock:
            data = self._read_raw()
        raw = data["habits"].get(name)
        if raw is None:
            raise HabitNotFoundError(f"Habit {name!r} not found.")
        return Habit.from_dict(name, raw)

    # ------------------------------------------------------------------
    # Goals – public API
    # ------------------------------------------------------------------

    def add_goal(
        self,
        name: str,
        due_date: Optional[str] = None,
        priority: Optional[str] = None,
        milestones: Optional[list[str]] = None,
    ) -> None:
        """
        Register a new goal.  Silently no-ops if the goal already exists
        (its due_date/priority are left untouched — use
        ``set_goal_metadata()`` to change an existing goal).

        Raises
        ------
        ValueError
            If *name* is empty/too long, *due_date* isn't an ISO date, or
            *priority* isn't one of the accepted values.
        """
        _validate_name(name, _MAX_GOAL_NAME_LEN, "Goal")
        due_date = _validate_due_date(due_date)
        priority = _validate_priority(priority)

        with self._lock:
            data = self._read_raw()
            if name in data["goals"]:
                logger.debug("Goal %r already exists; skipping.", name)
                return
            goal = Goal(name=name, due_date=due_date, priority=priority, milestones=milestones)
            data["goals"][name] = goal.to_dict()
            self._write_raw(data)

        logger.info("Goal added: %r (due_date=%s, priority=%s)", name, due_date, priority)

    def update_goal(self, name: str, progress: float) -> None:
        """
        Set *progress* (0.0 – 1.0) for *name*, auto-completing at 1.0.

        Raises
        ------
        ValueError
            If *name* is empty or *progress* is not a finite number.
        GoalNotFoundError
            If the goal does not exist.
        """
        _validate_name(name, _MAX_GOAL_NAME_LEN, "Goal")
        if not isinstance(progress, (int, float)) or not (
            _PROGRESS_MIN <= float(progress) <= _PROGRESS_MAX
        ):
            raise ValueError(
                f"progress must be a number between {_PROGRESS_MIN} and {_PROGRESS_MAX}."
            )

        with self._lock:
            data = self._read_raw()
            raw = data["goals"].get(name)
            if raw is None:
                raise GoalNotFoundError(
                    f"Goal {name!r} not found. Add it first with add_goal()."
                )
            goal = Goal.from_dict(name, raw)
            goal.update_progress(float(progress))
            data["goals"][name] = goal.to_dict()
            self._write_raw(data)

        logger.debug("Goal %r updated: progress=%.2f", name, progress)

    def set_goal_status(self, name: str, status: str) -> None:
        """
        Manually set the status of a goal (``active``, ``completed``, ``abandoned``).

        Raises
        ------
        GoalNotFoundError
            If the goal does not exist.
        ValueError
            If *status* is not one of the accepted values.
        """
        _validate_name(name, _MAX_GOAL_NAME_LEN, "Goal")

        with self._lock:
            data = self._read_raw()
            raw = data["goals"].get(name)
            if raw is None:
                raise GoalNotFoundError(f"Goal {name!r} not found.")
            goal = Goal.from_dict(name, raw)
            goal.set_status(status)
            data["goals"][name] = goal.to_dict()
            self._write_raw(data)

        logger.info("Goal %r status set to %r.", name, status)

    def set_goal_metadata(
        self,
        name: str,
        due_date: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> None:
        """
        Update ``due_date`` and/or ``priority`` on an existing goal. Pass
        ``None`` (the default) for a field to leave it as-is — there's no
        way to clear a value once set, matching ``update_goal()``'s
        no-partial-clear semantics. A call with both fields ``None`` is a
        no-op.

        Raises
        ------
        GoalNotFoundError
            If the goal does not exist.
        ValueError
            If *due_date* isn't an ISO date or *priority* is invalid.
        """
        _validate_name(name, _MAX_GOAL_NAME_LEN, "Goal")
        due_date = _validate_due_date(due_date)
        priority = _validate_priority(priority)
        if due_date is None and priority is None:
            return

        with self._lock:
            data = self._read_raw()
            raw = data["goals"].get(name)
            if raw is None:
                raise GoalNotFoundError(f"Goal {name!r} not found.")
            goal = Goal.from_dict(name, raw)
            if due_date is not None:
                goal.due_date = due_date
            if priority is not None:
                goal.priority = priority
            goal.updated_at = _utc_now()
            data["goals"][name] = goal.to_dict()
            self._write_raw(data)

        logger.info("Goal %r metadata updated (due_date=%s, priority=%s).", name, due_date, priority)

    def get_at_risk_goals(
        self,
        days_threshold: int = 7,
        progress_threshold: float = 0.5,
        today: Optional[date] = None,
    ) -> dict[str, Goal]:
        """
        Return active goals that are due within *days_threshold* days
        (including already overdue ones) and below *progress_threshold*
        progress. Goals without a ``due_date`` are never at risk — there's
        nothing to measure them against.
        """
        effective_today = today or _today_utc()
        at_risk: dict[str, Goal] = {}
        for name, goal in self.get_goals().items():
            if goal.status != "active":
                continue
            days_left = goal.days_until_due(effective_today)
            if days_left is None:
                continue
            if days_left <= days_threshold and goal.progress < progress_threshold:
                at_risk[name] = goal
        return at_risk

    # -- milestones (#124, #130) ----------------------------------------

    def set_goal_milestones(self, name: str, titles: list[str]) -> list[dict[str, Any]]:
        """Replace *name*'s milestones; progress is re-derived from them."""
        _validate_name(name, _MAX_GOAL_NAME_LEN, "Goal")
        with self._lock:
            data = self._read_raw()
            raw = data["goals"].get(name)
            if raw is None:
                raise GoalNotFoundError(f"Goal {name!r} not found.")
            goal = Goal.from_dict(name, raw)
            goal.set_milestones(titles)
            data["goals"][name] = goal.to_dict()
            self._write_raw(data)
        return goal.milestones

    def complete_goal_milestone(self, name: str, ref: Any) -> Optional[dict[str, Any]]:
        """Tick a milestone (1-based number or title fragment). None when *ref* matches nothing."""
        _validate_name(name, _MAX_GOAL_NAME_LEN, "Goal")
        with self._lock:
            data = self._read_raw()
            raw = data["goals"].get(name)
            if raw is None:
                raise GoalNotFoundError(f"Goal {name!r} not found.")
            goal = Goal.from_dict(name, raw)
            hit = goal.complete_milestone(ref)
            if hit is None:
                return None
            data["goals"][name] = goal.to_dict()
            self._write_raw(data)
        done = sum(m["done"] for m in goal.milestones)
        return {"title": hit["title"], "done": done, "total": len(goal.milestones),
                "progress": goal.progress, "goal_completed": goal.status == "completed"}

    def remove_goal(self, name: str) -> None:
        """
        Delete a goal permanently.

        Raises
        ------
        GoalNotFoundError
            If the goal does not exist.
        """
        _validate_name(name, _MAX_GOAL_NAME_LEN, "Goal")

        with self._lock:
            data = self._read_raw()
            if name not in data["goals"]:
                raise GoalNotFoundError(f"Goal {name!r} not found.")
            del data["goals"][name]
            self._write_raw(data)

        logger.info("Goal removed: %r", name)

    def get_goals(self) -> dict[str, Goal]:
        """Return all goals as ``{name: Goal}``."""
        with self._lock:
            data = self._read_raw()
        return {
            name: Goal.from_dict(name, raw)
            for name, raw in data["goals"].items()
        }

    def get_goal(self, name: str) -> Goal:
        """
        Return a single goal by name.

        Raises
        ------
        GoalNotFoundError
            If the goal does not exist.
        """
        _validate_name(name, _MAX_GOAL_NAME_LEN, "Goal")
        with self._lock:
            data = self._read_raw()
        raw = data["goals"].get(name)
        if raw is None:
            raise GoalNotFoundError(f"Goal {name!r} not found.")
        return Goal.from_dict(name, raw)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    # -- generic state access (focus, badges, nudges) -------------------

    def _update_state(self, fn: Callable[[dict[str, Any]], Any]) -> Any:
        """Run *fn(data)* under the lock and persist the result."""
        with self._lock:
            data = self._read_raw()
            result = fn(data)
            self._write_raw(data)
        return result

    def _peek_state(self) -> dict[str, Any]:
        with self._lock:
            return self._read_raw()

    # -- focus sessions / Pomodoro (#122) --------------------------------

    def start_focus(self, minutes: int = extras.DEFAULT_FOCUS_MINUTES, task: str = "",
                    now: Optional[datetime] = None) -> dict[str, Any]:
        """Start a focus session. Raises ValueError on a bad length or if one is already running."""
        if isinstance(minutes, bool) or not 1 <= int(minutes) <= extras.MAX_FOCUS_MINUTES:
            raise ValueError(f"A focus session must be 1 to {extras.MAX_FOCUS_MINUTES} minutes.")
        moment = now or self.now()

        def apply(data: dict[str, Any]) -> dict[str, Any]:
            focus = data.setdefault("focus", {})
            if focus.get("active"):
                raise ValueError("A focus session is already running.")
            return extras.focus_start(focus, moment, int(minutes), task)
        return self._update_state(apply)

    def stop_focus(self, now: Optional[datetime] = None) -> Optional[dict[str, Any]]:
        """End the running session and log it; None when nothing was running."""
        moment = now or self.now()
        return self._update_state(lambda d: extras.focus_finish(d.setdefault("focus", {}), moment))

    def active_focus(self, now: Optional[datetime] = None) -> Optional[dict[str, Any]]:
        """The running session with ``remaining`` minutes added, or None."""
        active = self._peek_state().get("focus", {}).get("active")
        if not active:
            return None
        return {**active, "remaining": extras.focus_remaining(active, now or self.now())}

    def completed_focus_today(self, now: Optional[datetime] = None) -> int:
        """Finished focus sessions that started today (local), for the long-break rule."""
        today = (now or self.now()).astimezone(self.tz).date().isoformat()
        sessions = self._peek_state().get("focus", {}).get("sessions", [])
        return sum(1 for s in sessions if s.get("completed") and str(s.get("start", ""))[:10] == today)

    def focus_stats(self, now: Optional[datetime] = None, days: int = 7) -> dict[str, Any]:
        return extras.focus_stats(self._peek_state().get("focus", {}), now or self.now(), days)

    # -- badges (#127) ---------------------------------------------------

    def earned_badges(self) -> dict[str, str]:
        """{badge id: ISO date earned}."""
        return dict(self._peek_state().get("badges", {}))

    def award_badges(self, today: Optional[date] = None) -> list[str]:
        """Record any newly earned badges and return their ids (empty when nothing is new)."""
        stamp = (today or _today_utc()).isoformat()

        def apply(data: dict[str, Any]) -> list[str]:
            habits = {n: Habit.from_dict(n, r) for n, r in data["habits"].items()}
            goals = {n: Goal.from_dict(n, r) for n, r in data["goals"].items()}
            earned = extras.evaluate_badges(habits, goals, data.get("focus", {}))
            have = data.setdefault("badges", {})
            new = [b for b in extras.BADGES if b in earned and b not in have]
            for b in new:
                have[b] = stamp
            return new
        return self._update_state(apply)

    # -- smart nudges (#129) ---------------------------------------------

    def habits_due_for_nudge(self, now: Optional[datetime] = None, lateness_minutes: int = 60,
                             day_end_minute: int = 22 * 60) -> list[dict[str, Any]]:
        """
        Habits usually logged by a certain time that haven't been today, and haven't been
        nudged today. A habit is due once *lateness_minutes* have passed its usual time;
        nothing is due after *day_end_minute*. Paused habits are skipped.
        """
        moment = (now or self.now()).astimezone(self.tz)
        today_local = moment.date()
        minute_now = moment.hour * 60 + moment.minute
        if minute_now >= day_end_minute:
            return []
        data = self._peek_state()
        nudged = data.get("nudges", {})
        due = []
        for name, raw in data["habits"].items():
            h = Habit.from_dict(name, raw)
            typical = extras.typical_minute(h.times)
            if typical is None or h.paused_on(today_local):
                continue
            if h.last_done in (today_local.isoformat(), _today_utc().isoformat()):
                continue
            if nudged.get(name) == today_local.isoformat():
                continue
            if minute_now >= typical + lateness_minutes:
                due.append({"name": name, "typical_minute": typical, "streak": h.streak})
        return due

    def mark_nudged(self, name: str, today: Optional[date] = None) -> None:
        day = (today or self.now().date()).isoformat()
        self._update_state(lambda d: d.setdefault("nudges", {}).__setitem__(name, day))

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary suitable for logging or a status API."""
        with self._lock:
            data = self._read_raw()

        habits = data["habits"]
        goals = data["goals"]

        active_goals = sum(
            1 for g in goals.values() if g.get("status") == "active"
        )
        completed_goals = sum(
            1 for g in goals.values() if g.get("status") == "completed"
        )
        top_streak = max(
            (h.get("streak", 0) for h in habits.values()), default=0
        )

        return {
            "total_habits": len(habits),
            "top_streak": top_streak,
            "total_goals": len(goals),
            "active_goals": active_goals,
            "completed_goals": completed_goals,
        }