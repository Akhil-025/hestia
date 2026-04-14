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
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PATH = Path("data/artemis_state.json")
_EMPTY_STATE: dict[str, Any] = {"habits": {}, "goals": {}}

_MAX_HABIT_NAME_LEN = 256
_MAX_GOAL_NAME_LEN = 256
_PROGRESS_MIN = 0.0
_PROGRESS_MAX = 1.0


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

    __slots__ = ("name", "streak", "last_done")

    def __init__(self, name: str, streak: int = 0, last_done: str = "") -> None:
        self.name = name
        self.streak = streak
        self.last_done = last_done  # ISO date string or ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"streak": self.streak, "last_done": self.last_done}

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> "Habit":
        return cls(
            name=name,
            streak=int(raw.get("streak", 0)),
            last_done=str(raw.get("last_done", "")),
        )

    # ------------------------------------------------------------------
    # Business logic
    # ------------------------------------------------------------------

    def complete(self, today: date) -> None:
        """
        Mark the habit as completed for *today*, updating the streak.

        Rules
        -----
        - Same day  → no-op (idempotent).
        - Yesterday → extend streak.
        - Older / never done → reset streak to 1.
        """
        today_iso = today.isoformat()
        yesterday_iso = (today - timedelta(days=1)).isoformat()

        if self.last_done == today_iso:
            logger.debug("Habit %r already completed today; no-op.", self.name)
            return

        if self.last_done == yesterday_iso:
            self.streak += 1
        else:
            self.streak = 1

        self.last_done = today_iso
        logger.debug(
            "Habit %r completed. streak=%d last_done=%s",
            self.name,
            self.streak,
            self.last_done,
        )


class Goal:
    """In-memory representation of a single goal."""

    __slots__ = ("name", "progress", "status", "created_at", "updated_at")

    _VALID_STATUSES = frozenset({"active", "completed", "abandoned"})

    def __init__(
        self,
        name: str,
        progress: float = 0.0,
        status: str = "active",
        created_at: str = "",
        updated_at: str = "",
    ) -> None:
        self.name = name
        self.progress = progress
        self.status = status
        self.created_at = created_at or _utc_now()
        self.updated_at = updated_at or _utc_now()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "progress": self.progress,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> "Goal":
        return cls(
            name=name,
            progress=float(raw.get("progress", 0.0)),
            status=str(raw.get("status", "active")),
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
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

    def set_status(self, status: str) -> None:
        if status not in self._VALID_STATUSES:
            raise ValueError(
                f"Invalid status {status!r}. Valid: {self._VALID_STATUSES}"
            )
        self.status = status
        self.updated_at = _utc_now()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _validate_name(name: str, max_len: int, label: str) -> None:
    if not name or not name.strip():
        raise ValueError(f"{label} name must be a non-empty string.")
    if len(name) > max_len:
        raise ValueError(f"{label} name exceeds maximum length of {max_len}.")


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
    """

    def __init__(self, path: str | Path = _DEFAULT_PATH) -> None:
        self._path = Path(path).resolve()
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

    def complete_habit(self, name: str, today: Optional[date] = None) -> None:
        """
        Mark *name* as completed for today (or an injected *today* for testing).

        Raises
        ------
        ValueError
            If *name* is empty.
        HabitNotFoundError
            If the habit does not exist.
        """
        _validate_name(name, _MAX_HABIT_NAME_LEN, "Habit")
        effective_today = today or _today_utc()

        with self._lock:
            data = self._read_raw()
            raw = data["habits"].get(name)
            if raw is None:
                raise HabitNotFoundError(
                    f"Habit {name!r} not found. Add it first with add_habit()."
                )
            habit = Habit.from_dict(name, raw)
            habit.complete(effective_today)
            data["habits"][name] = habit.to_dict()
            self._write_raw(data)

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

    def add_goal(self, name: str) -> None:
        """
        Register a new goal.  Silently no-ops if the goal already exists.

        Raises
        ------
        ValueError
            If *name* is empty or too long.
        """
        _validate_name(name, _MAX_GOAL_NAME_LEN, "Goal")

        with self._lock:
            data = self._read_raw()
            if name in data["goals"]:
                logger.debug("Goal %r already exists; skipping.", name)
                return
            goal = Goal(name=name)
            data["goals"][name] = goal.to_dict()
            self._write_raw(data)

        logger.info("Goal added: %r", name)

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