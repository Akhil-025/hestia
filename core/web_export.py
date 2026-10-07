"""
core/web_export.py - per-module data export for the web UI (backlog #190).

``GET /api/export/<module>/<dataset>?format=csv|json`` downloads one table of
one module. The catalog below says which datasets exist; each getter is
read-only and returns a list of flat dicts.

CSV safety: a cell that starts with ``=``, ``+``, ``-``, ``@``, tab or CR is
treated by Excel/Sheets as a formula, and some of these values come from the
outside world (a receipt merchant name, an email-derived reminder). Such text
cells are prefixed with an apostrophe. Real numbers are left alone.
"""
from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# "all time" for the day-window getters in Apollo's database.
_ALL_DAYS = 36500
_ROW_CAP = 200_000
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _facts(ui) -> list[dict]:
    out: list[dict] = []
    offset = 0
    while len(out) < _ROW_CAP:
        page = ui.memory.db.get_all_facts(limit=1000, offset=offset)
        out.extend(page)
        if len(page) < 1000:
            break
        offset += 1000
    return out


def _flatten_goal(name: str, g: Any) -> dict:
    d = g.to_dict()
    d["name"] = name
    d["milestones"] = json.dumps(d.get("milestones") or [], ensure_ascii=False)
    return d


def _habit_rows(ui) -> list[dict]:
    rows = []
    for name, h in ui.artemis.tracker.get_habits().items():
        d = h.to_dict()
        d["name"] = name
        for k in ("history", "pauses", "times"):
            if k in d:
                d[k] = json.dumps(d[k], ensure_ascii=False)
        rows.append(d)
    return rows


def _habit_completions(ui) -> list[dict]:
    rows = []
    for name, info in sorted(ui.artemis.tracker.habit_history().items()):
        for day in info["dates"]:
            rows.append({"habit": name, "date": day})
    return rows


def _iris_files(ui) -> list[dict]:
    rows = ui.iris.db.get_all_files(limit=_ROW_CAP)
    return [{k: v for k, v in r.items() if k not in ("embedding",)} for r in rows]


def _apollo(getter: str) -> Callable[[Any], list[dict]]:
    return lambda ui: getattr(ui.apollo.db, getter)(_ALL_DAYS)


# module -> (available attr on the UI, label, {dataset: (label, getter)})
CATALOG: dict[str, dict[str, Any]] = {
    "mnemosyne": {
        "label": "Memory", "needs": "memory",
        "datasets": {
            "facts": ("Facts", _facts),
            "notes": ("Notes", lambda ui: ui.memory.db.get_by_intent("take_note", _ROW_CAP)),
            "history": ("Interaction history",
                        lambda ui: ui.memory.db.get_recent_interactions(_ROW_CAP)),
            "summaries": ("Summaries",
                          lambda ui: ui.memory.db.get_recent_summaries(n=_ROW_CAP)),
            "goals": ("Goals", lambda ui: ui.memory.db.get_goals(status="active")
                      + ui.memory.db.get_goals(status="completed")),
        },
    },
    "chronos": {
        "label": "Reminders", "needs": "memory",
        "datasets": {
            "reminders": ("Reminders (all statuses)",
                          lambda ui: ui.memory.db.list_reminders(status=None)),
        },
    },
    "artemis": {
        "label": "Habits & goals", "needs": "artemis",
        "datasets": {
            "habits": ("Habits", _habit_rows),
            "completions": ("Habit completions", _habit_completions),
            "goals": ("Goals", lambda ui: [_flatten_goal(n, g) for n, g
                                           in ui.artemis.tracker.get_goals().items()]),
        },
    },
    "apollo": {
        "label": "Health", "needs": "apollo",
        "datasets": {
            "sleep": ("Sleep", _apollo("get_sleep")),
            "weight": ("Weight", _apollo("get_weight")),
            "water": ("Water", _apollo("get_water")),
            "mood": ("Mood", _apollo("get_mood")),
            "workouts": ("Workouts", _apollo("get_workouts")),
            "steps": ("Steps", _apollo("get_steps")),
            "meals": ("Meals", _apollo("get_meals")),
        },
    },
    "pluto": {
        "label": "Finance", "needs": "pluto",
        "datasets": {
            "expenses": ("Expenses",
                         lambda ui: ui.pluto.pf_manager.db.get_expenses(_ROW_CAP)),
            "investments": ("Investments",
                            lambda ui: ui.pluto.pf_manager.db.get_investments()),
        },
    },
    "athena": {
        "label": "Documents", "needs": "athena",
        "datasets": {
            "documents": ("Indexed documents",
                          lambda ui: ui.athena.rag.list_document_sources()),
        },
    },
    "iris": {
        "label": "Photos", "needs": "iris",
        "datasets": {"files": ("Media library (metadata)", _iris_files)},
    },
}


def available(ui) -> list[dict]:
    """The catalog restricted to modules this install actually has."""
    out = []
    for mod, spec in CATALOG.items():
        if getattr(ui, spec["needs"], None) is None:
            continue
        out.append({
            "module": mod, "label": spec["label"],
            "datasets": [{"id": k, "label": v[0]} for k, v in spec["datasets"].items()],
        })
    return out


def get_rows(ui, module: str, dataset: str) -> Optional[list[dict]]:
    """Rows for one dataset, or None if the module/dataset is unknown or off."""
    spec = CATALOG.get(module)
    if spec is None or getattr(ui, spec["needs"], None) is None:
        return None
    entry = spec["datasets"].get(dataset)
    if entry is None:
        return None
    rows = entry[1](ui) or []
    return [dict(r) for r in rows[:_ROW_CAP]]


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    text = str(value)
    if text.startswith(_FORMULA_PREFIXES):
        return "'" + text
    return text


def to_csv(rows: list[dict]) -> str:
    """UTF-8 CSV with a BOM (so Excel reads accents) and a stable column order."""
    columns: list[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow([_cell(c) for c in columns])
    for r in rows:
        w.writerow([_cell(r.get(c)) for c in columns])
    return "\ufeff" + buf.getvalue()


def to_json(module: str, dataset: str, rows: list[dict]) -> str:
    return json.dumps(
        {"module": module, "dataset": dataset, "count": len(rows),
         "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "rows": rows},
        indent=2, ensure_ascii=False, default=str,
    )
