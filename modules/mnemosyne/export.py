"""
modules/mnemosyne/export.py

One export builder shared by the engine (export_memory intent), the web API
(/api/mnemosyne/export) and scripts/export_memory.py (#44). Pages through the
fact table, so there is no 1,000-row ceiling on a backup.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def _all_facts(db) -> list[dict]:
    out: list[dict] = []
    offset = 0
    while True:
        page = db.get_all_facts(limit=1000, offset=offset)
        out.extend(page)
        if len(page) < 1000:
            return out
        offset += 1000


def collect(db) -> dict:
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "facts": _all_facts(db),
        "summaries": db.get_recent_summaries(n=1_000_000),
        "goals": db.get_goals(status="active") + db.get_goals(status="completed"),
    }


def render(payload: dict, fmt: str = "json") -> str:
    if fmt in ("markdown", "md"):
        facts, summaries, goals = payload["facts"], payload["summaries"], payload["goals"]
        lines = ["# Hestia memory export", f"_Exported {payload['exported_at']}_", "",
                 f"## Facts ({len(facts)})"]
        lines += [f"- **{f['key']}**: {f['value']}" for f in facts]
        lines += ["", f"## Summaries ({len(summaries)})"]
        lines += [f"- _{s.get('period_start', '?')}_ ({s.get('topic', 'General')}): {s.get('content', '')}"
                  for s in summaries]
        lines += ["", f"## Goals ({len(goals)})"]
        lines += [f"- [{g.get('status', '?')}] {g.get('text', '')}" for g in goals]
        return "\n".join(lines)
    return json.dumps(payload, indent=2, default=str)


def build_export(db, fmt: str = "json") -> str:
    return render(collect(db), fmt)


def write_export(db, directory: str, fmt: str = "json") -> dict:
    fmt = "markdown" if fmt in ("md", "markdown") else "json"
    payload = collect(db)
    os.makedirs(directory, exist_ok=True)
    ext = "md" if fmt == "markdown" else "json"
    path = os.path.join(directory, f"hestia_memory_{datetime.now():%Y%m%d_%H%M%S}.{ext}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(payload, fmt))
    return {"path": path, "format": fmt, "facts": len(payload["facts"]),
            "summaries": len(payload["summaries"]), "goals": len(payload["goals"])}
