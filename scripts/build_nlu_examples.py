#!/usr/bin/env python3
"""
Convert the legacy monolithic config/nlu_prompt.txt into data (Phase 1.2):

    config/nlu_examples.yaml   examples (utterance -> intent, entities) + per-section notes

Section = a "# ---- NAME ----" header; the "#" comment lines under it are the
disambiguation notes for that group. Re-runnable and idempotent.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "config" / "nlu_prompt.txt"
DST = ROOT / "config" / "nlu_examples.yaml"


def parse(text: str):
    sections, examples = {}, []
    section, notes, pending = "general", [], None
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        m = re.match(r"^#\s*-{3,}\s*(.+?)\s*-{3,}\s*$", line)
        if m:
            if notes:
                sections.setdefault(section, []).extend(notes)
            section, notes, pending = m.group(1).strip(), [], None
            continue
        if line.startswith("#"):
            notes.append(line.lstrip("# ").rstrip())
            continue
        if line.startswith("User:"):
            pending = line[5:].strip()
            continue
        if pending and line.startswith("{"):
            try:
                obj = json.loads(line)
                examples.append({
                    "utterance": pending,
                    "intent": obj.get("intent"),
                    "entities": obj.get("entities") or {},
                    "section": section,
                })
            except json.JSONDecodeError:
                pass
            pending = None
    if notes:
        sections.setdefault(section, []).extend(notes)
    return sections, examples


def main() -> int:
    import yaml
    sections, examples = parse(SRC.read_text(encoding="utf-8"))
    notes = {k: " ".join(x for x in v if x) for k, v in sections.items()}
    DST.write_text(yaml.safe_dump(
        {"notes": notes, "examples": examples}, allow_unicode=True,
        sort_keys=False, width=100), encoding="utf-8")
    print(f"wrote {DST}: {len(examples)} examples, {len(notes)} note groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
