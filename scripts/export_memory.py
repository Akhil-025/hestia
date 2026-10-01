#!/usr/bin/env python3
"""
scripts/export_memory.py - back up Hestia's memory without starting Hestia (#44).

    python scripts/export_memory.py                     # JSON file next to the database
    python scripts/export_memory.py --format markdown   # readable Markdown instead
    python scripts/export_memory.py --stdout            # print instead of writing a file
    python scripts/export_memory.py --db path/to/mnemosyne.db --out backups/

Opens the SQLite database directly (read-only use), so it works while Hestia is
stopped and needs neither ChromaDB nor Ollama.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export Hestia's facts, summaries and goals.")
    ap.add_argument("--db", help="path to mnemosyne.db (default: the configured one)")
    ap.add_argument("--out", help="directory for the export file (default: <db dir>/exports)")
    ap.add_argument("--format", choices=("json", "markdown"), default="json")
    ap.add_argument("--stdout", action="store_true", help="print the export instead of writing a file")
    args = ap.parse_args(argv)

    from modules.mnemosyne.config import get_config
    from modules.mnemosyne.db import MnemosyneDB
    from modules.mnemosyne.export import build_export, write_export

    db_path = args.db or get_config().db_path
    if not os.path.exists(db_path):
        print(f"No database at {db_path}", file=sys.stderr)
        return 1
    db = MnemosyneDB(db_path)
    if args.stdout:
        print(build_export(db, args.format))
        return 0
    out = args.out or str(Path(db_path).parent / "exports")
    info = write_export(db, out, args.format)
    print(f"Exported {info['facts']} facts, {info['summaries']} summaries, "
          f"{info['goals']} goals to {info['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
