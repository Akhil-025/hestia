#!/usr/bin/env python3
"""One-off: remove junk facts ('what: my name', 'hi my name: akhil').
Run scripts/export_memory.py FIRST. Dry-run by default; pass --apply to delete.
"""
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    from modules.mnemosyne.config import get_config
    from modules.mnemosyne.db import MnemosyneDB
    from modules.mnemosyne.keys import is_junk
    path = a.db or get_config().db_path
    if not os.path.exists(path):
        print(f"No database at {path}", file=sys.stderr); return 1
    db = MnemosyneDB(path)
    junk = [f["key"] for f in db.get_all_facts(limit=1000) if is_junk(f["key"], f["value"])]
    for k in junk:
        print(("DELETE " if a.apply else "would delete ") + repr(k))
    if a.apply and junk:
        print(f"deleted {db.delete_facts(junk)} row(s)")
    elif not junk:
        print("no junk facts found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
