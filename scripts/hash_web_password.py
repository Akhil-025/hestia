"""
scripts/hash_web_password.py - make a value for ``webui.password_hash`` (#186).

    python scripts/hash_web_password.py

Asks for a password (nothing is echoed), then prints one line to paste into
config/laptop_config.yaml under ``webui:``. The config file then holds only a
salted hash, never the password.

Whoever gets the hash can try to guess the password offline, so pick a
passphrase that is long rather than clever.
"""
from __future__ import annotations

import getpass
import sys

MIN_LENGTH = 8


def main() -> int:
    try:
        from werkzeug.security import generate_password_hash
    except ImportError:
        print("Flask/Werkzeug is not installed: pip install flask", file=sys.stderr)
        return 1

    first = getpass.getpass("New web UI password: ")
    if len(first) < MIN_LENGTH:
        print(f"Use at least {MIN_LENGTH} characters.", file=sys.stderr)
        return 1
    if getpass.getpass("Again: ") != first:
        print("Those did not match.", file=sys.stderr)
        return 1

    hashed = generate_password_hash(first)
    print("\nAdd this under `webui:` in config/laptop_config.yaml, then restart Hestia:\n")
    # Single-quoted YAML: the hash contains '$' but never a single quote.
    print(f"  password_hash: '{hashed}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
