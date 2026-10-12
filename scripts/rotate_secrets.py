"""Secrets rotation checklist with expiry warnings (backlog #199).

    python scripts/rotate_secrets.py                  # status table (same as `status`)
    python scripts/rotate_secrets.py status --strict  # exit 1 if anything is due/overdue
    python scripts/rotate_secrets.py checklist telegram   # how to rotate one secret
    python scripts/rotate_secrets.py record telegram      # "I rotated it today"
    python scripts/rotate_secrets.py record telegram --date 2026-09-01
    python scripts/rotate_secrets.py make-data-key        # new key for #198 (printed once)
    python scripts/rotate_secrets.py reencrypt            # encrypt old rows / move to the newest key

It only ever shows whether a secret is *set* and how old it is. Values are never
printed. Rotation dates live in data/secrets_rotation.json (data/ is gitignored).
Nothing here changes a secret for you; each checklist says where to rotate it.

Ages count from the date you ran `record`. A secret that is set but has no
recorded date is reported as "unknown age" and counts as due, because "never
rotated since I set it up" is exactly what this script exists to surface.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "data" / "secrets_rotation.json"
GOOGLE_TOKEN_PATH = ROOT / "data" / "google_token.json"
WARN_DAYS = 14  # "expiring soon" window


@dataclass(frozen=True)
class SecretSpec:
    name: str
    env: tuple            # environment variables that hold it (any one set = "set")
    max_age_days: int     # rotate at least this often
    steps: tuple          # the checklist
    note: str = ""


SECRETS: List[SecretSpec] = [
    SecretSpec("telegram", ("TELEGRAM_BOT_TOKEN",), 180, (
        "Open @BotFather, run /revoke, pick the bot: it issues a new token and the old one stops working.",
        "Put the new token in .env as TELEGRAM_BOT_TOKEN.",
        "Restart Hestia and send the bot a message to confirm it answers.",
    )),
    SecretSpec("web_ui", ("HESTIA_WEB_PASSWORD", "HESTIA_WEB_API_KEY", "HESTIA_WEB_SECRET"), 180, (
        "Make a new password hash: python scripts/hash_web_password.py (or set HESTIA_WEB_PASSWORD).",
        "Change HESTIA_WEB_API_KEY if scripts use it, and update those scripts.",
        "Change HESTIA_WEB_SECRET to any new long random string: every open session is signed out, which is the point.",
        "Restart Hestia and sign in again.",
    )),
    SecretSpec("sync_api", ("HESTIA_SYNC_API_KEY",), 180, (
        "Generate a new random key and set HESTIA_SYNC_API_KEY on the server.",
        "Update the same key on every client that calls /sync/*.",
        "Restart Hestia and run one pull to confirm.",
    )),
    SecretSpec("data_key", ("HESTIA_DATA_KEY", "HESTIA_DATA_KEY_FILE"), 0, (
        "Rotating the data key re-encrypts stored fields. Add the new key as the FIRST entry of "
        "HESTIA_DATA_KEY (comma-separated: new,old), restart, and run python scripts/rotate_secrets.py "
        "reencrypt (see core/crypto_store.py).",
        "Once it reports 0 fields left on the old key, remove the old key.",
    ), note="no fixed age: rotate it only if it may have leaked"),
    SecretSpec("todoist", ("TODOIST_API_TOKEN",), 365, (
        "Todoist -> Settings -> Integrations -> Developer: regenerate the API token.",
        "Update TODOIST_API_TOKEN (or todoist.api_token) and restart.",
    )),
    SecretSpec("anthropic", ("ANTHROPIC_API_KEY",), 180, (
        "Create a new key in the Anthropic console, then delete the old one.",
        "Update ANTHROPIC_API_KEY and restart.",
    )),
    SecretSpec("gemini", ("GEMINI_API_KEY",), 180, (
        "Create a new key in Google AI Studio, then delete the old one.",
        "Update GEMINI_API_KEY and restart.",
    )),
    SecretSpec("omdb", ("OMDB_API_KEY",), 365, (
        "OMDb keys cannot be rotated in place: request a new one at omdbapi.com and update OMDB_API_KEY.",
    )),
    SecretSpec("spotify", ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"), 365, (
        "Spotify developer dashboard -> your app -> Settings -> reset the client secret.",
        "Update SPOTIFY_CLIENT_SECRET and restart.",
    )),
    SecretSpec("ieee", ("IEEE_API_KEY",), 365, (
        "Request a replacement key from the IEEE developer portal and update IEEE_API_KEY.",
        "Check it with python scripts/check_ieee.py.",
    )),
    SecretSpec("fred", ("FRED_API_KEY",), 365, (
        "Request a new key at fred.stlouisfed.org and update FRED_API_KEY.",
    )),
    SecretSpec("kite", ("KITE_API_KEY", "KITE_ACCESS_TOKEN"), 1, (
        "The Kite access token expires daily: log in to Kite Connect and set a fresh KITE_ACCESS_TOKEN "
        "for the session. Do not store it long-term.",
    ), note="expires daily by design"),
]
BY_NAME = {s.name: s for s in SECRETS}


# ── state ────────────────────────────────────────────────────────────────────

def load_state(path: Path = STATE_PATH) -> Dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: str(v) for k, v in data.get("rotated", {}).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def save_state(state: Dict[str, str], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"rotated": state}, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_dotenv_into(env: Dict[str, str], path: Path) -> Dict[str, str]:
    """Merge simple KEY=VALUE lines of a .env file under the real environment."""
    merged: Dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            merged[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    merged.update(env)
    return merged


# ── evaluation (pure, tested) ────────────────────────────────────────────────

@dataclass
class Row:
    name: str
    status: str          # not set | ok | due soon | overdue | unknown age | info
    detail: str
    is_set: bool
    needs_attention: bool


def _parse_date(text: str) -> Optional[date]:
    try:
        return date.fromisoformat(text[:10])
    except (TypeError, ValueError):
        return None


def evaluate(env: Dict[str, str], state: Dict[str, str], today: date,
             specs: List[SecretSpec] = SECRETS) -> List[Row]:
    rows: List[Row] = []
    for spec in specs:
        is_set = any((env.get(v) or "").strip() for v in spec.env)
        if not is_set:
            rows.append(Row(spec.name, "not set", "not configured (nothing to rotate)", False, False))
            continue
        rotated = _parse_date(state.get(spec.name, ""))
        if rotated is None:
            rows.append(Row(spec.name, "unknown age",
                            f"set, but no rotation date recorded: run `record {spec.name}` "
                            f"once it is fresh" + (f" ({spec.note})" if spec.note else ""),
                            True, spec.max_age_days > 0))
            continue
        age = (today - rotated).days
        if spec.max_age_days <= 0:
            rows.append(Row(spec.name, "ok", f"rotated {rotated} ({age} days ago); {spec.note}", True, False))
            continue
        left = spec.max_age_days - age
        if left < 0:
            rows.append(Row(spec.name, "overdue", f"rotated {rotated}; {-left} days past its "
                                                  f"{spec.max_age_days}-day limit", True, True))
        elif 0 < left <= min(WARN_DAYS, spec.max_age_days // 4):
            rows.append(Row(spec.name, "due soon", f"rotated {rotated}; {left} days left", True, True))
        else:
            rows.append(Row(spec.name, "ok", f"rotated {rotated}; {left} days left", True, False))
    return rows


def google_token_row(path: Path, now: datetime) -> Optional[Row]:
    """Google OAuth token: a refresh token that is missing or an access-token expiry far in the past."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Row("google_oauth", "overdue", "token file is unreadable: re-run Google authorisation", True, True)
    if not isinstance(data, dict) or not data.get("refresh_token"):
        return Row("google_oauth", "overdue",
                   "no refresh token in the token file: delete it and re-authorise", True, True)
    expiry = data.get("expiry")
    exp = None
    if isinstance(expiry, str):
        try:
            exp = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
        except ValueError:
            exp = None
    if exp is None:
        return Row("google_oauth", "info", "refresh token present (access-token expiry not recorded)", True, False)
    # Access tokens last ~1h and renew themselves; a long-stale expiry means the
    # token has not been used, and a testing-mode Google app's refresh token
    # lapses after 7 days of that.
    idle = (now - exp).days
    if idle > 6:
        return Row("google_oauth", "overdue",
                   f"last renewed about {idle} days ago; a Google app in testing mode drops idle "
                   f"refresh tokens after 7 days: re-authorise if Calendar/Gmail fail", True, True)
    return Row("google_oauth", "ok", "refresh token present and recently used", True, False)


def render(rows: List[Row]) -> str:
    width = max(len(r.name) for r in rows)
    out = [f"{'secret':<{width}}  {'status':<11}  detail", "-" * (width + 40)]
    for r in rows:
        mark = "!" if r.needs_attention else " "
        out.append(f"{r.name:<{width}}  {r.status:<11}  {mark} {r.detail}")
    due = [r.name for r in rows if r.needs_attention]
    out.append("")
    out.append(("Needs attention: " + ", ".join(due) + ". See `checklist <name>`.") if due
               else "Nothing due.")
    return "\n".join(out)


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_status(args, env, today, state) -> int:
    rows = evaluate(env, state, today)
    g = google_token_row(Path(args.google_token), datetime.now(timezone.utc))
    if g:
        rows.append(g)
    print(render(rows))
    return 1 if (args.strict and any(r.needs_attention for r in rows)) else 0


def cmd_checklist(args, **_) -> int:
    spec = BY_NAME.get(args.name)
    if not spec:
        print(f"Unknown secret '{args.name}'. Known: {', '.join(BY_NAME)}", file=sys.stderr)
        return 2
    limit = f"every {spec.max_age_days} days" if spec.max_age_days else "only if it may have leaked"
    print(f"Rotate {spec.name} ({limit}). Variables: {', '.join(spec.env)}")
    for i, step in enumerate(spec.steps, 1):
        print(f"  {i}. {step}")
    print(f"  {len(spec.steps) + 1}. Then record it:  python scripts/rotate_secrets.py record {spec.name}")
    return 0


def cmd_record(args, today, state, state_path, **_) -> int:
    spec = BY_NAME.get(args.name)
    if not spec:
        print(f"Unknown secret '{args.name}'. Known: {', '.join(BY_NAME)}", file=sys.stderr)
        return 2
    when = _parse_date(args.date) if args.date else today
    if when is None or when > today:
        print("--date must be YYYY-MM-DD and not in the future.", file=sys.stderr)
        return 2
    state[spec.name] = when.isoformat()
    save_state(state, state_path)
    print(f"Recorded: {spec.name} rotated {when}.")
    return 0


def cmd_make_data_key(**_) -> int:
    from cryptography.fernet import Fernet  # cryptography is already a dependency
    print(Fernet.generate_key().decode())
    print("\nPut that in .env as HESTIA_DATA_KEY (or in a 600-permission file named by "
          "HESTIA_DATA_KEY_FILE). It is shown once and not saved anywhere. Keep a backup "
          "copy off this disk: without it encrypted fields cannot be recovered.", file=sys.stderr)
    return 0


def cmd_reencrypt(args, env, **_) -> int:
    """Encrypt existing plain rows, or move old-key rows to the primary key (#198)."""
    import shutil
    import sqlite3
    sys.path.insert(0, str(ROOT))
    from core import crypto_store as cs

    material = cs.load_key_material(env)
    if not material:
        print("No data key: set HESTIA_DATA_KEY (or HESTIA_DATA_KEY_FILE) first.", file=sys.stderr)
        return 2
    cs.configure(enabled=True, key_material=material)
    targets = {"apollo": Path(args.apollo_db), "pluto": Path(args.pluto_db)}
    total = 0
    for name, db_path in targets.items():
        if not db_path.is_file():
            print(f"{name}: {db_path} not found, skipped.")
            continue
        backup = db_path.with_suffix(db_path.suffix + ".pre-encrypt.bak")
        if not backup.exists():
            shutil.copy2(db_path, backup)          # a plain-text copy: delete it once you have checked
        conn = sqlite3.connect(db_path)
        try:
            changed = 0
            for table, cols in cs.PROTECTED_FIELDS[name]:
                try:
                    changed += cs.encrypt_existing(conn, table, cols)
                except sqlite3.OperationalError as exc:   # table or column absent in an older DB
                    print(f"{name}.{table}: skipped ({exc})")
        finally:
            conn.close()
        total += changed
        print(f"{name}: {changed} value(s) written under the primary key. Backup: {backup}")
    print(f"Done: {total} value(s) converted."
          + (" The .bak files hold the OLD plain text; delete them once you have checked the data." if total else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state", default=str(STATE_PATH), help="rotation dates file")
    ap.add_argument("--env-file", default=str(ROOT / ".env"))
    ap.add_argument("--google-token", default=str(GOOGLE_TOKEN_PATH))
    sub = ap.add_subparsers(dest="cmd")
    st = sub.add_parser("status", help="show what is set and what is due (default)")
    st.add_argument("--strict", action="store_true", help="exit 1 if anything needs attention")
    cl = sub.add_parser("checklist", help="steps to rotate one secret")
    cl.add_argument("name")
    rc = sub.add_parser("record", help="note that a secret was rotated")
    rc.add_argument("name")
    rc.add_argument("--date", help="YYYY-MM-DD (default today)")
    sub.add_parser("make-data-key", help="generate a key for encryption at rest")
    re_ = sub.add_parser("reencrypt", help="encrypt existing rows / move them to the newest data key")
    re_.add_argument("--apollo-db", default=str(ROOT / "data" / "apollo" / "apollo.db"))
    re_.add_argument("--pluto-db", default=str(ROOT / "data" / "pluto" / "pluto.db"))
    return ap


def main(argv: Optional[List[str]] = None, today: Optional[date] = None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.cmd or "status"
    if cmd == "status" and not hasattr(args, "strict"):
        args.strict = False
    today = today or date.today()
    state_path = Path(args.state)
    env = _load_dotenv_into(dict(os.environ), Path(args.env_file))
    handlers: Dict[str, Callable] = {"status": cmd_status, "checklist": cmd_checklist,
                                     "record": cmd_record, "make-data-key": cmd_make_data_key,
                                     "reencrypt": cmd_reencrypt}
    return handlers[cmd](args=args, env=env, today=today, state=load_state(state_path),
                         state_path=state_path) if cmd != "status" else \
        cmd_status(args, env, today, load_state(state_path))


if __name__ == "__main__":
    sys.exit(main())
