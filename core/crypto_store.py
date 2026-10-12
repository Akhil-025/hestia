"""Field-level encryption at rest for the most sensitive free text (backlog #198).

What is covered, and what is not
--------------------------------
Covered when ``security.encrypt_at_rest: true`` and a key is available:

* Apollo (health): the ``notes`` text on workouts, sleep, mood, weight, meals
  and pain logs.
* Pluto (finance): the expense ``description``.

Not covered, deliberately: numbers (amounts, durations, weights), dates,
categories and names, because SQL sorts, sums and filters on them; Mnemosyne's
interaction log and facts (search, the vector index and summaries all read the
plain text, so encrypting only the SQLite copy would give a false sense of
safety); Athena and Iris indexes. A stolen disk still shows *that* you logged
a workout and *how much* you spent, just not what you wrote about it. This is
field-level encryption, not a whole-database one (SQLCipher would need a
different SQLite build).

Format and behaviour
--------------------
A stored value is ``enc:v1:<Fernet token>``. Anything without that prefix is
plain text and is returned unchanged, so turning encryption on never breaks
existing rows; ``encrypt_existing`` (or ``scripts/rotate_secrets.py reencrypt``)
converts them. Turning it off later still reads encrypted rows as long as the
key is set. Writes are plain when encryption is off.

Keys: ``HESTIA_DATA_KEY`` (one Fernet key, or ``new,old,...`` comma-separated:
the first encrypts, all decrypt, which is how rotation works) or
``HESTIA_DATA_KEY_FILE`` / ``security.key_file`` pointing at a file with the
same content. Make one with ``python scripts/rotate_secrets.py make-data-key``.

Failure modes are quiet and safe: an encrypted value with no key, or the wrong
key, reads back as ``UNREADABLE`` (a short bracketed notice), never raises and
never returns ciphertext as if it were text. Enabling encryption with no usable
key raises at startup rather than silently storing plain text.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

logger = logging.getLogger(__name__)

PREFIX = "enc:v1:"
UNREADABLE = "[encrypted: key missing or wrong]"


class CryptoConfigError(RuntimeError):
    """Encryption was requested but cannot be done safely."""


class _State:
    lock = threading.Lock()
    fernet: Optional[MultiFernet] = None   # decrypt capability (all keys)
    enabled = False                        # encrypt new writes?
    warned = False


def _parse_keys(material: str) -> List[bytes]:
    keys = [k.strip() for k in (material or "").replace("\n", ",").split(",") if k.strip()]
    out = []
    for k in keys:
        try:
            Fernet(k.encode())
        except (ValueError, TypeError) as exc:
            raise CryptoConfigError(
                "A data key is not a valid Fernet key (make one with "
                "`python scripts/rotate_secrets.py make-data-key`).") from exc
        out.append(k.encode())
    return out


def load_key_material(env: Optional[dict] = None, key_file: Optional[str] = None) -> str:
    """Key text from HESTIA_DATA_KEY, else the key file, else ''."""
    env = os.environ if env is None else env
    direct = (env.get("HESTIA_DATA_KEY") or "").strip()
    if direct:
        return direct
    path = (env.get("HESTIA_DATA_KEY_FILE") or key_file or "").strip()
    if path:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise CryptoConfigError(f"Cannot read the data key file '{path}': {exc.strerror}") from exc
    return ""


def configure(*, enabled: bool, key_material: str = "") -> None:
    """Set up (or reset) the process-wide state. Raises if enabled with no key."""
    keys = _parse_keys(key_material)
    with _State.lock:
        _State.fernet = MultiFernet([Fernet(k) for k in keys]) if keys else None
        _State.enabled = bool(enabled)
        _State.warned = False
        if enabled and not keys:
            _State.enabled = False
            raise CryptoConfigError(
                "security.encrypt_at_rest is on but no data key is set. Set HESTIA_DATA_KEY "
                "(or HESTIA_DATA_KEY_FILE) first, or turn the option off.")


def configure_from_config(cfg: Optional[dict], env: Optional[dict] = None) -> bool:
    """Called once at startup. Returns whether new writes will be encrypted.

    A key is loaded even when encryption is off, so rows written earlier stay readable.
    """
    sec = (cfg or {}).get("security") or {}
    enabled = bool(sec.get("encrypt_at_rest", False))
    material = load_key_material(env, sec.get("key_file"))
    configure(enabled=enabled, key_material=material)
    return _State.enabled


def reset() -> None:
    with _State.lock:
        _State.fernet, _State.enabled, _State.warned = None, False, False


def is_enabled() -> bool:
    return _State.enabled


def is_encrypted(value) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def encrypt_text(value):
    """Ciphertext for new writes when enabled; otherwise ``value`` unchanged.

    ``None``, empty strings and already-encrypted values pass through, so calling
    it twice is harmless.
    """
    if not _State.enabled or _State.fernet is None:
        return value
    if value is None or not isinstance(value, str) or value == "" or is_encrypted(value):
        return value
    return PREFIX + _State.fernet.encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_text(value):
    """Plain text for ``value``. Plain input is returned as is; never raises."""
    if not is_encrypted(value):
        return value
    f = _State.fernet
    if f is None:
        _warn_once("Encrypted rows found but no data key is set.")
        return UNREADABLE
    try:
        return f.decrypt(value[len(PREFIX):].encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError, UnicodeError):
        _warn_once("An encrypted row could not be decrypted with the configured key(s).")
        return UNREADABLE


def _warn_once(msg: str) -> None:
    if not _State.warned:
        _State.warned = True
        logger.warning(msg)


def needs_rewrite(value) -> bool:
    """True if ``value`` should be rewritten: plain text while encryption is on,
    or encrypted under a non-primary (older) key that still decrypts."""
    if not _State.enabled or _State.fernet is None or not isinstance(value, str) or value == "":
        return False
    if not is_encrypted(value):
        return True
    if _primary_decrypts(value):
        return False
    return decrypt_text(value) != UNREADABLE


def _primary_decrypts(value: str) -> bool:
    primary = _State.fernet._fernets[0]  # type: ignore[union-attr]
    try:
        primary.decrypt(value[len(PREFIX):].encode("ascii"))
        return True
    except (InvalidToken, ValueError):
        return False


class DecryptingRow(sqlite3.Row):
    """``sqlite3.Row`` that decrypts string values as they are read.

    Use as ``conn.row_factory = DecryptingRow``. ``row["notes"]``, ``row[2]``,
    ``dict(row)`` and iteration all see plain text; unencrypted values cost one
    ``startswith`` check.
    """

    def __getitem__(self, key):
        v = super().__getitem__(key)
        if isinstance(key, slice):
            return tuple(decrypt_text(x) for x in v)
        return decrypt_text(v)

    def __iter__(self):
        return (decrypt_text(v) for v in super().__iter__())


# ── converting existing data ─────────────────────────────────────────────────

def encrypt_existing(conn: sqlite3.Connection, table: str, columns: Sequence[str],
                     key_column: str = "id") -> int:
    """Encrypt (or re-encrypt under the primary key) the given text columns in place.

    Returns the number of values rewritten. Safe to run repeatedly. Table and
    column names come from code, never from user input.
    """
    if not _State.enabled:
        raise CryptoConfigError("Encryption is not enabled; nothing to convert.")
    cols = ", ".join(columns)
    saved = conn.row_factory
    conn.row_factory = None            # raw tuples: we need the stored text, not decrypted text
    try:
        rows = conn.execute(f"SELECT {key_column}, {cols} FROM {table}").fetchall()  # noqa: S608
        changed = 0
        for row in rows:
            updates = {}
            for col, stored in zip(columns, row[1:]):
                if needs_rewrite(stored):
                    updates[col] = encrypt_text(decrypt_text(stored))
            if updates:
                sets = ", ".join(f"{c} = ?" for c in updates)
                conn.execute(f"UPDATE {table} SET {sets} WHERE {key_column} = ?",  # noqa: S608
                             (*updates.values(), row[0]))
                changed += len(updates)
        conn.commit()
        return changed
    finally:
        conn.row_factory = saved


# Tables and columns covered, as (db file key, table, columns). One place so the
# docs, the converter and the tests cannot drift apart.
PROTECTED_FIELDS = {
    "apollo": [("workouts", ("notes",)), ("sleep_logs", ("notes",)), ("mood_logs", ("notes",)),
               ("weight_logs", ("notes",)), ("meal_logs", ("notes",)), ("pain_logs", ("notes",))],
    "pluto": [("expenses", ("description",))],
}
