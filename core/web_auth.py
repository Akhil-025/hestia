"""
core/web_auth.py - password login for the web UI (backlog #186).

The web UI already had an optional shared API key (``X-API-Key``). That is
fine for scripts but awkward for a person: the key sits in session storage and
nothing signs you out. This adds the thing HEARTH.txt asked for - a password,
a login page and a signed session cookie - and leaves the API key working
alongside it for scripts.

Pieces
------
``PasswordAuth``   checks a password (hashed, or plain from the environment)
                   and throttles repeated wrong guesses per client address.
``resolve_secret_key``  where the cookie-signing key comes from.
``safe_next``      keeps "?next=" from becoming an open redirect.

Nothing here knows about Flask; ``web_ui.py`` does the cookie handling.

Setting a password
------------------
Preferred: ``python scripts/hash_web_password.py`` prints a hash to paste into
``webui.password_hash`` in laptop_config.yaml (the file never holds the
password itself). Also accepted: the ``HESTIA_WEB_PASSWORD`` environment
variable, or ``webui.password`` (plain text in a file that is easy to commit
by accident - it works, and logs a warning).
"""
from __future__ import annotations

import hmac
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MIN_SECRET_LEN = 32


class PasswordAuth:
    def __init__(
        self,
        password: Optional[str] = None,
        password_hash: Optional[str] = None,
        *,
        max_failures: int = 5,
        lockout_seconds: float = 60.0,
        clock=time.monotonic,
    ) -> None:
        self._password = password or None
        self._hash = password_hash or None
        self.max_failures = max(1, int(max_failures))
        self.lockout_seconds = float(lockout_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        # ip -> [consecutive failures, locked_until (monotonic) or 0]
        self._fails: dict[str, list[float]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._password or self._hash)

    # -- checking ---------------------------------------------------------

    def verify(self, candidate: str) -> bool:
        """True if *candidate* is the password. Constant-time for plain text."""
        if not self.enabled or not isinstance(candidate, str) or not candidate:
            return False
        if self._hash:
            try:
                from werkzeug.security import check_password_hash
                if check_password_hash(self._hash, candidate):
                    return True
            except Exception:
                logger.exception("[WebAuth] password hash could not be checked")
        if self._password:
            return hmac.compare_digest(
                self._password.encode("utf-8"), candidate.encode("utf-8")
            )
        return False

    # -- throttling -------------------------------------------------------

    def locked_for(self, ip: str) -> int:
        """Seconds left on *ip*'s lockout (0 = not locked)."""
        with self._lock:
            rec = self._fails.get(ip)
            if not rec:
                return 0
            left = rec[1] - self._clock()
            if left <= 0 and rec[1]:
                # Lockout over: forget the strikes so one more typo is not an
                # instant new lockout.
                self._fails.pop(ip, None)
                return 0
            return int(left) + 1 if left > 0 else 0

    def record_failure(self, ip: str) -> int:
        """Count a wrong password. Returns lockout seconds (0 = none yet)."""
        with self._lock:
            now = self._clock()
            # Drop stale entries so the table cannot grow without bound.
            if len(self._fails) > 1000:
                self._fails = {k: v for k, v in self._fails.items() if v[1] > now}
            rec = self._fails.setdefault(ip, [0, 0.0])
            rec[0] += 1
            if rec[0] >= self.max_failures:
                rec[0] = 0
                rec[1] = now + self.lockout_seconds
                return int(self.lockout_seconds)
            return 0

    def record_success(self, ip: str) -> None:
        with self._lock:
            self._fails.pop(ip, None)


def resolve_secret_key(configured: Optional[str], key_file: Optional[str]) -> str:
    """The key that signs session cookies.

    Order: an explicit setting (``webui.secret_key`` / ``HESTIA_WEB_SECRET``),
    then a key generated once and kept in *key_file* (so a restart does not
    sign everyone out), then a throwaway in-memory key if the file cannot be
    written. A throwaway key only means logging in again after a restart.
    """
    if configured and len(configured) >= MIN_SECRET_LEN:
        return configured
    if configured:
        logger.warning(
            "[WebAuth] secret_key is shorter than %d characters; ignoring it.",
            MIN_SECRET_LEN,
        )
    if key_file:
        path = Path(key_file)
        try:
            if path.exists():
                existing = path.read_text(encoding="utf-8").strip()
                if len(existing) >= MIN_SECRET_LEN:
                    return existing
            fresh = secrets.token_hex(32)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(fresh)
            os.replace(tmp, path)
            return fresh
        except OSError:
            logger.warning(
                "[WebAuth] could not store a session key at %s; sessions will "
                "end when Hestia restarts.", key_file,
            )
    return secrets.token_hex(32)


def safe_next(target: Optional[str], default: str = "/") -> str:
    """A post-login redirect target that stays on this site.

    Only a plain path is allowed. ``//evil.example``, ``https://evil.example``
    and ``/\\evil.example`` (browsers treat the backslash as a slash) fall back
    to *default*.
    """
    if not target or not isinstance(target, str):
        return default
    if not target.startswith("/") or target.startswith("//") or "\\" in target:
        return default
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc:
        return default
    if any(ch in target for ch in ("\r", "\n")):
        return default
    return target
