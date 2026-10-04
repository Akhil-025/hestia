# modules/hephaestus/monitor.py
"""
Page monitors for Hephaestus (backlog #101 scheduled checks, #109 change
detection).

Three small, separately testable pieces:

  validate_monitor_url()  refuses URLs a standing, unattended job must never
                          fetch (local/private addresses, embedded logins).
  analyse_change()        decides whether a new snapshot is a *meaningful*
                          change versus the last one. Pure, no I/O.
  MonitorStore            SQLite persistence: monitors, last snapshot, and a
                          queue of alerts that survives a restart.

What counts as meaningful
-------------------------
Pages churn constantly (view counters, clocks, rotating banners), so "the text
differs" is never enough. The rules are deliberately simple and explainable:

  * keywords given      -> alert only when a keyword *newly appears*.
  * price tracking      -> alert when the price falls, or first reaches the
                           target. A rise updates the baseline silently.
  * neither             -> alert when at least ``min_changed_words`` alphabetic
                           words were added/removed. Digits-only changes
                           (counters, timestamps) never count by themselves,
                           and re-ordering the same words is not a change.

The first successful check only records a baseline; it never alerts (except a
price already at or under the target).
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import sqlite3
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Optional
from urllib.parse import urlparse

MIN_INTERVAL_MINUTES = 30
DEFAULT_INTERVAL_MINUTES = 24 * 60
MAX_INTERVAL_MINUTES = 30 * 24 * 60
MAX_SNAPSHOT_CHARS = 60_000
FAILURE_ALERT_AFTER = 3          # consecutive failed checks before telling the user
DEFAULT_MIN_CHANGED_WORDS = 3
_EXCERPT_TOKEN_CAP = 4000        # skip the (slower) excerpt diff beyond this

_ALPHA_WORD = re.compile(r"[a-z]{2,}", re.I)
_PRICE_RE = re.compile(
    r"(₹|\bRs\.?|\bINR|US\$|\$|€|£)\s*"
    r"(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)",
    re.I,
)
_BLOCKED_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home", ".corp")


# ---------------------------------------------------------------------------
# URL guard
# ---------------------------------------------------------------------------

def validate_monitor_url(url: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(clean_url, None)`` or ``(None, reason)``.

    Only http(s) URLs to public-looking hosts are accepted: a monitor runs
    unattended and repeatedly, so it must not be steerable at the router page,
    a local service, or a link carrying credentials. This checks the address
    as written; it does not resolve DNS, so a public name that points at a
    private address is not caught.
    """
    raw = (url or "").strip()
    if not raw:
        return None, "no URL given"
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parts = urlparse(raw)
        host = (parts.hostname or "").lower()
    except ValueError:
        return None, "that isn't a valid URL"
    if parts.scheme not in ("http", "https"):
        return None, "only http and https pages can be watched"
    if not host or " " in host:
        return None, "that isn't a valid URL"
    if parts.username or parts.password:
        return None, "URLs with a login embedded in them aren't allowed"
    if host == "localhost" or host.endswith(_BLOCKED_SUFFIXES):
        return None, "local addresses can't be watched"
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        if "." not in host:
            return None, "that doesn't look like a public website"
    else:
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            return None, "local or private addresses can't be watched"
    return raw, None


def host_of(url: str) -> str:
    try:
        return (urlparse(url if "://" in url else "https://" + url).hostname or "").lower()
    except ValueError:
        return ""


# ---------------------------------------------------------------------------
# Change detection (pure)
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Collapse all whitespace; the only normalisation snapshots get."""
    return " ".join((text or "").split())[:MAX_SNAPSHOT_CHARS]


def extract_price(text: str) -> tuple[Optional[float], str]:
    """First currency amount in *text* as ``(value, symbol)``, else ``(None, "")``."""
    m = _PRICE_RE.search(text or "")
    if not m:
        return None, ""
    try:
        return float(m.group(2).replace(",", "")), m.group(1)
    except ValueError:
        return None, ""


def format_price(value: float, symbol: str = "") -> str:
    body = f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{symbol}{body}" if symbol else body


def _alpha_counter(text: str) -> Counter:
    return Counter(w.lower() for w in _ALPHA_WORD.findall(text or ""))


def word_changes(old: str, new: str) -> tuple[int, int]:
    """``(added, removed)`` counts of alphabetic words, ignoring order."""
    a, b = _alpha_counter(old), _alpha_counter(new)
    return sum((b - a).values()), sum((a - b).values())


def change_excerpt(old: str, new: str, limit: int = 90) -> str:
    """A short 'what's new' phrase for the alert, or ``""`` when the pages
    are too large to diff cheaply."""
    a, b = old.split(), new.split()
    if len(a) > _EXCERPT_TOKEN_CAP or len(b) > _EXCERPT_TOKEN_CAP:
        return ""
    added: list[str] = []
    for tag, _i1, _i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("insert", "replace"):
            added.extend(b[j1:j2])
        if len(" ".join(added)) > limit * 2:
            break
    text = " ".join(added)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def analyse_change(
    old_text: Optional[str],
    new_text: str,
    *,
    keywords: tuple[str, ...] | list[str] = (),
    track_price: bool = False,
    old_price: Optional[float] = None,
    target_price: Optional[float] = None,
    min_changed_words: int = DEFAULT_MIN_CHANGED_WORDS,
) -> dict[str, Any]:
    """Compare a new snapshot with the previous one.

    Returns ``{"alerts": [str, ...], "price": float|None, "symbol": str,
    "baseline": bool}``. ``old_text is None`` means this is the first check.
    """
    alerts: list[str] = []
    baseline = old_text is None
    price, symbol = extract_price(new_text) if track_price else (None, "")

    if keywords and not baseline:
        old_l, new_l = old_text.lower(), new_text.lower()
        for kw in keywords:
            k = kw.strip().lower()
            if k and k in new_l and k not in old_l:
                alerts.append(f"“{kw.strip()}” now appears")

    if track_price and price is not None:
        at_target = target_price is not None and price <= target_price
        was_at_target = target_price is not None and old_price is not None and old_price <= target_price
        if old_price is not None and price < old_price:
            msg = f"price dropped from {format_price(old_price, symbol)} to {format_price(price, symbol)}"
            if at_target:
                msg += " — at or below your target"
            alerts.append(msg)
        elif at_target and not was_at_target:
            alerts.append(f"price is {format_price(price, symbol)}, at or below your target")

    if not keywords and not track_price and not baseline:
        if hashlib.sha1(old_text.encode("utf-8", "ignore")).digest() != \
                hashlib.sha1(new_text.encode("utf-8", "ignore")).digest():
            added, removed = word_changes(old_text, new_text)
            if added + removed >= max(1, int(min_changed_words)):
                excerpt = change_excerpt(old_text, new_text)
                msg = f"{added} words added, {removed} removed"
                alerts.append(f"{msg}: “{excerpt}”" if excerpt else msg)

    return {"alerts": alerts, "price": price, "symbol": symbol, "baseline": baseline}


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


class MonitorStore:
    """SQLite store for page monitors and their pending alerts.

    Same conventions as modules/ares/db.py: one connection, one lock, WAL when
    on disk, schema created idempotently on open.
    """

    def __init__(self, db_path: str = ":memory:", max_monitors: int = 20) -> None:
        self.max_monitors = max_monitors
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if db_path != ":memory:":
            with self._conn:
                self._conn.execute("PRAGMA journal_mode=WAL;")
        with self._conn:
            self._conn.executescript("""
CREATE TABLE IF NOT EXISTS monitors (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    url              TEXT NOT NULL,
    keywords         TEXT NOT NULL DEFAULT '[]',
    track_price      INTEGER NOT NULL DEFAULT 0,
    target_price     REAL,
    selector         TEXT,
    scraper          TEXT,
    interval_minutes INTEGER NOT NULL,
    created_at       TEXT NOT NULL,
    last_checked     TEXT,
    last_changed     TEXT,
    snapshot         TEXT,
    last_price       REAL,
    currency         TEXT NOT NULL DEFAULT '',
    failures         INTEGER NOT NULL DEFAULT 0,
    failure_alerted  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS monitor_alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    monitor_id  INTEGER,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    delivered   INTEGER NOT NULL DEFAULT 0
);
""")

    # -- monitors -------------------------------------------------------

    def add(
        self, *, name: str, url: str, now: datetime, keywords=(), track_price=False,
        target_price: Optional[float] = None, selector: Optional[str] = None,
        scraper: Optional[str] = None, interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    ) -> int:
        interval = max(MIN_INTERVAL_MINUTES, min(int(interval_minutes), MAX_INTERVAL_MINUTES))
        with self._lock, self._conn:
            if self._conn.execute(
                "SELECT 1 FROM monitors WHERE lower(name)=lower(?)", (name,)
            ).fetchone():
                raise ValueError("duplicate")
            if self._conn.execute("SELECT COUNT(*) FROM monitors").fetchone()[0] >= self.max_monitors:
                raise ValueError("limit")
            cur = self._conn.execute(
                "INSERT INTO monitors (name,url,keywords,track_price,target_price,selector,"
                "scraper,interval_minutes,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (name, url, json.dumps([k for k in keywords if k]), int(bool(track_price)),
                 target_price, selector, scraper, interval, _iso(now)),
            )
            return int(cur.lastrowid)

    def list(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM monitors ORDER BY id").fetchall()
        return [self._row(r) for r in rows]

    def find(self, text: str) -> list[dict]:
        """Monitors whose name or URL matches *text* (exact first, else substring)."""
        t = (text or "").strip().lower()
        if not t:
            return []
        monitors = self.list()
        exact = [m for m in monitors if t in (m["name"].lower(), m["url"].lower())]
        if exact:
            return exact
        return [m for m in monitors if t in m["name"].lower() or t in m["url"].lower()]

    def remove(self, monitor_id: int) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM monitors WHERE id=?", (monitor_id,))
            self._conn.execute("DELETE FROM monitor_alerts WHERE monitor_id=?", (monitor_id,))
            return cur.rowcount > 0

    def due(self, now: datetime, slack_minutes: int = 5) -> list[dict]:
        """Monitors never checked, or whose interval (less *slack*) has elapsed.
        The slack stops a 30-minute heartbeat from skipping a 30-minute monitor
        because the tick landed a few seconds early."""
        out = []
        for m in self.list():
            last = _parse(m["last_checked"])
            if last is None or now - last >= timedelta(minutes=m["interval_minutes"] - slack_minutes):
                out.append(m)
        return out

    def record_success(
        self, monitor_id: int, *, snapshot: str, now: datetime, changed: bool,
        price: Optional[float], currency: str,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE monitors SET snapshot=?, last_checked=?, failures=0, failure_alerted=0, "
                "last_changed=CASE WHEN ? THEN ? ELSE last_changed END, "
                "last_price=COALESCE(?, last_price), "
                "currency=CASE WHEN ? IS NOT NULL THEN ? ELSE currency END WHERE id=?",
                (snapshot, _iso(now), int(changed), _iso(now), price, price, currency, monitor_id),
            )

    def record_failure(self, monitor_id: int, now: datetime) -> tuple[int, bool]:
        """Count a failed check. Returns ``(consecutive_failures, should_alert)``;
        the alert fires once per failure streak, not on every retry."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE monitors SET failures=failures+1, last_checked=? WHERE id=?",
                (_iso(now), monitor_id),
            )
            row = self._conn.execute(
                "SELECT failures, failure_alerted FROM monitors WHERE id=?", (monitor_id,)
            ).fetchone()
            if row is None:
                return 0, False
            should = row["failures"] >= FAILURE_ALERT_AFTER and not row["failure_alerted"]
            if should:
                self._conn.execute("UPDATE monitors SET failure_alerted=1 WHERE id=?", (monitor_id,))
            return int(row["failures"]), bool(should)

    # -- alerts ---------------------------------------------------------

    def add_alert(self, monitor_id: Optional[int], text: str, now: datetime) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO monitor_alerts (monitor_id,text,created_at) VALUES (?,?,?)",
                (monitor_id, text, _iso(now)),
            )

    def pending_alerts(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, monitor_id, text, created_at FROM monitor_alerts "
                "WHERE delivered=0 ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_delivered(self, alert_ids: list[int]) -> None:
        if not alert_ids:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                "UPDATE monitor_alerts SET delivered=1 WHERE id=?", [(i,) for i in alert_ids]
            )

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        try:
            d["keywords"] = json.loads(d.get("keywords") or "[]")
        except ValueError:
            d["keywords"] = []
        d["track_price"] = bool(d["track_price"])
        return d
