"""
modules/pluto/alerts.py

Price-movement and news alerts (backlog #132).

Opt-in: nothing is fetched until the user sets a threshold with
``set_price_alert`` (so a heartbeat tick never makes surprise network calls).
Once on, ``check()`` is called from the heartbeat and:

* looks at the stocks you hold plus a watchlist (at most MAX_WATCH names) and
  speaks when one has moved by at least the threshold percent between its last
  two daily closes, once per ticker per day per direction;
* optionally checks recent headlines for those names and speaks when a headline
  from the last two days contains a risk or event word (probe, downgrade, results,
  acquisition ...). That is keyword matching, not reading: the alert quotes the
  headline so you can judge it, and says it is only a keyword match.

It stays quiet in quiet hours (nothing is recorded, so the alert comes out later),
checks at most every CHECK_INTERVAL_S, stops at the first throttled response
instead of hammering the data source, and never raises.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from typing import Any, Callable, Optional

from .logging_config import get_logger
from .market_data import MarketDataError, MarketDataThrottled, fetch_price_history, normalise_ticker
from .planning import QUIET_HOURS, parse_amount

logger = get_logger(__name__)

THRESHOLD_KEY, WATCH_KEY, NEWS_KEY = "price_alert_threshold", "price_alert_watch", "price_alert_news"
DEFAULT_THRESHOLD, MIN_THRESHOLD, MAX_THRESHOLD = 5.0, 0.5, 50.0
MAX_WATCH = 15
MAX_NEWS_NAMES = 6
MAX_ITEMS_PER_ALERT = 3
CHECK_INTERVAL_S = 1800
NEWS_MAX_AGE_DAYS = 2
NEWS_WORDS = re.compile(
    r"\b(probe|raid|fraud|lawsuit|sues?|penalty|fine[sd]?|downgrade[sd]?|upgrade[sd]?|resigns?|"
    r"acquisition|acquires?|merger|buyback|dividend|results|profit warning|bankruptcy|default|"
    r"recall|ban(?:ned)?|investigation|sebi|ed)\b", re.I)
_OFF = {"off", "stop", "disable", "none", "clear"}


def _name_ok(raw: Any) -> Optional[str]:
    text = re.sub(r"[^A-Za-z0-9&.\-_ ]", "", str(raw or "")).strip()[:30]
    return text or None


class PriceAlerts:
    def __init__(self, db: Any, history_fn: Callable = fetch_price_history,
                 news_fn: Optional[Callable[[str], list]] = None,
                 now_fn: Callable[[], datetime] = datetime.now,
                 clock: Callable[[], float] = time.monotonic):
        self.db, self._history, self._news = db, history_fn, news_fn
        self._now, self._clock = now_fn, clock
        self._last_check: Optional[float] = None

    # -- settings ----------------------------------------------------------------

    def _threshold(self) -> Optional[float]:
        raw = self.db.get_setting(THRESHOLD_KEY)
        try:
            v = float(raw) if raw is not None else None
        except ValueError:
            return None
        return v if v and v > 0 else None

    def _watch(self) -> list[str]:
        try:
            data = json.loads(self.db.get_setting(WATCH_KEY) or "[]")
        except ValueError:
            return []
        return [str(x) for x in data][:MAX_WATCH] if isinstance(data, list) else []

    def configure(self, entities: dict) -> dict:
        """Set or show the alert: threshold, watch/unwatch a ticker, news on/off."""
        changed = []
        raw_thr = entities.get("threshold")
        if raw_thr not in (None, ""):
            if str(raw_thr).strip().lower() in _OFF:
                self.db.delete_setting(THRESHOLD_KEY)
                return _reply("Price alerts are off.", 0.9)
            v = parse_amount(str(raw_thr).replace("%", "").replace("percent", ""))
            if v is None or not MIN_THRESHOLD <= v <= MAX_THRESHOLD:
                return _reply(f"The move needs to be between {MIN_THRESHOLD:g}% and {MAX_THRESHOLD:g}%.", 0.4)
            self.db.set_setting(THRESHOLD_KEY, repr(v))
            changed.append(f"alert at {v:g}% moves")
        watch = self._watch()
        add, remove = _name_ok(entities.get("watch") or entities.get("add")), _name_ok(entities.get("unwatch") or entities.get("remove"))
        if add and add.lower() not in (w.lower() for w in watch):
            if len(watch) >= MAX_WATCH:
                return _reply(f"The watchlist holds {MAX_WATCH} names at most.", 0.5)
            watch.append(add)
            changed.append(f"watching {add}")
        if remove:
            keep = [w for w in watch if w.lower() != remove.lower()]
            if len(keep) != len(watch):
                watch = keep
                changed.append(f"stopped watching {remove}")
        if changed and (add or remove):
            self.db.set_setting(WATCH_KEY, json.dumps(watch))
        news = str(entities.get("news") or "").strip().lower()
        if news in ("on", "yes", "true", "off", "no", "false"):
            self.db.set_setting(NEWS_KEY, "on" if news in ("on", "yes", "true") else "off")
            changed.append("headline alerts " + ("on" if news in ("on", "yes", "true") else "off"))
        if changed and self._threshold() is None and (add or remove or news):
            self.db.set_setting(THRESHOLD_KEY, repr(DEFAULT_THRESHOLD))
            changed.append(f"alert at the default {DEFAULT_THRESHOLD:g}%")
        return _reply(self._status(changed), 0.9, {"threshold": self._threshold(), "watch": self._watch(),
                                                   "news": self.db.get_setting(NEWS_KEY) != "off"})

    def _status(self, changed: list) -> str:
        thr = self._threshold()
        head = ("Updated: " + ", ".join(changed) + ". ") if changed else ""
        if thr is None:
            return head + "Price alerts are off. Say e.g. 'alert me when my stocks move 5 percent' to turn them on."
        news = "on" if self.db.get_setting(NEWS_KEY) != "off" else "off"
        watch = self._watch()
        return (head + f"Price alerts are on at {thr:g}% (stocks you hold" +
                (f" plus {', '.join(watch)}" if watch else "") + f"). Headline keyword alerts are {news}. "
                "Checked about every 30 minutes, quiet 22:00-07:00.")

    # -- names to look at ------------------------------------------------------------

    def _names(self) -> list[str]:
        seen, out = set(), []
        held = [r["name"] for r in self.db.get_investments() if (r.get("type") or "stock") == "stock"]
        for n in [*held, *self._watch()]:
            k = normalise_ticker(n)
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out[:MAX_WATCH]

    # -- heartbeat ---------------------------------------------------------------------

    def check(self) -> Optional[str]:
        thr = self._threshold()
        if thr is None:
            return None
        now = self._now()
        lo, hi = QUIET_HOURS
        if now.hour >= lo or now.hour < hi:
            return None
        tick = self._clock()
        if self._last_check is not None and tick - self._last_check < CHECK_INTERVAL_S:
            return None
        self._last_check = tick
        names = self._names()
        if not names:
            return None
        today = now.date().isoformat()
        items: list[tuple[str, str]] = []          # (alert key, spoken text)
        for sym in names:
            try:
                s = self._history(sym, "1mo")
            except MarketDataThrottled:
                logger.info("price alerts: data source throttled, stopping this round")
                break
            except MarketDataError:
                continue
            if len(s.closes) < 2 or not s.closes[-2]:
                continue
            move = (s.closes[-1] / s.closes[-2] - 1.0) * 100
            if abs(move) >= thr:
                key = f"px:{today}:{sym}:{'up' if move > 0 else 'down'}"
                if not self.db.alert_already_sent(key):
                    items.append((key, f"{sym} is {'up' if move > 0 else 'down'} {abs(move):.1f}% on its last close"))
        if self._news and self.db.get_setting(NEWS_KEY) != "off":
            items += self._news_items(names[:MAX_NEWS_NAMES], now)
        if not items:
            return None
        items = items[:MAX_ITEMS_PER_ALERT]
        for key, _ in items:
            self.db.mark_alert_sent(key)
        return "Heads up: " + "; ".join(t for _, t in items) + "."

    def _news_items(self, names: list[str], now: datetime) -> list[tuple[str, str]]:
        out = []
        for sym in names:
            try:
                headlines = self._news(sym) or []
            except Exception:
                continue
            for h in headlines:
                title = str(h.get("title") or "")
                if not NEWS_WORDS.search(title):
                    continue
                pub = h.get("published")
                try:
                    age = (now.date() - datetime.strptime(pub, "%Y-%m-%d").date()).days if pub else None
                except ValueError:
                    age = None
                if age is None or age > NEWS_MAX_AGE_DAYS:
                    continue
                key = "news:" + hashlib.sha1(f"{sym}|{title}".encode()).hexdigest()[:16]
                if self.db.alert_already_sent(key):
                    continue
                out.append((key, f"headline on {sym} (keyword match only): {title[:110]}"))
                break                                  # one headline per name per round
        return out


def _reply(text: str, confidence: float = 0.9, data: Optional[dict] = None) -> dict:
    return {"response": text, "data": data or {}, "confidence": confidence}
