"""
core/whatif.py  (backlog #160: lightweight what-if simulator)

"What if I cancel Netflix?" / "What if I stop meditating?" / "What if I slept
an hour less?" Each of these has a downstream effect that already sits in
Hestia's own data. This projects it from that data: plain arithmetic over
numbers the modules already hold, with the working shown.

What it deliberately is not
---------------------------
* Not a Monte Carlo engine. Ares already has one (``simulate_outcomes``, for
  decisions under uncertainty); this has no random draws, no distributions
  and no forecasts of your behaviour. It answers "if this one thing changed
  and nothing else did, what do my own numbers say?".
* Not a writer. Nothing here changes any module's data. It only reads.
* Not an advisor. It reports effects, then stops. It does not tell you
  whether to do the thing.

Three kinds of change are understood, because those are the three whose
effect can be worked out from data Hestia really has:

  cut_expense   a subscription or recurring charge (Pluto's detected
                recurring list), or a stated amount.  -> money freed up,
                and the effect on this month's "safe to spend today".
  quit_habit    a habit Artemis tracks.  -> the streak that resets, how
                often you've actually been doing it, and what in Apollo or
                your goals sits near it.
  change_sleep  "an hour less sleep", "only 5 hours".  -> the new 7-night
                average against the line where the burnout check starts
                treating sleep as a rest signal.

Anything else gets an honest "I can't project that from my data" plus the
shapes it does understand, rather than a made-up number.

Like ``core/consensus.py``, it reads sibling modules through objects handed
to it (``pluto``/``artemis``/``apollo``). Nothing is imported from a module,
so the "no module calls another" rule holds: this is orchestrator-level, not
a module.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Keep in step with modules/apollo/engine.py's _LOW_SLEEP_THRESHOLD, the
# average below which the burnout/rest check counts sleep as a rest signal.
# tests/test_whatif.py asserts the two are equal, so they can't drift.
LOW_SLEEP_HOURS = 6.0

_QUIT_WORDS = re.compile(
    r"\b(stop|stopped|quit|quitting|drop|dropping|skip|skipping|give up|giving up|"
    r"abandon|no longer|without|cut out|cancel)\b", re.I)
_CUT_WORDS = re.compile(
    r"\b(cancel|cancelling|canceling|cut|cutting|drop|dropping|stop|stopping|"
    r"unsubscribe|reduce|reducing|quit|quitting|end|ending|without|no more)\b", re.I)
_SLEEP_WORDS = re.compile(r"\b(sleep|sleeping|slept|sleeps)\b", re.I)
_FITNESS = re.compile(
    r"\b(run|running|jog|gym|workout|exercise|lift|lifting|yoga|walk|walking|"
    r"swim|swimming|cycle|cycling|stretch|stretching|train|training)\b", re.I)
_STOP = frozenset({"a", "an", "the", "my", "of", "to", "and", "for", "in", "on",
                   "daily", "every", "day", "habit", "goal", "do", "doing"})

SUPPORTED_EXAMPLES = (
    "what if I cancel my Netflix subscription",
    "what if I stop meditating (a habit you track)",
    "what if I slept an hour less",
)


_SUFFIXES = ("ating", "ation", "ate", "ing", "ed", "es", "s", "e")


def _stem(word: str) -> str:
    """Crude suffix stripping so 'meditating', 'meditate' and 'meditation'
    all compare equal. Deliberately simple: a wrong match here only means a
    habit isn't recognised, in which case the user is asked, never that a
    wrong habit is projected, because a habit must still match by name."""
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    if len(word) > 3 and word[-1] == word[-2]:
        word = word[:-1]
    return word


def _words(text: str) -> set[str]:
    return {
        _stem(w) for w in re.findall(r"[a-z]+", (text or "").lower())
        if w not in _STOP and len(w) > 2
    }


def _money(value: float, fmt: Optional[Callable[[float], str]] = None) -> str:
    if fmt is not None:
        try:
            return fmt(value)
        except Exception:
            pass
    return f"{value:,.0f}" if abs(value - round(value)) < 0.005 else f"{value:,.2f}"


def parse_amount_per_month(text: str) -> Optional[float]:
    """An amount the user stated, normalised to per month, or None.

    "cut 500 a month", "save 1,200 per week" (x 52/12), "spend 50 less every
    day" (x 365/12), "6000 a year" (/12). A bare number with no period is
    taken as monthly, the unit people mean by default for bills.
    """
    m = re.search(r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", text or "")
    if not m:
        return None
    try:
        amount = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    if m.group(2):
        amount *= 1000
    # A bare small number is a count ("cancel 3 subscriptions"), not money.
    if amount < 10:
        return None
    tail = (text or "")[m.end():].lower()
    if re.search(r"\b(a|per|every|each)\s+(week|wk)\b|\bweekly\b", tail):
        return amount * 52 / 12
    if re.search(r"\b(a|per|every|each)\s+day\b|\bdaily\b", tail):
        return amount * 365 / 12
    if re.search(r"\b(a|per|every|each)\s+(year|yr)\b|\b(yearly|annually)\b", tail):
        return amount / 12
    return amount


def parse_sleep_change(text: str) -> Optional[dict[str, Any]]:
    """``{"delta": hours}`` for "an hour less" / "2 hours more", or
    ``{"target": hours}`` for "only 5 hours of sleep"; None if neither."""
    t = (text or "").lower()
    if not _SLEEP_WORDS.search(t):
        return None
    m = re.search(r"(\d+(?:\.\d+)?|an|one|half an|half)\s*(?:hours?|hrs?|h)\b", t)
    if not m:
        return None
    raw = m.group(1)
    hours = {"an": 1.0, "one": 1.0, "half an": 0.5, "half": 0.5}.get(raw)
    if hours is None:
        hours = float(raw)
    if re.search(r"\b(less|fewer|lose|losing|lost|cut|cutting|shorter|early|earlier)\b", t):
        return {"delta": -hours}
    if re.search(r"\b(more|extra|additional|longer|add|adding)\b", t):
        return {"delta": hours}
    if re.search(r"\b(only|just|exactly|about|around|sleep|slept|sleeping)\b", t) and 2 <= hours <= 14:
        return {"target": hours}
    return None


class WhatIfEngine:
    """Project the downstream effect of one proposed change from existing data."""

    def __init__(
        self,
        pluto: Optional[Any] = None,
        artemis: Optional[Any] = None,
        apollo: Optional[Any] = None,
        today: Optional[Callable[[], date]] = None,
    ) -> None:
        self.pluto = pluto
        self.artemis = artemis
        self.apollo = apollo
        self._today = today or (lambda: datetime.now(timezone.utc).date())

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def project(self, query: str, entities: Optional[dict] = None) -> dict:
        """``{"response", "data", "confidence"}``. Never raises."""
        try:
            return self._project(query or "", entities or {})
        except Exception:
            logger.exception("what-if: projection failed.")
            return {
                "response": "I couldn't work that out from my data just now.",
                "data": {"projected": False},
                "confidence": 0.2,
            }

    def _project(self, query: str, entities: dict) -> dict:
        kind = str(entities.get("change") or entities.get("kind") or "").strip().lower()
        order = ([kind] if kind in ("quit_habit", "cut_expense", "change_sleep") else []) + [
            "quit_habit", "cut_expense", "change_sleep"]
        for k in dict.fromkeys(order):
            result = getattr(self, "_" + k)(query, entities)
            if result is not None:
                return result
        return self._unsupported()

    # ------------------------------------------------------------------
    # quit_habit
    # ------------------------------------------------------------------

    def _habits(self) -> dict:
        tracker = getattr(self.artemis, "tracker", None)
        if tracker is None:
            return {}
        try:
            return dict(tracker.get_habits())
        except Exception:
            logger.exception("what-if: reading Artemis habits failed.")
            return {}

    @staticmethod
    def _match_habit(query: str, habits: dict) -> Optional[str]:
        q = query.lower()
        q_words = _words(query)
        best, best_score = None, 0
        for name in habits:
            n = name.lower().strip()
            if not n:
                continue
            if re.search(r"\b" + re.escape(n) + r"\b", q):
                score = 100 + len(n)
            else:
                nw = _words(n)
                score = len(nw & q_words) * 10 if nw and nw <= q_words else 0
            if score > best_score:
                best, best_score = name, score
        return best

    def _quit_habit(self, query: str, entities: dict) -> Optional[dict]:
        habits = self._habits()
        if not habits:
            return None
        explicit = str(entities.get("habit") or "").strip()
        name = explicit if explicit in habits else self._match_habit(query, habits)
        if name is None or not (_QUIT_WORDS.search(query) or explicit):
            return None
        habit = habits[name]
        today = self._today()
        done, possible = 0, 0
        try:
            done, possible = habit.window_stats(today - timedelta(days=29), today, today)
        except Exception:
            logger.debug("what-if: window_stats unavailable for %r", name, exc_info=True)

        streak = int(getattr(habit, "streak", 0) or 0)
        best = int(getattr(habit, "best_streak", 0) or 0)
        total = int(getattr(habit, "total_completions", 0) or 0)
        facts: list[str] = []
        if streak > 0:
            facts.append(
                f"your {streak}-day '{name}' streak would reset to zero"
                + (f" (your best is {best})" if best > streak else " (it's your best so far)")
            )
        else:
            facts.append(f"'{name}' has no live streak right now, so there's no streak to lose")
        rate = None
        if possible:
            rate = done / possible
            per_week = rate * 7
            facts.append(
                f"in the last 30 days you did it {done} of {possible} days "
                f"({rate:.0%}, about {per_week:.1f} times a week), and that's what would stop"
            )
        elif total:
            facts.append(f"you've logged it {total} time(s) in all, but not enough history for a recent rate")

        linked: list[str] = []
        if _FITNESS.search(name) or _FITNESS.search(query):
            workouts = self._workouts_last_week()
            if workouts is not None:
                linked.append(
                    f"Apollo logged {workouts} workout(s) in the last 7 days; if this habit is how "
                    "those happen, that's what's at stake"
                )
        goal = self._related_goal(name)
        if goal:
            linked.append(goal)

        text = f"If you stopped '{name}': " + "; ".join(facts) + "."
        if linked:
            text += " Nearby: " + "; ".join(linked) + "."
        text += (" That's arithmetic on what you've logged, not a prediction of how you'd actually behave.")
        return {
            "response": text,
            "data": {
                "projected": True, "kind": "quit_habit", "habit": name,
                "streak": streak, "best_streak": best, "total_completions": total,
                "last_30_done": done, "last_30_possible": possible,
                "completion_rate": rate, "linked": linked,
            },
            "confidence": 0.85,
        }

    def _workouts_last_week(self) -> Optional[int]:
        db = getattr(self.apollo, "db", None)
        try:
            return int(db.workout_count(7)) if db is not None else None
        except Exception:
            return None

    def _related_goal(self, habit_name: str) -> Optional[str]:
        tracker = getattr(self.artemis, "tracker", None)
        if tracker is None:
            return None
        hw = _words(habit_name)
        try:
            for gname, goal in tracker.get_goals().items():
                if getattr(goal, "status", "active") != "active":
                    continue
                if hw & _words(gname):
                    return f"your goal '{gname}' is {round(goal.progress * 100)}% done"
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # cut_expense
    # ------------------------------------------------------------------

    def _recurring(self) -> list[dict]:
        if self.pluto is None:
            return []
        try:
            result = self.pluto.handle("recurring_expenses", {}, {})
            rows = (result or {}).get("data", {}).get("recurring") or []
            return [r for r in rows if r.get("active")]
        except Exception:
            logger.exception("what-if: reading Pluto recurring charges failed.")
            return []

    @staticmethod
    def _match_recurring(query: str, rows: list[dict]) -> Optional[dict]:
        q_words = _words(query)
        best, best_hits = None, 0
        for r in rows:
            rw = _words(r.get("key") or r.get("description") or "")
            hits = len(rw & q_words)
            if rw and hits and hits >= min(len(rw), 2) and hits > best_hits:
                best, best_hits = r, hits
        return best

    def _fmt(self) -> Optional[Callable[[float], str]]:
        planner = getattr(self.pluto, "planner", None)
        return getattr(planner, "_m", None)

    def _cut_expense(self, query: str, entities: dict) -> Optional[dict]:
        if not _CUT_WORDS.search(query):
            return None
        rows = self._recurring()
        item = self._match_recurring(query, rows)
        stated = None
        if entities.get("amount") not in (None, ""):
            try:
                stated = float(str(entities["amount"]).replace(",", ""))
            except ValueError:
                stated = None
        if stated is None and item is None:
            stated = parse_amount_per_month(query)
        if item is None and stated is None:
            return None

        fmt = self._fmt()
        if item is not None:
            monthly = float(item["monthly_cost"])
            label = item["description"]
            basis = (f"'{label}' costs {_money(float(item['typical_amount']), fmt)} "
                     f"{item['cadence']}, about {_money(monthly, fmt)} a month")
        else:
            monthly = float(stated)
            label = "that spending"
            basis = f"you said about {_money(monthly, fmt)} a month"

        parts = [
            f"If you cut {label}: {basis}.",
            f"That frees up roughly {_money(monthly * 3, fmt)} over 3 months, "
            f"{_money(monthly * 6, fmt)} over 6 and {_money(monthly * 12, fmt)} over a year.",
        ]
        data: dict[str, Any] = {
            "projected": True, "kind": "cut_expense", "item": label,
            "monthly_saving": round(monthly, 2),
            "saving_3m": round(monthly * 3, 2), "saving_6m": round(monthly * 6, 2),
            "saving_12m": round(monthly * 12, 2), "matched_recurring": item is not None,
        }

        sts = self._safe_to_spend()
        if sts is not None:
            budget = float(sts.get("budget") or 0)
            if budget > 0:
                share = monthly / budget
                parts.append(f"That's {share:.0%} of your monthly budget of {_money(budget, fmt)}.")
                data["budget_share"] = round(share, 4)
            if item is not None and self._still_due_this_month(item):
                days_left = max(1, int(sts.get("days_left") or 1))
                amount = float(item["typical_amount"])
                new_remaining = float(sts["remaining"]) + amount
                before = float(sts.get("per_day") or 0.0)
                after = max(0.0, new_remaining) / days_left
                parts.append(
                    f"It's still due this month, so skipping it would lift today's safe-to-spend "
                    f"from {_money(before, fmt)} to {_money(after, fmt)} a day "
                    f"for the next {days_left} day(s)."
                )
                data.update({"safe_per_day_before": round(before, 2),
                             "safe_per_day_after": round(after, 2), "days_left": days_left})
            elif item is not None:
                parts.append("It isn't due again before month end, so this month's numbers wouldn't move.")
        return {"response": " ".join(parts), "data": data, "confidence": 0.85}

    def _safe_to_spend(self) -> Optional[dict]:
        if self.pluto is None:
            return None
        try:
            res = self.pluto.handle("safe_to_spend", {}, {}) or {}
            data = res.get("data") or {}
            return data if "remaining" in data else None
        except Exception:
            logger.exception("what-if: reading safe-to-spend failed.")
            return None

    def _still_due_this_month(self, item: dict) -> bool:
        """Same rule Pluto's safe-to-spend uses: the next charge falls after
        today and before the month ends."""
        try:
            due = date.fromisoformat(str(item.get("next_expected")))
        except ValueError:
            return False
        today = self._today()
        nxt = date(today.year + (today.month == 12), today.month % 12 + 1, 1)
        return today < due < nxt

    # ------------------------------------------------------------------
    # change_sleep
    # ------------------------------------------------------------------

    def _change_sleep(self, query: str, entities: dict) -> Optional[dict]:
        change = parse_sleep_change(query)
        if change is None:
            return None
        db = getattr(self.apollo, "db", None)
        try:
            avg = db.avg_sleep(7) if db is not None else None
        except Exception:
            avg = None
        if avg is None:
            return {
                "response": ("I can't project a sleep change because I don't have any sleep logged "
                             "in the last week. Log a few nights and ask again."),
                "data": {"projected": False, "kind": "change_sleep"},
                "confidence": 0.5,
            }
        avg = float(avg)
        new = float(change["target"]) if "target" in change else avg + float(change["delta"])
        new = max(0.0, new)
        diff = new - avg
        weekly = diff * 7
        if abs(diff) < 0.05:
            return {
                "response": f"That's about what you already get: your 7-night average is {avg:.1f}h.",
                "data": {"projected": True, "kind": "change_sleep", "current_avg": avg, "new_avg": new},
                "confidence": 0.8,
            }
        direction = "more" if diff > 0 else "less"
        text = (
            f"Your 7-night sleep average is {avg:.1f}h. Sleeping {abs(diff):.1f}h {direction} "
            f"a night would make it {new:.1f}h, which is {abs(weekly):.1f}h {direction} over a week."
        )
        crosses = avg >= LOW_SLEEP_HOURS > new
        if crosses:
            text += (f" That drops you below {LOW_SLEEP_HOURS:.0f}h, the line where my burnout "
                     "check starts counting sleep as a reason to rest.")
        elif new < LOW_SLEEP_HOURS:
            text += (f" You'd be under {LOW_SLEEP_HOURS:.0f}h, which my burnout check already "
                     "treats as a rest signal.")
        elif avg < LOW_SLEEP_HOURS <= new:
            text += f" That would bring you back above the {LOW_SLEEP_HOURS:.0f}h line."
        else:
            text += f" You'd stay above the {LOW_SLEEP_HOURS:.0f}h line."
        return {
            "response": text,
            "data": {"projected": True, "kind": "change_sleep", "current_avg": avg,
                     "new_avg": round(new, 2), "weekly_change_hours": round(weekly, 2),
                     "crosses_low_sleep_line": crosses},
            "confidence": 0.85,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _unsupported() -> dict:
        examples = "; ".join(f"'{e}'" for e in SUPPORTED_EXAMPLES)
        return {
            "response": (
                "I can't project that from my data. I can work out the effect of cutting "
                f"a recurring charge, dropping a habit I track, or changing your sleep, "
                f"for example: {examples}."
            ),
            "data": {"projected": False, "supported": list(SUPPORTED_EXAMPLES)},
            "confidence": 0.4,
        }
