"""
modules/pluto/planning.py

Money-planning features built only on what Pluto already stores (the local
SQLite ``expenses`` and ``investments`` tables):

  #133  recurring-expense (subscription) detection
  #134  per-category budgets with variance alerts
  #266  "safe to spend today"
  #135  financial health score (savings rate, volatility, diversification)
  #138  scenario planning ("what if I invest X a month for Y years")
  #137  tax-year CSV export

Everything numeric is a pure function that takes its inputs (and "today") as
arguments, so it is tested without a database, a clock or the network. The
``PlanningManager`` at the bottom is the thin layer that reads the database,
calls those functions, and turns the result into a reply.

Honest limits, repeated in the replies where they matter:
  * Dates come from SQLite's ``CURRENT_TIMESTAMP``, which is UTC, so an expense
    logged in the first hours of a month in India can land in the previous one.
  * Recurring detection needs three or more charges of a similar size at a
    regular gap; two charges is not a pattern.
  * The health score is a rule of thumb built from three simple measures, not
    financial advice, and it says which measures it could not compute.
  * Scenario results assume a constant return. Real returns vary year to year.
  * The tax export lists transactions with a hint where a category often
    matters for Indian tax filing; it does not decide what is deductible.
"""
from __future__ import annotations

import csv
import math
import re
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .logging_config import get_logger

logger = get_logger(__name__)

MAX_AMOUNT = 1_000_000_000.0          # one entry or one budget; larger is a typo
WARN_FRACTION = 0.8                   # budget warning threshold
MIN_RECURRING_OCCURRENCES = 3
QUIET_HOURS = (22, 7)                 # no spoken alerts between 22:00 and 07:00

_MONTH_WORDS = {
    "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
    "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
    "oct", "october", "nov", "november", "dec", "december",
}

# ---------------------------------------------------------------------------
# Amounts
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
_SIGNED_NUM_RE = re.compile(r"([-\u2212]?\s?\d+(?:\.\d+)?)")
_MULTIPLIERS = (
    (re.compile(r"\b(?:crore|crores|cr)\b"), 10_000_000.0),
    (re.compile(r"\b(?:lakh|lakhs|lac|lacs)\b"), 100_000.0),
    (re.compile(r"\b(?:k|thousand)\b"), 1_000.0),
)


def parse_amount(raw: Any) -> Optional[float]:
    """A positive, finite amount from whatever the NLU handed over.

    Accepts numbers and strings like ``"₹1,500"``, ``"1500 rupees"``, ``"5k"``,
    ``"2.5 lakh"``. Returns ``None`` for zero, negatives, anything non-finite,
    bools, or anything above ``MAX_AMOUNT``. Never raises.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        text = str(raw).strip().lower().replace(",", "")
        if re.match(r"^\s*[-\u2212(]", text):          # "-500", "(500)"
            return None
        if re.search(r"\d\s*e\s*[-+]?\d", text):      # "1e999": reading it as 1 would be wrong
            return None
        match = _NUM_RE.search(text)
        if not match:
            return None
        try:
            value = float(match.group(1))
        except ValueError:
            return None
        rest = text[match.end():]
        for pattern, factor in _MULTIPLIERS:
            if pattern.match(rest.strip()):
                value *= factor
                break
    if not math.isfinite(value) or value <= 0 or value > MAX_AMOUNT:
        return None
    return value


def money(value: float, symbol: str = "₹") -> str:
    """Whole-unit money for spoken replies: ``₹12,345``."""
    return f"{symbol}{value:,.0f}"


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def parse_day(value: Any) -> Optional[date]:
    """``YYYY-MM-DD`` (optionally followed by a time) -> date, else None."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    m = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", str(value or ""))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def month_bounds(today: date) -> tuple[date, date]:
    """First day of *today*'s month and first day of the next."""
    start = today.replace(day=1)
    nxt = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
    return start, nxt


def days_left_in_month(today: date) -> int:
    """Days remaining in the month including *today* (never below 1)."""
    _, nxt = month_bounds(today)
    return max(1, (nxt - today).days)


def financial_year_bounds(start_year: int) -> tuple[date, date]:
    """Indian financial year starting April of *start_year*: (first day, first day after)."""
    return date(start_year, 4, 1), date(start_year + 1, 4, 1)


def current_financial_year_start(today: date) -> int:
    return today.year if today.month >= 4 else today.year - 1


def parse_financial_year(raw: Any, today: date) -> Optional[int]:
    """"2025-26", "2025/26", "FY2025", "2025" -> 2025; empty -> the current year; junk -> None."""
    if raw is None or str(raw).strip() == "":
        return current_financial_year_start(today)
    m = re.search(r"(20\d{2})", str(raw))
    if not m:
        return None
    # "2025-26" names the year that STARTS in 2025; the first year is taken at face value.
    year = int(m.group(1))
    return year if 1990 <= year <= today.year + 1 else None


# ---------------------------------------------------------------------------
# Recurring expenses (#133)
# ---------------------------------------------------------------------------

_CADENCES = (   # (name, low gap, high gap, charges per month)
    ("weekly", 6, 8, 52 / 12),
    ("fortnightly", 13, 16, 26 / 12),
    ("monthly", 27, 33, 1.0),
    ("quarterly", 85, 95, 1 / 3),
    ("yearly", 360, 370, 1 / 12),
)


def normalise_description(text: str) -> str:
    """Key for "the same payee": lower-case letters only, no months or years."""
    words = re.findall(r"[a-z]+", (text or "").lower())
    return " ".join(w for w in words if w not in _MONTH_WORDS)


_CALENDAR_MONTHS = {"monthly": 1, "quarterly": 3, "yearly": 12}


def add_months(d: date, months: int) -> date:
    """*d* plus whole calendar months, clamped to the target month's last day (Jan 31 + 1 -> Feb 28)."""
    index = d.year * 12 + (d.month - 1) + months
    year, month = divmod(index, 12)
    month += 1
    last = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)).day
    return date(year, month, min(d.day, last))


def _cadence_for(gap: float) -> Optional[tuple[str, float]]:
    for name, lo, hi, per_month in _CADENCES:
        if lo <= gap <= hi:
            return name, per_month
    return None


def detect_recurring(expenses: Iterable[dict], today: date) -> list[dict]:
    """Subscriptions and other regular charges found in *expenses*.

    A charge is recurring when the same normalised description appears at least
    ``MIN_RECURRING_OCCURRENCES`` times on different days, the gaps between
    them are all close to one standard cadence, and the amounts are within 25%
    of their median. Each result carries ``active`` (the next charge is not yet
    a full cycle overdue) and ``price_changed`` (the latest amount differs from
    the typical one by more than 5%). Newest-active first, then by monthly cost.
    """
    groups: dict[str, list[tuple[date, float, str]]] = defaultdict(list)
    for row in expenses:
        d = parse_day(row.get("logged_at"))
        try:
            amt = float(row.get("amount"))
        except (TypeError, ValueError):
            continue
        key = normalise_description(str(row.get("description", "")))
        if d is None or not key or not math.isfinite(amt) or amt <= 0:
            continue
        groups[key].append((d, amt, str(row.get("description", ""))))

    found: list[dict] = []
    for key, items in groups.items():
        items.sort(key=lambda x: x[0])
        by_day: dict[date, tuple[float, str]] = {}
        for d, amt, desc in items:                       # one charge per day
            by_day[d] = (amt, desc)
        days = sorted(by_day)
        if len(days) < MIN_RECURRING_OCCURRENCES:
            continue
        gaps = [(b - a).days for a, b in zip(days, days[1:])]
        median_gap = statistics.median(gaps)
        cadence = _cadence_for(median_gap)
        if cadence is None:
            continue
        tolerance = max(3.0, 0.15 * median_gap)
        if any(abs(g - median_gap) > tolerance for g in gaps):
            continue
        amounts = [by_day[d][0] for d in days]
        typical = statistics.median(amounts)
        if any(abs(a - typical) > 0.25 * typical for a in amounts):
            continue
        name, per_month = cadence
        last_day = days[-1]
        # Monthly-ish bills fall on the same day of the month, not every 30.4 days.
        step = _CALENDAR_MONTHS.get(name)
        next_expected = (add_months(last_day, step) if step
                         else last_day + timedelta(days=round(median_gap)))
        overdue_after = next_expected + timedelta(days=max(7, round(median_gap)))
        latest = amounts[-1]
        found.append({
            "description": by_day[last_day][1].strip() or key,
            "key": key,
            "cadence": name,
            "typical_amount": round(typical, 2),
            "latest_amount": round(latest, 2),
            "count": len(days),
            "last_date": last_day.isoformat(),
            "next_expected": next_expected.isoformat(),
            "monthly_cost": round(typical * per_month, 2),
            "active": today <= overdue_after,
            "price_changed": abs(latest - typical) > 0.05 * typical,
            "category": None,
        })
    found.sort(key=lambda r: (not r["active"], -r["monthly_cost"], r["key"]))
    return found


def upcoming_recurring_total(recurring: Iterable[dict], today: date,
                             only_categories: Optional[set[str]] = None) -> float:
    """Recurring charges expected after *today* and before the month ends."""
    _, nxt = month_bounds(today)
    total = 0.0
    for r in recurring:
        if not r.get("active"):
            continue
        if only_categories is not None and (r.get("category") or "").lower() not in only_categories:
            continue
        due = parse_day(r.get("next_expected"))
        if due and today < due < nxt:
            total += float(r["typical_amount"])
    return total


# ---------------------------------------------------------------------------
# Budgets (#134) and safe to spend (#266)
# ---------------------------------------------------------------------------

def budget_status(budgets: dict[str, float], spent: dict[str, float]) -> list[dict]:
    """One row per budget: spent, percent used, remaining and a state.

    ``state`` is ``ok`` below 80%, ``warn`` from 80% up to the limit, and
    ``over`` once spending exceeds it. Keys are compared case-insensitively;
    the budget's own spelling is what is shown. Worst first.
    """
    spent_l = {k.lower(): v for k, v in spent.items()}
    rows = []
    for cat, limit in budgets.items():
        s = float(spent_l.get(cat.lower(), 0.0))
        pct = (s / limit * 100.0) if limit > 0 else 0.0
        state = "over" if s > limit else "warn" if s >= limit * WARN_FRACTION else "ok"
        rows.append({"category": cat, "limit": limit, "spent": s, "percent": round(pct, 1),
                     "remaining": limit - s, "state": state})
    order = {"over": 0, "warn": 1, "ok": 2}
    rows.sort(key=lambda r: (order[r["state"]], -r["percent"], r["category"].lower()))
    return rows


def safe_to_spend(budget: float, spent: float, upcoming: float, today: date) -> dict:
    """What is left per day, after money already spent and recurring bills still to come."""
    left_days = days_left_in_month(today)
    remaining = budget - spent - upcoming
    return {
        "budget": budget, "spent": spent, "upcoming_recurring": upcoming,
        "remaining": remaining, "days_left": left_days,
        "per_day": max(0.0, remaining) / left_days,
        "overspent": remaining < 0,
    }


# ---------------------------------------------------------------------------
# Financial health score (#135)
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def savings_rate_score(income: float, avg_monthly_spend: float) -> Optional[dict]:
    if income <= 0:
        return None
    rate = (income - avg_monthly_spend) / income
    return {"name": "savings rate", "value": rate, "score": round(_clamp(rate / 0.20) * 100),
            "detail": f"saving {rate:.0%} of income (20% or more scores full marks)"}


def volatility_score(weekly_totals: list[float]) -> Optional[dict]:
    """Steadier week-to-week spending scores higher. Needs four or more weeks."""
    if len(weekly_totals) < 4:
        return None
    mean = statistics.fmean(weekly_totals)
    if mean <= 0:
        return None
    cv = statistics.pstdev(weekly_totals) / mean
    return {"name": "spending steadiness", "value": cv, "score": round((1 - _clamp(cv / 1.0)) * 100),
            "detail": f"weekly spending varies by {cv:.0%} of its average"}


def diversification_score(holding_values: list[float]) -> Optional[dict]:
    """Spread across holdings, by cost. One holding scores 0; five equal ones score 100."""
    values = [v for v in holding_values if v > 0 and math.isfinite(v)]
    if not values:
        return None
    total = sum(values)
    hhi = sum((v / total) ** 2 for v in values)
    score = round(_clamp((1 - hhi) / (1 - 1 / 5)) * 100)
    return {"name": "diversification", "value": hhi, "score": score,
            "detail": f"{len(values)} holding(s), largest is {max(values) / total:.0%} of cost"}


_HEALTH_WEIGHTS = {"savings rate": 0.4, "spending steadiness": 0.3, "diversification": 0.3}


def health_score(components: list[Optional[dict]]) -> dict:
    """Weighted average of whichever components could be computed.

    Returns ``{"score": None, ...}`` when fewer than two are available, because
    one number alone isn't a health score.
    """
    have = [c for c in components if c]
    missing = [n for n in _HEALTH_WEIGHTS if n not in {c["name"] for c in have}]
    if len(have) < 2:
        return {"score": None, "components": have, "missing": missing, "label": "not enough data"}
    wsum = sum(_HEALTH_WEIGHTS[c["name"]] for c in have)
    score = round(sum(c["score"] * _HEALTH_WEIGHTS[c["name"]] for c in have) / wsum)
    label = ("strong" if score >= 75 else "decent" if score >= 55
             else "needs attention" if score >= 35 else "weak")
    return {"score": score, "components": have, "missing": missing, "label": label}


def weekly_totals(expenses: Iterable[dict], today: date, weeks: int = 8) -> list[float]:
    """Spend per full 7-day block over the last *weeks* weeks, oldest first.

    Weeks before the first logged expense are dropped, so a new user is not
    scored on weeks they weren't tracking.
    """
    rows = []
    for e in expenses:
        d = parse_day(e.get("logged_at"))
        try:
            a = float(e.get("amount"))
        except (TypeError, ValueError):
            continue
        if d and math.isfinite(a) and a > 0:
            rows.append((d, a))
    if not rows:
        return []
    first = min(d for d, _ in rows)
    totals = []
    for w in range(weeks, 0, -1):
        end = today - timedelta(days=7 * (w - 1))          # exclusive
        start = end - timedelta(days=7)
        if start < first:
            continue
        totals.append(sum(a for d, a in rows if start <= d < end))
    return totals


def monthly_spend_average(expenses: Iterable[dict], today: date, months: int = 3) -> Optional[float]:
    """Average spend over the last *months* COMPLETE calendar months that have any spending."""
    per: dict[tuple[int, int], float] = defaultdict(float)
    for e in expenses:
        d = parse_day(e.get("logged_at"))
        try:
            a = float(e.get("amount"))
        except (TypeError, ValueError):
            continue
        if d and math.isfinite(a) and a > 0:
            per[(d.year, d.month)] += a
    cur = (today.year, today.month)
    complete = sorted((k for k in per if k < cur), reverse=True)[:months]
    if not complete:
        return None
    return statistics.fmean(per[k] for k in complete)


# ---------------------------------------------------------------------------
# Scenario planning (#138)
# ---------------------------------------------------------------------------

MAX_YEARS = 60


def scenario(monthly: float, years: int, annual_return_pct: float,
             initial: float = 0.0) -> dict:
    """Value of investing *monthly* for *years* at a constant annual return.

    Contributions are made at the start of each month and growth compounds
    monthly at the rate equivalent to the annual figure. Returns the final
    value, total contributed, the gain, and a year-by-year table. Raises
    ``ValueError`` for inputs outside sensible bounds.
    """
    if not (math.isfinite(monthly) and 0 <= monthly <= MAX_AMOUNT):
        raise ValueError("monthly amount out of range")
    if not (math.isfinite(initial) and 0 <= initial <= MAX_AMOUNT):
        raise ValueError("starting amount out of range")
    if not (isinstance(years, int) and 1 <= years <= MAX_YEARS):
        raise ValueError(f"years must be between 1 and {MAX_YEARS}")
    if not (math.isfinite(annual_return_pct) and -50 <= annual_return_pct <= 100):
        raise ValueError("return must be between -50% and 100% a year")
    if monthly == 0 and initial == 0:
        raise ValueError("nothing to invest")
    r = (1 + annual_return_pct / 100.0) ** (1 / 12) - 1
    value, contributed = initial, initial
    table = []
    for month in range(1, years * 12 + 1):
        value += monthly
        contributed += monthly
        value *= 1 + r
        if month % 12 == 0:
            table.append({"year": month // 12, "value": value, "contributed": contributed})
    return {"final": value, "contributed": contributed, "gain": value - contributed,
            "years": years, "annual_return_pct": annual_return_pct, "table": table}


# ---------------------------------------------------------------------------
# Tax export (#137)
# ---------------------------------------------------------------------------

_TAX_HINTS = {
    "health": "Medical insurance premiums or specified illnesses can matter for 80D/80DDB; bills alone usually don't.",
    "education": "Children's tuition fees can matter for 80C; other courses usually don't.",
}


def csv_safe(value: Any) -> str:
    """Neutralise spreadsheet formulas: a cell starting with = + - @ is prefixed with '."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def investment_hint(name: str, type_: str) -> str:
    blob = f"{name} {type_}".lower()
    if "elss" in blob:
        return "ELSS funds can matter for 80C."
    if any(w in blob for w in ("ppf", "nps", "epf", "tax saver")):
        return "Often relevant to 80C / 80CCD: check the scheme."
    return ""


def write_tax_csvs(directory: Path, fy_start: int, expenses: list[dict],
                   investments: list[dict]) -> list[Path]:
    """Write ``expenses_FY<yy-yy>.csv`` and ``investments_FY<yy-yy>.csv``."""
    label = f"FY{fy_start}-{str(fy_start + 1)[-2:]}"
    directory.mkdir(parents=True, exist_ok=True)
    exp_path = directory / f"expenses_{label}.csv"
    inv_path = directory / f"investments_{label}.csv"
    with open(exp_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "description", "category", "amount", "tax_hint"])
        for e in sorted(expenses, key=lambda r: str(r.get("logged_at"))):
            cat = str(e.get("category", ""))
            w.writerow([str(e.get("logged_at"))[:10], csv_safe(e.get("description")),
                        csv_safe(cat), f"{float(e.get('amount', 0)):.2f}",
                        _TAX_HINTS.get(cat.lower(), "")])
    with open(inv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "name", "type", "quantity", "buy_price", "cost", "tax_hint"])
        for i in sorted(investments, key=lambda r: str(r.get("logged_at"))):
            q = float(i.get("quantity") or 0)
            p = float(i.get("buy_price") or 0)
            w.writerow([str(i.get("logged_at"))[:10], csv_safe(i.get("name")), csv_safe(i.get("type")),
                        f"{q:g}", f"{p:.2f}", f"{q * p:.2f}",
                        investment_hint(str(i.get("name", "")), str(i.get("type", "")))])
    return [exp_path, inv_path]


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

def _reply(text: str, data: Optional[dict] = None, confidence: float = 0.9) -> dict:
    return {"response": text, "data": data or {}, "confidence": confidence}


class PlanningManager:
    """Reads Pluto's SQLite data, runs the pure functions above, and words the result."""

    def __init__(self, db: Any, currency: str = "₹", export_dir: Optional[Path] = None,
                 today_fn: Callable[[], date] = date.today,
                 now_fn: Callable[[], datetime] = datetime.now):
        self.db = db
        self.currency = currency
        self.export_dir = export_dir
        self._today = today_fn
        self._now = now_fn

    def _m(self, v: float) -> str:
        return money(v, self.currency)

    # -- helpers -------------------------------------------------------------

    def _canonical_category(self, raw: str) -> str:
        from .personal_finance import _CATEGORY_KEYWORDS
        key = (raw or "").strip().lower()
        for name, words in _CATEGORY_KEYWORDS.items():
            if key == name.lower() or key in words:
                return name
        if key in ("other", "misc", "miscellaneous"):
            return "Other"
        return " ".join(w.capitalize() for w in key.split())[:40]

    def _month_spend(self, today: date) -> dict[str, float]:
        start, nxt = month_bounds(today)
        rows = self.db.get_totals_between(start.isoformat(), nxt.isoformat())
        return {r["category"]: float(r["total"]) for r in rows}

    def _recurring(self, today: date) -> list[dict]:
        rows = self.db.get_expenses_between("1970-01-01", (today + timedelta(days=1)).isoformat())
        found = detect_recurring(rows, today)
        cat_by_key: dict[str, str] = {}
        for r in rows:
            cat_by_key[normalise_description(str(r.get("description", "")))] = str(r.get("category", ""))
        for f in found:
            f["category"] = cat_by_key.get(f["key"])
        return found

    # -- intents ---------------------------------------------------------------

    def set_budget(self, entities: dict) -> dict:
        raw_q = str(entities.get("raw_query") or "").lower()
        category_raw = str(entities.get("category") or entities.get("name") or "").strip()
        if not category_raw:
            return _reply("Which category is the budget for, and how much a month? "
                          "For example: set a food budget of 8000.", confidence=0.5)
        category = self._canonical_category(category_raw)
        amount = parse_amount(entities.get("amount"))
        if amount is None and any(w in raw_q for w in ("remove", "delete", "clear", "cancel")):
            removed = self.db.delete_budget(category)
            return _reply(f"Removed your {category} budget." if removed
                          else f"You don't have a {category} budget set.", {"category": category})
        if amount is None:
            return _reply(f"How much a month for {category}?", confidence=0.5)
        self.db.set_budget(category, amount)
        return _reply(f"Set your {category} budget to {self._m(amount)} a month.",
                      {"category": category, "monthly_limit": amount}, 0.95)

    def budget_status(self, entities: Optional[dict] = None) -> dict:
        today = self._today()
        budgets = self.db.get_budgets()
        if not budgets:
            return _reply("You haven't set any budgets yet. Try: set a food budget of 8000.", confidence=0.8)
        rows = budget_status(budgets, self._month_spend(today))
        want = self._canonical_category(str((entities or {}).get("category") or "")) if (entities or {}).get("category") else None
        if want:
            rows = [r for r in rows if r["category"].lower() == want.lower()] or rows
        lines = []
        for r in rows:
            if r["state"] == "over":
                tail = f"{self._m(-r['remaining'])} over"
            else:
                tail = f"{self._m(r['remaining'])} left"
            lines.append(f"{r['category']}: {self._m(r['spent'])} of {self._m(r['limit'])} "
                         f"({r['percent']:.0f}%), {tail}")
        worst = rows[0]
        head = {"over": f"You're over budget on {worst['category']}.",
                "warn": f"{worst['category']} is close to its limit.",
                "ok": "You're within every budget this month."}[worst["state"]]
        return _reply(head + "\n" + "\n".join(lines), {"budgets": rows}, 0.95)

    def recurring_expenses(self) -> dict:
        today = self._today()
        found = self._recurring(today)
        if not found:
            return _reply("I don't see any regular charges yet. I need at least three similar charges "
                          "at a steady gap to call something a subscription.", confidence=0.85)
        active = [f for f in found if f["active"]]
        lapsed = [f for f in found if not f["active"]]
        total = sum(f["monthly_cost"] for f in active)
        lines = []
        for f in active[:8]:
            note = f", latest was {self._m(f['latest_amount'])}" if f["price_changed"] else ""
            lines.append(f"{f['description']}: {self._m(f['typical_amount'])} {f['cadence']}{note}")
        text = (f"I found {len(active)} regular charge(s) costing about {self._m(total)} a month:\n"
                + "\n".join(lines))
        if len(active) > 8:
            text += f"\n...and {len(active) - 8} more."
        if lapsed:
            text += f"\n{len(lapsed)} older one(s) look like they've stopped."
        return _reply(text, {"recurring": found, "monthly_total": total}, 0.85)

    def safe_to_spend(self) -> dict:
        today = self._today()
        budgets = self.db.get_budgets()
        income = self._income()
        spent_by_cat = self._month_spend(today)
        recurring = self._recurring(today)
        if budgets:
            scope = {c.lower() for c in budgets}
            budget_total = sum(budgets.values())
            spent = sum(v for k, v in spent_by_cat.items() if k.lower() in scope)
            upcoming = upcoming_recurring_total(recurring, today, scope)
            basis = "your category budgets"
        elif income:
            budget_total = income
            spent = sum(spent_by_cat.values())
            upcoming = upcoming_recurring_total(recurring, today)
            basis = "your monthly income"
        else:
            return _reply("I need a budget to work that out. Set one (for example, a food budget of "
                          "8000) or tell me your monthly income.", confidence=0.7)
        r = safe_to_spend(budget_total, spent, upcoming, today)
        if r["overspent"]:
            text = (f"You're already past {basis} for this month, by {self._m(-r['remaining'])}. "
                    "Safe to spend today: nothing more.")
        else:
            text = (f"You can spend about {self._m(r['per_day'])} a day for the next {r['days_left']} day(s), "
                    f"based on {basis}: {self._m(r['remaining'])} left after {self._m(spent)} spent")
            text += (f" and {self._m(upcoming)} of regular bills still due." if upcoming else ".")
        return _reply(text, r, 0.9)

    def set_income(self, entities: dict) -> dict:
        amount = parse_amount(entities.get("amount"))
        if amount is None:
            return _reply("What's your monthly income?", confidence=0.5)
        self.db.set_setting("monthly_income", repr(float(amount)))
        return _reply(f"Noted: your monthly income is {self._m(amount)}.",
                      {"monthly_income": amount}, 0.95)

    def _income(self) -> Optional[float]:
        raw = self.db.get_setting("monthly_income")
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) and v > 0 else None

    def financial_health(self) -> dict:
        today = self._today()
        income = self._income()
        expenses = self.db.get_expenses_between("1970-01-01", (today + timedelta(days=1)).isoformat())
        avg = monthly_spend_average(expenses, today)
        comps = [
            savings_rate_score(income, avg) if (income and avg is not None) else None,
            volatility_score(weekly_totals(expenses, today)),
            diversification_score([float(i.get("quantity") or 0) * float(i.get("buy_price") or 0)
                                   for i in self.db.get_investments()]),
        ]
        result = health_score(comps)
        if result["score"] is None:
            why = []
            if not income:
                why.append("tell me your monthly income")
            why.append("keep logging expenses for a month or two")
            why.append("track an investment or two")
            return _reply("I can't score your finances yet. To get there: " + "; ".join(why) + ".",
                          result, 0.7)
        lines = [f"{c['name']}: {c['score']}/100, {c['detail']}" for c in result["components"]]
        text = f"Your financial health score is {result['score']} out of 100, which is {result['label']}.\n" + "\n".join(lines)
        if result["missing"]:
            text += "\nNot counted (no data yet): " + ", ".join(result["missing"]) + "."
        text += "\nThis is a rule of thumb, not financial advice."
        return _reply(text, result, 0.85)

    def scenario_plan(self, entities: dict) -> dict:
        monthly = parse_amount(entities.get("amount") or entities.get("monthly"))
        initial = parse_amount(entities.get("initial")) or 0.0
        if monthly is None and not initial:
            return _reply("How much would you invest each month, for how many years, and at what yearly "
                          "return? For example: invest 10000 a month for 15 years at 12 percent.", confidence=0.5)
        ym = _SIGNED_NUM_RE.search(str(entities.get("years") if entities.get("years") is not None else ""))
        years = int(float(ym.group(1).replace("\u2212", "-").replace(" ", ""))) if ym else 0
        if years <= 0:
            return _reply("For how many years?", confidence=0.5)
        rate_raw = entities.get("rate") if entities.get("rate") is not None else entities.get("return")
        # A return can be negative ("-5"), so unlike amounts this keeps the sign.
        m = _SIGNED_NUM_RE.search(str(rate_raw)) if rate_raw is not None else None
        if not m:
            return _reply("What yearly return should I assume? 12 percent is a common long-run equity "
                          "assumption, though nothing is guaranteed.", confidence=0.5)
        rate = float(m.group(1).replace("\u2212", "-").replace(" ", ""))
        try:
            base = scenario(monthly or 0.0, years, rate, initial)
            low = scenario(monthly or 0.0, years, max(-50.0, rate - 3), initial)
            high = scenario(monthly or 0.0, years, min(100.0, rate + 3), initial)
        except ValueError as exc:
            return _reply(f"I can't run that: {exc}.", confidence=0.4)
        text = (f"Investing {self._m(monthly or 0)} a month"
                + (f" from a {self._m(initial)} start" if initial else "")
                + f" for {years} year(s) at {rate:g}% a year grows to about {self._m(base['final'])}. "
                f"You'd put in {self._m(base['contributed'])}, so the gain is about {self._m(base['gain'])}. "
                f"At {max(-50.0, rate - 3):g}% it's {self._m(low['final'])}; at {min(100.0, rate + 3):g}% "
                f"it's {self._m(high['final'])}. Real returns vary year to year, so treat this as an illustration.")
        return _reply(text, {"base": base, "low": low["final"], "high": high["final"]}, 0.9)

    def export_tax(self, entities: dict) -> dict:
        today = self._today()
        fy = parse_financial_year(entities.get("year"), today)
        if fy is None:
            return _reply("Which financial year? For example: 2025-26.", confidence=0.5)
        if self.export_dir is None:
            return _reply("I don't have a folder to save exports in.", confidence=0.0)
        start, end = financial_year_bounds(fy)
        expenses = self.db.get_expenses_between(start.isoformat(), end.isoformat())
        invs = [i for i in self.db.get_investments()
                if (d := parse_day(i.get("logged_at"))) and start <= d < end]
        if not expenses and not invs:
            return _reply(f"Nothing logged for the financial year starting April {fy}.", confidence=0.8)
        paths = write_tax_csvs(self.export_dir, fy, expenses, invs)
        text = (f"Exported {len(expenses)} expense(s) and {len(invs)} investment(s) for "
                f"April {fy} to March {fy + 1} to {paths[0].parent}. "
                "The tax_hint column only points out where a category often matters; "
                "it doesn't decide what you can claim.")
        return _reply(text, {"files": [str(p) for p in paths]}, 0.9)

    # -- heartbeat -------------------------------------------------------------

    def check_budget_alerts(self) -> Optional[str]:
        """Text to speak when a budget newly crosses 80% or its limit this month, else None.

        Each (month, category, state) is announced once: it is recorded when
        returned, so the next tick stays quiet. During quiet hours nothing is
        returned and nothing is recorded, so the alert comes out later.
        """
        now = self._now()
        lo, hi = QUIET_HOURS
        if now.hour >= lo or now.hour < hi:
            return None
        today = now.date()
        budgets = self.db.get_budgets()
        if not budgets:
            return None
        month_key = today.strftime("%Y-%m")
        new = []
        for r in budget_status(budgets, self._month_spend(today)):
            if r["state"] == "ok":
                continue
            key = f"budget:{month_key}:{r['category'].lower()}:{r['state']}"
            if self.db.alert_already_sent(key):
                continue
            if r["state"] == "warn" and self.db.alert_already_sent(
                    f"budget:{month_key}:{r['category'].lower()}:over"):
                continue
            new.append((key, r))
        if not new:
            return None
        for key, _ in new:
            self.db.mark_alert_sent(key)
        parts = []
        for _, r in new:
            if r["state"] == "over":
                parts.append(f"{r['category']} is over budget by {self._m(-r['remaining'])}")
            else:
                parts.append(f"{r['category']} has used {r['percent']:.0f}% of its {self._m(r['limit'])} budget")
        return "Heads up: " + "; ".join(parts) + "."
