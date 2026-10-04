"""
modules/pluto/rebalance.py

Rebalancing suggestions from drift against a target allocation (backlog #142).

Two ideas that sound alike and are not:

* ``optimize_portfolio`` (portfolio.py) asks "what mix would have had the best
  risk-adjusted return?" from past prices, and it picks the targets for you.
* This module takes targets YOU set ("60% stocks, 30% mutual funds, 10%
  crypto"), measures how far your holdings have drifted from them, and says
  what to move to get back. No market view is involved.

Targets are kept either by asset class (stock / mutual_fund / crypto, the
``type`` Pluto already assigns when you track an investment) or by individual
holding, never a mix of both, because a holding would be counted twice.
They must add up to 100 and must cover everything you hold: a held class with
no target is an error, not silently treated as "sell it all".

Valuation: live price x quantity where a live price can be fetched, otherwise
cost (quantity x buy price), and the reply says how many holdings were valued
at cost. All prices are assumed to be in one currency (Pluto's), as in the
rest of the module; mixed-currency portfolios are not handled (#136).

Two ways to get back to target:

* rebalance: sell what is over, buy what is under, total unchanged.
* add new money: ``contribution`` is spread over the under-weight areas only,
  so nothing is sold. It will not fully fix a large drift, and says how far it got.

Nothing here knows about tax, fees or minimum lot sizes. The reply says so.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Optional

from .logging_config import get_logger
from .market_data import normalise_ticker
from .planning import parse_amount

logger = get_logger(__name__)

SETTING_KEY = "target_allocation"
DEFAULT_THRESHOLD_PTS = 5.0
MIN_THRESHOLD_PTS, MAX_THRESHOLD_PTS = 0.5, 50.0
MAX_TARGETS = 20
SUM_TOLERANCE = 0.5            # targets may add up to 100 +/- this, then are scaled to exactly 100
MIN_MOVE_FRACTION = 0.005      # ignore moves smaller than 0.5% of the portfolio

ASSET_CLASSES = {
    "stock": "stock", "stocks": "stock", "equity": "stock", "equities": "stock",
    "shares": "stock", "share": "stock",
    "mutual fund": "mutual_fund", "mutual funds": "mutual_fund", "mutual_fund": "mutual_fund",
    "mutual_funds": "mutual_fund", "fund": "mutual_fund", "funds": "mutual_fund",
    "mf": "mutual_fund", "index fund": "mutual_fund", "index funds": "mutual_fund",
    "crypto": "crypto", "cryptocurrency": "crypto", "cryptocurrencies": "crypto",
    "coins": "crypto",
}

_FILLER = {
    "set", "my", "the", "target", "targets", "allocation", "allocations", "to", "as", "be",
    "portfolio", "please", "update", "change", "make", "it", "is", "should", "want", "i", "a",
    "in", "into", "for", "of", "at", "percent", "pct", "and", "with", "keep", "split", "ratio",
}
_CLEAR_WORDS = re.compile(r"\b(clear|remove|delete|reset|forget|drop)\b", re.I)
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])\s*(?:%|percent|pct)?", re.I)


class RebalanceError(ValueError):
    """A problem with the targets or holdings, phrased for the user."""


def _label(text: str) -> str:
    words = [w for w in re.split(r"[\s:=,;]+", text.strip()) if w and w.lower().strip("%") not in _FILLER]
    return " ".join(words).strip(" -_.")


def parse_targets(raw: Any) -> dict[str, float]:
    """``"60 stocks, 30 mutual funds, 10 crypto"`` or a dict -> {label: percent}.

    Labels that name an asset class are normalised to ``stock`` / ``mutual_fund``
    / ``crypto``; anything else is kept as a holding name. Raises RebalanceError.
    """
    pairs: list[tuple[str, float]] = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                pairs.append((str(k), float(str(v).replace("%", "").strip())))
            except (TypeError, ValueError):
                raise RebalanceError(f"I couldn't read {v!r} as a percentage for {k!r}.")
    elif isinstance(raw, str) and raw.strip():
        for chunk in re.split(r"[,;\n]", raw):
            nums = list(_NUMBER.finditer(chunk))
            if not nums:
                if _label(chunk):
                    raise RebalanceError(f"There's no percentage next to {_label(chunk)!r}.")
                continue
            before = _label(chunk[: nums[0].start()])
            after = _label(chunk[nums[-1].end():])
            # "stocks 60 funds 40" (name first) vs "60 stocks 40 funds" (number first).
            # One number: use whichever side has a name, the one after it if both do.
            label_first = bool(before) and (len(nums) > 1 or not after)
            for k, m in enumerate(nums):
                if label_first:
                    start = nums[k - 1].end() if k else 0
                    text = chunk[start: m.start()]
                else:
                    end = nums[k + 1].start() if k + 1 < len(nums) else len(chunk)
                    text = chunk[m.end(): end]
                label = _label(text)
                if not label:
                    raise RebalanceError("Each percentage needs a name next to it, e.g. '60 stocks'.")
                pairs.append((label, float(m.group(1))))
    else:
        raise RebalanceError("Tell me the split, e.g. '60 stocks, 30 mutual funds, 10 crypto'.")

    if not pairs:
        raise RebalanceError("Tell me the split, e.g. '60 stocks, 30 mutual funds, 10 crypto'.")
    if len(pairs) > MAX_TARGETS:
        raise RebalanceError(f"That's {len(pairs)} targets; the limit is {MAX_TARGETS}.")

    targets: dict[str, float] = {}
    for label, pct in pairs:
        if not math.isfinite(pct) or pct < 0 or pct > 100:
            raise RebalanceError(f"{label}: a target has to be between 0 and 100 percent.")
        key = ASSET_CLASSES.get(label.lower(), label)
        if key in targets:
            raise RebalanceError(f"{label!r} appears twice in the targets.")
        targets[key] = pct

    total = sum(targets.values())
    if abs(total - 100.0) > SUM_TOLERANCE:
        raise RebalanceError(f"Those targets add up to {total:g}%, not 100%.")
    if total != 100.0:
        targets = {k: v * 100.0 / total for k, v in targets.items()}
    return targets


def target_mode(targets: dict[str, float]) -> str:
    """'type' when every label is an asset class, 'holding' when none is."""
    classes = {k for k in targets if k in ("stock", "mutual_fund", "crypto")}
    if len(classes) == len(targets):
        return "type"
    if not classes:
        return "holding"
    raise RebalanceError(
        "Targets mix asset classes (" + ", ".join(sorted(classes)) + ") with individual holdings; "
        "use one or the other, otherwise a holding is counted twice."
    )


class RebalanceAdvisor:
    def __init__(self, db: Any = None, currency: str = "\u20b9",
                 price_fn: Optional[Callable[[str, str], Optional[float]]] = None):
        self.db = db
        self.currency = currency
        self.price_fn = price_fn

    # ---- targets --------------------------------------------------------

    def get_targets(self) -> Optional[dict]:
        if self.db is None:
            return None
        raw = self.db.get_setting(SETTING_KEY)
        if not raw:
            return None
        try:
            stored = json.loads(raw)
            return {"mode": stored["mode"], "targets": {k: float(v) for k, v in stored["targets"].items()}}
        except (ValueError, KeyError, TypeError, AttributeError):
            logger.warning("rebalance: stored target allocation is unreadable; ignoring it")
            return None

    def set_targets(self, entities: dict) -> dict:
        if self.db is None:
            return _err("I can't reach the finance database right now.")
        raw = entities.get("targets") or entities.get("allocation") or entities.get("raw_query") or ""
        if isinstance(raw, str) and _CLEAR_WORDS.search(raw) and not _NUMBER.search(raw):
            existed = self.get_targets() is not None
            self.db.delete_setting(SETTING_KEY)
            return _ok("Target allocation cleared." if existed else "You had no target allocation set.")
        if (not raw) or (isinstance(raw, str) and not _NUMBER.search(raw)):
            current = self.get_targets()
            if current:
                return _ok("Your target allocation:\n" + _targets_text(current["targets"]),
                           data=current)
            return _ok("You haven't set a target allocation yet. Tell me the split, e.g. "
                       "'set my target allocation to 60 stocks, 30 mutual funds, 10 crypto'.",
                       confidence=0.6)
        try:
            targets = parse_targets(raw)
            mode = target_mode(targets)
        except RebalanceError as e:
            return _err(str(e))
        self.db.set_setting(SETTING_KEY, json.dumps({"mode": mode, "targets": targets}))
        kind = "by asset class" if mode == "type" else "by holding"
        return _ok(f"Target allocation saved ({kind}):\n{_targets_text(targets)}\n"
                   "Say 'rebalance my portfolio' to see how far you've drifted.",
                   data={"mode": mode, "targets": targets})

    # ---- holdings -------------------------------------------------------

    def _holdings(self) -> tuple[list[dict], list[str]]:
        """Aggregate rows by name -> ([{name,type,qty,cost,price,value,at_cost}], names without a quantity)."""
        agg: dict[str, dict] = {}
        for inv in self.db.get_investments():
            name = (inv.get("name") or "").strip()
            if not name:
                continue
            h = agg.setdefault(name.lower(), {"name": name, "type": inv.get("type") or "stock",
                                              "qty": 0.0, "cost": 0.0})
            qty = float(inv.get("quantity") or 0.0)
            h["qty"] += max(qty, 0.0)
            h["cost"] += max(qty, 0.0) * float(inv.get("buy_price") or 0.0)
        holdings, no_qty = [], []
        for h in agg.values():
            if h["qty"] <= 0:
                no_qty.append(h["name"])
                continue
            price = None
            if self.price_fn is not None:
                try:
                    price = self.price_fn(h["name"], h["type"])
                except Exception as e:
                    logger.warning("rebalance: live price failed for %r: %s", h["name"], e)
            if price is not None and not (isinstance(price, (int, float)) and math.isfinite(price) and price > 0):
                price = None
            h["price"] = price
            h["at_cost"] = price is None
            h["value"] = h["qty"] * price if price is not None else h["cost"]
            holdings.append(h)
        return holdings, no_qty

    # ---- the check ------------------------------------------------------

    def suggest(self, entities: dict) -> dict:
        if self.db is None:
            return _err("I can't reach the finance database right now.")
        stored = self.get_targets()
        if not stored:
            return _err("Set your target split first, e.g. 'set my target allocation to 60 stocks, "
                        "30 mutual funds, 10 crypto'.")
        targets, mode = stored["targets"], stored["mode"]

        threshold = DEFAULT_THRESHOLD_PTS
        raw_threshold = entities.get("threshold")
        if raw_threshold not in (None, ""):
            try:
                threshold = float(str(raw_threshold).replace("%", "").strip())
            except ValueError:
                return _err(f"I couldn't read {raw_threshold!r} as a number of percentage points.")
            if not math.isfinite(threshold):
                return _err("The drift threshold must be a real number of percentage points.")
            threshold = min(MAX_THRESHOLD_PTS, max(MIN_THRESHOLD_PTS, threshold))

        contribution = 0.0
        raw_contrib = entities.get("contribution") or entities.get("new_money")
        if raw_contrib not in (None, ""):
            contribution = parse_amount(raw_contrib) or 0.0
            if contribution <= 0:
                return _err(f"I couldn't read {raw_contrib!r} as an amount of new money.")

        holdings, no_qty = self._holdings()
        values: dict[str, float] = {}
        bucket_of = (lambda h: h["type"]) if mode == "type" else (lambda h: h["name"])
        display: dict[str, str] = {}         # target key -> the holding's own name, for output
        for h in holdings:
            if h["value"] > 0:
                key = bucket_of(h)
                if mode == "holding":
                    # "reliance", "RELIANCE.NS" and "Reliance" are the same holding
                    match = next((t for t in targets
                                  if t.lower() == key.lower() or normalise_ticker(t) == normalise_ticker(key)), None)
                    if match is not None:
                        display[match] = h["name"]
                        key = match
                values[key] = values.get(key, 0.0) + h["value"]
        total = sum(values.values())
        if total <= 0:
            return _err("I can't value your holdings yet. Track some with a quantity and a buy price first.")

        missing = sorted(k for k in values if k not in targets)
        if missing:
            return _err("You hold " + ", ".join(missing) + " but your target has no entry for "
                        + ("it" if len(missing) == 1 else "them")
                        + ". Add it (use 0 if you want none) with 'set my target allocation ...'.")

        rows = []
        for key in sorted(set(targets) | set(values), key=lambda k: -targets.get(k, 0.0)):
            cur = values.get(key, 0.0)
            tgt = targets.get(key, 0.0)
            rows.append({
                "bucket": display.get(key, key), "value": cur, "actual_pct": cur / total * 100, "target_pct": tgt,
                "drift_pts": cur / total * 100 - tgt,
                "to_target": tgt / 100 * total - cur,          # + buy, - sell, to land exactly on target
            })
        breached = [r for r in rows if abs(r["drift_pts"]) >= threshold]
        at_cost = [h["name"] for h in holdings if h["at_cost"]]

        if contribution > 0:
            moves = self._new_money_moves(rows, total, contribution)
        elif breached:
            floor = max(1.0, MIN_MOVE_FRACTION * total)
            moves = [{"bucket": r["bucket"], "amount": r["to_target"]} for r in rows
                     if abs(r["to_target"]) >= floor]
            moves.sort(key=lambda m: m["amount"])           # sells first
        else:
            moves = []

        if mode == "holding":
            by_name = {h["name"].lower(): h for h in holdings}   # rows already carry the holding's own name
            for m in moves:
                h = by_name.get(m["bucket"].lower())
                if h and h["price"]:
                    m["shares"] = round(abs(m["amount"]) / h["price"], 4)

        text = self._render(rows, total, threshold, breached, moves, contribution, at_cost, no_qty, mode)
        return _ok(text, data={
            "mode": mode, "total_value": total, "threshold_pts": threshold,
            "rows": rows, "moves": moves, "contribution": contribution,
            "valued_at_cost": at_cost, "no_quantity": no_qty,
            "needs_rebalance": bool(breached),
        }, confidence=0.85 if not at_cost else 0.7)

    @staticmethod
    def _new_money_moves(rows: list[dict], total: float, contribution: float) -> list[dict]:
        """Spread ``contribution`` over under-weight buckets in proportion to their shortfall."""
        new_total = total + contribution
        short = {r["bucket"]: max(0.0, r["target_pct"] / 100 * new_total - r["value"]) for r in rows}
        s = sum(short.values())
        if s <= 0:
            return []
        return [{"bucket": b, "amount": contribution * v / s} for b, v in short.items() if v > 0]

    def _render(self, rows, total, threshold, breached, moves, contribution, at_cost, no_qty, mode) -> str:
        cur = self.currency
        head = "Rebalance check" + (f" with {cur}{contribution:,.0f} of new money" if contribution else "")
        lines = [f"{head} (portfolio {cur}{total:,.0f}, drift threshold {threshold:g} points):"]
        width = max(10, max(len(r["bucket"]) for r in rows))
        lines.append(f"  {'':{width}}  {'now':>7} {'target':>7} {'drift':>9}")
        for r in rows:
            flag = ""
            if abs(r["drift_pts"]) >= threshold:
                flag = "  over" if r["drift_pts"] > 0 else "  under"
            lines.append(f"  {r['bucket']:{width}}  {r['actual_pct']:6.1f}% {r['target_pct']:6.1f}% "
                         f"{r['drift_pts']:+8.1f}{flag}")
        if contribution:
            after_total = total + contribution
            moved = {m["bucket"]: m["amount"] for m in moves}
            if not moves:
                lines.append("Nothing is under target, so there is nowhere to put new money without selling.")
            else:
                lines.append("Put the new money here (no selling):")
                for m in sorted(moves, key=lambda m: -m["amount"]):
                    lines.append("  " + self._move_text(m, buy=True))
                worst = max(abs(r["target_pct"] - (r["value"] + moved.get(r["bucket"], 0.0)) / after_total * 100)
                            for r in rows)
                lines.append(f"After that the furthest any area sits from target is {worst:.1f} points"
                             + (" (new money alone does not fully fix the drift)." if worst >= threshold else "."))
        elif not breached:
            lines.append(f"Everything is within {threshold:g} points of target, so no moves are needed.")
        else:
            lines.append("To get back to target:")
            for m in moves:
                lines.append("  " + self._move_text(m, buy=m["amount"] > 0))
            lines.append(f"(Or say 'rebalance with {cur}10000 new money' to fix it by buying only.)")
        if mode == "type" and moves and not contribution:
            lines.append("Moves are per asset class; which holding to sell or buy within a class is up to you.")
        if at_cost:
            lines.append(f"Valued at what you paid, since no live price was available: {', '.join(at_cost)}.")
        if no_qty:
            lines.append(f"Left out, no quantity recorded: {', '.join(no_qty)}.")
        lines.append("This ignores tax, brokerage and minimum lot sizes. Selling can trigger capital gains tax. "
                     "It is arithmetic against your own targets, not advice on what to hold.")
        return "\n".join(lines)

    def _move_text(self, m: dict, buy: bool) -> str:
        verb = "Buy" if buy else "Sell"
        text = f"{verb} about {self.currency}{abs(m['amount']):,.0f} of {m['bucket']}"
        if m.get("shares"):
            text += f" (about {m['shares']:g} units at the live price)"
        return text


def _targets_text(targets: dict[str, float]) -> str:
    return "\n".join(f"  {k}: {v:g}%" for k, v in sorted(targets.items(), key=lambda kv: -kv[1]))


def _ok(response: str, data: Optional[dict] = None, confidence: float = 0.9) -> dict:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict:
    return {"response": response, "data": {}, "confidence": 0.0}
