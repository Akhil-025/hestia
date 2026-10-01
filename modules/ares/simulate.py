# modules/ares/simulate.py
"""
Lightweight Monte Carlo for numeric decisions (backlog #156).

Deliberately LLM-free: every number in the output comes from numbers the
user supplied, so there is nothing for a small local model to invent.

Two ways to describe an option:

  scenarios   a list of {probability, value} pairs ("60% chance of 10 lakh,
              40% chance of losing 2 lakh"). Probabilities may be given as
              fractions (0.6) or percents (60).
  range       low / likely / high, sampled from a triangular distribution
              ("between 50k and 200k, most likely 100k").

An optional fixed `cost` is subtracted from every outcome.
"""
from __future__ import annotations

import random
import re
import statistics

DEFAULT_RUNS = 10_000
MIN_RUNS = 1_000
MAX_RUNS = 200_000

_SUFFIX = {
    "k": 1e3, "m": 1e6, "mn": 1e6,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
    "crore": 1e7, "crores": 1e7, "cr": 1e7,
}
_AMOUNT_BODY = (
    r"\d[\d,]*(?:\.\d+)?\s*(?:k|mn|m|lakhs?|lacs?|crores?|cr)?(?![a-z0-9])"
)
_AMOUNT_RE = re.compile(r"(-|−)?\s*[₹$€£]?\s*(" + _AMOUNT_BODY + ")", re.IGNORECASE)
_LOSS_RE = re.compile(r"\b(los[te]|losing|loss|lose|cost|costing|spend|spending)\b", re.IGNORECASE)
_CURRENCY_WORDS = re.compile(r"\b(rs\.?|inr|usd|rupees?|dollars?)\b", re.IGNORECASE)


class SimulationInputError(ValueError):
    """Bad or missing numbers; the message is safe to show the user."""


def parse_amount(value) -> float:
    """'2 lakh', '₹1,50,000', '75k', '-3m', 1200 -> float. Raises SimulationInputError."""
    if isinstance(value, bool):
        raise SimulationInputError(f"{value!r} is not a number.")
    if isinstance(value, (int, float)):
        return float(value)
    text = _CURRENCY_WORDS.sub("", str(value or "")).strip()
    m = re.fullmatch(r"(-|−)?\s*[₹$€£]?\s*(\d[\d,]*(?:\.\d+)?)\s*([a-z]*)", text, re.IGNORECASE)
    if not m:
        raise SimulationInputError(f"I couldn't read {value!r} as a number.")
    num = float(m.group(2).replace(",", ""))
    suffix = m.group(3).lower()
    if suffix:
        if suffix not in _SUFFIX:
            raise SimulationInputError(f"I couldn't read {value!r} as a number.")
        num *= _SUFFIX[suffix]
    return -num if m.group(1) else num


def _prob(value) -> float:
    p = parse_amount(str(value).replace("%", "")) if not isinstance(value, (int, float)) else float(value)
    if p < 0:
        raise SimulationInputError("Probabilities can't be negative.")
    return p


def parse_text(text: str) -> dict | None:
    """
    Best-effort extraction of one option from free text. Returns an option dict
    ({"scenarios": [...]} or {"low":..,"likely":..,"high":..}) or None.
    """
    text = text or ""

    # "60% chance of 10 lakh, 40% chance of losing 2 lakh"
    pct = list(re.finditer(r"(\d+(?:\.\d+)?)\s*%", text))
    scenarios = []
    for i, m in enumerate(pct):
        end = pct[i + 1].start() if i + 1 < len(pct) else len(text)
        segment = text[m.end():end]
        am = _AMOUNT_RE.search(segment)
        if not am:
            continue
        value = parse_amount(am.group(2))
        if am.group(1) or _LOSS_RE.search(segment[:am.start()]):
            value = -abs(value)
        # "%" is explicit here, so convert to a fraction now rather than
        # leaving _normalise() to guess whether "1" means 1% or 100%.
        scenarios.append({"probability": float(m.group(1)) / 100.0, "value": value})
    if scenarios:
        return {"scenarios": scenarios}

    # "between 50k and 200k, most likely 100k"
    rng = re.search(
        r"between\s+(?:[₹$€£]\s*)?(" + _AMOUNT_BODY + r")\s*(?:and|to|-)\s*(?:[₹$€£]\s*)?(" + _AMOUNT_BODY + ")",
        text, re.IGNORECASE,
    )
    if rng:
        low, high = parse_amount(rng.group(1)), parse_amount(rng.group(2))
        mode = re.search(
            r"(?:most likely|likely|typically|usually|mode)\s*(?:is|at|around|about|~)?\s*(?:[₹$€£]\s*)?("
            + _AMOUNT_BODY + ")",
            text[rng.end():], re.IGNORECASE,
        )
        return {
            "low": low, "high": high,
            "likely": parse_amount(mode.group(1)) if mode else None,
        }
    return None


def _normalise(option: dict) -> dict:
    """Validate one option and return {name, kind, scenarios|low/likely/high, cost, notes}."""
    notes: list[str] = []
    name = str(option.get("name") or "Option").strip() or "Option"
    cost = parse_amount(option["cost"]) if option.get("cost") not in (None, "") else 0.0

    raw = option.get("scenarios")
    if raw:
        rows = []
        for s in raw:
            if not isinstance(s, dict):
                raise SimulationInputError("Each scenario needs a probability and a value.")
            p = s.get("probability", s.get("p"))
            v = s.get("value", s.get("payoff", s.get("amount")))
            if p is None or v is None:
                raise SimulationInputError("Each scenario needs a probability and a value.")
            rows.append([_prob(p), parse_amount(v) - cost])
        total = sum(r[0] for r in rows)
        if total <= 0:
            raise SimulationInputError("The probabilities add up to zero.")
        # Any single value above 1 means the user is speaking in percents
        # (60/40). Judging by the total instead would misread 0.8 + 0.8.
        if max(r[0] for r in rows) > 1.0:
            notes.append("Probabilities were read as percents.")
            for r in rows:
                r[0] /= 100.0
            total /= 100.0
        if total > 1.01:
            notes.append(f"Probabilities added up to {total:.0%}; scaled them to 100%.")
            for r in rows:
                r[0] /= total
            total = 1.0
        elif total < 0.99:
            rest = 1.0 - total
            notes.append(f"Probabilities added up to {total:.0%}; assumed a {rest:.0%} chance of nothing happening (value {-cost:g}).")
            rows.append([rest, 0.0 - cost])
        return {"name": name, "kind": "scenarios", "scenarios": rows, "cost": cost, "notes": notes}

    if option.get("low") is not None and option.get("high") is not None:
        low, high = parse_amount(option["low"]), parse_amount(option["high"])
        if low > high:
            low, high = high, low
        likely = option.get("likely")
        if likely in (None, ""):
            mode = (low + high) / 2
            notes.append("No 'most likely' value given; assumed the midpoint.")
        else:
            mode = parse_amount(likely)
            if not low <= mode <= high:
                raise SimulationInputError("'Most likely' has to sit between the low and high values.")
        return {"name": name, "kind": "range", "low": low - cost, "mode": mode - cost,
                "high": high - cost, "cost": cost, "notes": notes}

    raise SimulationInputError("No scenarios or low/high range to simulate.")


def _draw(opt: dict, rng: random.Random, runs: int) -> list[float]:
    if opt["kind"] == "range":
        lo, mo, hi = opt["low"], opt["mode"], opt["high"]
        if lo == hi:
            return [lo] * runs
        return [rng.triangular(lo, hi, mo) for _ in range(runs)]
    values = [r[1] for r in opt["scenarios"]]
    weights = [r[0] for r in opt["scenarios"]]
    return rng.choices(values, weights=weights, k=runs)


def _expected(opt: dict) -> float:
    if opt["kind"] == "range":
        return (opt["low"] + opt["mode"] + opt["high"]) / 3.0
    return sum(p * v for p, v in opt["scenarios"])


def _pct(sorted_vals: list[float], q: float) -> float:
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def run(options: list[dict], runs: int = DEFAULT_RUNS, seed: int | None = None) -> dict:
    """Simulate every option. Raises SimulationInputError on bad input."""
    if not options:
        raise SimulationInputError("Nothing to simulate.")
    runs = max(MIN_RUNS, min(MAX_RUNS, int(runs)))
    rng = random.Random(seed)

    results, draws = [], []
    for idx, raw in enumerate(options, 1):
        raw = dict(raw)
        raw.setdefault("name", f"Option {idx}" if len(options) > 1 else "Option")
        opt = _normalise(raw)
        vals = _draw(opt, rng, runs)
        draws.append(vals)
        ordered = sorted(vals)
        results.append({
            "name": opt["name"],
            "expected_value": _expected(opt),
            "mean": statistics.fmean(vals),
            "median": _pct(ordered, 0.5),
            "stdev": statistics.pstdev(vals),
            "p5": _pct(ordered, 0.05),
            "p95": _pct(ordered, 0.95),
            "worst": ordered[0],
            "best": ordered[-1],
            "p_loss": sum(1 for v in vals if v < 0) / runs,
            "notes": opt["notes"],
        })

    order = sorted(range(len(results)), key=lambda i: results[i]["expected_value"], reverse=True)
    out = {"runs": runs, "options": [results[i] for i in order], "p_top_beats_next": None}
    if len(order) > 1:
        a, b = draws[order[0]], draws[order[1]]
        out["p_top_beats_next"] = sum(1 for x, y in zip(a, b) if x > y) / runs
    return out


def _fmt(x: float) -> str:
    if abs(x) >= 100:
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def format_result(topic: str, result: dict) -> str:
    lines = [f"Monte Carlo: {topic.title()} ({result['runs']:,} runs)", ""]
    for r in result["options"]:
        if len(result["options"]) > 1 or r["name"] != "Option":
            lines.append(f"OPTION: {r['name']}")
        lines += [
            f"  Expected value : {_fmt(r['expected_value'])}",
            f"  Median         : {_fmt(r['median'])}",
            f"  Typical range  : {_fmt(r['p5'])} to {_fmt(r['p95'])}  (5th to 95th percentile)",
            f"  Worst / best   : {_fmt(r['worst'])} / {_fmt(r['best'])}",
            f"  Chance of loss : {r['p_loss']:.0%}",
        ]
        for n in r["notes"]:
            lines.append(f"  Note: {n}")
        lines.append("")
    if result["p_top_beats_next"] is not None:
        top, nxt = result["options"][0], result["options"][1]
        lines.append(
            f"By expected value alone, {top['name']} leads; it beats {nxt['name']} "
            f"in {result['p_top_beats_next']:.0%} of runs."
        )
        if top["p_loss"] > nxt["p_loss"]:
            lines.append(f"It also carries a higher chance of loss ({top['p_loss']:.0%} vs {nxt['p_loss']:.0%}).")
        lines.append("")
    lines.append("These results only reflect the numbers you gave me; no other data was used.")
    return "\n".join(lines).strip()
