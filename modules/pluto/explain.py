"""
modules/pluto/explain.py

Plain-language rendering of a quant-score breakdown (backlog #145).

``MarketIntelligenceManager.generate_quant_score`` returns a number plus, since
this change, a ``breakdown`` describing how the number was built. This module
turns that structure into text. It has no heavy imports (no polars/xgboost), so
it can be tested and reused anywhere.

Two breakdown shapes exist, matching the two scoring methods:

``technical_heuristic``
    ``score = 0.5 + sum(contribution)``, exactly (before clipping to 0-1).
    Each component's ``contribution`` is in score points, so the lines add up.

``ml_model``
    XGBoost's per-feature contributions (``pred_contribs``) for the row that was
    scored. They are in the model's raw output units, which for a classifier is
    log-odds rather than probability, so they explain direction and relative
    weight but do not add up to the 0-1 score.
"""

from __future__ import annotations

from typing import Any, Optional


def _lean(signal: float) -> str:
    if signal > 0.1:
        return "bullish"
    if signal < -0.1:
        return "bearish"
    return "neutral"


def _fmt_raw(component: dict) -> str:
    raw = component.get("raw")
    if raw is None:
        return "n/a"
    unit = component.get("raw_unit", "")
    return f"{raw:+.2f}{unit}" if unit.startswith("%") else f"{raw:.1f}{unit}"


def render_score_breakdown(ticker: str, quant: dict) -> str:
    """Multi-line explanation of one ``generate_quant_score`` result."""
    method = quant.get("method")
    score = quant.get("score")
    reliability = quant.get("reliability", 0.0)
    breakdown: Optional[dict] = quant.get("breakdown")

    if method == "no_data" or score is None:
        return (f"There isn't enough price history for {ticker} to score it, so there is "
                "nothing to break down yet.")

    lines = [f"Quant score for {ticker}: {score:.2f} (0 = very bearish, 0.5 = neutral, 1 = very bullish)"]

    if not breakdown:
        lines.append("No breakdown is available for this score.")
        return "\n".join(lines)

    if method == "technical_heuristic":
        lines.append("Method: fixed technical-indicator formula, not a trained model. How it adds up:")
        lines.append(f"  {'start (neutral)':44} {breakdown.get('base', 0.5):.3f}")
        for c in breakdown.get("components", []):
            note = "  [capped]" if c.get("capped") else ""
            lines.append(
                f"  {c['label']:44} {c['contribution']:+.3f}   "
                f"({_fmt_raw(c)}, weight {c['weight']:.0%}, reads {_lean(c['signal'])}){note}"
            )
        lines.append(f"  {'total':44} {breakdown.get('score_unclipped', score):.3f}")
        if breakdown.get("clipped"):
            lines.append("  (the total fell outside 0-1 and was clipped to the score above)")
        biggest = max(breakdown.get("components", []), key=lambda c: abs(c["contribution"]), default=None)
        if biggest and abs(biggest["contribution"]) > 1e-9:
            direction = "up" if biggest["contribution"] > 0 else "down"
            lines.append(f"Biggest push: {biggest['label'].split(' (')[0]}, pulling the score {direction}.")
    elif method == "ml_model":
        lines.append("Method: trained XGBoost model. Each feature's contribution to this prediction:")
        comps = sorted(breakdown.get("components", []), key=lambda c: -abs(c["contribution"]))
        for c in comps:
            lines.append(f"  {c['label']:24} {c['contribution']:+.4f}   (value {c['raw']:.4g})")
        if breakdown.get("base") is not None:
            lines.append(f"  {'model baseline':24} {breakdown['base']:+.4f}")
        lines.append(breakdown.get("note", ""))
    else:
        lines.append(f"Method: {method}.")

    lines.append(
        f"Data behind it: reliability {reliability:.0%} "
        "(how much clean price history there was; it says nothing about bullish or bearish)."
    )
    lines.append("This describes how the number was produced, not whether it is right. Not financial advice.")
    return "\n".join(l for l in lines if l)


def heuristic_breakdown(momentum_pct: Optional[float], momentum_signal: float,
                        trend_pct: Optional[float], trend_signal: float,
                        rsi: Optional[float], rsi_signal: float,
                        weights: tuple[float, float, float],
                        score: float) -> dict:
    """Assemble the heuristic's breakdown dict. Pure arithmetic; no scoring decisions."""
    w_m, w_t, w_r = weights
    rows = [
        ("momentum", "Momentum (avg of last 5 daily returns)", momentum_pct, "%/day", momentum_signal, w_m),
        ("trend", "Trend (latest close vs 20-day average)", trend_pct, "%", trend_signal, w_t),
        ("rsi", "RSI mean-reversion (14-day)", rsi, "", rsi_signal, w_r),
    ]
    components = [
        {
            "key": key, "label": label,
            "raw": None if raw is None else float(raw), "raw_unit": unit,
            "signal": float(signal), "weight": float(weight),
            "contribution": 0.5 * float(weight) * float(signal),
            "capped": abs(float(signal)) >= 1.0,
        }
        for key, label, raw, unit, signal, weight in rows
    ]
    unclipped = 0.5 + sum(c["contribution"] for c in components)
    return {
        "base": 0.5,
        "components": components,
        "score_unclipped": unclipped,
        "clipped": not (0.0 <= unclipped <= 1.0),
        "final_score": float(score),
    }
