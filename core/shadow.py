"""
core/shadow.py

Shadow mode for new intent handlers (backlog #16): route to the new handler
but also run the old path, log both outputs, and diff them before switching.

A rule names one NLU intent, the *candidate* handler (module + intent) that is
meant to replace whatever Hecate would have chosen (the *baseline*), and a mode:

``shadow``  The baseline answers you. The candidate runs in the background on
            the same input; only the comparison is recorded. (Start here.)
``canary``  The candidate answers you. The baseline runs in the background and
            is compared. (Next step, once ``shadow`` agrees.)
``live``    The candidate answers; nothing is compared. (Promoted.)

THE SAFETY RULE
---------------
Shadow and canary run BOTH handlers for one request. A handler that writes
(logs an expense, sends an email, sets a reminder) would do its work twice.
So a rule may only use ``shadow``/``canary`` if it declares ``read_only: true``,
asserting that both the baseline and the candidate only read. A rule that
doesn't is rejected at load time and listed in ``ShadowRules.rejected``; it is
never silently run unsafely. Nothing here can verify that claim for you.

What "agree" means
------------------
Replies are compared as normalised text with ``difflib``: identical, or a
similarity ratio at/above ``similarity`` (default 0.85), counts as agreement.
That is a text comparison, not a judgement that the two answers are equally
good: a candidate can agree with a wrong baseline. ``verdict`` therefore says
"ready to promote" only when enough comparisons exist, agreement is high and
the candidate never errored; a human still flips the mode.

Records go to ``logs/shadow.jsonl`` (rotating); ``python main.py
--shadow-report`` prints the summary.

Config::

    shadow:
      enabled: true
      rules:
        - name: weather_v2
          intent: get_weather
          candidate: {module: chronos, intent: get_weather_v2}
          mode: shadow
          read_only: true
          min_samples: 20
          agreement: 0.9
"""
from __future__ import annotations

import difflib
import json
import logging
import logging.handlers
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

MODES = ("shadow", "canary", "live")
DEFAULT_SIMILARITY = 0.85
DEFAULT_MIN_SAMPLES = 20
DEFAULT_AGREEMENT = 0.9
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").lower()).strip()


def compare_replies(a: str, b: str, similarity: float = DEFAULT_SIMILARITY) -> dict:
    """Compare two reply strings. ``agree`` is identical-or-similar."""
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return {"identical": True, "similarity": 1.0, "agree": True}
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    return {"identical": False, "similarity": round(ratio, 3), "agree": ratio >= similarity}


@dataclass
class ShadowRule:
    name: str
    intent: str
    candidate_module: str
    candidate_intent: str
    mode: str = "shadow"
    read_only: bool = False
    min_samples: int = DEFAULT_MIN_SAMPLES
    agreement: float = DEFAULT_AGREEMENT
    similarity: float = DEFAULT_SIMILARITY


class ShadowRules:
    """Validated rules from config; invalid ones are kept in ``rejected``."""

    def __init__(self, rules: Optional[list] = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self._by_intent: dict[str, ShadowRule] = {}
        self.rejected: list[str] = []
        for raw in rules or []:
            self._add(raw)

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "ShadowRules":
        cfg = cfg or {}
        return cls(cfg.get("rules") or [], enabled=bool(cfg.get("enabled", False)))

    def _add(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            self.rejected.append(f"not a mapping: {raw!r}")
            return
        name = str(raw.get("name") or raw.get("intent") or "?")
        cand = raw.get("candidate") or {}
        mode = str(raw.get("mode", "shadow")).lower()
        problems = []
        if not raw.get("intent"):
            problems.append("missing intent")
        if not (isinstance(cand, dict) and cand.get("module") and cand.get("intent")):
            problems.append("candidate needs module and intent")
        if mode not in MODES:
            problems.append(f"mode must be one of {MODES}")
        if mode in ("shadow", "canary") and raw.get("read_only") is not True:
            problems.append("shadow/canary run both handlers, so the rule must declare read_only: true")
        if problems:
            self.rejected.append(f"{name}: " + "; ".join(problems))
            logger.warning("Shadow rule %s rejected: %s", name, "; ".join(problems))
            return
        rule = ShadowRule(
            name=name, intent=str(raw["intent"]).strip().lower(),
            candidate_module=str(cand["module"]), candidate_intent=str(cand["intent"]),
            mode=mode, read_only=bool(raw.get("read_only", False)),
            min_samples=int(raw.get("min_samples", DEFAULT_MIN_SAMPLES)),
            agreement=float(raw.get("agreement", DEFAULT_AGREEMENT)),
            similarity=float(raw.get("similarity", DEFAULT_SIMILARITY)),
        )
        self._by_intent[rule.intent] = rule

    def rule_for(self, intent: str) -> Optional[ShadowRule]:
        if not self.enabled:
            return None
        return self._by_intent.get((intent or "").strip().lower())

    @property
    def rules(self) -> list[ShadowRule]:
        return list(self._by_intent.values())


class ShadowRecorder:
    """Runs the off-path handler, compares, and writes one JSON line each."""

    def __init__(self, rules: ShadowRules, log_path: "str | Path" = "logs/shadow.jsonl",
                 runner: Optional[Callable] = None, synchronous: bool = False) -> None:
        self.rules = rules
        self.log_path = Path(log_path)
        self._runner = runner            # (module, intent, entities, context) -> dict | None
        self.synchronous = synchronous
        self._lock = threading.Lock()
        self._log = None
        self._open_log()

    def _open_log(self) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                self.log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            lg = logging.getLogger(f"hestia.shadow.{id(self)}")
            lg.propagate = False
            lg.setLevel(logging.INFO)
            lg.addHandler(handler)
            self._log = lg
        except OSError as exc:
            logger.debug("Shadow log unavailable: %s", exc)

    def set_runner(self, runner: Callable) -> None:
        self._runner = runner

    # -- recording --------------------------------------------------------

    def observe(self, rule: ShadowRule, *, query: str, entities: dict, context: dict,
                served_by: str, served_target: tuple, served_text: str,
                served_ms: float, other_target: tuple) -> None:
        """Run *other_target* off-path and record how it compares to what was served."""
        def _work() -> None:
            t0 = time.perf_counter()
            text, error = "", None
            try:
                raw = self._runner(other_target[0], other_target[1], entities, context) \
                    if self._runner else None
                if raw is None:
                    error = "handler unavailable (missing, refused, or its breaker is open)"
                else:
                    text = str(raw.get("response", "")) if isinstance(raw, dict) else str(raw)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            other_ms = (time.perf_counter() - t0) * 1000
            cmp = compare_replies(served_text, text, rule.similarity) if error is None else \
                {"identical": False, "similarity": 0.0, "agree": False}
            candidate_is_other = served_by == "baseline"
            self._write({
                "ts": time.time(), "rule": rule.name, "mode": rule.mode,
                "query": (query or "")[:300], "served_by": served_by,
                "served": f"{served_target[0]}.{served_target[1]}",
                "other": f"{other_target[0]}.{other_target[1]}",
                "served_reply": served_text[:500], "other_reply": text[:500],
                "served_ms": round(served_ms, 1), "other_ms": round(other_ms, 1),
                "candidate_error": error if candidate_is_other else None,
                "baseline_error": error if not candidate_is_other else None,
                **cmp,
            })
        if self.synchronous:
            _work()
        else:
            threading.Thread(target=_work, daemon=True, name="ShadowCompare").start()

    def _write(self, record: dict) -> None:
        if self._log is None:
            return
        try:
            with self._lock:
                self._log.info(json.dumps(record, ensure_ascii=False, default=str))
        except Exception:
            logger.debug("Shadow write failed.", exc_info=True)

    # -- reporting --------------------------------------------------------

    def records(self) -> list[dict]:
        out = []
        try:
            with self.log_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        return out

    def report(self) -> dict[str, dict]:
        """Per-rule statistics from the log."""
        by_rule: dict[str, list[dict]] = {}
        for rec in self.records():
            by_rule.setdefault(rec.get("rule", "?"), []).append(rec)
        rules = {r.name: r for r in self.rules.rules}
        out: dict[str, dict] = {}
        for name, recs in by_rule.items():
            n = len(recs)
            agree = sum(1 for r in recs if r.get("agree"))
            cand_err = sum(1 for r in recs if r.get("candidate_error"))
            base_err = sum(1 for r in recs if r.get("baseline_error"))
            sims = [float(r.get("similarity", 0.0)) for r in recs]
            rule = rules.get(name)
            rate = agree / n if n else 0.0
            ready = bool(
                rule and n >= rule.min_samples and rate >= rule.agreement and cand_err == 0)
            out[name] = {
                "samples": n, "agree": agree, "agreement_rate": round(rate, 3),
                "mean_similarity": round(sum(sims) / n, 3) if n else 0.0,
                "candidate_errors": cand_err, "baseline_errors": base_err,
                "mode": rule.mode if rule else "(rule removed)",
                "ready_to_promote": ready,
                "needed": rule.min_samples if rule else None,
            }
        return out

    def disagreements(self, rule: Optional[str] = None, limit: int = 10) -> list[dict]:
        rows = [r for r in self.records()
                if not r.get("agree") and (rule is None or r.get("rule") == rule)]
        return rows[-limit:]

    def summary(self) -> str:
        rep = self.report()
        if not self.rules.rules and not rep and not self.rules.rejected:
            return "No shadow rules are configured."
        lines = []
        for name, s in rep.items():
            verdict = ("READY to promote (a person still has to change the mode)"
                       if s["ready_to_promote"] else
                       f"not ready ({s['samples']}/{s['needed']} samples)"
                       if s["needed"] and s["samples"] < s["needed"] else "not ready")
            lines.append(
                f"{name} [{s['mode']}]: {s['samples']} compared, "
                f"{s['agreement_rate']:.0%} agree, mean similarity {s['mean_similarity']:.2f}, "
                f"candidate errors {s['candidate_errors']} -> {verdict}")
        for r in self.rules.rules:
            if r.name not in rep:
                lines.append(f"{r.name} [{r.mode}]: nothing compared yet")
        for bad in self.rules.rejected:
            lines.append(f"REJECTED rule {bad}")
        return "\n".join(lines)
