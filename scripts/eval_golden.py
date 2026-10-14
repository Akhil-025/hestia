#!/usr/bin/env python3
"""
scripts/eval_golden.py  (Hardening Plan, Phase 6.3-6.4)

Evaluate tests/golden/eval_set.yaml and report:
  * overall + per-category + per-intent accuracy
  * entity accuracy (expected entities must appear in the result)
  * false-write rate on adversarial cases (a question/statement that must not write)
  * latency p50/p95 per resolution stage

    python scripts/eval_golden.py                       # deterministic layer, no Ollama (CI)
    python scripts/eval_golden.py --live                # whole set through real NLU (needs Ollama)
    python scripts/eval_golden.py --live --json out.json   # save results for calibrate_thresholds.py
    python scripts/eval_golden.py --min-accuracy 0.95 --max-false-write 0

Exit 1 when a threshold is missed (use as a pre-push gate; CI has no Ollama so it
runs the deterministic layer only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WRITE_INTENTS = {"take_note", "learn_fact", "save_name", "remove_habit", "delete_notes",
                 "forget_fact", "add_habit", "pluto_log_expense", "set_reminder",
                 "add_goal", "remove_goal", "create_event", "send_email"}
DETERMINISTIC_SOURCES = {"alias", "slots", "fast", "cache", "classifier_stage1"}


def _pct(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))] if vals else 0.0


def build_nlu(live: bool):
    from core.nlu import HestiaNLU
    nlu = HestiaNLU(cache_ttl_seconds=0)
    if not live:
        nlu._health_check = lambda: False      # never reaches Ollama
    return nlu


def run(cases, nlu, live: bool):
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        res = nlu.understand(c["utterance"], None) or {}
        ms = (time.perf_counter() - t0) * 1000
        src = res.get("source") or ("fast" if (res.get("confidence") or 0) >= 0.98 else "fallback")
        intent = res.get("intent")
        unresolved = (not live) and src not in DETERMINISTIC_SOURCES
        if c.get("must_not_write") is not None:
            ok = (intent not in WRITE_INTENTS) or unresolved
            false_write = intent in WRITE_INTENTS and not unresolved
        else:
            ok = (intent == c["intent"]) and not unresolved
            false_write = False
        ents = res.get("entities") or {}
        exp = c.get("entities") or {}
        ent_ok = all(str(ents.get(k, "")).strip().lower() == str(v).strip().lower()
                     for k, v in exp.items()) if exp and ok else None
        rows.append({"utterance": c["utterance"], "expected": c.get("intent"),
                     "got": intent, "category": c["category"], "source": src,
                     "confidence": res.get("confidence"), "ok": ok,
                     "false_write": false_write, "entity_ok": ent_ok,
                     "unresolved": unresolved, "ms": round(ms, 2)})
    return rows


def summarise(rows):
    n = len(rows) or 1
    scored = [r for r in rows if not r["unresolved"] or r["category"] == "adversarial"]
    cat = defaultdict(lambda: [0, 0])
    per_intent = defaultdict(lambda: [0, 0])
    for r in rows:
        cat[r["category"]][1] += 1
        cat[r["category"]][0] += int(r["ok"])
        if r["expected"]:
            per_intent[r["expected"]][1] += 1
            per_intent[r["expected"]][0] += int(r["ok"])
    adv = [r for r in rows if r["category"] == "adversarial"]
    ents = [r for r in rows if r["entity_ok"] is not None]
    lat = defaultdict(list)
    for r in rows:
        lat[r["source"]].append(r["ms"])
    return {
        "cases": len(rows),
        "accuracy": round(sum(r["ok"] for r in rows) / n, 4),
        "unresolved": sum(r["unresolved"] for r in rows),
        "false_write_rate": round(sum(r["false_write"] for r in adv) / (len(adv) or 1), 4),
        "entity_accuracy": round(sum(r["entity_ok"] for r in ents) / (len(ents) or 1), 4) if ents else None,
        "by_category": {k: round(v[0] / v[1], 4) for k, v in cat.items()},
        "worst_intents": sorted(((round(v[0] / v[1], 3), k, v[1]) for k, v in per_intent.items()
                                 if v[0] < v[1]))[:15],
        "latency_ms": {s: {"n": len(v), "p50": round(_pct(v, .5), 2), "p95": round(_pct(v, .95), 2)}
                       for s, v in lat.items()},
    }


def main(argv=None) -> int:
    import yaml
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default=str(ROOT / "tests" / "golden" / "eval_set.yaml"))
    ap.add_argument("--live", action="store_true", help="include llm-layer cases; needs Ollama")
    ap.add_argument("--json", help="write per-case results here")
    ap.add_argument("--min-accuracy", type=float, default=None)
    ap.add_argument("--max-false-write", type=float, default=0.0)
    a = ap.parse_args(argv)
    cases = yaml.safe_load(Path(a.set).read_text(encoding="utf-8"))
    if not a.live:
        cases = [c for c in cases if c["layer"] == "deterministic"]
    rows = run(cases, build_nlu(a.live), a.live)
    s = summarise(rows)
    print(json.dumps(s, indent=2, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": s, "rows": rows}, indent=1), encoding="utf-8")
    bad = []
    if a.min_accuracy is not None and s["accuracy"] < a.min_accuracy:
        bad.append(f"accuracy {s['accuracy']} < {a.min_accuracy}")
    if s["false_write_rate"] > a.max_false_write:
        bad.append(f"false-write rate {s['false_write_rate']} > {a.max_false_write}")
    for b in bad:
        print("FAIL:", b, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
