#!/usr/bin/env python3
"""
scripts/calibrate_thresholds.py  (Hardening Plan, Phase 6.4)

Pick confidence thresholds from MEASURED data instead of guessing. Input is the
JSON from `eval_golden.py --live --json out.json`. For each candidate threshold
it reports accuracy of the answers above it, the share of traffic they cover,
and the false-write rate, then recommends:

  execute   lowest threshold whose accuracy >= --target-acc (default 0.97)
            and has no false writes -> run the action
  validate  below execute, still >= 0.5 of the way -> ask "did you mean ...?"
  clarify   below validate -> ask a clarifying question
  fallback  below that -> chat

    python scripts/calibrate_thresholds.py out.json --target-acc 0.97
"""
import argparse
import json
import sys


def table(rows):
    steps = [i / 20 for i in range(4, 20)]
    out = []
    for t in steps:
        sel = [r for r in rows if (r.get("confidence") or 0) >= t]
        if not sel:
            continue
        acc = sum(r["ok"] for r in sel) / len(sel)
        fw = sum(r["false_write"] for r in sel)
        out.append({"threshold": round(t, 2), "coverage": round(len(sel) / len(rows), 3),
                    "accuracy": round(acc, 4), "false_writes": fw})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--target-acc", type=float, default=0.97)
    a = ap.parse_args(argv)
    rows = [r for r in json.load(open(a.results))["rows"] if r.get("confidence") is not None]
    if not rows:
        print("no rows with confidence", file=sys.stderr)
        return 1
    tbl = table(rows)
    for r in tbl:
        print(r)
    execute = next((r["threshold"] for r in tbl
                    if r["accuracy"] >= a.target_acc and r["false_writes"] == 0), None)
    if execute is None:
        print("\nNo threshold reaches the target; keep confirmations on and improve the model/examples.")
        return 2
    validate = round(max(0.0, execute - 0.2), 2)
    clarify = round(max(0.0, validate - 0.2), 2)
    print(f"\nRecommended: execute>={execute}  validate>={validate}  clarify>={clarify}  else fallback")
    print("Set in config: nlu.thresholds: {execute: %s, validate: %s, clarify: %s}" % (execute, validate, clarify))
    return 0


if __name__ == "__main__":
    sys.exit(main())
