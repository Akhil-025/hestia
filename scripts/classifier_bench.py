#!/usr/bin/env python
"""
scripts/classifier_bench.py - measure the intent classifier backends (backlog #25).

Trains each backend in memory (it never touches your live model files), scores
it on the held-out golden prompts (``hestia_test_prompts.md``, removed from
every training set), and prints one comparable row per run.

  coverage   share of golden prompts it ANSWERED; the rest it declines and the
             normal routing tiers handle them.
  precision  of the ones it answered, how many were right (with a 95% interval).

    python scripts/classifier_bench.py                          # tfidf, the baseline
    python scripts/classifier_bench.py --save-baseline          # record it in data/classifier_baseline.json
    python scripts/classifier_bench.py --backends tfidf,embedding,ensemble --augment both --show-wrong

The embedding and ensemble backends need ``sentence-transformers`` and the
all-MiniLM-L6-v2 model (the one the rest of Hestia already uses). The golden
set is small (~100 cases): differences of a few cases are noise, so read the
interval, not just the point value.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASELINE_PATH = Path("data") / "classifier_baseline.json"


def run_one(backend: str, augment: bool, *, embedder=None, embedding_model: Optional[str] = None,
            device: str = "cpu", cases=None, examples=None) -> dict:
    """Train *backend* in a temp dir and score it on the golden prompts."""
    from core.classifier_data import collect_examples, evaluate_on_golden, wilson_interval
    from core.intent_classifier import ClassifierService
    from modules.hecate.intent_registry import ALL_INTENTS

    exs = list(examples) if examples is not None else collect_examples(augment=augment)
    with tempfile.TemporaryDirectory() as tmp:
        svc = ClassifierService(model_path=Path(tmp) / "m.npz", mode="assist", backend=backend,
                                examples_fn=lambda: exs, valid_intents=ALL_INTENTS,
                                embedding_model=embedding_model, device=device, embedder=embedder)
        res = svc.train()
        if not res.get("ok"):
            return {"backend": backend, "augment": augment, "error": res.get("error", "training failed")}
        ev = evaluate_on_golden(svc, cases)
        st = res["stats"]
        lo, hi = wilson_interval(ev["right"], ev["answered"])
        return {
            "backend": backend, "augment": augment, "examples": st.examples, "classes": st.classes,
            "train_seconds": round(st.seconds, 1), "thresholds": list(svc._clf.default_thresholds),
            "cases": ev["cases"], "answered": ev["answered"], "right": ev["right"],
            "coverage": ev["coverage"], "precision": ev["precision"], "precision_ci": [lo, hi],
            "calibration": st.calibration, "wrong": ev["wrong"],
        }


def format_row(r: dict) -> str:
    tag = f"{r['backend']:<10} {'aug' if r['augment'] else '   '}"
    if "error" in r:
        return f"{tag}  FAILED: {r['error']}"
    lo, hi = r["precision_ci"]
    th = f"{r['thresholds'][0]:.2f}/{r['thresholds'][1]:.2f}"
    return (f"{tag}  {r['examples']:>5} ex  answered {r['answered']:>2}/{r['cases']} "
            f"({r['coverage']:.0%})  right {r['right']:>2}/{r['answered']:<2} "
            f"precision {r['precision']:.1%} [{lo:.0%}-{hi:.0%}]  thresholds {th}  {r['train_seconds']}s")


def run_bench(backends: list[str], augment_modes: list[bool], *, embedder=None,
              embedding_model: Optional[str] = None, device: str = "cpu",
              out: Callable[[str], None] = print) -> list[dict]:
    rows = []
    for backend in backends:
        for aug in augment_modes:
            r = run_one(backend, aug, embedder=embedder, embedding_model=embedding_model, device=device)
            out(format_row(r))
            rows.append(r)
    return rows


def save_baseline(row: dict, path: Path = BASELINE_PATH) -> None:
    keep = {k: row[k] for k in ("backend", "augment", "examples", "classes", "cases", "answered",
                                "right", "coverage", "precision", "precision_ci", "thresholds")}
    keep["recorded"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    keep["wrong"] = row["wrong"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keep, indent=2), encoding="utf-8")


def main(argv: Optional[list[str]] = None, out: Callable[[str], None] = print,
         embedder=None, baseline_path: Path = BASELINE_PATH) -> int:
    ap = argparse.ArgumentParser(description="Score intent classifier backends on the held-out golden prompts.")
    ap.add_argument("--backends", default="tfidf", help="comma list of: tfidf, embedding, ensemble")
    ap.add_argument("--augment", choices=("off", "on", "both"), default="off")
    ap.add_argument("--embedding-model", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-baseline", action="store_true",
                    help="record the tfidf / no-augmentation run as data/classifier_baseline.json")
    ap.add_argument("--show-wrong", action="store_true", help="list the answered-but-wrong prompts")
    args = ap.parse_args(argv)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    bad = [b for b in backends if b not in ("tfidf", "embedding", "ensemble")]
    if bad:
        out(f"Unknown backend(s): {', '.join(bad)} (the fine-tuned model is scored by scripts/finetune_classifier.py).")
        return 2
    modes = {"off": [False], "on": [True], "both": [False, True]}[args.augment]
    rows = run_bench(backends, modes, embedder=embedder, embedding_model=args.embedding_model,
                     device=args.device, out=out)

    if args.show_wrong:
        for r in rows:
            for w in r.get("wrong", []):
                out(f"  [{r['backend']}] wrong: {w['prompt']!r} expected {w['expected']!r}, got {w['got']!r}")
    if args.save_baseline:
        base = next((r for r in rows if r["backend"] == "tfidf" and not r["augment"] and "error" not in r), None)
        if base is None:
            out("--save-baseline needs the tfidf backend with augmentation off in this run.")
            return 2
        save_baseline(base, baseline_path)
        out(f"Baseline saved to {baseline_path}.")
    elif baseline_path.is_file():
        try:
            b = json.loads(baseline_path.read_text(encoding="utf-8"))
            out(f"Saved baseline ({b['recorded'][:10]}): answered {b['coverage']:.0%}, "
                f"precision {b['precision']:.1%} on {b['cases']} cases.")
        except (ValueError, KeyError):
            out(f"({baseline_path} is unreadable; re-save it with --save-baseline.)")
    return 1 if any("error" in r for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
