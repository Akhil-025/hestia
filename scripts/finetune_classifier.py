#!/usr/bin/env python
"""
scripts/finetune_classifier.py - fine-tune a distilbert-sized intent classifier,
and keep it only if it earns its place (backlog #25).

    python scripts/finetune_classifier.py                       # distilbert-base-uncased, GPU if there is one
    python scripts/finetune_classifier.py --augment --epochs 20
    python scripts/finetune_classifier.py --base distilbert-base-multilingual-cased   # for Hinglish
    python scripts/finetune_classifier.py --device cpu          # works, just slow

What it does
  1. Builds the same training set the classifier uses (prompt examples, aliases,
     your --label corrections, confident log lines; golden prompts held out;
     optional --augment variants) and sets ~12% aside as a validation split.
  2. Trains the base model; the epoch with the best validation accuracy is kept.
  3. Calibrates the answer/decline thresholds on the validation split.
  4. Scores it on the held-out golden prompts next to the TF-IDF baseline trained
     on exactly the same data.
  5. Promotion gate (core/classifier_data.gate_decision): the model replaces
     ``--out`` only if it answers at least 5 points more of the golden prompts
     without losing more than 2 points of precision (and stays above 90%).
     Otherwise it is left as ``<out>.candidate`` for you to inspect and exit
     code 3 says it was rejected. ``--no-promote`` never installs. A promoted
     model's predecessor is kept as ``<out>.previous``.
  Even a promoted model does nothing until the config says
  ``classifier.backend: transformer``; the swap stays a manual decision.

Requirements: ``pip install torch transformers``. For the RTX 4050 install a CUDA
build of torch (pytorch.org); the default CPU build trains on CPU. The first run
downloads the base model (~260 MB). The training loop itself has NOT been run
by the author (no torch or network where this was written); the gate exists
because of that.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPORT_PATH = Path("data") / "classifier_finetune_report.json"
DEFAULT_OUT = Path("data") / "intent_distilbert"
VAL_FRACTION = 0.12


def split_train_val(examples, frac: float = VAL_FRACTION, seed: int = 0):
    """``(train, val)``. Validation holds only real (non-augmented) examples of
    intents that keep at least two other examples in training, and every
    augmented variant of a validation example is dropped from training, so
    validation never sees a near-copy of what it is judging."""
    from core.classifier_backends import AUGMENT_SOURCE, _group
    by_intent: dict[str, set] = {}
    for e in examples:
        by_intent.setdefault(e.intent, set()).add(_group(e))
    eligible = sorted({_group(e) for e in examples
                       if e.source != AUGMENT_SOURCE and len(by_intent[e.intent]) >= 3})
    random.Random(seed).shuffle(eligible)
    chosen = set(eligible[: int(len(eligible) * frac)])
    val = [e for e in examples if e.source != AUGMENT_SOURCE and _group(e) in chosen]
    train = [e for e in examples if _group(e) not in chosen]
    return train, val


def check_dependencies() -> Optional[str]:
    missing = []
    for mod in ("torch", "transformers"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return (f"missing: {', '.join(missing)}. Install with:  pip install torch transformers  "
            f"(for the RTX 4050 use a CUDA build of torch from pytorch.org)") if missing else None


def run_finetune(
    *,
    base_model: str,
    epochs: int,
    batch_size: int,
    device: str,
    out_dir: Path,
    augment: bool,
    promote: bool,
    report_path: Path = REPORT_PATH,
    out: Callable[[str], None] = print,
    train_fn=None,
    scorer_fn=None,
    examples=None,
    cases=None,
) -> int:
    """The whole pipeline. ``train_fn`` / ``scorer_fn`` / ``examples`` / ``cases``
    exist so tests can run it without torch; the real ones are the defaults."""
    from core.classifier_backends import choose_thresholds
    from core.classifier_data import collect_examples, evaluate_on_golden, gate_decision, wilson_interval
    from core.intent_classifier import ClassifierService, TrainStats
    from core.transformer_classifier import (
        TransformerIntentClassifier, load_torch_scorer, train_transformer, write_labels,
    )
    from modules.hecate.intent_registry import ALL_INTENTS

    train_fn = train_fn or train_transformer
    scorer_fn = scorer_fn or load_torch_scorer
    exs = list(examples) if examples is not None else collect_examples(augment=augment)
    train, val = split_train_val(exs)
    out(f"{len(exs)} examples -> {len(train)} train, {len(val)} validation "
        f"({len({e.intent for e in train})} intents)")

    # Baseline: the TF-IDF model on exactly the same data, scored the same way.
    tmp_model = Path(report_path).parent / ".baseline_tmp.npz"
    base_svc = ClassifierService(model_path=tmp_model, mode="assist",
                                 examples_fn=lambda: exs, valid_intents=ALL_INTENTS)
    if not base_svc.train().get("ok"):
        out("Could not train the TF-IDF baseline; nothing to compare against.")
        return 1
    tmp_model.unlink(missing_ok=True)
    baseline = evaluate_on_golden(base_svc, cases)
    out(f"Baseline (tfidf): answered {baseline['answered']}/{baseline['cases']} ({baseline['coverage']:.0%}), "
        f"precision {baseline['precision']:.1%}")

    out(f"Fine-tuning {base_model} ...")
    cand = Path(str(out_dir) + ".candidate")
    if cand.exists():
        shutil.rmtree(cand)
    info = train_fn(train, val, cand, base_model=base_model, epochs=epochs,
                    batch_size=batch_size, device=device, log=out)

    # Thresholds from the validation split, not from the golden prompts.
    scorer = scorer_fn(cand, info.get("device", "cpu"))
    probe = TransformerIntentClassifier(scorer, info["classes"])
    classes = set(info["classes"])
    preds = [(e.intent, probe.predict(e.text, k=3)) for e in val if e.intent in classes]
    cal = choose_thresholds(preds, target_precision=0.95) if preds else {}
    thresholds = (cal["min_prob"], cal["min_margin"]) if cal else None
    stats = TrainStats(examples=len(train), classes=len(info["classes"]), seconds=info.get("seconds", 0.0),
                       backend="transformer", calibration=cal, sources={"validation": len(val)})
    write_labels(cand, info["classes"], thresholds, base_model, stats,
                 extra={"best_val_accuracy": info.get("best_val_accuracy")})

    svc = ClassifierService(model_path=cand, mode="assist", backend="transformer",
                            valid_intents=ALL_INTENTS, device=info.get("device", "cpu"))
    svc._clf = TransformerIntentClassifier(scorer, info["classes"], thresholds, stats)
    candidate = evaluate_on_golden(svc, cases)
    lo, hi = wilson_interval(candidate["right"], candidate["answered"])
    out(f"Fine-tuned: answered {candidate['answered']}/{candidate['cases']} ({candidate['coverage']:.0%}), "
        f"precision {candidate['precision']:.1%} [{lo:.0%}-{hi:.0%}], thresholds {thresholds}")

    passed, reasons = gate_decision(baseline, candidate)
    promoted = False
    if passed and promote:
        previous = Path(str(out_dir) + ".previous")
        if out_dir.exists():
            if previous.exists():
                shutil.rmtree(previous)
            out_dir.rename(previous)
        cand.rename(out_dir)
        promoted = True
    report = {"base_model": base_model, "augment": augment, "epochs": epochs, "device": info.get("device"),
              "baseline": {k: v for k, v in baseline.items() if k != "wrong"},
              "candidate": {k: v for k, v in candidate.items() if k != "wrong"},
              "candidate_wrong": candidate["wrong"][:20], "thresholds": list(thresholds) if thresholds else None,
              "best_val_accuracy": info.get("best_val_accuracy"), "gate_passed": passed,
              "gate_reasons": reasons, "promoted": promoted}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for r in reasons:
        out(f"  gate: {r}")
    if promoted:
        out(f"PROMOTED to {out_dir}. To use it set  classifier.backend: transformer  and "
            f"classifier.model_path: {out_dir}  in your config. Report: {report_path}")
        return 0
    if passed:
        out(f"Passed the gate but --no-promote was given; model left at {cand}. Report: {report_path}")
        return 0
    out(f"REJECTED by the gate: not installed. Left at {cand} for inspection; the TF-IDF model stays. "
        f"Report: {report_path}")
    return 3


def main(argv: Optional[list[str]] = None, **injected) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune a small transformer intent classifier (gated).")
    ap.add_argument("--base", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--no-promote", action="store_true", help="train and score, but never install")
    args = ap.parse_args(argv)
    if not injected.get("train_fn"):
        problem = check_dependencies()
        if problem:
            print(f"Cannot fine-tune: {problem}", file=sys.stderr)
            return 2
    return run_finetune(base_model=args.base, epochs=args.epochs, batch_size=args.batch_size,
                        device=args.device, out_dir=Path(args.out), augment=args.augment,
                        promote=not args.no_promote, **injected)


if __name__ == "__main__":
    raise SystemExit(main())
