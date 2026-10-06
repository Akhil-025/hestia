"""
core/classifier_data.py

Training data for the trained intent classifier (backlog #4): where the
labelled queries come from, and how it is scored.

Sources, in the order they are trusted
--------------------------------------
``label``   Queries you labelled by hand (``data/classifier_labels.jsonl``,
            written by ``add_label`` / ``scripts/train_classifier.py --label``).
            Weighted x3: these are the corrections that matter most.
``prompt``  The few-shot examples in ``config/nlu_prompt.txt`` (``User:`` line
            followed by a JSON line naming the intent).
``alias``   Phrases in ``config/intent_aliases.yaml`` (exact answers by
            construction).
``log``     Past classifications from ``logs/routing.jsonl`` that the LLM was
            confident about (>= 0.8) and that you did NOT flag with
            ``report_mistake`` afterwards. Records classified by the cache,
            an alias or this classifier itself are skipped, so the model never
            trains on its own output.

Held-out prompts
----------------
Any query that appears in ``hestia_test_prompts.md`` is removed from the
training set, so ``evaluate_on_golden`` measures prompts the model has never
seen. Without that the accuracy figure would just be memorisation.

Every label is checked against ``ALL_INTENTS``; an unknown intent is dropped,
never trained on.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Iterable, Optional

from core.intent_classifier import Example, normalise as _normalise
from modules.hecate.intent_registry import ALL_INTENTS, INTENT_MODULE_MAP

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_PATH = _ROOT / "config" / "nlu_prompt.txt"
DEFAULT_ALIAS_PATH = _ROOT / "config" / "intent_aliases.yaml"
DEFAULT_GOLDEN_PATH = _ROOT / "hestia_test_prompts.md"
DEFAULT_LABELS_PATH = Path("data") / "classifier_labels.jsonl"

LABEL_WEIGHT = 3.0
MIN_LOG_CONFIDENCE = 0.8

def normalise(text: str) -> str:
    """Comparison key: lower-cased, numbers collapsed, whitespace collapsed."""
    return " ".join(_normalise(text).split())


_USER_LINE = re.compile(r"^User:\s*(.+?)\s*$")


# ---------------------------------------------------------------------------
# Individual sources
# ---------------------------------------------------------------------------

def prompt_examples(path: "str | Path" = DEFAULT_PROMPT_PATH) -> list[Example]:
    """``User:`` / JSON pairs from the NLU prompt file."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[Example] = []
    for i, line in enumerate(lines[:-1]):
        m = _USER_LINE.match(line.strip())
        if not m:
            continue
        nxt = lines[i + 1].strip()
        if not nxt.startswith("{"):
            continue
        try:
            intent = json.loads(nxt).get("intent")
        except ValueError:
            continue
        if isinstance(intent, str) and intent in ALL_INTENTS:
            out.append(Example(m.group(1), intent, 1.0, "prompt"))
    return out


def alias_examples(path: "str | Path" = DEFAULT_ALIAS_PATH) -> list[Example]:
    """Every phrase in the alias file, as a training example."""
    try:
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    out: list[Example] = []
    for intent, phrases in data.items():
        if intent not in ALL_INTENTS or not isinstance(phrases, list):
            continue
        for p in phrases:
            text = p.get("phrase") if isinstance(p, dict) else p
            if isinstance(text, str) and text.strip():
                out.append(Example(text.strip(), intent, 1.0, "alias"))
    return out


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    rows.append(rec)
    except OSError:
        pass
    return rows


def label_examples(path: "str | Path" = DEFAULT_LABELS_PATH) -> list[Example]:
    """Hand-labelled queries (``{"query": ..., "intent": ...}`` per line)."""
    out: list[Example] = []
    for rec in _read_jsonl(Path(path)):
        q, intent = rec.get("query"), rec.get("intent")
        if isinstance(q, str) and q.strip() and intent in ALL_INTENTS:
            out.append(Example(q.strip(), intent, LABEL_WEIGHT, "label"))
    return out


def add_label(query: str, intent: str,
              path: "str | Path" = DEFAULT_LABELS_PATH) -> bool:
    """Append one hand label. False (and nothing written) for an unknown intent."""
    query = (query or "").strip()
    if not query or intent not in ALL_INTENTS:
        return False
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"query": query, "intent": intent}, ensure_ascii=False) + "\n")
    return True


def log_examples(routing_log: "str | Path" = "logs/routing.jsonl",
                 feedback_log: "str | Path" = "logs/feedback.jsonl",
                 min_confidence: float = MIN_LOG_CONFIDENCE) -> list[Example]:
    """Confident past LLM classifications that weren't flagged as wrong."""
    wrong = {
        (normalise(r.get("query", "")), r.get("intent", ""))
        for r in _read_jsonl(Path(feedback_log))
    }
    out: list[Example] = []
    for rec in _read_jsonl(Path(routing_log)):
        if rec.get("source", "nlu") != "nlu":
            continue
        q, intent = rec.get("query"), rec.get("intent")
        if not isinstance(q, str) or intent not in ALL_INTENTS or intent == "chat":
            continue
        try:
            conf = float(rec.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if conf < min_confidence or (normalise(q), intent) in wrong:
            continue
        out.append(Example(q, intent, 1.0, "log"))
    return out


def feedback_count(feedback_log: "str | Path" = "logs/feedback.jsonl") -> int:
    return len(_read_jsonl(Path(feedback_log)))


# ---------------------------------------------------------------------------
# Golden prompts (held out)
# ---------------------------------------------------------------------------

def golden_cases(path: "str | Path" = DEFAULT_GOLDEN_PATH):
    """Scored cases from ``hestia_test_prompts.md`` (module- and intent-level)."""
    from scripts.eval_intents import parse_golden_dataset
    try:
        return [c for c in parse_golden_dataset(path) if c.kind in ("module", "intent")]
    except OSError:
        return []


def held_out_queries(path: "str | Path" = DEFAULT_GOLDEN_PATH) -> set[str]:
    """Normalised text of EVERY quoted prompt in the golden file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return set()
    return {normalise(m) for m in re.findall(r'^\s*-\s+"((?:[^"\\]|\\.)*)"', text, re.M)}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def collect_examples(
    *,
    prompt_path: "str | Path" = DEFAULT_PROMPT_PATH,
    alias_path: "str | Path" = DEFAULT_ALIAS_PATH,
    labels_path: "str | Path" = DEFAULT_LABELS_PATH,
    routing_log: "str | Path" = "logs/routing.jsonl",
    feedback_log: "str | Path" = "logs/feedback.jsonl",
    golden_path: "str | Path" = DEFAULT_GOLDEN_PATH,
    hold_out_golden: bool = True,
) -> list[Example]:
    """All training examples, golden prompts removed, exact duplicates merged
    (the most trusted source wins, so a hand label overrides a log line)."""
    held = held_out_queries(golden_path) if hold_out_golden else set()
    ordered = (
        label_examples(labels_path)
        + prompt_examples(prompt_path)
        + alias_examples(alias_path)
        + log_examples(routing_log, feedback_log)
    )
    seen: dict[str, Example] = {}
    for ex in ordered:
        key = normalise(ex.text)
        if not key or key in held:
            continue
        # A label for the same text always replaces a weaker source's answer.
        if key not in seen:
            seen[key] = ex
    return list(seen.values())


def make_examples_fn(**kwargs) -> Callable[[], list[Example]]:
    """A zero-argument callable for ``ClassifierService(examples_fn=...)``."""
    return lambda: collect_examples(**kwargs)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def evaluate_on_golden(service, cases: Optional[Iterable] = None) -> dict:
    """Score a ``ClassifierService`` on the golden prompts.

    Reports, separately, how often it ANSWERED, and how often the answers were
    right (module-level for Section 1 cases, exact intent for Section 2). A
    model that declines is not counted wrong: declining hands the query to the
    existing tiers, which is the designed behaviour.
    """
    cases = list(cases) if cases is not None else golden_cases()
    answered = right = 0
    wrong: list[dict] = []
    for case in cases:
        pred = service.classify(case.prompt)
        if pred is None:
            continue
        answered += 1
        if case.kind == "module":
            ok = INTENT_MODULE_MAP.get(pred.intent) == case.expected
        else:
            ok = pred.intent == case.expected and (
                case.forbidden is None or pred.intent != case.forbidden)
        if ok:
            right += 1
        else:
            wrong.append({"prompt": case.prompt, "expected": case.expected,
                          "got": pred.intent, "probability": pred.probability})
    n = len(cases)
    return {
        "cases": n, "answered": answered, "right": right,
        "coverage": round(answered / n, 3) if n else 0.0,
        "precision": round(right / answered, 3) if answered else 0.0,
        "wrong": wrong,
    }
