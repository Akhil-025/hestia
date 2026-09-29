"""
scripts/eval_intents.py

Confusion-matrix eval over hestia_test_prompts.md as a golden dataset
(backlog #21).

Why parse a markdown file instead of hand-writing a fixture
-------------------------------------------------------------
`hestia_test_prompts.md` already exists, was written by hand specifically
to probe the NLU's known weak spots (it says so in its own header — "built
from god_function.md and config/nlu_prompt.txt... the exact ambiguity
traps the codebase's own comments call out"), and a chunk of it already
carries explicit ground truth in the prose: lines like

    - "List my habits" → must be `list_habits`, never `get_goals`

are annotations a human already made. Re-typing those into a second
fixture file would just create a second copy that can silently drift from
the first. So this script parses the existing file instead.

What actually has ground truth
-------------------------------
Not every line in the file does, and this script is honest about that
rather than inventing labels:

- **Section 1** ("Per-Module Sanity Checks"): every bullet under a
  `**ModuleName (...)**` header is a MODULE-level expectation — the
  query should route to that module. Entity/exact-intent correctness
  isn't checked here, just "did it land in the right god's hands."
- **Section 2** ("Boundary Cases the Devs Already Know Are Fragile"):
  lines with an explicit "→ `intent_name`" (optionally "must be
  `X`, never `Y`") are INTENT-level expectations — the exact
  intent, not just the module.
- **Everything else** (Sections 3-10: ambiguous routing, cross-module
  synthesis, negation/repair, low-confidence/refusal, security, voice-
  specific, load/consistency) is deliberately NOT scored pass/fail. The
  file's own text says so — "Phrases that plausibly belong to more than
  one god," "Test both readings" — grading those against a single
  correct answer would be fabricating ground truth the file's author
  explicitly declined to assert. Those are still run and logged, in a
  separate `manual_review` section of the report, because the actual
  classification is genuinely useful to look at by eye.

Usage
-----
    python scripts/eval_intents.py                  # full run against Ollama
    python scripts/eval_intents.py --threshold 0.75  # CI gate, non-zero exit below it
    python scripts/eval_intents.py --dataset-only    # parser smoke test, no NLU calls

CI integration: exits 1 if accuracy falls below --threshold (default 0.7)
on the scored (module + intent) cases, OR if Ollama is unreachable and
--require-ollama is passed. Without that flag, an unreachable Ollama exits
0 with a warning — see the module docstring rationale in
tests/test_eval_parser.py's companion test file for why the parser itself
(the part that never needs Ollama) is what CI actually gates on by
default, with the live-model run as an opt-in, not a hard CI requirement
for a project with no CI Ollama instance.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modules.hecate.intent_registry import ALL_INTENTS, strip_module_prefix

_DEFAULT_DATASET = _REPO_ROOT / "hestia_test_prompts.md"

# Display name (as it appears in "**Display Name (...)**") -> actual
# registered module name. Only Section 1's headers are matched against
# this; anything not in this map is treated as ordinary prose (a
# subsection heading like "**Habit vs. Goal**", not a module header).
_MODULE_DISPLAY_MAP: dict[str, str] = {
    "Hestia": "core",
    "Mnemosyne": "mnemosyne",
    "Hermes": "hermes",
    "Hephaestus": "hephaestus",
    "Chronos": "chronos",
    "Athena": "athena",
    "Iris": "iris",
    "Artemis": "artemis",
    "Ares": "ares",
    "Apollo": "apollo",
    "Orpheus": "orpheus",
    "Metis": "metis",
    "Dionysus": "dionysus",
    "Pluto": "pluto",
    # Hecate deliberately excluded: "No direct prompts; test indirectly
    # via ambiguous routing" — its section has no bullets to parse anyway.
}

_HEADER_RE = re.compile(r"^\*\*([A-Za-z]+)\s*\(")
_BULLET_RE = re.compile(r'^-\s+"((?:[^"\\]|\\.)*)"')
_SECTION_RE = re.compile(r"^##\s+(\d+)\.")


def _build_unprefixed_lookup() -> dict[str, str]:
    """
    Map an intent's bare, module-prefix-stripped form back to its one
    registered (prefixed) name — e.g. "rewrite_style" -> "orpheus_rewrite_style".

    hestia_test_prompts.md's "→ `intent_name`" annotations were written in
    the bare, conceptual form ("rewrite_style"), while
    intent_registry.py's actual keys carry the module prefix
    ("orpheus_rewrite_style") — see CONTRIBUTING.md's checklist item 2 for
    why that prefix convention exists. Without resolving one to the other
    here, this script would flag a perfectly correct classification as a
    failure just because the markdown and the registry write the same
    intent's name two different ways.

    A bare form shared by two different modules' intents (ambiguous) is
    deliberately left unresolved — falling through to "no match found" is
    safer than guessing which module's version a hand-written annotation
    meant.
    """
    grouped: dict[str, list[str]] = {}
    for intent in ALL_INTENTS:
        grouped.setdefault(strip_module_prefix(intent), []).append(intent)
    return {bare: only[0] for bare, only in grouped.items() if len(only) == 1}


_UNPREFIXED_LOOKUP = _build_unprefixed_lookup()


def _resolve_intent(name: str) -> str:
    """Resolve a bare intent name to its registered form; pass through
    already-registered or unresolvable names unchanged."""
    if name in ALL_INTENTS:
        return name
    return _UNPREFIXED_LOOKUP.get(name, name)


@dataclass
class EvalCase:
    prompt: str
    kind: str                       # "module" | "intent" | "manual"
    section: int
    expected: Optional[str] = None  # module name, or intent name
    forbidden: Optional[str] = None  # intent name that must NOT be chosen
    note: str = ""                  # the rest of the line, for the report


@dataclass
class CaseResult:
    case: EvalCase
    intent: str
    module: str
    confidence: float
    passed: Optional[bool]  # None for manual/unscored cases


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def scored(self) -> list[CaseResult]:
        return [r for r in self.results if r.passed is not None]

    @property
    def manual(self) -> list[CaseResult]:
        return [r for r in self.results if r.passed is None]

    @property
    def accuracy(self) -> float:
        scored = self.scored
        if not scored:
            return 1.0
        return sum(1 for r in scored if r.passed) / len(scored)

    def confusion(self) -> dict[str, dict[str, int]]:
        """{expected: {actual: count}} over intent-level cases only."""
        matrix: dict[str, dict[str, int]] = {}
        for r in self.results:
            if r.case.kind != "intent":
                continue
            row = matrix.setdefault(r.case.expected or "?", {})
            row[r.intent] = row.get(r.intent, 0) + 1
        return matrix

    def format(self) -> str:
        lines = [
            f"Scored: {len(self.scored)}  Accuracy: {self.accuracy:.1%}  "
            f"Manual/unscored: {len(self.manual)}",
        ]
        failures = [r for r in self.scored if not r.passed]
        if failures:
            lines.append(f"\n{len(failures)} failure(s):")
            for r in failures:
                if r.case.kind == "module":
                    lines.append(
                        f"  MODULE  {r.case.prompt!r}\n"
                        f"          expected module={r.case.expected!r}, "
                        f"got module={r.module!r} via intent={r.intent!r}"
                    )
                else:
                    lines.append(
                        f"  INTENT  {r.case.prompt!r}\n"
                        f"          expected={r.case.expected!r}"
                        + (f" (forbidden={r.case.forbidden!r})" if r.case.forbidden else "")
                        + f", got={r.intent!r} (confidence={r.confidence:.2f})"
                    )
        else:
            lines.append("\nNo failures among scored cases.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_golden_dataset(path: str | Path = _DEFAULT_DATASET) -> list[EvalCase]:
    """
    Parse hestia_test_prompts.md into EvalCase objects.

    Deterministic and needs no network/LLM access — see
    tests/test_eval_parser.py for the tests that exercise this function
    directly against the real file, in CI, every run.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()

    cases: list[EvalCase] = []
    section = 0
    current_module: Optional[str] = None

    for raw_line in lines:
        line = raw_line.strip()

        sec_match = _SECTION_RE.match(line)
        if sec_match:
            section = int(sec_match.group(1))
            current_module = None
            continue

        # Only Section 1's bold headers name a module; elsewhere "**...**"
        # is just emphasis on a subsection label (e.g. "**Habit vs. Goal**").
        if section == 1:
            header_match = _HEADER_RE.match(line)
            if header_match:
                current_module = _MODULE_DISPLAY_MAP.get(header_match.group(1))
                continue

        bullet_match = _BULLET_RE.match(line)
        if not bullet_match:
            continue
        prompt = bullet_match.group(1)
        rest = line[bullet_match.end():]

        if section == 1 and current_module:
            cases.append(EvalCase(
                prompt=prompt, kind="module", section=section,
                expected=current_module, note=rest.strip(),
            ))
            continue

        if section == 2 and "→" in rest:
            arrow_part = rest.split("→", 1)[1]
            if "?" in arrow_part:
                # An explicit "or ... ? Test both readings"-style hedge
                # even inside Section 2 — don't fabricate a single answer.
                cases.append(EvalCase(
                    prompt=prompt, kind="manual", section=section, note=rest.strip(),
                ))
                continue
            backticks = re.findall(r"`([a-z][a-z0-9_]*)`", arrow_part)
            if backticks:
                resolved = [_resolve_intent(b) for b in backticks]
                forbidden = None
                if "never" in arrow_part and len(resolved) >= 2:
                    forbidden = resolved[1]
                cases.append(EvalCase(
                    prompt=prompt, kind="intent", section=section,
                    expected=resolved[0], forbidden=forbidden, note=rest.strip(),
                ))
                continue

        # Anything with a quoted prompt but no ground truth we can safely
        # assert — still worth running and eyeballing, just not scored.
        cases.append(EvalCase(prompt=prompt, kind="manual", section=section, note=rest.strip()))

    return cases


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_eval(nlu, hecate, cases: list[EvalCase], active_modules: list[str]) -> EvalReport:
    """
    Classify every case through the real NLU + Hecate and score it.

    `nlu` and `hecate` are duck-typed (HestiaNLU / HecateEngine instances)
    so this can be driven by a fake in tests without booting Ollama.
    """
    report = EvalReport()
    for case in cases:
        try:
            nlu_result = nlu.understand(case.prompt, context=None)
        except Exception:
            report.results.append(CaseResult(
                case=case, intent="<nlu error>", module="<nlu error>",
                confidence=0.0, passed=False if case.kind != "manual" else None,
            ))
            continue

        intent = nlu_result.get("intent", "chat")
        confidence = float(nlu_result.get("confidence", 0.0) or 0.0)
        try:
            decision = hecate.decide(case.prompt, nlu_result, active_modules)
            module = decision.get("primary", "?")
        except Exception:
            module = "<hecate error>"

        if case.kind == "module":
            passed = module == case.expected
        elif case.kind == "intent":
            passed = intent == case.expected and (
                case.forbidden is None or intent != case.forbidden
            )
        else:
            passed = None

        report.results.append(CaseResult(
            case=case, intent=intent, module=module,
            confidence=confidence, passed=passed,
        ))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(_DEFAULT_DATASET))
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument(
        "--dataset-only", action="store_true",
        help="Parse and summarize the dataset without calling the NLU.",
    )
    parser.add_argument(
        "--require-ollama", action="store_true",
        help="Exit 1 (instead of 0 with a warning) if Ollama is unreachable.",
    )
    args = parser.parse_args(argv)

    cases = parse_golden_dataset(args.dataset)
    by_kind = {"module": 0, "intent": 0, "manual": 0}
    for c in cases:
        by_kind[c.kind] += 1
    print(
        f"Parsed {len(cases)} case(s) from {args.dataset}: "
        f"{by_kind['module']} module-level, {by_kind['intent']} intent-level, "
        f"{by_kind['manual']} manual/unscored."
    )
    if args.dataset_only:
        return 0

    from core.nlu import HestiaNLU
    from modules.hecate import HecateEngine

    nlu = HestiaNLU()
    if not nlu._health_check():
        message = "Ollama is unreachable; cannot run a live eval."
        if args.require_ollama:
            print(f"ERROR: {message}", file=sys.stderr)
            return 1
        print(f"WARNING: {message} Skipping live scoring.")
        return 0

    hecate = HecateEngine()
    active_modules = sorted({m for m in _MODULE_DISPLAY_MAP.values()})
    report = run_eval(nlu, hecate, cases, active_modules)
    print(report.format())

    if report.accuracy < args.threshold:
        print(
            f"\nAccuracy {report.accuracy:.1%} is below the {args.threshold:.0%} "
            f"threshold.", file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
