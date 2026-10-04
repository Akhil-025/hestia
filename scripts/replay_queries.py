"""
scripts/replay_queries.py

Replay real (anonymised) past queries against the NLU to catch classification
regressions when the prompt or model changes (backlog #215).

    # 1. Build a corpus from your own routing log (anonymised on the way out):
    python scripts/replay_queries.py extract --log logs/routing.jsonl \\
        --out data/nlu_replay_corpus.jsonl --name Rohan --name Priya

    # 2. Before editing config/nlu_prompt.txt, record how it does today:
    python scripts/replay_queries.py run --save data/nlu_replay_before.json

    # 3. After editing it, replay and compare with that run:
    python scripts/replay_queries.py run --against data/nlu_replay_before.json

    python scripts/replay_queries.py anonymise "mail priya@example.com 9876543210"   # preview

Corpus: one JSON object per line, ``{"query": ..., "expected": "<intent>",
"verified": false}``. Rows made by ``extract`` copy the intent the NLU chose at
the time and are marked ``"verified": false``: that is what the model *said*,
not what is *true*. A replay against unverified rows detects **change**, not
correctness. Edit the file, fix any wrong ``expected`` and set ``"verified":
true``; ``--verified-only`` then scores only what you have confirmed.

What a comparison reports: queries that were right and are now wrong
(regressions), wrong and now right (fixes), and, for unverified rows, any whose
intent changed at all. ``run`` exits 1 on a regression (or accuracy below
``--threshold``), 2 when the corpus is empty or the model can't be reached.

Anonymising is best effort and is for the *corpus*, which you may share or
commit: emails, URLs, phone and long numbers, IP addresses, Windows user
folders and any ``--name`` you give are replaced by placeholders. It cannot
recognise a name it wasn't told, or a street address, so read the file before
sharing it. ``logs/routing.jsonl`` itself is never modified.

The pure parts (anonymise, extract, replay, compare) take plain data and a
``classify`` callable, so tests/test_qa_tools.py runs them with no model.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = ROOT / "logs" / "routing.jsonl"
DEFAULT_CORPUS = ROOT / "data" / "nlu_replay_corpus.jsonl"

# Sources that are cheap shortcuts, not the model: replaying them would test
# the alias file / cache, not the prompt.
_NON_MODEL_SOURCES = frozenset({"cache", "alias", "multi_intent", "dry_run"})

# ---------------------------------------------------------------------------
# Anonymising
# ---------------------------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_WIN_USER = re.compile(r"(?i)([A-Z]:\\Users\\)[^\\\s]+")
_NIX_USER = re.compile(r"(/(?:home|Users)/)[^/\s]+")
_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
# 7+ digits, allowing the usual separators: phone numbers, account/card numbers.
_LONG_NUMBER = re.compile(r"(?<![\w.])\+?\d(?:[\s().-]?\d){6,}(?![\w])")


def anonymise(text: str, names: Iterable[str] = ()) -> str:
    """Replace personal identifiers with placeholders. Amounts, times and short
    numbers are kept: they are what the NLU needs to classify the query."""
    out = text or ""
    out = _EMAIL.sub("<email>", out)
    out = _URL.sub("<url>", out)
    out = _WIN_USER.sub(r"\1<user>", out)
    out = _NIX_USER.sub(r"\1<user>", out)
    out = _IPV4.sub("<ip>", out)
    out = _LONG_NUMBER.sub("<number>", out)
    for name in sorted({n.strip() for n in names if n and n.strip()}, key=len, reverse=True):
        out = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", "<name>", out, flags=re.IGNORECASE)
    return out


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    query: str
    expected: str
    verified: bool = False


def _norm_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def extract_rows(
    records: Iterable[dict],
    *,
    names: Iterable[str] = (),
    min_confidence: float = 0.0,
    limit: Optional[int] = None,
    valid_intents: Optional[Iterable[str]] = None,
) -> list[Row]:
    """Turn routing-log records into corpus rows.

    Dropped: records without a query or intent, ones answered by a shortcut
    rather than the model, chat (nothing to regress), anything below
    *min_confidence*, an intent the registry no longer has, and duplicates
    (compared case- and whitespace-insensitively, after anonymising). Newest
    records are kept when *limit* bites.
    """
    allowed = set(valid_intents) if valid_intents is not None else None
    names = list(names)
    seen: set[str] = set()
    rows: list[Row] = []
    for rec in reversed(list(records)):
        if not isinstance(rec, dict):
            continue
        query, intent = rec.get("query"), rec.get("intent")
        if not isinstance(query, str) or not query.strip() or not isinstance(intent, str):
            continue
        if rec.get("source") in _NON_MODEL_SOURCES or intent == "chat":
            continue
        try:
            if float(rec.get("confidence") or 0.0) < min_confidence:
                continue
        except (TypeError, ValueError):
            continue
        if allowed is not None and intent not in allowed:
            continue
        clean = anonymise(query, names)
        key = _norm_query(clean)
        if key in seen:
            continue
        seen.add(key)
        rows.append(Row(clean, intent, False))
        if limit is not None and len(rows) >= limit:
            break
    rows.reverse()
    return rows


def read_jsonl(path: Path) -> list[dict]:
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except OSError:
        return []
    return out


def load_corpus(path: Path) -> list[Row]:
    rows = []
    for obj in read_jsonl(path):
        q, e = obj.get("query"), obj.get("expected")
        if isinstance(q, str) and q.strip() and isinstance(e, str) and e:
            rows.append(Row(q, e, bool(obj.get("verified", False))))
    return rows


def write_corpus(path: Path, rows: Sequence[Row]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Replay and comparison
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Outcome:
    query: str
    expected: str
    got: str
    verified: bool

    @property
    def correct(self) -> bool:
        return self.got == self.expected


def replay(rows: Sequence[Row], classify: Callable[[str], str]) -> list[Outcome]:
    """Classify every row. An exception becomes ``got="<error>"``, never a stop."""
    out = []
    for r in rows:
        try:
            got = classify(r.query)
        except Exception as exc:                       # noqa: BLE001
            got = f"<error: {type(exc).__name__}>"
        out.append(Outcome(r.query, r.expected, str(got), r.verified))
    return out


def accuracy(outcomes: Sequence[Outcome], verified_only: bool = False) -> Optional[float]:
    pool = [o for o in outcomes if o.verified] if verified_only else list(outcomes)
    return sum(o.correct for o in pool) / len(pool) if pool else None


@dataclass
class Comparison:
    regressions: list[tuple[Outcome, str]]    # was right, now wrong: (now, previous got)
    fixes: list[tuple[Outcome, str]]          # was wrong, now right
    changed: list[tuple[Outcome, str]]        # intent changed, neither side right/verified
    new: int                                  # queries the earlier run didn't have

    @property
    def clean(self) -> bool:
        return not self.regressions


def compare(now: Sequence[Outcome], before: Sequence[Outcome]) -> Comparison:
    prev = {o.query: o for o in before}
    reg, fix, chg, new = [], [], [], 0
    for o in now:
        p = prev.get(o.query)
        if p is None:
            new += 1
            continue
        if p.correct and not o.correct:
            reg.append((o, p.got))
        elif o.correct and not p.correct:
            fix.append((o, p.got))
        elif o.got != p.got:
            chg.append((o, p.got))
    return Comparison(reg, fix, chg, new)


def outcomes_to_json(outcomes: Sequence[Outcome]) -> str:
    return json.dumps([asdict(o) for o in outcomes], indent=1, ensure_ascii=False)


def outcomes_from_json(text: str) -> list[Outcome]:
    try:
        raw = json.loads(text)
    except ValueError:
        return []
    out = []
    for obj in raw if isinstance(raw, list) else []:
        if isinstance(obj, dict) and all(k in obj for k in ("query", "expected", "got")):
            out.append(Outcome(obj["query"], obj["expected"], obj["got"], bool(obj.get("verified"))))
    return out


def format_report(outcomes: Sequence[Outcome], cmp: Optional[Comparison] = None,
                  verified_only: bool = False, show: int = 15) -> str:
    acc = accuracy(outcomes, verified_only)
    pool = [o for o in outcomes if o.verified] if verified_only else list(outcomes)
    n_ver = sum(o.verified for o in outcomes)
    lines = [f"{len(pool)} queries replayed ({n_ver} verified, {len(outcomes) - n_ver} unverified)."]
    if acc is not None:
        lines.append(f"Matches the corpus: {acc:.1%}"
                     + ("" if verified_only or n_ver == len(outcomes) else
                        "  (unverified rows are the model's own past answers: this measures change)"))
    wrong = [o for o in pool if not o.correct]
    if wrong and cmp is None:
        lines.append("Differences from the corpus:")
        lines += [f"  {o.query!r}: expected {o.expected}, got {o.got}" for o in wrong[:show]]
        if len(wrong) > show:
            lines.append(f"  ... and {len(wrong) - show} more")
    if cmp is not None:
        lines.append(f"Compared with the earlier run: {len(cmp.regressions)} regressed, "
                     f"{len(cmp.fixes)} fixed, {len(cmp.changed)} changed, {cmp.new} new.")
        for title, items in (("REGRESSED", cmp.regressions), ("fixed", cmp.fixes), ("changed", cmp.changed)):
            for o, before in items[:show]:
                lines.append(f"  {title}: {o.query!r}: {before} -> {o.got} (expected {o.expected})")
            if len(items) > show:
                lines.append(f"  ... and {len(items) - show} more {title.lower()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_classifier(args) -> Callable[[str], str]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from core.nlu import HestiaNLU
    kwargs = {"model": args.model, "host": args.host, "port": args.port}
    if args.prompt:
        kwargs["prompt_path"] = args.prompt
    if not args.with_aliases:
        kwargs.update(alias_path=None, cache_ttl_seconds=0)
    nlu = HestiaNLU(**kwargs)
    if not nlu._health_check():
        raise ConnectionError("Ollama is unreachable")
    return lambda q: nlu.understand(q).get("intent", "chat")


def _cmd_extract(args) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from modules.hecate.intent_registry import ALL_INTENTS
    records = read_jsonl(Path(args.log))
    if not records:
        print(f"No routing records found in {args.log}.", file=sys.stderr)
        return 2
    rows = extract_rows(records, names=args.name, min_confidence=args.min_confidence,
                        limit=args.limit, valid_intents=ALL_INTENTS)
    out = Path(args.out)
    if out.exists() and not args.force:
        keep = {r.query: r for r in load_corpus(out)}          # never lose hand-verified labels
        rows = [keep.get(r.query, r) for r in rows] + [r for q, r in keep.items()
                                                       if q not in {x.query for x in rows}]
    write_corpus(out, rows)
    print(f"Wrote {len(rows)} anonymised queries to {out}. Read it before sharing, and set "
          '"verified": true on rows you have checked.')
    return 0


def _cmd_run(args) -> int:
    rows = [r for r in load_corpus(Path(args.corpus)) if r.verified or not args.verified_only]
    if not rows:
        print(f"No usable rows in {args.corpus}. Run `extract` first.", file=sys.stderr)
        return 2
    try:
        classify = _build_classifier(args)
    except Exception as exc:                           # noqa: BLE001
        print(f"Can't reach the model: {exc}", file=sys.stderr)
        return 2
    outcomes = replay(rows, classify)
    cmp = None
    if args.against:
        before = outcomes_from_json(Path(args.against).read_text(encoding="utf-8")) \
            if Path(args.against).exists() else []
        if not before:
            print(f"Can't read an earlier run from {args.against}.", file=sys.stderr)
            return 2
        cmp = compare(outcomes, before)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(outcomes_to_json(outcomes), encoding="utf-8")
        print(f"Saved this run to {args.save}.")
    print(format_report(outcomes, cmp, args.verified_only))
    acc = accuracy(outcomes, args.verified_only)
    if cmp is not None and not cmp.clean:
        return 1
    if args.threshold is not None and acc is not None and acc < args.threshold:
        print(f"Below the {args.threshold:.0%} threshold.", file=sys.stderr)
        return 1
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Replay anonymised past queries against the NLU.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="build a corpus from the routing log")
    ex.add_argument("--log", default=str(DEFAULT_LOG))
    ex.add_argument("--out", default=str(DEFAULT_CORPUS))
    ex.add_argument("--name", action="append", default=[], help="a name to scrub (repeatable)")
    ex.add_argument("--min-confidence", type=float, default=0.0)
    ex.add_argument("--limit", type=int, default=None, help="keep the newest N")
    ex.add_argument("--force", action="store_true", help="overwrite instead of merging with the existing corpus")

    run = sub.add_parser("run", help="classify the corpus and report")
    run.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    run.add_argument("--save", help="write this run's outcomes here")
    run.add_argument("--against", help="compare with an earlier --save file")
    run.add_argument("--verified-only", action="store_true")
    run.add_argument("--threshold", type=float, default=None)
    run.add_argument("--with-aliases", action="store_true", help="also use the alias file and cache")
    run.add_argument("--model", default="mistral")
    run.add_argument("--host", default="localhost")
    run.add_argument("--port", type=int, default=11434)
    run.add_argument("--prompt", default=None, help="alternative prompt file to try")

    an = sub.add_parser("anonymise", help="show what a string looks like after scrubbing")
    an.add_argument("text")
    an.add_argument("--name", action="append", default=[])

    args = ap.parse_args(argv)
    if args.cmd == "anonymise":
        print(anonymise(args.text, args.name))
        return 0
    return {"extract": _cmd_extract, "run": _cmd_run}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
