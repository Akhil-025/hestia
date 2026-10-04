"""
scripts/smoke_test.py

Pre-deploy gate (backlog #212): boot the real Hestia, send ten canonical
queries, and fail unless every one gets a proper reply.

    python scripts/smoke_test.py                       # boot, run, exit 0 / 1
    python scripts/smoke_test.py --config config/laptop_config.yaml
    python scripts/smoke_test.py --queries my_queries.txt     # one query per line
    python scripts/smoke_test.py --routing-only        # no handlers run at all
    python scripts/smoke_test.py --json                # machine-readable result

What it checks, per query: the reply is non-empty, is not one of Hestia's own
failure messages ("something went wrong", "my backend isn't responding", a
traceback...), and arrived within ``--max-seconds``. With ``--routing-only``
it uses ``Hestia.resolve_only`` instead, so nothing executes and nothing is
written; it then checks the query was routed to a module that accepts it.

The default queries are read-only: none logs, saves, sends or schedules
anything, so running this on your real install leaves your data alone. They
do exercise the live path (Ollama, the NLU, Hecate, a handful of modules),
which is the point: this tells you the whole thing starts and answers, not
that any one feature is right. Exit codes: 0 all passed, 1 a query failed,
2 Hestia would not start.

The pure parts (``looks_like_error``, ``evaluate``, ``run_queries``) take the
callables they need, so tests/test_qa_tools.py covers them without booting
anything. Only ``boot_hestia`` imports ``main``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

CANONICAL_QUERIES: tuple[str, ...] = (
    "hello",
    "what time is it",
    "what's today's date",
    "what's the weather like",
    "what can you do",
    "what is twelve times twelve",
    "tell me a short joke",
    "what do you remember about me",
    "how are my habits going",
    "do I have any reminders today",
)

# Replies that mean the pipeline fell over even though it returned text.
_FAILURE_MARKERS = (
    "something went wrong",
    "my backend isn't responding",
    "my backend is not responding",
    "i had trouble understanding",
    "traceback (most recent call last)",
    "internal error",
    "exception:",
    "is not defined",
    "object has no attribute",
    "keyerror",
    "typeerror",
    "attributeerror",
    "valueerror",
    "nonetype",
    "connection refused",
    "max retries exceeded",
)


@dataclass
class QueryResult:
    query: str
    ok: bool
    seconds: float
    reply: str = ""
    problem: str = ""


def looks_like_error(reply: object) -> Optional[str]:
    """Why *reply* is not an acceptable answer, or None if it is."""
    if reply is None:
        return "no reply"
    text = str(reply).strip()
    if not text:
        return "empty reply"
    low = text.lower()
    for marker in _FAILURE_MARKERS:
        if marker in low:
            return f"reply looks like an error (contains {marker!r})"
    return None


def evaluate(query: str, reply: object, seconds: float, max_seconds: float) -> QueryResult:
    problem = looks_like_error(reply)
    if problem is None and seconds > max_seconds:
        problem = f"took {seconds:.1f}s, over the {max_seconds:g}s limit"
    return QueryResult(query, problem is None, round(seconds, 2),
                       str(reply or "")[:300], problem or "")


def run_queries(
    ask: Callable[[str], object],
    queries: Sequence[str],
    *,
    max_seconds: float = 60.0,
    clock: Callable[[], float] = time.perf_counter,
) -> list[QueryResult]:
    """Send each query through *ask*. An exception is a failure, never a crash."""
    results: list[QueryResult] = []
    for q in queries:
        start = clock()
        try:
            reply: object = ask(q)
        except Exception as exc:                       # noqa: BLE001 - this is the gate
            results.append(QueryResult(q, False, round(clock() - start, 2), "",
                                       f"raised {type(exc).__name__}: {exc}"))
            continue
        results.append(evaluate(q, reply, clock() - start, max_seconds))
    return results


def routing_problem(decision: object) -> Optional[str]:
    """Why a ``resolve_only`` result is not an acceptable routing, or None."""
    if not isinstance(decision, dict):
        return "no routing decision"
    if "error" in decision:
        return str(decision["error"])
    if not decision.get("primary"):
        return "no module chosen"
    if decision.get("primary_can_handle") is False:
        return (f"routed to {decision.get('primary')!r}, which cannot handle "
                f"{decision.get('dispatch_intent')!r}")
    return None


def run_routing_only(
    resolve: Callable[[str], object],
    queries: Sequence[str],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> list[QueryResult]:
    results: list[QueryResult] = []
    for q in queries:
        start = clock()
        try:
            decision = resolve(q)
        except Exception as exc:                       # noqa: BLE001
            results.append(QueryResult(q, False, round(clock() - start, 2), "",
                                       f"raised {type(exc).__name__}: {exc}"))
            continue
        problem = routing_problem(decision)
        summary = ""
        if isinstance(decision, dict) and not problem:
            summary = f"{decision.get('intent')} -> {decision.get('primary')}"
        results.append(QueryResult(q, problem is None, round(clock() - start, 2),
                                   summary, problem or ""))
    return results


def load_queries(path: str) -> list[str]:
    """One query per line; blank lines and ``#`` comments are ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def format_report(results: Sequence[QueryResult]) -> str:
    out = []
    for r in results:
        mark = "ok  " if r.ok else "FAIL"
        out.append(f"[{mark}] {r.seconds:6.2f}s  {r.query}")
        if not r.ok:
            out.append(f"         {r.problem}")
            if r.reply:
                out.append(f"         reply: {r.reply[:120]}")
    failed = sum(1 for r in results if not r.ok)
    out.append("")
    out.append(f"{len(results) - failed} of {len(results)} queries passed."
               + ("" if not failed else "  Do not deploy."))
    return "\n".join(out)


def exit_code(results: Sequence[QueryResult]) -> int:
    return 0 if results and all(r.ok for r in results) else 1


def boot_hestia(config_path: Optional[str] = None):
    """Construct the real application. Imported lazily: it is heavy, and the
    rest of this module must stay importable without it."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import main as hestia_main
    return hestia_main.Hestia(config_path=config_path) if config_path else hestia_main.Hestia()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Boot Hestia and send canonical queries.")
    ap.add_argument("--config", default=None, help="config file (default: main.py's default)")
    ap.add_argument("--queries", help="file with one query per line instead of the built-in ten")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="per-query time limit")
    ap.add_argument("--routing-only", action="store_true", help="resolve routing, execute nothing")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)

    queries = load_queries(args.queries) if args.queries else list(CANONICAL_QUERIES)
    if not queries:
        print("No queries to run.", file=sys.stderr)
        return 2
    try:
        hestia = boot_hestia(args.config)
    except Exception as exc:                           # noqa: BLE001
        print(f"Hestia would not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        if args.routing_only:
            results = run_routing_only(hestia.resolve_only, queries)
        else:
            results = run_queries(hestia.process_text, queries, max_seconds=args.max_seconds)
    finally:
        shutdown = getattr(hestia, "_shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:                          # noqa: BLE001
                pass
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print(format_report(results))
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
