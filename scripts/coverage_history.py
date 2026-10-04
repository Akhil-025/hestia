"""
scripts/coverage_history.py

Coverage reporting for ``run_tests.py --coverage`` (backlog #207).

Two jobs:

1. Turn per-file (statements, missing) numbers into a short report: the total,
   the change since the last recorded run, and the least-covered files.
2. Append each run to a small CSV (``tests/coverage_history.csv`` by default)
   so the trend is visible in ``git log`` / a spreadsheet and a drop gets
   called out the next time someone runs the suite.

Everything here is pure or takes its collaborators as arguments, so it is
tested without ``coverage`` installed (tests/test_qa_tools.py). Only
``run_with_coverage`` touches the real library, and it takes the module as a
parameter for the same reason.

What the number means: *line* coverage of ``core/``, ``modules/`` and
``main.py`` while the test suite runs. It says which lines never executed, not
whether the tests assert anything useful about the ones that did; that is what
``scripts/mutation_check.py`` (#214) is for.
"""
from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

HISTORY_FIELDS = ("date", "commit", "percent", "statements", "covered", "files", "tests_ok")
DEFAULT_SOURCES = ("core", "modules", "main.py")
DEFAULT_OMIT = ("*/tests/*", "*/__pycache__/*", "*/_stubs/*")
# A fall of more than this many percentage points since the last run is called out.
REGRESSION_POINTS = 1.0


@dataclass(frozen=True)
class Summary:
    percent: float
    statements: int
    covered: int
    files: int
    worst: tuple[tuple[str, float, int], ...]   # (file, percent, lines not run)


def summarise(per_file: dict[str, tuple[int, int]], worst_n: int = 5) -> Summary:
    """``per_file`` maps path -> ``(statements, missing)``.

    Files with no statements (empty ``__init__.py``) are left out of the file
    count and the ranking: they can't be covered or uncovered.
    """
    rows = {f: (s, m) for f, (s, m) in per_file.items() if s > 0}
    statements = sum(s for s, _ in rows.values())
    missing = sum(min(m, s) for s, m in rows.values())
    covered = statements - missing
    percent = round(100.0 * covered / statements, 2) if statements else 100.0
    ranked = sorted(
        ((f, round(100.0 * (s - min(m, s)) / s, 1), min(m, s)) for f, (s, m) in rows.items()),
        key=lambda r: (-r[2], r[1], r[0]),          # most uncovered lines first
    )
    return Summary(percent, statements, covered, len(rows), tuple(ranked[:worst_n]))


# ---------------------------------------------------------------------------
# History file
# ---------------------------------------------------------------------------

def read_history(path: str) -> list[dict[str, str]]:
    """All recorded rows, oldest first. A missing or unreadable file is empty."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return [r for r in csv.DictReader(fh) if r.get("percent")]
    except (OSError, csv.Error):
        return []


def append_history(path: str, summary: Summary, *, commit: str = "", tests_ok: bool = True,
                   when: Optional[datetime] = None) -> None:
    """Add one row, writing the header first if the file is new."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M")
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(HISTORY_FIELDS)
        writer.writerow([stamp, commit, f"{summary.percent:.2f}", summary.statements,
                         summary.covered, summary.files, int(tests_ok)])


def previous_percent(history: list[dict[str, str]]) -> Optional[float]:
    for row in reversed(history):
        try:
            return float(row["percent"])
        except (KeyError, ValueError):
            continue
    return None


def describe_change(previous: Optional[float], now: float) -> str:
    """One line comparing this run with the last recorded one."""
    if previous is None:
        return "First recorded run, nothing to compare with yet."
    delta = round(now - previous, 2)
    if abs(delta) < 0.005:
        return f"Unchanged since the last run ({previous:.2f}%)."
    arrow = "Up" if delta > 0 else "Down"
    text = f"{arrow} {abs(delta):.2f} points from {previous:.2f}%."
    if delta < -REGRESSION_POINTS:
        text += f" That is a drop of more than {REGRESSION_POINTS:g} point(s): check what lost coverage."
    return text


def format_report(summary: Summary, change: str) -> str:
    lines = [
        "",
        f"Coverage: {summary.percent:.2f}%  ({summary.covered:,} of {summary.statements:,} "
        f"statements, {summary.files} files)",
        change,
    ]
    if summary.worst:
        lines.append("Most uncovered lines:")
        lines += [f"  {f}  {pct:.1f}%  ({miss:,} lines not run)" for f, pct, miss in summary.worst]
    return "\n".join(lines)


def git_commit(root: str) -> str:
    """Short hash of HEAD, or ``""`` outside a git checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def split_cov_args(argv: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Pull this script's flags out of *argv*; everything else is for pytest.

    ``--coverage``            measure and report
    ``--no-record``           don't append to the history file
    ``--cov-fail-under N``    exit non-zero when coverage is below N percent
    """
    opts: dict[str, Any] = {"coverage": False, "record": True, "fail_under": None}
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--coverage":
            opts["coverage"] = True
        elif a == "--no-record":
            opts["record"] = False
        elif a == "--cov-fail-under" and i + 1 < len(argv):
            opts["fail_under"] = float(argv[i + 1])
            opts["coverage"] = True
            i += 1
        elif a.startswith("--cov-fail-under="):
            opts["fail_under"] = float(a.split("=", 1)[1])
            opts["coverage"] = True
        else:
            rest.append(a)
        i += 1
    return opts, rest


def iter_source_files(root: str, sources: Iterable[str] = DEFAULT_SOURCES) -> list[str]:
    """Every ``.py`` file under the measured sources, so a module no test ever
    imports counts as 0% instead of silently not existing."""
    found: list[str] = []
    for src in sources:
        path = os.path.join(root, src)
        if os.path.isfile(path) and path.endswith(".py"):
            found.append(path)
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "_stubs", "tests")]
            found += [os.path.join(dirpath, f) for f in filenames if f.endswith(".py")]
    return sorted(set(found))


def collect_per_file(cov: Any, files: Iterable[str], root: str) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for f in files:
        try:
            _name, statements, _excluded, missing, _text = cov.analysis2(f)
        except Exception:          # a file coverage can't parse (e.g. a syntax error)
            continue
        out[os.path.relpath(f, root).replace(os.sep, "/")] = (len(statements), len(missing))
    return out


def run_with_coverage(
    pytest_args: list[str],
    *,
    pytest_main: Callable[[list[str]], int],
    coverage_module: Any,
    root: str,
    history_path: str,
    record: bool = True,
    fail_under: Optional[float] = None,
    printer: Callable[[str], None] = print,
) -> int:
    """Run the suite under coverage, print a report, optionally record it."""
    cov = coverage_module.Coverage(
        source=[os.path.join(root, s) for s in DEFAULT_SOURCES if os.path.exists(os.path.join(root, s))],
        omit=list(DEFAULT_OMIT),
    )
    cov.start()
    try:
        code = int(pytest_main(pytest_args))
    finally:
        cov.stop()
    try:
        cov.save()
    except Exception:
        pass
    measured = set(cov.get_data().measured_files())
    per_file = collect_per_file(cov, sorted(measured | set(iter_source_files(root))), root)
    summary = summarise(per_file)
    history = read_history(history_path)
    printer(format_report(summary, describe_change(previous_percent(history), summary.percent)))
    if record:
        append_history(history_path, summary, commit=git_commit(root), tests_ok=(code == 0))
        printer(f"Recorded in {os.path.relpath(history_path, root)}.")
    if fail_under is not None and summary.percent < fail_under:
        printer(f"Coverage {summary.percent:.2f}% is below the required {fail_under:g}%.")
        return code or 2
    return code
