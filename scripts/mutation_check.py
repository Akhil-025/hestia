"""
scripts/mutation_check.py

A small mutation tester for Hestia's critical logic (backlog #214).

A test that passes proves little if it would *still* pass after the code it
covers was broken. This script breaks the code on purpose, one change at a
time ("a mutant"): flip a comparison, swap and/or, drop a `not`, nudge a
number, make a function return None. Then it runs the tests. A mutant the
tests catch is *killed*; one that slips through is a *survivor*, and each
survivor names a line your tests don't actually pin down.

    python scripts/mutation_check.py modules/hecate/engine.py \\
        --tests tests/test_hecate.py tests/test_registry_contract.py \\
                tests/test_hecate_routing_tiers.py
    python scripts/mutation_check.py core/intent_aliases.py --tests tests/test_intent_aliases.py --max 40
    python scripts/mutation_check.py FILE --list          # show the mutants, run nothing

Options: ``--max N`` (random sample of N mutants, ``--seed`` fixes it),
``--min-score 0.8`` (exit 1 below that kill rate), ``--runner CMD`` (default
``python -m pytest -x -q``; the test files are appended).

Each mutant's cached bytecode is deleted before its test run (see
``drop_bytecode``) so the tests really execute the mutated source.

Safety: the file is changed *in place* while mutants run. The original is first
copied to ``<file>.mutbak`` and restored in a ``finally`` block, and if a
previous run was killed mid-way the leftover backup is restored before
anything else happens. Don't edit the file while this runs, and run it on a
clean git tree so ``git diff`` stays a second safety net. Mutants are written
with ``ast.unparse``, so comments and layout are lost while a mutant is live;
the original text is what gets restored.

Not every survivor is a gap: some mutants are equivalent (a changed number in a
log message, a swapped comparison that can't change behaviour). Read the list
rather than chasing 100%. Mark a line ``# pragma: no mutate`` to exclude it.
Docstrings, logging calls and ``__repr__``/``__str__`` bodies are skipped
automatically. Timeouts count as killed, since an infinite loop is a change a
test run would notice.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import random
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Optional

BACKUP_SUFFIX = ".mutbak"

_COMPARE_SWAP = {
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE, ast.LtE: ast.Gt,
    ast.Gt: ast.LtE, ast.GtE: ast.Lt, ast.In: ast.NotIn, ast.NotIn: ast.In,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is,
}
_BINOP_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add}
_LOG_NAMES = frozenset({"logger", "log", "logging", "print"})


@dataclass(frozen=True)
class Mutant:
    index: int
    lineno: int
    description: str


# ---------------------------------------------------------------------------
# Finding and applying mutations
# ---------------------------------------------------------------------------

def _skipped_lines(tree: ast.AST, source: str) -> set[int]:
    """Lines never mutated: docstrings, logging/print calls, __repr__, pragmas."""
    skip: set[int] = set()
    for i, line in enumerate(source.splitlines(), 1):
        if "pragma: no mutate" in line:
            skip.add(i)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                skip.update(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
            if isinstance(node, ast.FunctionDef) and node.name in ("__repr__", "__str__"):
                skip.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            base = func.value if isinstance(func, ast.Attribute) else func
            if isinstance(base, ast.Name) and base.id in _LOG_NAMES:
                skip.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return skip


class _Mutator(ast.NodeTransformer):
    """Walks the tree in a fixed order. With ``target=None`` it only counts
    candidates; with a target index it applies exactly that one change."""

    def __init__(self, skip: set[int], target: Optional[int] = None) -> None:
        self.skip = skip
        self.target = target
        self.count = 0
        self.found: list[Mutant] = []

    def _hit(self, node: ast.AST, description: str) -> bool:
        if getattr(node, "lineno", 0) in self.skip:
            return False
        idx = self.count
        self.count += 1
        self.found.append(Mutant(idx, node.lineno, description))
        return idx == self.target

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        for i, op in enumerate(node.ops):
            swap = _COMPARE_SWAP.get(type(op))
            if swap and self._hit(node, f"{type(op).__name__} -> {swap.__name__}"):
                node.ops[i] = swap()
        return node

    def visit_BoolOp(self, node: ast.BoolOp):
        self.generic_visit(node)
        new = ast.Or if isinstance(node.op, ast.And) else ast.And
        if self._hit(node, f"{type(node.op).__name__} -> {new.__name__}"):
            node.op = new()
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and self._hit(node, "removed `not`"):
            return node.operand
        return node

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        swap = _BINOP_SWAP.get(type(node.op))
        if swap and self._hit(node, f"{type(node.op).__name__} -> {swap.__name__}"):
            node.op = swap()
        return node

    def visit_Constant(self, node: ast.Constant):
        v = node.value
        if isinstance(v, bool):
            if self._hit(node, f"{v} -> {not v}"):
                return ast.copy_location(ast.Constant(value=not v), node)
        elif isinstance(v, (int, float)) and abs(v) < 1e9:
            if self._hit(node, f"{v!r} -> {v + 1!r}"):
                return ast.copy_location(ast.Constant(value=v + 1), node)
        return node

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        if self._hit(node, "negated `if` condition"):
            node.test = ast.copy_location(ast.UnaryOp(op=ast.Not(), operand=node.test), node.test)
        return node

    def visit_Return(self, node: ast.Return):
        self.generic_visit(node)
        v = node.value
        if v is not None and not (isinstance(v, ast.Constant) and v.value is None):
            if self._hit(node, "return value -> None"):
                node.value = ast.copy_location(ast.Constant(value=None), v)
        return node


def list_mutants(source: str) -> list[Mutant]:
    tree = ast.parse(source)
    m = _Mutator(_skipped_lines(tree, source))
    m.visit(tree)
    return m.found


def apply_mutant(source: str, index: int) -> str:
    """Source with exactly mutant *index* applied. Raises ``IndexError`` if
    there is no such mutant, ``SyntaxError`` if the result doesn't parse."""
    tree = ast.parse(source)
    m = _Mutator(_skipped_lines(tree, source), target=index)
    tree = m.visit(tree)
    if index >= m.count:
        raise IndexError(index)
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)
    compile(out, "<mutant>", "exec")
    return out


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def recover_backup(path: str) -> bool:
    """Restore *path* from a leftover backup (a previous run was killed)."""
    bak = path + BACKUP_SUFFIX
    if os.path.exists(bak):
        shutil.copyfile(bak, path)
        os.remove(bak)
        return True
    return False


def drop_bytecode(path: str) -> None:
    """Delete *path*'s cached bytecode.

    Python trusts a ``.pyc`` whose recorded mtime (whole seconds) and size match
    the source. Mutants are written within the same second, and many have the
    same size as the file they replace (``1.0`` -> ``2.0``), so without this a
    run can silently execute the *previous* mutant or the original: false kills
    and false survivors alike.
    """
    try:
        os.remove(importlib.util.cache_from_source(path))
    except (OSError, ValueError):
        pass


def subprocess_runner(command: list[str], tests: list[str], timeout: float) -> Callable[[], bool]:
    """A callable returning True when the tests PASS (the mutant survived)."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def run() -> bool:
        try:
            proc = subprocess.run(command + tests, capture_output=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return False
        return proc.returncode == 0
    return run


@dataclass
class Result:
    total: int
    tested: int
    killed: int
    survivors: list[tuple[Mutant, str]]
    baseline_ok: bool = True

    @property
    def score(self) -> float:
        return self.killed / self.tested if self.tested else 1.0


def run_mutants(
    path: str,
    runner: Callable[[], bool],
    *,
    max_mutants: Optional[int] = None,
    seed: int = 0,
    progress: Callable[[str], None] = lambda s: None,
) -> Result:
    """Mutate *path* one change at a time, calling ``runner()`` after each.

    ``runner()`` must return True when the test suite passes. The unmutated
    file is checked first: if its tests already fail there is nothing to
    measure and ``baseline_ok`` is False. The file is always restored.
    """
    recover_backup(path)
    with open(path, encoding="utf-8", newline="") as fh:
        original = fh.read()
    normalised = original.replace("\r\n", "\n")
    mutants = list_mutants(normalised)
    drop_bytecode(path)
    if not runner():
        return Result(len(mutants), 0, 0, [], baseline_ok=False)
    chosen = mutants
    if max_mutants is not None and len(mutants) > max_mutants:
        chosen = sorted(random.Random(seed).sample(mutants, max_mutants), key=lambda m: m.index)
    lines = normalised.splitlines()
    killed = 0
    survivors: list[tuple[Mutant, str]] = []
    bak = path + BACKUP_SUFFIX
    shutil.copyfile(path, bak)
    try:
        for n, m in enumerate(chosen, 1):
            try:
                mutated = apply_mutant(normalised, m.index)
            except (SyntaxError, IndexError, ValueError, RecursionError):
                continue            # couldn't build a valid mutant; not counted
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(mutated)
            drop_bytecode(path)
            survived = runner()
            progress(f"[{n}/{len(chosen)}] line {m.lineno} {m.description}: "
                     f"{'SURVIVED' if survived else 'killed'}")
            if survived:
                src = lines[m.lineno - 1].strip() if 0 < m.lineno <= len(lines) else ""
                survivors.append((m, src))
            else:
                killed += 1
    finally:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(original)
        drop_bytecode(path)
        if os.path.exists(bak):
            os.remove(bak)
    return Result(len(mutants), killed + len(survivors), killed, survivors)


def format_result(path: str, r: Result) -> str:
    if not r.baseline_ok:
        return (f"{path}: the tests already fail on the unmodified file, so there is "
                "nothing to measure. Fix them first.")
    lines = [f"{path}: {r.killed} of {r.tested} mutants killed "
             f"({r.score:.0%}); {r.total} possible, {r.tested} tried."]
    if r.survivors:
        lines.append("Survivors (changes the tests did not notice):")
        for m, src in r.survivors:
            lines.append(f"  line {m.lineno}: {m.description}    {src[:90]}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Mutation-test one source file.")
    ap.add_argument("file")
    ap.add_argument("--tests", nargs="+", default=["tests"], help="test files/dirs to run")
    ap.add_argument("--runner", default=f"{sys.executable} -m pytest -x -q -p no:cacheprovider",
                    help="test command; the test paths are appended")
    ap.add_argument("--max", type=int, default=None, help="try at most N mutants (random sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per test run")
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--list", action="store_true", help="list mutants and exit")
    args = ap.parse_args(argv)

    path = os.path.abspath(args.file)
    if not os.path.isfile(path):
        print(f"No such file: {args.file}", file=sys.stderr)
        return 2
    if recover_backup(path):
        print(f"Restored {args.file} from a leftover backup of an interrupted run.", file=sys.stderr)
    if args.list:
        with open(path, encoding="utf-8") as fh:
            for m in list_mutants(fh.read().replace("\r\n", "\n")):
                print(f"#{m.index:<4} line {m.lineno:<5} {m.description}")
        return 0
    runner = subprocess_runner(shlex.split(args.runner), args.tests, args.timeout)
    result = run_mutants(path, runner, max_mutants=args.max, seed=args.seed,
                         progress=lambda s: print(s, flush=True))
    print(format_result(args.file, result))
    if not result.baseline_ok:
        return 2
    if args.min_score is not None and result.score < args.min_score:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
