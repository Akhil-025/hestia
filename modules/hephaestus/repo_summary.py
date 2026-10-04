# modules/hephaestus/repo_summary.py
"""
Local code-repository summary for Hephaestus (backlog #110).

Point it at a folder and get the shape of the project: languages and size,
where the code lives, entry points and manifests, how much of it has tests,
and a short list of things worth a look (very long functions and files,
TODO/FIXME markers, Python files that don't parse, bare ``except:``).

It is a deterministic scan, not an LLM review: every number can be checked,
and nothing in the output is file *content* — only names, counts and line
numbers. It never follows symlinks, skips dependency/build folders, and caps
how much it reads, so pointing it at a huge folder is safe (``truncated`` says
so when a cap was hit).
"""
from __future__ import annotations

import ast
import os
import re
from collections import Counter
from typing import Any

_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".idea", ".vscode", ".mypy_cache", ".pytest_cache", ".tox",
    ".next", "target", "site-packages", ".cache", ".ruff_cache",
})
_LANGS = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".java": "Java", ".kt": "Kotlin", ".go": "Go", ".rs": "Rust",
    ".c": "C", ".h": "C", ".cpp": "C++", ".hpp": "C++", ".cs": "C#", ".rb": "Ruby",
    ".php": "PHP", ".swift": "Swift", ".sh": "Shell", ".ps1": "PowerShell",
    ".html": "HTML", ".css": "CSS", ".sql": "SQL", ".yaml": "YAML", ".yml": "YAML",
    ".json": "JSON", ".toml": "TOML", ".md": "Markdown",
}
# Config/data/docs languages are listed but not counted as "code" for ratios.
_NON_CODE = frozenset({"YAML", "JSON", "TOML", "Markdown", "HTML", "CSS"})
_MANIFESTS = (
    "requirements.txt", "pyproject.toml", "setup.py", "Pipfile", "package.json",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "Gemfile", "composer.json",
    "Dockerfile", "docker-compose.yml", "Makefile",
)
_ENTRY_NAMES = frozenset({
    "main.py", "__main__.py", "app.py", "manage.py", "cli.py", "run.py", "server.py",
    "index.js", "main.js", "server.js", "main.go", "main.rs", "Program.cs",
})
_MARKER = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b")
_MAX_FILES = 20_000
_MAX_READ_BYTES = 1_000_000
LONG_FUNCTION_LINES = 80
LONG_FILE_LINES = 1000


def _is_test_file(rel: str) -> bool:
    parts = rel.lower().replace("\\", "/").split("/")
    base = parts[-1]
    return (
        any(p in ("tests", "test", "__tests__", "spec") for p in parts[:-1])
        or base.startswith("test_") or base.endswith(("_test.py", ".test.js", ".test.ts", ".spec.js", ".spec.ts"))
    )


def _python_findings(source: str, rel: str, out: dict[str, Any]) -> None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        out["unparsable"].append(rel)
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1
            if length >= LONG_FUNCTION_LINES:
                out["long_functions"].append((length, f"{rel}:{node.lineno} {node.name}()"))
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            out["bare_excepts"] += 1


def summarize_repo(path: str) -> dict[str, Any]:
    """Scan *path* and return a plain dict (see ``format_summary``).

    Raises ``NotADirectoryError`` if *path* isn't a directory."""
    root = os.path.realpath(path)
    if not os.path.isdir(root):
        raise NotADirectoryError(path)

    lang_files: Counter = Counter()
    lang_lines: Counter = Counter()
    top_dirs: Counter = Counter()
    manifests: list[str] = []
    entry_points: list[str] = []
    sizes: list[tuple[int, str]] = []
    markers: Counter = Counter()
    py = {"long_functions": [], "bare_excepts": 0, "unparsable": []}
    code_files = test_files = total_files = 0
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in _SKIP_DIRS and not os.path.islink(os.path.join(dirpath, d)))
        for fname in sorted(filenames):
            full = os.path.join(dirpath, fname)
            if os.path.islink(full):
                continue
            total_files += 1
            if total_files > _MAX_FILES:
                truncated = True
                break
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if fname in _MANIFESTS and rel.count("/") <= 1:
                manifests.append(rel)
            if fname in _ENTRY_NAMES and rel.count("/") <= 1:
                entry_points.append(rel)
            lang = _LANGS.get(os.path.splitext(fname)[1].lower())
            if not lang:
                continue
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            lang_files[lang] += 1
            top = rel.split("/")[0] if "/" in rel else "(root)"
            if lang not in _NON_CODE:
                code_files += 1
                test_files += _is_test_file(rel)
            if size > _MAX_READ_BYTES:
                continue
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    source = fh.read()
            except OSError:
                continue
            lines = source.count("\n") + (1 if source and not source.endswith("\n") else 0)
            nonblank = sum(1 for ln in source.splitlines() if ln.strip())
            lang_lines[lang] += nonblank
            if lang not in _NON_CODE:
                top_dirs[top] += nonblank
                sizes.append((lines, rel))
                markers.update(m.group(1) for m in _MARKER.finditer(source))
            if lang == "Python":
                _python_findings(source, rel, py)
        if truncated:
            break

    code_lines = sum(n for lang, n in lang_lines.items() if lang not in _NON_CODE)
    sizes.sort(reverse=True)
    return {
        "path": root,
        "truncated": truncated,
        "files_scanned": min(total_files, _MAX_FILES),
        "code_files": code_files,
        "test_files": test_files,
        "code_lines": code_lines,
        "languages": [(l, lang_files[l], lang_lines[l]) for l, _ in lang_lines.most_common()],
        "top_dirs": top_dirs.most_common(6),
        "manifests": manifests,
        "entry_points": entry_points,
        "largest_files": [(n, r) for n, r in sizes[:5]],
        "long_files": [(n, r) for n, r in sizes if n >= LONG_FILE_LINES][:5],
        "long_functions": sorted(py["long_functions"], reverse=True)[:5],
        "bare_excepts": py["bare_excepts"],
        "unparsable": py["unparsable"][:5],
        "markers": dict(markers),
    }


def format_summary(s: dict[str, Any], name: str = "") -> str:
    """Short spoken-style summary of a ``summarize_repo`` result."""
    label = name or os.path.basename(s["path"].rstrip("/\\")) or s["path"]
    if not s["code_files"]:
        return f"I didn't find any source code in {label}."
    code_langs = [(l, f, n) for l, f, n in s["languages"] if l not in _NON_CODE]
    langs = ", ".join(f"{l} ({n:,} lines)" for l, _f, n in code_langs[:3])
    parts = [
        f"{label}: {s['code_files']:,} code files, about {s['code_lines']:,} lines — mostly {langs}."
    ]
    if s["top_dirs"]:
        parts.append("Most of the code is in " + ", ".join(
            f"{d} ({n:,})" for d, n in s["top_dirs"][:3]) + ".")
    if s["entry_points"] or s["manifests"]:
        bits = []
        if s["entry_points"]:
            bits.append("entry points " + ", ".join(s["entry_points"][:3]))
        if s["manifests"]:
            bits.append("manifests " + ", ".join(s["manifests"][:4]))
        parts.append("It has " + " and ".join(bits) + ".")
    ratio = s["test_files"] / s["code_files"]
    parts.append(
        f"{s['test_files']:,} of the code files are tests ({ratio:.0%})."
        if s["test_files"] else "I found no test files."
    )
    flags = []
    if s["long_functions"]:
        n, where = s["long_functions"][0]
        flags.append(f"{len(s['long_functions'])} very long function(s), the longest {where} at {n} lines")
    if s["long_files"]:
        n, where = s["long_files"][0]
        flags.append(f"{len(s['long_files'])} file(s) over {LONG_FILE_LINES} lines, the largest {where} ({n:,})")
    if s["bare_excepts"]:
        flags.append(f"{s['bare_excepts']} bare except clause(s)")
    if s["unparsable"]:
        flags.append(f"{len(s['unparsable'])} Python file(s) that don't parse, e.g. {s['unparsable'][0]}")
    if s["markers"]:
        flags.append(", ".join(f"{n} {k}" for k, n in sorted(s["markers"].items())))
    if flags:
        parts.append("Worth a look: " + "; ".join(flags) + ".")
    if s["truncated"]:
        parts.append(f"The folder was very large, so I stopped after {s['files_scanned']:,} files.")
    return " ".join(parts)
