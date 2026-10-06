# modules/hephaestus/repo_summary.py
"""
Local code-repository summary for Hephaestus (backlog #110).

Point it at a folder and get the shape of the project: languages and size,
where the code lives, entry points and manifests, how much of it has tests,
and a short list of things worth a look.

The scan is deterministic and offline. Every number can be checked, and the
summary holds only names, counts and line numbers, never file *content*. What
it looks for depends on the language:

* Python: very long functions, files that don't parse, bare ``except:``,
  ``eval``/``exec`` calls and mutable default arguments (parsed with ``ast``).
* JavaScript/TypeScript, Go and Rust: very long functions (found by brace
  matching after comments and strings are blanked out), plus empty ``catch``
  blocks, ``eval``, ``any``/``@ts-ignore`` (TS), ``panic`` (Go) and
  ``unwrap``/``unsafe`` (Rust). These are heuristics, not a parser: a file
  whose braces don't balance after blanking is skipped for function lengths
  rather than guessed at.
* Every language: TODO/FIXME markers, long files, test ratio.
* Dependencies: package counts from requirements.txt, package.json, go.mod
  and Cargo.toml, and how many Python requirements are unpinned.

``build_review_prompt`` builds the *optional* LLM review (see the engine's
``repo_review`` setting). That is the one path that reads file content, and
only a few redacted, size-capped excerpts of the files the scan flagged.

It never follows symlinks, skips dependency/build folders, and caps how much
it reads, so pointing it at a huge folder is safe (``truncated`` says so when a
cap was hit).
"""
from __future__ import annotations

import ast
import json
import os
import re
from collections import Counter
from typing import Any, Optional

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

# Languages whose function length is measured by brace matching.
_BRACE_LANGS = frozenset({"JavaScript", "TypeScript", "Go", "Rust"})
_JS_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "catch", "function", "return", "else", "do",
    "with", "try", "super", "constructor_",
})
_JS_FUNC = re.compile(
    r"\bfunction\s*\*?\s*([A-Za-z_$][\w$]*)\s*\("
    r"|\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?"
    r"(?:function\b|\([^)]*\)\s*(?::\s*[^={;]+)?=>|[A-Za-z_$][\w$]*\s*=>)"
    r"|^[ \t]*(?:(?:public|private|protected|static|async|override|get|set)\s+)*"
    r"([A-Za-z_$][\w$]*)\s*\([^()]*\)\s*(?::\s*[^{;=]+)?\{",
    re.M,
)
_GO_FUNC = re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\(", re.M)
_RUST_FUNC = re.compile(r"\bfn\s+([A-Za-z_]\w*)")
_FUNC_PATTERNS = {
    "JavaScript": _JS_FUNC, "TypeScript": _JS_FUNC, "Go": _GO_FUNC, "Rust": _RUST_FUNC,
}
_EMPTY_CATCH = re.compile(r"\bcatch\s*(?:\([^)]*\))?\s*\{\s*\}")
_EVAL_CALL = re.compile(r"(?<![\w$.])eval\s*\(")
_TS_ANY = re.compile(r":\s*any\b")
_TS_SUPPRESS = re.compile(r"@ts-(?:ignore|nocheck)\b")
_RUST_UNWRAP = re.compile(r"\.unwrap\(\)")
_RUST_UNSAFE = re.compile(r"\bunsafe\s*(?:\{|fn\b|impl\b)")
_GO_PANIC = re.compile(r"(?<![\w.])panic\(")
_MAX_REVIEW_FILE_LINES = 60
_SECRET_LINE = re.compile(
    r"(?i)(api[_-]?key|secret|token|passw(or)?d|passwd|bearer|private[_-]?key|"
    r"authorization|aws_access|client[_-]?secret)\s*[:=]|-----BEGIN [A-Z ]*PRIVATE KEY"
)


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
            defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
            if any(isinstance(d, (ast.List, ast.Dict, ast.Set)) for d in defaults):
                out["mutable_defaults"] += 1
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            out["bare_excepts"] += 1
        elif (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("eval", "exec")
        ):
            out["dynamic_exec"] += 1


def _blank_noncode(source: str, lang: str) -> str:
    """Return *source* with comments and the insides of string literals
    replaced by spaces (newlines kept), so braces and keywords in them can't
    confuse brace matching. Line numbers and offsets are preserved.

    Handles ``//`` and ``/* */`` comments and ``'``, ``"`` and backtick strings.
    In Rust a ``'`` is a string delimiter only when it closes within a few
    characters (``'x'``, ``'\\n'``), since ``'a`` is a lifetime. A JavaScript
    regex literal containing a quote can still fool this; the caller treats an
    unbalanced result as "can't tell" and skips function lengths for the file.
    """
    out = []
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = source.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and nxt == "*":
            j = source.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in source[i:j]))
            i = j
        elif c in "\"`" or (c == "'" and (lang != "Rust" or _rust_char_at(source, i))):
            quote = c
            j = i + 1
            while j < n:
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == quote or (source[j] == "\n" and quote != "`"):
                    break  # closing quote, or an unterminated plain string ending at the line
                j += 1
            j = min(j, n)
            closed = j < n and source[j] == quote
            out.append(quote + "".join(ch if ch == "\n" else " " for ch in source[i + 1:j])
                       + (quote if closed else ""))
            i = j + 1 if closed else j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _rust_char_at(source: str, i: int) -> bool:
    """True if the ``'`` at *i* starts a Rust char literal rather than a lifetime."""
    return bool(re.match(r"'(?:\\.|[^'\\\n])'", source[i:i + 6]))


def _brace_function_lengths(clean: str, lang: str) -> Optional[list[tuple[int, int, str]]]:
    """``[(length_in_lines, start_line, name), ...]`` for each function found
    in comment/string-blanked source, or None if the braces don't balance."""
    if clean.count("{") != clean.count("}"):
        return None
    pattern = _FUNC_PATTERNS[lang]
    found: list[tuple[int, int, str]] = []
    for m in pattern.finditer(clean):
        name = next((g for g in m.groups() if g), "")
        if not name or name in _JS_KEYWORDS:
            continue
        open_at = clean.find("{", m.end() - 1 if m.group(0).endswith("{") else m.end())
        if open_at == -1:
            continue
        # A ';' before the brace means a declaration with no body (Rust trait
        # method, Go interface, TS overload), not a function to measure.
        between = clean[m.end():open_at] if open_at >= m.end() else ""
        if ";" in between:
            continue
        depth, j = 0, open_at
        while j < len(clean):
            ch = clean[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            return None
        start_line = clean.count("\n", 0, m.start()) + 1
        end_line = clean.count("\n", 0, j) + 1
        found.append((end_line - start_line + 1, start_line, name))
    return found


def _brace_findings(source: str, rel: str, lang: str, out: dict[str, Any], is_test: bool) -> None:
    """JavaScript/TypeScript, Go and Rust findings (see the module docstring)."""
    clean = _blank_noncode(source, lang)
    lengths = _brace_function_lengths(clean, lang)
    if lengths is None:
        out["unbalanced"].append(rel)
    else:
        for length, line, name in lengths:
            if length >= LONG_FUNCTION_LINES:
                out["long_functions"].append((length, f"{rel}:{line} {name}()"))
    if lang in ("JavaScript", "TypeScript"):
        out["empty_catches"] += len(_EMPTY_CATCH.findall(clean))
        out["dynamic_exec"] += len(_EVAL_CALL.findall(clean))
        if lang == "TypeScript":
            out["ts_any"] += len(_TS_ANY.findall(clean))
            out["ts_suppressions"] += len(_TS_SUPPRESS.findall(source))
    elif lang == "Go" and not is_test:
        out["go_panics"] += len(_GO_PANIC.findall(clean))
    elif lang == "Rust" and not is_test:
        out["rust_unwraps"] += len(_RUST_UNWRAP.findall(clean))
        out["rust_unsafe"] += len(_RUST_UNSAFE.findall(clean))


def _dependency_counts(root: str, manifests: list[str]) -> dict[str, dict[str, int]]:
    """Package counts from the manifests found near the top of the project.
    Only counts are returned, never package names or versions."""
    out: dict[str, dict[str, int]] = {}
    for rel in manifests:
        base = os.path.basename(rel)
        full = os.path.join(root, rel)
        try:
            if os.path.getsize(full) > _MAX_READ_BYTES:
                continue
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if base == "requirements.txt":
            lines = [ln.split("#", 1)[0].strip() for ln in text.splitlines()]
            pkgs = [ln for ln in lines if ln and not ln.startswith(("-", "git+", "http"))]
            unpinned = [p for p in pkgs if not re.search(r"(==|~=|>=|<=|<|>|@)", p)]
            out[rel] = {"packages": len(pkgs), "unpinned": len(unpinned)}
        elif base == "package.json":
            try:
                data = json.loads(text)
            except ValueError:
                continue
            if isinstance(data, dict):
                runtime = data.get("dependencies") or {}
                dev = data.get("devDependencies") or {}
                out[rel] = {
                    "packages": len(runtime) if isinstance(runtime, dict) else 0,
                    "dev_packages": len(dev) if isinstance(dev, dict) else 0,
                }
        elif base == "go.mod":
            block = re.findall(r"^\s*require\s*\((.*?)\)", text, re.S | re.M)
            n = sum(1 for body in block for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("//"))
            n += len(re.findall(r"^\s*require\s+[^(\s]", text, re.M))
            out[rel] = {"packages": n}
        elif base == "Cargo.toml":
            m = re.search(r"^\[dependencies\]\s*$(.*?)(?=^\[|\Z)", text, re.S | re.M)
            n = sum(1 for ln in (m.group(1).splitlines() if m else [])
                    if re.match(r"^\s*[A-Za-z0-9_-]+\s*=", ln))
            out[rel] = {"packages": n}
    return out


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
    py = {
        "long_functions": [], "bare_excepts": 0, "unparsable": [],
        "mutable_defaults": 0, "dynamic_exec": 0, "unbalanced": [],
        "empty_catches": 0, "ts_any": 0, "ts_suppressions": 0,
        "go_panics": 0, "rust_unwraps": 0, "rust_unsafe": 0,
    }
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
            elif lang in _BRACE_LANGS:
                _brace_findings(source, rel, lang, py, _is_test_file(rel))
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
        "mutable_defaults": py["mutable_defaults"],
        "dynamic_exec": py["dynamic_exec"],
        "empty_catches": py["empty_catches"],
        "ts_any": py["ts_any"],
        "ts_suppressions": py["ts_suppressions"],
        "go_panics": py["go_panics"],
        "rust_unwraps": py["rust_unwraps"],
        "rust_unsafe": py["rust_unsafe"],
        "unbalanced": py["unbalanced"][:5],
        "dependencies": _dependency_counts(root, manifests),
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
    for key, text in (
        ("dynamic_exec", "{n} eval/exec call(s)"),
        ("mutable_defaults", "{n} function(s) with a mutable default argument"),
        ("empty_catches", "{n} empty catch block(s)"),
        ("ts_any", "{n} use(s) of the TypeScript any type"),
        ("ts_suppressions", "{n} @ts-ignore/@ts-nocheck comment(s)"),
        ("go_panics", "{n} panic call(s) in Go code"),
        ("rust_unwraps", "{n} Rust unwrap() call(s)"),
        ("rust_unsafe", "{n} Rust unsafe block(s)"),
    ):
        if s.get(key):
            flags.append(text.format(n=s[key]))
    unpinned = sum(d.get("unpinned", 0) for d in (s.get("dependencies") or {}).values())
    if unpinned:
        flags.append(f"{unpinned} Python requirement(s) with no version pinned")
    if s["markers"]:
        flags.append(", ".join(f"{n} {k}" for k, n in sorted(s["markers"].items())))
    deps = s.get("dependencies") or {}
    if deps:
        parts.append("Dependencies: " + ", ".join(
            f"{d.get('packages', 0)} in {rel}" for rel, d in list(deps.items())[:3]) + ".")
    if flags:
        parts.append("Worth a look: " + "; ".join(flags) + ".")
    if s.get("unbalanced"):
        parts.append(
            f"I couldn't measure function lengths in {len(s['unbalanced'])} file(s) "
            "(unbalanced braces), e.g. " + s["unbalanced"][0] + "."
        )
    if s["truncated"]:
        parts.append(f"The folder was very large, so I stopped after {s['files_scanned']:,} files.")
    return " ".join(parts)


# ----------------------------------------------------------------------------
# Optional LLM review (backlog #110)
# ----------------------------------------------------------------------------

def _redact(line: str) -> str:
    """Blank a line that looks like it assigns a secret."""
    return "[line removed: looks like a secret]" if _SECRET_LINE.search(line) else line


def _review_files(root: str, summary: dict[str, Any], max_files: int) -> list[str]:
    """Relative paths worth showing a reviewer, most interesting first: entry
    points, then files the scan flagged. Always inside *root*."""
    wanted: list[str] = list(summary.get("entry_points") or [])
    for _n, where in (summary.get("long_functions") or []):
        wanted.append(where.split(":", 1)[0])
    wanted += [r for _n, r in (summary.get("long_files") or [])]
    wanted += list(summary.get("unparsable") or [])
    seen: list[str] = []
    for rel in wanted:
        if rel in seen:
            continue
        full = os.path.realpath(os.path.join(root, rel))
        if not (full == root or full.startswith(root.rstrip(os.sep) + os.sep)):
            continue
        if os.path.islink(os.path.join(root, rel)) or not os.path.isfile(full):
            continue
        seen.append(rel)
        if len(seen) >= max_files:
            break
    return seen


def build_review_prompt(summary: dict[str, Any], max_chars: int = 6000, max_files: int = 4) -> str:
    """Prompt asking a local model to review a project from its scan result
    plus short excerpts of the files the scan flagged.

    Only the first ``_MAX_REVIEW_FILE_LINES`` lines of at most *max_files*
    files are included, lines that look like they assign a secret are removed,
    and the whole prompt is cut to *max_chars*. The model is told the scan
    numbers are measured facts and to say so when it is guessing.
    """
    root = summary["path"]
    header = [
        "You are reviewing a software project for its owner. Below is a measured "
        "scan of the project followed by short excerpts of a few files the scan "
        "flagged. The scan numbers are facts; do not contradict them. Base every "
        "other claim on the excerpts and say plainly when you are guessing. Give "
        "at most five concrete, prioritised suggestions in plain sentences, no "
        "preamble.",
        "",
        "SCAN:",
        format_summary(summary),
        "",
    ]
    budget = max(500, int(max_chars)) - sum(len(h) + 1 for h in header)
    body: list[str] = []
    for rel in _review_files(root, summary, max(0, int(max_files))):
        if budget <= 200:
            break
        try:
            with open(os.path.join(root, rel), "r", encoding="utf-8", errors="replace") as fh:
                lines = [ln.rstrip("\n") for _, ln in zip(range(_MAX_REVIEW_FILE_LINES), fh)]
        except OSError:
            continue
        excerpt = "\n".join(_redact(ln) for ln in lines)[: max(0, budget - len(rel) - 40)]
        block = f"--- {rel} (first {len(lines)} lines) ---\n{excerpt}\n"
        body.append(block)
        budget -= len(block)
    prompt = "\n".join(header + body)
    return prompt[: max(500, int(max_chars))]
