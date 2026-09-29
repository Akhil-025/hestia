"""
core/text_export.py

Small, dependency-free helpers for exporting a piece of writing to a plain
text or Markdown file (backlog #168).

Shared by Orpheus (`export_creation`) and Metis (`export_session`) so the
two modules agree on filename rules, formats and, most importantly, on
*where they are allowed to write*.

Safety rules
------------
- Files are only ever written inside the export directory the caller
  supplies. A user-provided ``filename`` is reduced to its basename and
  slugified, so ``../../etc/passwd`` or ``C:\\Windows\\x`` can never escape it.
- An existing file is never overwritten: a numeric suffix is added instead
  (``poem.md`` -> ``poem-2.md``). Exports are cheap; losing one is not.
- Only ``txt`` and ``md`` are supported. Unknown formats fall back to
  Markdown rather than raising, since the caller is usually a voice/chat
  intent where "export it as a doc" should still produce *something*.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

VALID_FORMATS: tuple[str, ...] = ("md", "txt")
DEFAULT_FORMAT = "md"
DEFAULT_EXPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "exports"

_FORMAT_ALIASES: dict[str, str] = {
    "md": "md", "markdown": "md", ".md": "md",
    "txt": "txt", "text": "txt", "plain": "txt", "plaintext": "txt",
    "plain text": "txt", ".txt": "txt",
}
_MAX_STEM_LEN = 60
_MAX_UNIQUE_ATTEMPTS = 1000

# Types whose line breaks are part of the content. Markdown collapses a
# single newline into a space, which would turn a poem into a paragraph.
VERSE_TYPES: frozenset[str] = frozenset({"poem", "lyrics", "haiku"})


def normalise_format(value: object) -> str:
    """Map a free-form format word ("Markdown", ".txt") to ``md``/``txt``."""
    key = str(value or "").strip().lower()
    return _FORMAT_ALIASES.get(key, DEFAULT_FORMAT)


def slugify(value: str, fallback: str = "export") -> str:
    """
    Reduce *value* to a filesystem-safe lowercase slug.

    Keeps ASCII letters, digits, ``-`` and ``_``; everything else becomes a
    single hyphen. Never returns an empty string and never returns a name
    that is only dots (``.``/``..``).
    """
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text).strip("-_")
    text = text[:_MAX_STEM_LEN].strip("-_")
    return text or fallback


def safe_stem(filename: str, fallback: str) -> str:
    """
    Turn a user-supplied filename into a safe stem.

    Directory components (either slash style) and any extension are
    dropped, so the result can only ever name a file inside the export
    directory.
    """
    base = re.split(r"[\\/]", str(filename or "").strip())[-1]
    base = re.sub(r"\.(md|markdown|txt)$", "", base, flags=re.IGNORECASE)
    return slugify(base, fallback)


def unique_path(directory: Path, stem: str, ext: str) -> Path:
    """Return a path in *directory* that does not exist yet."""
    candidate = directory / f"{stem}.{ext}"
    n = 2
    while candidate.exists():
        if n > _MAX_UNIQUE_ATTEMPTS:
            raise FileExistsError(f"Too many existing exports named {stem!r}.")
        candidate = directory / f"{stem}-{n}.{ext}"
        n += 1
    return candidate


def render_document(
    title: str,
    body: str,
    fmt: str = DEFAULT_FORMAT,
    meta: list[tuple[str, str]] | None = None,
    sections: list[tuple[str, str]] | None = None,
    preserve_lines: bool = False,
) -> str:
    """
    Render a titled document as Markdown or plain text.

    Parameters
    ----------
    title:
        Document heading.
    body:
        Main text.
    meta:
        Ordered ``(label, value)`` pairs shown under the title (type,
        date, version...). Pairs with an empty value are skipped.
    sections:
        Extra ``(heading, text)`` blocks appended after the body (e.g. a
        writing session's critique or a creation's version history).
    preserve_lines:
        Keep single line breaks in Markdown (poems, lyrics). Implemented
        as a trailing double-space "hard break" so the file still reads
        naturally as plain text.
    """
    fmt = normalise_format(fmt)
    meta_pairs = [(k, v) for k, v in (meta or []) if str(v or "").strip()]
    extra = [(h, t) for h, t in (sections or []) if str(t or "").strip()]
    title = (title or "Untitled").strip()
    body = (body or "").strip("\n")

    if fmt == "md":
        out: list[str] = [f"# {title}", ""]
        if meta_pairs:
            out.append(" · ".join(f"**{k}:** {v}" for k, v in meta_pairs))
            out.append("")
        out.append(_hard_breaks(body) if preserve_lines else body)
        for heading, text in extra:
            out += ["", f"## {heading}", "", text.strip("\n")]
        return "\n".join(out).rstrip() + "\n"

    out = [title, "=" * max(len(title), 3), ""]
    if meta_pairs:
        out += [f"{k}: {v}" for k, v in meta_pairs]
        out.append("")
    out.append(body)
    for heading, text in extra:
        out += ["", heading, "-" * max(len(heading), 3), text.strip("\n")]
    return "\n".join(out).rstrip() + "\n"


def _hard_breaks(text: str) -> str:
    """Add Markdown hard breaks to every non-blank line but the last of a stanza."""
    lines = text.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and nxt.strip():
            out.append(line.rstrip() + "  ")
        else:
            out.append(line.rstrip())
    return "\n".join(out)


def write_export(
    directory: Path | str | None,
    stem: str,
    fmt: str,
    text: str,
) -> Path:
    """
    Write *text* to a new file in *directory* and return its path.

    Creates the directory if needed. Raises ``OSError`` on I/O failure —
    callers are expected to turn that into a friendly ``_err`` response.
    """
    ext = normalise_format(fmt)
    out_dir = Path(directory) if directory else DEFAULT_EXPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = unique_path(out_dir, slugify(stem), ext)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path