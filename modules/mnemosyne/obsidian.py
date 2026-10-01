"""
modules/mnemosyne/obsidian.py

Obsidian vault sync (backlog #39): watch a configured vault folder, ingest
its markdown notes with wikilink-aware chunking, and write structured notes
Hestia generates back into the vault.

Safety model — this touches a folder of the user's own writing, so the
rules are strict and enforced in code, not by convention:

* **Off by default.** Nothing here runs unless a vault path is configured
  *and* ``enabled`` is true (see ``MnemosyneEngine.configure_obsidian``).
* **Read-only on existing notes.** Ingest only ever reads. Write-back goes
  exclusively into one dedicated subfolder (default ``Hestia/``) and only
  ever creates *new* files — a name collision gets a numeric suffix, it
  never overwrites, appends to, or edits anything. The subfolder is also
  excluded from ingest so Hestia never re-reads its own output.
* **No escaping the vault.** Generated filenames are sanitised and the
  final path is resolved and checked to lie inside the write-back folder;
  symlinks that point outside the vault are skipped on read.

Ingest is incremental: each note's content hash is stored, so a sync only
re-embeds notes that changed, and notes deleted from the vault have their
chunks and graph entries removed.

The parsing/chunking functions are pure (text in, data out) so they can be
tested without a vault on disk.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_SUBFOLDER = "Hestia"
_MAX_NOTE_BYTES = 2_000_000          # skip pathological files
_DEFAULT_CHUNK_CHARS = 800
_IGNORED_DIRS = frozenset({".obsidian", ".trash", ".git", ".stfolder", "node_modules"})

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_WIKILINK_RE = re.compile(r"(!?)\[\[([^\]\|#\^]+)(#[^\]\|]*)?(\^[^\]\|]*)?(?:\|([^\]]*))?\]\]")
_TAG_RE = re.compile(r"(?<![\w/&])#([A-Za-z][\w\-/]*)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


# ---------------------------------------------------------------------------
# Parsing (pure)
# ---------------------------------------------------------------------------

def parse_note(text: str) -> tuple[dict, str]:
    """
    Split a note into ``(frontmatter, body)``.

    Handles the YAML subset Obsidian actually writes — ``key: value``,
    ``key: [a, b]`` and ``key:`` followed by ``- item`` lines. Anything
    fancier is left as a raw string rather than guessed at; a malformed
    block is treated as body text, never an error.
    """
    text = (text or "").lstrip("\ufeff")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            end = i
            break
    if end is None:
        return {}, text

    meta: dict[str, Any] = {}
    current_list: Optional[str] = None
    for raw in lines[1:end]:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        item = re.match(r"^\s+-\s+(.*)$", raw)
        if item and current_list:
            meta[current_list].append(_unquote(item.group(1)))
            continue
        m = re.match(r"^([A-Za-z0-9_\-]+)\s*:\s*(.*)$", raw)
        if not m:
            current_list = None
            continue
        key, value = m.group(1), m.group(2).strip()
        if value == "":
            meta[key] = []
            current_list = key
        elif value.startswith("[") and value.endswith("]"):
            meta[key] = [_unquote(v) for v in value[1:-1].split(",") if v.strip()]
            current_list = None
        else:
            meta[key] = _unquote(value)
            current_list = None
    return meta, "\n".join(lines[end + 1:]).lstrip("\n")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _outside_code(body: str) -> str:
    """*body* with fenced code blocks and inline code blanked out."""
    out, fenced = [], False
    for line in body.splitlines():
        if _FENCE_RE.match(line):
            fenced = not fenced
            out.append("")
            continue
        out.append("" if fenced else re.sub(r"`[^`]*`", "", line))
    return "\n".join(out)


def extract_wikilinks(body: str) -> list[str]:
    """
    Link targets in document order, de-duplicated case-insensitively.
    ``[[Note|alias]]``, ``[[Note#Heading]]`` and ``![[Note]]`` embeds all
    resolve to ``Note``; links inside code are ignored.
    """
    seen, out = set(), []
    for m in _WIKILINK_RE.finditer(_outside_code(body)):
        target = m.group(2).strip()
        key = target.lower()
        if target and key not in seen:
            seen.add(key)
            out.append(target)
    return out


def extract_tags(frontmatter: dict, body: str) -> list[str]:
    tags: list[str] = []
    fm = frontmatter.get("tags") or frontmatter.get("tag") or []
    if isinstance(fm, str):
        fm = [t for t in re.split(r"[,\s]+", fm) if t]
    for t in fm:
        t = str(t).lstrip("#").strip()
        if t:
            tags.append(t)
    for m in _TAG_RE.finditer(_outside_code(body)):
        tags.append(m.group(1))
    seen, out = set(), []
    for t in tags:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def plain_text(text: str) -> str:
    """Wikilinks rendered as the words a reader sees: ``[[A|b]]`` -> ``b``."""
    def _sub(m: re.Match) -> str:
        if m.group(1):              # ![[embed]] — keep the name, it's content
            return m.group(2).strip()
        return (m.group(5) or m.group(2)).strip()
    return _WIKILINK_RE.sub(_sub, text)


# ---------------------------------------------------------------------------
# Chunking (pure)
# ---------------------------------------------------------------------------

def _in_wikilink(text: str, pos: int) -> bool:
    before = text[:pos]
    return before.rfind("[[") > before.rfind("]]")


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split one over-long paragraph at sentence ends, never inside a wikilink."""
    if len(text) <= max_chars:
        return [text]
    pieces, start = [], 0
    last_ok = None
    for m in re.finditer(r"(?<=[.!?])\s+", text):
        if _in_wikilink(text, m.start()):
            continue
        if m.start() - start > max_chars and last_ok is not None:
            pieces.append(text[start:last_ok].strip())
            start = last_ok
        last_ok = m.end()
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)

    # A single sentence longer than max_chars: hard-split on whitespace, but
    # again never through a wikilink.
    final: list[str] = []
    for p in pieces:
        while len(p) > max_chars * 1.5:
            cut = p.rfind(" ", 0, max_chars)
            while cut > 0 and _in_wikilink(p, cut):
                cut = p.rfind(" ", 0, cut)
            if cut <= 0:
                break
            final.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            final.append(p)
    return final


def _sections(body: str) -> list[tuple[list[str], list[str]]]:
    """``[(heading_path, paragraphs)]``; fenced code is kept as one paragraph."""
    sections: list[tuple[list[str], list[str]]] = []
    path: list[tuple[int, str]] = []
    paras: list[str] = []
    buf: list[str] = []
    fenced = False

    def flush_buf() -> None:
        if buf:
            text = "\n".join(buf).strip("\n")
            if text.strip():
                paras.append(text)
            buf.clear()

    def flush_section() -> None:
        flush_buf()
        if paras:
            sections.append(([h for _, h in path], list(paras)))
            paras.clear()

    for line in body.splitlines():
        if _FENCE_RE.match(line):
            if not fenced:
                flush_buf()
            buf.append(line)
            fenced = not fenced
            if not fenced:
                flush_buf()
            continue
        if fenced:
            buf.append(line)
            continue
        hm = _HEADING_RE.match(line)
        if hm:
            flush_section()
            level = len(hm.group(1))
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, hm.group(2)))
            continue
        if not line.strip():
            flush_buf()
        else:
            buf.append(line)
    flush_section()
    return sections


def chunk_note(title: str, body: str, max_chars: int = _DEFAULT_CHUNK_CHARS) -> list[dict]:
    """
    Wikilink-aware chunks for one note body (frontmatter already removed).

    * Chunks follow the heading structure — a section is never merged with
      its neighbour, so a chunk answers for one topic.
    * Paragraphs are packed up to *max_chars*; a fenced code block stays in
      one piece; a long paragraph splits at sentence ends, and never inside
      a ``[[wikilink]]``.
    * Each chunk's embedded ``text`` is prefixed ``Title › Heading › Sub``
      (context the chunk alone would lack) and has wikilinks rendered as
      plain words; its ``links`` are the real link targets found *in that
      chunk*, so graph/back-link data survives the plain-text rendering.
    """
    chunks: list[dict] = []
    for heading_path, paragraphs in _sections(body):
        context = " › ".join([title] + heading_path)
        units: list[str] = []
        for p in paragraphs:
            first_line = p.splitlines()[0] if p.strip() else ""
            if _FENCE_RE.match(first_line):
                units.append(p)
            else:
                units.extend(_split_long(p, max_chars))

        current: list[str] = []
        size = 0
        for unit in units:
            if current and size + len(unit) + 2 > max_chars:
                chunks.append(_make_chunk(context, heading_path, current, len(chunks)))
                current, size = [], 0
            current.append(unit)
            size += len(unit) + 2
        if current:
            chunks.append(_make_chunk(context, heading_path, current, len(chunks)))
    return chunks


def _make_chunk(context: str, heading_path: list[str], units: list[str], index: int) -> dict:
    raw = "\n\n".join(units)
    return {
        "index": index,
        "heading": " > ".join(heading_path),
        "text": f"{context}\n{plain_text(raw)}",
        "links": extract_wikilinks(raw),
    }


# ---------------------------------------------------------------------------
# Write-back naming (pure)
# ---------------------------------------------------------------------------

_BAD_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|#^\[\]\x00-\x1f]')


def safe_filename(title: str, max_len: int = 80) -> str:
    """A cross-platform, Obsidian-safe note filename stem (never empty)."""
    name = _BAD_FILENAME_CHARS.sub(" ", title or "")
    name = " ".join(name.split()).strip(" .")
    return (name[:max_len].strip(" .")) or "Untitled"


# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS obsidian_files (
    path TEXT PRIMARY KEY,          -- vault-relative, forward slashes
    content_hash TEXT,
    title TEXT,
    chunk_count INTEGER DEFAULT 0,
    synced_at TEXT
);
"""


def _chunk_id(rel_path: str, index: int) -> str:
    return f"obsidian:{rel_path}:{index}"


class ObsidianVault:
    """
    A configured vault plus its sync state. ``db_path`` is Mnemosyne's
    SQLite file (the sync state lives in its own table there).
    """

    def __init__(
        self,
        vault_path: str,
        db_path: str,
        subfolder: str = DEFAULT_SUBFOLDER,
        chunk_chars: int = _DEFAULT_CHUNK_CHARS,
    ) -> None:
        if not vault_path:
            raise ValueError("ObsidianVault requires a vault_path.")
        self.root = Path(vault_path).expanduser()
        self.subfolder = safe_filename(subfolder or DEFAULT_SUBFOLDER)
        self.chunk_chars = chunk_chars
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    # -- discovery -------------------------------------------------------

    @property
    def exists(self) -> bool:
        return self.root.is_dir()

    @property
    def writeback_dir(self) -> Path:
        return self.root / self.subfolder

    def scan(self) -> list[Path]:
        """
        Every ingestible ``.md`` file, sorted. Skips hidden/tool folders,
        the write-back subfolder, oversized files, and symlinks that resolve
        outside the vault.
        """
        if not self.exists:
            return []
        root = self.root.resolve()
        skip_top = self.subfolder.lower()
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_dir = Path(dirpath).relative_to(self.root)
            dirnames[:] = [
                d for d in dirnames
                if d not in _IGNORED_DIRS and not d.startswith(".")
                and not (rel_dir == Path(".") and d.lower() == skip_top)
            ]
            for fn in filenames:
                if not fn.lower().endswith(".md") or fn.startswith("."):
                    continue
                p = Path(dirpath) / fn
                try:
                    resolved = p.resolve()
                    resolved.relative_to(root)           # symlink escape check
                    if resolved.stat().st_size > _MAX_NOTE_BYTES:
                        logger.info("Obsidian: skipping oversized note %s", p)
                        continue
                except (OSError, ValueError):
                    continue
                found.append(p)
        return sorted(found)

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # -- sync ------------------------------------------------------------

    def sync(
        self,
        add_chunks: Optional[Callable[[list[dict]], None]] = None,
        delete_chunks: Optional[Callable[[list[str]], None]] = None,
        graph: Any = None,
    ) -> dict:
        """
        Bring the index in line with the vault. Returns
        ``{"added","updated","removed","unchanged","chunks","errors"}``.

        *add_chunks* receives dicts ``{id, text, metadata}`` per note;
        *delete_chunks* receives chunk ids to drop; *graph* is a
        ``KnowledgeGraph`` (optional). One unreadable note is logged and
        counted in ``errors`` — it never aborts the rest of the sync.
        """
        stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0, "chunks": 0, "errors": 0}
        if not self.exists:
            logger.warning("Obsidian vault %s not found; nothing to sync.", self.root)
            return stats

        files = self.scan()
        stems = {p.stem.lower() for p in files}
        known = {
            r["path"]: dict(r)
            for r in self._conn.execute("SELECT * FROM obsidian_files").fetchall()
        }
        now = datetime.now(timezone.utc).isoformat()
        present: set[str] = set()

        for path in files:
            rel = self._rel(path)
            present.add(rel)
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
                digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
                prev = known.get(rel)
                if prev and prev["content_hash"] == digest:
                    stats["unchanged"] += 1
                    continue

                frontmatter, body = parse_note(raw)
                title = str(frontmatter.get("title") or path.stem)
                chunks = chunk_note(title, body, self.chunk_chars)
                tags = extract_tags(frontmatter, body)
                links = extract_wikilinks(body)

                if prev and delete_chunks and prev["chunk_count"]:
                    delete_chunks([_chunk_id(rel, i) for i in range(prev["chunk_count"])])
                if add_chunks and chunks:
                    add_chunks([
                        {
                            "id": _chunk_id(rel, c["index"]),
                            "text": c["text"],
                            "metadata": {
                                "type": "note",
                                "created_at": now,
                                "source": "obsidian",
                                "path": rel,
                                "title": title,
                                "heading": c["heading"],
                                "links": ", ".join(c["links"]),
                            },
                        }
                        for c in chunks
                    ])
                if graph is not None:
                    self._update_graph(graph, rel, title, links, tags, stems)

                with self._lock, self._conn:
                    self._conn.execute(
                        """
                        INSERT INTO obsidian_files (path, content_hash, title, chunk_count, synced_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(path) DO UPDATE SET
                            content_hash = excluded.content_hash, title = excluded.title,
                            chunk_count = excluded.chunk_count, synced_at = excluded.synced_at
                        """,
                        (rel, digest, title, len(chunks), now),
                    )
                stats["updated" if prev else "added"] += 1
                stats["chunks"] += len(chunks)
            except Exception:
                logger.exception("Obsidian: failed to sync %s; continuing.", rel)
                stats["errors"] += 1

        for rel, prev in known.items():
            if rel in present:
                continue
            try:
                if delete_chunks and prev["chunk_count"]:
                    delete_chunks([_chunk_id(rel, i) for i in range(prev["chunk_count"])])
                if graph is not None:
                    graph.remove_source(f"note:{rel}")
                with self._lock, self._conn:
                    self._conn.execute("DELETE FROM obsidian_files WHERE path = ?", (rel,))
                stats["removed"] += 1
            except Exception:
                logger.exception("Obsidian: failed to remove %s; continuing.", rel)
                stats["errors"] += 1
        return stats

    @staticmethod
    def _update_graph(graph: Any, rel: str, title: str, links: list[str],
                      tags: list[str], stems: set[str]) -> None:
        source = f"note:{rel}"
        graph.remove_source(source)       # replace, don't accumulate stale links
        entities = [(title, "note")]
        relations = []
        for target in links:
            is_note = target.lower() in stems
            entities.append((target, "note" if is_note else "concept"))
            relations.append((title, "links to", target))
        for tag in tags:
            entities.append((tag, "tag"))
            relations.append((title, "tagged", tag))
        graph.add_extraction(source, entities, relations)

    # -- write-back ------------------------------------------------------

    def write_note(
        self,
        title: str,
        body: str,
        tags: Optional[Iterable[str]] = None,
        links: Optional[Iterable[str]] = None,
    ) -> Path:
        """
        Create a NEW note in the write-back subfolder and return its path.

        Never overwrites: if the filename is taken, ``" 2"``, ``" 3"`` … is
        appended. Raises ``FileNotFoundError`` if the vault itself doesn't
        exist (Hestia won't create someone's vault for them) and
        ``ValueError`` if the resolved path would leave the write-back
        folder.
        """
        if not self.exists:
            raise FileNotFoundError(f"Obsidian vault not found: {self.root}")
        folder = self.writeback_dir
        folder.mkdir(parents=True, exist_ok=True)
        folder_resolved = folder.resolve()

        stem = safe_filename(title)
        candidate = folder / f"{stem}.md"
        n = 1
        while candidate.exists():
            n += 1
            candidate = folder / f"{stem} {n}.md"
        if candidate.resolve().parent != folder_resolved:
            raise ValueError("Refusing to write outside the Hestia folder.")

        tag_list = [t.lstrip("#") for t in (tags or []) if str(t).strip()]
        front = ["---", "generated_by: hestia",
                 f"created: {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
        if tag_list:
            front.append("tags: [" + ", ".join(tag_list) + "]")
        front.append("---")
        link_block = ""
        link_list = [str(l).strip() for l in (links or []) if str(l).strip()]
        if link_list:
            link_block = "\n\n## Related\n" + "\n".join(f"- [[{safe_filename(l)}]]" for l in link_list)

        content = "\n".join(front) + f"\n\n# {title}\n\n{body.strip()}{link_block}\n"
        # "x" mode: fail instead of clobbering if another process raced us.
        with open(candidate, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        return candidate

    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(chunk_count), 0) AS c FROM obsidian_files"
        ).fetchone()
        return {"notes": row["n"], "chunks": row["c"], "vault": str(self.root),
                "exists": self.exists}
