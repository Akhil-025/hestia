"""
core/web_search.py - "find everything related to X" (backlog #184).

One query is sent to Mnemosyne (facts, notes, past conversations, summaries),
Athena (document passages) and Iris (photos and videos) at the same time. Each
source answers on its own: if Athena is slow or Iris is switched off, the other
two still show up, and the response says which source failed and why.

Nothing here calls the language model. Athena's own "search" intent writes an
answer; this uses its retrieval step only (``AthenaEngine.search_sources``).
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 200
DEFAULT_TIMEOUT = 25.0


def snippet(text: Any, query: str, width: int = 180) -> str:
    """A short excerpt of *text* around the first occurrence of the query."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= width:
        return text
    pos = text.lower().find(query.lower())
    if pos < 0:
        return text[:width].rstrip() + "…"
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def _memory_hits(ui, q: str, limit: int) -> list[dict]:
    mem = ui.memory
    hits: list[dict] = []
    seen: set[tuple] = set()

    def add(kind: str, title: str, text: Any, meta: dict) -> None:
        key = (kind, title, str(text)[:80])
        if key in seen:
            return
        seen.add(key)
        hits.append({"source": "mnemosyne", "kind": kind, "title": title,
                     "snippet": snippet(text, q), "meta": meta})

    for f in mem.db.search_facts(q, limit):
        add("fact", f["key"], f["value"], {"updated_at": f.get("updated_at")})

    for row in mem.db.search_interactions(q, limit * 2):
        kind = "note" if row.get("intent") == "take_note" else "history"
        shown = row["query"] if kind == "note" else f"{row['query']} → {row['response']}"
        add(kind, (row["query"] or "")[:80], shown,
            {"intent": row.get("intent"), "at": row.get("pushed_at")})

    store = getattr(mem, "vector_store", None)
    if store is not None:
        try:
            for r in store.search(q, n_results=limit, where={"type": {"$eq": "summary"}}):
                if r.get("score", 0) >= 0.3:
                    add("summary", "Conversation summary", r["text"], {"score": r.get("score")})
            for r in store.search(q, n_results=limit, where={"type": {"$eq": "fact"}}):
                if r.get("score", 0) >= 0.3:
                    md = r.get("metadata") or {}
                    add("fact", md.get("key") or "Fact", r["text"],
                        {"score": r.get("score"), "semantic": True})
        except Exception:
            # Semantic search is a bonus over the exact matches above.
            logger.debug("[WebUI] semantic memory search failed", exc_info=True)
    return hits[: limit * 3]


def _athena_hits(ui, q: str, limit: int) -> list[dict]:
    return [{
        "source": "athena", "kind": "document",
        "title": d.get("file_name") or "Document",
        "snippet": snippet(d.get("text"), q),
        "meta": {"page": d.get("page"), "subject": d.get("subject"),
                 "score": d.get("score")},
    } for d in ui.athena.search_sources(q, limit)]


def _iris_hits(ui, q: str, limit: int) -> list[dict]:
    out = []
    for r in ui.iris.search_records(q, limit):
        tags = r.get("tags")
        out.append({
            "source": "iris", "kind": "photo",
            "title": str(r.get("file_path") or "").replace("\\", "/").rsplit("/", 1)[-1] or "Media file",
            "snippet": snippet(r.get("caption") or "", q),
            "meta": {"path": r.get("file_path"), "tags": tags,
                     "sensitive": bool(r.get("is_sensitive"))},
        })
    return out


def search_all(ui, query: str, limit: int = 8, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Run the three searches in parallel; always returns a complete payload."""
    q = (query or "").strip()[:MAX_QUERY_LENGTH]
    limit = max(1, min(int(limit), 25))
    sources: list[tuple[str, Callable[[], list[dict]]]] = []
    if ui.memory is not None:
        sources.append(("mnemosyne", lambda: _memory_hits(ui, q, limit)))
    if getattr(ui, "athena", None) is not None:
        sources.append(("athena", lambda: _athena_hits(ui, q, limit)))
    if getattr(ui, "iris", None) is not None:
        sources.append(("iris", lambda: _iris_hits(ui, q, limit)))

    groups: dict[str, dict] = {
        name: {"ok": False, "items": [], "error": "not available"}
        for name in ("mnemosyne", "athena", "iris")
    }
    if not q or not sources:
        return {"query": q, "groups": groups, "total": 0}

    pool = ThreadPoolExecutor(max_workers=len(sources), thread_name_prefix="web-search")
    futures = {pool.submit(fn): name for name, fn in sources}
    done, pending = wait(futures, timeout=timeout)
    for fut in done:
        name = futures[fut]
        try:
            items = fut.result()
            groups[name] = {"ok": True, "items": items, "error": None}
        except Exception as exc:
            logger.exception("[WebUI] %s search failed", name)
            groups[name] = {"ok": False, "items": [], "error": type(exc).__name__}
    for fut in pending:
        groups[futures[fut]] = {"ok": False, "items": [], "error": "timed out"}
        fut.cancel()
    pool.shutdown(wait=False)   # a stuck source must not hold this request

    return {"query": q, "groups": groups,
            "total": sum(len(g["items"]) for g in groups.values())}
