"""
modules/athena/services/citation_graph_service.py

Cross-document citation graph (backlog #60): which of YOUR indexed papers cite
which of your other indexed papers.

How it works
------------
1. ``rag.list_document_sources`` says which files are indexed and where they live.
2. Each file is read again from disk (not from the chunk index, which has had its
   URLs/DOIs and line breaks removed) to get its own identity (title, arXiv id,
   DOI, year) and its reference list. See ``modules/athena/bibliography.py``.
3. ``bibliography.match_references`` links a reference to a library document only
   on evidence (DOI, arXiv id, or the document's title inside the entry).
4. The result is a plain dict that ``citation_graph_view`` draws.

Parsing a PDF is the slow part, so each file's parse is cached (keyed by path,
mtime, size and ``PARSER_VERSION``) and a second request only re-reads changed files.

Limits, stated plainly
----------------------
* A paper whose reference list cannot be found (scanned PDF, no "References"
  heading) can be cited but cannot itself cite anything; its node says why.
* Only citations between documents in the library are drawn. Everything else is
  listed as "cited by several of your papers but not in your library" when two or
  more papers share it.
* A graph limited to one subject cannot show citations to other subjects.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

from modules.athena import bibliography as bib

try:                                    # "pymupdf" is the real name; a test stub may shadow "fitz"
    import pymupdf as _pdf              # type: ignore
except ImportError:                     # pragma: no cover
    try:
        import fitz as _pdf             # type: ignore
    except ImportError:
        _pdf = None

logger = logging.getLogger(__name__)

_TAIL_PAGES = 15                        # how far from the end of a PDF to look for "References"
_EVIDENCE_CHARS = 500
_RELIABLE_YEAR_SOURCES = ("arxiv", "front-matter")


# ---------------------------------------------------------------------------
# Reading one document
# ---------------------------------------------------------------------------

def _page_lines(page) -> list[tuple[str, float]]:
    lines = []
    for block in (page.get_text("dict") or {}).get("blocks", []):
        for ln in block.get("lines", []):
            spans = ln.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if text:
                lines.append((text, max((s.get("size", 0) for s in spans), default=0)))
    return lines


def _read_pdf(path: str, file_name: str) -> tuple[bib.Identity, list[bib.Reference], str]:
    if _pdf is None:
        raise RuntimeError("PyMuPDF is not installed")
    with _pdf.open(path) as doc:
        n = len(doc)
        if n == 0:
            return bib.identity_from_filename(file_name), [], "no_text"
        first = doc[0]
        page_text = bib.text_before_references(first.get_text("text") or "")
        meta = doc.metadata or {}
        ident = bib.identity_from_first_page(
            _page_lines(first), page_text, file_name, meta.get("title") or "", meta.get("author") or "")
        if not any((doc[i].get_text("text") or "").strip() for i in range(min(n, 3))):
            return ident, [], "no_text"
        tail, found_heading = [], False
        for i in range(n - 1, max(-1, n - 1 - _TAIL_PAGES), -1):
            text = doc[i].get_text("text") or ""
            tail.append(text)
            if bib.has_reference_heading(text):
                found_heading = True
                break
    if not found_heading:
        return ident, [], "no_reference_section"
    refs, found = bib.parse_references("\n".join(reversed(tail)), early_cutoff=False)
    if not found:
        return ident, [], "no_reference_section"
    return ident, refs, ("ok" if refs else "empty_reference_section")


def _read_other(path: str, file_name: str) -> tuple[bib.Identity, list[bib.Reference], str]:
    from modules.athena.document_processor import extract_text_from_file
    pages = extract_text_from_file(path)
    text = "\n".join(p.get("text", "") for p in pages)
    if not text.strip():
        return bib.identity_from_filename(file_name), [], "no_text"
    ident = bib.identity_from_markdown(text, file_name) if file_name.lower().endswith(".md") else None
    if ident is None:
        own = bib.text_before_references(text)
        head = [(re.sub(r"^#+\s*", "", ln.strip()), 0.0) for ln in own.splitlines() if ln.strip()][:40]
        ident = bib.identity_from_first_page(head, own[:3000], file_name)
    refs, found = bib.parse_references(text)
    if not found:
        return ident, [], "no_reference_section"
    return ident, refs, ("ok" if refs else "empty_reference_section")


def read_document(path: str, file_name: str) -> dict[str, Any]:
    """Parse one file: {"identity", "references", "status"}. Never raises."""
    try:
        if not os.path.isfile(path):
            return {"identity": bib.identity_from_filename(file_name), "references": [], "status": "missing"}
        reader = _read_pdf if path.lower().endswith(".pdf") else _read_other
        ident, refs, status = reader(path, file_name)
    except Exception:
        logger.warning("Could not read %s for the citation graph.", path, exc_info=True)
        return {"identity": bib.identity_from_filename(file_name), "references": [], "status": "unreadable"}
    return {"identity": ident, "references": refs, "status": status}


# ---------------------------------------------------------------------------
# Parse cache
# ---------------------------------------------------------------------------

class _ParseCache:
    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        if path and os.path.isfile(path):
            try:
                self._data = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Ignoring unreadable citation cache %s", path)

    @staticmethod
    def signature(path: str) -> str:
        try:
            st = os.stat(path)
            return f"{int(st.st_mtime)}:{st.st_size}:v{bib.PARSER_VERSION}"
        except OSError:
            return ""

    def get(self, path: str):
        sig = self.signature(path)
        entry = self._data.get(path)
        if not sig or not entry or entry.get("sig") != sig:
            return None
        return {"identity": bib.Identity.from_dict(entry["identity"]),
                "references": [bib.Reference.from_dict(r) for r in entry["references"]],
                "status": entry["status"]}

    def put(self, path: str, parsed: dict) -> None:
        sig = self.signature(path)
        if not sig or parsed["status"] in ("unreadable", "missing"):
            return                                   # a transient failure must not be remembered
        self._data[path] = {"sig": sig, "identity": parsed["identity"].to_dict(), "status": parsed["status"],
                            "references": [r.to_dict() for r in parsed["references"]]}

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            try:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path + ".tmp"
                Path(tmp).write_text(json.dumps(self._data), encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError:
                logger.warning("Could not save the citation cache to %s", self.path, exc_info=True)


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------

class CitationGraphService:
    def __init__(self, rag, cache_path: Optional[str] = None) -> None:
        self.rag = rag
        self._cache = _ParseCache(cache_path)

    def build(self, subject: Optional[str] = None) -> dict[str, Any]:
        sources = self.rag.list_document_sources(subject=subject)
        parsed: dict[str, dict] = {}
        meta: dict[str, dict] = {}
        for src in sources:
            key = f"{src['subject']}::{src['file_name']}"
            if key in parsed:
                continue
            path = src.get("file_path") or ""
            result = self._cache.get(path) if path else None
            if result is None:
                result = read_document(path, src["file_name"])
                if path:
                    self._cache.put(path, result)
            parsed[key], meta[key] = result, src
        self._cache.save()

        keys = list(parsed)
        ids = {k: f"n{i}" for i, k in enumerate(keys, start=1)}
        docs = [bib.LibraryDoc(k, parsed[k]["identity"]) for k in keys]
        outcome = bib.match_references({k: parsed[k]["references"] for k in keys}, docs)

        cites: dict[str, int] = {k: 0 for k in keys}
        cited_by: dict[str, int] = {k: 0 for k in keys}
        links = []
        for lk in outcome.links:
            src_ident, dst_ident = parsed[lk.source]["identity"], parsed[lk.target]["identity"]
            suspicious = ""
            if (src_ident.year and dst_ident.year and dst_ident.year > src_ident.year
                    and src_ident.year_source in _RELIABLE_YEAR_SOURCES
                    and dst_ident.year_source in _RELIABLE_YEAR_SOURCES):
                suspicious = "The cited paper is dated later than the paper citing it"
            d = lk.to_dict()
            d.update({"source": ids[lk.source], "target": ids[lk.target], "suspicious": suspicious,
                      "evidence": parsed[lk.source]["references"][lk.ref_index].raw[:_EVIDENCE_CHARS]})
            links.append(d)
            cites[lk.source] += 1
            cited_by[lk.target] += 1

        nodes = []
        for k in keys:
            ident = parsed[k]["identity"]
            nodes.append({
                "id": ids[k], "label": ident.title or meta[k]["file_name"], "file_name": meta[k]["file_name"],
                "subject": meta[k]["subject"], "year": ident.year, "year_source": ident.year_source,
                "authors": list(ident.authors), "arxiv_id": ident.arxiv_id, "doi": ident.doi,
                "title_source": ident.title_source, "status": parsed[k]["status"],
                "references": len(parsed[k]["references"]), "cites": cites[k], "cited_by": cited_by[k],
            })

        # References that no library document matched, shared by 2+ of your papers.
        groups: dict[str, dict] = {}
        for src_key, refs in outcome.unresolved.items():
            for ref in refs:
                gk = bib.reference_group_key(ref)
                if gk:
                    g = groups.setdefault(gk, {"title": ref.title_guess, "reference": ref.raw[:200],
                                               "year": ref.years[0] if ref.years else None, "by": set()})
                    g["by"].add(src_key)
        missing = sorted(({"title": g["title"], "reference": g["reference"], "year": g["year"],
                           "cited_by_count": len(g["by"])} for g in groups.values() if len(g["by"]) >= 2),
                         key=lambda m: -m["cited_by_count"])

        top = sorted((n for n in nodes if n["cited_by"] > 0), key=lambda n: (-n["cited_by"], n["label"]))[:5]
        connected = {l["source"] for l in links} | {l["target"] for l in links}
        no_refs = [n for n in nodes if n["status"] != "ok"]
        notes = []
        if no_refs:
            notes.append(f"{len(no_refs)} of {len(nodes)} documents have no readable reference list, "
                         "so they can be cited but their own citations are unknown.")
        if outcome.ambiguous:
            notes.append(f"{len(outcome.ambiguous)} reference(s) matched more than one of your documents equally "
                         "well, so no arrow was drawn for them.")
        if subject:
            notes.append(f"Only documents under '{subject}' are shown, so citations to other subjects are missing.")
        return {
            "subject": subject or "", "nodes": nodes, "links": links,
            "stats": {"documents": len(nodes), "links": len(links), "connected_documents": len(connected),
                      "with_reference_lists": len(nodes) - len(no_refs), "ambiguous": len(outcome.ambiguous),
                      "most_cited": [{"id": n["id"], "label": n["label"], "cited_by": n["cited_by"]} for n in top]},
            "missing_but_cited": missing[:20], "notes": notes,
        }


# ---------------------------------------------------------------------------
# Asking about a graph
# ---------------------------------------------------------------------------

def find_node(graph: dict, query: str) -> tuple[Optional[dict], list[dict]]:
    """(the one matching node, []) or (None, candidates) when zero or several match."""
    q = (query or "").strip().lower()
    if not q:
        return None, []
    nodes = graph.get("nodes", [])
    for n in nodes:
        if q in (n["file_name"].lower(), n["file_name"].rsplit(".", 1)[0].lower(), n["label"].lower(),
                 n["arxiv_id"].lower()):
            return n, []
    hits = [n for n in nodes if q in n["label"].lower() or q in n["file_name"].lower()
            or any(q in a.lower() for a in n["authors"])]
    return (hits[0], []) if len(hits) == 1 else (None, hits)


def describe(graph: dict, focus: Optional[dict] = None) -> str:
    """A reply that reads well aloud."""
    s = graph["stats"]
    if not s["documents"]:
        return "There are no indexed documents to build a citation graph from yet."
    if focus is not None:
        by_id = {n["id"]: n for n in graph["nodes"]}
        out = [by_id[l["target"]]["label"] for l in graph["links"] if l["source"] == focus["id"]]
        inc = [by_id[l["source"]]["label"] for l in graph["links"] if l["target"] == focus["id"]]
        parts = [f"{focus['label']} cites {len(out)} of your papers"
                 + (f" ({'; '.join(out[:4])})" if out else "")
                 + f" and is cited by {len(inc)}" + (f" ({'; '.join(inc[:4])})" if inc else "") + "."]
        if focus["status"] != "ok" and not out:
            parts.append("Its own reference list could not be read, so the first number may be low.")
        return " ".join(parts)
    if not s["links"]:
        msg = (f"I looked through {s['documents']} document(s) and found no citations between them.")
        if s["with_reference_lists"] == 0:
            msg += " None of them has a reference list I could read."
        return msg
    top = s["most_cited"][0]
    msg = (f"{s['links']} citation(s) link {s['connected_documents']} of your {s['documents']} documents. "
           f"The most cited is {top['label']}, by {top['cited_by']} of your other papers.")
    if graph["missing_but_cited"]:
        msg += f" {len(graph['missing_but_cited'])} work(s) are cited by several of your papers but not in your library."
    return msg


_GENERIC_SUBJECT = {"", "my", "all", "everything", "my papers", "my documents", "my library", "my notes",
                    "papers", "documents", "library", "my files", "them", "it"}


def parse_graph_request(raw: str) -> tuple[str, str]:
    """(subject, focus) from phrasings like "who cites attention is all you need" or "citation graph for physics"."""
    text = (raw or "").strip().rstrip("?.! ")
    m = re.search(r"\bwho\s+cites\s+(.+)$", text, re.I) or re.search(r"\bwhat\s+does\s+(.+?)\s+cite\b", text, re.I) \
        or re.search(r"\b(?:centred|centered|focus(?:ed)?)\s+on\s+(.+)$", text, re.I)
    if m:
        return "", m.group(1).strip(" \"'")
    m = re.search(r"\b(?:for|of|on|about|in)\s+(.+)$", text, re.I)
    subject = m.group(1).strip() if m else ""
    subject = re.sub(r"\s+(?:as|in|to)\s+(?:an?\s+)?(?:dot|graphviz|json|html)\b.*$", "", subject, flags=re.I)
    subject = re.sub(r"\s+(?:papers|documents|notes|library)$", "", subject, flags=re.I).strip(" \"'")
    return ("" if subject.lower() in _GENERIC_SUBJECT else subject), ""
