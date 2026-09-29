r"""
modules/athena/models.py

Data classes shared across the Athena pipeline.
Replaces the old C:\Athena\models\ package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Source document (one retrieved chunk) ────────────────────────────────────

@dataclass
class SourceDocument:
    text: str
    file_name: str
    file_path: str
    page_number: int
    subject: Optional[str] = None
    module: Optional[str] = None
    chunk_number: Optional[int] = None
    score: Optional[float] = None
    # backlog #64: the merged `score` above is what search/ranking
    # actually uses; these two are the breakdown BEHIND it, populated
    # only when the search that produced this document computed both
    # (hybrid search always does — see MergedLocalRAG's docstring).
    # Optional and defaulting to None rather than 0.0, so a caller can
    # tell "not computed for this search" apart from "computed as zero".
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None

    def to_dict(self, include_score_breakdown: bool = False) -> Dict[str, Any]:
        d = {
            "text":         self.text,
            "file_name":    self.file_name,
            "file_path":    self.file_path,
            "page":         self.page_number,
            "subject":      self.subject,
            "module":       self.module,
            "chunk_number": self.chunk_number,
            "score":        self.score,
        }
        # Opt-in (backlog #64's "in debug mode") rather than always-on:
        # the breakdown is genuinely useful for tuning retrieval but is
        # noise in every normal search response, and doubles the field
        # count of what's usually a list of several sources.
        if include_score_breakdown:
            d["semantic_score"] = self.semantic_score
            d["bm25_score"] = self.bm25_score
        return d


# ── Wrapper around a raw RAG response ────────────────────────────────────────

@dataclass
class SearchResults:
    documents:       List[str]             = field(default_factory=list)
    metadatas:       List[Dict[str, Any]]  = field(default_factory=list)
    scores:          List[float]           = field(default_factory=list)
    semantic_scores: List[float]           = field(default_factory=list)
    bm25_scores:     List[float]           = field(default_factory=list)
    query:           str                   = ""
    total_results:   int                   = 0

    @classmethod
    def from_rag_response(cls, response: Any) -> "SearchResults":
        """
        Build SearchResults from either:
        - old dict-based RAG responses
        - new SearchResponse objects
        """

        def read(name: str, default):
            if isinstance(response, dict):
                return response.get(name, default)
            return getattr(response, name, default)

        return cls(
            documents       = read("documents", []),
            metadatas       = read("metadatas", []),
            scores          = read("scores", []),
            semantic_scores = read("semantic_scores", []),
            bm25_scores     = read("bm25_scores", []),
            query           = read("query", ""),
            total_results   = read("total_results", 0),
        )

    def to_source_documents(self) -> List[SourceDocument]:
        """Convert to a flat list of SourceDocument objects."""
        sources: List[SourceDocument] = []
        scores = self.scores or [0.0] * len(self.documents)
        # These two are shorter than `documents` whenever the search that
        # produced this SearchResults didn't compute a breakdown (a pure-
        # semantic or pure-BM25 search, or an old-style dict response) —
        # index past the end just means "not available for this one".
        semantic = self.semantic_scores or []
        bm25 = self.bm25_scores or []

        for i, (doc, md, score) in enumerate(zip(self.documents, self.metadatas, scores)):
            if not doc:
                continue
            md = md or {}
            sources.append(SourceDocument(
                text         = doc,
                file_name    = md.get("file_name", "unknown"),
                file_path    = md.get("file_path", ""),
                page_number  = int(md.get("page_number", 0)),
                subject      = md.get("subject"),
                module       = md.get("module"),
                chunk_number = md.get("chunk_number"),
                score        = float(score),
                semantic_score = semantic[i] if i < len(semantic) else None,
                bm25_score     = bm25[i] if i < len(bm25) else None,
            ))
        return sources


# ── Final query result ────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    question:      str
    answer:        str
    sources:       List[SourceDocument]    = field(default_factory=list)
    cached:        bool                    = False
    mode:          str                     = "local"
    total_sources: int                     = 0
    metrics:       Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question":      self.question,
            "answer":        self.answer,
            "sources":       [s.to_dict() for s in self.sources],
            "cached":        self.cached,
            "mode":          self.mode,
            "total_sources": self.total_sources,
            "metrics":       self.metrics,
        }