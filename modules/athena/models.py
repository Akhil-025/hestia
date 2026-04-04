"""
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text":         self.text,
            "file_name":    self.file_name,
            "file_path":    self.file_path,
            "page":         self.page_number,
            "subject":      self.subject,
            "module":       self.module,
            "chunk_number": self.chunk_number,
            "score":        self.score,
        }


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
    def from_rag_response(cls, response: Dict[str, Any]) -> "SearchResults":
        """Build a SearchResults from the dict returned by MergedLocalRAG.search()."""
        return cls(
            documents       = response.get("documents",       []),
            metadatas       = response.get("metadatas",       []),
            scores          = response.get("scores",          []),
            semantic_scores = response.get("semantic_scores", []),
            bm25_scores     = response.get("bm25_scores",     []),
            query           = response.get("query",           ""),
            total_results   = response.get("total_results",   0),
        )

    def to_source_documents(self) -> List[SourceDocument]:
        """Convert to a flat list of SourceDocument objects."""
        sources: List[SourceDocument] = []
        scores = self.scores or [0.0] * len(self.documents)

        for doc, md, score in zip(self.documents, self.metadatas, scores):
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
