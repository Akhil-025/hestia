"""
modules/athena/local_rag.py

Hybrid retrieval engine: dense embeddings (ChromaDB) + sparse BM25.

Design notes
------------
- ChromaDB and SentenceTransformers are initialised once at construction;
  failures raise immediately rather than silently degrading.
- BM25 index is rebuilt lazily and protected by a dedicated lock so that
  concurrent ingestion and search do not race.
- Every public method is fully typed, documented, and never raises;
  errors are logged and surfaced as empty/False returns.
- Distance → score conversion uses the numerically stable cosine formula
  (requires L2-normalised embeddings) rather than min-max normalisation.
- Ingestion is idempotent: files already present in the collection are
  skipped without touching the database.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Optional

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from modules.athena.config import get_config
from modules.athena.pdf_processor import (
    PDFProcessor,
    get_organization_structure,
    get_supported_files,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLLECTION_NAME = "engineering_documents"
_MIN_CHUNK_CHARS = 40
_MAX_SEARCH_RESULTS = 50
_BM25_CANDIDATE_MULTIPLIER = 3   # fetch 3× n_results before re-ranking
_SCORE_MIN = 0.0
_SCORE_MAX = 1.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RAGError(Exception):
    """Base exception for MergedLocalRAG failures."""


class ChromaInitError(RAGError):
    """Raised when ChromaDB cannot be initialised."""


class EmbedderInitError(RAGError):
    """Raised when the SentenceTransformer model cannot be loaded."""


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchResult:
    """A single ranked retrieval result."""

    document: str
    metadata: dict[str, Any]
    score: float                   # final hybrid (or semantic-only) score
    semantic_score: float = 0.0
    bm25_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "metadata": self.metadata,
            "score": self.score,
            "semantic_score": self.semantic_score,
            "bm25_score": self.bm25_score,
        }


@dataclass
class SearchResponse:
    """
    Structured response returned by every search method.

    Callers that previously consumed raw dicts can call ``to_dict()``.
    """

    results: list[SearchResult]
    query: str

    # ------------------------------------------------------------------
    # Convenience accessors (preserve the original dict-key contract)
    # ------------------------------------------------------------------

    @property
    def documents(self) -> list[str]:
        return [r.document for r in self.results]

    @property
    def metadatas(self) -> list[dict[str, Any]]:
        return [r.metadata for r in self.results]

    @property
    def scores(self) -> list[float]:
        return [r.score for r in self.results]

    @property
    def semantic_scores(self) -> list[float]:
        return [r.semantic_score for r in self.results]

    @property
    def bm25_scores(self) -> list[float]:
        return [r.bm25_score for r in self.results]

    @property
    def total_results(self) -> int:
        return len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "metadatas": self.metadatas,
            "scores": self.scores,
            "semantic_scores": self.semantic_scores,
            "bm25_scores": self.bm25_scores,
            "query": self.query,
            "total_results": self.total_results,
        }


@dataclass
class IngestionStats:
    total_files: int = 0
    total_chunks: int = 0
    by_subject: dict[str, dict[str, int]] = field(default_factory=dict)
    by_module: dict[str, dict[str, int]] = field(default_factory=dict)
    # backlog #57/#61: distinguishes "nothing to do" from "processed but
    # extracted nothing" from "genuinely new content" — a single
    # total_chunks==0 used to mean all three, which made a "what's new"
    # digest impossible to build from this alone.
    new_files: int = 0
    updated_files: int = 0
    unchanged_files: int = 0
    failed_files: int = 0

    def record(self, file_info: dict[str, str], chunks: int, status: str = "new") -> None:
        self.total_files += 1
        self.total_chunks += chunks

        if status == "new":
            self.new_files += 1
        elif status == "updated":
            self.updated_files += 1
        elif status == "unchanged":
            self.unchanged_files += 1
        else:
            self.failed_files += 1

        subj = file_info.get("subject") or "unknown"
        self.by_subject.setdefault(subj, {"files": 0, "chunks": 0})
        self.by_subject[subj]["files"] += 1
        self.by_subject[subj]["chunks"] += chunks

        mod = file_info.get("module") or "unknown"
        key = f"{subj}/{mod}"
        self.by_module.setdefault(key, {"files": 0, "chunks": 0})
        self.by_module[key]["files"] += 1
        self.by_module[key]["chunks"] += chunks

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_files": self.total_files,
            "total_chunks": self.total_chunks,
            "by_subject": self.by_subject,
            "by_module": self.by_module,
            "new_files": self.new_files,
            "updated_files": self.updated_files,
            "unchanged_files": self.unchanged_files,
            "failed_files": self.failed_files,
        }


# ---------------------------------------------------------------------------
# Internal BM25 state (kept in one place to simplify locking)
# ---------------------------------------------------------------------------

@dataclass
class _BM25State:
    index: Optional[BM25Okapi] = None
    corpus: list[str] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def clear(self) -> None:
        self.index = None
        self.corpus = []
        self.metadata = []

    @property
    def ready(self) -> bool:
        return self.index is not None and bool(self.corpus)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MergedLocalRAG:
    """
    Hybrid retrieval engine combining dense (ChromaDB) and sparse (BM25) search.

    Parameters
    ----------
    persist_directory:
        Path where ChromaDB stores its data.  Defaults to config value.
    model_name:
        HuggingFace model name for SentenceTransformer.
    embed_batch_size:
        Number of texts encoded per GPU/CPU call.
    enable_bm25:
        When ``True`` hybrid scoring is used; otherwise pure semantic search.

    Thread-safety
    -------------
    - The BM25 index is protected by ``_bm25_lock`` (RLock).
    - ChromaDB's PersistentClient is internally thread-safe for reads;
      mutating operations (``add``, ``delete``) are serialised via
      ``_chroma_write_lock``.
    - The SentenceTransformer embedder is loaded once and then read-only.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        model_name: Optional[str] = None,
        embed_batch_size: Optional[int] = None,
        enable_bm25: Optional[bool] = None,
    ) -> None:
        cfg = get_config()

        self.persist_directory: str = persist_directory or cfg.chroma_persist_dir
        self.model_name: str = model_name or cfg.embedding_model
        self.embed_batch_size: int = embed_batch_size or cfg.embed_batch_size
        self.enable_bm25: bool = (
            enable_bm25 if enable_bm25 is not None else cfg.enable_bm25
        )

        self._chroma_write_lock = threading.Lock()
        self._bm25_lock = threading.RLock()
        self._bm25 = _BM25State()

        self._client, self._collection = self._init_chroma()
        self._embedder: SentenceTransformer = self._init_embedder()
        self._pdf_processor = PDFProcessor()

        # backlog #63: relevance feedback, stored alongside (not inside)
        # ChromaDB — see core/chunk_feedback_store.py for why a separate
        # small SQLite store fits this better than Chroma metadata.
        from core.chunk_feedback_store import ChunkFeedbackStore
        self._feedback = ChunkFeedbackStore(
            str(Path(self.persist_directory) / "chunk_feedback.db")
        )

        logger.info(
            "MergedLocalRAG ready (model=%s, bm25=%s, dir=%s)",
            self.model_name,
            self.enable_bm25,
            self.persist_directory,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_chroma(self) -> tuple[chromadb.PersistentClient, chromadb.Collection]:
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"description": "Athena Knowledge Base"},
            )
            logger.info("ChromaDB initialised (%s).", self.persist_directory)
            return client, collection
        except Exception as exc:
            raise ChromaInitError(
                f"Failed to initialise ChromaDB at {self.persist_directory!r}: {exc}"
            ) from exc

    def _init_embedder(self) -> SentenceTransformer:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        try:
            embedder = SentenceTransformer(self.model_name, device=device)
            logger.info("Embedding model %r loaded on %s.", self.model_name, device)
            return embedder
        except Exception as exc:
            raise EmbedderInitError(
                f"Failed to load embedding model {self.model_name!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """
        Encode *texts* in batches and return L2-normalised embeddings.

        Raises
        ------
        RAGError
            If encoding fails for any reason.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        try:
            for start in range(0, len(texts), self.embed_batch_size):
                batch = texts[start : start + self.embed_batch_size]
                raw = self._embedder.encode(
                    batch,
                    show_progress_bar=False,
                    normalize_embeddings=True,  # cosine similarity via dot product
                )
                all_embeddings.extend(
                    raw.tolist() if hasattr(raw, "tolist") else [list(e) for e in raw]
                )
        except Exception as exc:
            raise RAGError(f"Embedding failed: {exc}") from exc

        return all_embeddings

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        file_info: dict[str, str],
        rebuild_bm25: bool = True,
    ) -> tuple[int, str]:
        """
        Ingest a single file into ChromaDB.

        The operation is idempotent AND change-aware (backlog #61): a file
        that hasn't changed since it was last ingested is skipped without
        touching the database, same as before. A file whose content HAS
        changed (detected via a cheap mtime+size signature — see
        `_file_signature` — not a full content hash, which would mean
        reading every large PDF/document on every ingestion run just to
        check whether it changed) has its OLD chunks deleted first, then
        is re-ingested fresh. Before this, a modified file with the same
        name/subject was silently treated as already-ingested forever —
        an update to a document was invisible to the index with no error
        or warning anywhere.

        Parameters
        ----------
        file_info:
            Dict with at minimum ``full_path``, ``file_name``, ``subject``,
            and ``module`` keys.
        rebuild_bm25:
            Rebuild the BM25 index after ingestion.  Pass ``False`` when
            bulk-ingesting to avoid rebuilding after every file.

        Returns
        -------
        tuple[int, str]
            (chunks added, status) where status is one of "new",
            "updated", "unchanged", or "failed" — see IngestionStats,
            which this return value exists to feed.
        """
        file_path = file_info.get("full_path", "")
        if not file_path:
            logger.warning("ingest_file: file_info missing 'full_path'; skipping.")
            return 0, "failed"

        try:
            signature = _file_signature(file_path)
        except OSError:
            logger.warning("ingest_file: could not stat %s; skipping.", file_path)
            return 0, "failed"

        existing_signature = self._get_ingested_signature(file_info)
        if existing_signature == signature:
            logger.debug("Unchanged since last ingestion: %s", file_path)
            return 0, "unchanged"

        status = "updated" if existing_signature is not None else "new"
        if existing_signature is not None:
            removed = self._delete_file_chunks(file_info)
            logger.info(
                "Content changed since last ingestion — removed %d stale "
                "chunk(s) for %s before re-ingesting.", removed, file_path,
            )

        ext = Path(file_path).suffix.lower()

        try:
            chunks = self._extract_chunks(file_info, ext)
        except Exception:
            logger.exception("Chunk extraction failed for %s.", file_path)
            return 0, "failed"

        if not chunks:
            logger.warning("No text extracted from %s.", file_path)
            return 0, "failed"

        ids, documents, metadatas = _prepare_batch(file_info, chunks, file_path, signature)

        try:
            embeddings = self._embed(documents)
        except RAGError:
            logger.exception("Embedding failed for %s; skipping ingestion.", file_path)
            return 0, "failed"

        try:
            with self._chroma_write_lock:
                self._collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
        except Exception:
            logger.exception("ChromaDB add failed for %s.", file_path)
            return 0, "failed"

        logger.info("Ingested %d chunk(s) from %s (%s).", len(chunks), file_path, status)

        if self.enable_bm25 and rebuild_bm25:
            self._rebuild_bm25()

        return len(chunks), status

    def ingest_directory(
        self,
        data_dir: Optional[str] = None,
        rebuild_bm25: bool = True,
    ) -> dict[str, Any]:
        """
        Ingest all supported files under *data_dir*.

        Returns
        -------
        dict
            Aggregated ingestion statistics, including the new/updated/
            unchanged/failed breakdown (backlog #57, #61).
        """
        resolved = data_dir or str(get_config().data_dir)
        files = get_supported_files(resolved)
        stats = IngestionStats()

        if not files:
            logger.warning("No supported files found in %s.", resolved)
            return stats.to_dict()

        for fi in files:
            n, status = self.ingest_file(fi, rebuild_bm25=False)
            stats.record(fi, n, status)

        if self.enable_bm25 and rebuild_bm25:
            self._rebuild_bm25()

        logger.info(
            "Directory ingestion complete: %d chunk(s) from %d file(s) "
            "(%d new, %d updated, %d unchanged, %d failed).",
            stats.total_chunks, stats.total_files,
            stats.new_files, stats.updated_files, stats.unchanged_files, stats.failed_files,
        )
        return stats.to_dict()

    def list_files(self, subject: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Distinct ingested files (file_name, subject, module, chunk_count),
        optionally scoped to one subject. Used by the multi-document
        features below (#54, #55, #56, #65) to know what documents exist
        to synthesize across, before running any search.
        """
        try:
            where = {"subject": {"$eq": subject}} if subject else None
            raw = self._collection.get(include=["metadatas"], where=where)
            md_list = _unwrap(raw.get("metadatas", []))
        except Exception:
            logger.exception("list_files failed.")
            return []

        by_file: dict[tuple[str, str], dict[str, Any]] = {}
        for md in md_list:
            if not md or not md.get("file_name"):
                continue
            key = (md.get("file_name"), md.get("subject", "unknown"))
            entry = by_file.setdefault(key, {
                "file_name": md.get("file_name"),
                "subject": md.get("subject", "unknown"),
                "module": md.get("module", "unknown"),
                "chunk_count": 0,
            })
            entry["chunk_count"] += 1
        return sorted(by_file.values(), key=lambda f: f["file_name"])

    def get_chunks_for_file(
        self, file_name: str, subject: Optional[str] = None, limit: int = 30,
    ) -> list[str]:
        """
        Raw chunk texts for one specific file, in original chunk order —
        NOT relevance-ranked, since "gather this document's content" and
        "find content relevant to a query" are different needs. Used for
        literature review / research-gap / document-comparison synthesis,
        where the point is representative coverage of a document, not a
        ranked subset for one query.
        """
        where: dict[str, Any] = {"file_name": {"$eq": file_name}}
        if subject:
            where = {"$and": [where, {"subject": {"$eq": subject}}]}
        try:
            raw = self._collection.get(
                where=where, include=["documents", "metadatas"], limit=max(limit, 1) * 4,
            )
        except Exception:
            logger.exception("get_chunks_for_file failed for %s.", file_name)
            return []

        docs = _unwrap(raw.get("documents", []))
        metas = _unwrap(raw.get("metadatas", []))
        paired = sorted(
            zip(docs, metas),
            key=lambda dm: (dm[1] or {}).get("chunk_number", 0),
        )
        return [d for d, _ in paired[:limit] if d]

    def get_changes_since_last_check(self, data_dir: Optional[str] = None) -> dict[str, Any]:
        """
        A dry preview of what `ingest_directory` WOULD do, without
        actually ingesting anything (backlog #57 — "what's new since I
        last checked"). Compares each file's current signature against
        what's already indexed; nothing is written to the database or the
        BM25 index either way.
        """
        resolved = data_dir or str(get_config().data_dir)
        files = get_supported_files(resolved)
        new_files: list[str] = []
        updated_files: list[str] = []
        unchanged_count = 0

        for fi in files:
            file_path = fi.get("full_path", "")
            try:
                signature = _file_signature(file_path)
            except OSError:
                continue
            existing = self._get_ingested_signature(fi)
            if existing is None:
                new_files.append(fi.get("relative_path", fi.get("file_name", file_path)))
            elif existing != signature:
                updated_files.append(fi.get("relative_path", fi.get("file_name", file_path)))
            else:
                unchanged_count += 1

        return {
            "new_files": new_files,
            "updated_files": updated_files,
            "unchanged_count": unchanged_count,
            "total_files_scanned": len(files),
        }

    # ------------------------------------------------------------------
    # Search – public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: Optional[int] = None,
        subject_filter: Optional[str] = None,
        module_filter: Optional[str] = None,
    ) -> SearchResponse:
        """
        Search the knowledge base, using hybrid mode when BM25 is enabled.

        Parameters
        ----------
        query:
            Natural-language search string.
        n_results:
            Maximum results to return (capped at ``_MAX_SEARCH_RESULTS``).
        subject_filter:
            Restrict results to a specific subject.
        module_filter:
            Restrict results to a specific module within the subject.

        Returns
        -------
        SearchResponse
            Always returns a valid object; empty on error.
        """
        if not query or not query.strip():
            logger.warning("search() called with empty query.")
            return _empty_response(query)

        n = _clamp(n_results or get_config().default_search_results, 1, _MAX_SEARCH_RESULTS)

        if self.enable_bm25:
            response = self._hybrid_search(query, n, subject_filter, module_filter)
        else:
            response = self._semantic_search(query, n, subject_filter, module_filter)

        return self._apply_feedback_weighting(response)

    def mark_feedback(self, chunk_metadata: dict[str, Any], relevant: bool) -> None:
        """
        Record that a previously-returned chunk was (ir)relevant to the
        query it was returned for (backlog #63). *chunk_metadata* is the
        metadata dict already attached to a SearchResult/SourceDocument —
        the caller doesn't need to know how chunk ids are constructed.
        """
        chunk_id = _chunk_id_from_metadata(chunk_metadata)
        self._feedback.record(chunk_id, relevant)
        logger.info(
            "Feedback recorded: chunk=%s relevant=%s", chunk_id, relevant
        )

    def _apply_feedback_weighting(self, response: SearchResponse) -> SearchResponse:
        """
        Down-weight (never zero out) results with a history of being
        marked irrelevant, then re-sort by the adjusted score.

        Never fully excludes a chunk regardless of how much negative
        feedback it has — a chunk irrelevant to one question can still be
        exactly right for a different one, so this is a demotion, not a
        blacklist. Floors at 10% of the original score after enough
        negative feedback rather than approaching zero, for the same
        reason.
        """
        if not response.results:
            return response

        feedback = self._feedback.get_all()
        if not feedback:
            return response  # no feedback recorded yet — nothing to adjust

        import dataclasses
        adjusted_results = []
        for result in response.results:
            chunk_id = _chunk_id_from_metadata(result.metadata)
            counts = feedback.get(chunk_id)
            if not counts or not counts["irrelevant_count"]:
                adjusted_results.append(result)
                continue
            multiplier = max(0.1, 1 - counts["irrelevant_count"] * 0.2)
            # SearchResult is a frozen dataclass — build a new instance
            # with the adjusted score rather than mutating in place.
            adjusted_results.append(dataclasses.replace(result, score=result.score * multiplier))

        adjusted_results.sort(key=lambda r: r.score, reverse=True)
        response.results = adjusted_results
        return response

    def _semantic_search(
        self,
        query: str,
        n_results: int,
        subject_filter: Optional[str],
        module_filter: Optional[str],
    ) -> SearchResponse:
        try:
            embeddings = self._embed([query])
            where = _build_where_filter(subject_filter, module_filter)
            total = self._collection.count()
            if total == 0:
                return _empty_response(query)

            raw = self._collection.query(
                query_embeddings=embeddings,
                n_results=min(n_results, total),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            docs = _unwrap(raw.get("documents", []))
            metas = _unwrap(raw.get("metadatas", []))
            distances = _unwrap(raw.get("distances", []))
            scores = _distances_to_scores(distances or [0.0] * len(docs))

            results = [
                SearchResult(
                    document=doc,
                    metadata=meta,
                    score=score,
                    semantic_score=score,
                )
                for doc, meta, score in zip(docs, metas, scores)
            ]
            logger.debug("Semantic search: %d result(s) for %r.", len(results), query[:60])
            return SearchResponse(results=results, query=query)

        except RAGError:
            logger.exception("Embedding failed during semantic search.")
            return _empty_response(query)
        except Exception:
            logger.exception("Semantic search failed.")
            return _empty_response(query)

    def _hybrid_search(
        self,
        query: str,
        n_results: int,
        subject_filter: Optional[str],
        module_filter: Optional[str],
        semantic_weight: Optional[float] = None,
    ) -> SearchResponse:
        cfg = get_config()
        weight = semantic_weight if semantic_weight is not None else cfg.semantic_weight
        weight = _clamp_f(weight, 0.0, 1.0)

        try:
            # ── Semantic candidates ───────────────────────────────────
            combined = self._semantic_candidates(
                query,
                n_results * _BM25_CANDIDATE_MULTIPLIER,
                subject_filter,
                module_filter,
            )

            # ── BM25 candidates ───────────────────────────────────────
            self._merge_bm25(combined, query, n_results, subject_filter, module_filter)

            # ── Hybrid score ──────────────────────────────────────────
            for entry in combined.values():
                entry["hybrid_score"] = (
                    weight * entry.get("semantic_score", 0.0)
                    + (1.0 - weight) * entry.get("bm25_score", 0.0)
                )

            ranked = sorted(
                combined.values(),
                key=lambda x: x["hybrid_score"],
                reverse=True,
            )[:n_results]

            results = [
                SearchResult(
                    document=r["document"],
                    metadata=r["metadata"],
                    score=r["hybrid_score"],
                    semantic_score=r.get("semantic_score", 0.0),
                    bm25_score=r.get("bm25_score", 0.0),
                )
                for r in ranked
            ]
            logger.debug("Hybrid search: %d result(s) for %r.", len(results), query[:60])
            return SearchResponse(results=results, query=query)

        except RAGError:
            logger.exception("Embedding failed during hybrid search.")
            return _empty_response(query)
        except Exception:
            logger.exception("Hybrid search failed.")
            return _empty_response(query)

    # ------------------------------------------------------------------
    # Search helpers (private)
    # ------------------------------------------------------------------

    def _semantic_candidates(
        self,
        query: str,
        n_results: int,
        subject_filter: Optional[str],
        module_filter: Optional[str],
    ) -> dict[str, dict[str, Any]]:
        embeddings = self._embed([query])
        where = _build_where_filter(subject_filter, module_filter)
        total = self._collection.count()
        if total == 0:
            return {}

        raw = self._collection.query(
            query_embeddings=embeddings,
            n_results=min(n_results, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = _unwrap(raw.get("documents", []))
        metas = _unwrap(raw.get("metadatas", []))
        distances = _unwrap(raw.get("distances", []))
        scores = _distances_to_scores(distances or [0.0] * len(docs))

        combined: dict[str, dict[str, Any]] = {}
        for doc, meta, score in zip(docs, metas, scores):
            did = _doc_key(meta)
            combined[did] = {
                "document": doc,
                "metadata": meta,
                "semantic_score": score,
                "bm25_score": 0.0,
            }
        return combined

    def _merge_bm25(
        self,
        combined: dict[str, dict[str, Any]],
        query: str,
        n_results: int,
        subject_filter: Optional[str],
        module_filter: Optional[str],
    ) -> None:
        with self._bm25_lock:
            if not self._bm25.ready:
                self._rebuild_bm25_locked()
            if not self._bm25.ready:
                logger.debug("BM25 index unavailable; skipping BM25 candidates.")
                return

            tokenized = query.lower().split()
            raw_scores = self._bm25.index.get_scores(tokenized)  # type: ignore[union-attr]

            candidates: list[tuple[int, float, dict]] = []
            for idx, score in enumerate(raw_scores):
                if idx >= len(self._bm25.metadata):
                    break
                meta = self._bm25.metadata[idx]
                if subject_filter and meta.get("subject") != subject_filter:
                    continue
                if module_filter and meta.get("module") != module_filter:
                    continue
                candidates.append((idx, float(score), meta))

            candidates.sort(key=lambda x: x[1], reverse=True)
            top = candidates[: n_results * _BM25_CANDIDATE_MULTIPLIER]

            if not top:
                return

            max_score = max(s for _, s, _ in top) or 1.0

            for idx, score, meta in top:
                did = _doc_key(meta)
                normalised = score / max_score
                if did in combined:
                    combined[did]["bm25_score"] = normalised
                else:
                    combined[did] = {
                        "document": self._bm25.corpus[idx],
                        "metadata": meta,
                        "semantic_score": 0.0,
                        "bm25_score": normalised,
                    }

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        with self._bm25_lock:
            self._rebuild_bm25_locked()

    def _rebuild_bm25_locked(self) -> None:
        """Must be called with ``_bm25_lock`` held."""
        try:
            raw = self._collection.get(include=["documents", "metadatas"])
            documents = _unwrap(raw.get("documents", []))
            metadatas = _unwrap(raw.get("metadatas", []))

            if not documents:
                self._bm25.clear()
                logger.debug("BM25: collection empty; index cleared.")
                return

            tokenized = [doc.lower().split() for doc in documents]
            self._bm25.index = BM25Okapi(tokenized)
            self._bm25.corpus = documents
            self._bm25.metadata = metadatas
            logger.info("BM25 index rebuilt (%d document(s)).", len(documents))

        except Exception:
            logger.exception("Failed to rebuild BM25 index.")
            self._bm25.clear()

    # ------------------------------------------------------------------
    # Ingestion helpers (private)
    # ------------------------------------------------------------------

    def _get_ingested_signature(self, file_info: dict[str, str]) -> Optional[str]:
        """
        The `file_signature` stored in metadata for this file's chunks, if
        any were previously ingested — None if the file has never been
        ingested. Used by `ingest_file` to decide new vs. updated vs.
        unchanged (backlog #61).
        """
        try:
            result = self._collection.get(
                where={
                    "$and": [
                        {"file_name": file_info.get("file_name", "")},
                        {"subject": file_info.get("subject", "unknown")},
                    ]
                },
                include=["metadatas"],
                limit=1,
            )
            metadatas = result.get("metadatas") if result else None
            if metadatas:
                return metadatas[0].get("file_signature")
            return None
        except Exception:
            logger.debug("_get_ingested_signature check failed; treating as never ingested.")
            return None

    def _delete_file_chunks(self, file_info: dict[str, str]) -> int:
        """Delete every previously-ingested chunk for this file (by name+subject)."""
        where = {
            "$and": [
                {"file_name": file_info.get("file_name", "")},
                {"subject": file_info.get("subject", "unknown")},
            ]
        }
        try:
            existing = self._collection.get(where=where, limit=10_000)
            ids = existing.get("ids", []) if existing else []
            if not ids:
                return 0
            with self._chroma_write_lock:
                self._collection.delete(ids=ids)
            return len(ids)
        except Exception:
            logger.exception(
                "_delete_file_chunks failed for %s; stale chunks may remain.",
                file_info.get("file_name"),
            )
            return 0

    def _resolve_chunk_config(self, ext: str) -> tuple[int, int]:
        """
        (chunk_size, chunk_overlap) for *ext* — the single place that
        looks up `AthenaConfig.chunk_config_by_type` and falls back to the
        global chunk_size/chunk_overlap for an extension not listed there
        (backlog #62).
        """
        by_type = getattr(get_config(), "chunk_config_by_type", {}) or {}
        return by_type.get(
            ext, (self._pdf_processor.chunk_size, self._pdf_processor.chunk_overlap)
        )

    def _extract_chunks(
        self, file_info: dict[str, str], ext: str
    ) -> list[dict[str, Any]]:
        file_path = file_info["full_path"]
        chunk_size, chunk_overlap = self._resolve_chunk_config(ext)

        if ext == ".pdf":
            return self._pdf_processor.process_pdf(
                file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

        from modules.athena.document_processor import extract_text_from_file

        pages = extract_text_from_file(file_path)
        chunks: list[dict[str, Any]] = []
        for page in pages:
            for idx, text in enumerate(
                self._pdf_processor.semantic_chunking(
                    page["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap
                ),
                start=1,
            ):
                if len(text.strip()) < _MIN_CHUNK_CHARS:
                    continue
                chunks.append({**page, "chunk_number": idx, "text": text})
        return chunks

    # ------------------------------------------------------------------
    # Collection management – public
    # ------------------------------------------------------------------

    def get_collection_stats(self) -> dict[str, Any]:
        """Return aggregate counts and a list of known subjects / modules."""
        try:
            raw = self._collection.get(include=["metadatas"])
            md_list = _unwrap(raw.get("metadatas", []))
            subjects: set[str] = set()
            modules: set[str] = set()
            for md in md_list:
                if not md:
                    continue
                if md.get("subject"):
                    subjects.add(md["subject"])
                if md.get("module"):
                    modules.add(md["module"])
            return {
                "total_chunks": len(md_list),
                "subjects": sorted(subjects),
                "modules": sorted(modules),
                "persist_directory": self.persist_directory,
                "embedding_model": self.model_name,
            }
        except Exception:
            logger.exception("get_collection_stats failed.")
            return {"total_chunks": 0, "subjects": [], "modules": []}

    def get_organization_info(self) -> dict[str, Any]:
        """Return collection stats combined with the on-disk file structure."""
        stats = self.get_collection_stats()
        try:
            structure = get_organization_structure(str(get_config().data_dir))
        except Exception:
            logger.exception("get_organization_structure failed.")
            structure = {}
        return {"database_stats": stats, "file_structure": structure}

    def clear_database(self) -> bool:
        """
        Permanently delete all documents and rebuild an empty collection.

        Returns
        -------
        bool
            ``True`` on success.
        """
        try:
            with self._chroma_write_lock:
                self._client.delete_collection(_COLLECTION_NAME)
                self._collection = self._client.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    metadata={"description": "Athena Knowledge Base"},
                )
            with self._bm25_lock:
                self._bm25.clear()
            logger.info("Collection cleared.")
            return True
        except Exception:
            logger.exception("clear_database failed.")
            return False


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _unwrap(data: Any) -> list:
    """Flatten the first level of ChromaDB's nested-list responses."""
    if not data:
        return []
    if isinstance(data[0], (list, tuple)):
        return data[0]
    return list(data)


def _build_where_filter(
    subject: Optional[str] = None,
    module: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Build a ChromaDB ``where`` clause from optional filter values."""
    conditions: list[dict[str, str]] = []
    if subject:
        conditions.append({"subject": subject})
    if module:
        conditions.append({"module": module})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _doc_key(meta: dict[str, Any]) -> str:
    """Stable deduplication key from chunk metadata."""
    return (
        f"{meta.get('file_name', 'unk')}"
        f"::p{meta.get('page_number', 0)}"
        f"::c{meta.get('chunk_number', 0)}"
    )


def _chunk_id_from_metadata(metadata: dict[str, Any]) -> str:
    """
    Reconstruct the same id `_chunk_id` would have generated at ingestion
    time, but from a SearchResult/SourceDocument's METADATA (subject,
    module, file_name, page_number, chunk_number) rather than from
    `file_info` + a fresh chunk dict — this is what `mark_feedback` and
    `_apply_feedback_weighting` use to identify a chunk that was already
    returned from a search, where only the metadata is available, not the
    original file_info used to ingest it.
    """
    return (
        f"{metadata.get('subject', 'unknown')}"
        f"::{metadata.get('module', 'unknown')}"
        f"::{metadata.get('file_name', 'file')}"
        f"::p{metadata.get('page_number', 0)}"
        f"::c{metadata.get('chunk_number', 0)}"
    )


def _chunk_id(file_info: dict[str, str], chunk: dict[str, Any]) -> str:
    """Globally unique ID for a single chunk (used as the ChromaDB document id)."""
    safe_name = Path(file_info.get("full_path", "file")).name
    return (
        f"{file_info.get('subject', 'unknown')}"
        f"::{file_info.get('module', 'unknown')}"
        f"::{safe_name}"
        f"::p{chunk.get('page_number', 0)}"
        f"::c{chunk.get('chunk_number', 0)}"
    )


def _file_signature(file_path: str) -> str:
    """
    Cheap change-detection signature: mtime + size, not a content hash.

    A full hash would mean reading every file's entire content on every
    ingestion run just to check whether it changed — for a folder of
    large PDFs, that's most of the cost of re-ingesting them anyway. A
    modified file almost always has a different mtime or size (and if
    someone deliberately preserves both while changing content, that's a
    genuinely rare edge case this trade-off accepts).
    """
    import os
    st = os.stat(file_path)
    return f"{int(st.st_mtime)}:{st.st_size}"


def _prepare_batch(
    file_info: dict[str, str],
    chunks: list[dict[str, Any]],
    file_path: str,
    file_signature: str = "",
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Convert extracted chunks into parallel id / document / metadata lists."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    base_name = Path(file_path).name

    for chunk in chunks:
        ids.append(_chunk_id(file_info, chunk))
        documents.append(chunk["text"])
        metadatas.append(
            {
                "file_name": chunk.get("file_name") or base_name,
                "file_path": chunk.get("file_path") or file_path,
                "subject": file_info.get("subject") or "unknown",
                "module": file_info.get("module") or "unknown",
                "page_number": chunk.get("page_number") or 0,
                "chunk_number": chunk.get("chunk_number") or 0,
                "total_pages": chunk.get("total_pages") or 0,
                # backlog #61 — read back by _get_ingested_signature on the
                # next ingestion run to detect whether this file changed.
                "file_signature": file_signature,
            }
        )

    return ids, documents, metadatas


def _distances_to_scores(distances: list[float]) -> list[float]:
    """
    Convert L2 distances from ChromaDB to [0, 1] similarity scores.

    With L2-normalised embeddings, cosine distance ∈ [0, 2], so:
        similarity = 1 - distance / 2

    This is numerically stable and does not artificially compress the
    score range the way min-max normalisation does when results cluster.
    """
    scores: list[float] = []
    for d in distances:
        if not isfinite(d):
            scores.append(_SCORE_MIN)
        else:
            scores.append(max(_SCORE_MIN, min(_SCORE_MAX, 1.0 - d / 2.0)))
    return scores


def _empty_response(query: str) -> SearchResponse:
    return SearchResponse(results=[], query=query)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _clamp_f(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))