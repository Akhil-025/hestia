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
import os
import tempfile
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

    def record(self, file_info: dict[str, str], chunks: int) -> None:
        self.total_files += 1
        self.total_chunks += chunks

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
    ) -> int:
        """
        Ingest a single file into ChromaDB.

        The operation is idempotent: if the file is already present it is
        skipped without touching the database.

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
        int
            Number of chunks added (0 if skipped or on error).
        """
        file_path = file_info.get("full_path", "")
        if not file_path:
            logger.warning("ingest_file: file_info missing 'full_path'; skipping.")
            return 0

        if self._is_ingested(file_info):
            logger.debug("Already ingested: %s", file_path)
            return 0

        ext = Path(file_path).suffix.lower()

        try:
            chunks = self._extract_chunks(file_info, ext)
        except Exception:
            logger.exception("Chunk extraction failed for %s.", file_path)
            return 0

        if not chunks:
            logger.warning("No text extracted from %s.", file_path)
            return 0

        ids, documents, metadatas = _prepare_batch(file_info, chunks, file_path)

        try:
            embeddings = self._embed(documents)
        except RAGError:
            logger.exception("Embedding failed for %s; skipping ingestion.", file_path)
            return 0

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
            return 0

        logger.info("Ingested %d chunk(s) from %s.", len(chunks), file_path)

        if self.enable_bm25 and rebuild_bm25:
            self._rebuild_bm25()

        return len(chunks)

    # Backward compatibility alias
    ingest_pdf = ingest_file

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
            Aggregated ingestion statistics.
        """
        resolved = data_dir or str(get_config().data_dir)
        files = get_supported_files(resolved)
        stats = IngestionStats()

        if not files:
            logger.warning("No supported files found in %s.", resolved)
            return stats.to_dict()

        for fi in files:
            n = self.ingest_file(fi, rebuild_bm25=False)
            stats.record(fi, n)

        if self.enable_bm25 and rebuild_bm25:
            self._rebuild_bm25()

        logger.info(
            "Directory ingestion complete: %d chunk(s) from %d file(s).",
            stats.total_chunks,
            stats.total_files,
        )
        return stats.to_dict()

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
            return self._hybrid_search(query, n, subject_filter, module_filter)
        return self._semantic_search(query, n, subject_filter, module_filter)

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

    def _is_ingested(self, file_info: dict[str, str]) -> bool:
        try:
            result = self._collection.get(
                where={
                    "$and": [
                        {"file_name": file_info.get("file_name", "")},
                        {"subject": file_info.get("subject", "unknown")},
                    ]
                },
                limit=1,
            )
            return bool(result and result.get("ids"))
        except Exception:
            logger.debug("_is_ingested check failed; assuming not ingested.")
            return False

    def _extract_chunks(
        self, file_info: dict[str, str], ext: str
    ) -> list[dict[str, Any]]:
        file_path = file_info["full_path"]

        if ext == ".pdf":
            return self._pdf_processor.process_pdf(file_path)

        from modules.athena.document_processor import extract_text_from_file

        pages = extract_text_from_file(file_path)
        chunks: list[dict[str, Any]] = []
        for page in pages:
            for idx, text in enumerate(
                self._pdf_processor.semantic_chunking(page["text"]), start=1
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


def _prepare_batch(
    file_info: dict[str, str],
    chunks: list[dict[str, Any]],
    file_path: str,
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