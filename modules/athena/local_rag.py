"""
modules/athena/local_rag.py

ChromaDB vector store + SentenceTransformers + optional BM25 hybrid search.
Fixed for Hestia integration: all imports are now relative.
"""

import logging
import os
from math import isfinite
from typing import Any, Dict, List, Optional

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from modules.athena.config import get_config
from modules.athena.pdf_processor import (
    PDFProcessor,
    get_supported_files,
    get_organization_structure,
)

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "engineering_documents"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap_chroma(data: Any) -> list:
    if not data:
        return []
    if isinstance(data[0], (list, tuple)):
        return data[0]
    return data


def _build_where_filter(
    subject: Optional[str] = None,
    module: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    conditions: List[Dict[str, str]] = []
    if subject:
        conditions.append({"subject": subject})
    if module:
        conditions.append({"module": module})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _empty_result(query: str) -> Dict[str, Any]:
    return {
        "documents": [],
        "metadatas": [],
        "scores": [],
        "semantic_scores": [],
        "bm25_scores": [],
        "query": query,
        "total_results": 0,
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MergedLocalRAG:
    """Hybrid retrieval engine: dense embeddings (ChromaDB) + sparse BM25."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        model_name: Optional[str] = None,
        embed_batch_size: Optional[int] = None,
        enable_bm25: Optional[bool] = None,
    ) -> None:
        config = get_config()

        self.persist_directory: str = persist_directory or config.chroma_persist_dir
        self.model_name: str        = model_name        or config.embedding_model
        self.embed_batch_size: int  = embed_batch_size  or config.embed_batch_size
        self.enable_bm25: bool = (
            enable_bm25 if enable_bm25 is not None else config.enable_bm25
        )

        self._initialize_chroma()
        self._initialize_embedder()
        self.pdf_processor = PDFProcessor()

        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[str] = []
        self.bm25_metadata: List[Dict[str, Any]] = []

        logger.info(
            "MergedLocalRAG ready (model=%s, bm25=%s)",
            self.model_name, self.enable_bm25,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_chroma(self) -> None:
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"description": "Athena Knowledge Base"},
            )
            logger.info("ChromaDB initialized (%s)", self.persist_directory)
        except Exception:
            logger.exception("Failed to initialize ChromaDB")
            raise

    def _initialize_embedder(self) -> None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.embedder = SentenceTransformer(self.model_name, device=device)
            logger.info("Embedding model loaded on %s: %s", device, self.model_name)
        except Exception:
            logger.exception("Failed to load embedding model: %s", self.model_name)
            raise

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            emb = self.embedder.encode(batch, show_progress_bar=False)
            emb_list = emb.tolist() if hasattr(emb, "tolist") else [list(e) for e in emb]
            all_embeddings.extend(emb_list)
        return all_embeddings

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_id(file_info: Dict[str, str], chunk: Dict[str, Any]) -> str:
        safe_name = os.path.basename(file_info.get("full_path", "file"))
        return (
            f"{file_info.get('subject', 'unknown')}"
            f"::{file_info.get('module', 'unknown')}"
            f"::{safe_name}"
            f"::p{chunk.get('page_number', 0)}"
            f"::c{chunk.get('chunk_number', 0)}"
        )

    def _is_already_ingested(self, file_info: Dict[str, str]) -> bool:
        try:
            result = self.collection.get(
                where={
                    "$and": [
                        {"file_name": file_info["file_name"]},
                        {"subject": file_info.get("subject", "unknown")},
                    ]
                },
                limit=1,
            )
            return bool(result and result.get("ids"))
        except Exception:
            return False

    def ingest_file(self, file_info: Dict[str, str], rebuild_bm25: bool = True) -> int:
        if self._is_already_ingested(file_info):
            logger.info("Already ingested: %s", file_info["file_name"])
            return 0

        file_path: str = file_info["full_path"]
        ext = os.path.splitext(file_path)[1].lower()

        try:
            chunks = self._extract_chunks(file_info, ext)
            if not chunks:
                logger.warning("No text extracted from %s", file_path)
                return 0

            ids, documents, metadatas = self._prepare_batch(file_info, chunks, file_path)
            embeddings = self._embed_texts(documents)

            self.collection.add(
                ids=ids, documents=documents,
                metadatas=metadatas, embeddings=embeddings,
            )
            logger.info("Added %d chunks from %s", len(chunks), file_path)

            if self.enable_bm25 and rebuild_bm25:
                self._rebuild_bm25_index()

            return len(chunks)

        except Exception:
            logger.exception("Failed to ingest %s", file_path)
            return 0

    ingest_pdf = ingest_file  # backward compat

    def _extract_chunks(self, file_info: Dict[str, str], ext: str) -> List[Dict[str, Any]]:
        file_path = file_info["full_path"]

        if ext == ".pdf":
            return self.pdf_processor.process_pdf(file_path)

        from modules.athena.document_processor import extract_text_from_file
        pages = extract_text_from_file(file_path)
        chunks: List[Dict[str, Any]] = []
        for page in pages:
            for idx, text in enumerate(
                self.pdf_processor.semantic_chunking(page["text"]), start=1
            ):
                if len(text.strip()) < 40:
                    continue
                chunks.append({**page, "chunk_number": idx, "text": text})
        return chunks

    @staticmethod
    def _prepare_batch(
        file_info: Dict[str, str],
        chunks: List[Dict[str, Any]],
        file_path: str,
    ) -> tuple:
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(MergedLocalRAG._chunk_id(file_info, chunk))
            documents.append(chunk["text"])
            metadatas.append({
                "file_name":    chunk.get("file_name", os.path.basename(file_path)),
                "file_path":    chunk.get("file_path", file_path),
                "subject":      file_info.get("subject"),
                "module":       file_info.get("module"),
                "page_number":  chunk.get("page_number"),
                "chunk_number": chunk.get("chunk_number"),
                "total_pages":  chunk.get("total_pages"),
            })

        return ids, documents, metadatas

    def ingest_directory(
        self,
        data_dir: Optional[str] = None,
        rebuild_bm25: bool = True,
    ) -> Dict[str, Any]:
        if data_dir is None:
            data_dir = str(get_config().data_dir)

        files = get_supported_files(data_dir)
        results: Dict[str, Any] = {
            "total_files": 0, "total_chunks": 0,
            "by_subject": {}, "by_module": {},
        }

        if not files:
            logger.warning("No supported files in %s", data_dir)
            return results

        for fi in files:
            n = self.ingest_file(fi, rebuild_bm25=False)
            results["total_files"] += 1
            results["total_chunks"] += n

            subj = fi.get("subject", "unknown")
            results["by_subject"].setdefault(subj, {"files": 0, "chunks": 0})
            results["by_subject"][subj]["files"]  += 1
            results["by_subject"][subj]["chunks"] += n

            mod_key = f"{subj}/{fi.get('module', 'unknown')}"
            results["by_module"].setdefault(mod_key, {"files": 0, "chunks": 0})
            results["by_module"][mod_key]["files"]  += 1
            results["by_module"][mod_key]["chunks"] += n

        if self.enable_bm25 and rebuild_bm25:
            self._rebuild_bm25_index()

        logger.info(
            "Ingested %d chunks from %d files",
            results["total_chunks"], results["total_files"],
        )
        return results

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------

    def _rebuild_bm25_index(self) -> None:
        try:
            raw = self.collection.get(include=["documents", "metadatas"])
            documents = _unwrap_chroma(raw.get("documents", []))
            metadatas = _unwrap_chroma(raw.get("metadatas", []))

            if not documents:
                self.bm25 = None
                self.bm25_corpus = []
                self.bm25_metadata = []
                return

            tokenized = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized)
            self.bm25_corpus = documents
            self.bm25_metadata = metadatas
            logger.info("BM25 index built (%d documents)", len(documents))

        except Exception:
            logger.exception("Failed to build BM25 index")
            self.bm25 = None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: Optional[int] = None,
        subject_filter: Optional[str] = None,
        module_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        n_results = n_results or get_config().default_search_results
        n_results = min(n_results, 3)

        if self.enable_bm25:
            return self.hybrid_search(query, n_results, subject_filter, module_filter)
        return self._semantic_search(query, n_results, subject_filter, module_filter)

    def _semantic_search(
        self,
        query: str,
        n_results: int,
        subject_filter: Optional[str] = None,
        module_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            emb = self._embed_texts([query])
            where = _build_where_filter(subject_filter, module_filter)

            raw = self.collection.query(
                query_embeddings=emb,
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            docs  = _unwrap_chroma(raw.get("documents", []))
            mds   = _unwrap_chroma(raw.get("metadatas", []))
            dists = _unwrap_chroma(raw.get("distances", []))
            scores = self._distances_to_scores(dists or [0.0] * len(docs))

            return {
                "documents": docs, "metadatas": mds,
                "scores": scores, "semantic_scores": scores,
                "bm25_scores": [0.0] * len(docs),
                "query": query, "total_results": len(docs),
            }
        except Exception:
            logger.exception("Semantic search failed")
            return _empty_result(query)

    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        subject_filter: Optional[str] = None,
        module_filter: Optional[str] = None,
        semantic_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        config = get_config()
        semantic_weight = (
            semantic_weight if semantic_weight is not None else config.semantic_weight
        )

        try:
            combined = self._semantic_candidates(query, n_results, subject_filter, module_filter)

            if self.enable_bm25:
                self._merge_bm25_candidates(combined, query, n_results, subject_filter, module_filter)

            for entry in combined.values():
                entry["hybrid_score"] = (
                    semantic_weight * entry.get("semantic_score", 0.0)
                    + (1.0 - semantic_weight) * entry.get("bm25_score", 0.0)
                )

            ranked = sorted(
                combined.values(), key=lambda x: x["hybrid_score"], reverse=True
            )[:n_results]

            return {
                "documents":       [r["document"]                    for r in ranked],
                "metadatas":       [r["metadata"]                    for r in ranked],
                "scores":          [r["hybrid_score"]                for r in ranked],
                "semantic_scores": [r.get("semantic_score", 0.0)     for r in ranked],
                "bm25_scores":     [r.get("bm25_score", 0.0)         for r in ranked],
                "query":           query,
                "total_results":   len(ranked),
            }

        except Exception:
            logger.exception("Hybrid search failed")
            return _empty_result(query)

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def _semantic_candidates(self, query, n_results, subject_filter, module_filter):
        emb = self._embed_texts([query])
        where = _build_where_filter(subject_filter, module_filter)

        raw = self.collection.query(
            query_embeddings=emb,
            n_results=n_results * 3,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs  = _unwrap_chroma(raw.get("documents", []))
        mds   = _unwrap_chroma(raw.get("metadatas", []))
        dists = _unwrap_chroma(raw.get("distances", []))
        scores = self._distances_to_scores(dists or [0.0] * len(docs))

        combined: Dict[str, Dict[str, Any]] = {}
        for doc, md, score in zip(docs, mds, scores):
            did = self._doc_id(md)
            combined[did] = {
                "document": doc, "metadata": md,
                "semantic_score": score, "bm25_score": 0.0,
            }
        return combined

    def _merge_bm25_candidates(self, combined, query, n_results, subject_filter, module_filter):
        if self.bm25 is None:
            self._rebuild_bm25_index()
        if self.bm25 is None:
            return

        tokenized_q = query.lower().split()
        raw_scores  = self.bm25.get_scores(tokenized_q)

        candidates: List[tuple] = []
        for idx, score in enumerate(raw_scores):
            md = self.bm25_metadata[idx] if idx < len(self.bm25_metadata) else {}
            if subject_filter and md.get("subject") != subject_filter:
                continue
            if module_filter and md.get("module") != module_filter:
                continue
            candidates.append((idx, score, md))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[: n_results * 3]
        max_score = max((s for _, s, _ in top), default=1.0) or 1.0

        for idx, score, md in top:
            did = self._doc_id(md)
            normalised = score / max_score
            if did in combined:
                combined[did]["bm25_score"] = normalised
            else:
                combined[did] = {
                    "document":       self.bm25_corpus[idx],
                    "metadata":       md,
                    "semantic_score": 0.0,
                    "bm25_score":     normalised,
                }

    @staticmethod
    def _doc_id(md: Dict[str, Any]) -> str:
        return (
            f"{md.get('file_name', 'unk')}"
            f"_p{md.get('page_number', 0)}"
            f"_c{md.get('chunk_number', 0)}"
        )

    @staticmethod
    def _distances_to_scores(distances: List[float]) -> List[float]:
        finite = [d for d in distances if isfinite(d)]
        if not finite:
            return [0.0] * len(distances)
        min_d, max_d = min(finite), max(finite)
        denom = (max_d - min_d) or 1.0
        return [
            max(0.0, min(1.0, 1.0 - (d - min_d) / denom)) if isfinite(d) else 0.0
            for d in distances
        ]

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            raw = self.collection.get(include=["metadatas"])
            md_list = _unwrap_chroma(raw.get("metadatas", []))
            subjects: set = set()
            modules:  set = set()
            for md in md_list:
                if not md:
                    continue
                if md.get("subject"):
                    subjects.add(md["subject"])
                if md.get("module"):
                    modules.add(md["module"])
            return {
                "total_chunks":      len(md_list),
                "subjects":          sorted(subjects),
                "modules":           sorted(modules),
                "persist_directory": self.persist_directory,
                "embedding_model":   self.model_name,
            }
        except Exception:
            logger.exception("Failed to get collection stats")
            return {"total_chunks": 0, "subjects": [], "modules": []}

    def get_organization_info(self) -> Dict[str, Any]:
        stats = self.get_collection_stats()
        try:
            structure = get_organization_structure(str(get_config().data_dir))
        except Exception:
            structure = {}
        return {"database_stats": stats, "file_structure": structure}

    def clear_database(self) -> bool:
        try:
            self.client.delete_collection(_COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(name=_COLLECTION_NAME)
            self.bm25 = None
            self.bm25_corpus = []
            self.bm25_metadata = []
            logger.info("Database cleared")
            return True
        except Exception:
            logger.exception("Failed to clear database")
            return False
