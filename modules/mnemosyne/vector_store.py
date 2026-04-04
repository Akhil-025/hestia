"""
modules/mnemosyne/vector_store.py

ChromaDB vector store for Mnemosyne, using SentenceTransformers for embeddings.
"""
import logging
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "mnemosyne_memory"

class MnemosyneVectorStore:
    def __init__(self, chroma_dir: str, embedding_model: str):
        self.chroma_dir = chroma_dir
        self.embedding_model_name = embedding_model
        self._lock = threading.Lock()
        self._embedder: Optional[SentenceTransformer] = None
        self._initialize_chroma()
        logger.info(f"MnemosyneVectorStore initialized (dir={chroma_dir}, model={embedding_model})")

    def _initialize_chroma(self):
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"description": "Mnemosyne Memory Store"},
        )

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedder = SentenceTransformer(self.embedding_model_name, device=device)
            logger.info(f"Embedding model loaded on {device}: {self.embedding_model_name}")
        return self._embedder

    def _embed(self, texts: List[str]) -> List[List[float]]:
        embedder = self._get_embedder()
        emb = embedder.encode(texts, show_progress_bar=False)
        if hasattr(emb, "tolist"):
            return emb.tolist()
        return [list(e) for e in emb]

    def add(self, text: str, metadata: dict, doc_id: str) -> None:
        # Ensure required metadata
        if "type" not in metadata or "created_at" not in metadata:
            raise ValueError("metadata must include 'type' and 'created_at'")
        with self._lock:
            embedding = self._embed([text])[0]
            self.collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],
            )

    def search(self, query: str, n_results: int = 5, where: dict = None) -> List[dict]:
        if self.collection.count() == 0:
            return []
        embedding = self._embed([query])[0]
        try:
            raw = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results, self.collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.debug(f"ChromaDB query failed: {e}")
            return []
        docs      = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        ids       = raw.get("ids", [[]])[0]
        scores    = self._distances_to_scores(distances)
        logger.info(f"MnemosyneVectorStore search returned {len(docs)} results")
        return [
            {"text": text, "metadata": meta, "score": score, "id": doc_id}
            for text, meta, score, doc_id in zip(docs, metadatas, scores, ids)
        ]

    def delete(self, doc_id: str) -> None:
        with self._lock:
            self.collection.delete(ids=[doc_id])

    @staticmethod
    def _distances_to_scores(distances: List[float]) -> List[float]:
        from math import isfinite
        finite = [d for d in distances if isfinite(d)]
        if not finite:
            return [0.0] * len(distances)
        min_d, max_d = min(finite), max(finite)
        denom = (max_d - min_d) or 1.0
        return [max(0.0, min(1.0, 1.0 - (d - min_d) / denom)) if isfinite(d) else 0.0 for d in distances]
