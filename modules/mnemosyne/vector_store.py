"""
modules/mnemosyne/vector_store.py

ChromaDB vector store for Mnemosyne, using SentenceTransformers for embeddings.
"""
import logging
import threading
from typing import Optional, List

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

    def add(self, text: str, metadata: dict, doc_id: str, embed_text: Optional[str] = None) -> None:
        """
        Store *text* as the document. *embed_text*, when given, is what gets
        embedded instead (the stored/returned document stays *text*). Facts
        use this so a question about the KEY ("my favourite color") can match
        a fact whose stored value is just "teal".
        """
        # Ensure required metadata
        if "type" not in metadata or "created_at" not in metadata:
            raise ValueError("metadata must include 'type' and 'created_at'")
        with self._lock:
            embedding = self._embed([embed_text if embed_text else text])[0]
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
            logger.error(f"ChromaDB query failed: {e}", exc_info=True)
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

    def similarity(self, a: str, b: str) -> float:
        """
        Absolute similarity between two texts on the same 1/(1+d) scale
        search() reports (d = Chroma's squared-L2), so thresholds written
        against search scores (e.g. the #42 dedupe cut-off) stay valid.
        """
        va, vb = self._embed([a, b])
        d = sum((x - y) ** 2 for x, y in zip(va, vb))
        return self._distances_to_scores([d])[0]

    def reembed_facts(self) -> int:
        """
        Re-embed every stored fact as "<key words>: <value>" (documents and
        metadata untouched). Facts written before key-aware embedding were
        embedded from the bare value; run this once to upgrade them.
        """
        with self._lock:
            raw = self.collection.get(where={"type": {"$eq": "fact"}}, include=["documents", "metadatas"])
            ids = raw.get("ids") or []
            docs = raw.get("documents") or []
            metas = raw.get("metadatas") or []
            n = 0
            for doc_id, doc, meta in zip(ids, docs, metas):
                key = (meta or {}).get("key") or doc_id
                emb = self._embed([f"{str(key).replace('_', ' ')}: {doc}"])[0]
                self.collection.upsert(ids=[doc_id], documents=[doc], metadatas=[meta], embeddings=[emb])
                n += 1
            return n

    @staticmethod
    def _distances_to_scores(distances: List[float]) -> List[float]:
        # NOTE: this used to min-max normalize *within the returned batch*
        # (score = 1 - (d - min(batch)) / (max(batch) - min(batch))). That
        # guarantees the single closest result in any batch scores 1.0 no
        # matter how far away it actually is — so a caller trying to reject
        # "nothing here is relevant" by thresholding this score can never
        # succeed, since the best-of-a-bad-batch always looks perfect. This
        # collection is created without an explicit hnsw:space (see
        # _initialize_chroma), so Chroma defaults to squared L2 distance,
        # not cosine — there's no fixed [0, 2] range to rescale against.
        # Converted to an absolute, batch-independent similarity instead:
        # 1/(1 + distance) is monotonically decreasing, distance=0 -> 1.0,
        # and it doesn't rescale based on what else happened to be in this
        # particular result set. Embedding magnitudes are roughly stable
        # for a fixed sentence-transformers model, so a fixed downstream
        # threshold (see MnemosyneEngine._MIN_RELEVANCE) is meaningful
        # against this — tune it empirically against your model's actual
        # distance distribution for relevant vs. irrelevant pairs.
        from math import isfinite
        return [1.0 / (1.0 + d) if isfinite(d) and d >= 0 else 0.0 for d in distances]