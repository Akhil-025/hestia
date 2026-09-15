"""

modules/iris/embeddings.py

CLIP-based semantic image search for Iris, replacing the caption-only
search noted as a limitation in README.md's roadmap ("CLIP-based
semantic image search"). Both `sentence-transformers` and `chromadb`
were already in requirements.txt and `chroma_dir` was already wired
up in config/laptop_config.yaml — this was clearly planned but never
actually built; this file builds it.

Two pieces:
  - ClipEmbedder: wraps a sentence-transformers CLIP model
    ("clip-ViT-B-32") to embed both images and text queries into the
    same vector space, so "find photos of a dog on a beach" can be
    compared directly against image embeddings without needing a
    caption to have mentioned "dog" or "beach".
  - ImageVectorIndex: a thin wrapper around a persistent Chroma
    collection at `IrisConfig.chroma_dir`, storing one embedding per
    ingested image keyed by Iris's own file_id.

Both degrade to a clearly-flagged "unavailable" state (never raise
past their public methods) if `sentence-transformers`/`torch` or
`chromadb` aren't installed, or if the CLIP model can't be downloaded
(offline machine, first run with no cached weights, etc.) — Iris as a
whole must keep working on caption/tag search alone in that case.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CLIP_MODEL_NAME = "clip-ViT-B-32"
_COLLECTION_NAME = "iris_images"


class ClipEmbedder:
    """
    Lazily-loaded CLIP text/image encoder. The model is not loaded at
    construction time (import + first-run weight download can take
    several seconds to minutes) — only on first actual `embed_*` call,
    so IrisEngine.__init__ stays fast and doesn't fail Iris entirely if
    the model can't be loaded on a given machine.
    """

    def __init__(self, model_name: str = _CLIP_MODEL_NAME):
        self._model_name = model_name
        self._model = None
        self._load_failed = False

    @property
    def available(self) -> bool:
        """True once a working model is loaded; does not itself trigger a load."""
        return self._model is not None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info("[Iris] CLIP model %r loaded.", self._model_name)
            return True
        except Exception as e:
            self._load_failed = True
            logger.warning(
                "[Iris] CLIP model unavailable (%s) — semantic image search "
                "disabled; falling back to caption/tag search only.", e
            )
            return False

    def embed_image(self, path: "str | Path") -> Optional[list[float]]:
        if not self._ensure_loaded():
            return None
        try:
            from PIL import Image
            with Image.open(path) as img:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                vector = self._model.encode(img, convert_to_numpy=True)
            return vector.tolist()
        except Exception as e:
            logger.warning("[Iris] Failed to embed image %s: %s", path, e)
            return None

    def embed_text(self, text: str) -> Optional[list[float]]:
        if not text or not text.strip():
            return None
        if not self._ensure_loaded():
            return None
        try:
            vector = self._model.encode(text, convert_to_numpy=True)
            return vector.tolist()
        except Exception as e:
            logger.warning("[Iris] Failed to embed text query %r: %s", text, e)
            return None


class ImageVectorIndex:
    """Persistent Chroma-backed store of one CLIP embedding per Iris file_id."""

    def __init__(self, chroma_dir: "str | Path"):
        self._chroma_dir = str(chroma_dir)
        self._client = None
        self._collection = None
        self._init_failed = False

    @property
    def available(self) -> bool:
        return self._ensure_client()

    def _ensure_client(self) -> bool:
        if self._collection is not None:
            return True
        if self._init_failed:
            return False
        try:
            import chromadb
            Path(self._chroma_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._chroma_dir)
            self._collection = self._client.get_or_create_collection(_COLLECTION_NAME)
            return True
        except Exception as e:
            self._init_failed = True
            logger.warning("[Iris] Chroma vector index unavailable: %s", e)
            return False

    def upsert(self, file_id: int, embedding: list[float]) -> bool:
        if not self._ensure_client():
            return False
        try:
            self._collection.upsert(
                ids=[str(file_id)],
                embeddings=[embedding],
                metadatas=[{"file_id": file_id}],
            )
            return True
        except Exception as e:
            logger.warning("[Iris] Failed to store embedding for file_id=%s: %s", file_id, e)
            return False

    def query(self, embedding: list[float], top_k: int = 10) -> list[tuple[int, float]]:
        """Return [(file_id, distance), ...] nearest neighbours, ascending distance."""
        if not self._ensure_client():
            return []
        try:
            count = self._collection.count()
            if count == 0:
                return []
            result = self._collection.query(
                query_embeddings=[embedding], n_results=min(top_k, count)
            )
        except Exception as e:
            logger.warning("[Iris] Vector query failed: %s", e)
            return []

        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        out: list[tuple[int, float]] = []
        for raw_id, dist in zip(ids, distances):
            try:
                out.append((int(raw_id), float(dist)))
            except (TypeError, ValueError):
                continue
        return out

    def delete(self, file_id: int) -> None:
        if not self._ensure_client():
            return
        try:
            self._collection.delete(ids=[str(file_id)])
        except Exception as e:
            logger.warning("[Iris] Failed to delete embedding for file_id=%s: %s", file_id, e)
