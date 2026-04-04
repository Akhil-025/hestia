"""
modules/athena/config.py

Self-contained config for Athena when running inside Hestia.
No external config.json required — all defaults are sensible.
Override any value by passing kwargs to AthenaConfig().
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Root of the Athena module (modules/athena/)
_MODULE_DIR = Path(__file__).parent

# Hestia project root (two levels up: modules/athena → modules → hestia/)
_PROJECT_ROOT = _MODULE_DIR.parent.parent


@dataclass
class AthenaConfig:
    # ── Storage ──────────────────────────────────────────────────────────────
    # ChromaDB lives inside Hestia's data folder, not Athena's old location.
    chroma_persist_dir: str = str(_PROJECT_ROOT / "data" / "athena" / "chroma_db")
    data_dir: str           = str(_PROJECT_ROOT / "data" / "athena" / "documents")
    cache_dir: str          = str(_PROJECT_ROOT / "data" / "athena" / "cache")

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_model: str  = "all-MiniLM-L6-v2"
    embed_batch_size: int = 32

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int    = 1000
    chunk_overlap: int = 200

    # ── Retrieval ────────────────────────────────────────────────────────────
    default_search_results: int = 10
    max_chunks_local: int       = 5
    max_chunks_cloud: int       = 3
    enable_bm25: bool           = True
    semantic_weight: float      = 0.7   # 0.0 = pure BM25, 1.0 = pure semantic

    def __post_init__(self) -> None:
        # Ensure all directories exist at startup
        for attr in ("chroma_persist_dir", "data_dir", "cache_dir"):
            Path(getattr(self, attr)).mkdir(parents=True, exist_ok=True)


# ── Paths helper (used by llm_cache.py) ──────────────────────────────────────
class _Paths:
    def __init__(self, cfg: AthenaConfig) -> None:
        self._cfg = cfg

    @property
    def CACHE_DIR(self) -> Path:
        return Path(self._cfg.cache_dir)


# ── Module-level singletons ───────────────────────────────────────────────────
_config: AthenaConfig | None = None


def get_config() -> AthenaConfig:
    """Return the module-level AthenaConfig singleton."""
    global _config
    if _config is None:
        _config = AthenaConfig()
    return _config


def set_config(cfg: AthenaConfig) -> None:
    """Replace the singleton (useful for testing or custom paths)."""
    global _config
    _config = cfg


# Singleton paths object — matches the old `from config import paths` usage
paths = _Paths(get_config())
