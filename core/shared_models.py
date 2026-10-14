"""
core/shared_models.py

One process-wide cache for SentenceTransformer models.

Why this exists
---------------
Mnemosyne, Athena, the intent classifier and Pluto each used to construct
their own ``SentenceTransformer("all-MiniLM-L6-v2")``. Four independent
copies of the same weights (plus four copies of the tokenizer and inference
buffers) sit in RAM for no benefit. ``get_sentence_transformer`` loads a
given (model, device) pair once and hands every caller the same instance.

Behaviour notes
---------------
* Thread-safe: concurrent first calls load the model exactly once.
* ``sentence_transformers`` is imported on every call (cheap once it is in
  ``sys.modules``) so a missing library still raises ``ImportError`` at the
  call site, exactly as a direct ``from sentence_transformers import ...``
  would. Callers that already wrap model loading in try/except keep working.
* ``device=None`` auto-picks CUDA when available, else CPU.
* ``clear_cache()`` drops every cached model (used by tests and available to
  callers that want to free RAM, e.g. after a long idle period).
"""
from __future__ import annotations

import gc
import logging
import threading
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_cache: Dict[Tuple[str, str], Any] = {}
_lock = threading.Lock()


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_sentence_transformer(model_name: str, device: Optional[str] = None) -> Any:
    """Return the shared ``SentenceTransformer`` for *model_name* / *device*."""
    # Import first, outside the cache lookup: if the library is missing (or a
    # test has blocked it) this raises ImportError just like a direct import.
    from sentence_transformers import SentenceTransformer

    resolved = _resolve_device(device)
    key = (model_name, resolved)

    model = _cache.get(key)
    if model is not None:
        return model

    with _lock:
        model = _cache.get(key)
        if model is None:
            model = SentenceTransformer(model_name, device=resolved)
            _cache[key] = model
            logger.info("Shared embedding model loaded on %s: %s", resolved, model_name)
    return model


def loaded_models() -> list[Tuple[str, str]]:
    """``[(model_name, device), ...]`` currently held in the cache."""
    with _lock:
        return list(_cache.keys())


def clear_cache() -> None:
    """Drop every cached model so its memory can be reclaimed."""
    with _lock:
        _cache.clear()
    gc.collect()
