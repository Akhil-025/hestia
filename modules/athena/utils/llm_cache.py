"""
modules/athena/utils/llm_cache.py

LLM response caching — fixed for Hestia (relative config import).
"""
import json
import hashlib
from pathlib import Path

from modules.athena.config import get_config, paths


def question_hash(question: str, context_ids: list) -> str:
    key = question + "|" + "|".join(context_ids or [])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load_cached_answer(qhash: str):
    cache_file = paths.CACHE_DIR / f"{qhash}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Cache read error: {e}")
    return None


def save_cached_answer(qhash: str, payload: dict):
    cache_file = paths.CACHE_DIR / f"{qhash}.json"
    try:
        cache_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"Cache write error: {e}")
