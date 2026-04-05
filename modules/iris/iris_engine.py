"""
modules/iris/iris_engine.py

IrisEngine: single entry point for Iris image/video/audio ingestion and search.
"""
from modules.base import BaseModule  
import logging
import asyncio
import json
from .config import get_config, IrisConfig
from pathlib import Path
from .db import IrisDB
from .ingestion import FileIngestor
try:
    from .analyser import IrisAnalyser
except ImportError:
    IrisAnalyser = None
    logging.warning("[Iris] requests library not available, IrisAnalyser disabled.")

logger = logging.getLogger(__name__)

class IrisEngine(BaseModule):               
    name = "iris"

    _SEARCH_KEYWORDS  = {"photo", "picture", "image", "media", "gallery"}
    _INGEST_KEYWORDS  = {"ingest", "import", "add", "index", "scan", "catalog"}
    _ANALYSE_KEYWORDS = {"analyse", "analyze", "describe", "caption"}

    def __init__(self, hestia_llm=None):
        try:
            self.config: IrisConfig = get_config()
            self.db = IrisDB(self.config.db_path)
            self.ingestor = FileIngestor(self.config, self.db)
            self.hestia_llm = hestia_llm
            ollama_host = "127.0.0.1"
            ollama_port = 11434
            self.analyser = IrisAnalyser(self.db, ollama_host, ollama_port, "llava:7b") if IrisAnalyser else None
            logger.info("[Iris] Engine ready")
        except Exception as e:
            logger.error(f"[Iris] Engine init failed: {e}")
            raise


    def can_handle(self, intent: str) -> bool:   
        return intent in {"iris_search", "iris_ingest", "iris_analyse", "iris_status", "iris_query"}

    def handle(self, intent: str, entities: dict, context: dict) -> dict:   
        raw = entities.get("raw_query", context.get("raw_query", "")).lower()

        if intent == "iris_ingest" or any(k in raw for k in self._INGEST_KEYWORDS):
            stats = self.ingest()
            return {
                "response": (
                    f"Media ingestion complete. "
                    f"{stats.get('ingested', 0)} files processed, "
                    f"{stats.get('duplicates_skipped', 0)} duplicates skipped."
                ),
                "data": stats,
                "confidence": 1.0,
            }
        elif intent == "iris_analyse" or any(k in raw for k in self._ANALYSE_KEYWORDS):
            response = self.analyse(limit=20)
            return {"response": response, "data": {}, "confidence": 0.9}
        else:
            result = self.search(raw or entities.get("query", ""))
            return {
                "response": result or "No matching media found.",
                "data": {},
                "confidence": 0.85 if result else 0.3,
            }

    def get_context(self) -> dict:            
        try:
            s = self.stats()
            return {
                "iris_total":    s.get("total_files", 0),
                "iris_processed": s.get("processed", 0),
                "iris_pending":  s.get("pending", 0),
            }
        except Exception:
            return {}

    def ingest(self, source_dir: str = None) -> dict:
        try:
            dir_to_use = source_dir or self.config.source_dir
            stats = asyncio.run(self.ingestor.process_directory(
                Path(dir_to_use), recursive=True
            ))
            return stats
        except Exception as e:
            logger.error(f"[Iris] Ingest error: {e}")
            return {"ingested": 0, "duplicates_skipped": 0, "errors": 1, "total_size": 0}

    def search(self, query: str, limit: int = 10) -> str:
        try:
            results_caption = self.db.search_files_by_caption(query, limit)
            results_tags = self.db.search_files_by_tags(query, limit)
            # Deduplicate by file_path
            seen = set()
            combined = []
            unique_map = {}

            for r in results_caption + results_tags:
                fp = r.get("file_path")
                if not fp:
                    continue
                if fp not in unique_map:
                    unique_map[fp] = r
                else:
                    # merge metadata if needed (prefer captioned version)
                    if r.get("caption") and not unique_map[fp].get("caption"):
                        unique_map[fp] = r

            combined = list(unique_map.values())
            if not combined:
                return "No photos found matching that."
            lines = [f"Found {len(combined)} photos:"]
            for i, r in enumerate(combined, 1):
                path = r.get("file_path", "?")
                caption = r.get("caption") or "(no caption)"
                raw_tags = r.get("tags")
                try:
                    tags_list = json.loads(raw_tags) if raw_tags else []
                    tags = ", ".join(tags_list)
                except:
                    tags = raw_tags or ""
                lines.append(f"{i}. {path} — {caption} [{tags}]")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[Iris] Search error: {e}")
            return "No photos found matching that."

    def analyse(self, limit: int = 10) -> str:
        if not self.analyser:
            return "IrisAnalyser is not available."
        result = self.analyser.run_batch(limit)
        analysed = result.get("analysed", 0)
        errors = result.get("errors", 0)
        return f"Analysed {analysed} photos. {errors} errors."

    def stats(self) -> dict:
        try:
            return {
                "total_files": self.db.file_count(),
                "processed": self.db.processed_count(),
                "pending": self.db.pending_count(),
            }
        except Exception as e:
            logger.error(f"[Iris] Stats error: {e}")
            return {"total_files": 0, "processed": 0, "pending": 0}

    def status(self) -> str:
        try:
            s = self.stats()
            return f"Iris has {s['total_files']} photos indexed, {s['processed']} analysed, {s['pending']} pending."
        except Exception as e:
            logger.error(f"[Iris] Status error: {e}")
            return "Iris status unavailable."
