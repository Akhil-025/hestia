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
from .embeddings import ClipEmbedder, ImageVectorIndex
try:
    from .analyser import IrisAnalyser
except ImportError:
    IrisAnalyser = None
    logging.warning("[Iris] requests library not available, IrisAnalyser disabled.")

logger = logging.getLogger(__name__)

class IrisEngine(BaseModule):               
    name = "iris"

    def __init__(self, hestia_llm=None, embedder=None, vector_index=None):
        try:
            self.config: IrisConfig = get_config()
            self.db = IrisDB(self.config.db_path)
            self.ingestor = FileIngestor(self.config, self.db)
            self.hestia_llm = hestia_llm

            # CLIP semantic search (README roadmap item — see embeddings.py).
            # Both are injectable for tests; default to real implementations
            # that degrade to unavailable on their own if torch/chromadb
            # aren't installed, so Iris as a whole never fails to construct
            # because of this.
            self.embedder = embedder if embedder is not None else ClipEmbedder()
            self.vector_index = (
                vector_index if vector_index is not None
                else ImageVectorIndex(self.config.chroma_dir)
            )

            ollama_host = "127.0.0.1"
            ollama_port = 11434
            self.analyser = (
                IrisAnalyser(
                    self.db, ollama_host, ollama_port, "llava:7b",
                    embedder=self.embedder, vector_index=self.vector_index,
                )
                if IrisAnalyser else None
            )
            logger.info("[Iris] Engine ready")
        except Exception as e:
            logger.error(f"[Iris] Engine init failed: {e}")
            raise


    def can_handle(self, intent: str) -> bool:   
        # Accepts both the full prefixed intent name (as emitted by the NLU
        # / used by Hecate's Tier-1.5 direct routing) and the stripped form
        # produced by HestiaOrchestrator._strip_module_prefix() before
        # dispatch — without this, "iris_search" gets stripped to "search"
        # and can_handle() would reject it, silently falling back to chat.
        return intent in {
            "iris_search", "iris_ingest", "iris_analyse", "iris_status", "iris_query",
            "search", "ingest", "analyse", "status", "query",
        }

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        raw = entities.get("raw_query", context.get("raw_query", "")).lower()

        # Normalise: accept both the "iris_"-prefixed intent (as emitted by
        # the NLU / used by Hecate's Tier-1.5 direct routing) and the
        # stripped form HestiaOrchestrator._strip_module_prefix() actually
        # passes to handle() for every non-trigger-tier dispatch ("iris_"
        # is in the orchestrator's known prefix list). Branching on the
        # prefixed form only, as this used to, meant the normal dispatch
        # path — intent="search"/"ingest"/"analyse"/"status" — never
        # matched anything here and silently fell through to the blank,
        # 0.0-confidence response below, even though can_handle() had
        # already reported True for that exact intent.
        canonical = intent[len("iris_"):] if intent.startswith("iris_") else intent

        if canonical == "ingest":
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

        elif canonical == "analyse":
            response = self.analyse(limit=20)
            return {"response": response, "data": {}, "confidence": 0.9}

        elif canonical == "status":
            # Previously had no branch at all — can_handle() accepted
            # "status"/"iris_status" but handle() silently dropped it into
            # the blank fallback below.
            return {"response": self.status(), "data": {}, "confidence": 0.9}

        elif canonical in {"search", "query"}:
            result = self.search(raw or entities.get("query", ""))
            return {
                "response": result or "No matching media found.",
                "data": {},
                "confidence": 0.85 if result else 0.3,
            }

        return {"response": "I'm not sure how to handle that media request.", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        try:
            s = self.stats()
            recent = self.db.get_all_files(limit=3)
            recent_captions = [
                f.get("caption", "") for f in recent if f.get("caption")
            ]
            return {
                "iris_total":         s.get("total_files", 0),
                "iris_processed":     s.get("processed", 0),
                "iris_pending":       s.get("pending", 0),
                "iris_recent_captions": recent_captions,
            }
        except Exception:
            return {}

    def ingest(self, source_dir: str = None) -> dict:
        try:
            dir_to_use = source_dir or self.config.source_dir
            coro = self.ingestor.process_directory(
                Path(dir_to_use), recursive=True
            )

            try:
                loop = asyncio.get_running_loop()

                # ⚠️ If we're already inside the event loop thread
                if loop.is_running():
                    # Offload to thread to avoid blocking loop
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(lambda: asyncio.run(coro))
                        return future.result()

            except RuntimeError:
                # No running loop
                return asyncio.run(coro)

        except Exception as e:
            logger.error(f"[Iris] Ingest error: {e}")
            return {
                "ingested": 0,
                "duplicates_skipped": 0,
                "errors": 1,
                "total_size": 0,
            }

    def _semantic_matches(self, query: str, limit: int) -> list[dict]:
        """
        CLIP-embedding-based matches, ranked nearest-first. Returns [] (not
        an error) whenever the embedder/vector_index aren't available or the
        query can't be embedded — this is a ranking *enhancement* over
        caption/tag search, never a hard dependency for search() to work.
        """
        try:
            query_vector = self.embedder.embed_text(query)
            if query_vector is None:
                return []
            hits = self.vector_index.query(query_vector, top_k=limit)
            matches = []
            for file_id, _distance in hits:
                record = self.db.get_file(file_id)
                if record:
                    matches.append(record)
            return matches
        except Exception as e:
            logger.warning(f"[Iris] Semantic search failed, falling back to caption/tag: {e}")
            return []

    def search(self, query: str, limit: int = 10) -> "str | None":
        try:
            results_semantic = self._semantic_matches(query, limit)
            results_caption = self.db.search_files_by_caption(query, limit)
            results_tags = self.db.search_files_by_tags(query, limit)
            # Deduplicate by file_path. Semantic hits are listed first so
            # `dict`-insertion order (preserved by unique_map.values() below)
            # keeps them ranked ahead of plain substring caption/tag matches,
            # which have no real relevance ordering of their own.
            combined = []
            unique_map = {}

            for r in results_semantic + results_caption + results_tags:
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
                # Falsy on purpose: handle()'s `result or "No matching
                # media found."` / `0.85 if result else 0.3` logic relies
                # on search() returning something falsy when nothing
                # matched. Returning a non-empty "not found" sentinel
                # string here (as this used to) made that check always
                # truthy, so every zero-result search was reported back
                # to the orchestrator at 0.85 confidence — indistinguishable
                # from a real match.
                return None
            lines = [f"Found {len(combined)} photos:"]
            for i, r in enumerate(combined, 1):
                path = r.get("file_path", "?")
                caption = r.get("caption") or "(no caption)"
                raw_tags = r.get("tags")
                try:
                    tags_list = json.loads(raw_tags) if raw_tags else []
                    tags = ", ".join(tags_list)
                # Narrowed from a bare `except:` — raw_tags is a DB column
                # that's expected to hold either JSON or nothing; the only
                # realistic failures are malformed JSON or an unexpected
                # non-string type. A bare except also silently swallows
                # KeyboardInterrupt/SystemExit, which has no business being
                # caught while formatting a search result string.
                except (json.JSONDecodeError, TypeError):
                    tags = raw_tags or ""
                lines.append(f"{i}. {path} — {caption} [{tags}]")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[Iris] Search error: {e}")
            return None

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