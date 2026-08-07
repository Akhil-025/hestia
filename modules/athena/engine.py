"""
modules/athena/engine.py

AthenaEngine — the single public entry point Hestia calls.
"""
import logging

from modules.base import BaseModule   
from modules.athena.local_rag import MergedLocalRAG
from modules.athena.hestia_llm_adapter import HestiaLLMAdapter
from modules.athena.services.query_service import QueryService

logger = logging.getLogger(__name__)


class AthenaEngine(BaseModule): 
    name = "athena"

    _INTENTS = {
        "athena_search",
        "query_documents",
        "search_documents",
        # HestiaOrchestrator._strip_module_prefix() strips "athena_" off
        # "athena_search" before dispatch, turning it into "search" — add
        # the stripped form too or can_handle() rejects it and the query
        # silently falls back to chat.
        "search",
        # Ingestion — previously Athena had no way to be triggered by voice
        # / chat / Hecate's text-trigger tier at all (unlike Iris, which
        # has a matching "iris_ingest"/"ingest" pair). Without this, users
        # had no way to get documents into the RAG index short of calling
        # the private _ingest() method directly from a Python shell, so
        # every search silently returned "No relevant information found".
        "athena_ingest",
        "ingest_documents",
        "ingest",
        # Status / stats — same rationale: expose stats() as a dispatchable
        # intent so "how many documents have I indexed" etc. can reach it,
        # matching Iris's "status" intent.
        "athena_status",
        "status",
    }

    def __init__(self, hestia_llm) -> None:
        self.rag           = MergedLocalRAG()
        self.llm           = HestiaLLMAdapter(hestia_llm)
        self.query_service = QueryService(self.rag, self.llm)

    def can_handle(self, intent: str) -> bool:              # ADD
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:   # ADD
        # Normalise: accept both the "athena_"-prefixed intent (as emitted
        # by the NLU / used by Hecate's Tier-1.5 direct routing) and the
        # stripped form HestiaOrchestrator._strip_module_prefix() actually
        # passes to handle() for every normal dispatch. Branching on the
        # prefixed form only would mean the ordinary dispatch path never
        # matches anything here, even though can_handle() already reported
        # True for that exact intent.
        canonical = intent[len("athena_"):] if intent.startswith("athena_") else intent

        if canonical in ("ingest", "ingest_documents"):
            return self._handle_ingest(entities, context)

        if canonical == "status":
            return self._handle_status()

        return self._handle_search(entities, context)

    def _handle_search(self, entities: dict, context: dict) -> dict:
        query = (
            entities.get("query")
            or entities.get("raw_query")
            or context.get("raw_query", "")
        )
        if not query:
            return {"response": "What would you like me to look up?", "data": {}, "confidence": 0.0}
        try:
            result = self.query_service.execute(query)
            return {
                "response":   result.answer,
                "data":       {"sources": [s.to_dict() for s in result.sources]},
                "confidence": 0.9,
            }
        except Exception:
            logger.exception("Athena query failed for query=%r", query[:80])
            return {"response": "I had trouble searching your documents.", "data": {}, "confidence": 0.0}

    def _handle_ingest(self, entities: dict, context: dict) -> dict:
        data_dir = entities.get("data_dir") or entities.get("path")
        try:
            stats = self._ingest(data_dir)
            files = stats.get("total_files", 0)
            chunks = stats.get("total_chunks", 0)
            if files == 0:
                response = (
                    "No new documents to ingest — add files to your Athena "
                    "documents folder and try again."
                )
            else:
                response = (
                    f"Ingestion complete. Processed {files} file(s) into "
                    f"{chunks} chunk(s)."
                )
            return {"response": response, "data": stats, "confidence": 1.0}
        except Exception:
            logger.exception("Athena ingestion failed for data_dir=%r", data_dir)
            return {
                "response": "I had trouble ingesting your documents.",
                "data": {},
                "confidence": 0.0,
            }

    def _handle_status(self) -> dict:
        try:
            s = self.stats()
            chunks = s.get("total_chunks", 0)
            subjects = s.get("subjects", [])
            if chunks == 0:
                response = "Your document index is empty — nothing has been ingested yet."
            else:
                response = (
                    f"Your document index has {chunks} chunk(s) across "
                    f"{len(subjects)} subject(s): {', '.join(subjects) or 'none'}."
                )
            return {"response": response, "data": s, "confidence": 0.9}
        except Exception:
            logger.exception("Athena status check failed")
            return {"response": "I couldn't check your document index.", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        try:
            s = self.stats()
            return {
                "athena_chunks":   s.get("total_chunks", 0),
                "athena_subjects": s.get("subjects", []),
                "athena_modules":  s.get("modules", []),
                "athena_ready":    s.get("total_chunks", 0) > 0,
            }
        except Exception:
            return {}

    def _query(self, q: str) -> str:
        """Run the full RAG pipeline and return the answer string."""
        result = self.query_service.execute(q)
        return result.answer

    def _ingest(self, data_dir: str | None = None) -> dict:
        """Ingest all documents under data_dir (or the configured default)."""
        return self.rag.ingest_directory(data_dir)

    def stats(self) -> dict:
        """Return ChromaDB collection stats."""
        return self.rag.get_collection_stats()