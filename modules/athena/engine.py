"""
modules/athena/engine.py

AthenaEngine — the single public entry point Hestia calls.
"""
from modules.base import BaseModule   
from modules.athena.local_rag import MergedLocalRAG
from modules.athena.hestia_llm_adapter import HestiaLLMAdapter
from modules.athena.services.query_service import QueryService


class AthenaEngine(BaseModule): 
    name = "athena"

    _INTENTS = {
        "athena_search",
        "query_documents",
        "search_documents",
    }

    def __init__(self, hestia_llm) -> None:
        self.rag           = MergedLocalRAG()
        self.llm           = HestiaLLMAdapter(hestia_llm)
        self.query_service = QueryService(self.rag, self.llm)

    def can_handle(self, intent: str) -> bool:              # ADD
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:   # ADD
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
        except Exception as e:
            return {"response": "I had trouble searching your documents.", "data": {}, "confidence": 0.0}

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
