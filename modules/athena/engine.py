"""
modules/athena/engine.py

AthenaEngine — the single public entry point Hestia calls.
"""
from modules.athena.local_rag import MergedLocalRAG
from modules.athena.hestia_llm_adapter import HestiaLLMAdapter
from modules.athena.services.query_service import QueryService


class AthenaEngine:
    """
    Usage inside Hestia:

        from modules.athena.engine import AthenaEngine
        athena = AthenaEngine(hestia_llm=self.nlu.llm)
        answer = athena.query("explain entropy from my notes")
    """

    def __init__(self, hestia_llm) -> None:
        self.rag           = MergedLocalRAG()
        self.llm           = HestiaLLMAdapter(hestia_llm)
        self.query_service = QueryService(self.rag, self.llm)

    def query(self, q: str) -> str:
        """Run the full RAG pipeline and return the answer string."""
        result = self.query_service.execute(q)
        return result.answer

    def ingest(self, data_dir: str | None = None) -> dict:
        """Ingest all documents under data_dir (or the configured default)."""
        return self.rag.ingest_directory(data_dir)

    def stats(self) -> dict:
        """Return ChromaDB collection stats."""
        return self.rag.get_collection_stats()
