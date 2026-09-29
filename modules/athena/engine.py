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
        # backlog #57 — dry preview of what a re-ingest would find, with
        # nothing actually written to the index.
        "athena_check_updates",
        "check_updates",
        # backlog #63 — mark a previously-returned source as (ir)relevant.
        "athena_mark_feedback",
        "mark_feedback",
        # backlog #54, #56, #65, #55 — synthesis over already-ingested
        # content, and a bibliography of what a search/synthesis drew on.
        "athena_literature_review",
        "literature_review",
        "athena_research_gaps",
        "research_gaps",
        "athena_compare_documents",
        "compare_documents",
        "athena_get_citations",
        "get_citations",
        # backlog #69 — translate an already-ingested document.
        "athena_translate_document",
        "translate_document",
    }

    def __init__(self, hestia_llm) -> None:
        self.rag           = MergedLocalRAG()
        self.llm           = HestiaLLMAdapter(hestia_llm)
        self.query_service = QueryService(self.rag, self.llm)
        # backlog #54, #56, #65 — shares the same rag + llm adapter, so
        # nothing here needs its own connection/config.
        from modules.athena.services.synthesis_service import SynthesisService
        self.synthesis = SynthesisService(self.rag, self.llm)

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

        if canonical == "check_updates":
            return self._handle_check_updates(entities)

        if canonical == "mark_feedback":
            return self._handle_mark_feedback(entities)

        if canonical == "literature_review":
            return self._handle_literature_review(entities)

        if canonical == "research_gaps":
            return self._handle_research_gaps(entities)

        if canonical == "compare_documents":
            return self._handle_compare_documents(entities, context)

        if canonical == "get_citations":
            return self._handle_get_citations(entities)

        if canonical == "translate_document":
            return self._handle_translate_document(entities)

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

        # backlog #64: "debug" (or "show_scores") in entities opts into
        # the semantic/BM25 score breakdown per source and the retrieval
        # metrics — off by default because it roughly doubles the field
        # count of what's usually a short list of sources, and most
        # callers just want the answer.
        debug = bool(entities.get("debug") or entities.get("show_scores"))

        try:
            result = self.query_service.execute(query)
            data: dict = {
                "sources": [s.to_dict(include_score_breakdown=debug) for s in result.sources],
            }
            if debug and result.metrics is not None:
                data["metrics"] = result.metrics
            return {
                "response":   result.answer,
                "data":       data,
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
            new = stats.get("new_files", 0)
            updated = stats.get("updated_files", 0)
            unchanged = stats.get("unchanged_files", 0)
            if files == 0:
                response = (
                    "No new documents to ingest — add files to your Athena "
                    "documents folder and try again."
                )
            elif new == 0 and updated == 0:
                # Every file was already up to date (backlog #61) — say so
                # plainly rather than "processed N files" implying work
                # was done when nothing actually changed.
                response = f"Everything's already up to date ({unchanged} file(s) unchanged)."
            else:
                parts = []
                if new:
                    parts.append(f"{new} new")
                if updated:
                    parts.append(f"{updated} updated")
                if unchanged:
                    parts.append(f"{unchanged} unchanged")
                response = (
                    f"Ingestion complete: {', '.join(parts)} file(s), "
                    f"{chunks} chunk(s) added."
                )
            return {"response": response, "data": stats, "confidence": 1.0}
        except Exception:
            logger.exception("Athena ingestion failed for data_dir=%r", data_dir)
            return {
                "response": "I had trouble ingesting your documents.",
                "data": {},
                "confidence": 0.0,
            }

    def _handle_mark_feedback(self, entities: dict) -> dict:
        """
        Mark a specific, previously-returned source as relevant or not
        (backlog #63). *entities* carries the same fields a
        SourceDocument.to_dict() already returns (file_name, subject,
        module, page_number, chunk_number) — the natural shape for a
        caller (a future "thumbs down on this source" UI action, or a
        voice follow-up referencing the last search's sources) that
        already has a source dict in hand from a prior search response,
        rather than needing to know Athena's internal chunk-id format.
        """
        required = ("file_name", "subject", "module", "page_number", "chunk_number")
        if not all(k in entities for k in required):
            return {
                "response": "I need the source's file, subject, module, page, and chunk to record that.",
                "data": {}, "confidence": 0.0,
            }
        relevant = bool(entities.get("relevant", False))
        try:
            self.rag.mark_feedback(entities, relevant)
        except Exception:
            logger.exception("mark_feedback failed for entities=%r", entities)
            return {"response": "I couldn't record that feedback.", "data": {}, "confidence": 0.0}

        verb = "relevant" if relevant else "not relevant"
        return {
            "response": f"Noted — I'll treat that source as {verb} going forward.",
            "data": {}, "confidence": 0.9,
        }

    def _handle_literature_review(self, entities: dict) -> dict:
        subject = (entities.get("subject") or entities.get("topic") or "").strip()
        if not subject:
            return {"response": "Which subject should I review?", "data": {}, "confidence": 0.0}
        try:
            result = self.synthesis.generate_literature_review(subject)
        except Exception:
            logger.exception("literature_review failed for subject=%r", subject)
            return {"response": "I had trouble generating that review.", "data": {}, "confidence": 0.0}
        return {"response": result["review"], "data": result, "confidence": 0.85}

    def _handle_research_gaps(self, entities: dict) -> dict:
        subject = (entities.get("subject") or entities.get("topic") or "").strip()
        if not subject:
            return {"response": "Which subject should I look for gaps in?", "data": {}, "confidence": 0.0}
        try:
            result = self.synthesis.detect_research_gaps(subject)
        except Exception:
            logger.exception("research_gaps failed for subject=%r", subject)
            return {"response": "I had trouble analyzing that.", "data": {}, "confidence": 0.0}
        return {"response": result["gaps"], "data": result, "confidence": 0.85}

    def _handle_compare_documents(self, entities: dict, context: dict) -> dict:
        question = entities.get("query") or entities.get("question") or context.get("raw_query", "")
        file_names = entities.get("file_names") or entities.get("files") or []
        if isinstance(file_names, str):
            file_names = [f.strip() for f in file_names.split(",") if f.strip()]
        if len(file_names) < 2:
            return {
                "response": "Tell me which two or more documents to compare (by file name).",
                "data": {}, "confidence": 0.0,
            }
        try:
            result = self.synthesis.compare_documents(
                question or "How do these documents compare?",
                file_names, subject=entities.get("subject"),
            )
        except Exception:
            logger.exception("compare_documents failed for files=%r", file_names)
            return {"response": "I had trouble comparing those documents.", "data": {}, "confidence": 0.0}
        return {"response": result["answer"], "data": result, "confidence": 0.85}

    def _handle_get_citations(self, entities: dict) -> dict:
        """
        A bibliography for the sources of the MOST RECENT search this
        engine ran, or an explicit list passed in `entities["sources"]`
        (the shape a caller already holding a prior search response's
        `data["sources"]` can pass directly). File-based citations only —
        see modules/athena/services/citation_service.py's module
        docstring for why real academic metadata isn't available.
        """
        from modules.athena.services.citation_service import CitationRegistry

        sources = entities.get("sources")
        if not sources:
            return {
                "response": "I don't have a recent search to build a bibliography from.",
                "data": {}, "confidence": 0.0,
            }
        registry = CitationRegistry()
        registry.add_sources(sources)
        fmt = (entities.get("format") or "apa").lower()
        text = registry.to_bibtex() if fmt == "bibtex" else registry.to_apa_style()
        if not text:
            return {"response": "No citable sources found.", "data": {}, "confidence": 0.5}
        return {
            "response": text,
            "data": {"format": fmt, "count": len(registry)},
            "confidence": 0.9,
        }

    def _handle_translate_document(self, entities: dict) -> dict:
        file_name = (entities.get("file_name") or entities.get("file") or "").strip()
        target_language = (entities.get("target_language") or entities.get("language") or "").strip()
        if not file_name or not target_language:
            return {
                "response": "Which document, and which language should I translate it into?",
                "data": {}, "confidence": 0.0,
            }
        try:
            result = self.synthesis.translate_document(
                file_name, target_language, subject=entities.get("subject")
            )
        except Exception:
            logger.exception(
                "translate_document failed for file=%r language=%r", file_name, target_language
            )
            return {"response": "I had trouble translating that document.", "data": {}, "confidence": 0.0}
        return {"response": result["translation"], "data": result, "confidence": 0.85}

    def _handle_check_updates(self, entities: dict) -> dict:
        """
        "What's new since I last checked" (backlog #57) — a dry preview,
        no ingestion performed.
        """
        data_dir = entities.get("data_dir") or entities.get("path")
        try:
            changes = self.rag.get_changes_since_last_check(data_dir)
        except Exception:
            logger.exception("Athena check-updates failed for data_dir=%r", data_dir)
            return {"response": "I couldn't check for updates.", "data": {}, "confidence": 0.0}

        new_files = changes.get("new_files", [])
        updated_files = changes.get("updated_files", [])
        if not new_files and not updated_files:
            response = f"No changes — all {changes.get('unchanged_count', 0)} file(s) are up to date."
        else:
            parts = []
            if new_files:
                parts.append(f"{len(new_files)} new file(s): {', '.join(new_files[:5])}")
            if updated_files:
                parts.append(f"{len(updated_files)} updated file(s): {', '.join(updated_files[:5])}")
            response = "; ".join(parts) + ". Say \"ingest documents\" to index them."
        return {"response": response, "data": changes, "confidence": 0.9}

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