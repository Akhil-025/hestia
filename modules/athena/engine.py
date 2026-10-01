"""
modules/athena/engine.py

AthenaEngine — the single public entry point Hestia calls.
"""
import logging
import re

from modules.base import BaseModule   
from modules.athena.local_rag import MergedLocalRAG
from modules.athena.hestia_llm_adapter import HestiaLLMAdapter
from modules.athena.services.query_service import QueryService

logger = logging.getLogger(__name__)

_SCORE_TOGGLE_RE = re.compile(
    r"\b(?:turn|switch)\s+(on|off)\s+(?:the\s+)?(?:retrieval\s+|search\s+)?(?:scores?|score breakdown|debug)\b", re.I)
_SCORE_ONCE_RE = re.compile(
    r"^\s*debug(?:\s+search)?\b[\s:,-]*|\b(?:search\s+)?(?:show|with|include)\s+(?:me\s+)?(?:the\s+)?"
    r"(?:retrieval\s+|search\s+)?(?:scores?|score breakdown)\b(?:\s+(?:for|on|about))?[\s:,-]*", re.I)


def _format_score_breakdown(sources: list, metrics) -> str:
    """Spoken/typed form of the per-source semantic + BM25 scores (#64)."""
    def f(v):
        return "n/a" if v is None else f"{float(v):.2f}"
    lines = ["Retrieval scores:"]
    for i, s in enumerate(sources[:5], 1):
        page = f", page {s.get('page')}" if s.get("page") else ""
        lines.append(f"{i}. {s.get('file_name') or 'source'}{page}: semantic {f(s.get('semantic_score'))}, "
                     f"BM25 {f(s.get('bm25_score'))}, combined {f(s.get('score'))}")
    if isinstance(metrics, dict):
        extras = ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
        if extras:
            lines.append(f"({extras})")
    return "\n".join(lines)


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
        # backlog #53, #51, #52, #66 — export a report (pdf/latex/pptx) and
        # draft a research methodology.
        "athena_generate_report",
        "generate_report",
        "athena_methodology",
        "methodology",
        # backlog #60 — which indexed papers cite which, drawn as a graph.
        "athena_citation_graph",
        "citation_graph",
    }

    def __init__(self, hestia_llm) -> None:
        self.rag           = MergedLocalRAG()
        self.llm           = HestiaLLMAdapter(hestia_llm)
        self.query_service = QueryService(self.rag, self.llm)
        # backlog #54, #56, #65 — shares the same rag + llm adapter, so
        # nothing here needs its own connection/config.
        from modules.athena.services.synthesis_service import SynthesisService
        self.synthesis = SynthesisService(self.rag, self.llm)
        # The sources of the most recent search, so a follow-up like "the second
        # source wasn't relevant" (#63) or "give me a bibliography" (#55) has
        # something to refer to. Single user, so one list on the engine.
        self._last_sources: list = []
        self._citation_graphs = None        # built lazily (#60): reading PDFs is slow
        # Session toggle for the semantic/BM25 score breakdown (#64).
        self._debug_scores: bool = False

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

        if canonical == "generate_report":
            return self._handle_generate_report(entities, context)

        if canonical == "methodology":
            return self._handle_methodology(entities, context)

        if canonical == "citation_graph":
            return self._handle_citation_graph(entities, context)

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
        debug = bool(entities.get("debug") or entities.get("show_scores") or self._debug_scores)

        # #64: natural-language switches ("turn on retrieval scores") and a
        # one-off ("show scores for X") - an alias sets the intent only, so the
        # phrase has to be read out of the query here.
        raw_text = str(entities.get("raw_query") or context.get("raw_query") or query)
        toggle = _SCORE_TOGGLE_RE.search(raw_text)
        if toggle:
            self._debug_scores = toggle.group(1).lower() == "on"
            state = "on" if self._debug_scores else "off"
            return {"response": f"Retrieval score breakdown is {state}.",
                    "data": {"debug_scores": self._debug_scores}, "confidence": 0.95}
        if _SCORE_ONCE_RE.search(raw_text) or _SCORE_ONCE_RE.search(query):
            debug = True
        cleaned = _SCORE_ONCE_RE.sub(" ", query).strip(" ,:-")
        if cleaned != query:
            query = cleaned
            if not query:
                return {"response": "What should I look up?", "data": {}, "confidence": 0.0}

        # backlog #58/#59: "find the graph that shows X" / "which table lists Y"
        # is answered from figure/table chunks directly (no LLM synthesis).
        from modules.athena.pdf_structures import infer_content_type
        content_type = entities.get("content_type") or infer_content_type(query)
        if content_type in ("figure", "table"):
            structured = self._handle_structured_search(query, content_type, debug)
            if structured is not None:
                self._remember_sources((structured.get("data") or {}).get("sources"))
                return structured

        try:
            result = self.query_service.execute(query)
            data: dict = {
                "sources": [s.to_dict(include_score_breakdown=debug) for s in result.sources],
            }
            if debug and result.metrics is not None:
                data["metrics"] = result.metrics
            self._remember_sources(data["sources"])
            answer = result.answer
            if debug and data["sources"]:
                answer = f"{answer}\n\n{_format_score_breakdown(data['sources'], result.metrics)}"
            return {
                "response":   answer,
                "data":       data,
                "confidence": 0.9,
            }
        except Exception:
            logger.exception("Athena query failed for query=%r", query[:80])
            return {"response": "I had trouble searching your documents.", "data": {}, "confidence": 0.0}

    def _handle_structured_search(self, query: str, content_type: str, debug: bool) -> dict | None:
        """
        Figure/table lookup. Returns None to fall back to a normal search when
        the request can't be served this way (an error occurred).
        """
        noun = "figure" if content_type == "figure" else "table"
        try:
            if not self.rag.has_content_type(content_type):
                return {
                    "response": (
                        f"No {noun}s are indexed yet. Re-index your PDFs "
                        "(say \"ingest documents\") so I can find them."
                    ),
                    "data": {"sources": []}, "confidence": 0.4,
                }
            found = self.rag.search_by_content_type(query, content_type, n_results=3)
            if not found.results:
                return {"response": f"I couldn't find a {noun} matching that.",
                        "data": {"sources": []}, "confidence": 0.4}
            lines = []
            for r in found.results:
                m = r.metadata
                lines.append(f"{m.get('file_name', 'a document')}, page {m.get('page_number', '?')}: "
                             + r.document.split("\n", 1)[0])
            sources = [
                {"text": r.document, "file_name": r.metadata.get("file_name"),
                 "file_path": r.metadata.get("file_path"), "page": r.metadata.get("page_number"),
                 "subject": r.metadata.get("subject"), "module": r.metadata.get("module"),
                 "chunk_number": r.metadata.get("chunk_number"), "score": r.score,
                 **({"semantic_score": r.semantic_score} if debug else {})}
                for r in found.results
            ]
            return {"response": f"Best matching {noun}(s): " + " | ".join(lines),
                    "data": {"sources": sources, "content_type": content_type}, "confidence": 0.85}
        except Exception:
            logger.exception("Structured %s search failed; falling back to normal search.", noun)
            return None

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

    def _remember_sources(self, sources) -> None:
        if isinstance(sources, list):
            self._last_sources = [s for s in sources if isinstance(s, dict)]

    _FEEDBACK_REQUIRED = ("file_name", "subject", "module", "page_number", "chunk_number")
    _ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
                 "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "last": -1}

    def _resolve_feedback_source(self, entities: dict):
        """
        The chunk a feedback request means: a complete metadata dict if the caller
        supplied one (a web UI holding a source dict), else one of the last
        search's sources picked by 1-based `index`, an ordinal word in the
        request ("the second source"), or a file name. Returns (metadata, None),
        or (None, question) when it can't tell which source was meant.
        """
        e = dict(entities)
        if "page_number" not in e and e.get("page") is not None:
            e["page_number"] = e["page"]       # search results say "page", feedback said "page_number"
        if all(k in e for k in self._FEEDBACK_REQUIRED):
            return e, None
        sources = self._last_sources
        if not sources:
            return None, ("I need the source's file, subject, module, page, and chunk to record that.")
        raw = " ".join(str(entities.get(k) or "") for k in ("raw_query", "source", "which")).lower()
        idx = None
        v = entities.get("index")
        if isinstance(v, int) and not isinstance(v, bool):
            idx = v
        elif isinstance(v, str) and v.strip().lstrip("-").isdigit():
            idx = int(v)
        if idx is None:
            m = re.search(r"\b(?:source|result|number|no\.?|#)\s*(\d+)\b", raw) or re.search(r"\b(\d+)(?:st|nd|rd|th)\b", raw)
            if m:
                idx = int(m.group(1))
        if idx is None:
            m = re.search(r"\b(first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th)\b", raw)
            if m:
                idx = self._ORDINALS[m.group(1)]
        chosen = None
        if idx is not None:
            n = len(sources)
            if idx == -1:
                chosen = sources[-1]
            elif 1 <= idx <= n:
                chosen = sources[idx - 1]
            else:
                return None, f"I only showed {n} source(s). Which one do you mean?"
        else:
            name = str(entities.get("file_name") or "").lower()
            if name:
                hits = [s for s in sources if name in str(s.get("file_name") or "").lower()]
                if len({h.get("file_name") for h in hits}) == 1:
                    chosen = hits[0]
            elif len(sources) == 1:
                chosen = sources[0]
        if chosen is None:
            return None, "Which source - the first, second, third? Or give me its file name."
        meta = {"file_name": chosen.get("file_name"), "subject": chosen.get("subject"),
                "module": chosen.get("module"), "page_number": chosen.get("page"),
                "chunk_number": chosen.get("chunk_number")}
        return {k: v for k, v in meta.items() if v is not None}, None

    @staticmethod
    def _feedback_polarity(entities: dict) -> bool:
        if "relevant" in entities:
            v = entities["relevant"]
            return v.strip().lower() in ("true", "yes", "1", "relevant") if isinstance(v, str) else bool(v)
        raw = str(entities.get("raw_query") or "").lower()
        if re.search(r"\b(?:not|n't|isn'?t|wasn'?t|irrelevant|useless|unhelpful|wrong|bad)\b", raw):
            return False
        return bool(re.search(r"\b(?:relevant|useful|helpful|good|right)\b", raw))

    def _handle_mark_feedback(self, entities: dict) -> dict:
        """
        Mark a previously-returned source as relevant or not (backlog #63). Works
        from a complete metadata dict, or - the path a person actually has -
        from the sources of the last search: "the second source wasn't relevant".
        """
        metadata, question = self._resolve_feedback_source(entities)
        if metadata is None:
            return {"response": question, "data": {}, "confidence": 0.0}
        relevant = self._feedback_polarity(entities)
        try:
            self.rag.mark_feedback(metadata, relevant)
        except Exception:
            logger.exception("mark_feedback failed for entities=%r", entities)
            return {"response": "I couldn't record that feedback.", "data": {}, "confidence": 0.0}

        verb = "relevant" if relevant else "not relevant"
        name = metadata.get("file_name")
        what = f"the source from {name}" if name else "that source"
        return {
            "response": f"Noted — I'll treat {what} as {verb} going forward.",
            "data": {"file_name": name, "relevant": relevant}, "confidence": 0.9,
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

        sources = entities.get("sources") or self._last_sources
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

    # ------------------------------------------------------------------
    # backlog #53, #51, #52, #66 — generated files
    # ------------------------------------------------------------------

    def _export_dir(self) -> str:
        from pathlib import Path
        from modules.athena.config import get_config
        return str(Path(get_config().data_dir).parent / "exports")

    def _handle_generate_report(self, entities: dict, context: dict) -> dict:
        from modules.athena import generation as gen

        raw = (entities.get("raw_query") or context.get("raw_query") or "")
        subject = (entities.get("subject") or entities.get("topic") or "").strip()
        fmt_entity = entities.get("format")
        if not subject or not fmt_entity:
            parsed_subject, parsed_fmt = gen.parse_report_request(raw)
            subject = subject or parsed_subject
            fmt_entity = fmt_entity or parsed_fmt
        if not subject:
            return {"response": "Which subject should the report cover?", "data": {}, "confidence": 0.0}
        try:
            fmt = gen.normalise_format(fmt_entity)
            model = gen.build_report_from_subject(self.synthesis, subject, entities.get("title"))
            outline_fn = (lambda m: gen.outline_with_llm(self.llm, m)) if fmt == "pptx" else None
            files = gen.render(model, fmt, self._export_dir(), subject, outline_fn)
        except gen.GenerationError as exc:
            return {"response": str(exc), "data": {}, "confidence": 0.3}
        except Exception:
            logger.exception("generate_report failed for subject=%r", subject)
            return {"response": "I had trouble creating that report.", "data": {}, "confidence": 0.0}
        return {
            "response": f"Your {fmt.upper() if fmt != 'latex' else 'LaTeX'} report on {subject} is ready: "
                        + ", ".join(files),
            "data": {"format": fmt, "files": files, "subject": subject,
                     "references": len(model.references)},
            "confidence": 0.9,
        }

    def _handle_methodology(self, entities: dict, context: dict) -> dict:
        from modules.athena import generation as gen

        raw = (entities.get("raw_query") or context.get("raw_query") or "")
        question = (entities.get("question") or entities.get("topic") or entities.get("query") or "").strip()
        fmt_entity = entities.get("format")
        if not question:
            question = gen.parse_methodology_request(raw)
        if not fmt_entity:
            fmt_entity = gen.parse_report_request(raw)[1]
        try:
            fmt = gen.normalise_format(fmt_entity)
            method = gen.generate_methodology(self.llm, question)
            model = gen.methodology_to_report(method)
            outline_fn = (lambda m: gen.deterministic_outline(m)) if fmt == "pptx" else None
            files = gen.render(model, fmt, self._export_dir(), "methodology-" + question, outline_fn)
        except gen.GenerationError as exc:
            return {"response": str(exc), "data": {}, "confidence": 0.3}
        except Exception:
            logger.exception("methodology failed for question=%r", question[:80])
            return {"response": "I had trouble drafting that methodology.", "data": {}, "confidence": 0.0}
        return {
            "response": f"Methodology drafted: {method['research_question']} "
                        f"(design: {method['design']}). File: " + ", ".join(files),
            "data": {"methodology": method, "format": fmt, "files": files},
            "confidence": 0.85,
        }

    # ------------------------------------------------------------------
    # backlog #60 — citation graph
    # ------------------------------------------------------------------

    def _citation_service(self):
        if self._citation_graphs is None:
            from pathlib import Path
            from modules.athena.config import get_config
            from modules.athena.services.citation_graph_service import CitationGraphService
            cache = str(Path(get_config().cache_dir) / "citation_parse.json")
            self._citation_graphs = CitationGraphService(self.rag, cache)
        return self._citation_graphs

    def citation_graph(self, subject: str | None = None) -> dict:
        """The citation graph as a dict (nodes, links, stats) - what the web API serves."""
        return self._citation_service().build(subject or None)

    def citation_graph_html(self, subject: str | None = None, focus: str | None = None) -> str:
        from modules.athena.citation_graph_view import render_html
        from modules.athena.services.citation_graph_service import find_node
        graph = self.citation_graph(subject)
        node, _ = find_node(graph, focus or "")
        return render_html(graph, node["id"] if node else None)

    def _handle_citation_graph(self, entities: dict, context: dict) -> dict:
        from datetime import date
        from modules.athena.citation_graph_view import write_graph_files
        from modules.athena.services import citation_graph_service as cg

        raw = str(entities.get("raw_query") or context.get("raw_query") or "")
        subject = str(entities.get("subject") or entities.get("topic") or "").strip()
        focus = str(entities.get("focus") or entities.get("paper") or entities.get("file_name") or "").strip()
        if not subject and not focus and raw:
            subject, focus = cg.parse_graph_request(raw)
        formats = ["html", "json"] + (["dot"] if re.search(r"\b(?:dot|graphviz)\b", raw, re.I)
                                      or str(entities.get("format") or "").lower() in ("dot", "graphviz") else [])
        try:
            if subject:
                known = {f["subject"].lower(): f["subject"] for f in self.rag.list_files()}
                if subject.lower() not in known:
                    have = ", ".join(sorted(known.values())) or "none yet"
                    return {"response": f"I don't have a subject called {subject}. Your subjects: {have}.",
                            "data": {}, "confidence": 0.3}
                subject = known[subject.lower()]
            graph = self.citation_graph(subject)
            node, candidates = cg.find_node(graph, focus) if focus else (None, [])
            if focus and node is None:
                names = "; ".join(c["label"] for c in candidates[:4])
                return {"response": ("Which paper do you mean: " + names + "?") if candidates
                        else f"I couldn't find a paper matching {focus}.", "data": {}, "confidence": 0.3}
            files = write_graph_files(graph, self._export_dir(), "citation-graph-" + (subject or "all"),
                                      date.today().isoformat(), tuple(formats), node["id"] if node else None)
        except Exception:
            logger.exception("citation_graph failed for subject=%r", subject)
            return {"response": "I had trouble building the citation graph.", "data": {}, "confidence": 0.0}
        text = cg.describe(graph, node)
        if graph["stats"]["documents"]:
            text += " Saved: " + ", ".join(files)
        return {"response": text, "data": {"files": files, "stats": graph["stats"], "subject": subject,
                                           "focus": node["file_name"] if node else None},
                "confidence": 0.9 if graph["stats"]["documents"] else 0.5}

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

    def ingest_status(self) -> dict:
        """Live progress of the current (or most recent) ingestion (backlog #70)."""
        return self.rag.progress.snapshot()

    def start_ingest_background(self, data_dir: str | None = None) -> bool:
        """
        Run an ingestion on a worker thread so the web UI isn't blocked for
        minutes. Returns False, starting nothing, if one is already running.
        Progress is read via ingest_status().
        """
        import threading

        # Atomic claim: checking progress.running alone would let two quick
        # calls both pass before either thread has marked itself running.
        flag = self.__dict__.setdefault("_bg_ingest_flag", threading.Lock())
        if not flag.acquire(blocking=False):
            return False
        if self.rag.progress.snapshot()["running"] or self.rag._ingest_lock.locked():
            flag.release()
            return False

        # Mark the run as started NOW, before the thread is scheduled, so a
        # status poll right after this call never sees "idle".
        self.rag.progress.start(0)

        def _run() -> None:
            try:
                self._ingest(data_dir)
            except Exception as exc:
                logger.exception("Background Athena ingestion failed for data_dir=%r", data_dir)
                self.rag.progress.finish(error=type(exc).__name__)
            finally:
                flag.release()

        threading.Thread(target=_run, daemon=True, name="AthenaIngest").start()
        return True

    def stats(self) -> dict:
        """Return ChromaDB collection stats."""
        return self.rag.get_collection_stats()