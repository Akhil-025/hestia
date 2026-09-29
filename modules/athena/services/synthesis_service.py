"""
modules/athena/services/synthesis_service.py

Multi-document synthesis over already-ingested content (backlog #54, #56,
#65): literature review generation, research-gap detection, and
comparative queries across documents.

All three share the same shape — gather representative content from one
or more already-ingested documents (via `MergedLocalRAG.get_chunks_for_file`
/ a targeted `search`, never a fresh document fetch — this operates
entirely on what's already in the index), build one prompt, ask the LLM
to synthesize, return prose. Kept as one small service rather than three
separate ones since the gather-then-synthesize shape and error handling
are identical; only the prompt differs.

None of these methods are wired to a user-facing intent yet in THIS
change — see `modules/athena/engine.py` for the handlers that call them
and `CONTRIBUTING.md` for the intent-registration checklist if extending
this further.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# How much raw chunk text to feed the LLM per document — generous enough
# for real synthesis, capped so three or four documents together still
# fit comfortably in a local model's context window.
_MAX_CHARS_PER_DOCUMENT = 4000
_CHUNKS_PER_DOCUMENT = 15


class SynthesisService:
    def __init__(self, rag, ai) -> None:
        """
        *rag*: a MergedLocalRAG (or anything with list_files/get_chunks_for_file/search).
        *ai*: a HestiaLLMAdapter (or anything with .generate(prompt) -> {"text", "error"}).
        """
        self.rag = rag
        self.ai = ai

    # ------------------------------------------------------------------
    # Shared gathering + LLM-call plumbing
    # ------------------------------------------------------------------

    def _gather_subject_material(self, subject: str) -> list[tuple[str, str]]:
        """[(file_name, joined_text)] for every file ingested under *subject*."""
        files = self.rag.list_files(subject=subject)
        material: list[tuple[str, str]] = []
        for f in files:
            chunks = self.rag.get_chunks_for_file(
                f["file_name"], subject=subject, limit=_CHUNKS_PER_DOCUMENT
            )
            text = "\n".join(chunks)[:_MAX_CHARS_PER_DOCUMENT]
            if text.strip():
                material.append((f["file_name"], text))
        return material

    def _call_llm(self, prompt: str, empty_message: str) -> str:
        """Shared error handling for every synthesis call in this service."""
        try:
            result = self.ai.generate(prompt)
        except Exception:
            logger.exception("SynthesisService: LLM call failed.")
            return ""
        if not isinstance(result, dict):
            return str(result).strip()
        if result.get("error"):
            logger.warning("SynthesisService: LLM returned an error: %s", result["error"])
            return ""
        text = (result.get("text") or "").strip()
        return text or ""

    # ------------------------------------------------------------------
    # #54 — literature review generator
    # ------------------------------------------------------------------

    def generate_literature_review(self, subject: str) -> dict[str, Any]:
        """
        Synthesize a literature-review-style overview across every
        document ingested under *subject* — recurring themes, points of
        agreement/disagreement between sources, and an overall summary.
        Returns {"review": str, "sources": [file_name, ...]} — empty
        review with an explanatory message if there's nothing ingested
        for the subject, never raises.
        """
        material = self._gather_subject_material(subject)
        if not material:
            return {
                "review": f"I don't have any documents ingested under \"{subject}\" to review.",
                "sources": [],
            }

        sections = "\n\n".join(
            f"--- {name} ---\n{text}" for name, text in material
        )
        prompt = (
            f"You are writing a literature review covering the following "
            f"{len(material)} source document(s) on \"{subject}\". "
            f"Identify recurring themes, note where sources agree or "
            f"disagree, and summarize the overall state of the material. "
            f"Cite sources by their file name in parentheses where relevant. "
            f"Write 3-5 paragraphs.\n\n{sections}"
        )
        review = self._call_llm(
            prompt, f"I couldn't generate a review for \"{subject}\"."
        )
        if not review:
            review = f"I had trouble generating a review for \"{subject}\"."
        return {"review": review, "sources": [name for name, _ in material]}

    # ------------------------------------------------------------------
    # #56 — research-gap detection
    # ------------------------------------------------------------------

    def detect_research_gaps(self, subject: str) -> dict[str, Any]:
        """
        Look for recurring "limitations", "future work", or open-question
        language across a subject's ingested documents and synthesize
        what gaps the sources themselves point to. This is explicitly
        about what the SOURCES SAY is unresolved, not an independent
        judgement Hestia forms about the field.
        """
        material = self._gather_subject_material(subject)
        if not material:
            return {
                "gaps": f"I don't have any documents ingested under \"{subject}\" to analyze.",
                "sources": [],
            }

        sections = "\n\n".join(
            f"--- {name} ---\n{text}" for name, text in material
        )
        prompt = (
            f"Read the following {len(material)} source document(s) on "
            f"\"{subject}\". Identify limitations, open questions, or "
            f"suggested future work that the SOURCES THEMSELVES mention — "
            f"do not invent gaps the sources don't point to. If multiple "
            f"sources mention a similar gap, note that it's recurring. "
            f"If no such gaps are discussed, say so plainly rather than "
            f"inventing some. Cite sources by file name.\n\n{sections}"
        )
        gaps = self._call_llm(
            prompt, f"I couldn't analyze gaps for \"{subject}\"."
        )
        if not gaps:
            gaps = f"I had trouble analyzing research gaps for \"{subject}\"."
        return {"gaps": gaps, "sources": [name for name, _ in material]}

    # ------------------------------------------------------------------
    # #65 — multi-document comparative queries
    # ------------------------------------------------------------------

    def compare_documents(
        self, question: str, file_names: list[str], subject: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Answer *question* by comparing content ACROSS the named files
        specifically — not a general subject-wide synthesis, a targeted
        comparison of exactly the documents named. Returns which of the
        requested files actually had content to compare, since a name
        that doesn't match anything ingested is a likely typo worth
        surfacing rather than silently ignoring.
        """
        material: list[tuple[str, str]] = []
        missing: list[str] = []
        for name in file_names:
            chunks = self.rag.get_chunks_for_file(
                name, subject=subject, limit=_CHUNKS_PER_DOCUMENT
            )
            text = "\n".join(chunks)[:_MAX_CHARS_PER_DOCUMENT]
            if text.strip():
                material.append((name, text))
            else:
                missing.append(name)

        if len(material) < 2:
            found = ", ".join(name for name, _ in material) or "none"
            return {
                "answer": (
                    f"I need at least two documents with content to compare. "
                    f"Found: {found}."
                    + (f" Not found: {', '.join(missing)}." if missing else "")
                ),
                "compared": [name for name, _ in material],
                "missing": missing,
            }

        sections = "\n\n".join(f"--- {name} ---\n{text}" for name, text in material)
        prompt = (
            f"Compare the following {len(material)} documents to answer this "
            f"question: \"{question}\"\n\n"
            f"Explicitly note where the documents agree, disagree, or one "
            f"covers something the other doesn't. Reference each document "
            f"by its file name.\n\n{sections}"
        )
        answer = self._call_llm(prompt, "I couldn't compare those documents.")
        if not answer:
            answer = "I had trouble comparing those documents."
        return {
            "answer": answer,
            "compared": [name for name, _ in material],
            "missing": missing,
        }

    # ------------------------------------------------------------------
    # #69 — "translate this document" pipeline
    # ------------------------------------------------------------------

    def translate_document(
        self, file_name: str, target_language: str, subject: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Translate an already-ingested document's content into
        *target_language* via the same local LLM used for everything
        else in this service — no separate translation API/library, since
        the model itself handles this reasonably for common languages.

        Chunked internally at roughly `_MAX_CHARS_PER_DOCUMENT`-sized
        pieces and translated piece by piece, then rejoined — a document
        long enough to need chunking for ingestion is long enough to need
        it here too, for the same context-window reasons.
        """
        chunks = self.rag.get_chunks_for_file(
            file_name, subject=subject, limit=_CHUNKS_PER_DOCUMENT * 3
        )
        if not chunks:
            return {
                "translation": f"I don't have \"{file_name}\" ingested to translate.",
                "file_name": file_name, "target_language": target_language,
                "chunks_translated": 0,
            }

        # Regroup raw chunks into ~_MAX_CHARS_PER_DOCUMENT-sized pieces
        # rather than translating each small ingestion chunk separately —
        # fewer, larger LLM calls, and translation quality benefits from
        # more surrounding context per call than a single short chunk
        # would give it.
        pieces: list[str] = []
        buffer = ""
        for chunk in chunks:
            if len(buffer) + len(chunk) > _MAX_CHARS_PER_DOCUMENT and buffer:
                pieces.append(buffer)
                buffer = chunk
            else:
                buffer = f"{buffer}\n{chunk}" if buffer else chunk
        if buffer:
            pieces.append(buffer)

        translated_pieces: list[str] = []
        for piece in pieces:
            prompt = (
                f"Translate the following text into {target_language}. "
                f"Return ONLY the translation, no commentary or preamble.\n\n{piece}"
            )
            translated = self._call_llm(prompt, "")
            translated_pieces.append(translated or f"[translation failed for this section]")

        translation = "\n\n".join(translated_pieces)
        return {
            "translation": translation,
            "file_name": file_name,
            "target_language": target_language,
            "chunks_translated": len(pieces),
        }
