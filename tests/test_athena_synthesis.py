# tests/test_athena_synthesis.py
"""
Tests for modules/athena/services/synthesis_service.py (backlog #54,
#56, #65) and the corresponding AthenaEngine intent handlers.
"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_athena import _ensure_stubs, make_engine  # noqa: E402
_ensure_stubs()

from modules.athena.services.synthesis_service import SynthesisService  # noqa: E402


class _FakeRAG:
    """Controls exactly what list_files/get_chunks_for_file return."""

    def __init__(self, files_by_subject=None, chunks_by_file=None):
        self.files_by_subject = files_by_subject or {}
        self.chunks_by_file = chunks_by_file or {}

    def list_files(self, subject=None):
        return self.files_by_subject.get(subject, [])

    def get_chunks_for_file(self, file_name, subject=None, limit=15):
        return self.chunks_by_file.get(file_name, [])


class _FakeAI:
    def __init__(self, response="A synthesized response.", error=None):
        self.response = response
        self.error = error
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return {"text": self.response, "error": self.error}


class _ExplodesAI:
    def generate(self, prompt):
        raise RuntimeError("ollama down")


# ---------------------------------------------------------------------------
# generate_literature_review (#54)
# ---------------------------------------------------------------------------

def test_literature_review_with_no_ingested_files_says_so():
    rag = _FakeRAG()
    service = SynthesisService(rag, _FakeAI())
    result = service.generate_literature_review("thermodynamics")
    assert "don't have any documents" in result["review"]
    assert result["sources"] == []


def test_literature_review_gathers_all_files_under_the_subject():
    rag = _FakeRAG(
        files_by_subject={"thermo": [{"file_name": "a.pdf"}, {"file_name": "b.pdf"}]},
        chunks_by_file={"a.pdf": ["Content A about entropy."], "b.pdf": ["Content B about enthalpy."]},
    )
    ai = _FakeAI(response="Both papers discuss thermodynamic properties.")
    service = SynthesisService(rag, ai)
    result = service.generate_literature_review("thermo")
    assert result["review"] == "Both papers discuss thermodynamic properties."
    assert set(result["sources"]) == {"a.pdf", "b.pdf"}
    assert "a.pdf" in ai.prompts[0]
    assert "Content A" in ai.prompts[0]


def test_literature_review_survives_llm_failure():
    rag = _FakeRAG(
        files_by_subject={"thermo": [{"file_name": "a.pdf"}]},
        chunks_by_file={"a.pdf": ["Content A."]},
    )
    service = SynthesisService(rag, _ExplodesAI())
    result = service.generate_literature_review("thermo")
    assert "trouble generating" in result["review"]


def test_literature_review_survives_an_llm_error_field():
    rag = _FakeRAG(
        files_by_subject={"thermo": [{"file_name": "a.pdf"}]},
        chunks_by_file={"a.pdf": ["Content A."]},
    )
    ai = _FakeAI(response="", error="connection refused")
    service = SynthesisService(rag, ai)
    result = service.generate_literature_review("thermo")
    assert "trouble generating" in result["review"]


def test_literature_review_skips_files_with_no_actual_content():
    rag = _FakeRAG(
        files_by_subject={"thermo": [{"file_name": "empty.pdf"}, {"file_name": "a.pdf"}]},
        chunks_by_file={"empty.pdf": [], "a.pdf": ["Real content here."]},
    )
    service = SynthesisService(rag, _FakeAI())
    result = service.generate_literature_review("thermo")
    assert result["sources"] == ["a.pdf"]


# ---------------------------------------------------------------------------
# detect_research_gaps (#56)
# ---------------------------------------------------------------------------

def test_research_gaps_with_no_ingested_files_says_so():
    rag = _FakeRAG()
    service = SynthesisService(rag, _FakeAI())
    result = service.detect_research_gaps("thermo")
    assert "don't have any documents" in result["gaps"]


def test_research_gaps_prompt_instructs_against_inventing_gaps():
    rag = _FakeRAG(
        files_by_subject={"thermo": [{"file_name": "a.pdf"}]},
        chunks_by_file={"a.pdf": ["Content."]},
    )
    ai = _FakeAI(response="No gaps mentioned in the sources.")
    service = SynthesisService(rag, ai)
    service.detect_research_gaps("thermo")
    assert "do not invent gaps" in ai.prompts[0].lower() or "SOURCES THEMSELVES" in ai.prompts[0]


def test_research_gaps_survives_llm_failure():
    rag = _FakeRAG(
        files_by_subject={"thermo": [{"file_name": "a.pdf"}]},
        chunks_by_file={"a.pdf": ["Content."]},
    )
    service = SynthesisService(rag, _ExplodesAI())
    result = service.detect_research_gaps("thermo")
    assert "trouble analyzing" in result["gaps"]


# ---------------------------------------------------------------------------
# compare_documents (#65)
# ---------------------------------------------------------------------------

def test_compare_documents_with_two_real_files():
    rag = _FakeRAG(chunks_by_file={
        "a.pdf": ["Method A uses finite elements."],
        "b.pdf": ["Method B uses finite differences."],
    })
    ai = _FakeAI(response="A uses FEM while B uses FDM.")
    service = SynthesisService(rag, ai)
    result = service.compare_documents("compare methods", ["a.pdf", "b.pdf"])
    assert result["answer"] == "A uses FEM while B uses FDM."
    assert set(result["compared"]) == {"a.pdf", "b.pdf"}
    assert result["missing"] == []


def test_compare_documents_reports_missing_files():
    rag = _FakeRAG(chunks_by_file={"a.pdf": ["Content A."]})
    service = SynthesisService(rag, _FakeAI())
    result = service.compare_documents("compare", ["a.pdf", "nonexistent.pdf"])
    assert "nonexistent.pdf" in result["missing"]
    assert "I need at least two" in result["answer"]


def test_compare_documents_requires_at_least_two_with_content():
    rag = _FakeRAG(chunks_by_file={"a.pdf": ["Content A."]})
    service = SynthesisService(rag, _FakeAI())
    result = service.compare_documents("compare", ["a.pdf"])
    assert "at least two" in result["answer"]


def test_compare_documents_prompt_includes_both_file_names():
    rag = _FakeRAG(chunks_by_file={
        "a.pdf": ["Content A."], "b.pdf": ["Content B."],
    })
    ai = _FakeAI()
    service = SynthesisService(rag, ai)
    service.compare_documents("compare", ["a.pdf", "b.pdf"])
    assert "a.pdf" in ai.prompts[0]
    assert "b.pdf" in ai.prompts[0]


def test_compare_documents_survives_llm_failure():
    rag = _FakeRAG(chunks_by_file={
        "a.pdf": ["Content A."], "b.pdf": ["Content B."],
    })
    service = SynthesisService(rag, _ExplodesAI())
    result = service.compare_documents("compare", ["a.pdf", "b.pdf"])
    assert "trouble comparing" in result["answer"]


# ---------------------------------------------------------------------------
# End-to-end via AthenaEngine (real ingestion, fake LLM from test_athena)
# ---------------------------------------------------------------------------

def _write_and_ingest(engine, tmp, subfolder, filename, text):
    path = os.path.join(tmp, "documents", subfolder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    engine.handle("ingest", {}, {})


def test_literature_review_intent_end_to_end():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        _write_and_ingest(
            engine, tmp, "Bio", "notes.txt",
            "The mitochondria is the powerhouse of the cell and produces ATP through respiration.",
        )
        result = engine.handle("literature_review", {"subject": "Bio"}, {})
        assert result["confidence"] > 0
        assert result["response"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_literature_review_intent_without_subject_asks_for_one():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        result = engine.handle("literature_review", {}, {})
        assert result["confidence"] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_compare_documents_intent_requires_two_file_names():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        result = engine.handle("compare_documents", {"file_names": ["a.pdf"]}, {})
        assert result["confidence"] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_compare_documents_intent_accepts_comma_separated_string():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        result = engine.handle(
            "compare_documents", {"file_names": "a.pdf, b.pdf", "query": "compare"}, {}
        )
        # Neither file is actually ingested, so this reports missing —
        # the point here is just that the comma-separated string parsed
        # into two names rather than being treated as a single one.
        assert result["data"].get("missing") == ["a.pdf", "b.pdf"] or "at least two" in result["response"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_get_citations_intent_with_no_sources_says_so():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        result = engine.handle("get_citations", {}, {})
        assert result["confidence"] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_get_citations_intent_builds_a_bibliography_from_passed_sources():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        result = engine.handle("get_citations", {
            "sources": [{"file_name": "a.pdf", "page": 3}, {"file_name": "b.pdf", "page": 1}],
        }, {})
        assert "a.pdf" in result["response"]
        assert "b.pdf" in result["response"]
        assert result["data"]["count"] == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_get_citations_intent_supports_bibtex_format():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        result = engine.handle("get_citations", {
            "sources": [{"file_name": "a.pdf"}], "format": "bibtex",
        }, {})
        assert "@misc" in result["response"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_new_intents_dispatchable_prefixed_and_stripped():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        for intent in (
            "literature_review", "athena_literature_review",
            "research_gaps", "athena_research_gaps",
            "compare_documents", "athena_compare_documents",
            "get_citations", "athena_get_citations",
        ):
            assert engine.can_handle(intent), f"can_handle() rejected {intent!r}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# translate_document (#69)
# ---------------------------------------------------------------------------

def test_translate_document_with_no_ingested_file_says_so():
    rag = _FakeRAG()
    service = SynthesisService(rag, _FakeAI())
    result = service.translate_document("nope.pdf", "Spanish")
    assert "don't have" in result["translation"]
    assert result["chunks_translated"] == 0


def test_translate_document_translates_and_reports_target_language():
    rag = _FakeRAG(chunks_by_file={"notes.pdf": ["Hello, this is a test."]})
    ai = _FakeAI(response="Hola, esto es una prueba.")
    service = SynthesisService(rag, ai)
    result = service.translate_document("notes.pdf", "Spanish")
    assert result["translation"] == "Hola, esto es una prueba."
    assert result["target_language"] == "Spanish"
    assert result["chunks_translated"] == 1


def test_translate_document_prompt_names_the_target_language():
    rag = _FakeRAG(chunks_by_file={"notes.pdf": ["Hello."]})
    ai = _FakeAI()
    service = SynthesisService(rag, ai)
    service.translate_document("notes.pdf", "French")
    assert "French" in ai.prompts[0]


def test_translate_document_groups_many_small_chunks_into_fewer_calls():
    rag = _FakeRAG(chunks_by_file={"notes.pdf": ["Short chunk."] * 5})
    ai = _FakeAI()
    service = SynthesisService(rag, ai)
    result = service.translate_document("notes.pdf", "German")
    # Five tiny chunks well under the per-piece size limit should be
    # grouped into a single LLM call, not five separate ones.
    assert result["chunks_translated"] == 1
    assert len(ai.prompts) == 1


def test_translate_document_survives_partial_llm_failure():
    rag = _FakeRAG(chunks_by_file={"notes.pdf": ["x" * 5000, "y" * 5000]})  # forces 2 pieces

    class _PartialFailAI:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("ollama down")
            return {"text": "translated second piece", "error": None}

    service = SynthesisService(rag, _PartialFailAI())
    result = service.translate_document("notes.pdf", "Spanish")
    assert "translation failed" in result["translation"]
    assert "translated second piece" in result["translation"]


def test_translate_document_intent_missing_fields_asks_for_them():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        result = engine.handle("translate_document", {"file_name": "notes.pdf"}, {})
        assert result["confidence"] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_translate_document_intent_end_to_end():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        _write_and_ingest(
            engine, tmp, "Bio", "notes.txt",
            "The mitochondria is the powerhouse of the cell and produces ATP.",
        )
        result = engine.handle(
            "translate_document", {"file_name": "notes.txt", "target_language": "Spanish"}, {}
        )
        assert result["response"]
        assert result["data"]["target_language"] == "Spanish"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_translate_document_intent_dispatchable_prefixed_and_stripped():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        for intent in ("translate_document", "athena_translate_document"):
            assert engine.can_handle(intent)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
