# tests/test_athena_pdf_structures.py
"""
Tests for backlog #58 (table extraction) and #59 (figure-caption indexing).

PDFs are generated on the fly with reportlab and read back with the real
pdfplumber. The Athena suite stubs PyMuPDF, so the end-to-end ingest tests
feed the extracted chunks through the RAG directly.
"""
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

pytest.importorskip("pdfplumber")
reportlab = pytest.importorskip("reportlab")

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

from test_athena import make_engine  # noqa: E402
from modules.athena import pdf_structures as ps  # noqa: E402


def build_pdf(path, with_table=True, with_figures=True):
    styles = getSampleStyleSheet()
    story = [Paragraph("A Study of Image Classifiers", styles["Title"]),
             Paragraph("Figure 2 shows the loss in the experiment described later.", styles["Normal"]),
             Spacer(1, 12)]
    if with_table:
        story.append(Paragraph("Table 1: Accuracy of models on ImageNet", styles["Normal"]))
        data = [["Model", "Accuracy", "Params"],
                ["ResNet", "92.1", "25M"],
                ["VGG", "90.4", "138M"],
                ["ViT", "94.7", "86M"]]
        t = Table(data)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black)]))
        story += [t, Spacer(1, 18)]
    if with_figures:
        story += [Paragraph("Figure 2: Training loss over epochs for each model. Lower is better.", styles["Normal"]),
                  Spacer(1, 12),
                  Paragraph("Fig. 3 - Confusion matrix of the best classifier.", styles["Normal"])]
    story += [PageBreak(), Paragraph("Plain second page with ordinary prose only. " * 5, styles["Normal"])]
    SimpleDocTemplate(path, pagesize=A4).build(story)


@pytest.fixture
def pdf(tmp_path):
    p = str(tmp_path / "paper.pdf")
    build_pdf(p)
    return p


# ---- pure helpers ----------------------------------------------------------

def test_clean_cell_collapses_whitespace_and_none():
    assert ps.clean_cell("  a \n b ") == "a b"
    assert ps.clean_cell(None) == ""
    assert ps.clean_cell(3.5) == "3.5"


def test_normalise_table_drops_empty_rows_and_columns_and_pads_ragged_rows():
    rows = [["A", "", "B"], ["1", None, "2"], [None, None, None], ["3", ""]]
    assert ps.normalise_table(rows) == [["A", "B"], ["1", "2"], ["3", ""]]


@pytest.mark.parametrize("rows", [None, [], [["only header", "x"]], [["a"], ["b"], ["c"]]])
def test_normalise_table_rejects_things_that_are_not_tables(rows):
    assert ps.normalise_table(rows) is None


def test_table_text_labels_every_value_with_its_column():
    texts = ps.table_to_texts([["Model", "Accuracy"], ["ResNet", "92.1"]], page=4,
                              label="Table 1", caption="Accuracy of models")
    assert len(texts) == 1
    assert texts[0].startswith("Table 1 (page 4): Accuracy of models\nColumns: Model | Accuracy\n")
    assert "Model: ResNet; Accuracy: 92.1" in texts[0]


def test_table_text_skips_empty_cells_and_names_blank_headers():
    texts = ps.table_to_texts([["", "Score"], ["a", ""], ["b", "5"]], page=1)
    assert "Column 1 | Score" in texts[0]
    assert "Column 1: a" in texts[0] and "Column 1: b; Score: 5" in texts[0]


def test_header_only_table_yields_no_chunks():
    assert ps.table_to_texts([["A", "B"], ["", ""]], page=1) == []


def test_large_table_splits_by_rows_and_repeats_the_header_in_every_chunk():
    rows = [["Name", "Value"]] + [[f"item{i}", str(i)] for i in range(200)]
    texts = ps.table_to_texts(rows, page=2, label="Table 9", max_chars=400)
    assert len(texts) > 3
    assert all(t.startswith("Table 9 (page 2)\nColumns: Name | Value\n") for t in texts)
    joined = "\n".join(texts)
    assert all(f"Name: item{i}; Value: {i}" in joined for i in range(200))


def test_caption_regex_accepts_real_forms_and_rejects_body_text():
    caps = ps.extract_page_captions(
        "Intro text.\nFigure 3: Loss curves.\nFig. 4 - Confusion matrix.\nTable 2. Results.\n"
        "Figure 5 shows the trend in detail.\nAs seen in Table 1 the values rise.")
    assert [(c["kind"], c["label"]) for c in caps] == [
        ("figure", "Figure 3"), ("figure", "Figure 4"), ("table", "Table 2")]


def test_wrapped_caption_is_joined_until_a_sentence_ends():
    caps = ps.extract_page_captions(
        "Figure 1: Overview of the proposed pipeline with\nthree stages: encode, align and decode.\n"
        "Body text continues here.")
    assert caps[0]["text"] == "Overview of the proposed pipeline with three stages: encode, align and decode."


def test_single_line_caption_ending_in_a_period_does_not_swallow_the_next_line():
    caps = ps.extract_page_captions("Figure 1: A short caption.\nThe next paragraph starts here.")
    assert caps[0]["text"] == "A short caption."


def test_consecutive_captions_are_not_merged():
    caps = ps.extract_page_captions("Figure 1: first\nFigure 2: second")
    assert [c["text"] for c in caps] == ["first", "second"]


def test_caption_length_is_capped():
    caps = ps.extract_page_captions("Figure 1: " + "word " * 400)
    assert len(caps[0]["text"]) <= 500


@pytest.mark.parametrize("query,expected", [
    ("find the graph that shows training loss", "figure"),
    ("show me the chart of accuracy", "figure"),
    ("Which table lists the model parameters?", "table"),
    ("where is the figure about attention", "figure"),
    ("please find the plot of loss vs epochs", "figure"),
    ("explain graph theory to me", None),
    ("what is a table", "table"),
    ("summarise the paper", None),
    ("", None),
])
def test_infer_content_type_only_fires_on_find_style_requests(query, expected):
    assert ps.infer_content_type(query) == expected


def test_lexical_overlap_ignores_filler_words():
    assert ps.lexical_overlap("find the graph that shows training loss",
                              "Figure 2 (page 3): Training loss over epochs") == 1.0
    assert ps.lexical_overlap("find the graph that shows training loss", "unrelated words here") == 0.0
    assert ps.lexical_overlap("find the graph", "anything") == 0.0


# ---- real PDFs -------------------------------------------------------------

def test_table_is_extracted_from_a_real_pdf_with_caption_and_labelled_rows(pdf):
    chunks = ps.extract_structures(pdf, index_figures=False)
    tables = [c for c in chunks if c["content_type"] == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert t["page_number"] == 1 and t["chunk_number"] == 1000 and t["file_name"] == "paper.pdf"
    assert t["text"].startswith("Table 1 (page 1): Accuracy of models on ImageNet")
    assert "Model: ResNet; Accuracy: 92.1; Params: 25M" in t["text"]
    assert "Model: ViT; Accuracy: 94.7; Params: 86M" in t["text"]


def test_figure_captions_are_extracted_and_body_mentions_are_not(pdf):
    chunks = ps.extract_structures(pdf, extract_tables=False)
    figs = [c for c in chunks if c["content_type"] == "figure"]
    assert [f["chunk_number"] for f in figs] == [2000, 2001]
    assert figs[0]["text"] == "Figure 2 (page 1): Training loss over epochs for each model. Lower is better."
    assert figs[1]["text"] == "Figure 3 (page 1): Confusion matrix of the best classifier."
    assert all("shows the loss" not in f["text"] for f in figs)


def test_chunk_dicts_carry_everything_the_ingest_pipeline_needs(pdf):
    for c in ps.extract_structures(pdf):
        assert {"text", "file_name", "file_path", "page_number", "chunk_number",
                "total_pages", "total_chunks", "content_type"} <= set(c)
        assert c["total_pages"] == 2 and c["file_path"] == pdf


def test_page_without_structures_adds_nothing(tmp_path):
    p = str(tmp_path / "plain.pdf")
    build_pdf(p, with_table=False, with_figures=False)
    assert ps.extract_structures(p) == []


def test_both_features_can_be_switched_off(pdf):
    assert ps.extract_structures(pdf, extract_tables=False, index_figures=False) == []


def test_unreadable_or_missing_pdf_returns_empty_not_an_exception(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    assert ps.extract_structures(str(bad)) == []
    assert ps.extract_structures(str(tmp_path / "missing.pdf")) == []


def test_max_pages_limits_the_scan(tmp_path):
    p = str(tmp_path / "two.pdf")
    build_pdf(p)
    assert ps.extract_structures(p, max_pages=0) == []


def test_chunk_ids_never_collide_with_text_chunks(pdf):
    from modules.athena.local_rag import _chunk_id
    fi = {"full_path": pdf, "subject": "ml", "module": "vision"}
    ids = {_chunk_id(fi, c) for c in ps.extract_structures(pdf)}
    ids.add(_chunk_id(fi, {"page_number": 1, "chunk_number": 1}))
    assert len(ids) == len(ps.extract_structures(pdf)) + 1


# ---- processor, metadata, signature ---------------------------------------

def test_processor_adds_structure_chunks_and_respects_config(pdf, monkeypatch):
    from modules.athena.config import get_config
    from modules.athena.pdf_processor import PDFProcessor
    got = PDFProcessor._structured_chunks(pdf)
    assert {c["content_type"] for c in got} == {"table", "figure"}
    monkeypatch.setattr(get_config(), "extract_tables", False, raising=False)
    monkeypatch.setattr(get_config(), "index_figure_captions", False, raising=False)
    assert PDFProcessor._structured_chunks(pdf) == []


def test_processor_swallows_extractor_crashes(pdf, monkeypatch):
    from modules.athena.pdf_processor import PDFProcessor
    monkeypatch.setattr(ps, "extract_structures", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert PDFProcessor._structured_chunks(pdf) == []


def test_metadata_records_content_type_with_text_as_default():
    from modules.athena.local_rag import _prepare_batch
    fi = {"full_path": "/x/p.pdf", "subject": "ml", "module": "v"}
    _, _, metas = _prepare_batch(
        fi, [{"text": "plain", "page_number": 1, "chunk_number": 1},
             {"text": "t", "page_number": 1, "chunk_number": 1000, "content_type": "table"}], "/x/p.pdf")
    assert [m["content_type"] for m in metas] == ["text", "table"]


def test_pdf_signature_changes_once_so_old_pdfs_get_reprocessed(tmp_path, monkeypatch):
    from modules.athena.config import get_config
    from modules.athena.local_rag import _file_signature
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4")
    t = tmp_path / "a.txt"
    t.write_text("x")
    assert _file_signature(str(f)).endswith(":s1")
    assert not _file_signature(str(t)).endswith(":s1")
    monkeypatch.setattr(get_config(), "extract_tables", False, raising=False)
    monkeypatch.setattr(get_config(), "index_figure_captions", False, raising=False)
    assert not _file_signature(str(f)).endswith(":s1")


# ---- engine: ingest -> search ----------------------------------------------

@pytest.fixture
def engine_with_pdf(pdf):
    tmp = tempfile.mkdtemp()
    engine = make_engine(tmp)
    folder = os.path.join(tmp, "documents", "ML")
    os.makedirs(folder)
    shutil.copy(pdf, os.path.join(folder, "paper.pdf"))
    from modules.athena.pdf_processor import PDFProcessor
    real_structured = PDFProcessor._structured_chunks
    engine.rag._pdf_processor.process_pdf = lambda path, **kw: [
        {"text": "Body prose about convolutional classifiers and their training.",
         "file_name": "paper.pdf", "file_path": path, "page_number": 1,
         "chunk_number": 1, "total_chunks": 1, "total_pages": 2}
    ] + real_structured(path)
    yield engine
    shutil.rmtree(tmp, ignore_errors=True)


def test_ingest_indexes_table_and_figure_chunks_with_their_content_type(engine_with_pdf):
    engine_with_pdf.handle("ingest", {}, {})
    assert engine_with_pdf.rag.has_content_type("table")
    assert engine_with_pdf.rag.has_content_type("figure")
    assert not engine_with_pdf.rag.has_content_type("equation")


def test_find_the_graph_returns_the_right_caption_and_page(engine_with_pdf):
    engine_with_pdf.handle("ingest", {}, {})
    r = engine_with_pdf.handle("search", {"query": "find the graph that shows training loss over epochs"}, {})
    assert "paper.pdf, page 1: Figure 2 (page 1)" in r["response"]
    top = r["data"]["sources"][0]
    assert top["page"] == 1 and "Training loss" in top["text"]
    assert r["data"]["content_type"] == "figure"


def test_which_table_finds_the_table_by_its_contents(engine_with_pdf):
    engine_with_pdf.handle("ingest", {}, {})
    r = engine_with_pdf.handle("search", {"query": "which table lists accuracy for ResNet"}, {})
    assert r["data"]["content_type"] == "table"
    assert "Accuracy: 92.1" in r["data"]["sources"][0]["text"]


def test_explicit_content_type_entity_is_honoured(engine_with_pdf):
    engine_with_pdf.handle("ingest", {}, {})
    r = engine_with_pdf.handle("search", {"query": "confusion matrix", "content_type": "figure"}, {})
    assert "Fig" in r["data"]["sources"][0]["text"] or "Figure" in r["data"]["sources"][0]["text"]


def test_structured_search_with_nothing_indexed_explains_how_to_fix_it():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        r = engine.handle("search", {"query": "find the graph that shows loss"}, {})
        assert "No figures are indexed yet" in r["response"] and "ingest" in r["response"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ordinary_questions_mentioning_graphs_still_use_normal_search(engine_with_pdf):
    engine_with_pdf.handle("ingest", {}, {})
    called = []
    engine_with_pdf._handle_structured_search = lambda *a, **k: called.append(1)
    engine_with_pdf.handle("search", {"query": "explain graph theory"}, {})
    assert called == []


def test_a_structured_search_error_falls_back_to_normal_search(engine_with_pdf):
    engine_with_pdf.handle("ingest", {}, {})
    engine_with_pdf.rag.search_by_content_type = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    r = engine_with_pdf.handle("search", {"query": "find the graph that shows training loss"}, {})
    assert "content_type" not in r["data"]
    assert isinstance(r["data"].get("sources"), list)
    assert r["confidence"] > 0
