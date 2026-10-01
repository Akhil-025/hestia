# tests/test_athena_citation_graph.py
"""
Tests for the cross-document citation graph (backlog #60): reference parsing and
matching (modules/athena/bibliography.py), the graph service, the renderers and the
`athena_citation_graph` intent.

PDFs are generated with reportlab and read back with the real PyMuPDF. The service
imports `pymupdf` (not `fitz`) because tests/test_athena.py's stub setup can put a fake
`fitz` first on sys.path. No network, no LLM.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_athena import _ensure_stubs, make_engine  # noqa: E402
_ensure_stubs()

from modules.athena import bibliography as bib  # noqa: E402
from modules.athena import citation_graph_view as view  # noqa: E402
from modules.athena.services import citation_graph_service as cg  # noqa: E402

pymupdf = pytest.importorskip("pymupdf")
reportlab_canvas = pytest.importorskip("reportlab.pdfgen.canvas")

T_ATTN = "Attention Is All You Need"
T_RESNET = "Deep Residual Learning for Image Recognition"
T_ADAM = "Adam: A Method for Stochastic Optimization"


def _make_pdf(path, title, body_lines, refs=None, arxiv=None):
    c = reportlab_canvas.Canvas(path)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(60, 790, title)
    c.setFont("Helvetica", 10)
    y = 760
    if arxiv:
        c.drawString(60, y, f"arXiv:{arxiv} [cs.LG] 6 Dec 2017"); y -= 16
    for line in body_lines:
        c.drawString(60, y, line); y -= 14
    if refs is not None:
        c.showPage()
        c.setFont("Helvetica-Bold", 14); c.drawString(60, 790, "References")
        c.setFont("Helvetica", 9)
        y = 770
        for r in refs:
            c.drawString(60, y, r); y -= 13
    c.save()


class _Rag:
    def __init__(self, sources):
        self.sources = sources

    def list_document_sources(self, subject=None):
        return [s for s in self.sources if not subject or s["subject"] == subject]


@pytest.fixture()
def library(tmp_path):
    d = tmp_path / "docs"; d.mkdir()
    attn = str(d / "attn.pdf"); res = str(d / "resnet.pdf"); adam = str(d / "adam.pdf"); notes = str(d / "scan.pdf")
    _make_pdf(attn, T_ATTN, ["We study sequence models."],
              ['[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," CVPR, 2016.',
               '[2] D. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv:1412.6980, 2014.',
               '[3] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, 1997.'],
              arxiv="1706.03762")
    _make_pdf(res, T_RESNET, ["Residual networks."],
              ['[1] D. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv:1412.6980, 2014.',
               '[2] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, 1997.'])
    _make_pdf(adam, T_ADAM, ["Adaptive moments."], arxiv="1412.6980")
    _make_pdf(notes, "Lecture notes without a bibliography", ["Just notes."])
    srcs = [{"file_name": os.path.basename(p), "subject": "ml", "module": "m", "file_path": p}
            for p in (attn, res, adam, notes)]
    return srcs, d


def _node(g, file_name):
    return next(n for n in g["nodes"] if n["file_name"] == file_name)


# ---- parsing ---------------------------------------------------------------

def test_reference_section_is_found_and_split():
    text = "Body.\nReferences\n[1] A. One, \"First paper title here,\" 2015.\n[2] B. Two, \"Second paper title here,\" 2016.\n"
    refs, found = bib.parse_references(text)
    assert found and len(refs) == 2 and refs[0].years == [2015]


def test_no_reference_heading_means_not_found():
    assert bib.parse_references("just prose, no list") == ([], False)


def test_doi_and_arxiv_ids_are_read_from_entries():
    refs, _ = bib.parse_references("References\n[1] X. Y, \"A title of some length,\" doi:10.1109/CVPR.2016.90.\n"
                                   "[2] Z. W, \"Another long enough title,\" arXiv:1412.6980, 2014.\n")
    assert refs[0].doi == "10.1109/cvpr.2016.90" and refs[1].arxiv_id == "1412.6980"


def test_short_generic_title_never_matches_by_title():
    doc = bib.LibraryDoc("d", bib.Identity(title="Attention", title_source="font-size"))
    ref = bib.parse_reference('[1] A. B, "Attention is all you need," 2017.')
    assert bib.match_references({"s": [ref]}, [doc]).links == []


def test_two_equally_good_documents_make_no_link():
    d1 = bib.LibraryDoc("d1", bib.Identity(title="Learning Deep Features Quickly", title_source="font-size"))
    d2 = bib.LibraryDoc("d2", bib.Identity(title="Learning Deep Features Quickly", title_source="font-size"))
    ref = bib.parse_reference('[1] A. B, "Learning Deep Features Quickly," 2015.')
    out = bib.match_references({"s": [ref]}, [d1, d2])
    assert out.links == [] and len(out.ambiguous) == 1


# ---- service on real PDFs --------------------------------------------------

def test_graph_links_follow_the_reference_lists(library):
    srcs, d = library
    g = cg.CitationGraphService(_Rag(srcs), str(d / "cache.json")).build()
    pairs = {(next(n for n in g["nodes"] if n["id"] == l["source"])["file_name"],
              next(n for n in g["nodes"] if n["id"] == l["target"])["file_name"]) for l in g["links"]}
    assert pairs == {("attn.pdf", "resnet.pdf"), ("attn.pdf", "adam.pdf"), ("resnet.pdf", "adam.pdf")}
    assert _node(g, "adam.pdf")["cited_by"] == 2 and _node(g, "attn.pdf")["cited_by"] == 0
    assert g["stats"]["most_cited"][0]["label"].startswith("Adam")


def test_identity_is_read_from_the_first_page(library):
    srcs, d = library
    g = cg.CitationGraphService(_Rag(srcs), None).build()
    assert _node(g, "attn.pdf")["label"] == T_ATTN
    assert _node(g, "attn.pdf")["arxiv_id"] == "1706.03762" and _node(g, "attn.pdf")["year"] == 2017


def test_document_without_reference_list_is_flagged_not_guessed(library):
    srcs, _ = library
    g = cg.CitationGraphService(_Rag(srcs), None).build()
    n = _node(g, "scan.pdf")
    assert n["status"] == "no_reference_section" and n["cites"] == 0
    assert any("no readable reference list" in note for note in g["notes"])


def test_reference_cited_by_two_papers_but_not_in_library_is_reported(library):
    srcs, _ = library
    g = cg.CitationGraphService(_Rag(srcs), None).build()
    assert [m["title"] for m in g["missing_but_cited"]] == ["Long short-term memory"] or \
        any("memory" in (m["title"] or m["reference"]).lower() for m in g["missing_but_cited"])
    assert g["missing_but_cited"][0]["cited_by_count"] == 2


def test_cache_is_used_and_invalidated_by_a_changed_file(library, monkeypatch):
    srcs, d = library
    cache = str(d / "cache.json")
    cg.CitationGraphService(_Rag(srcs), cache).build()
    calls = []
    real = cg.read_document
    monkeypatch.setattr(cg, "read_document", lambda p, n: calls.append(n) or real(p, n))
    cg.CitationGraphService(_Rag(srcs), cache).build()
    assert calls == []                                   # everything came from the cache
    path = srcs[2]["file_path"]
    st = os.stat(path); os.utime(path, (st.st_atime, st.st_mtime + 10))
    cg.CitationGraphService(_Rag(srcs), cache).build()
    assert calls == ["adam.pdf"]


def test_missing_and_corrupt_files_do_not_raise(tmp_path):
    bad = tmp_path / "bad.pdf"; bad.write_bytes(b"not a pdf")
    srcs = [{"file_name": "bad.pdf", "subject": "s", "module": "m", "file_path": str(bad)},
            {"file_name": "gone.pdf", "subject": "s", "module": "m", "file_path": str(tmp_path / "gone.pdf")}]
    g = cg.CitationGraphService(_Rag(srcs), str(tmp_path / "c.json")).build()
    assert {n["status"] for n in g["nodes"]} == {"unreadable", "missing"} and g["links"] == []
    assert not (tmp_path / "c.json").read_text() or "bad.pdf" not in (tmp_path / "c.json").read_text()


def test_markdown_paper_files_are_cited_by_arxiv_id(tmp_path):
    md = tmp_path / "arxiv_1412.6980.md"
    md.write_text("# Adam: A Method for Stochastic Optimization\n\n- arXiv: 1412.6980\n- Authors: D. Kingma, J. Ba\n"
                  "- Published: 2014-12-22\n\n## Abstract\nWe introduce Adam.\n", encoding="utf-8")
    paper = tmp_path / "paper.md"
    paper.write_text("# My survey of optimisers in practice\n\nBody text.\n\n## References\n"
                     "[1] D. Kingma and J. Ba, Adam, arXiv:1412.6980, 2014.\n", encoding="utf-8")
    srcs = [{"file_name": p.name, "subject": "s", "module": "m", "file_path": str(p)} for p in (md, paper)]
    g = cg.CitationGraphService(_Rag(srcs), None).build()
    assert len(g["links"]) == 1 and g["links"][0]["method"] == "arxiv" and g["links"][0]["confidence"] == 1.0
    assert "arXiv:1412.6980" in g["links"][0]["evidence"]


def test_subject_filter_limits_documents(library):
    srcs, _ = library
    srcs[0] = dict(srcs[0], subject="other")
    g = cg.CitationGraphService(_Rag(srcs), None).build("ml")
    assert len(g["nodes"]) == 3 and g["subject"] == "ml"


def test_anachronistic_link_is_marked_suspicious():
    ident_old = bib.Identity(title="An old paper about things", year=2010, year_source="arxiv")
    ident_new = bib.Identity(title="A newer paper about other things", year=2020, year_source="arxiv")

    class R(_Rag):
        pass
    svc = cg.CitationGraphService(R([{"file_name": "old.md", "subject": "s", "module": "m", "file_path": ""},
                                     {"file_name": "new.md", "subject": "s", "module": "m", "file_path": ""}]), None)
    refs = [bib.parse_reference('[1] A. B, "A newer paper about other things," 2020.')]
    parsed = {"old.md": {"identity": ident_old, "references": refs, "status": "ok"},
              "new.md": {"identity": ident_new, "references": [], "status": "no_reference_section"}}
    svc._cache.get = lambda path: None
    orig = cg.read_document
    cg.read_document = lambda p, n: parsed[n]
    try:
        g = svc.build()
    finally:
        cg.read_document = orig
    assert g["links"] and "later than" in g["links"][0]["suspicious"]


# ---- asking about the graph ------------------------------------------------

def test_find_node_exact_unique_and_ambiguous(library):
    srcs, _ = library
    g = cg.CitationGraphService(_Rag(srcs), None).build()
    assert cg.find_node(g, "adam.pdf")[0]["file_name"] == "adam.pdf"
    assert cg.find_node(g, "residual")[0]["file_name"] == "resnet.pdf"
    node, cands = cg.find_node(g, "pdf")
    assert node is None and len(cands) == 4


def test_describe_for_whole_graph_and_for_one_paper(library):
    srcs, _ = library
    g = cg.CitationGraphService(_Rag(srcs), None).build()
    assert "3 citation(s)" in cg.describe(g)
    text = cg.describe(g, cg.find_node(g, "attn.pdf")[0])
    assert "cites 2 of your papers" in text and "cited by 0" in text


@pytest.mark.parametrize("q,expected", [
    ("show the citation graph", ("", "")),
    ("citation graph for physics papers", ("physics", "")),
    ("who cites attention is all you need", ("", "attention is all you need")),
    ("what does bert.pdf cite", ("", "bert.pdf")),
])
def test_parse_graph_request(q, expected):
    assert cg.parse_graph_request(q) == expected


# ---- renderers -------------------------------------------------------------

def test_html_is_self_contained_and_script_safe():
    g = {"subject": "", "nodes": [{"id": "n1", "label": "</script><b>x", "file_name": "a.pdf", "subject": "s", "year": None,
                                   "authors": [], "arxiv_id": "", "doi": "", "status": "ok", "references": 0,
                                   "cites": 0, "cited_by": 0}],
         "links": [], "stats": {"documents": 1, "links": 0, "connected_documents": 0, "most_cited": []},
         "missing_but_cited": [], "notes": []}
    html = view.render_html(g)
    assert html.count("</script>") == 3                  # data + layout + ui only: the label's was escaped
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "") and "https://" not in html


def test_dot_output_and_files(library, tmp_path):
    srcs, _ = library
    g = cg.CitationGraphService(_Rag(srcs), None).build()
    dot = view.to_dot(g)
    assert dot.startswith("digraph citations") and "->" in dot and "scan" not in dot
    files = view.write_graph_files(g, str(tmp_path / "out"), "My Graph!", "2026-10-01", ("html", "json", "dot", "bogus"))
    assert [os.path.basename(f) for f in files] == ["my-graph-2026-10-01.html", "my-graph-2026-10-01.json", "my-graph-2026-10-01.dot"]
    assert json.loads(open(files[1], encoding="utf-8").read())["stats"]["links"] == 3


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_layout_is_deterministic_and_keeps_nodes_in_bounds(tmp_path):
    script = view.LAYOUT_JS + """
    var mk = function () { return [{id:'a',year:2015},{id:'b',year:2017},{id:'c',year:2019}]; };
    var links = [{source:'b',target:'a'},{source:'c',target:'b'}];
    var r1 = CitationLayout.layout(mk(), links, {width:900,height:600}), r2 = CitationLayout.layout(mk(), links, {width:900,height:600});
    var ok = r1.every(function (n, i) { return n.x === r2[i].x && n.y === r2[i].y && isFinite(n.x) && n.x >= 0 && n.x <= 900 && n.y >= 0 && n.y <= 600; });
    var y = CitationLayout.layout(mk(), links, {width:900,height:600,mode:'year'});
    console.log(JSON.stringify({ok: ok, ordered: y[0].x < y[1].x && y[1].x < y[2].x}));
    """
    f = tmp_path / "t.js"; f.write_text(script)
    out = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=30)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {"ok": True, "ordered": True}, out.stderr


# ---- the intent ------------------------------------------------------------

@pytest.fixture()
def engine(library):
    srcs, _ = library
    tmp = tempfile.mkdtemp()
    try:
        eng = make_engine(tmp)
        eng.rag.list_document_sources = lambda subject=None: [s for s in srcs if not subject or s["subject"] == subject]
        eng.rag.list_files = lambda subject=None: [{"file_name": s["file_name"], "subject": s["subject"]} for s in srcs]
        yield eng
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_intent_is_registered_and_handled(engine):
    from modules.hecate.intent_registry import INTENT_MODULE_MAP
    assert INTENT_MODULE_MAP["athena_citation_graph"] == "athena"
    for intent in ("athena_citation_graph", "citation_graph"):
        assert engine.can_handle(intent)


def test_intent_writes_html_and_json_and_speaks_a_summary(engine):
    r = engine.handle("citation_graph", {}, {"raw_query": "show the citation graph"})
    assert r["confidence"] == 0.9 and "3 citation(s)" in r["response"]
    assert sorted(os.path.splitext(f)[1] for f in r["data"]["files"]) == [".html", ".json"]
    assert all(os.path.isfile(f) for f in r["data"]["files"])


def test_intent_focus_on_one_paper(engine):
    r = engine.handle("citation_graph", {"raw_query": "who cites adam.pdf"}, {})
    assert "cited by 2" in r["response"] and r["data"]["focus"] == "adam.pdf"


def test_intent_unknown_subject_and_unknown_paper_are_explained(engine):
    assert "don't have a subject called nope" in engine.handle("citation_graph", {"subject": "nope"}, {})["response"]
    assert "couldn't find a paper" in engine.handle("citation_graph", {"focus": "zzz"}, {})["response"]


def test_intent_with_dot_format(engine):
    r = engine.handle("citation_graph", {}, {"raw_query": "draw the citation graph as dot"})
    assert any(f.endswith(".dot") for f in r["data"]["files"])


def test_engine_html_for_web_view(engine):
    html = engine.citation_graph_html(None, "adam.pdf")
    assert '"focus": "n3"' in html


def test_a_paper_does_not_take_the_identity_of_a_paper_it_cites(tmp_path):
    """Regression: a short document's own reference list used to supply its arXiv id."""
    paper = tmp_path / "paper.md"
    paper.write_text("# My survey of optimisers in practice\n\nBody text.\n\n## References\n"
                     "[1] D. Kingma and J. Ba, Adam, arXiv:1412.6980, 2014.\n", encoding="utf-8")
    ident = cg.read_document(str(paper), "paper.md")["identity"]
    assert ident.arxiv_id == "" and ident.title == "My survey of optimisers in practice"


def test_two_entry_reference_lists_are_split():
    refs, _ = bib.parse_references("References\n[1] A. One, \"First paper title here,\" 2015.\n[2] B. Two, \"Second paper title here,\" 2016.\n")
    assert len(refs) == 2


# ---- web routes ------------------------------------------------------------

def test_web_routes_serve_json_and_a_sandboxable_page(engine):
    from web_ui import HestiaWebUI
    client = HestiaWebUI(memory=None, athena=engine).app.test_client()
    data = client.get("/api/athena/citation-graph").get_json()
    assert data["stats"]["links"] == 3
    page = client.get("/api/athena/citation-graph/view?focus=adam.pdf")
    assert page.status_code == 200 and page.mimetype == "text/html"
    assert "default-src 'none'" in page.headers["Content-Security-Policy"] and b'"focus": "n3"' in page.data
    assert client.get("/api/athena/citation-graph?subject=nope").get_json()["nodes"] == []


def test_web_routes_say_disabled_without_athena():
    from web_ui import HestiaWebUI
    client = HestiaWebUI(memory=None).app.test_client()
    assert client.get("/api/athena/citation-graph").status_code == 503
    assert client.get("/api/athena/citation-graph/view").status_code == 503
