# tests/test_athena_progress_ocr.py
"""
Tests for backlog #70 (ingestion progress reporting) and #68 (OCR language
auto-detection). Reuses tests/test_athena.py's make_engine fixture.

Language behaviour is tested through the injectable probe function and by
faking the installed-language list, never by relying on real foreign OCR.
"""
import os
import shutil
import sys
import tempfile
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_athena import make_engine  # noqa: E402
from modules.athena.progress import IngestionProgress  # noqa: E402
from modules.athena import ocr_language as ocr  # noqa: E402


def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _wait_idle(engine, secs=5):
    deadline = time.time() + secs
    while engine.ingest_status()["running"] and time.time() < deadline:
        time.sleep(0.02)


# ===========================================================================
# #70 progress
# ===========================================================================

def test_progress_starts_idle():
    snap = IngestionProgress().snapshot()
    assert snap["running"] is False and snap["total"] == 0 and snap["percent"] == 0
    assert snap["elapsed_seconds"] is None and snap["result"] is None


def test_progress_tracks_files_percent_and_failures():
    p = IngestionProgress()
    p.start(4)
    p.begin_file("a.pdf")
    assert p.snapshot()["current"] == "a.pdf" and p.snapshot()["running"]
    p.file_done("a.pdf", "new")
    p.file_done("b.pdf", "failed")
    snap = p.snapshot()
    assert (snap["done"], snap["total"], snap["percent"]) == (2, 4, 50)
    assert snap["failed"] == ["b.pdf"]


def test_progress_finish_records_result_and_clears_current():
    p = IngestionProgress()
    p.start(1)
    p.begin_file("a.txt")
    p.file_done("a.txt", "new")
    p.finish({"new_files": 1})
    snap = p.snapshot()
    assert snap["running"] is False and snap["current"] is None
    assert snap["result"] == {"new_files": 1} and snap["percent"] == 100
    assert snap["elapsed_seconds"] is not None


def test_empty_run_reads_as_complete_not_zero_percent():
    p = IngestionProgress()
    p.start(0)
    p.finish({"total_files": 0})
    assert p.snapshot()["percent"] == 100


def test_a_new_run_resets_the_previous_runs_state():
    p = IngestionProgress()
    p.start(2)
    p.file_done("x", "failed")
    p.finish({"old": True}, error="Boom")
    p.start(3)
    snap = p.snapshot()
    assert snap["done"] == 0 and snap["failed"] == [] and snap["result"] is None
    assert snap["error"] is None and snap["total"] == 3


def test_failure_list_is_capped():
    p = IngestionProgress()
    p.start(100)
    for i in range(100):
        p.file_done(f"f{i}", "failed")
    assert len(p.snapshot()["failed"]) == 20


def test_eta_only_while_running_and_after_some_progress():
    p = IngestionProgress()
    p.start(10)
    assert p.snapshot()["eta_seconds"] is None
    p.file_done("a", "new")
    time.sleep(0.05)
    p.file_done("b", "new")
    assert p.snapshot()["eta_seconds"] is not None
    p.finish({})
    assert p.snapshot()["eta_seconds"] is None


def test_directory_ingest_reports_progress_and_final_result():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        write_file(os.path.join(tmp, "documents", "Bio", "a.txt"), "The mitochondria is the powerhouse of the cell.")
        write_file(os.path.join(tmp, "documents", "Bio", "b.txt"), "Photosynthesis converts light into chemical energy.")
        seen = []
        orig = engine.rag.ingest_file

        def spy(fi, rebuild_bm25=True):
            snap = engine.ingest_status()
            seen.append((snap["running"], snap["current"], snap["done"]))
            return orig(fi, rebuild_bm25=rebuild_bm25)

        engine.rag.ingest_file = spy
        engine._ingest()
        assert [s[0] for s in seen] == [True, True]
        assert {s[1] for s in seen} == {"a.txt", "b.txt"}
        assert [s[2] for s in seen] == [0, 1]
        final = engine.ingest_status()
        assert final["running"] is False and final["done"] == 2 and final["percent"] == 100
        assert final["result"]["new_files"] == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ingest_status_before_any_run_is_idle():
    tmp = tempfile.mkdtemp()
    try:
        assert make_engine(tmp).ingest_status()["running"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_crash_mid_run_marks_the_run_finished_with_an_error():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        write_file(os.path.join(tmp, "documents", "Bio", "a.txt"), "Some reasonably long text for chunking here.")

        def boom(fi, rebuild_bm25=True):
            raise RuntimeError("disk gone")

        engine.rag.ingest_file = boom
        with pytest.raises(RuntimeError):
            engine._ingest()
        snap = engine.ingest_status()
        assert snap["running"] is False and snap["error"] == "RuntimeError"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_background_ingest_runs_and_second_start_is_refused_while_running():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        write_file(os.path.join(tmp, "documents", "Bio", "a.txt"), "The mitochondria is the powerhouse of the cell.")
        gate, entered = threading.Event(), threading.Event()
        orig = engine.rag.ingest_file

        def slow(fi, rebuild_bm25=True):
            entered.set()
            gate.wait(5)
            return orig(fi, rebuild_bm25=rebuild_bm25)

        engine.rag.ingest_file = slow
        assert engine.start_ingest_background() is True
        assert entered.wait(5)
        assert engine.ingest_status()["running"] is True
        assert engine.start_ingest_background() is False
        gate.set()
        _wait_idle(engine)
        assert engine.ingest_status()["result"]["new_files"] == 1
        assert engine.start_ingest_background() is True
        _wait_idle(engine)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_status_reads_running_immediately_after_start_with_no_idle_gap():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        write_file(os.path.join(tmp, "documents", "Bio", "a.txt"), "The mitochondria is the powerhouse of the cell.")
        gate = threading.Event()
        orig = engine.rag.ingest_file
        engine.rag.ingest_file = lambda fi, rebuild_bm25=True: (gate.wait(5), orig(fi, rebuild_bm25=rebuild_bm25))[1]
        assert engine.start_ingest_background() is True
        assert engine.ingest_status()["running"] is True
        gate.set()
        _wait_idle(engine)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_background_failure_is_reported_not_left_running_forever():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        engine._ingest = lambda data_dir=None: (_ for _ in ()).throw(RuntimeError("boom"))
        assert engine.start_ingest_background() is True
        _wait_idle(engine)
        snap = engine.ingest_status()
        assert snap["running"] is False and snap["error"] == "RuntimeError"
        assert engine.start_ingest_background() is True
        _wait_idle(engine)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_web_endpoints_start_and_report_ingestion():
    pytest.importorskip("flask")
    from web_ui import HestiaWebUI

    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        write_file(os.path.join(tmp, "documents", "Bio", "a.txt"), "The mitochondria is the powerhouse of the cell.")
        c = HestiaWebUI(memory=None, athena=engine).app.test_client()

        assert c.get("/api/athena/ingest-status").get_json()["running"] is False
        r = c.post("/api/athena/ingest")
        assert r.status_code == 202 and r.get_json() == {"started": True}
        deadline = time.time() + 5
        while c.get("/api/athena/ingest-status").get_json()["running"] and time.time() < deadline:
            time.sleep(0.02)
        snap = c.get("/api/athena/ingest-status").get_json()
        assert snap["percent"] == 100 and snap["result"]["new_files"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_web_ingest_is_503_when_athena_is_disabled():
    pytest.importorskip("flask")
    from web_ui import HestiaWebUI

    c = HestiaWebUI(memory=None, athena=None).app.test_client()
    assert c.post("/api/athena/ingest").status_code == 503
    assert c.get("/api/athena/ingest-status").get_json()["running"] is False


# ===========================================================================
# #68 OCR language
# ===========================================================================

_FRENCH = ("Le chat dort sur le canapé pendant que la pluie tombe doucement sur "
           "les toits de la ville. Nous avons décidé de rester à la maison ce soir.")
_ENGLISH = ("The cat sleeps on the sofa while the rain falls gently on the roofs "
            "of the city. We decided to stay at home this evening.")


def test_detect_language_identifies_french_and_english():
    pytest.importorskip("langdetect")
    assert ocr.detect_language(_FRENCH) == "fr"
    assert ocr.detect_language(_ENGLISH) == "en"


def test_detect_language_refuses_to_guess_from_short_or_empty_text():
    assert ocr.detect_language("bonjour") is None
    assert ocr.detect_language("") is None
    assert ocr.detect_language(None) is None


def test_choose_language_keeps_english_alongside_an_installed_pack():
    assert ocr.choose_ocr_language("fr", {"eng", "fra"}) == "fra+eng"
    assert ocr.choose_ocr_language("zh-cn", {"eng", "chi_sim"}) == "chi_sim+eng"


def test_choose_language_falls_back_to_english_when_pack_missing_or_unknown():
    assert ocr.choose_ocr_language("fr", {"eng"}) == "eng"
    assert ocr.choose_ocr_language("xx", {"eng", "fra"}) == "eng"
    assert ocr.choose_ocr_language(None, {"eng", "fra"}) == "eng"
    assert ocr.choose_ocr_language("en", {"eng", "fra"}) == "eng"


def test_probe_string_is_english_only_when_nothing_else_is_installed():
    assert ocr.probe_language_string({"eng"}) == "eng"
    assert ocr.probe_language_string({"eng", "fra", "deu"}) == "eng+fra+deu"


def test_probe_string_is_capped_so_the_probe_pass_stays_fast():
    everything = {"eng", "fra", "deu", "spa", "ita", "por", "rus", "hin"}
    assert len(ocr.probe_language_string(everything).split("+")) == 4


def test_pipeline_detects_a_french_scan_and_uses_the_french_pack(monkeypatch):
    # langdetect may not be installed: inject the detector so the pipeline logic is tested regardless.
    monkeypatch.setattr(ocr, "detect_language", lambda text: "fr")
    calls = []

    def probe(img, lang):
        calls.append(lang)
        return _FRENCH

    assert ocr.auto_ocr_language(probe, object(), installed={"eng", "fra"}) == "fra+eng"
    assert calls == ["eng+fra"]


def test_pipeline_skips_the_slow_probe_when_only_english_is_installed():
    def probe(img, lang):
        raise AssertionError("must not run")

    assert ocr.auto_ocr_language(probe, object(), installed={"eng"}) == "eng"


def test_pipeline_never_raises_and_defaults_to_english():
    def probe(img, lang):
        raise RuntimeError("tesseract crashed")

    assert ocr.auto_ocr_language(probe, object(), installed={"eng", "fra"}) == "eng"
    assert ocr.auto_ocr_language(lambda i, l: "xx", object(), installed={"eng", "fra"}) == "eng"


def test_installed_languages_excludes_osd_and_survives_tesseract_errors(monkeypatch):
    import pytesseract
    monkeypatch.setattr(pytesseract, "get_languages", lambda config="": ["eng", "osd", "fra"])
    assert ocr.installed_languages() == {"eng", "fra"}

    def boom(config=""):
        raise RuntimeError("no tesseract")

    monkeypatch.setattr(pytesseract, "get_languages", boom)
    assert ocr.installed_languages() == {"eng"}


@pytest.fixture
def processor():
    from modules.athena.pdf_processor import PDFProcessor
    return PDFProcessor()


def test_tesseract_ocr_uses_the_documents_language(processor, monkeypatch):
    used = []
    monkeypatch.setattr(processor, "_tesseract_text", lambda img, lang: (used.append(lang), "texte")[1])
    processor._ocr_lang = "fra+eng"
    assert processor._tesseract_ocr(object()) == "texte"
    assert used == ["fra+eng"]


def test_tesseract_ocr_defaults_to_english_before_a_language_is_chosen(processor, monkeypatch):
    used = []
    monkeypatch.setattr(processor, "_tesseract_text", lambda img, lang: (used.append(lang), "text")[1])
    if hasattr(processor, "_ocr_lang"):
        del processor._ocr_lang
    processor._tesseract_ocr(object())
    assert used == ["eng"]


def test_tesseract_failure_still_returns_empty_text(processor, monkeypatch):
    def boom(img, lang):
        raise RuntimeError("x")

    monkeypatch.setattr(processor, "_tesseract_text", boom)
    assert processor._tesseract_ocr(object()) == ""


def test_auto_language_can_be_switched_off_in_config(processor, monkeypatch):
    from modules.athena.config import get_config
    monkeypatch.setattr(get_config(), "ocr_auto_language", False, raising=False)
    monkeypatch.setattr(ocr, "auto_ocr_language", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")))
    assert processor._pick_ocr_language(object()) == "eng"


def test_pick_language_delegates_to_the_pipeline(processor, monkeypatch):
    monkeypatch.setattr(ocr, "auto_ocr_language", lambda probe, img, installed=None: "fra+eng")
    assert processor._pick_ocr_language(object()) == "fra+eng"
