# tests/test_athena.py
"""
Regression tests for modules/athena/engine.py.

Run with:  python3 tests/test_athena.py
(or:       python -m pytest tests/test_athena.py -v)

These cover the real bug found and fixed in AthenaEngine:

1. AthenaEngine could only ever *search* — it had no "ingest" intent at
   all (can_handle() rejected it, handle() had no branch for it), and
   Hecate had no text-trigger phrases for it either (unlike Iris, which
   has a full "ingest my photos" -> intent="ingest" pair). The only way
   to populate the RAG index was to call the private, undocumented
   engine._ingest() method directly from a Python shell — nothing a user
   could reach through voice, chat, or the declared BaseModule contract.
   As a result every real search silently returned "No relevant
   information found in your documents.", forever, on a fresh install.

2. Same story for "status": there was no way to ask "how many documents
   have I indexed" and reach AthenaEngine.stats() through handle().

These tests use lightweight stand-ins for the heavy ML dependencies
(chromadb / sentence-transformers / rank_bm25) so they can run without
those packages installed, while still exercising the real
AthenaEngine.handle()/can_handle() code paths end to end.
"""
import sys
import os
import shutil
import tempfile

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)

_STUBS = os.path.join(os.path.dirname(__file__), "_stubs")
sys.path.insert(0, _STUBS)  # only used if the real heavy deps aren't installed


def _ensure_stubs():
    """Install minimal stand-ins for chromadb/sentence_transformers/torch
    if they aren't already installed, so this file can run in a bare
    environment. No-ops if the real packages are present."""
    os.makedirs(_STUBS, exist_ok=True)
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
        import fitz  # noqa: F401
        return
    except ImportError:
        pass

    with open(os.path.join(_STUBS, "torch.py"), "w") as f:
        f.write(
            "class _Cuda:\n"
            "    @staticmethod\n"
            "    def is_available(): return False\n"
            "    @staticmethod\n"
            "    def set_device(i): pass\n"
            "cuda = _Cuda()\n"
        )

    with open(os.path.join(_STUBS, "fitz.py"), "w") as f:
        f.write(
            "def open(*a, **k):\n"
            "    raise RuntimeError('fitz stub - PDF extraction not used in this test')\n"
        )

    st_dir = os.path.join(_STUBS, "sentence_transformers")
    os.makedirs(st_dir, exist_ok=True)
    with open(os.path.join(st_dir, "__init__.py"), "w") as f:
        f.write(
            "import hashlib\n"
            "class SentenceTransformer:\n"
            "    def __init__(self, model_name, device='cpu'):\n"
            "        self.model_name = model_name\n"
            "    def encode(self, batch, show_progress_bar=False, normalize_embeddings=True):\n"
            "        out = []\n"
            "        for text in batch:\n"
            "            h = hashlib.sha256(text.encode()).digest()\n"
            "            vec = [b / 255.0 for b in h[:8]]\n"
            "            norm = sum(v * v for v in vec) ** 0.5 or 1.0\n"
            "            out.append([v / norm for v in vec])\n"
            "        return _FakeArr(out)\n"
            "class _FakeArr(list):\n"
            "    def tolist(self): return list(self)\n"
        )

    ch_dir = os.path.join(_STUBS, "chromadb")
    os.makedirs(ch_dir, exist_ok=True)
    with open(os.path.join(ch_dir, "__init__.py"), "w") as f:
        f.write(
            "class Collection:\n"
            "    def __init__(self, name, metadata=None):\n"
            "        self.name = name; self.metadata = metadata\n"
            "        self._ids = []; self._docs = []; self._metas = []; self._embs = []\n"
            "    def count(self): return len(self._ids)\n"
            "    def add(self, ids, documents, metadatas, embeddings):\n"
            "        for i, d, m, e in zip(ids, documents, metadatas, embeddings):\n"
            "            if i in self._ids: continue\n"
            "            self._ids.append(i); self._docs.append(d)\n"
            "            self._metas.append(m); self._embs.append(e)\n"
            "    def get(self, where=None, include=None, limit=None):\n"
            "        idxs = list(range(len(self._ids)))\n"
            "        if where: idxs = [i for i in idxs if _match_where(self._metas[i], where)]\n"
            "        if limit: idxs = idxs[:limit]\n"
            "        result = {'ids': [self._ids[i] for i in idxs]}\n"
            "        if include:\n"
            "            if 'documents' in include: result['documents'] = [self._docs[i] for i in idxs]\n"
            "            if 'metadatas' in include: result['metadatas'] = [self._metas[i] for i in idxs]\n"
            "        return result\n"
            "    def query(self, query_embeddings, n_results, where=None, include=None):\n"
            "        idxs = list(range(len(self._ids)))\n"
            "        if where: idxs = [i for i in idxs if _match_where(self._metas[i], where)]\n"
            "        qe = query_embeddings[0]\n"
            "        def dist(i):\n"
            "            e = self._embs[i]\n"
            "            return sum((a - b) ** 2 for a, b in zip(qe, e)) ** 0.5\n"
            "        idxs.sort(key=dist); idxs = idxs[:n_results]\n"
            "        return {'documents': [[self._docs[i] for i in idxs]],\n"
            "                'metadatas': [[self._metas[i] for i in idxs]],\n"
            "                'distances': [[dist(i) for i in idxs]]}\n"
            "def _match_where(meta, where):\n"
            "    if '$and' in where: return all(_match_where(meta, c) for c in where['$and'])\n"
            "    return all(meta.get(k) == v for k, v in where.items())\n"
            "class _Client:\n"
            "    def __init__(self, path):\n"
            "        self.path = path; self._collections = {}\n"
            "    def get_or_create_collection(self, name, metadata=None):\n"
            "        if name not in self._collections:\n"
            "            self._collections[name] = Collection(name, metadata)\n"
            "        return self._collections[name]\n"
            "    def delete_collection(self, name): self._collections.pop(name, None)\n"
            "def PersistentClient(path): return _Client(path)\n"
        )


_ensure_stubs()

from modules.athena.engine import AthenaEngine  # noqa: E402


class _FakeLLM:
    def generate(self, prompt, fmt=None):
        return (
            "This is a fake generated answer that references the provided "
            "context about the ingested test document in enough detail."
        )


def make_engine(tmp_dir):
    """Build an AthenaEngine backed by an isolated, empty chroma dir so
    tests don't collide with each other or with real user data."""
    from modules.athena.config import AthenaConfig, set_config
    cfg = AthenaConfig(
        chroma_persist_dir=os.path.join(tmp_dir, "chroma_db"),
        data_dir=os.path.join(tmp_dir, "documents"),
        cache_dir=os.path.join(tmp_dir, "cache"),
    )
    set_config(cfg)
    return AthenaEngine(_FakeLLM())


# ---------------------------------------------------------------------------
# Bug 1: ingest must be a real, dispatchable intent (stripped + prefixed).
# ---------------------------------------------------------------------------

def test_ingest_stripped_and_prefixed_both_produce_a_response():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        for intent in ("ingest", "athena_ingest", "ingest_documents"):
            assert engine.can_handle(intent), f"can_handle() rejected {intent!r}"
            r = engine.handle(intent, {}, {})
            assert r["response"], f"blank response for intent={intent!r}"
            assert r["confidence"] == 1.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ingest_actually_indexes_documents_end_to_end():
    """The ingest intent must make subsequently-searched content findable —
    this is the actual bug: previously there was no way to reach ingestion
    at all, so search always returned nothing."""
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        docs_dir = os.path.join(tmp, "documents", "Biology", "Cells")
        os.makedirs(docs_dir, exist_ok=True)
        with open(os.path.join(docs_dir, "notes.txt"), "w") as f:
            f.write(
                "The mitochondria is the powerhouse of the cell. It "
                "generates ATP used as a source of chemical energy."
            )

        ingest_result = engine.handle("ingest", {}, {})
        assert ingest_result["confidence"] == 1.0
        assert ingest_result["data"].get("total_chunks", 0) > 0

        status_result = engine.handle("status", {}, {})
        assert "1" in status_result["response"] or status_result["data"].get("total_chunks", 0) > 0

        search_result = engine.handle(
            "search", {"query": "what does the mitochondria do"}, {}
        )
        assert search_result["response"]
        assert search_result["data"]["sources"], "ingested document was not found by search"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Bug 2: status must be a real, dispatchable intent.
# ---------------------------------------------------------------------------

def test_status_stripped_and_prefixed_both_produce_a_response():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        for intent in ("status", "athena_status"):
            assert engine.can_handle(intent), f"can_handle() rejected {intent!r}"
            r = engine.handle(intent, {}, {})
            assert r["response"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_search_with_empty_index_reports_low_confidence_not_a_crash():
    tmp = tempfile.mkdtemp()
    try:
        engine = make_engine(tmp)
        r = engine.handle("search", {"query": "anything"}, {})
        assert r["response"]
        assert r["data"]["sources"] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    failures = []
    tests = [
        (name, fn) for name, fn in list(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures.append(name)
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failures.append(name)
            print(f"ERROR {name}: {e!r}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        sys.exit(1)