# tests/test_mnemosyne.py
"""
Regression tests for modules/mnemosyne/engine.py and summariser.py.

Run with:  python3 tests/test_mnemosyne.py
(or:       python -m pytest tests/test_mnemosyne.py -v)

These cover two real bugs found and fixed in Mnemosyne:

1. engine.py's `_handle_get_user_info` / `_handle_learn_fact` /
   `_handle_forget_fact` read entity values with
   `entities.get("key", "")`. That default only covers a *missing* key —
   if the NLU emits the key explicitly as None (present but empty,
   observed elsewhere in this codebase for other entities), `.get()`
   returns None rather than the default, and `.strip()` on it raised
   AttributeError. handle()'s top-level try/except caught it and
   returned a generic "Something went wrong" instead of the graceful
   fallback (e.g. falling through to semantic recall) the rest of the
   pipeline already relies on. Fixed with `(entities.get("key") or "")`,
   matching the pattern modules/hestia/core_module.py already uses for
   the exact same entity.

2. summariser.py called `self.hestia_llm.generate(prompt)` without
   `fmt="json"`, even though it immediately does `json.loads()` on the
   result and the prompt only *asks* for pure JSON in free text. Every
   other module in this codebase that needs structured LLM output
   (Ares, Orpheus, Dionysus, core/nlu.py) passes `fmt="json"` explicitly
   — core/llm.py's HestiaLLM.generate() docstring says as much. Without
   it, a real local model (e.g. mistral via Ollama) commonly wraps its
   answer in prose ("Sure! Here's the summary: {...}"), json.loads()
   raises, and the summariser silently and permanently backs off
   (10-50 interaction cooldown) — so periodic memory summarisation
   effectively never succeeds in practice.

These tests use lightweight stand-ins for the heavy ML dependencies
(chromadb / sentence-transformers / torch) so they can run without
those packages installed, while still exercising the real
MnemosyneEngine/Summariser code paths end to end.
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
            "    def upsert(self, ids, documents, metadatas, embeddings):\n"
            "        for i, d, m, e in zip(ids, documents, metadatas, embeddings):\n"
            "            if i in self._ids:\n"
            "                idx = self._ids.index(i)\n"
            "                self._docs[idx] = d; self._metas[idx] = m; self._embs[idx] = e\n"
            "            else:\n"
            "                self._ids.append(i); self._docs.append(d)\n"
            "                self._metas.append(m); self._embs.append(e)\n"
            "    def delete(self, ids):\n"
            "        for i in ids:\n"
            "            if i in self._ids:\n"
            "                idx = self._ids.index(i)\n"
            "                del self._ids[idx]; del self._docs[idx]\n"
            "                del self._metas[idx]; del self._embs[idx]\n"
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
            "                'distances': [[dist(i) for i in idxs]],\n"
            "                'ids': [[self._ids[i] for i in idxs]]}\n"
            "def _match_where(meta, where):\n"
            "    if '$and' in where: return all(_match_where(meta, c) for c in where['$and'])\n"
            "    for k, v in where.items():\n"
            "        actual = meta.get(k)\n"
            "        if isinstance(v, dict) and '$eq' in v:\n"
            "            if actual != v['$eq']: return False\n"
            "        elif actual != v: return False\n"
            "    return True\n"
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

from modules.mnemosyne.engine import MnemosyneEngine  # noqa: E402
from modules.mnemosyne.config import MnemosyneConfig, set_config  # noqa: E402


class _FakeLLM:
    """Simulates a real local model (e.g. mistral via Ollama) that only
    returns clean JSON when the caller enforces fmt='json' — otherwise it
    wraps the answer in prose, same as observed in practice."""

    def generate(self, prompt, fmt=None):
        if fmt == "json":
            return '{"summary": "User discussed cooking and travel plans in useful detail.", "topic": "lifestyle"}'
        return 'Sure! Here is a summary: {"summary": "..."}'


def make_engine(tmp_dir, summarise_every_n=5):
    cfg = MnemosyneConfig(
        db_path=os.path.join(tmp_dir, "m.db"),
        chroma_dir=os.path.join(tmp_dir, "chroma_db"),
        summarise_every_n=summarise_every_n,
    )
    set_config(cfg)
    return MnemosyneEngine(_FakeLLM()), cfg


# ---------------------------------------------------------------------------
# Bug 1: entities present-but-None must not crash handle().
# ---------------------------------------------------------------------------

def test_get_user_info_with_none_key_falls_through_gracefully():
    tmp = tempfile.mkdtemp()
    try:
        engine, _ = make_engine(tmp)
        r = engine.handle("get_user_info", {"key": None}, {})
        assert r["confidence"] != 0.0 or "went wrong" not in r["response"]
        assert "went wrong" not in r["response"], f"crashed instead of falling through: {r}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_learn_fact_with_none_key_and_value_does_not_crash():
    tmp = tempfile.mkdtemp()
    try:
        engine, _ = make_engine(tmp)
        r = engine.handle("learn_fact", {"key": None, "value": None}, {})
        assert r["response"] == "What should I remember?"
        assert r["confidence"] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_forget_fact_with_none_key_does_not_crash():
    tmp = tempfile.mkdtemp()
    try:
        engine, _ = make_engine(tmp)
        r = engine.handle("forget_fact", {"key": None}, {})
        assert r["response"] == "Which fact should I forget?"
        assert r["confidence"] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# forget_fact: confirmation gating
#
# Forgetting a fact is irreversible and "forget fact" is exactly the kind
# of short phrase STT gets wrong, so the first call must preview what would
# be forgotten and ask for confirmation instead of deleting it immediately
# (see HestiaOrchestrator's confirmation mechanism in
# modules/hestia/orchestrator.py). Only a call carrying
# entities["_confirmed"] = True actually forgets anything.
# ---------------------------------------------------------------------------

def test_forget_fact_unknown_key_reports_nothing_to_forget_without_asking():
    tmp = tempfile.mkdtemp()
    try:
        engine, _ = make_engine(tmp)
        r = engine.handle("forget_fact", {"key": "favourite_colour"}, {})
        assert r.get("needs_confirmation", False) is False
        assert "don't have anything" in r["response"].lower()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_forget_fact_first_call_asks_for_confirmation_and_forgets_nothing():
    tmp = tempfile.mkdtemp()
    try:
        engine, _ = make_engine(tmp)
        engine.learn("favourite_colour", "teal")

        r = engine.handle("forget_fact", {"key": "favourite_colour"}, {})

        assert r.get("needs_confirmation") is True
        assert "teal" in r["response"]
        assert r["confirm_intent"] == "forget_fact"
        assert r["confirm_entities"] == {"key": "favourite_colour"}
        # Still there — nothing was forgotten yet.
        assert engine.db.get_fact("favourite_colour") == "teal"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_forget_fact_confirmed_call_actually_forgets():
    tmp = tempfile.mkdtemp()
    try:
        engine, _ = make_engine(tmp)
        engine.learn("favourite_colour", "teal")

        preview = engine.handle("forget_fact", {"key": "favourite_colour"}, {})
        assert preview.get("needs_confirmation") is True

        r = engine.handle("forget_fact", {"key": "favourite_colour", "_confirmed": True}, {})
        assert r["response"] == "Forgotten: favourite colour."
        assert engine.db.get_fact("favourite_colour") is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_get_user_info_with_real_key_still_works():
    tmp = tempfile.mkdtemp()
    try:
        engine, _ = make_engine(tmp)
        engine.learn("user_name", "Alex")
        r = engine.handle("get_user_info", {"key": "user_name"}, {})
        assert r["response"] == "Your name is Alex."
        assert r["confidence"] == 0.95
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Bug 2: summarisation must actually succeed against a realistic local LLM.
# ---------------------------------------------------------------------------

def test_summarisation_succeeds_with_realistic_llm_output():
    tmp = tempfile.mkdtemp()
    try:
        engine, cfg = make_engine(tmp, summarise_every_n=5)
        assert engine.summariser is not None

        for i in range(cfg.summarise_every_n):
            engine.push(f"msg {i}", f"resp {i}", "chat")

        assert engine.summariser._failure_count == 0, (
            "summariser backed off — fmt='json' was not enforced on the LLM call"
        )
        status = engine.status()
        assert status["summaries"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_learn_and_recall_end_to_end():
    tmp = tempfile.mkdtemp()
    try:
        engine, _ = make_engine(tmp)
        engine.learn("favourite_color", "teal")
        response = engine.remember("what is my favourite color")
        assert response, "learned fact was not retrievable via semantic recall"
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