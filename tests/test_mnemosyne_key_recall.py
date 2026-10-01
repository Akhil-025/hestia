"""
Facts are embedded as "<key words>: <value>" so recall by KEY works, while the
stored document stays the bare value and the #42 dedupe still compares values.

These tests check the mechanism (what gets embedded, how dedupe scores), not
embedding quality, so they behave the same under the stub embedders.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_mnemosyne import make_engine  # noqa: E402


class _RecordingStore:
    """add() accepts embed_text, like the real MnemosyneVectorStore."""

    def __init__(self):
        self.calls = []

    def add(self, text, metadata, doc_id, embed_text=None):
        self.calls.append({"text": text, "doc_id": doc_id, "embed_text": embed_text})

    def search(self, *a, **k):
        return []


def test_learn_embeds_key_and_value_but_stores_bare_value(tmp_path):
    eng, _ = make_engine(str(tmp_path))
    eng.vector_store = _RecordingStore()
    eng.learn("favourite_color", "teal")
    assert eng.vector_store.calls == [
        {"text": "teal", "doc_id": "favourite_color", "embed_text": "favourite color: teal"}
    ]   # document stays the bare value, so _format_result still says "Your favourite color is teal."


def test_store_without_embed_text_param_still_gets_the_fact(tmp_path):
    eng, _ = make_engine(str(tmp_path))
    calls = []

    class OldStyleStore:                              # add() has no embed_text parameter
        def add(self, text, metadata, doc_id):
            calls.append((text, doc_id))

        def search(self, *a, **k):
            return []

    eng.vector_store = OldStyleStore()
    eng.learn("favourite_color", "teal")
    assert calls == [("teal", "favourite_color")]


def _store_with(similarity_value):
    class Store:
        def search(self, query, n_results=5, where=None):
            # Low search score on purpose: a "key: value" embedding scores lower
            # than a bare-value one against a bare-value query.
            return [{"text": "teal", "score": 0.3, "id": "favourite_color",
                     "metadata": {"key": "favourite_color", "type": "fact"}}]

        def similarity(self, a, b):
            return similarity_value

        def add(self, text, metadata, doc_id):
            pass

    return Store()


def test_dedupe_scores_value_against_value_not_the_mixed_search_score(tmp_path):
    eng, _ = make_engine(str(tmp_path))
    eng.vector_store = _store_with(0.99)
    assert eng._find_duplicate_value("preferred_colour", "teal") == "favourite_color"


def test_dedupe_rejects_when_values_differ(tmp_path):
    eng, _ = make_engine(str(tmp_path))
    eng.vector_store = _store_with(0.4)
    assert eng._find_duplicate_value("preferred_colour", "crimson") is None


def test_dedupe_never_matches_its_own_key(tmp_path):
    eng, _ = make_engine(str(tmp_path))
    eng.vector_store = _store_with(1.0)
    assert eng._find_duplicate_value("favourite_color", "teal") is None


class _FakeCollection:
    """Just enough of a Chroma collection for reembed_facts; independent of the chromadb stub."""

    def __init__(self):
        self.upserts = []

    def get(self, where=None, include=None):
        return {
            "ids": ["favourite_color"],
            "documents": ["teal"],
            "metadatas": [{"type": "fact", "key": "favourite_color", "created_at": "2026-01-01T00:00:00+00:00"}],
        }

    def upsert(self, ids, documents, metadatas, embeddings):
        self.upserts.append((ids, documents, metadatas, embeddings))


def test_reindex_upgrades_legacy_value_only_facts(tmp_path):
    import threading
    from modules.mnemosyne.vector_store import MnemosyneVectorStore

    vs = object.__new__(MnemosyneVectorStore)        # skip chroma/model init
    vs._lock = threading.Lock()
    vs.collection = _FakeCollection()
    embedded = []
    vs._embed = lambda texts: (embedded.append(list(texts)) or [[0.0, 1.0] for _ in texts])

    eng, _ = make_engine(str(tmp_path))
    eng.vector_store = vs
    assert eng.reindex_facts() == 1
    assert embedded == [["favourite color: teal"]]
    ids, docs, metas, embs = vs.collection.upserts[0]
    assert docs == ["teal"]                           # stored document untouched
    assert metas[0]["created_at"] == "2026-01-01T00:00:00+00:00"   # metadata untouched


def test_reindex_is_a_noop_without_support(tmp_path):
    eng, _ = make_engine(str(tmp_path))
    eng.vector_store = object()
    assert eng.reindex_facts() == 0
