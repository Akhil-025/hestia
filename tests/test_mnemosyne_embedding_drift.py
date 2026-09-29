# tests/test_mnemosyne_embedding_drift.py
"""
Tests for embedding drift (backlog #46): the risk that a fact's stored
vector goes stale relative to its current text — e.g. a fact's value is
updated but the old embedding is left in place, so semantic search keeps
matching queries against outdated content, or two different values end up
sharing one embedding because a write silently didn't happen.

These test the INVARIANTS that prevent that, using a local fake vector
store that records every call precisely (real embedding math isn't the
point here — call discipline is).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_mnemosyne import make_engine  # noqa: E402


class _RecordingVectorStore:
    """
    Records every add()/delete() call with its arguments, so a test can
    assert exactly how many times (and with what content) the embedder
    was actually invoked — the thing that matters for drift.
    """

    def __init__(self):
        self.add_calls = []
        self.delete_calls = []
        self._docs = {}  # doc_id -> (text, metadata)

    def add(self, text, metadata, doc_id):
        self.add_calls.append((text, dict(metadata), doc_id))
        self._docs[doc_id] = (text, dict(metadata))

    def delete(self, doc_id):
        self.delete_calls.append(doc_id)
        self._docs.pop(doc_id, None)

    def search(self, query, n_results=5, where=None):
        return []

    def current_text_for(self, doc_id):
        return self._docs.get(doc_id, (None, None))[0]


def make_engine_with_recorder(tmp_path):
    eng, _ = make_engine(str(tmp_path))
    recorder = _RecordingVectorStore()
    eng.vector_store = recorder
    return eng, recorder


# ---------------------------------------------------------------------------
# Updating a fact must re-embed, not reuse the old vector
# ---------------------------------------------------------------------------

def test_updating_a_facts_value_triggers_a_fresh_embed_call(tmp_path):
    engine, recorder = make_engine_with_recorder(tmp_path)
    engine.learn("favorite_color", "blue")
    engine.learn("favorite_color", "green")  # same key, new value
    assert len(recorder.add_calls) == 2
    assert recorder.add_calls[0][0] == "blue"
    assert recorder.add_calls[1][0] == "green"


def test_the_stored_vector_reflects_only_the_latest_value(tmp_path):
    # Chroma's upsert (not a plain insert) is what makes this true in
    # production — the fake here mirrors that same replace-not-append
    # semantics, so this test is really asserting the CALLER (learn())
    # doesn't route an update through some other path that would leave
    # both an old and a new vector alive under the same doc_id.
    engine, recorder = make_engine_with_recorder(tmp_path)
    engine.learn("favorite_color", "blue")
    engine.learn("favorite_color", "green")
    assert recorder.current_text_for("favorite_color") == "green"


def test_updating_a_fact_does_not_leave_a_second_doc_id_behind(tmp_path):
    engine, recorder = make_engine_with_recorder(tmp_path)
    engine.learn("favorite_color", "blue")
    engine.learn("favorite_color", "green")
    # Same doc_id (the key) reused for the update — not a new one.
    doc_ids = [call[2] for call in recorder.add_calls]
    assert doc_ids == ["favorite_color", "favorite_color"]


def test_forgetting_a_fact_removes_its_vector(tmp_path):
    engine, recorder = make_engine_with_recorder(tmp_path)
    engine.learn("favorite_color", "blue")
    engine.forget("favorite_color")
    assert "favorite_color" in recorder.delete_calls
    assert recorder.current_text_for("favorite_color") is None


def test_forgetting_a_fact_that_was_never_embedded_does_not_raise(tmp_path):
    engine, recorder = make_engine_with_recorder(tmp_path)
    engine.forget("never_existed")
    assert "never_existed" in recorder.delete_calls  # attempted, harmlessly


# ---------------------------------------------------------------------------
# The real MnemosyneVectorStore.add() always upserts, never a plain insert
# ---------------------------------------------------------------------------
#
# This is the production-code half of the guarantee the fake above
# assumes: MnemosyneVectorStore.add() must call collection.upsert(), which
# replaces an existing doc_id's embedding, rather than collection.add(),
# which for most vector stores either errors or silently creates a
# duplicate on a repeated id — either of which is a real drift bug.

def test_vector_store_add_calls_upsert_not_insert():
    import inspect
    from modules.mnemosyne.vector_store import MnemosyneVectorStore
    source = inspect.getsource(MnemosyneVectorStore.add)
    assert "upsert" in source
    assert ".add(" not in source.replace("def add(", "")  # the method's own signature line excluded


# ---------------------------------------------------------------------------
# Deduplication (#42) must not silently skip re-embedding a real update
# ---------------------------------------------------------------------------
#
# A subtle interaction: if dedup's "does a similar VALUE already exist"
# check ran on every learn() call (not just genuinely NEW keys), updating
# an existing fact to a value that happens to resemble another fact's
# value could skip writing the update entirely — the exact silent-drift
# failure mode this whole test file exists to catch.

def test_dedup_check_never_applies_to_an_update_of_an_existing_key(tmp_path):
    engine, recorder = make_engine_with_recorder(tmp_path)
    engine.learn("fact_a", "I love hiking")
    engine.learn("fact_b", "I love hiking outdoors")  # different key, similar value

    def fake_search(*args, **kwargs):
        # Simulate a near-duplicate match against fact_a for ANY search —
        # if learn() checked dedup on an update to fact_b, it would
        # (wrongly) skip writing fact_b's real update below.
        return [{"text": "I love hiking", "metadata": {"type": "fact", "key": "fact_a"}, "score": 0.99, "id": "fact_a"}]

    recorder.search = fake_search
    engine.learn("fact_b", "actually I prefer swimming")  # update to an EXISTING key

    assert recorder.current_text_for("fact_b") == "actually I prefer swimming"


def test_dedup_only_intercepts_genuinely_new_keys(tmp_path):
    engine, recorder = make_engine_with_recorder(tmp_path)
    engine.learn("fact_a", "I love hiking")

    def fake_search(*args, **kwargs):
        return [{"text": "I love hiking", "metadata": {"type": "fact", "key": "fact_a"}, "score": 0.99, "id": "fact_a"}]

    recorder.search = fake_search
    result = engine.learn("fact_c", "I love hiking")  # brand-new key, duplicate value

    assert result["deduplicated"] is True
    assert recorder.current_text_for("fact_c") is None  # no vector written for the dup
