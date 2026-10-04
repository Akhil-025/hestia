# tests/test_voice_latency.py
"""
Latency regression budget for the part of the voice round trip that is
Hestia's own code (backlog #211).

The models (speech-to-text, the LLM behind the NLU, speech synthesis) cost
hundreds of milliseconds to seconds and vary by machine; they are measured on
a real install by scripts/voice_latency.py against config/latency_budget.json.
What can be pinned on every test run is everything else: routing, validation,
context bookkeeping, locking, sentence splitting. With the model replaced by an
instant fake that is pure overhead, and if it grows (an accidental O(n^2)
loop, a lock held across a slow call, a regex that backtracks) it adds to every
single spoken turn.

Budgets are deliberately loose, 10-50x what this takes on a laptop, so a
loaded CI machine doesn't flake them while a real regression (which is
usually a 100x jump) still fails. Tighten them if they turn out too slack.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import core.nlu as nlu_module
from core.tts import split_speakable
from scripts.voice_latency import percentile
from test_pipeline_integration import Pipeline, ScriptedModel

TURN_P95_BUDGET_MS = 60.0          # NLU + Hecate + dispatch with an instant model
LATE_VS_EARLY_FACTOR = 4.0         # the 200th turn may not be this much slower than the 1st
LONG_REPLY_SPLIT_BUDGET_MS = 50.0  # sentence splitting a 20,000-character reply
CONCURRENT_TURNS_BUDGET_S = 10.0   # 8 threads x 25 turns


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(nlu_module.time, "sleep", lambda *_: None)


def _pipeline():
    p = Pipeline(ScriptedModel.intent("read_email"))
    return p


def _timed_turns(p, n, prefix="any new mail"):
    out = []
    for i in range(n):
        start = time.perf_counter()
        p.ask(f"{prefix} {i}")             # distinct text, so the NLU cache can't help
        out.append((time.perf_counter() - start) * 1000.0)
    return out


def test_turn_overhead_is_inside_the_budget():
    p = _pipeline()
    _timed_turns(p, 5)                      # warm-up
    samples = _timed_turns(p, 200)
    assert percentile(samples, 95) < TURN_P95_BUDGET_MS, (
        f"p95 {percentile(samples, 95):.1f} ms; median {percentile(samples, 50):.2f} ms"
    )


def test_turns_do_not_get_slower_as_a_session_goes_on():
    p = _pipeline()
    _timed_turns(p, 5)
    early = percentile(_timed_turns(p, 50, "first batch"), 50)
    _timed_turns(p, 500, "middle")
    late = percentile(_timed_turns(p, 50, "last batch"), 50)
    # a small absolute allowance so a sub-millisecond median can't flake the ratio
    assert late <= early * LATE_VS_EARLY_FACTOR + 2.0, f"early {early:.2f} ms, late {late:.2f} ms"


def test_the_context_window_stays_bounded():
    p = _pipeline()
    _timed_turns(p, 400)
    assert len(p.orch._ctx.recent_intents) < 50


def test_concurrent_turns_all_complete_without_errors():
    p = _pipeline()
    errors, replies = [], []

    def worker(k):
        for i in range(25):
            try:
                replies.append(p.orch.dispatch(f"mail {k}-{i}", {"intent": "read_email", "entities": {}, "confidence": 0.95}))
            except Exception as exc:                       # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=CONCURRENT_TURNS_BUDGET_S)
    elapsed = time.perf_counter() - start
    assert not any(t.is_alive() for t in threads), "a turn is stuck: likely a lock held too long"
    assert errors == [] and len(replies) == 200 and set(replies) == {"You have 2 emails."}
    assert elapsed < CONCURRENT_TURNS_BUDGET_S


def test_a_slow_module_does_not_stop_other_modules_answering():
    """A call that takes a while must not hold the orchestrator's lock: while a
    slow handler runs, a fast turn on another thread still completes."""
    p = _pipeline()
    gate = threading.Event()
    original = p.chronos.handle

    def slow(intent, entities, context):
        gate.wait(timeout=5)
        return original(intent, entities, context)
    p.chronos.handle = slow

    slow_thread = threading.Thread(
        target=lambda: p.orch.dispatch("what time", {"intent": "get_time", "entities": {}, "confidence": 0.95}))
    slow_thread.start()
    time.sleep(0.05)
    start = time.perf_counter()
    reply = p.orch.dispatch("mail", {"intent": "read_email", "entities": {}, "confidence": 0.95})
    fast_ms = (time.perf_counter() - start) * 1000.0
    gate.set()
    slow_thread.join(timeout=5)
    assert reply == "You have 2 emails." and fast_ms < 1000.0


def test_splitting_a_very_long_reply_into_speakable_chunks_is_fast():
    text = ("This is a sentence that goes on for a little while, Dr. Smith said. " * 300)
    start = time.perf_counter()
    chunks, rest = split_speakable(text)
    ms = (time.perf_counter() - start) * 1000.0
    assert chunks and ms < LONG_REPLY_SPLIT_BUDGET_MS, f"{ms:.1f} ms"


def test_splitting_pathological_input_does_not_backtrack():
    for text in ("." * 20000, "a" * 20000, ". " * 10000, "Dr. " * 5000, "!?" * 10000):
        start = time.perf_counter()
        split_speakable(text)
        assert (time.perf_counter() - start) * 1000.0 < 500.0, repr(text[:10])


def test_boundary_regex_matches_exactly_what_the_old_quadratic_one_did():
    """The lookbehind in core/tts.py is a speed fix only; prove it changed no result."""
    import re
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import given, settings, strategies as st
    from core.tts import _BOUNDARY_RE

    old = re.compile(r"""([.!?]+["')\]]*)(\s+)|(\n+)""")
    alphabet = st.sampled_from(list("ab Z.!?\"')]\n\t,;:-") + ["Dr.", "e.g.", "...", "?!"])

    @settings(max_examples=500, deadline=None)
    @given(st.lists(alphabet, max_size=40).map("".join))
    def check(text):
        assert [m.span() for m in _BOUNDARY_RE.finditer(text)] == [m.span() for m in old.finditer(text)]
        assert [m.groups() for m in _BOUNDARY_RE.finditer(text)] == [m.groups() for m in old.finditer(text)]

    check()
