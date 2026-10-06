# tests/test_event_queue.py
"""Backlog #20 (and #10 across processes): the durable local queue."""
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.event_bus import EventBus
from core.event_queue import EventBridge, EventQueue, QueueRPC, QueueWorker, RPCTimeout


def _q(tmp_path, origin="A", **kw):
    return EventQueue(tmp_path / "ipc" / "e.db", origin=origin, **kw)


def _wait(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_publish_and_consume_in_order(tmp_path):
    q = _q(tmp_path)
    q.start_from_beginning("c")
    for i in range(3):
        q.publish("t", {"i": i})
    assert [e["payload"]["i"] for e in q.consume("c")] == [0, 1, 2]
    assert q.consume("c") == []                    # each event once per consumer


def test_two_consumers_each_get_every_event(tmp_path):
    q = _q(tmp_path)
    q.start_from_beginning("a"); q.start_from_beginning("b")
    q.publish("t", {"x": 1})
    assert len(q.consume("a")) == 1 and len(q.consume("b")) == 1


def test_new_consumer_does_not_replay_history(tmp_path):
    q = _q(tmp_path)
    q.publish("t", {"old": True})
    assert q.consume("fresh") == []
    q.publish("t", {"new": True})
    assert [e["payload"] for e in q.consume("fresh")] == [{"new": True}]


def test_topic_filter_and_prefix_wildcard(tmp_path):
    q = _q(tmp_path)
    q.start_from_beginning("c")
    q.publish("voice.say", {}); q.publish("speak", {}); q.publish("rpc.query", {})
    assert [e["topic"] for e in q.consume("c", ["voice.*", "speak"])] == ["voice.say", "speak"]
    assert q.consume("c") == []                    # filtered-out events aren't re-read


def test_events_survive_a_reopen(tmp_path):
    q = _q(tmp_path)
    q.start_from_beginning("c")
    q.publish("t", {"kept": 1})
    again = _q(tmp_path, origin="B")
    assert again.consume("c")[0]["payload"] == {"kept": 1}
    assert again.consume("c")[0:1] == []


def test_origin_is_recorded(tmp_path):
    a, b = _q(tmp_path, "A"), _q(tmp_path, "B")
    b.start_from_beginning("c")
    a.publish("t", {})
    assert b.consume("c")[0]["origin"] == "A"


def test_old_events_are_purged(tmp_path):
    q = _q(tmp_path, retention_seconds=0.05)
    q.publish("t", {})
    q._last_purge = 0.0
    time.sleep(0.15)
    q.publish("t", {})
    assert q.stats()["events"] == 1                # the first was purged by the second publish


def test_unserialisable_payload_is_stringified_not_fatal(tmp_path):
    q = _q(tmp_path)
    q.start_from_beginning("c")
    q.publish("t", {"obj": object()})
    assert "object" in q.consume("c")[0]["payload"]["obj"]


def test_concurrent_publishers_lose_nothing(tmp_path):
    path = tmp_path / "ipc" / "e.db"
    EventQueue(path, "init").start_from_beginning("c")

    def pub(n):
        q = EventQueue(path, f"p{n}")
        for i in range(25):
            q.publish("t", {"n": n, "i": i})
    threads = [threading.Thread(target=pub, args=(n,)) for n in range(4)]
    [t.start() for t in threads]; [t.join() for t in threads]
    got = EventQueue(path, "r").consume("c", limit=1000)
    assert len(got) == 100


def test_worker_delivers_and_survives_a_callback_error(tmp_path):
    q = _q(tmp_path)
    seen = []

    def cb(ev):
        if ev["payload"].get("bad"):
            raise RuntimeError("callback bug")
        seen.append(ev["payload"])
    w = QueueWorker(q, "w", ["t"], cb, interval=0.02)
    w.start()
    q.publish("t", {"bad": True}); q.publish("t", {"ok": 1})
    assert _wait(lambda: seen == [{"ok": 1}])
    w.stop()


# ---------------------------------------------------------------- RPC

def test_rpc_round_trip(tmp_path):
    server_q, client_q = _q(tmp_path, "core"), _q(tmp_path, "voice")
    w = QueueRPC(server_q, "core.q", interval=0.02).serve("query", lambda b: {"echo": b["text"]})
    w.start()
    out = QueueRPC(client_q, "voice", interval=0.02).call("query", {"text": "hi"}, timeout=5)
    w.stop()
    assert out == {"echo": "hi"}


def test_rpc_handler_error_is_reported_to_the_caller(tmp_path):
    sq, cq = _q(tmp_path, "core"), _q(tmp_path, "voice")

    def boom(_): raise ValueError("bad input")
    w = QueueRPC(sq, "core.q", interval=0.02).serve("query", boom)
    w.start()
    with pytest.raises(RuntimeError, match="bad input"):
        QueueRPC(cq, "voice", interval=0.02).call("query", {}, timeout=5)
    w.stop()


def test_rpc_times_out_when_nobody_answers(tmp_path):
    with pytest.raises(RPCTimeout):
        QueueRPC(_q(tmp_path), "voice", interval=0.02).call("query", {}, timeout=0.3)


def test_request_sent_while_server_was_down_is_answered_after_restart(tmp_path):
    cq = _q(tmp_path, "voice")
    result = {}

    def caller():
        result["v"] = QueueRPC(cq, "voice", interval=0.02).call("query", {"text": "late"}, timeout=5)
    t = threading.Thread(target=caller); t.start()
    time.sleep(0.3)                                # request is sitting in the queue
    w = QueueRPC(_q(tmp_path, "core"), "core.q", interval=0.02).serve("query", lambda b: b["text"])
    w.start()
    t.join(6); w.stop()
    assert result.get("v") == "late"


# ---------------------------------------------------------------- bridge

def test_bridge_carries_an_event_between_two_buses(tmp_path):
    bus_a, bus_b = EventBus(), EventBus()
    got = []
    bus_b.on("speak", got.append)
    ba = EventBridge(bus_a, _q(tmp_path, "jobs"), "bridge.jobs", export=["speak"], interval=0.02)
    bb = EventBridge(bus_b, _q(tmp_path, "core"), "bridge.core", imports=["speak"], interval=0.02)
    bb.start()
    bus_a.emit_sync("speak", {"text": "reminder"})
    assert _wait(lambda: got)
    assert got[0]["text"] == "reminder" and got[0]["_remote"] and got[0]["_origin"] == "jobs"
    ba.stop(); bb.stop()


def test_bridge_never_echoes_imported_events_back(tmp_path):
    bus_a, bus_b = EventBus(), EventBus()
    got_a, got_b = [], []
    bus_a.on("speak", got_a.append); bus_b.on("speak", got_b.append)
    qa, qb = _q(tmp_path, "A"), _q(tmp_path, "B")
    ba = EventBridge(bus_a, qa, "bridge.A", export=["speak"], imports=["speak"], interval=0.02)
    bb = EventBridge(bus_b, qb, "bridge.B", export=["speak"], imports=["speak"], interval=0.02)
    ba.start(); bb.start()
    bus_a.emit_sync("speak", {"text": "once"})
    assert _wait(lambda: got_b)
    time.sleep(0.4)
    assert len(got_b) == 1 and len(got_a) == 1     # no ping-pong
    ba.stop(); bb.stop()


def test_bridge_ignores_topics_not_exported(tmp_path):
    bus = EventBus()
    q = _q(tmp_path)
    q.start_from_beginning("probe")
    br = EventBridge(bus, q, "b", export=["speak"])
    bus.emit_sync("private_thing", {"x": 1})
    assert q.consume("probe") == []
    br.stop()


def test_bridge_stop_unsubscribes(tmp_path):
    bus = EventBus()
    q = _q(tmp_path)
    q.start_from_beginning("probe")
    br = EventBridge(bus, q, "b", export=["speak"])
    br.stop()
    bus.emit_sync("speak", {"text": "x"})
    assert q.consume("probe") == []
