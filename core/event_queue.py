"""
core/event_queue.py

A small durable local queue that lets Hestia's processes talk to each other
(backlog #20, and the cross-process half of #10).

Why SQLite and not Redis/ZeroMQ
-------------------------------
The goal is "a local queue, without going all the way to Docker microservices".
SQLite in WAL mode is already a dependency, survives a process crash (a message
sent while the receiver restarts is still there when it comes back), needs no
server, and is plenty for the volume of one person talking to one assistant.
It is polled (default every 100 ms), so latency is roughly that. It is not a
high-throughput broker and is not meant to be.

Model
-----
* ``publish(topic, payload)`` appends to a log table.
* ``consume(consumer, topics)`` returns events newer than that consumer's saved
  cursor, so each named consumer sees each event once, and a consumer that was
  down simply catches up. Different consumers each get their own copy.
* ``QueueRPC`` builds request/reply on top: ``call()`` publishes ``rpc.<name>``
  with a correlation id and waits for ``rpc.reply.<id>``.
* ``EventBridge`` mirrors chosen topics between this queue and a process's
  in-memory ``EventBus``. Events that arrive from the queue are tagged so they
  are never exported again (no echo loops).
* Old rows are purged (default after 1 hour).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data") / "ipc" / "events.db"
RETENTION_SECONDS = 3600.0
POLL_INTERVAL = 0.1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    topic   TEXT NOT NULL,
    origin  TEXT NOT NULL,
    ts      REAL NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic, id);
CREATE TABLE IF NOT EXISTS cursors (
    consumer TEXT PRIMARY KEY,
    last_id  INTEGER NOT NULL
);
"""


class EventQueue:
    """The shared log. One instance per process; the file is shared."""

    def __init__(self, path: "str | Path" = DEFAULT_PATH, origin: Optional[str] = None,
                 retention_seconds: float = RETENTION_SECONDS) -> None:
        self.path = Path(path)
        self.origin = origin or f"proc-{os.getpid()}"
        self.retention = float(retention_seconds)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._last_purge = 0.0
        with self._conn() as c:
            c.executescript(_SCHEMA)

    # -- connections (sqlite connections are per-thread) -------------------

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    # -- publish / consume --------------------------------------------------

    def publish(self, topic: str, payload: Any = None, origin: Optional[str] = None) -> int:
        body = json.dumps(payload if payload is not None else {}, ensure_ascii=False, default=str)
        cur = self._conn().execute(
            "INSERT INTO events (topic, origin, ts, payload) VALUES (?,?,?,?)",
            (topic, origin or self.origin, time.time(), body))
        self._maybe_purge()
        return int(cur.lastrowid)

    def latest_id(self) -> int:
        row = self._conn().execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()
        return int(row[0])

    def start_from_now(self, consumer: str) -> None:
        """Create the consumer's cursor at the current end if it has none, so a
        brand-new consumer doesn't replay history."""
        self._conn().execute(
            "INSERT OR IGNORE INTO cursors (consumer, last_id) VALUES (?,?)",
            (consumer, self.latest_id()))

    def start_from_beginning(self, consumer: str) -> None:
        """Create the consumer's cursor at the start of the log if it has none."""
        self._conn().execute(
            "INSERT OR IGNORE INTO cursors (consumer, last_id) VALUES (?,0)", (consumer,))

    def consume(self, consumer: str, topics: Optional[Iterable[str]] = None,
                limit: int = 100) -> list[dict]:
        """New events for *consumer* (optionally only *topics*, where a trailing
        ``*`` is a prefix match). The cursor advances past everything scanned,
        including events filtered out, so they aren't re-read."""
        conn = self._conn()
        row = conn.execute("SELECT last_id FROM cursors WHERE consumer=?", (consumer,)).fetchone()
        if row is None:
            self.start_from_now(consumer)
            row = conn.execute("SELECT last_id FROM cursors WHERE consumer=?", (consumer,)).fetchone()
        last = int(row[0])
        rows = conn.execute(
            "SELECT id, topic, origin, ts, payload FROM events WHERE id>? ORDER BY id LIMIT ?",
            (last, limit)).fetchall()
        if not rows:
            return []
        patterns = list(topics) if topics is not None else None
        out = []
        for id_, topic, origin, ts, payload in rows:
            if patterns is None or _topic_in(topic, patterns):
                try:
                    data = json.loads(payload)
                except ValueError:
                    data = {}
                out.append({"id": id_, "topic": topic, "origin": origin, "ts": ts, "payload": data})
        conn.execute("UPDATE cursors SET last_id=? WHERE consumer=?", (int(rows[-1][0]), consumer))
        return out

    def _maybe_purge(self) -> None:
        now = time.time()
        if now - self._last_purge < 60:
            return
        self._last_purge = now
        try:
            self._conn().execute("DELETE FROM events WHERE ts < ?", (now - self.retention,))
        except sqlite3.Error:
            logger.debug("Queue purge failed.", exc_info=True)

    def stats(self) -> dict:
        conn = self._conn()
        n = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        cursors = {c: i for c, i in conn.execute("SELECT consumer, last_id FROM cursors")}
        return {"events": n, "latest_id": self.latest_id(), "consumers": cursors,
                "path": str(self.path)}


def _topic_in(topic: str, patterns: list) -> bool:
    for p in patterns:
        if p == topic or (p.endswith("*") and topic.startswith(p[:-1])):
            return True
    return False


# ---------------------------------------------------------------------------
# Polling worker
# ---------------------------------------------------------------------------

class QueueWorker:
    """Polls the queue on a daemon thread and hands each event to a callback."""

    def __init__(self, queue: EventQueue, consumer: str, topics: Optional[Iterable[str]],
                 callback: Callable[[dict], None], interval: float = POLL_INTERVAL,
                 from_beginning: bool = False) -> None:
        self.queue, self.consumer = queue, consumer
        self.topics = list(topics) if topics is not None else None
        self.callback, self.interval = callback, interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if from_beginning:
            queue.start_from_beginning(consumer)
        else:
            queue.start_from_now(consumer)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"Queue[{self.consumer}]")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    def poll_once(self) -> int:
        events = self.queue.consume(self.consumer, self.topics)
        for ev in events:
            try:
                self.callback(ev)
            except Exception:
                logger.exception("Queue callback for %s failed on %s.", self.consumer, ev.get("topic"))
        return len(events)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                busy = self.poll_once()
            except Exception:
                logger.exception("Queue poll failed.")
                busy = 0
            if not busy:
                self._stop.wait(self.interval)


# ---------------------------------------------------------------------------
# Request / reply
# ---------------------------------------------------------------------------

class RPCTimeout(Exception):
    pass


class QueueRPC:
    """Request/reply over the queue.

    ``serve(name, handler)`` (core side) answers requests; ``call(name, payload)``
    (voice side) sends one and waits for the answer. Requests are consumed by a
    single named server, so each is answered once.
    """

    def __init__(self, queue: EventQueue, consumer: str, interval: float = POLL_INTERVAL) -> None:
        self.queue, self.consumer, self.interval = queue, consumer, interval
        self._worker: Optional[QueueWorker] = None

    def serve(self, name: str, handler: Callable[[dict], Any]) -> QueueWorker:
        topic = f"rpc.{name}"

        def _on(ev: dict) -> None:
            payload = ev["payload"] or {}
            cid = payload.get("cid")
            try:
                result = handler(payload.get("body") or {})
                reply = {"ok": True, "result": result}
            except Exception as exc:
                logger.exception("RPC handler %s failed.", name)
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if cid:
                self.queue.publish(f"rpc.reply.{cid}", reply)

        # Pick up requests sent while we were down (a restarting core still answers).
        self._worker = QueueWorker(self.queue, self.consumer, [topic], _on,
                                   self.interval, from_beginning=True)
        return self._worker

    def call(self, name: str, body: Optional[dict] = None, timeout: float = 120.0) -> Any:
        cid = uuid.uuid4().hex
        reply_topic = f"rpc.reply.{cid}"
        consumer = f"{self.consumer}.reply.{cid}"
        self.queue.start_from_now(consumer)
        self.queue.publish(f"rpc.{name}", {"cid": cid, "body": body or {}})
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                for ev in self.queue.consume(consumer, [reply_topic]):
                    rep = ev["payload"]
                    if rep.get("ok"):
                        return rep.get("result")
                    raise RuntimeError(rep.get("error", "remote call failed"))
                time.sleep(self.interval)
        finally:
            try:
                self.queue._conn().execute("DELETE FROM cursors WHERE consumer=?", (consumer,))
            except sqlite3.Error:
                pass
        raise RPCTimeout(f"no reply to {name!r} within {timeout:g}s (is the core process running?)")


# ---------------------------------------------------------------------------
# Bus <-> queue bridge
# ---------------------------------------------------------------------------

class EventBridge:
    """Mirrors selected topics between an in-memory ``EventBus`` and the queue.

    ``export``: bus topics sent out to the queue. ``imports``: queue topics
    re-emitted onto the local bus. A re-emitted event carries ``_remote=True``
    in its payload and is never exported again, so exporting and importing the
    same topic in two processes cannot echo forever.
    """

    def __init__(self, bus, queue: EventQueue, consumer: str,
                 export: Iterable[str] = (), imports: Iterable[str] = (),
                 interval: float = POLL_INTERVAL) -> None:
        self.bus, self.queue = bus, queue
        self.export = set(export)
        self.imports = list(imports)
        self._callbacks: list[tuple[str, Callable]] = []
        for topic in self.export:
            cb = self._make_exporter(topic)
            self.bus.on(topic, cb)
            self._callbacks.append((topic, cb))
        self._worker = QueueWorker(queue, consumer, self.imports, self._on_event, interval) \
            if self.imports else None

    def _make_exporter(self, topic: str) -> Callable:
        def _export(data: Any) -> None:
            if isinstance(data, dict) and data.get("_remote"):
                return
            try:
                self.queue.publish(topic, data if isinstance(data, dict) else {"value": data})
            except Exception:
                logger.exception("Could not export %s.", topic)
        _export.__qualname__ = f"bridge.export[{topic}]"
        return _export

    def _on_event(self, ev: dict) -> None:
        if ev["origin"] == self.queue.origin:
            return
        payload = dict(ev["payload"]) if isinstance(ev["payload"], dict) else {"value": ev["payload"]}
        payload["_remote"] = True
        payload.setdefault("_origin", ev["origin"])
        self.bus.emit(ev["topic"], payload)

    def start(self) -> None:
        if self._worker:
            self._worker.start()

    def stop(self) -> None:
        if self._worker:
            self._worker.stop()
        for topic, cb in self._callbacks:
            try:
                self.bus.off(topic, cb)
            except Exception:
                pass
        self._callbacks.clear()
