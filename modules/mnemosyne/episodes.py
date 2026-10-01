"""
modules/mnemosyne/episodes.py

Episodic memory clustering (backlog #43): group related interactions into
"episodes" ("everything about the GATE prep", "the Thursday flight booking")
so long-range recall can answer "what were we doing about X" rather than
returning isolated messages.

Algorithm
---------
Online, chronological clustering. Each interaction is compared with the
episodes that are still "open" (touched within ``max_age_hours``) and joins
the most similar one, or starts a new episode:

* similarity is cosine similarity between the interaction and the
  episode's running centroid;
* a message that follows the episode's last message within ``gap_minutes``
  only needs *half* the usual similarity — a follow-up like "and how much
  does that cost?" shares almost no words with its question but is plainly
  the same conversation;
* very short replies ("yes", "thanks") never start an episode: they join
  the episode they answer, or are ignored if there isn't one.

The vectors are pluggable. With an embedding function (sentence-transformers,
as used by the rest of Mnemosyne) similarity is semantic; without one it
falls back to bag-of-words, which still groups by shared vocabulary. A
centroid remembers which kind it is, so switching embedding availability
never compares incompatible vectors — it just starts new episodes.

``assign_interactions`` is a pure function (state in, state out, no clock,
no database). ``EpisodeStore`` persists the open episodes' centroids so each
run continues where the last one stopped instead of re-clustering history.
"""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenising / vectors
# ---------------------------------------------------------------------------

_STOP = frozenset(
    """
    the and for are but not you your yours with that this these those have has
    had was were been being from into onto about what which who whom whose when
    where why how can could would should will shall may might must its it's
    i'm i've i'll don't doesn't didn't isn't aren't wasn't can't won't
    please thanks thank okay yes yeah nope hey hello just also very really
    some any all more most much many there here then than them they their
    tell show give let make get got want need like know think say said
    hestia remind reminder you're welcome sure great cool alright
    """.split()
)
_TOKEN_RE = re.compile(r"[a-z][a-z0-9']{2,}")

SPARSE = "sparse"
DENSE = "dense"


def tokenize(text: str) -> list[str]:
    """Lower-cased content words, stop-words removed, plural 's' stripped."""
    out = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        if tok in _STOP:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


def _interaction_text(item: dict) -> str:
    # The user's words carry the topic; the response is context, so it gets
    # one pass while the user text is counted twice.
    user = item.get("query") or item.get("user_text") or ""
    resp = (item.get("response") or item.get("hestia_response") or "")[:200]
    return f"{user} {user} {resp}"


def sparse_vector(text: str) -> dict[str, float]:
    return dict(Counter(tokenize(text)))


def _cos_sparse(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _cos_dense(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def cosine(kind: str, a: Any, b: Any) -> float:
    return _cos_sparse(a, b) if kind == SPARSE else _cos_dense(a, b)


def _add_into(kind: str, centroid: Any, vec: Any) -> Any:
    """Return centroid + vec (running sum; cosine is scale-invariant)."""
    if kind == SPARSE:
        merged = dict(centroid or {})
        for k, v in vec.items():
            merged[k] = merged.get(k, 0.0) + v
        return merged
    if not centroid:
        return list(vec)
    return [x + y for x, y in zip(centroid, vec)]


# ---------------------------------------------------------------------------
# Pure clustering
# ---------------------------------------------------------------------------

@dataclass
class ClusterConfig:
    gap_minutes: float = 30.0          # follow-up window (halves the threshold)
    max_age_hours: float = 48.0        # how long an episode stays joinable
    sparse_threshold: float = 0.25
    dense_threshold: float = 0.55
    min_content_tokens: int = 2        # below this a message can't start an episode


@dataclass
class EpisodeState:
    id: Optional[int] = None           # None until persisted
    kind: str = SPARSE
    centroid: Any = field(default_factory=dict)
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    size: int = 0
    keywords: Counter = field(default_factory=Counter)
    intents: Counter = field(default_factory=Counter)
    member_ids: list = field(default_factory=list)   # NEW members this run
    dirty: bool = False

    def label(self, n: int = 3) -> str:
        ranked = sorted(self.keywords.items(), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(k for k, _ in ranked[:n]) or "misc"

    def top_intent(self) -> Optional[str]:
        skip = {"chat", "clarify_intent", ""}
        real = Counter({k: v for k, v in self.intents.items() if k not in skip})
        pool = real or self.intents
        return pool.most_common(1)[0][0] if pool else None


def parse_timestamp(value: Any) -> datetime:
    """SQLite CURRENT_TIMESTAMP ('YYYY-MM-DD HH:MM:SS') or ISO-8601, naive UTC."""
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def assign_interactions(
    episodes: list[EpisodeState],
    items: list[dict],
    vectors: Optional[Sequence[Sequence[float]]] = None,
    cfg: Optional[ClusterConfig] = None,
) -> dict[Any, EpisodeState]:
    """
    Assign chronologically-ordered *items* to *episodes* (mutated in place;
    new ones are appended). Each item needs ``id``, ``pushed_at`` and text
    (``query``/``response``). ``vectors`` (same length as *items*) switches
    to dense mode. Returns ``{item_id: EpisodeState}`` for every item that
    joined or started an episode (ignored filler is absent).
    """
    cfg = cfg or ClusterConfig()
    kind = DENSE if vectors is not None else SPARSE
    threshold = cfg.dense_threshold if kind == DENSE else cfg.sparse_threshold
    assignment: dict[Any, EpisodeState] = {}

    for idx, item in enumerate(items):
        when = parse_timestamp(item["pushed_at"])
        text = _interaction_text(item)
        # Filler is judged on what the USER said — Hestia's reply to "thanks"
        # ("You're welcome") has content words of its own and would make
        # every acknowledgement look like a fresh topic.
        tokens = tokenize(item.get("query") or item.get("user_text") or "")
        vec = list(vectors[idx]) if kind == DENSE else sparse_vector(text)

        # Episodes that can still take this message.
        open_eps = [
            ep for ep in episodes
            if ep.end is not None and ep.kind == kind
            and timedelta(0) <= when - ep.end <= timedelta(hours=cfg.max_age_hours)
        ]
        latest = max(open_eps, key=lambda e: e.end, default=None)

        best, best_score = None, 0.0
        for ep in open_eps:
            score = cosine(kind, vec, ep.centroid)
            in_followup_window = when - ep.end <= timedelta(minutes=cfg.gap_minutes)
            needed = threshold * (0.5 if in_followup_window else 1.0)
            if score >= needed and score >= best_score:
                best, best_score = ep, score

        is_filler = len(tokens) < cfg.min_content_tokens
        if best is None and is_filler:
            # "yes" / "thanks": belongs to the conversation it answers, but
            # never becomes an episode of its own.
            if latest is not None and when - latest.end <= timedelta(minutes=cfg.gap_minutes):
                best = latest
            else:
                continue

        if best is None:
            best = EpisodeState(kind=kind, centroid={} if kind == SPARSE else [], start=when)
            episodes.append(best)

        best.centroid = _add_into(kind, best.centroid, vec)
        best.end = when
        best.start = best.start or when
        best.size += 1
        best.keywords.update(tokenize(item.get("query") or item.get("user_text") or ""))
        if item.get("intent"):
            best.intents[item["intent"]] += 1
        best.member_ids.append(item["id"])
        best.dirty = True
        assignment[item["id"]] = best
    return assignment


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    label TEXT,
    kind TEXT,
    centroid TEXT,
    keywords TEXT,
    intents TEXT,
    start_at TEXT,
    end_at TEXT,
    size INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS episode_members (
    interaction_id INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS episode_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_end ON episodes(end_at);
CREATE INDEX IF NOT EXISTS idx_episode_members_ep ON episode_members(episode_id);
"""

_DT_FMT = "%Y-%m-%d %H:%M:%S"


class EpisodeStore:
    """
    Persists episodes in the same SQLite file as Mnemosyne's interaction log
    and reads new interactions straight from ``interaction_log`` (same
    database, so no second copy of the conversation is kept).
    """

    def __init__(self, db_path: str, cfg: Optional[ClusterConfig] = None) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or ClusterConfig()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.executescript(_SCHEMA)

    # -- helpers ---------------------------------------------------------

    def _get_state(self, key: str, default: str = "0") -> str:
        row = self._conn.execute(
            "SELECT value FROM episode_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def _set_state(self, key: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO episode_state (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _load_open(self, since: datetime) -> list[EpisodeState]:
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE end_at >= ?", (since.strftime(_DT_FMT),)
        ).fetchall()
        out = []
        for r in rows:
            out.append(EpisodeState(
                id=r["id"], kind=r["kind"] or SPARSE,
                centroid=json.loads(r["centroid"] or "null") or ({} if r["kind"] == SPARSE else []),
                start=parse_timestamp(r["start_at"]), end=parse_timestamp(r["end_at"]),
                size=r["size"] or 0,
                keywords=Counter(json.loads(r["keywords"] or "{}")),
                intents=Counter(json.loads(r["intents"] or "{}")),
            ))
        return out

    # -- clustering ------------------------------------------------------

    def cluster_new(
        self,
        embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
        batch_limit: int = 500,
    ) -> dict:
        """
        Cluster interactions logged since the last run. Returns
        ``{"processed", "episodes_touched", "new_episodes"}``. Never raises
        on an embedding failure — it falls back to bag-of-words for that
        batch and logs it.
        """
        last_id = int(self._get_state("last_interaction_id", "0"))
        rows = self._conn.execute(
            """
            SELECT id, user_text, hestia_response, intent, pushed_at
            FROM interaction_log WHERE id > ? ORDER BY id ASC LIMIT ?
            """,
            (last_id, batch_limit),
        ).fetchall()
        if not rows:
            return {"processed": 0, "episodes_touched": 0, "new_episodes": 0}

        items = [
            {"id": r["id"], "query": r["user_text"], "response": r["hestia_response"],
             "intent": r["intent"], "pushed_at": r["pushed_at"]}
            for r in rows
        ]

        vectors = None
        if embed_fn is not None:
            try:
                vectors = embed_fn([_interaction_text(i) for i in items])
                if len(vectors) != len(items):
                    raise ValueError("embedding count mismatch")
            except Exception:
                logger.exception("cluster_new: embedding failed; using bag-of-words.")
                vectors = None

        first = parse_timestamp(items[0]["pushed_at"])
        episodes = self._load_open(first - timedelta(hours=self.cfg.max_age_hours))
        known_ids = {e.id for e in episodes}
        assignment = assign_interactions(episodes, items, vectors, self.cfg)

        new_count = 0
        with self._lock, self._conn:
            for ep in episodes:
                if not ep.dirty:
                    continue
                if ep.id is None:
                    cur = self._conn.execute(
                        "INSERT INTO episodes (label, kind, centroid, keywords, intents, start_at, end_at, size) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        self._episode_params(ep),
                    )
                    ep.id = cur.lastrowid
                    new_count += 1
                else:
                    self._conn.execute(
                        "UPDATE episodes SET label=?, kind=?, centroid=?, keywords=?, intents=?, "
                        "start_at=?, end_at=?, size=? WHERE id=?",
                        self._episode_params(ep) + (ep.id,),
                    )
            for interaction_id, ep in assignment.items():
                self._conn.execute(
                    "INSERT OR REPLACE INTO episode_members (interaction_id, episode_id) VALUES (?, ?)",
                    (interaction_id, ep.id),
                )
            self._set_state("last_interaction_id", str(items[-1]["id"]))

        touched = {ep.id for ep in assignment.values()}
        return {
            "processed": len(items),
            "episodes_touched": len(touched),
            "new_episodes": len([i for i in touched if i not in known_ids]),
        }

    @staticmethod
    def _episode_params(ep: EpisodeState) -> tuple:
        return (
            ep.label(), ep.kind, json.dumps(ep.centroid),
            json.dumps(dict(ep.keywords)), json.dumps(dict(ep.intents)),
            ep.start.strftime(_DT_FMT) if ep.start else None,
            ep.end.strftime(_DT_FMT) if ep.end else None,
            ep.size,
        )

    # -- queries ---------------------------------------------------------

    def list_episodes(self, limit: int = 10, min_size: int = 1) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, start_at, end_at, size, intents FROM episodes "
            "WHERE size >= ? ORDER BY end_at DESC LIMIT ?",
            (min_size, limit),
        ).fetchall()
        return [self._row_to_summary(r) for r in rows]

    def _row_to_summary(self, r: sqlite3.Row) -> dict:
        intents = Counter(json.loads(r["intents"] or "{}"))
        skip = {"chat", "clarify_intent", ""}
        real = Counter({k: v for k, v in intents.items() if k not in skip})
        pool = real or intents
        return {
            "id": r["id"], "label": r["label"], "start": r["start_at"],
            "end": r["end_at"], "size": r["size"],
            "top_intent": pool.most_common(1)[0][0] if pool else None,
        }

    def find_episodes(self, query: str, limit: int = 3, min_size: int = 2) -> list[dict]:
        """
        Episodes best matching *query* by keyword overlap with each
        episode's accumulated keywords (weighted by how often each appeared),
        best first. Multi-message episodes only by default — a single
        stray message isn't an "episode" worth recalling.
        """
        terms = set(tokenize(query))
        if not terms:
            return []
        scored = []
        for r in self._conn.execute(
            "SELECT id, label, start_at, end_at, size, intents, keywords FROM episodes WHERE size >= ?",
            (min_size,),
        ).fetchall():
            kw = json.loads(r["keywords"] or "{}")
            total = sum(kw.values()) or 1
            score = sum(kw.get(t, 0) for t in terms) / total
            hits = sum(1 for t in terms if t in kw)
            if hits:
                scored.append((hits, score, r["end_at"], r))
        # Most distinct query terms matched, then density, then most recent.
        scored.sort(key=lambda s: (s[0], s[1], s[2]), reverse=True)
        return [self._row_to_summary(s[3]) for s in scored[:limit]]

    def get_members(self, episode_id: int, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT l.id, l.user_text, l.hestia_response, l.intent, l.pushed_at
            FROM episode_members m JOIN interaction_log l ON l.id = m.interaction_id
            WHERE m.episode_id = ? ORDER BY l.id ASC LIMIT ?
            """,
            (episode_id, limit),
        ).fetchall()
        return [
            {"id": r["id"], "query": r["user_text"], "response": r["hestia_response"],
             "intent": r["intent"], "pushed_at": r["pushed_at"]}
            for r in rows
        ]

    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(size), 0) AS members FROM episodes"
        ).fetchone()
        return {"episodes": row["n"], "clustered_interactions": row["members"]}
