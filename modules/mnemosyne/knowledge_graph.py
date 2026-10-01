"""
modules/mnemosyne/knowledge_graph.py

Knowledge graph over Mnemosyne's facts and notes (backlog #31), plus the
node/link export the web UI draws (backlog #32).

What it stores
--------------
* **Entities** — people, places, organisations, concepts, notes.
* **Edges** — a directed, labelled relation between two entities
  ("You --sister name--> Priya"), or an undirected ``related_to`` for plain
  co-occurrence in the same sentence.
* **Sources** — every entity and edge remembers *which* fact/note produced
  it (``source_ref`` such as ``fact:sister_name`` or ``note:Thermo.md``).
  Weights and mention counts are derived from the number of distinct
  sources, so re-extracting the same text is idempotent, and forgetting a
  fact cleanly removes exactly what only that fact contributed
  (``remove_source``) instead of leaving a ghost in the graph.

Extraction is two-tier, on purpose:
  1. ``extract_heuristic`` — rule-based, instant, no LLM. Runs inline when a
     fact is learned, so the graph is always current.
  2. ``extract_with_llm`` — richer subject/relation/object triples from the
     local model, used by the explicit rebuild. If the model is down or
     returns junk it falls back to (1); it never raises.

Like ``core/quiz_store.py`` and ``spaced_repetition.py`` this module owns
its tables (``CREATE TABLE IF NOT EXISTS`` on open), so an existing
database needs no migration.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
import sqlite3
import threading
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

USER_ENTITY = "You"          # the node every user-stated fact hangs off
RELATED_TO = "related to"    # undirected co-occurrence relation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kg_entities (
    id INTEGER PRIMARY KEY,
    norm TEXT NOT NULL UNIQUE,      -- lower-cased lookup key
    name TEXT NOT NULL,             -- display form (first spelling seen)
    type TEXT DEFAULT 'thing'
);
CREATE TABLE IF NOT EXISTS kg_entity_sources (
    entity_id INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    source_ref TEXT NOT NULL,
    PRIMARY KEY (entity_id, source_ref)
);
CREATE TABLE IF NOT EXISTS kg_edges (
    id INTEGER PRIMARY KEY,
    src INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    dst INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    UNIQUE (src, dst, relation)
);
CREATE TABLE IF NOT EXISTS kg_edge_sources (
    edge_id INTEGER NOT NULL REFERENCES kg_edges(id) ON DELETE CASCADE,
    source_ref TEXT NOT NULL,
    PRIMARY KEY (edge_id, source_ref)
);
CREATE TABLE IF NOT EXISTS kg_processed (
    source_ref TEXT PRIMARY KEY,
    content_hash TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_kg_edges_src ON kg_edges(src);
CREATE INDEX IF NOT EXISTS idx_kg_edges_dst ON kg_edges(dst);
CREATE INDEX IF NOT EXISTS idx_kg_entity_sources_ref ON kg_entity_sources(source_ref);
CREATE INDEX IF NOT EXISTS idx_kg_edge_sources_ref ON kg_edge_sources(source_ref);
"""


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_name(name: str) -> str:
    """Lower-case, punctuation-light, single-spaced lookup key."""
    cleaned = re.sub(r"[^\w\s\-&+.]", " ", (name or "").lower())
    cleaned = cleaned.replace("_", " ")
    return " ".join(cleaned.split()).strip(" .-")


# ---------------------------------------------------------------------------
# Extraction — heuristic tier
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    """
    a an the this that these those it its i me my mine we our us you your he
    she him her his hers they them their there here what which who whom
    whose when where why how and or but if then so because as of at by for
    from in into on onto to with without about above after before between
    is are was were be been being am do does did done have has had will
    would can could should may might must not no yes also just very
    today tomorrow yesterday tonight now monday tuesday wednesday thursday
    friday saturday sunday january february march april may june july
    august september october november december however therefore thus
    note notes remember fact ok okay please thanks hello hi hey
    """.split()
)

# A run of Capitalised words: "Priya", "Indian Institute of Technology".
# Only genuine name-internal connectors (of/de/du/van/von/da) may sit between
# capitalised words — "and"/"the"/"in" must NOT, or "Professor Rao and
# Carnot" fuses two people into one node.
_PROPER_RE = re.compile(
    r"\b[A-Z][\w'\-]*(?:\s+(?:of|de|du|van|von|da)\s+[A-Z][\w'\-]*|\s+[A-Z][\w'\-]*)*"
)
_WIKILINK_RE = re.compile(r"\[\[([^\]\|#]+)(?:#[^\]\|]*)?(?:\|[^\]]*)?\]\]")
_TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z][\w\-/]*)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

_MAX_ENTITIES_PER_SENTENCE = 6   # caps the O(n^2) co-occurrence fan-out
_MAX_FACT_VALUE_WORDS = 6        # longer values are prose, not an object


def _clean_candidate(text: str) -> Optional[str]:
    words = text.strip(" .,;:-'\"").split()
    # Drop leading/trailing stopwords ("The Indian Institute" -> keep whole,
    # but "The" alone or "I" is noise).
    while words and words[0].lower() in _STOPWORDS:
        words = words[1:]
    while words and words[-1].lower() in _STOPWORDS:
        words = words[:-1]
    if not words:
        return None
    candidate = " ".join(words)
    if len(candidate) < 2 or len(candidate) > 60:
        return None
    if candidate.isdigit():
        return None
    return candidate


def extract_entities_heuristic(text: str) -> list[tuple[str, str]]:
    """
    Pull ``(name, type)`` candidates out of *text* without an LLM.

    Sources, in confidence order: ``[[wikilinks]]`` (concept), ``#tags``
    (tag), then runs of Capitalised words (thing). Order is preserved and
    duplicates (case-insensitively) are dropped.
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(name: Optional[str], etype: str) -> None:
        if not name:
            return
        key = normalise_name(name)
        if key and key not in seen and key not in _STOPWORDS:
            seen.add(key)
            found.append((name, etype))

    for m in _WIKILINK_RE.finditer(text or ""):
        _add(_clean_candidate(m.group(1)), "concept")
    for m in _TAG_RE.finditer(text or ""):
        _add(m.group(1).replace("-", " ").replace("/", " / ").strip(), "tag")
    for m in _PROPER_RE.finditer(_WIKILINK_RE.sub(" ", text or "")):
        _add(_clean_candidate(m.group(0)), "thing")
    return found


def extract_from_fact(key: str, value: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """
    Turn one user-stated fact into entities and relations.

    ``sister_name = Priya`` becomes ``You --sister name--> Priya``: the
    fact's *key* is the relation label, which is exactly the wording the
    user would use to ask ("who is my sister"). Long values are prose, so
    instead of one giant object node they go through the text extractor and
    each proper noun is linked to You with the key as the relation.
    """
    key_label = normalise_name(key)
    value = (value or "").strip()
    entities: list[tuple[str, str]] = [(USER_ENTITY, "person")]
    relations: list[tuple[str, str, str]] = []
    if not key_label or not value:
        return entities, relations

    if len(value.split()) <= _MAX_FACT_VALUE_WORDS and not value.startswith("{"):
        entities.append((value, "thing"))
        relations.append((USER_ENTITY, key_label, value))
        return entities, relations

    for name, etype in extract_entities_heuristic(value):
        entities.append((name, etype))
        relations.append((USER_ENTITY, key_label, name))
    return entities, relations


def extract_cooccurrence(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """
    Entities for a block of prose, plus ``related to`` links between
    entities mentioned in the *same sentence* (paragraph-wide co-occurrence
    links everything to everything and drowns the graph in noise).
    """
    entities: list[tuple[str, str]] = []
    seen: set[str] = set()
    relations: list[tuple[str, str, str]] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text or ""):
        ents = extract_entities_heuristic(sentence)[:_MAX_ENTITIES_PER_SENTENCE]
        for name, etype in ents:
            k = normalise_name(name)
            if k not in seen:
                seen.add(k)
                entities.append((name, etype))
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                relations.append((ents[i][0], RELATED_TO, ents[j][0]))
    return entities, relations


# ---------------------------------------------------------------------------
# Extraction — LLM tier
# ---------------------------------------------------------------------------

_VALID_TYPES = frozenset({"person", "place", "organisation", "concept", "thing", "tag", "note", "paper"})

_LLM_PROMPT = """\
Extract a knowledge graph from the text below. Return ONLY valid JSON, no
preamble, in exactly this shape:

{{"entities": [{{"name": "...", "type": "person|place|organisation|concept|thing"}}],
  "relations": [{{"subject": "...", "relation": "short verb phrase", "object": "..."}}]}}

Rules: use only what the text states; keep relation labels under 4 words;
every subject/object must also appear in "entities"; at most {max_items} entities.

Text:
{text}
"""


def parse_llm_extraction(raw: Any) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """
    Validate model output into ``(entities, relations)``. Anything malformed
    is dropped piecemeal rather than rejecting the whole response, because
    a small local model commonly gets most of a list right.
    """
    try:
        parsed = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except (TypeError, ValueError):
        return [], []
    if not isinstance(parsed, dict):
        return [], []

    entities: list[tuple[str, str]] = []
    known: set[str] = set()
    for item in parsed.get("entities") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        etype = str(item.get("type") or "thing").strip().lower()
        if not name or len(name) > 80:
            continue
        if etype not in _VALID_TYPES:
            etype = "thing"
        entities.append((name, etype))
        known.add(normalise_name(name))

    relations: list[tuple[str, str, str]] = []
    for item in parsed.get("relations") or []:
        if not isinstance(item, dict):
            continue
        s = str(item.get("subject") or "").strip()
        r = str(item.get("relation") or "").strip()
        o = str(item.get("object") or "").strip()
        if not (s and r and o) or normalise_name(s) == normalise_name(o):
            continue
        if len(r.split()) > 6:
            continue
        for name in (s, o):   # a relation may mention an entity the list missed
            if normalise_name(name) not in known:
                entities.append((name, "thing"))
                known.add(normalise_name(name))
        relations.append((s, r.lower(), o))
    return entities, relations


def extract_with_llm(
    llm: Any, text: str, max_items: int = 12
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """LLM extraction with a heuristic fallback; never raises."""
    text = (text or "").strip()
    if not text:
        return [], []
    if llm is not None:
        try:
            raw = llm.generate(
                _LLM_PROMPT.format(text=text[:3000], max_items=max_items), fmt="json"
            )
            entities, relations = parse_llm_extraction(raw)
            if entities:
                return entities[:max_items], relations
        except Exception:
            logger.exception("extract_with_llm: model call failed; using heuristic.")
    return extract_cooccurrence(text)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


class KnowledgeGraph:
    """SQLite-backed entity/relation graph with source tracking."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.executescript(_SCHEMA)

    # -- writes ----------------------------------------------------------

    def _upsert_entity(self, name: str, etype: str, source_ref: str) -> Optional[int]:
        norm = normalise_name(name)
        if not norm:
            return None
        etype = etype if etype in _VALID_TYPES else "thing"
        row = self._conn.execute(
            "SELECT id, type FROM kg_entities WHERE norm = ?", (norm,)
        ).fetchone()
        if row is None:
            cur = self._conn.execute(
                "INSERT INTO kg_entities (norm, name, type) VALUES (?, ?, ?)",
                (norm, name.strip(), etype),
            )
            eid = cur.lastrowid
        else:
            eid = row["id"]
            # A specific type beats the generic default.
            if row["type"] == "thing" and etype != "thing":
                self._conn.execute(
                    "UPDATE kg_entities SET type = ? WHERE id = ?", (etype, eid)
                )
        self._conn.execute(
            "INSERT OR IGNORE INTO kg_entity_sources (entity_id, source_ref) VALUES (?, ?)",
            (eid, source_ref),
        )
        return eid

    def add_extraction(
        self,
        source_ref: str,
        entities: Iterable[tuple[str, str]],
        relations: Iterable[tuple[str, str, str]],
    ) -> dict:
        """
        Record one source's entities and relations. Idempotent: adding the
        same thing from the same source twice changes nothing. Returns
        ``{"entities": n, "edges": n}`` counts of what this source touched.
        """
        if not source_ref:
            raise ValueError("add_extraction() requires a source_ref.")
        touched_entities: set[int] = set()
        touched_edges: set[int] = set()
        with self._lock, self._conn:
            for name, etype in entities:
                eid = self._upsert_entity(name, etype, source_ref)
                if eid is not None:
                    touched_entities.add(eid)
            for subject, relation, obj in relations:
                relation = " ".join((relation or "").lower().split())
                if not relation:
                    continue
                a = self._upsert_entity(subject, "thing", source_ref)
                b = self._upsert_entity(obj, "thing", source_ref)
                if a is None or b is None or a == b:
                    continue
                if relation == RELATED_TO and a > b:
                    a, b = b, a   # undirected: one canonical row
                self._conn.execute(
                    "INSERT OR IGNORE INTO kg_edges (src, dst, relation) VALUES (?, ?, ?)",
                    (a, b, relation),
                )
                edge_id = self._conn.execute(
                    "SELECT id FROM kg_edges WHERE src = ? AND dst = ? AND relation = ?",
                    (a, b, relation),
                ).fetchone()["id"]
                self._conn.execute(
                    "INSERT OR IGNORE INTO kg_edge_sources (edge_id, source_ref) VALUES (?, ?)",
                    (edge_id, source_ref),
                )
                touched_edges.add(edge_id)
                touched_entities.update((a, b))
        return {"entities": len(touched_entities), "edges": len(touched_edges)}

    def remove_source(self, source_ref: str) -> dict:
        """
        Withdraw everything *source_ref* contributed. Entities and edges
        that no other source supports are deleted; shared ones stay.
        """
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM kg_edge_sources WHERE source_ref = ?", (source_ref,)
            )
            self._conn.execute(
                "DELETE FROM kg_entity_sources WHERE source_ref = ?", (source_ref,)
            )
            edges = self._conn.execute(
                "DELETE FROM kg_edges WHERE id NOT IN (SELECT edge_id FROM kg_edge_sources)"
            ).rowcount
            # An entity survives only while some source or some edge keeps it.
            entities = self._conn.execute(
                """
                DELETE FROM kg_entities
                WHERE id NOT IN (SELECT entity_id FROM kg_entity_sources)
                  AND id NOT IN (SELECT src FROM kg_edges)
                  AND id NOT IN (SELECT dst FROM kg_edges)
                """
            ).rowcount
            self._conn.execute(
                "DELETE FROM kg_processed WHERE source_ref = ?", (source_ref,)
            )
        return {"entities_removed": entities, "edges_removed": edges}

    def is_processed(self, source_ref: str, hash_: str) -> bool:
        row = self._conn.execute(
            "SELECT content_hash FROM kg_processed WHERE source_ref = ?", (source_ref,)
        ).fetchone()
        return row is not None and row["content_hash"] == hash_

    def mark_processed(self, source_ref: str, hash_: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO kg_processed (source_ref, content_hash) VALUES (?, ?)
                ON CONFLICT(source_ref) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    processed_at = CURRENT_TIMESTAMP
                """,
                (source_ref, hash_),
            )

    # -- lookups ---------------------------------------------------------

    def find_entity(self, name: str) -> Optional[dict]:
        """
        Resolve a spoken/typed name to an entity: exact, then substring
        (prefers the shortest containing name), then a close fuzzy match —
        so "priya" and "Priya" and "my sister priya" all land on one node.
        """
        norm = normalise_name(name)
        if not norm:
            return None
        row = self._conn.execute(
            "SELECT * FROM kg_entities WHERE norm = ?", (norm,)
        ).fetchone()
        if row:
            return dict(row)

        rows = self._conn.execute("SELECT * FROM kg_entities").fetchall()
        if not rows:
            return None
        contained = [r for r in rows if r["norm"] in norm.split() or f" {r['norm']} " in f" {norm} "]
        if contained:   # the longest known name appearing inside the query
            return dict(max(contained, key=lambda r: len(r["norm"])))
        containing = [r for r in rows if norm in r["norm"]]
        if containing:
            return dict(min(containing, key=lambda r: len(r["norm"])))
        best, best_ratio = None, 0.0
        for r in rows:
            ratio = difflib.SequenceMatcher(None, norm, r["norm"]).ratio()
            if ratio > best_ratio:
                best, best_ratio = r, ratio
        return dict(best) if best is not None and best_ratio >= 0.82 else None

    def neighbors(self, entity_id: int, limit: int = 25) -> list[dict]:
        """
        Direct connections of one entity, strongest first. Each row:
        ``{id, name, type, relation, direction, weight}`` where direction is
        ``out`` (entity -> other), ``in`` (other -> entity) or ``both``
        (undirected ``related to``).
        """
        rows = self._conn.execute(
            """
            SELECT e.id AS edge_id, e.src, e.dst, e.relation,
                   (SELECT COUNT(*) FROM kg_edge_sources s WHERE s.edge_id = e.id) AS weight
            FROM kg_edges e
            WHERE e.src = ? OR e.dst = ?
            """,
            (entity_id, entity_id),
        ).fetchall()
        out: list[dict] = []
        for r in rows:
            other_id = r["dst"] if r["src"] == entity_id else r["src"]
            other = self._conn.execute(
                "SELECT id, name, type FROM kg_entities WHERE id = ?", (other_id,)
            ).fetchone()
            if other is None:
                continue
            if r["relation"] == RELATED_TO:
                direction = "both"
            else:
                direction = "out" if r["src"] == entity_id else "in"
            out.append({
                "id": other["id"], "name": other["name"], "type": other["type"],
                "relation": r["relation"], "direction": direction,
                "weight": r["weight"],
            })
        out.sort(key=lambda n: (-n["weight"], n["name"].lower()))
        return out[:limit]

    def find_path(self, start_id: int, end_id: int, max_depth: int = 4) -> Optional[list[dict]]:
        """
        Shortest chain of connections between two entities (BFS, edges
        treated as undirected for reachability). Returns a list of hops
        ``{from, relation, direction, to}`` or None when not connected
        within *max_depth*. ``[]`` means start and end are the same node.
        """
        if start_id == end_id:
            return []
        adjacency: dict[int, list[tuple[int, str, str]]] = {}
        for r in self._conn.execute("SELECT src, dst, relation FROM kg_edges").fetchall():
            fwd = "both" if r["relation"] == RELATED_TO else "out"
            back = "both" if r["relation"] == RELATED_TO else "in"
            adjacency.setdefault(r["src"], []).append((r["dst"], r["relation"], fwd))
            adjacency.setdefault(r["dst"], []).append((r["src"], r["relation"], back))

        queue: deque[int] = deque([start_id])
        parent: dict[int, Optional[tuple[int, str, str]]] = {start_id: None}
        depth = {start_id: 0}
        while queue:
            node = queue.popleft()
            if depth[node] >= max_depth:
                continue
            for nxt, relation, direction in adjacency.get(node, []):
                if nxt in parent:
                    continue
                parent[nxt] = (node, relation, direction)
                depth[nxt] = depth[node] + 1
                if nxt == end_id:
                    return self._unwind_path(parent, end_id)
                queue.append(nxt)
        return None

    def _unwind_path(self, parent: dict, end_id: int) -> list[dict]:
        hops: list[dict] = []
        node = end_id
        while parent[node] is not None:
            prev, relation, direction = parent[node]
            hops.append({
                "from": self._entity_name(prev), "relation": relation,
                "direction": direction, "to": self._entity_name(node),
            })
            node = prev
        hops.reverse()
        return hops

    def _entity_name(self, entity_id: int) -> str:
        row = self._conn.execute(
            "SELECT name FROM kg_entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return row["name"] if row else "?"

    def sources_for(self, entity_id: int, limit: int = 10) -> list[str]:
        rows = self._conn.execute(
            "SELECT source_ref FROM kg_entity_sources WHERE entity_id = ? ORDER BY source_ref LIMIT ?",
            (entity_id, limit),
        ).fetchall()
        return [r["source_ref"] for r in rows]

    # -- export / stats --------------------------------------------------

    def graph_data(self, max_nodes: int = 150, min_weight: int = 1) -> dict:
        """
        Nodes and links for the force-directed view (backlog #32).

        Keeps the *max_nodes* best-connected entities (degree, then number
        of sources) and only the links whose both ends survived — a graph
        of everything is an unreadable hairball, and D3's force layout
        slows down well before a few hundred nodes.
        """
        max_nodes = max(1, min(int(max_nodes), 500))
        nodes = self._conn.execute(
            """
            SELECT n.id, n.name, n.type,
                   (SELECT COUNT(*) FROM kg_edges e WHERE e.src = n.id OR e.dst = n.id) AS degree,
                   (SELECT COUNT(*) FROM kg_entity_sources s WHERE s.entity_id = n.id) AS mentions
            FROM kg_entities n
            ORDER BY degree DESC, mentions DESC, n.name ASC
            LIMIT ?
            """,
            (max_nodes,),
        ).fetchall()
        keep = {r["id"] for r in nodes}
        links = []
        for r in self._conn.execute(
            """
            SELECT e.src, e.dst, e.relation,
                   (SELECT COUNT(*) FROM kg_edge_sources s WHERE s.edge_id = e.id) AS weight
            FROM kg_edges e
            """
        ).fetchall():
            if r["src"] in keep and r["dst"] in keep and r["weight"] >= min_weight:
                links.append({
                    "source": r["src"], "target": r["dst"],
                    "relation": r["relation"], "weight": r["weight"],
                })
        return {
            "nodes": [
                {"id": r["id"], "name": r["name"], "type": r["type"],
                 "degree": r["degree"], "mentions": r["mentions"]}
                for r in nodes
            ],
            "links": links,
        }

    def stats(self) -> dict:
        ents = self._conn.execute("SELECT COUNT(*) AS n FROM kg_entities").fetchone()["n"]
        edges = self._conn.execute("SELECT COUNT(*) AS n FROM kg_edges").fetchone()["n"]
        return {"entities": ents, "edges": edges}


# ---------------------------------------------------------------------------
# Natural-language rendering
# ---------------------------------------------------------------------------

def describe_connections(entity: dict, neighbours: list[dict], limit: int = 8) -> str:
    """One spoken-style answer to "what connects to X"."""
    name = entity["name"]
    if not neighbours:
        return f"I know {name}, but nothing is connected to it yet."
    parts = []
    for n in neighbours[:limit]:
        if n["direction"] == "out":
            parts.append(f"{name} — {n['relation']} → {n['name']}")
        elif n["direction"] == "in":
            parts.append(f"{n['name']} — {n['relation']} → {name}")
        else:
            parts.append(f"{n['name']}")
    related = [n["name"] for n in neighbours[:limit] if n["direction"] == "both"]
    labelled = [p for p, n in zip(parts, neighbours[:limit]) if n["direction"] != "both"]
    sentences = []
    if labelled:
        sentences.append("; ".join(labelled))
    if related:
        sentences.append("Related to " + ", ".join(related))
    more = len(neighbours) - limit
    suffix = f" (+{more} more)" if more > 0 else ""
    return f"{name}: " + ". ".join(sentences) + suffix + "."


def describe_path(path: Optional[list[dict]]) -> str:
    if path is None:
        return "I can't find a connection between those."
    if not path:
        return "Those are the same thing."
    steps = []
    for hop in path:
        if hop["direction"] == "in":
            steps.append(f"{hop['from']} ← {hop['relation']} ← {hop['to']}")
        elif hop["direction"] == "both":
            steps.append(f"{hop['from']} ~ {hop['to']}")
        else:
            steps.append(f"{hop['from']} → {hop['relation']} → {hop['to']}")
    return "Connection: " + "; then ".join(steps) + "."
