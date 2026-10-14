"""
core/entity_coerce.py

Generic boundary coercion for NLU entities (Hardening Plan, Phase 3.1).

The model sometimes emits the wrong *type* for an entity (``topic=['gaps']``
instead of ``"gaps"``; ``amount=500`` vs ``"500"``). Modules generally call
``.strip()`` / ``.lower()`` on string entities, so a list there raised
``AttributeError`` (the logged Ares crash). This module normalises values once,
before any module sees them.

Rules
-----
* ``None`` values are dropped.
* A list/tuple/set of scalars under a ``STRING_FIELDS`` key becomes a
  ", "-joined string. Any other key (and lists of non-scalars) is unchanged.
* Keys starting with '_' (internal orchestrator state) are never touched.
* Scalars (str/int/float/bool) are untouched, so numeric fields stay numeric.
* Never raises; returns a new dict.
"""
from __future__ import annotations

from typing import Any, Dict

# Keys that modules treat as free-text strings (they call .strip()/.lower()).
# Only these are flattened; every other key keeps its type, because modules
# legitimately use list values (e.g. Hermes ``ids``, ``attendees``).
STRING_FIELDS = frozenset({
    "topic", "query", "raw_query", "name", "content", "text", "title", "task",
    "description", "company", "location", "person", "subject", "category",
    "key", "value", "note", "message", "date", "time", "when", "goal",
    "habit", "event", "theme", "genre", "artist", "movie", "city", "source",
    "mode", "style", "language", "label", "pattern", "recipient", "sender",
})

_SCALARS = (str, int, float, bool)


def coerce_entities(entities: Any) -> Dict[str, Any]:
    if not isinstance(entities, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in entities.items():
        if value is None:
            continue
        # "_"-prefixed keys are internal orchestrator state (e.g. the ids a
        # confirmation preview is bound to) and must keep their types.
        if (isinstance(value, (list, tuple, set))
                and str(key).lower() in STRING_FIELDS):
            items = list(value)
            if all(isinstance(v, _SCALARS) for v in items):
                value = ", ".join(str(v).strip() for v in items
                                  if str(v).strip())
        out[key] = value
    return out
