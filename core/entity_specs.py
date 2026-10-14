"""
core/entity_specs.py

Per-intent entity contracts (Hardening Plan, Phase 3.1): field name, type,
required, and accepted aliases, for the most-used intents. Generic coercion
(core/entity_coerce.py) still covers every other intent.

``apply_spec(intent, entities)`` -> (entities, missing_required)
    * renames alias keys onto the canonical key (never overwriting it),
    * coerces values to the declared type (list -> str, "Rs 500" -> 500.0 for
      numbers, "true"/"yes" -> True for bools); un-coercible values are dropped
      so the module re-asks instead of crashing,
    * reports required fields that are absent.
Never raises.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

_NUM_RE = re.compile(r"[^\d.\-]")


@dataclass(frozen=True)
class F:
    name: str
    type: str = "str"            # str | num | bool
    required: bool = False
    aliases: Tuple[str, ...] = ()


SPECS: Dict[str, Tuple[F, ...]] = {
    "take_note": (F("content", required=True, aliases=("note", "text", "task")),),
    "learn_fact": (F("key", required=True, aliases=("fact_key",)),
                   F("value", required=True, aliases=("fact_value",))),
    "save_name": (F("name", required=True, aliases=("user_name", "value")),),
    "get_user_info": (F("key", aliases=("field", "info")),),
    "forget_fact": (F("key", aliases=("fact_key",)), F("pattern", aliases=("topic",))),
    "add_habit": (F("name", required=True, aliases=("habit", "content", "task", "title")),),
    "complete_habit": (F("name", required=True, aliases=("habit", "content", "task", "title")),),
    "remove_habit": (F("name", required=True, aliases=("habit", "content", "task", "title")),),
    "add_goal": (F("name", required=True, aliases=("goal", "content", "task", "title")),),
    "remove_goal": (F("name", required=True, aliases=("goal", "content", "task", "title")),),
    "set_reminder": (F("task", required=True, aliases=("content", "reminder", "note")),
                     F("date"), F("time")),
    "create_event": (F("title", required=True, aliases=("name", "event", "summary")),
                     F("date"), F("time"), F("location")),
    "pluto_log_expense": (F("amount", "num", True, ("cost", "price", "value")),
                          F("description", aliases=("category", "item", "note"))),
    "apollo_log_water": (F("amount", "num", False, ("ml", "water", "quantity")),),
    "apollo_log_weight": (F("weight", "num", True, ("mass", "kg", "lbs", "value")),),
    "apollo_log_mood": (F("mood", required=True, aliases=("emotion", "feeling")),),
    "get_weather": (F("location", aliases=("city", "place")),),
    "hephaestus_search_web": (F("query", required=True, aliases=("topic", "q")),),
    "athena_search": (F("query", required=True, aliases=("topic", "subject")),),
    "iris_search": (F("query", required=True, aliases=("object", "subject", "topic")),),
    "iris_find_person": (F("person", required=True, aliases=("name",)),),
    "send_email": (F("to", aliases=("recipient",)), F("subject"), F("body", aliases=("content", "text"))),
    "draft_email": (F("to", aliases=("recipient",)), F("subject"), F("body", aliases=("content", "text"))),
    "orpheus_write_poem": (F("topic", aliases=("subject", "theme")),),
    "orpheus_brainstorm": (F("topic", aliases=("subject", "theme")),),
    "ares_competitive_analysis": (F("topic", aliases=("company", "subject")),),
    "pluto_company_lookup": (F("company", required=True, aliases=("name", "ticker", "symbol")),),
}


def _to_str(v: Any) -> str:
    if isinstance(v, (list, tuple, set)):
        return ", ".join(str(x).strip() for x in v if str(x).strip())
    return str(v).strip()


def _to_num(v: Any):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    raw = str(v).strip()
    # Strip only currency markers/commas; "5k" or "2 lakh" are NOT plain
    # numbers, so they are returned untouched for the module's own parser.
    s = re.sub(r"(?i)^(?:rs\.?|inr|rupees?|usd|eur|[$€£₹])\s*|\s*(?:rs\.?|inr|rupees?)$", "", raw).replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return float(s) if "." in s else int(s)
    return raw or None


def _to_bool(v: Any):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "yes", "y", "1", "on"):
        return True
    if s in ("false", "no", "n", "0", "off"):
        return False
    return None


def requires_entities(intent: str) -> bool:
    """True if the intent has a required entity (so a classifier-only guess,
    which extracts no entities, is not enough)."""
    return any(f.required for f in SPECS.get(intent, ()))


def apply_spec(intent: str, entities: Any) -> Tuple[Dict[str, Any], List[str]]:
    ents: Dict[str, Any] = dict(entities) if isinstance(entities, dict) else {}
    spec = SPECS.get(intent)
    if not spec:
        return ents, []
    missing: List[str] = []
    for f in spec:
        for alias in f.aliases:
            if alias in ents and f.name not in ents:
                ents[f.name] = ents.pop(alias)
        if f.name in ents:
            raw = ents[f.name]
            val = None
            if raw is not None:
                val = {"str": _to_str, "num": _to_num, "bool": _to_bool}[f.type](raw)
            if val in (None, ""):
                ents.pop(f.name, None)
            else:
                ents[f.name] = val
        if f.required and f.name not in ents:
            missing.append(f.name)
    return ents, missing
