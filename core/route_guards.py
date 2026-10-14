"""
core/route_guards.py

Post-classification routing guards (Hardening Plan, Phase 2.4 / 3.3 / 3.6).
Each guard fixes a *logged* misroute using evidence in the utterance itself and
only ever moves an intent to a sibling in the same family:

* write -> read twin for question-form text ("What notes do I have?")
* creative requests (poem / story / brainstorm / lyrics) never go to
  Dionysus recommendations
* search is split by source: web / documents / photos
* photo-of-an-object requests ("pictures of my dog") are not person lookups
"""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from core.slot_extract import guard_write_intent

_CREATIVE_RE = re.compile(
    r"\b(poem|poetry|haiku|sonnet|limerick|lyrics|song about|short story|story about|"
    r"brainstorm|rewrite|rephrase|write (?:me )?a|ideas for)\b", re.I)
_REC_INTENTS = {"dionysus_recommend_movie", "dionysus_recommend_music",
                "dionysus_recommend_recipe", "suggest_activity"}
_WEB_RE = re.compile(r"\b(?:the )?(?:web|internet|online|google)\b", re.I)
_PHOTO_RE = re.compile(r"\b(photos?|pictures?|pics?|images?)\b", re.I)
_DOC_RE = re.compile(r"\b(documents?|docs?|files?|pdfs?|papers?)\b", re.I)
_SEARCH_VERB_RE = re.compile(r"^\s*(?:please\s+)?(?:search|find|look up|look for|google)\b", re.I)
_OBJECT_WORDS = frozenset({
    "dog", "dogs", "cat", "cats", "pet", "pets", "car", "cars", "food", "beach",
    "sunset", "flowers", "flower", "mountain", "mountains", "bike", "house",
    "baby", "puppy", "kitten", "bird", "birds", "tree", "trees", "sky",
})


def _query_from(text: str) -> str:
    m = re.search(r"\b(?:about|for|of)\s+(.+)$", text or "", re.I)
    return (m.group(1) if m else text or "").strip(" .?!")


def apply_guards(intent: str, entities: Dict[str, Any], text: str
                 ) -> Tuple[str, Dict[str, Any]]:
    ents = dict(entities or {})
    text = text or ""

    # 1. question-form write -> read twin
    guarded = guard_write_intent(intent, text)
    if guarded != intent:
        return guarded, {}

    # 2. creative requests are not recommendation requests
    if intent in _REC_INTENTS and _CREATIVE_RE.search(text):
        if re.search(r"\bbrainstorm|ideas for\b", text, re.I):
            return "orpheus_brainstorm", {"topic": _query_from(text)}
        if re.search(r"\bpoem|poetry|haiku|sonnet|limerick\b", text, re.I):
            return "orpheus_write_poem", {"topic": _query_from(text)}
        return "orpheus_creative_prompt", {"topic": _query_from(text)}

    # 3. search split by source
    if intent in ("athena_search", "hephaestus_search_web", "iris_search") \
            and _SEARCH_VERB_RE.search(text):
        if _WEB_RE.search(text) and intent != "hephaestus_search_web":
            return "hephaestus_search_web", {"query": _query_from(text)}
        if _PHOTO_RE.search(text) and intent != "iris_search" \
                and not _DOC_RE.search(text):
            return "iris_search", {"query": _query_from(text)}

    # 4. "pictures of my dog" is an object/entity photo search, not a person
    if intent == "iris_find_person":
        person = str(ents.get("person") or ents.get("name") or "").lower()
        words = set(re.findall(r"[a-z]+", person))
        if person.startswith("my ") and words & _OBJECT_WORDS or words <= _OBJECT_WORDS and words:
            return "iris_search", {"query": person.replace("my ", "").strip() or _query_from(text)}
    return intent, ents
