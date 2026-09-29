"""
core/intent_chains.py

Intent chaining/pipelines (backlog #24): "summarize this paper and add it
to my reading list" should pipe the SUMMARY into the save action, not
dispatch "add it to my reading list" on its own with nothing to add.

How this differs from multi-intent splitting (#13)
-----------------------------------------------------
`core/query_splitter.py` + `Hestia._try_multi_intent` already split a
compound query into two independent actions and dispatch both — that's
right for "log my workout and tell me the weather", where the two halves
share nothing. It is WRONG for "summarize this and add it to my reading
list": dispatching the second half on its own hands `take_note` the literal
text "add it to my reading list", which is not a summary of anything.

This module adds the missing piece: detecting when the SECOND segment of
an already-split compound query is referring back to the FIRST segment's
result — "add IT", "save THAT", "note IT down" — rather than carrying its
own content. When it is, the caller should pipe the first segment's
dispatched response text into the second segment's entities instead of
dispatching it as-is.

Scope, deliberately conservative
----------------------------------
Detection is a regex over the segment's own text, not a semantic guess —
the same design principle as `core.query_splitter`: a wrong detection
here costs one entity substitution, never a wrong action taken, because
the substituted content is real dispatched output from the FIRST half of
the SAME query the user just typed, not fabricated.

Only `take_note` is wired as a chain target for now — its entity shape
(`content`/`text`/`task`, with a permissive fallback) was verified
directly against `modules/hestia/core_module.py._take_note` rather than
assumed. Other plausible targets (`add_goal`, `learn_fact`, `add_habit`)
have entity shapes this module hasn't verified against their real
handlers, so adding them here is future work, not a guess baked in now —
see `CHAINABLE_TARGETS`' own comment for how to extend it once verified.
"""
from __future__ import annotations

import re
from typing import Optional

# target intent -> the entity key its handler reads the piped-in content
# from. EXTEND THIS ONLY after checking the real handler, the way
# "take_note" was checked against modules/hestia/core_module.py._take_note
# (module docstring above) — an unverified entry here would silently pipe
# content into a key the handler never reads.
CHAINABLE_TARGETS: dict[str, str] = {
    "take_note": "content",
}

# "add/save/note/track/remember" + an anaphoric object ("it"/"that"/
# "this"), optionally followed by more words ("...to my reading list").
# Anchored at the start of the segment: a mid-sentence "it" is too weak a
# signal on its own ("check if it works" is not a chain reference).
_ANAPHORIC_RE = re.compile(
    r"^(?:add|save|note|track|remember|keep|jot|file|log)\s+"
    r"(?:it|that|this)\b",
    re.IGNORECASE,
)


def detect_chain_reference(segment_text: str, target_intent: str) -> Optional[str]:
    """
    If *segment_text* looks like it refers back to a PRECEDING segment's
    result rather than carrying its own content, and *target_intent* is a
    verified chainable target, return the entity key that content should
    be piped into. Otherwise return None — meaning: dispatch normally,
    nothing to chain.
    """
    entity_key = CHAINABLE_TARGETS.get(target_intent)
    if entity_key is None:
        return None
    if not segment_text or not _ANAPHORIC_RE.match(segment_text.strip()):
        return None
    return entity_key


def apply_chain(
    entities: dict, entity_key: str, piped_content: str
) -> dict:
    """
    Return a copy of *entities* with *entity_key* set to *piped_content*.

    Always overwrites rather than only filling if absent: the whole point
    of detecting an anaphoric reference is that whatever the NLU itself
    extracted for this segment (if anything) is not real content — it's
    at best a restatement of "it"/"that", which is worse than the actual
    piped-in result.
    """
    updated = dict(entities or {})
    updated[entity_key] = piped_content
    return updated
