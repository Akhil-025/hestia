"""
core/query_splitter.py

Compound-query splitting (backlog #13): "log my workout and tell me the
weather" is two requests, not one, and used to be handled entirely by
whichever single intent the NLU happened to pick for the whole sentence
— usually whichever half came first, with the second half silently
dropped.

Design
------
Splitting text on conjunctions is easy; knowing whether a *given* split is
actually two separate requests (rather than one request that happens to
contain the word "and" — "log my workout and how I felt", "remind me to
buy milk and eggs") is not solvable by string matching alone. So this
module deliberately does the cheap, safe half only:

  `candidate_segments(text)` finds *plausible* split points and returns
  the candidate segments. It over-generates: for ambiguous input it
  returns more than one candidate split, or none, and never claims
  confidence it doesn't have.

The actual decision — "are these really two different intents, or was
splitting a mistake?" — is made by the CALLER (`Hestia.process_text`),
which classifies each candidate segment through the real NLU and only
commits to the split if the segments resolve to two DIFFERENT, concrete
(non-chat) registered intents. That keeps a wrong split cheap to detect
and impossible to act on by accident: the worst case of a bad split is
one extra NLU call that gets discarded, not a wrong action taken.
"""
from __future__ import annotations

import re
from typing import Optional

# Conjunctions checked in this order — longer/more explicit phrases first,
# so "and then" isn't matched as a lower-priority bare "and" first and
# left as two awkward fragments.
_CONNECTORS: tuple[str, ...] = (
    " and also ",
    " and then ",
    "; ",
    " and ",
)

# A hard floor on segment length (in characters, after stripping): splits
# that would produce a segment shorter than this are almost always a
# mistake ("mac and cheese" -> "mac", "cheese" — neither is a request on
# its own) and are discarded rather than offered as a candidate.
_MIN_SEGMENT_CHARS = 6

# Bare " and " is the connector most likely to appear INSIDE a single
# request rather than BETWEEN two ("bacon and eggs", "sooner and later",
# "back and forth", "salt and pepper"). Splitting on it is only offered
# when neither side looks like a short noun-list fragment — approximated
# here as "at least one side contains a verb-shaped first word", which is
# a heuristic, not a parse; see the module docstring for why the real
# decision is deferred to the caller's NLU classification instead of being
# made here.
_VERBY_STARTS = re.compile(
    r"^(log|track|tell|show|what|when|where|who|why|how|remind|set|add|"
    r"remove|delete|list|get|check|play|find|search|create|send|read|"
    r"give|update|mark|complete|start|stop|record|summar|analy|write|"
    r"forget|remember|compare)",
    re.IGNORECASE,
)


def candidate_segments(text: str) -> list[str]:
    """
    Return plausible split points for *text*, longest/most-explicit
    connector first. An empty or single-element result means "don't
    split" — the caller should fall through to normal single-query
    handling.
    """
    stripped = (text or "").strip()
    if not stripped:
        return []

    lowered = stripped.lower()
    for connector in _CONNECTORS:
        idx = lowered.find(connector)
        if idx == -1:
            continue

        left = stripped[:idx].strip()
        right = stripped[idx + len(connector):].strip()

        if len(left) < _MIN_SEGMENT_CHARS or len(right) < _MIN_SEGMENT_CHARS:
            continue

        if connector == " and " and not (
            _VERBY_STARTS.match(left) or _VERBY_STARTS.match(right)
        ):
            # Bare "and" with neither side looking like its own request —
            # more likely "bacon and eggs" than two commands. Try the next
            # (less common, so less risky) connector instead of splitting.
            continue

        return [left, right]

    return []


def looks_compound(text: str) -> bool:
    """Cheap pre-check: is it even worth running candidate_segments()?"""
    return bool(candidate_segments(text))
