"""
core/intent_aliases.py

Config-driven phrase aliases for intent classification (backlog #22).

The problem
-----------
Some phrasings are unambiguous to a human and reliably mis-handled by a
small local model — "log my sleep" and "I slept for seven hours" are the
same request, but only one of them looks like the few-shot examples in
``config/nlu_prompt.txt``. The previous answer to this was to keep adding
phrasings to a 41 KB prompt file, which makes every classification call
longer and slower without ever converging.

This module instead resolves known phrasings *before* the LLM is called:
a match short-circuits classification entirely (no Ollama round-trip, no
retry loop), and a miss costs one dict lookup plus a few prefix checks.

Contract
--------
- Aliases live in ``config/intent_aliases.yaml`` — data, not code, so a
  new phrasing is a config edit rather than a prompt-engineering session.
- Every alias target is validated against ``intent_registry.ALL_INTENTS``
  at load time. An alias pointing at a non-existent intent is dropped with
  a warning rather than silently producing an intent Hecate can't route
  and the NLU schema would reject. The registry stays the source of truth.
- Aliases never *override* a confident LLM classification, because they
  run first and only fire on a match; anything unmatched follows the
  normal path unchanged.
- Entity extraction is NOT attempted here. An alias supplies the intent
  only; entities come from the normal pipeline or from the module's own
  parsing of ``raw_query`` (which is how Apollo/Pluto already read
  amounts and durations). Returning a confident intent with empty
  entities is exactly what the existing ``_FAST_INTENTS`` regexes in
  core/nlu.py already do.

Matching rules (in priority order, first match wins):
  1. ``exact``    — the whole normalised query equals the phrase.
  2. ``prefix``   — the query starts with the phrase (so "log my sleep for
                    7 hours" matches the "log my sleep" alias).
  3. ``contains`` — the phrase appears as a whole-word substring. Lowest
                    priority and opt-in per alias, since it's the rule most
                    likely to fire on something it shouldn't.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from modules.hecate.intent_registry import ALL_INTENTS

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("config/intent_aliases.yaml")

# Confidence reported for an alias hit. Deliberately below the 0.98 the
# hardcoded _FAST_INTENTS regexes use (those match a whole utterance
# shape, e.g. "what time is it"), and at/above Hecate's 0.85 high-
# confidence threshold so a Tier-1 registry route is taken as normal.
_ALIAS_CONFIDENCE = 0.92

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalise(text: str) -> str:
    """Lower-case, strip punctuation, and collapse whitespace."""
    if not text:
        return ""
    lowered = _PUNCT_RE.sub(" ", text.lower())
    return _WS_RE.sub(" ", lowered).strip()


class IntentAliasResolver:
    """
    Loads alias definitions once and resolves queries against them.

    Constructed with no arguments in the normal case; tests pass
    ``aliases=`` directly to avoid touching the filesystem.
    """

    def __init__(
        self,
        path: str | Path | None = _DEFAULT_PATH,
        aliases: Optional[dict[str, Any]] = None,
        valid_intents: Optional[frozenset[str]] = None,
    ) -> None:
        self._valid = valid_intents if valid_intents is not None else ALL_INTENTS
        self._exact: dict[str, str] = {}
        self._prefix: list[tuple[str, str]] = []
        self._contains: list[tuple[str, str]] = []
        self.dropped: list[str] = []

        raw = aliases if aliases is not None else self._load(path)
        self._compile(raw or {})

    # -- loading --------------------------------------------------------

    @staticmethod
    def _load(path: str | Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        p = Path(path)
        if not p.exists():
            logger.debug("No intent alias file at %s; alias layer disabled.", p)
            return {}
        try:
            import yaml

            with p.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception as exc:
            logger.warning("Could not read intent aliases from %s: %s", p, exc)
            return {}
        if not isinstance(data, dict):
            logger.warning("%s must be a YAML mapping of intent -> phrases.", p)
            return {}
        return data

    def _compile(self, raw: dict[str, Any]) -> None:
        """
        Accepts either shape per intent:

            apollo_track_sleep:
              - "log my sleep"              # defaults to prefix matching
              - phrase: "i slept for"
                match: prefix
              - phrase: "hours of sleep"
                match: contains
        """
        for intent, entries in raw.items():
            if not isinstance(intent, str):
                continue
            if intent not in self._valid:
                # Registry is the source of truth — see module docstring.
                self.dropped.append(intent)
                logger.warning(
                    "intent_aliases: %r is not a registered intent "
                    "(modules/hecate/intent_registry.py); its %d alias(es) "
                    "were ignored.",
                    intent,
                    len(entries) if isinstance(entries, list) else 1,
                )
                continue
            if not isinstance(entries, list):
                entries = [entries]

            for entry in entries:
                if isinstance(entry, str):
                    phrase, mode = entry, "prefix"
                elif isinstance(entry, dict):
                    phrase = entry.get("phrase", "")
                    mode = str(entry.get("match", "prefix")).lower()
                else:
                    continue

                norm = normalise(str(phrase))
                if not norm:
                    continue
                if mode == "exact":
                    self._exact[norm] = intent
                elif mode == "contains":
                    self._contains.append((norm, intent))
                else:
                    self._prefix.append((norm, intent))

        # Longest phrase first, so "log my sleep quality" wins over
        # "log my sleep" when both are defined.
        self._prefix.sort(key=lambda pair: len(pair[0]), reverse=True)
        self._contains.sort(key=lambda pair: len(pair[0]), reverse=True)

    # -- resolution -----------------------------------------------------

    @property
    def count(self) -> int:
        """Total number of loaded alias phrases."""
        return len(self._exact) + len(self._prefix) + len(self._contains)

    def resolve(self, text: str) -> Optional[str]:
        """Return the aliased intent for *text*, or None if no alias matches."""
        q = normalise(text)
        if not q:
            return None

        hit = self._exact.get(q)
        if hit:
            return hit

        for phrase, intent in self._prefix:
            if q == phrase or q.startswith(phrase + " "):
                return intent

        for phrase, intent in self._contains:
            if re.search(r"\b" + re.escape(phrase) + r"\b", q):
                return intent

        return None

    def resolve_result(self, text: str) -> Optional[dict[str, Any]]:
        """
        Return a full NLU-shaped result for an alias hit, or None.

        Matches the dict shape ``HestiaNLU.understand`` returns so callers
        can use it interchangeably.
        """
        intent = self.resolve(text)
        if intent is None:
            return None
        return {
            "intent": intent,
            "entities": {},
            "response": "",
            "confidence": _ALIAS_CONFIDENCE,
            "source": "alias",
        }
