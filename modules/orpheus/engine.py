"""
modules/orpheus/engine.py

OrpheusEngine: creative writing module for poems, lyrics, brainstorming,
and creative prompt generation.

Design notes
------------
- All LLM calls are isolated behind typed helpers that never raise; failures
  produce graceful response dicts rather than propagating exceptions.
- JSON parsing is strict: the raw string is validated against an expected
  schema before being used, so a malformed LLM response never crashes a
  handler.
- Memory persistence is fire-and-forget: a failure there must not affect the
  creative output returned to the user.
- Prompts, default values, and valid option sets are module-level constants
  so they can be audited and changed without touching business logic.
- Every public method conforms to the BaseModule response contract:
  {response: str, data: dict, confidence: float}.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.ollama_client import generate
from modules.base import BaseModule
from .db import OrpheusDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "orpheus" / "orpheus.db"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "mistral"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 11434

_VALID_POEM_STYLES: frozenset[str] = frozenset(
    {"haiku", "sonnet", "free verse", "ballad", "limerick", "ode", "villanelle"}
)
_VALID_TONES: frozenset[str] = frozenset(
    {"melancholic", "joyful", "romantic", "dark", "playful", "reflective",
     "angry", "hopeful", "nostalgic", "humorous"}
)
_VALID_LENGTHS: frozenset[str] = frozenset({"short", "medium", "long"})
_VALID_MEDIUMS: frozenset[str] = frozenset({"writing", "art", "music"})
_VALID_GENRES: frozenset[str] = frozenset(
    {"pop", "hip-hop", "folk", "rock", "indie", "jazz", "classical",
     "country", "r&b", "electronic"}
)
_VALID_RHYME_SCHEMES: frozenset[str] = frozenset(
    {"ABAB", "AABB", "ABBA", "ABCABC", "free", "none"}
)

_DEFAULT_POEM_STYLE = "free verse"
_DEFAULT_TONE = "reflective"
_DEFAULT_LENGTH = "medium"
_DEFAULT_RHYME = "optional"
_DEFAULT_MEDIUM = "writing"
_DEFAULT_GENRE = "pop"
_DEFAULT_RHYME_SCHEME = "ABAB"
_DEFAULT_STRUCTURE = "verse-chorus-verse-chorus-bridge-chorus"
_DEFAULT_PROMPT_COUNT = 5
_MAX_PROMPT_COUNT = 20
_MEMORY_KEY_MAX_LEN = 40
_MEMORY_VALUE_MAX_LEN = 500

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_POEM_PROMPT = """\
You are Orpheus, a master poet.
Write a poem about: {topic}
Style  : {style}
Tone   : {tone}
Length : {length}
Rhyme  : {rhyme}

Write only the poem itself. No title prefix, no explanation.
Start directly with the first line."""

_BRAINSTORM_PROMPT = """\
You are Orpheus, a creative thinking assistant.
Brainstorm ideas for: {topic}

Respond with ONLY valid JSON in this structure:
{{
  "central_idea": "core concept in one sentence",
  "branches": [
    {{
      "theme": "theme name",
      "ideas": ["idea 1", "idea 2", "idea 3"],
      "unexpected_angle": "one surprising or counterintuitive idea"
    }}
  ],
  "cross_connections": ["connection between branch 1 and branch 2"],
  "first_action": "the single best idea to start with right now"
}}

Generate 4-5 branches. Be creative and specific. JSON only."""

_PROMPT_PROMPT = """\
You are Orpheus, a creative catalyst.
Generate {count} creative prompts for: {medium}
Theme or mood: {theme}

Respond with ONLY valid JSON:
{{
  "prompts": [
    {{
      "prompt": "the creative prompt itself",
      "medium": "writing | art | music",
      "difficulty": "easy | medium | challenging"
    }}
  ]
}}
JSON only."""

_LYRICS_PROMPT = """\
You are Orpheus, a master lyricist.
Write lyrics about: {topic}
Structure    : {structure}
Rhyme scheme : {rhyme_scheme}
Tone         : {tone}
Style/Genre  : {genre}

Write the full lyrics with clear section labels (VERSE 1, CHORUS, etc.).
Maintain the rhyme scheme consistently.
Write only the lyrics. No explanation."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OllamaConfig:
    model: str = _DEFAULT_MODEL
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "OllamaConfig":
        return cls(
            model=str(cfg.get("model", _DEFAULT_MODEL)),
            host=str(cfg.get("host", _DEFAULT_HOST)),
            port=int(cfg.get("port", _DEFAULT_PORT)),
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OrpheusError(Exception):
    """Base exception for OrpheusEngine failures."""


class LLMResponseError(OrpheusError):
    """Raised when the LLM returns an empty or unparseable response."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class OrpheusEngine(BaseModule):
    """
    Creative writing module: poems, lyrics, brainstorming, and prompt
    generation, all backed by a local LLM via Ollama.

    Parameters
    ----------
    ollama_cfg:
        Dict with optional keys ``model``, ``host``, ``port``.
    memory:
        Optional Mnemosyne memory engine for persisting notable creations.
    db_path:
        Override the default SQLite database path (useful in tests).
    """

    name = "orpheus"

    _INTENTS: frozenset[str] = frozenset(
        {
            "write_poem",
            "brainstorm",
            "creative_prompt",
            "generate_lyrics",
        }
    )

    def __init__(
        self,
        ollama_cfg: Optional[dict[str, Any]] = None,
        memory: Any = None,
        db_path: Optional[Path] = None,
    ) -> None:
        self._cfg = OllamaConfig.from_dict(ollama_cfg or {})
        self._memory = memory
        resolved = (db_path or _DB_PATH).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.db = OrpheusDB(str(resolved))
        logger.info(
            "OrpheusEngine ready (model=%s, db=%s).", self._cfg.model, resolved
        )

    # ------------------------------------------------------------------
    # BaseModule interface
    # ------------------------------------------------------------------

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        """
        Dispatch an intent to the appropriate handler.

        Never raises; all errors produce a graceful response dict.
        """
        try:
            return self._dispatch(intent, entities)
        except Exception:
            logger.exception(
                "OrpheusEngine.handle() raised for intent=%s.", intent
            )
            return _err("Something went wrong in the creative module.")

    def get_context(self) -> dict:
        """Return a lightweight context snapshot for NLU enrichment."""
        try:
            recent = self.db.get_all(limit=3)
            total = self.db.get_all(limit=10_000)
            return {
                "orpheus_recent_types": [r["type"] for r in recent],
                "orpheus_total_creations": len(total),
            }
        except Exception:
            logger.exception("get_context() failed.")
            return {}

    # ------------------------------------------------------------------
    # Dispatcher (private)
    # ------------------------------------------------------------------

    def _dispatch(self, intent: str, entities: dict) -> dict:
        if intent == "write_poem":
            return self._write_poem(entities)
        if intent == "brainstorm":
            return self._brainstorm(entities)
        if intent == "creative_prompt":
            return self._creative_prompt(entities)
        if intent == "generate_lyrics":
            return self._generate_lyrics(entities)
        return _err(f"Unknown intent: {intent!r}")

    # ------------------------------------------------------------------
    # LLM helpers (private)
    # ------------------------------------------------------------------

    def _llm_text(self, prompt: str) -> str:
        """
        Call the LLM for a plain-text response.

        Raises
        ------
        LLMResponseError
            If the LLM returns an empty string.
        """
        result = generate(
            prompt,
            model=self._cfg.model,
            host=self._cfg.host,
            port=self._cfg.port,
        )
        if not result or not result.strip():
            raise LLMResponseError("LLM returned an empty text response.")
        return result.strip()

    def _llm_json(self, prompt: str) -> dict[str, Any]:
        """
        Call the LLM for a JSON response and parse it.

        Raises
        ------
        LLMResponseError
            If the LLM returns an empty string or invalid JSON.
        """
        raw = generate(
            prompt,
            model=self._cfg.model,
            host=self._cfg.host,
            port=self._cfg.port,
            fmt="json",
        )
        if not raw or not raw.strip():
            raise LLMResponseError("LLM returned an empty JSON response.")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"LLM response is not valid JSON: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Memory persistence (private)
    # ------------------------------------------------------------------

    def _persist(self, key: str, content: str) -> None:
        """
        Persist a notable creation to Mnemosyne memory.

        Fire-and-forget: errors are logged but never propagate.
        """
        if not self._memory:
            return
        try:
            safe_key = (
                "orpheus_"
                + key[:_MEMORY_KEY_MAX_LEN].replace(" ", "_").lower()
            )
            self._memory.learn(safe_key, content[:_MEMORY_VALUE_MAX_LEN])
        except Exception:
            logger.exception("_persist() failed for key=%r; continuing.", key)

    # ------------------------------------------------------------------
    # Intent handlers (private)
    # ------------------------------------------------------------------

    def _write_poem(self, entities: dict) -> dict:
        """Generate a poem and persist it to DB and memory."""
        topic = _extract(entities, "topic", "raw_query")
        style = _normalise(entities.get("style", ""), _VALID_POEM_STYLES, _DEFAULT_POEM_STYLE)
        tone = _normalise(entities.get("tone", ""), _VALID_TONES, _DEFAULT_TONE)
        length = _normalise(entities.get("length", ""), _VALID_LENGTHS, _DEFAULT_LENGTH)
        rhyme = entities.get("rhyme") or _DEFAULT_RHYME

        missing = _collect_missing(
            ("topic", topic, "What should the poem be about?"),
            ("style", style if style != _DEFAULT_POEM_STYLE else "",
             f"Any style preference? ({', '.join(sorted(_VALID_POEM_STYLES))})"),
            ("tone", tone if tone != _DEFAULT_TONE else "",
             f"What tone? ({', '.join(sorted(_VALID_TONES))})"),
            ("length", length if length != _DEFAULT_LENGTH else "",
             "How long? (short / medium / long)"),
        )
        if missing:
            return _clarify(missing)

        try:
            poem = self._llm_text(
                _POEM_PROMPT.format(
                    topic=topic, style=style,
                    tone=tone, length=length, rhyme=rhyme,
                )
            )
        except LLMResponseError:
            logger.exception("_write_poem: LLM call failed.")
            return _err("I had trouble writing that poem. Please try again.")

        title = f"{style.title()} — {topic.title()}"

        _safe_db(
            self.db.save,
            "poem", poem,
            title=title,
            metadata=json.dumps({"style": style, "tone": tone, "topic": topic}),
        )
        self._persist(f"poem_{topic}", poem)

        return _ok(
            f"{title}\n\n{poem}",
            data={"title": title, "poem": poem, "style": style, "tone": tone},
            confidence=0.95,
        )

    def _brainstorm(self, entities: dict) -> dict:
        """Generate a structured brainstorm and format it for display."""
        topic = _extract(entities, "topic", "raw_query")
        if not topic:
            return _ok("What would you like to brainstorm about?", confidence=0.5)

        try:
            result = self._llm_json(_BRAINSTORM_PROMPT.format(topic=topic))
        except LLMResponseError:
            logger.exception("_brainstorm: LLM call failed.")
            return _err("I had trouble brainstorming that. Please try again.")

        if not _validate_brainstorm(result):
            logger.warning("_brainstorm: LLM response failed schema validation.")
            return _err("The brainstorm response was malformed. Please try again.")

        response = _format_brainstorm(topic, result)

        _safe_db(
            self.db.save,
            "brainstorm", response,
            title=f"Brainstorm — {topic}",
            metadata=json.dumps({"topic": topic}),
        )
        self._persist(f"brainstorm_{topic}", result.get("central_idea", ""))

        return _ok(response, data=result, confidence=0.95)

    def _creative_prompt(self, entities: dict) -> dict:
        """Generate a set of creative prompts for a given medium and theme."""
        medium = _normalise(
            entities.get("medium") or entities.get("type", ""),
            _VALID_MEDIUMS,
            _DEFAULT_MEDIUM,
        )
        theme = _extract(entities, "theme", "topic", "raw_query") or "anything"

        try:
            count = int(entities.get("count", _DEFAULT_PROMPT_COUNT))
            count = max(1, min(count, _MAX_PROMPT_COUNT))
        except (ValueError, TypeError):
            count = _DEFAULT_PROMPT_COUNT

        try:
            result = self._llm_json(
                _PROMPT_PROMPT.format(count=count, medium=medium, theme=theme)
            )
        except LLMResponseError:
            logger.exception("_creative_prompt: LLM call failed.")
            return _err("I had trouble generating prompts. Please try again.")

        prompts: list[dict] = result.get("prompts") or []
        if not prompts:
            logger.warning("_creative_prompt: LLM returned no prompts.")
            return _err("The prompt response was empty. Please try again.")

        response = _format_creative_prompts(medium, theme, prompts)

        _safe_db(
            self.db.save,
            "prompt", response,
            title=f"Prompts — {medium} / {theme}",
            metadata=json.dumps({"medium": medium, "theme": theme, "count": count}),
        )

        return _ok(response, data={"prompts": prompts}, confidence=0.95)

    def _generate_lyrics(self, entities: dict) -> dict:
        """Generate song lyrics and persist them to DB and memory."""
        topic = _extract(entities, "topic", "raw_query")
        genre = _normalise(entities.get("genre", ""), _VALID_GENRES, "")
        tone = _normalise(entities.get("tone", ""), _VALID_TONES, "")
        rhyme = _normalise(
            entities.get("rhyme_scheme", ""), _VALID_RHYME_SCHEMES, ""
        )
        structure = entities.get("structure", "").strip()

        missing = _collect_missing(
            ("topic", topic, "What should the song be about?"),
            ("genre", genre,
             f"What genre or style? ({', '.join(sorted(_VALID_GENRES))})"),
            ("structure", structure,
             "What structure? (verse-chorus, verse-chorus-bridge, just a verse)"),
            ("rhyme_scheme", rhyme,
             f"Rhyme scheme? ({', '.join(sorted(_VALID_RHYME_SCHEMES))})"),
        )
        if missing:
            return _clarify(missing)

        # Apply defaults for optional fields that passed the missing check via entities
        genre = genre or _DEFAULT_GENRE
        tone = tone or _DEFAULT_TONE
        rhyme = rhyme or _DEFAULT_RHYME_SCHEME
        structure = structure or _DEFAULT_STRUCTURE

        try:
            lyrics = self._llm_text(
                _LYRICS_PROMPT.format(
                    topic=topic, structure=structure,
                    rhyme_scheme=rhyme, tone=tone, genre=genre,
                )
            )
        except LLMResponseError:
            logger.exception("_generate_lyrics: LLM call failed.")
            return _err("I had trouble writing those lyrics. Please try again.")

        title = f"{genre.title()} — {topic.title()}"

        _safe_db(
            self.db.save,
            "lyrics", lyrics,
            title=title,
            metadata=json.dumps(
                {"genre": genre, "tone": tone,
                 "rhyme": rhyme, "structure": structure, "topic": topic}
            ),
        )
        self._persist(f"lyrics_{topic}", lyrics)

        return _ok(
            f"{title}\n\n{lyrics}",
            data={
                "title": title, "lyrics": lyrics,
                "genre": genre, "structure": structure, "rhyme": rhyme,
            },
            confidence=0.95,
        )


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _extract(entities: dict, *keys: str) -> str:
    """Return the first non-empty value found under any of *keys*."""
    for key in keys:
        value = entities.get(key, "")
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _normalise(value: str, valid: frozenset[str], default: str) -> str:
    """
    Return *value* if it is in *valid* (case-insensitive), else *default*.

    An empty *value* returns *default* without a warning.
    """
    stripped = value.strip().lower()
    if not stripped:
        return default
    # Exact match first
    if stripped in valid:
        return stripped
    # Prefix match (e.g. "folk rock" → "folk")
    for v in valid:
        if stripped.startswith(v) or v.startswith(stripped):
            return v
    logger.debug("_normalise: %r not in valid set; using default %r.", value, default)
    return default


def _collect_missing(*checks: tuple[str, str, str]) -> list[str]:
    """
    Return clarification questions for any field whose value is empty.

    Each *check* is a ``(field_name, current_value, question)`` triple.
    """
    return [question for _, value, question in checks if not value]


def _clarify(questions: list[str]) -> dict:
    return {
        "response": " ".join(questions),
        "data": {"needs_clarification": True},
        "confidence": 0.6,
    }


def _validate_brainstorm(data: Any) -> bool:
    """Return True if *data* has the minimum expected brainstorm structure."""
    if not isinstance(data, dict):
        return False
    if not isinstance(data.get("branches"), list):
        return False
    return True


def _format_brainstorm(topic: str, result: dict) -> str:
    """Render a brainstorm result dict as a readable tree."""
    lines: list[str] = [f"Brainstorm: {topic.title()}", ""]

    central = result.get("central_idea", "")
    if central:
        lines += [f"  ◈ {central}", ""]

    for branch in result.get("branches") or []:
        theme = branch.get("theme") or ""
        lines.append(f"  ┌─ {theme.upper()}")
        for idea in branch.get("ideas") or []:
            lines.append(f"  │   • {idea}")
        unexpected = branch.get("unexpected_angle", "")
        if unexpected:
            lines.append(f"  │   ↯ {unexpected}")
        lines.append("  │")

    connections: list[str] = result.get("cross_connections") or []
    if connections:
        lines.append("  CONNECTIONS")
        for conn in connections:
            lines.append(f"  ↔ {conn}")
        lines.append("")

    first = result.get("first_action", "")
    if first:
        lines += ["  START HERE", f"  → {first}"]

    return "\n".join(lines).strip()


def _format_creative_prompts(
    medium: str, theme: str, prompts: list[dict]
) -> str:
    """Render a list of creative prompt dicts as a numbered display."""
    lines: list[str] = [f"Creative Prompts — {medium.title()} ({theme})", ""]
    for i, p in enumerate(prompts, start=1):
        diff = p.get("difficulty", "")
        med = p.get("medium", medium)
        lines.append(f"  {i}. [{med.upper()} · {diff}]")
        lines.append(f"     {p.get('prompt', '')}")
        lines.append("")
    return "\n".join(lines).strip()


def _safe_db(fn: Any, *args: Any, **kwargs: Any) -> None:
    """
    Call a DB function, logging and swallowing any exception.

    Creative output must never be withheld because the DB write failed.
    """
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.exception("DB write failed (fn=%s); creative output unaffected.", fn)


def _ok(
    response: str,
    data: Optional[dict[str, Any]] = None,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict[str, Any]:
    return {"response": response, "data": {}, "confidence": 0.0}