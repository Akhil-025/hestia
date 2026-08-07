"""
modules/orpheus/engine.py

OrpheusEngine: creative writing module for poems, lyrics, short stories,
brainstorming, creative prompt generation, name generation, and working
with text the user already wrote — continuing it in-voice, critiquing it,
or rewriting it in a different style. Also surfaces the module's own
history so past creations can be recalled, not just written once and
forgotten.

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
_VALID_STORY_GENRES: frozenset[str] = frozenset(
    {"fantasy", "sci-fi", "mystery", "romance", "horror", "adventure",
     "literary", "comedy", "thriller", "fable"}
)
_VALID_NAME_CATEGORIES: frozenset[str] = frozenset(
    {"character", "band", "story title", "pet", "fantasy place",
     "pen name", "business"}
)
_VALID_CREATION_TYPES: frozenset[str] = frozenset(
    {"poem", "lyrics", "brainstorm", "prompt", "story",
     "continuation", "critique", "rewrite", "names"}
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
_DEFAULT_STORY_GENRE = "literary"
_DEFAULT_POV = "third person limited"
_DEFAULT_CONTINUE_LENGTH = "short"
_DEFAULT_NAME_CATEGORY = "character"
_DEFAULT_NAME_COUNT = 8
_MAX_NAME_COUNT = 20
_DEFAULT_CREATIONS_LIMIT = 5
_MAX_CREATIONS_LIMIT = 20
_MAX_INPUT_TEXT_LEN = 4000
_MEMORY_KEY_MAX_LEN = 40
_MEMORY_VALUE_MAX_LEN = 500

# Point-of-view aliases are handled separately from `_normalise()`'s generic
# prefix matching: "third person" is a genuine prefix of BOTH "third person
# limited" and "third person omniscient", so frozenset iteration order could
# make that match non-deterministic. An explicit, ordered alias table avoids
# the ambiguity entirely.
_POV_ALIASES: dict[str, str] = {
    "first": "first person",
    "first person": "first person",
    "1st person": "first person",
    "second": "second person",
    "second person": "second person",
    "2nd person": "second person",
    "omniscient": "third person omniscient",
    "third person omniscient": "third person omniscient",
    "3rd person omniscient": "third person omniscient",
    "limited": "third person limited",
    "third person limited": "third person limited",
    "3rd person limited": "third person limited",
    "third": "third person limited",
    "third person": "third person limited",
    "3rd person": "third person limited",
}

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

_STORY_PROMPT = """\
You are Orpheus, a master storyteller.
Write a short story about: {topic}
Genre  : {genre}
POV    : {pov}
Tone   : {tone}
Length : {length}

Write only the story itself. No title prefix, no explanation.
Start directly with the first line."""

_CONTINUE_PROMPT = """\
You are Orpheus, a creative writing collaborator with a sharp ear for voice \
and style.
Continue the piece of writing below in the SAME voice, tense, tone, and \
style as the original.
Do not repeat, summarize, or restate the original text — write only the \
new continuation.
Extension length: {length}
{direction_line}

--- ORIGINAL TEXT ---
{text}
--- END ORIGINAL TEXT ---

Write only the continuation."""

_CRITIQUE_PROMPT = """\
You are Orpheus, a master poet and a generous but honest writing mentor.
Give constructive creative feedback on the piece of writing below.
{focus_line}

--- TEXT ---
{text}
--- END TEXT ---

Respond with ONLY valid JSON in this structure:
{{
  "overall_impression": "one or two sentence honest reaction",
  "strengths": ["specific strength 1", "specific strength 2"],
  "areas_to_improve": ["specific, actionable suggestion 1", "specific, actionable suggestion 2"],
  "line_to_revisit": "one specific line or phrase from the text worth revising",
  "revision_example": "a rewritten version of that one line, as an example"
}}
JSON only."""

_REWRITE_PROMPT = """\
You are Orpheus, a master of voice and style.
Rewrite the text below to shift its style, PRESERVING its core meaning \
and content.
Target tone : {tone}
{style_line}

--- ORIGINAL TEXT ---
{text}
--- END ORIGINAL TEXT ---

Write only the rewritten text. No explanation, no commentary."""

_NAMES_PROMPT = """\
You are Orpheus, a creative naming consultant.
Generate {count} creative name options.
Category       : {category}
Theme/keywords : {theme}

Respond with ONLY valid JSON:
{{
  "names": [
    {{
      "name": "the name itself",
      "vibe": "one short phrase on the feeling or connotation of this name"
    }}
  ]
}}
JSON only."""


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
            "write_story",
            "continue_writing",
            "critique_writing",
            "rewrite_style",
            "generate_names",
            "get_creations",
        }
    )

    def __init__(
        self,
        ollama_cfg: Optional[dict[str, Any]] = None,
        memory: Any = None,
        db_path: Optional[Path] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self._cfg = OllamaConfig.from_dict(ollama_cfg or {})
        self._memory = memory
        self._llm_instance = llm  # HestiaLLM | None — preferred path
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
        if intent == "write_story":
            return self._write_story(entities)
        if intent == "continue_writing":
            return self._continue_writing(entities)
        if intent == "critique_writing":
            return self._critique_writing(entities)
        if intent == "rewrite_style":
            return self._rewrite_style(entities)
        if intent == "generate_names":
            return self._generate_names(entities)
        if intent == "get_creations":
            return self._get_creations(entities)
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
        if self._llm_instance is not None:
            result = self._llm_instance.generate(prompt)
        else:
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
        if self._llm_instance is not None:
            raw = self._llm_instance.generate(prompt, fmt="json")
        else:
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

        raw_prompts = result.get("prompts") if isinstance(result, dict) else None
        # Drop any non-dict entries defensively — same failure mode as
        # brainstorm's branches: a local LLM occasionally returns a plain
        # string instead of the requested {"prompt": ..., ...} object, which
        # would otherwise crash _format_creative_prompts() on `.get()`.
        prompts: list[dict] = [p for p in (raw_prompts or []) if isinstance(p, dict)]
        if not prompts:
            logger.warning("_creative_prompt: LLM returned no usable prompts.")
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
        genre = _normalise(entities.get("genre", ""), _VALID_GENRES, _DEFAULT_GENRE)
        tone = _normalise(entities.get("tone", ""), _VALID_TONES, _DEFAULT_TONE)
        rhyme = _normalise(
            entities.get("rhyme_scheme", ""), _VALID_RHYME_SCHEMES, _DEFAULT_RHYME_SCHEME
        )
        structure = entities.get("structure", "").strip() or _DEFAULT_STRUCTURE

        missing = _collect_missing(
            ("topic", topic, "What should the song be about?"),
        )
        if missing:
            return _clarify(missing)

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

    def _write_story(self, entities: dict) -> dict:
        """Generate a short story and persist it to DB and memory."""
        topic = _extract(entities, "topic", "raw_query")
        genre = _normalise(entities.get("genre", ""), _VALID_STORY_GENRES, _DEFAULT_STORY_GENRE)
        pov = _normalise_pov(entities.get("pov", "") or entities.get("point_of_view", ""))
        tone = _normalise(entities.get("tone", ""), _VALID_TONES, _DEFAULT_TONE)
        length = _normalise(entities.get("length", ""), _VALID_LENGTHS, _DEFAULT_LENGTH)

        missing = _collect_missing(
            ("topic", topic, "What should the story be about?"),
        )
        if missing:
            return _clarify(missing)

        try:
            story = self._llm_text(
                _STORY_PROMPT.format(
                    topic=topic, genre=genre, pov=pov, tone=tone, length=length,
                )
            )
        except LLMResponseError:
            logger.exception("_write_story: LLM call failed.")
            return _err("I had trouble writing that story. Please try again.")

        title = f"{genre.title()} Story — {topic.title()}"

        _safe_db(
            self.db.save,
            "story", story,
            title=title,
            metadata=json.dumps({"genre": genre, "pov": pov, "tone": tone, "topic": topic}),
        )
        self._persist(f"story_{topic}", story)

        return _ok(
            f"{title}\n\n{story}",
            data={"title": title, "story": story, "genre": genre, "pov": pov, "tone": tone},
            confidence=0.95,
        )

    def _continue_writing(self, entities: dict) -> dict:
        """Continue an existing piece of text in its own voice and style."""
        text = _extract(entities, "text", "content", "raw_query")[:_MAX_INPUT_TEXT_LEN]
        direction = _extract(entities, "direction", "instruction")
        length = _normalise(entities.get("length", ""), _VALID_LENGTHS, _DEFAULT_CONTINUE_LENGTH)

        missing = _collect_missing(
            ("text", text, "What's the text you'd like me to continue?"),
        )
        if missing:
            return _clarify(missing)

        direction_line = f"Direction: {direction}" if direction else ""

        try:
            continuation = self._llm_text(
                _CONTINUE_PROMPT.format(text=text, length=length, direction_line=direction_line)
            )
        except LLMResponseError:
            logger.exception("_continue_writing: LLM call failed.")
            return _err("I had trouble continuing that piece. Please try again.")

        _safe_db(
            self.db.save,
            "continuation", continuation,
            title="Continuation",
            metadata=json.dumps(
                {"length": length, "direction": direction, "original_excerpt": text[:200]}
            ),
        )

        return _ok(
            continuation,
            data={"continuation": continuation, "length": length},
            confidence=0.9,
        )

    def _critique_writing(self, entities: dict) -> dict:
        """Give structured creative feedback on a piece of text."""
        text = _extract(entities, "text", "content", "raw_query")[:_MAX_INPUT_TEXT_LEN]
        focus = _extract(entities, "focus", "aspect")

        missing = _collect_missing(
            ("text", text, "What would you like me to give feedback on?"),
        )
        if missing:
            return _clarify(missing)

        focus_line = f"Focus especially on: {focus}" if focus else ""

        try:
            result = self._llm_json(_CRITIQUE_PROMPT.format(text=text, focus_line=focus_line))
        except LLMResponseError:
            logger.exception("_critique_writing: LLM call failed.")
            return _err("I had trouble reviewing that. Please try again.")

        if not _validate_critique(result):
            logger.warning("_critique_writing: LLM response failed schema validation.")
            return _err("The feedback response was malformed. Please try again.")

        response = _format_critique(result)

        _safe_db(
            self.db.save,
            "critique", response,
            title="Critique",
            metadata=json.dumps({"focus": focus, "text_excerpt": text[:200]}),
        )

        return _ok(response, data=result, confidence=0.9)

    def _rewrite_style(self, entities: dict) -> dict:
        """Rewrite existing text in a different tone/style, same meaning."""
        text = _extract(entities, "text", "content", "raw_query")[:_MAX_INPUT_TEXT_LEN]
        tone = _normalise(entities.get("tone", ""), _VALID_TONES, _DEFAULT_TONE)
        style = _extract(entities, "style")

        missing = _collect_missing(
            ("text", text, "What text would you like me to rewrite?"),
        )
        if missing:
            return _clarify(missing)

        style_line = (
            f"Style       : {style}" if style
            else "Style       : (use your judgement to match the requested tone)"
        )

        try:
            rewritten = self._llm_text(
                _REWRITE_PROMPT.format(text=text, tone=tone, style_line=style_line)
            )
        except LLMResponseError:
            logger.exception("_rewrite_style: LLM call failed.")
            return _err("I had trouble rewriting that. Please try again.")

        _safe_db(
            self.db.save,
            "rewrite", rewritten,
            title=f"Rewrite ({tone})",
            metadata=json.dumps({"tone": tone, "style": style, "original_excerpt": text[:200]}),
        )

        return _ok(
            rewritten,
            data={"rewritten": rewritten, "tone": tone, "style": style},
            confidence=0.9,
        )

    def _generate_names(self, entities: dict) -> dict:
        """Generate a set of creative name options and persist them."""
        category = _normalise(
            entities.get("category") or entities.get("type", ""),
            _VALID_NAME_CATEGORIES, _DEFAULT_NAME_CATEGORY,
        )
        theme = _extract(entities, "theme", "topic", "keywords") or "anything"

        try:
            count = int(entities.get("count", _DEFAULT_NAME_COUNT))
            count = max(1, min(count, _MAX_NAME_COUNT))
        except (ValueError, TypeError):
            count = _DEFAULT_NAME_COUNT

        try:
            result = self._llm_json(
                _NAMES_PROMPT.format(count=count, category=category, theme=theme)
            )
        except LLMResponseError:
            logger.exception("_generate_names: LLM call failed.")
            return _err("I had trouble generating names. Please try again.")

        raw_names = result.get("names") if isinstance(result, dict) else None
        # Same defensive filter as brainstorm/creative_prompt: a local LLM
        # occasionally returns bare strings instead of {"name": ..., ...}.
        names: list[dict] = [
            n for n in (raw_names or []) if isinstance(n, dict) and n.get("name")
        ]
        if not names:
            logger.warning("_generate_names: LLM returned no usable names.")
            return _err("The name list came back empty. Please try again.")

        response = _format_names(category, theme, names)

        _safe_db(
            self.db.save,
            "names", response,
            title=f"Names — {category} / {theme}",
            metadata=json.dumps({"category": category, "theme": theme, "count": count}),
        )

        return _ok(response, data={"names": names}, confidence=0.92)

    def _get_creations(self, entities: dict) -> dict:
        """Recall past creations from Orpheus's own DB, optionally filtered."""
        raw_type = (entities.get("type") or entities.get("creation_type") or "").strip().lower()
        type_ = raw_type if raw_type in _VALID_CREATION_TYPES else None
        if raw_type and type_ is None:
            logger.debug("_get_creations: unrecognised type %r; ignoring filter.", raw_type)

        keyword = _extract(entities, "keyword", "query", "search")

        try:
            limit = int(entities.get("limit", _DEFAULT_CREATIONS_LIMIT))
            limit = max(1, min(limit, _MAX_CREATIONS_LIMIT))
        except (ValueError, TypeError):
            limit = _DEFAULT_CREATIONS_LIMIT

        try:
            rows = self.db.search(type_=type_, keyword=keyword or None, limit=limit)
        except Exception:
            logger.exception("_get_creations: DB read failed.")
            return _err("I couldn't pull up your past creations right now.")

        if not rows:
            return _ok(
                "I don't have any matching creations saved yet.",
                data={"creations": []}, confidence=0.7,
            )

        return _ok(_format_creations(rows), data={"creations": rows}, confidence=0.9)


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


def _normalise_pov(value: str) -> str:
    """
    Resolve a free-form point-of-view entity to a canonical value via
    `_POV_ALIASES`, defaulting to `_DEFAULT_POV` when unrecognised.

    Kept separate from `_normalise()` because "third person" is a genuine
    prefix of two distinct valid values (limited/omniscient); an ordered
    alias table resolves that deterministically where generic frozenset
    prefix-matching could not.
    """
    stripped = value.strip().lower()
    if not stripped:
        return _DEFAULT_POV
    if stripped in _POV_ALIASES:
        return _POV_ALIASES[stripped]
    for alias, canonical in _POV_ALIASES.items():
        if stripped.startswith(alias):
            return canonical
    return _DEFAULT_POV


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
    """Return True if *data* has the minimum expected brainstorm structure.

    Checks each branch is itself a dict, not just that ``branches`` is a
    list — local LLMs occasionally return a list of plain strings instead
    of the requested objects, which used to slip past this check and crash
    inside ``_format_brainstorm`` instead (caught only by the outer
    ``handle()`` try/except, which then discards a perfectly usable
    ``central_idea``/``first_action`` and reports a generic error instead
    of the specific "malformed" message).
    """
    if not isinstance(data, dict):
        return False
    branches = data.get("branches")
    if not isinstance(branches, list) or not branches:
        return False
    return all(isinstance(b, dict) for b in branches)


def _format_brainstorm(topic: str, result: dict) -> str:
    """Render a brainstorm result dict as a readable tree."""
    lines: list[str] = [f"Brainstorm: {topic.title()}", ""]

    central = result.get("central_idea", "")
    if central:
        lines += [f"  ◈ {central}", ""]

    for branch in result.get("branches") or []:
        if not isinstance(branch, dict):
            # Defensive: _validate_brainstorm() should have already rejected
            # this, but formatting stays crash-proof even if a caller skips
            # validation or the schema is loosened later.
            continue
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


def _validate_critique(data: Any) -> bool:
    """
    Return True if *data* has the minimum expected critique structure:
    a dict with at least one non-empty feedback list (strengths and/or
    areas_to_improve). Mirrors `_validate_brainstorm`'s defensive stance
    against a local LLM returning the wrong shape.
    """
    if not isinstance(data, dict):
        return False
    strengths = data.get("strengths")
    improve = data.get("areas_to_improve")
    if not isinstance(strengths, list) or not isinstance(improve, list):
        return False
    return bool(strengths) or bool(improve)


def _format_critique(result: dict) -> str:
    """Render a critique result dict as readable feedback."""
    lines: list[str] = ["Feedback", ""]

    overall = result.get("overall_impression", "")
    if overall:
        lines += [f"  {overall}", ""]

    strengths: list[str] = result.get("strengths") or []
    if strengths:
        lines.append("  STRENGTHS")
        for s in strengths:
            lines.append(f"  + {s}")
        lines.append("")

    improve: list[str] = result.get("areas_to_improve") or []
    if improve:
        lines.append("  TO IMPROVE")
        for i in improve:
            lines.append(f"  - {i}")
        lines.append("")

    line_to_revisit = result.get("line_to_revisit", "")
    revision = result.get("revision_example", "")
    if line_to_revisit and revision:
        lines += ["  TRY REVISING", f'  "{line_to_revisit}"', f"  → {revision}"]

    return "\n".join(lines).strip()


def _format_names(category: str, theme: str, names: list[dict]) -> str:
    """Render a list of name-option dicts as a numbered display."""
    lines: list[str] = [f"Name Ideas — {category.title()} ({theme})", ""]
    for i, n in enumerate(names, start=1):
        vibe = n.get("vibe", "")
        entry = f"  {i}. {n.get('name', '')}"
        if vibe:
            entry += f" — {vibe}"
        lines.append(entry)
    return "\n".join(lines).strip()


def _format_creations(rows: list[dict]) -> str:
    """Render a list of DB creation rows as a readable, dated index."""
    lines: list[str] = ["Past Creations", ""]
    for r in rows:
        creation_type = r.get("type", "")
        title = r.get("title") or creation_type.title()
        logged_at = r.get("logged_at", "")
        lines.append(f"  • [{creation_type}] {title}  ({logged_at})")
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