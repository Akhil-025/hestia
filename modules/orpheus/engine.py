"""
modules/orpheus/engine.py

OrpheusEngine: creative writing module for poems, lyrics, short stories,
brainstorming, creative prompt generation, name generation, and working
with text the user already wrote — continuing it in-voice, critiquing it,
or rewriting it in a different style. Also surfaces the module's own
history so past creations can be recalled, not just written once and
forgotten.

Beyond first drafts, Orpheus keeps an append-only version history for every
creation (revise / restore never overwrite the original draft), can export a
piece to a plain-text or Markdown file, and can hand a fresh draft to Metis
for a single critique-and-polish pass before you see it.

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
from core.text_export import (
    DEFAULT_EXPORT_DIR,
    VERSE_TYPES,
    normalise_format,
    render_document,
    safe_stem,
    slugify,
    write_export,
)
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
# Revising rewrites a *saved* piece in full, so (unlike the 4000-char
# clamp on pasted input, which just truncates the prompt) it must never
# silently cut the text: a truncated revision would become the newest
# version. Anything longer is refused instead.
_MAX_REVISE_LEN = 12_000
_NOTE_MAX_LEN = 120
_TRUE_WORDS: frozenset[str] = frozenset({"true", "yes", "y", "on", "1", "enable", "enabled"})
_FALSE_WORDS: frozenset[str] = frozenset({"false", "no", "n", "off", "0", "disable", "disabled", "none"})
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

_REVISE_PROMPT = """\
You are Orpheus, a careful creative editor.
Revise the piece below by applying ONLY this instruction: {instruction}

Change what the instruction implies and nothing else. Keep everything the \
instruction does not touch — the voice, imagery, form, line breaks and \
section labels — exactly as written.

--- CURRENT TEXT ---
{text}
--- END CURRENT TEXT ---

Write only the full revised text. No explanation, no commentary."""


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
    llm:
        Optional pre-built HestiaLLM instance.
    export_dir:
        Directory ``export_creation`` writes into (default ``data/exports``).
    polish_default:
        When True, every new poem/story/lyrics goes through one Metis
        critique-and-polish pass unless the request says ``polish: false``.
        Needs Metis attached via :meth:`attach_metis`; without it the pass
        is skipped and the draft is returned as written.
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
            "revise_creation",
            "get_versions",
            "restore_version",
            "export_creation",
        }
    )

    def __init__(
        self,
        ollama_cfg: Optional[dict[str, Any]] = None,
        memory: Any = None,
        db_path: Optional[Path] = None,
        llm: Optional[Any] = None,
        export_dir: Optional[Path | str] = None,
        polish_default: bool = False,
    ) -> None:
        self._cfg = OllamaConfig.from_dict(ollama_cfg or {})
        self._memory = memory
        self._llm_instance = llm  # HestiaLLM | None — preferred path
        self._metis: Any = None   # attached by main.py; optional
        self._export_dir = Path(export_dir) if export_dir else DEFAULT_EXPORT_DIR
        self._polish_default = bool(polish_default)
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
    # Wiring (called once from main.py)
    # ------------------------------------------------------------------

    def attach_metis(self, metis: Any) -> None:
        """
        Give Orpheus a Metis instance to use for the optional polish pass
        (backlog #270). Any object with ``polish_pass(text, kind=...)``
        works, which keeps this testable without a real Metis.
        """
        self._metis = metis

    def set_polish_default(self, enabled: bool) -> None:
        """Turn the automatic polish pass on or off at runtime."""
        self._polish_default = bool(enabled)

    @property
    def polish_default(self) -> bool:
        return self._polish_default

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
        if intent == "revise_creation":
            return self._revise_creation(entities)
        if intent == "get_versions":
            return self._get_versions(entities)
        if intent == "restore_version":
            return self._restore_version(entities)
        if intent == "export_creation":
            return self._export_creation(entities)
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
    # Finalising a new creation: save v1, optional polish pass (private)
    # ------------------------------------------------------------------

    def _wants_polish(self, entities: dict) -> bool:
        for key in ("polish", "polish_pass"):
            if key in entities and entities.get(key) not in (None, ""):
                return _truthy(entities[key], default=self._polish_default)
        return self._polish_default

    def _finalize_creation(
        self, type_: str, draft: str, title: str, metadata: dict,
        entities: dict,
    ) -> tuple[str, Optional[int], dict]:
        """
        Save *draft* as version 1 and, if asked, run one Metis polish pass.

        The draft is always stored first, so a polished result becomes
        version 2 and the original stays retrievable. Returns
        ``(final_text, creation_id, polish_info)``; *polish_info* is
        ``{"requested": False}`` when no pass was wanted.
        """
        creation_id = _safe_db(
            self.db.save, type_, draft,
            title=title, metadata=json.dumps(metadata),
        )
        info: dict[str, Any] = {"requested": False}
        if not self._wants_polish(entities):
            return draft, creation_id, info

        info = {"requested": True, "applied": False, "changed": False}
        if self._metis is None or not hasattr(self._metis, "polish_pass"):
            info["reason"] = "Metis isn't connected"
            return draft, creation_id, info
        try:
            result = self._metis.polish_pass(draft, kind="creative")
        except Exception:
            logger.exception("_finalize_creation: polish pass raised; keeping draft.")
            info["reason"] = "the polish pass failed"
            return draft, creation_id, info
        if not isinstance(result, dict) or not result.get("ok"):
            info["reason"] = (result or {}).get("reason") if isinstance(result, dict) else None
            info["reason"] = info["reason"] or "the polish pass didn't return anything usable"
            return draft, creation_id, info

        info["applied"] = True
        info["issues"] = list(result.get("issues") or [])
        polished = str(result.get("polished") or "").strip()
        if not result.get("changed") or not polished or polished == draft.strip():
            return draft, creation_id, info

        info["changed"] = True
        if creation_id is not None:
            try:
                version = self.db.add_version(
                    creation_id, polished, note="Metis polish pass",
                )
                info["version"] = version
            except Exception:
                logger.exception("_finalize_creation: could not store polished version.")
                info["changed"] = False
                info["reason"] = "the polished text couldn't be saved"
                return draft, creation_id, info
        return polished, creation_id, info

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

        final, creation_id, polish = self._finalize_creation(
            "poem", poem, title,
            {"style": style, "tone": tone, "topic": topic}, entities,
        )
        self._persist(f"poem_{topic}", final)

        return _ok(
            f"{title}\n\n{final}" + _polish_note(polish),
            data={"title": title, "poem": final, "style": style, "tone": tone,
                  "creation_id": creation_id, "polish": polish},
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

        final, creation_id, polish = self._finalize_creation(
            "lyrics", lyrics, title,
            {"genre": genre, "tone": tone, "rhyme": rhyme,
             "structure": structure, "topic": topic},
            entities,
        )
        self._persist(f"lyrics_{topic}", final)

        return _ok(
            f"{title}\n\n{final}" + _polish_note(polish),
            data={
                "title": title, "lyrics": final,
                "genre": genre, "structure": structure, "rhyme": rhyme,
                "creation_id": creation_id, "polish": polish,
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

        final, creation_id, polish = self._finalize_creation(
            "story", story, title,
            {"genre": genre, "pov": pov, "tone": tone, "topic": topic},
            entities,
        )
        self._persist(f"story_{topic}", final)

        return _ok(
            f"{title}\n\n{final}" + _polish_note(polish),
            data={"title": title, "story": final, "genre": genre, "pov": pov,
                  "tone": tone, "creation_id": creation_id, "polish": polish},
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

    # ------------------------------------------------------------------
    # Version history (#167) and export (#168)
    # ------------------------------------------------------------------

    def _resolve_creation(self, entities: dict) -> tuple[Optional[dict], Optional[dict]]:
        """
        Pick the creation an intent refers to.

        Order: explicit id (``creation_id`` / ``id``) -> newest match for
        the ``type`` / ``keyword`` filters -> newest creation overall.
        Returns ``(row, None)`` or ``(None, error_or_clarify_response)``.
        """
        raw_id = entities.get("creation_id")
        if raw_id in (None, ""):
            raw_id = entities.get("id")
        if raw_id not in (None, ""):
            cid = _to_int(raw_id)
            if cid is None or cid < 1:
                return None, _clarify(
                    ["Which creation do you mean? Give me its number, "
                     "e.g. 'creation 12'."]
                )
            try:
                row = self.db.get(cid)
            except Exception:
                logger.exception("_resolve_creation: DB read failed.")
                return None, _err("I couldn't look that creation up right now.")
            if row is None:
                return None, _ok(
                    f"I can't find a creation numbered {cid}.",
                    data={"found": False}, confidence=0.7,
                )
            return row, None

        raw_type = (entities.get("type") or entities.get("creation_type") or "")
        raw_type = str(raw_type).strip().lower()
        type_ = raw_type if raw_type in _VALID_CREATION_TYPES else None
        keyword = _extract(entities, "keyword", "query", "search")
        try:
            rows = self.db.search(type_=type_, keyword=keyword or None, limit=1)
        except Exception:
            logger.exception("_resolve_creation: DB read failed.")
            return None, _err("I couldn't look that creation up right now.")
        if not rows:
            return None, _ok(
                "I don't have a matching creation saved yet.",
                data={"found": False}, confidence=0.7,
            )
        return rows[0], None

    def _revise_creation(self, entities: dict) -> dict:
        """
        Revise a saved creation as a NEW version; the original is kept.
        """
        instruction = _extract(entities, "instruction", "direction", "change", "raw_query")
        row, problem = self._resolve_creation(entities)
        if problem is not None:
            return problem
        if not instruction:
            return _clarify(["How should I change it?"])

        current = str(row.get("content") or "")
        if len(current) > _MAX_REVISE_LEN:
            return _err(
                "That piece is too long for me to revise in one go without "
                "risking cutting it off. Try revising a section of it instead."
            )

        try:
            revised = self._llm_text(
                _REVISE_PROMPT.format(instruction=instruction, text=current)
            )
        except LLMResponseError:
            logger.exception("_revise_creation: LLM call failed.")
            return _err("I had trouble revising that. Nothing was changed.")

        try:
            version = self.db.add_version(
                row["id"], revised, note=f"revision: {instruction}"[:_NOTE_MAX_LEN],
                metadata=json.dumps({"instruction": instruction}),
            )
        except Exception:
            logger.exception("_revise_creation: could not store new version.")
            return _err("I revised it but couldn't save the new version, so nothing was changed.")

        title = row.get("title") or str(row.get("type", "")).title()
        return _ok(
            f"{title} (version {version})\n\n{revised}\n\n"
            f"Your earlier versions are kept — version 1 is still the original.",
            data={"creation_id": row["id"], "version": version,
                  "previous_version": version - 1, "revised": revised,
                  "title": title},
            confidence=0.9,
        )

    def _get_versions(self, entities: dict) -> dict:
        """List a creation's versions, or show the full text of one."""
        row, problem = self._resolve_creation(entities)
        if problem is not None:
            return problem
        try:
            versions = self.db.get_versions(row["id"])
        except Exception:
            logger.exception("_get_versions: DB read failed.")
            return _err("I couldn't pull up that version history right now.")

        title = row.get("title") or str(row.get("type", "")).title()
        wanted = entities.get("version")
        if wanted not in (None, ""):
            number = _to_int(wanted)
            match = next((v for v in versions if v["version"] == number), None)
            if match is None:
                have = ", ".join(f"v{v['version']}" for v in versions)
                return _ok(
                    f"'{title}' has no version {wanted}. It has: {have}.",
                    data={"creation_id": row["id"], "found": False},
                    confidence=0.7,
                )
            label = _version_label(match)
            return _ok(
                f"{title} — {label}\n\n{match['content']}",
                data={"creation_id": row["id"], "version": match["version"],
                      "content": match["content"]},
                confidence=0.9,
            )

        return _ok(
            _format_versions(title, row["id"], versions),
            data={"creation_id": row["id"], "title": title, "versions": versions,
                  "current_version": row.get("current_version", len(versions))},
            confidence=0.9,
        )

    def _restore_version(self, entities: dict) -> dict:
        """
        Make an earlier version current again — by appending a copy, so
        the history stays complete and nothing is lost.
        """
        row, problem = self._resolve_creation(entities)
        if problem is not None:
            return problem
        number = _to_int(entities.get("version"))
        if number is None or number < 1:
            return _clarify(["Which version should I restore? (e.g. 'version 1')"])
        try:
            target = self.db.get_version(row["id"], number)
            if target is None:
                versions = self.db.get_versions(row["id"])
                have = ", ".join(f"v{v['version']}" for v in versions)
                return _ok(
                    f"There's no version {number} of that piece. It has: {have}.",
                    data={"creation_id": row["id"], "found": False},
                    confidence=0.7,
                )
            if str(target["content"]).strip() == str(row.get("content", "")).strip():
                return _ok(
                    f"Version {number} is already the current text — nothing to restore.",
                    data={"creation_id": row["id"], "version": row.get("current_version"),
                          "restored": False},
                    confidence=0.8,
                )
            new_version = self.db.add_version(
                row["id"], target["content"], note=f"restored from v{number}",
            )
        except Exception:
            logger.exception("_restore_version: DB operation failed.")
            return _err("I couldn't restore that version right now.")

        title = row.get("title") or str(row.get("type", "")).title()
        return _ok(
            f"Restored version {number} of '{title}' as version {new_version}. "
            f"Every earlier version is still in the history.",
            data={"creation_id": row["id"], "restored_from": number,
                  "version": new_version, "restored": True},
            confidence=0.9,
        )

    def _export_creation(self, entities: dict) -> dict:
        """Write a creation to a .md or .txt file inside the export directory."""
        row, problem = self._resolve_creation(entities)
        if problem is not None:
            return problem

        fmt = normalise_format(entities.get("format") or entities.get("file_format"))
        title = row.get("title") or str(row.get("type", "")).title()
        content = str(row.get("content") or "")
        version = row.get("current_version") or 1

        try:
            versions = self.db.get_versions(row["id"])
        except Exception:
            logger.exception("_export_creation: could not read versions.")
            versions = []

        wanted = entities.get("version")
        if wanted not in (None, ""):
            number = _to_int(wanted)
            match = next((v for v in versions if v["version"] == number), None)
            if match is None:
                return _ok(
                    f"'{title}' has no version {wanted} to export.",
                    data={"creation_id": row["id"], "found": False}, confidence=0.7,
                )
            content, version = match["content"], match["version"]

        sections: list[tuple[str, str]] = []
        if _truthy(entities.get("include_history"), default=False) and len(versions) > 1:
            sections.append(("Version history", _format_history_lines(versions)))

        doc = render_document(
            title, content, fmt,
            meta=[("Type", str(row.get("type", ""))),
                  ("Version", str(version)),
                  ("Created", str(row.get("logged_at", "")))],
            sections=sections,
            preserve_lines=str(row.get("type", "")) in VERSE_TYPES,
        )

        requested = _extract(entities, "filename", "file_name", "name")
        stem = safe_stem(requested, "") if requested else ""
        stem = stem or f"orpheus-{row['id']}-{slugify(title, 'creation')}"
        try:
            path = write_export(self._export_dir, stem, fmt, doc)
        except OSError:
            logger.exception("_export_creation: could not write export file.")
            return _err("I couldn't write the export file. Check the export folder is writable.")

        return _ok(
            f"Exported '{title}' (version {version}) to {path}",
            data={"path": str(path), "format": fmt, "creation_id": row["id"],
                  "version": version},
            confidence=0.92,
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


def _truthy(value: Any, default: bool = False) -> bool:
    """
    Interpret a loosely-typed flag from NLU entities ("yes", "off", True).

    Unrecognised values fall back to *default* rather than guessing.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    word = str(value or "").strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    return default


def _to_int(value: Any) -> Optional[int]:
    """Parse "12", 12, "v2" or "version 3" to an int; None if there isn't one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    import re
    m = re.search(r"\d+", str(value or ""))
    return int(m.group()) if m else None


def _polish_note(polish: dict) -> str:
    """One-line footer describing what the optional polish pass did."""
    if not polish or not polish.get("requested"):
        return ""
    if not polish.get("applied"):
        reason = polish.get("reason") or "it couldn't run"
        return f"\n\n(Polish pass skipped: {reason}.)"
    if polish.get("changed"):
        return ("\n\n(Polished by Metis. Your original draft is saved as "
                "version 1 — ask for the version history to see it.)")
    return "\n\n(Metis's polish pass found nothing worth changing.)"


def _version_label(v: dict) -> str:
    label = f"version {v.get('version')}"
    note = (v.get("note") or "").strip()
    return f"{label} ({note})" if note else label


def _format_history_lines(versions: list[dict]) -> str:
    lines = []
    for v in versions:
        note = (v.get("note") or "").strip()
        when = v.get("created_at", "")
        chars = len(str(v.get("content") or ""))
        lines.append(
            f"- v{v.get('version')}"
            + (f" — {note}" if note else "")
            + f" ({when}, {chars} chars)"
        )
    return "\n".join(lines)


def _format_versions(title: str, creation_id: int, versions: list[dict]) -> str:
    lines = [f"Version history — {title} (#{creation_id})", ""]
    for v in versions:
        note = (v.get("note") or "").strip()
        when = v.get("created_at", "")
        chars = len(str(v.get("content") or ""))
        entry = f"  • v{v.get('version')}"
        if note:
            entry += f" — {note}"
        entry += f"  ({when}, {chars} chars)"
        lines.append(entry)
    if len(versions) > 1:
        lines += ["", "Say 'show version N' to read one, or 'restore version N' to bring it back."]
    return "\n".join(lines)


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
        version = r.get("current_version") or 1
        tag = f" v{version}" if version > 1 else ""
        prefix = f"#{r['id']} " if r.get("id") is not None else ""
        lines.append(f"  • {prefix}[{creation_type}] {title}{tag}  ({logged_at})")
    return "\n".join(lines).strip()


def _safe_db(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Call a DB function, logging and swallowing any exception.

    Creative output must never be withheld because the DB write failed.
    Returns the function's result (e.g. the new row id), or None on failure.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception("DB write failed (fn=%s); creative output unaffected.", fn)
        return None


def _ok(
    response: str,
    data: Optional[dict[str, Any]] = None,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict[str, Any]:
    return {"response": response, "data": {}, "confidence": 0.0}