"""
modules/metis/engine.py

MetisEngine: writing-assistance module (grammar/spelling correction,
clarity, style, tone, rewriting, content drafting, summarising, citation
formatting, consistency checks, readability, and writing stats).

Scope boundary
--------------
Metis owns *editing and utilitarian content generation* — it takes the
user's own text (or a content-generation brief) and returns corrected,
rewritten, or drafted text. It deliberately does NOT own:
  - Poems, lyrics, brainstorming, or creative prompts — that's Orpheus's
    territory (artistic/fictional creative writing). Routing the same verb
    ("write me a ...") to two different gods would be ambiguous, so the
    boundary is: fiction/verse -> Orpheus, everything else -> Metis.
  - Actually sending email or creating calendar events — that's Hermes.
    Metis can *draft* an email's text (`draft_content` with
    content_type="email"), but never transmits anything; the user (or a
    future explicit hand-off to Hermes) does that.
  - True web-scale plagiarism detection — that needs a live index of the
    web, which this module has no access to. `check_plagiarism` is
    intentionally honest about that limitation rather than pretending to
    scan the internet locally.

Writing-workflow features
-------------------------
- ``writing_session`` chains an Orpheus draft into a Metis critique and
  (optionally) a polish pass in one command; ``polish_text`` runs the same
  pass on text you supply. Orpheus can also request the pass itself via
  ``polish_pass()`` (see ``attach_orpheus``).
- ``learn_style`` builds a profile of the user's own voice from pasted
  samples (measured habits plus a short LLM summary). Rewrite / correct /
  clarity / draft / expand / shorten then respect it, unless the request
  says ``use_style: false``.
- shorten / expand / summarise / draft accept exact targets
  (``target_words``, ``max_words``, ``min_words``, ``target_grade``) and
  verify the result, retrying once if it missed.
- ``export_session`` writes a session or any saved item to .md / .txt.
- ``check_plagiarism`` spot-checks distinctive passages against a web
  search when one is attached, and reports the sources it found.

Design notes (mirrors modules/orpheus/engine.py conventions)
--------------------------------------------------------------
- All LLM calls are isolated behind typed helpers that never raise;
  failures produce graceful response dicts rather than propagating
  exceptions.
- JSON parsing is strict: the raw string is validated against an expected
  shape before use, so a malformed LLM response never crashes a handler.
- DB persistence is fire-and-forget: a failure there must never withhold
  the actual writing output from the user.
- Prompts, defaults, and valid option sets are module-level constants so
  they can be audited and changed without touching business logic.
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
from .analysis import (
    LengthTarget,
    compute_style_stats,
    count_words,
    describe_style_stats,
    distinctive_phrases,
    ease_label,
    normalise_for_match,
    parse_length_target,
    text_metrics,
)
from .db import MetisDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "metis" / "metis.db"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "mistral"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 11434

_VALID_TONES: frozenset[str] = frozenset(
    {"friendly", "professional", "confident", "empathetic",
     "diplomatic", "persuasive", "formal", "casual", "neutral"}
)
_VALID_REWRITE_GOALS: frozenset[str] = frozenset(
    {"clarity", "professionalism", "confidence", "politeness",
     "simplicity", "engagement", "formal", "casual"}
)
_VALID_CONTENT_TYPES: frozenset[str] = frozenset(
    {"email", "blog", "report", "social_post", "cover_letter",
     "job_application", "marketing_copy", "product_description",
     "meeting_agenda", "project_plan", "presentation_outline", "essay"}
)
_VALID_CITATION_STYLES: frozenset[str] = frozenset({"apa", "mla", "chicago"})
_VALID_LENGTHS: frozenset[str] = frozenset({"short", "medium", "long"})

_DEFAULT_TONE = "neutral"
_DEFAULT_REWRITE_GOAL = "clarity"
_DEFAULT_CONTENT_TYPE = "email"
_DEFAULT_CITATION_STYLE = "apa"
_DEFAULT_LENGTH = "medium"

_MEMORY_KEY_MAX_LEN = 40
_MEMORY_VALUE_MAX_LEN = 500
_INPUT_PREVIEW_MAX_LEN = 200
# Notes/critique-only intents don't persist a rewritten body worth
# resurfacing later, so only these write to Mnemosyne (fire-and-forget,
# same as Orpheus's _persist for poems/lyrics).
_MEMORY_WORTHY_TYPES: frozenset[str] = frozenset({"rewrite", "draft", "citation"})

# Style profile (#165)
_MIN_STYLE_SAMPLE_WORDS = 40
_MAX_STYLE_SAMPLE_CHARS = 20_000
_MAX_STYLE_SAMPLES = 200
_STYLE_NOTES_INPUT_CHARS = 3_000
_TRUE_WORDS: frozenset[str] = frozenset({"true", "yes", "y", "on", "1", "enable", "enabled"})
_FALSE_WORDS: frozenset[str] = frozenset({"false", "no", "n", "off", "0", "disable", "disabled", "none"})

# Writing session / polish (#164, #270)
_MAX_POLISH_ISSUES = 5
_SESSION_KIND_TO_ORPHEUS: dict[str, str] = {
    "poem": "write_poem",
    "lyrics": "generate_lyrics",
    "story": "write_story",
}
_SESSION_KIND_ALIASES: dict[str, str] = {
    "poem": "poem", "poetry": "poem", "haiku": "poem", "sonnet": "poem",
    "verse": "poem", "limerick": "poem", "ode": "poem",
    "lyrics": "lyrics", "lyric": "lyrics", "song": "lyrics", "songs": "lyrics",
    "story": "story", "stories": "story", "tale": "story", "fable": "story",
    "fiction": "story", "narrative": "story",
}
# Entities forwarded to Orpheus for the draft step.
_SESSION_PASSTHROUGH: tuple[str, ...] = (
    "style", "tone", "length", "rhyme", "genre", "pov", "point_of_view",
    "rhyme_scheme", "structure",
)

# Plagiarism spot-check (#169)
_DEFAULT_PLAGIARISM_PHRASES = 4
_MAX_PLAGIARISM_PHRASES = 6
_MAX_PLAGIARISM_TEXT_CHARS = 20_000
_PLAGIARISM_RESULTS_PER_PHRASE = 3
_MAX_PLAGIARISM_FETCHES = 6

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CORRECT_PROMPT = """\
You are Metis, a meticulous copy editor.{style_block}
Correct grammar, spelling, punctuation, subject-verb agreement, verb tense \
consistency, pronoun usage, and sentence structure in the text below. \
Do not change the meaning, tone, or voice — fix errors only.

Text:
{text}

Respond with ONLY valid JSON in this structure:
{{
  "corrected": "the fully corrected text",
  "changes": [
    {{"type": "grammar|spelling|punctuation|sentence_structure", \
"original": "short original fragment", "suggestion": "short corrected fragment"}}
  ]
}}
If there are no errors, return the text unchanged in "corrected" and an \
empty "changes" list. JSON only."""

_CLARITY_PROMPT = """\
You are Metis, a clarity editor.{style_block}
Simplify complex sentences, remove unnecessary words, and reduce wordiness \
in the text below, without losing meaning.

Text:
{text}

Respond with ONLY valid JSON:
{{
  "revised": "the clearer version of the text",
  "notes": ["short note on one specific simplification made"]
}}
JSON only."""

_STYLE_PROMPT = """\
You are Metis, a writing-style coach.{style_block}
Review the text below for vocabulary, sentence variety, active vs. passive \
voice, word choice, and stylistic consistency.

Text:
{text}

Respond with ONLY valid JSON:
{{
  "suggestions": [
    {{"category": "vocabulary|sentence_variety|voice|word_choice|consistency", \
"issue": "short description", "suggestion": "concrete fix"}}
  ],
  "revised": "the text rewritten applying the suggestions"
}}
JSON only."""

_TONE_DETECT_PROMPT = """\
You are Metis, a tone analyst.
Identify the emotional/professional tone of the text below and suggest the \
single target tone (from: friendly, professional, confident, empathetic, \
diplomatic, persuasive, formal, casual, neutral) that would best serve it \
if the writer wanted to improve reception.

Text:
{text}

Respond with ONLY valid JSON:
{{
  "detected_tone": "the tone as currently written",
  "description": "one sentence describing how it reads",
  "suggested_target": "one of the target tones listed above"
}}
JSON only."""

_TONE_SHIFT_PROMPT = """\
You are Metis, a tone editor.{style_block}
Rewrite the text below so it reads as {target_tone}, preserving its \
core meaning and factual content.

Text:
{text}

Write only the rewritten text. No explanation, no preamble."""

_REWRITE_PROMPT = """\
You are Metis, a rewriting assistant.{style_block}
Rewrite the text below to optimise for: {goal}.

Text:
{text}

Write only the rewritten text. No explanation, no preamble."""

_DRAFT_PROMPT = """\
You are Metis, a professional writing assistant.{style_block}
Draft a {content_type} based on this brief: {brief}
Desired tone: {tone}
Desired length: {length}
{target_block}
Write only the drafted content. No explanation, no preamble."""

_SUMMARY_PROMPT = """\
You are Metis, a summarisation assistant.
Summarise the text below at {length} length, preserving the key points.

Text:
{text}
{target_block}
Write only the summary. No explanation, no preamble."""

_EXPAND_PROMPT = """\
You are Metis, a writing assistant.{style_block}
Expand the text below with relevant supporting detail, examples, or \
context, roughly {length} in additional length. Keep the original voice.

Text:
{text}
{target_block}
Write only the expanded text. No explanation, no preamble."""

_SHORTEN_PROMPT = """\
You are Metis, a writing assistant.{style_block}
Shorten the text below to a {length} length while preserving its key \
meaning and tone.

Text:
{text}
{target_block}
Write only the shortened text. No explanation, no preamble."""

_OUTLINE_PROMPT = """\
You are Metis, an outlining assistant.
Generate an outline for: {topic}
Desired length: {length}

Respond with ONLY valid JSON:
{{
  "title": "outline title",
  "sections": [
    {{"heading": "section heading", "points": ["point 1", "point 2"]}}
  ]
}}
JSON only."""

_CONSISTENCY_PROMPT = """\
You are Metis, a consistency checker.
Check the text below for inconsistencies in capitalization, hyphenation, \
numbers, dates, formatting, and terminology (e.g. the same term spelled \
or capitalised two different ways).

Text:
{text}

Respond with ONLY valid JSON:
{{
  "issues": [
    {{"category": "capitalization|hyphenation|numbers|dates|formatting|terminology", \
"description": "what's inconsistent", "examples": ["example 1", "example 2"]}}
  ]
}}
If there are none, return an empty "issues" list. JSON only."""

_READABILITY_PROMPT = """\
You are Metis, a readability analyst.
Assess the text below for reading ease, sentence length variation, \
paragraph flow, transitions, and redundancy.

Text:
{text}

Respond with ONLY valid JSON:
{{
  "reading_ease": "easy|medium|difficult",
  "sentence_variety": "low|medium|high",
  "issues": ["short issue description"],
  "suggestions": ["short actionable suggestion"]
}}
JSON only."""

_CITATION_PROMPT = """\
You are Metis, a citation assistant.
Generate a citation in {style} style for this source: {source}

Respond with ONLY valid JSON:
{{
  "reference_entry": "the full reference-list / bibliography entry",
  "in_text": "the in-text / parenthetical citation form"
}}
If key details are missing, make reasonable placeholder assumptions and \
note them inside "reference_entry" in [brackets]. JSON only."""

_STYLE_NOTES_PROMPT = """\
You are Metis, a writing-voice analyst.
Read the writing samples below, all by the same person, and describe how \
this person writes so that an editor could imitate their voice. Describe \
habits (sentence rhythm, formality, humour, typical openings, favourite \
constructions, punctuation quirks) — do not summarise what the samples are \
about.

Samples:
{samples}

Respond with ONLY valid JSON:
{{
  "voice_summary": "two sentences describing this person's voice",
  "signature_traits": ["short trait", "short trait"],
  "avoid": ["thing this writer would never do, e.g. 'stiff corporate phrasing'"]
}}
Give 3-6 signature_traits and 0-3 avoid items. JSON only."""

_POLISH_CRITIQUE_PROMPT = """\
You are Metis, a tough but fair editor giving ONE round of feedback.
{kind_note}
Text:
{text}

List only concrete, fixable weaknesses (clichés, vague or flat wording, \
awkward rhythm, repetition, unclear passages, weak opening or ending). Do \
not list nitpicks. If the text is already strong, say so.

Respond with ONLY valid JSON:
{{
  "verdict": "good|needs_work",
  "issues": ["specific issue and where it occurs"]
}}
Give at most {max_issues} issues. If verdict is "good", issues may be empty. JSON only."""

_POLISH_REVISE_PROMPT = """\
You are Metis, an editor applying feedback.{style_block}
{kind_note}
Revise the text below to fix ONLY these issues:
{issues}

Keep everything that isn't a problem exactly as written.

Text:
{text}

Write only the full revised text. No explanation, no preamble."""

_POLISH_KIND_NOTES: dict[str, tuple[str, str]] = {
    "creative": (
        "This is a piece of creative writing (verse, lyrics or fiction). "
        "Judge it as art: protect its voice, imagery, form, line breaks and "
        "any section labels. Do not flatten it into plain prose.",
        "This is creative writing. Preserve its voice, imagery, form, line "
        "breaks and section labels. Do not turn it into plain prose.",
    ),
    "prose": (
        "This is functional writing (email, essay, report, etc.).",
        "This is functional writing. Keep the meaning and facts unchanged.",
    ),
}


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

class MetisError(Exception):
    """Base exception for MetisEngine failures."""


class LLMResponseError(MetisError):
    """Raised when the LLM returns an empty or unparseable response."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MetisEngine(BaseModule):
    """
    Writing-assistance module: correction, clarity, style, tone, rewriting,
    content drafting, summarising/expanding/shortening, outlining, citation
    formatting, consistency checks, readability, and writing stats — all
    backed by a local LLM via Ollama.

    Parameters
    ----------
    ollama_cfg:
        Dict with optional keys ``model``, ``host``, ``port``.
    memory:
        Optional Mnemosyne memory engine for persisting notable output
        (rewrites, drafts, citations).
    db_path:
        Override the default SQLite database path (useful in tests).
    llm:
        Optional pre-built HestiaLLM instance (preferred path, same as
        Orpheus) — falls back to ``core.ollama_client.generate`` if absent.
    export_dir:
        Directory ``export_session`` writes into (default ``data/exports``).
    """

    name = "metis"

    _INTENTS: frozenset[str] = frozenset(
        {
            "correct_text",
            "improve_clarity",
            "suggest_style",
            "detect_tone",
            "rewrite_text",
            "draft_content",
            "summarize_text",
            "expand_text",
            "shorten_text",
            "generate_outline",
            "check_plagiarism",
            "generate_citation",
            "check_consistency",
            "readability_report",
            "writing_stats",
            "polish_text",
            "writing_session",
            "learn_style",
            "show_style_profile",
            "clear_style_profile",
            "export_session",
        }
    )

    def __init__(
        self,
        ollama_cfg: Optional[dict[str, Any]] = None,
        memory: Any = None,
        db_path: Optional[Path] = None,
        llm: Optional[Any] = None,
        export_dir: Optional[Path | str] = None,
    ) -> None:
        self._cfg = OllamaConfig.from_dict(ollama_cfg or {})
        self._memory = memory
        self._llm_instance = llm  # HestiaLLM | None — preferred path
        self._orpheus: Any = None       # attached by main.py; optional
        self._search_fn: Any = None     # plagiarism spot-check; optional
        self._fetch_fn: Any = None
        self._export_dir = Path(export_dir) if export_dir else DEFAULT_EXPORT_DIR
        resolved = (db_path or _DB_PATH).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.db = MetisDB(str(resolved))
        logger.info(
            "MetisEngine ready (model=%s, db=%s).", self._cfg.model, resolved
        )

    # ------------------------------------------------------------------
    # BaseModule interface
    # ------------------------------------------------------------------

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        """
        Process the intent and return a response dict.

        Never raises; all errors produce a graceful response dict.
        """
        try:
            return self._dispatch(intent, entities)
        except Exception:
            logger.exception("MetisEngine.handle() raised for intent=%s.", intent)
            return _err("Something went wrong in the writing module.")

    def get_context(self) -> dict:
        """Return a lightweight context snapshot for NLU enrichment."""
        try:
            recent = self.db.get_all(limit=3)
            stats = self.db.get_stats()
            return {
                "metis_recent_types": [r["type"] for r in recent],
                "metis_total_items": stats.get("total", 0),
            }
        except Exception:
            logger.exception("get_context() failed.")
            return {}

    # ------------------------------------------------------------------
    # Wiring (called once from main.py)
    # ------------------------------------------------------------------

    def attach_orpheus(self, orpheus: Any) -> None:
        """Let ``writing_session`` hand drafting to Orpheus (#164)."""
        self._orpheus = orpheus

    def attach_web_search(self, search_fn: Any, fetch_fn: Any = None) -> None:
        """
        Give ``check_plagiarism`` a way to look passages up (#169).

        *search_fn(query, max_results=N)* must return a list of
        ``{"title": ..., "url": ...}`` dicts (``HestiaBrowserAgent
        .search_web_results`` fits). *fetch_fn(url)* — optional — returns a
        page's text (``get_page_text``) and lets Metis confirm that a
        passage really appears on the page a search returned, instead of
        reporting every search hit as a match.
        """
        self._search_fn = search_fn
        self._fetch_fn = fetch_fn

    # ------------------------------------------------------------------
    # Dispatcher (private)
    # ------------------------------------------------------------------

    def _dispatch(self, intent: str, entities: dict) -> dict:
        if intent == "correct_text":
            return self._correct_text(entities)
        if intent == "improve_clarity":
            return self._improve_clarity(entities)
        if intent == "suggest_style":
            return self._suggest_style(entities)
        if intent == "detect_tone":
            return self._detect_tone(entities)
        if intent == "rewrite_text":
            return self._rewrite_text(entities)
        if intent == "draft_content":
            return self._draft_content(entities)
        if intent == "summarize_text":
            return self._summarize_text(entities)
        if intent == "expand_text":
            return self._expand_text(entities)
        if intent == "shorten_text":
            return self._shorten_text(entities)
        if intent == "generate_outline":
            return self._generate_outline(entities)
        if intent == "check_plagiarism":
            return self._check_plagiarism(entities)
        if intent == "generate_citation":
            return self._generate_citation(entities)
        if intent == "check_consistency":
            return self._check_consistency(entities)
        if intent == "readability_report":
            return self._readability_report(entities)
        if intent == "writing_stats":
            return self._writing_stats()
        if intent == "polish_text":
            return self._polish_text(entities)
        if intent == "writing_session":
            return self._writing_session(entities)
        if intent == "learn_style":
            return self._learn_style(entities)
        if intent == "show_style_profile":
            return self._show_style_profile()
        if intent == "clear_style_profile":
            return self._clear_style_profile()
        if intent == "export_session":
            return self._export_session(entities)
        return _err(f"Unknown intent: {intent!r}")

    # ------------------------------------------------------------------
    # LLM helpers (private) — identical contract to OrpheusEngine's
    # ------------------------------------------------------------------

    def _llm_text(self, prompt: str) -> str:
        if self._llm_instance is not None:
            result = self._llm_instance.generate(prompt)
        else:
            result = generate(
                prompt, model=self._cfg.model,
                host=self._cfg.host, port=self._cfg.port,
            )
        if not result or not result.strip():
            raise LLMResponseError("LLM returned an empty text response.")
        return result.strip()

    def _llm_json(self, prompt: str) -> dict[str, Any]:
        if self._llm_instance is not None:
            raw = self._llm_instance.generate(prompt, fmt="json")
        else:
            raw = generate(
                prompt, model=self._cfg.model,
                host=self._cfg.host, port=self._cfg.port, fmt="json",
            )
        if not raw or not raw.strip():
            raise LLMResponseError("LLM returned an empty JSON response.")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"LLM response is not valid JSON: {exc}") from exc

    # ------------------------------------------------------------------
    # Persistence helpers (private)
    # ------------------------------------------------------------------

    def _persist(self, key: str, content: str) -> None:
        """Fire-and-forget Mnemosyne persistence, same as Orpheus._persist."""
        if not self._memory:
            return
        try:
            safe_key = "metis_" + key[:_MEMORY_KEY_MAX_LEN].replace(" ", "_").lower()
            self._memory.learn(safe_key, content[:_MEMORY_VALUE_MAX_LEN])
        except Exception:
            logger.exception("_persist() failed for key=%r; continuing.", key)

    def _save(self, type_: str, content: str, title: str, input_text: str,
               metadata: dict) -> Optional[int]:
        """
        Fire-and-forget DB write. Never withholds output on failure.

        Returns the new row id (or None if the write failed) so callers
        that need to refer back to the item — sessions, exports — can.
        """
        item_id = _safe_db(
            self.db.save, type_, content,
            title=title,
            input_preview=input_text[:_INPUT_PREVIEW_MAX_LEN],
            input_chars=len(input_text),
            output_chars=len(content),
            metadata=json.dumps(metadata),
        )
        if type_ in _MEMORY_WORTHY_TYPES:
            self._persist(f"{type_}_{title}", content)
        return item_id

    # ------------------------------------------------------------------
    # Style profile helpers (#165)
    # ------------------------------------------------------------------

    def _load_profile(self) -> Optional[dict]:
        """The stored voice profile as a dict, or None (never raises)."""
        try:
            row = self.db.get_style_profile()
            if not row:
                return None
            profile = json.loads(row["profile_json"])
            return profile if isinstance(profile, dict) else None
        except Exception:
            logger.exception("_load_profile() failed; continuing without a style profile.")
            return None

    def _style_block(self, entities: dict) -> str:
        """
        Prompt text asking the model to respect the user's learned voice.

        Empty when there is no profile, or when the request opted out with
        ``use_style: false``. Always safe to interpolate: the templates
        place it directly after the role line.
        """
        if "use_style" in entities and not _truthy(entities.get("use_style"), default=True):
            return ""
        profile = self._load_profile()
        return _style_clause(profile) if profile else ""

    # ------------------------------------------------------------------
    # Length-target helper (#166)
    # ------------------------------------------------------------------

    def _generate_with_target(
        self, prompt: str, target: Optional[LengthTarget]
    ) -> tuple[str, bool, int]:
        """
        Generate text and, if a target is set, check it and retry once.

        Returns ``(text, target_met, attempts)``. The retry tells the model
        exactly how it missed. If the retry is no closer, the first attempt
        is kept. A target that still can't be hit is reported by the caller,
        never hidden and never enforced by truncating the text.
        """
        text = self._llm_text(prompt)
        if target is None or not target.is_set:
            return text, True, 1
        issue = target.violation(text)
        if not issue:
            return text, True, 1

        retry_prompt = (
            f"{prompt}\n\nYour previous attempt was:\n{text}\n\n"
            f"Problem: {issue} Rewrite it so it satisfies every hard "
            f"requirement above. Write only the text."
        )
        try:
            second = self._llm_text(retry_prompt)
        except LLMResponseError:
            logger.warning("_generate_with_target: retry failed; keeping first attempt.")
            return text, False, 1
        best = second if target.score(second) <= target.score(text) else text
        return best, not target.violation(best), 2

    # ------------------------------------------------------------------
    # Intent handlers (private)
    # ------------------------------------------------------------------

    def _correct_text(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        missing = _collect_missing(("text", text, "What text should I correct?"))
        if missing:
            return _clarify(missing)

        style_block = self._style_block(entities)
        try:
            result = self._llm_json(
                _CORRECT_PROMPT.format(text=text, style_block=style_block)
            )
        except LLMResponseError:
            logger.exception("_correct_text: LLM call failed.")
            return _err("I had trouble correcting that. Please try again.")

        corrected = result.get("corrected") if isinstance(result, dict) else None
        if not corrected or not str(corrected).strip():
            logger.warning("_correct_text: LLM returned no corrected text.")
            return _err("The correction came back empty. Please try again.")

        changes = [c for c in (result.get("changes") or []) if isinstance(c, dict)]
        response = _format_corrections(corrected, changes)

        self._save("correction", corrected, "Correction", text,
                    {"num_changes": len(changes)})

        return _ok(response,
                    data={"corrected": corrected, "changes": changes,
                          "style_applied": bool(style_block)},
                    confidence=0.93)

    def _improve_clarity(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        missing = _collect_missing(("text", text, "What text should I clarify?"))
        if missing:
            return _clarify(missing)

        style_block = self._style_block(entities)
        try:
            result = self._llm_json(
                _CLARITY_PROMPT.format(text=text, style_block=style_block)
            )
        except LLMResponseError:
            logger.exception("_improve_clarity: LLM call failed.")
            return _err("I had trouble simplifying that. Please try again.")

        revised = result.get("revised") if isinstance(result, dict) else None
        if not revised or not str(revised).strip():
            return _err("The clarity pass came back empty. Please try again.")

        notes = [n for n in (result.get("notes") or []) if isinstance(n, str)]
        response = revised if not notes else (
            revised + "\n\n" + "\n".join(f"  • {n}" for n in notes)
        )

        self._save("clarity", revised, "Clarity pass", text, {"num_notes": len(notes)})

        return _ok(response,
                    data={"revised": revised, "notes": notes,
                          "style_applied": bool(style_block)},
                    confidence=0.92)

    def _suggest_style(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        missing = _collect_missing(("text", text, "What text should I review for style?"))
        if missing:
            return _clarify(missing)

        style_block = self._style_block(entities)
        try:
            result = self._llm_json(
                _STYLE_PROMPT.format(text=text, style_block=style_block)
            )
        except LLMResponseError:
            logger.exception("_suggest_style: LLM call failed.")
            return _err("I had trouble reviewing that. Please try again.")

        suggestions = [s for s in (result.get("suggestions") or []) if isinstance(s, dict)]
        revised = result.get("revised") if isinstance(result, dict) else ""
        response = _format_style_suggestions(suggestions, revised)

        self._save("style", revised or text, "Style review", text,
                    {"num_suggestions": len(suggestions)})

        return _ok(response,
                    data={"suggestions": suggestions, "revised": revised,
                          "style_applied": bool(style_block)},
                    confidence=0.9)

    def _detect_tone(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        missing = _collect_missing(("text", text, "What text should I analyse the tone of?"))
        if missing:
            return _clarify(missing)

        target = entities.get("target_tone", "")
        target = _normalise(target, _VALID_TONES, "") if target else ""

        try:
            detect = self._llm_json(_TONE_DETECT_PROMPT.format(text=text))
        except LLMResponseError:
            logger.exception("_detect_tone: LLM call failed.")
            return _err("I had trouble reading the tone there. Please try again.")

        detected = detect.get("detected_tone", "unclear") if isinstance(detect, dict) else "unclear"
        description = detect.get("description", "") if isinstance(detect, dict) else ""
        suggested = detect.get("suggested_target", "") if isinstance(detect, dict) else ""

        # Only shift tone when the caller explicitly asked for a target —
        # detection alone shouldn't silently also rewrite the user's text.
        rewritten = ""
        if target:
            try:
                rewritten = self._llm_text(
                    _TONE_SHIFT_PROMPT.format(
                        target_tone=target, text=text,
                        style_block=self._style_block(entities),
                    )
                )
            except LLMResponseError:
                logger.exception("_detect_tone: tone-shift LLM call failed.")

        response = f"Detected tone: {detected}."
        if description:
            response += f" {description}"
        if target and rewritten:
            response += f"\n\nRewritten to sound more {target}:\n\n{rewritten}"
        elif suggested:
            response += f" A more {suggested} tone might land better here."

        self._save("tone", rewritten or detected, "Tone analysis", text,
                    {"detected": detected, "target": target})

        return _ok(
            response,
            data={"detected_tone": detected, "suggested_target": suggested,
                  "rewritten": rewritten},
            confidence=0.9,
        )

    def _rewrite_text(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        goal = _normalise(entities.get("goal", ""), _VALID_REWRITE_GOALS, _DEFAULT_REWRITE_GOAL)
        missing = _collect_missing(("text", text, "What text should I rewrite?"))
        if missing:
            return _clarify(missing)

        style_block = self._style_block(entities)
        try:
            rewritten = self._llm_text(
                _REWRITE_PROMPT.format(goal=goal, text=text, style_block=style_block)
            )
        except LLMResponseError:
            logger.exception("_rewrite_text: LLM call failed.")
            return _err("I had trouble rewriting that. Please try again.")

        self._save("rewrite", rewritten, f"Rewrite ({goal})", text,
                    {"goal": goal, "style_applied": bool(style_block)})

        return _ok(rewritten,
                    data={"rewritten": rewritten, "goal": goal,
                          "style_applied": bool(style_block)},
                    confidence=0.93)

    def _draft_content(self, entities: dict) -> dict:
        brief = _extract(entities, "brief", "topic", "raw_query")
        content_type = _normalise(
            entities.get("content_type") or entities.get("type", ""),
            _VALID_CONTENT_TYPES, _DEFAULT_CONTENT_TYPE,
        )
        tone = _normalise(entities.get("tone", ""), _VALID_TONES, _DEFAULT_TONE)
        length = _normalise(entities.get("length", ""), _VALID_LENGTHS, _DEFAULT_LENGTH)

        missing = _collect_missing(
            ("brief", brief, "What should the content be about?"),
        )
        if missing:
            return _clarify(missing)
        target, target_problem = parse_length_target(entities)
        if target_problem:
            return _clarify([target_problem])

        style_block = self._style_block(entities)
        try:
            draft, met, attempts = self._generate_with_target(
                _DRAFT_PROMPT.format(
                    content_type=content_type.replace("_", " "),
                    brief=brief, tone=tone, length=length,
                    style_block=style_block,
                    target_block=_target_block(target),
                ),
                target,
            )
        except LLMResponseError:
            logger.exception("_draft_content: LLM call failed.")
            return _err("I had trouble drafting that. Please try again.")

        title = f"{content_type.replace('_', ' ').title()} — {brief[:40]}"

        self._save("draft", draft, title, brief,
                    {"content_type": content_type, "tone": tone, "length": length,
                     "style_applied": bool(style_block)})

        return _ok(
            f"{title}\n\n{draft}" + _target_footer(draft, target, met),
            data={"title": title, "draft": draft, "content_type": content_type,
                  "style_applied": bool(style_block),
                  **_target_data(draft, brief, target, met, attempts)},
            confidence=0.92,
        )

    def _summarize_text(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        length = _normalise(entities.get("length", ""), _VALID_LENGTHS, "short")
        missing = _collect_missing(("text", text, "What text should I summarise?"))
        if missing:
            return _clarify(missing)
        target, target_problem = parse_length_target(entities)
        if target_problem:
            return _clarify([target_problem])

        try:
            summary, met, attempts = self._generate_with_target(
                _SUMMARY_PROMPT.format(
                    length=length, text=text, target_block=_target_block(target)
                ),
                target,
            )
        except LLMResponseError:
            logger.exception("_summarize_text: LLM call failed.")
            return _err("I had trouble summarising that. Please try again.")

        self._save("summary", summary, "Summary", text, {"length": length})

        return _ok(summary + _target_footer(summary, target, met),
                    data={"summary": summary,
                          **_target_data(summary, text, target, met, attempts)},
                    confidence=0.93)

    def _expand_text(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        length = _normalise(entities.get("length", ""), _VALID_LENGTHS, _DEFAULT_LENGTH)
        missing = _collect_missing(("text", text, "What text should I expand?"))
        if missing:
            return _clarify(missing)
        target, target_problem = parse_length_target(entities)
        if target_problem:
            return _clarify([target_problem])

        style_block = self._style_block(entities)
        try:
            expanded, met, attempts = self._generate_with_target(
                _EXPAND_PROMPT.format(
                    length=length, text=text, style_block=style_block,
                    target_block=_target_block(target),
                ),
                target,
            )
        except LLMResponseError:
            logger.exception("_expand_text: LLM call failed.")
            return _err("I had trouble expanding that. Please try again.")

        self._save("expansion", expanded, "Expanded text", text, {"length": length})

        return _ok(expanded + _target_footer(expanded, target, met),
                    data={"expanded": expanded, "style_applied": bool(style_block),
                          **_target_data(expanded, text, target, met, attempts)},
                    confidence=0.9)

    def _shorten_text(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        length = _normalise(entities.get("length", ""), _VALID_LENGTHS, "short")
        missing = _collect_missing(("text", text, "What text should I shorten?"))
        if missing:
            return _clarify(missing)
        target, target_problem = parse_length_target(entities)
        if target_problem:
            return _clarify([target_problem])

        style_block = self._style_block(entities)
        try:
            shortened, met, attempts = self._generate_with_target(
                _SHORTEN_PROMPT.format(
                    length=length, text=text, style_block=style_block,
                    target_block=_target_block(target),
                ),
                target,
            )
        except LLMResponseError:
            logger.exception("_shorten_text: LLM call failed.")
            return _err("I had trouble shortening that. Please try again.")

        self._save("shortened", shortened, "Shortened text", text, {"length": length})

        return _ok(shortened + _target_footer(shortened, target, met),
                    data={"shortened": shortened, "style_applied": bool(style_block),
                          **_target_data(shortened, text, target, met, attempts)},
                    confidence=0.9)

    def _generate_outline(self, entities: dict) -> dict:
        topic = _extract(entities, "topic", "raw_query")
        length = _normalise(entities.get("length", ""), _VALID_LENGTHS, _DEFAULT_LENGTH)
        missing = _collect_missing(("topic", topic, "What should the outline be about?"))
        if missing:
            return _clarify(missing)

        try:
            result = self._llm_json(_OUTLINE_PROMPT.format(topic=topic, length=length))
        except LLMResponseError:
            logger.exception("_generate_outline: LLM call failed.")
            return _err("I had trouble outlining that. Please try again.")

        sections = [s for s in (result.get("sections") or []) if isinstance(s, dict)]
        if not sections:
            return _err("The outline came back empty. Please try again.")

        title = result.get("title") or topic.title()
        response = _format_outline(title, sections)

        self._save("outline", response, title, topic, {"num_sections": len(sections)})

        return _ok(response, data={"title": title, "sections": sections}, confidence=0.92)

    def _check_plagiarism(self, entities: dict) -> dict:
        """
        Spot-check a text's most distinctive passages and report sources.

        Metis has no web-scale index, so this never claims an originality
        score. What it does instead (#169):

        - pick a few distinctive passages (uncommon words, names, numbers);
        - if a web search is attached, look each one up as an exact-phrase
          query and, when page fetching is available, confirm the passage
          really appears on the returned page;
        - report the *sources* it found — title, URL and whether the match
          was confirmed — rather than a bare similarity number.

        With no search attached it says so honestly and hands back the
        passages as ready-made quoted queries for a manual check.
        """
        text = _extract(entities, "text", "content", "raw_query")
        missing = _collect_missing(("text", text, "What text should I check?"))
        if missing:
            return _clarify(missing)

        try:
            wanted = int(entities.get("max_phrases", _DEFAULT_PLAGIARISM_PHRASES))
        except (TypeError, ValueError):
            wanted = _DEFAULT_PLAGIARISM_PHRASES
        wanted = max(1, min(wanted, _MAX_PLAGIARISM_PHRASES))
        phrases = distinctive_phrases(text[:_MAX_PLAGIARISM_TEXT_CHARS], k=wanted)

        if self._search_fn is None or not phrases:
            return self._plagiarism_unsupported(text, phrases)

        sources: list[dict[str, Any]] = []
        matched_phrases: set[str] = set()
        checked = 0
        fetches = 0
        any_results = False
        for phrase in phrases:
            try:
                results = self._search_fn(
                    f'"{phrase}"', max_results=_PLAGIARISM_RESULTS_PER_PHRASE
                )
            except Exception:
                logger.exception("_check_plagiarism: search failed for one passage.")
                continue
            checked += 1
            any_results = any_results or bool(results)
            for hit in (results or [])[:_PLAGIARISM_RESULTS_PER_PHRASE]:
                if not isinstance(hit, dict):
                    continue
                url = str(hit.get("url") or "").strip()
                title = str(hit.get("title") or "").strip()
                status = "possible"
                if self._fetch_fn is not None and url and fetches < _MAX_PLAGIARISM_FETCHES:
                    fetches += 1
                    try:
                        page = self._fetch_fn(url) or ""
                        if normalise_for_match(phrase) in normalise_for_match(page):
                            status = "confirmed"
                        elif page.strip():
                            # The engine returned it, but the passage isn't
                            # on the page: a false lead, so don't report it.
                            continue
                    except Exception:
                        logger.debug("_check_plagiarism: could not fetch %r.", url)
                sources.append({"phrase": phrase, "title": title,
                                "url": url, "status": status})
                matched_phrases.add(phrase)

        if checked == 0:
            return self._plagiarism_unsupported(text, phrases, search_failed=True)

        response = _format_plagiarism(checked, sources, matched_phrases)
        if not any_results:
            # Zero hits for every passage is also what an unavailable
            # browser looks like from here, so don't over-claim.
            response += ("\n\nEvery search came back empty. That can mean the "
                         "passages are unique — or that web search isn't working "
                         "right now (e.g. the browser isn't available).")
        self._save("plagiarism_check", response, "Plagiarism spot-check", text,
                    {"checked": checked, "matched": len(matched_phrases)})

        return _ok(
            response,
            data={
                "supported": True,
                "phrases_checked": checked,
                "phrases_matched": len(matched_phrases),
                "match_ratio": round(len(matched_phrases) / checked, 2),
                "sources": sources,
            },
            confidence=0.75,
        )

    def _plagiarism_unsupported(self, text: str, phrases: list[str],
                                search_failed: bool = False) -> dict:
        """Honest fallback: no live search available (or it failed)."""
        response = (
            "I don't have a live web index to run a real plagiarism scan against, "
            "so I can't give you a reliable originality score. If you're worried "
            "about a specific passage, I can search the web for a distinctive "
            "sentence from it to spot-check for matches — or I can help make sure "
            "any borrowed material is properly cited instead."
        )
        if search_failed:
            response = ("I tried to search the web for your text's most distinctive "
                        "passages but the searches didn't work. ") + response
        if phrases:
            response += (
                "\n\nTo check by hand, paste these passages into a search engine "
                "with the quotes:\n" + "\n".join(f'  • "{p}"' for p in phrases)
            )
        self._save("plagiarism_check", response, "Plagiarism check (limited)", text,
                    {"phrases": len(phrases)})
        return _ok(
            response,
            data={"supported": False, "phrases": phrases,
                  "queries": [f'"{p}"' for p in phrases]},
            confidence=0.6,
        )

    def _generate_citation(self, entities: dict) -> dict:
        source = _extract(entities, "source", "raw_query")
        style = _normalise(entities.get("style", ""), _VALID_CITATION_STYLES,
                            _DEFAULT_CITATION_STYLE)
        missing = _collect_missing(("source", source, "What source should I cite?"))
        if missing:
            return _clarify(missing)

        try:
            result = self._llm_json(_CITATION_PROMPT.format(style=style, source=source))
        except LLMResponseError:
            logger.exception("_generate_citation: LLM call failed.")
            return _err("I had trouble generating that citation. Please try again.")

        entry = result.get("reference_entry") if isinstance(result, dict) else None
        in_text = result.get("in_text", "") if isinstance(result, dict) else ""
        if not entry:
            return _err("The citation came back empty. Please try again.")

        response = f"{style.upper()} reference entry:\n{entry}"
        if in_text:
            response += f"\n\nIn-text: {in_text}"

        self._save("citation", entry, f"Citation ({style})", source, {"style": style})

        return _ok(response, data={"reference_entry": entry, "in_text": in_text,
                                     "style": style}, confidence=0.85)

    def _check_consistency(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        missing = _collect_missing(("text", text, "What text should I check for consistency?"))
        if missing:
            return _clarify(missing)

        try:
            result = self._llm_json(_CONSISTENCY_PROMPT.format(text=text))
        except LLMResponseError:
            logger.exception("_check_consistency: LLM call failed.")
            return _err("I had trouble checking that. Please try again.")

        issues = [i for i in (result.get("issues") or []) if isinstance(i, dict)]
        response = _format_consistency(issues)

        self._save("consistency", response, "Consistency check", text,
                    {"num_issues": len(issues)})

        return _ok(response, data={"issues": issues}, confidence=0.88)

    def _readability_report(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        missing = _collect_missing(("text", text, "What text should I assess?"))
        if missing:
            return _clarify(missing)

        try:
            result = self._llm_json(_READABILITY_PROMPT.format(text=text))
        except LLMResponseError:
            logger.exception("_readability_report: LLM call failed.")
            return _err("I had trouble assessing readability. Please try again.")

        metrics = text_metrics(text)
        response = _format_readability(result if isinstance(result, dict) else {}, metrics)

        self._save("readability", response, "Readability report", text, {})

        data = dict(result) if isinstance(result, dict) else {}
        data["metrics"] = metrics
        return _ok(response, data=data, confidence=0.88)

    def _writing_stats(self) -> dict:
        try:
            stats = self.db.get_stats()
        except Exception:
            logger.exception("_writing_stats: DB read failed.")
            return _err("I couldn't fetch your writing stats right now.")

        response = _format_stats(stats)
        return _ok(response, data=stats, confidence=0.97)

    # ------------------------------------------------------------------
    # Polish pass and writing session (#164, #270)
    # ------------------------------------------------------------------

    def polish_pass(
        self,
        text: str,
        kind: str = "prose",
        entities: Optional[dict] = None,
        revise: bool = True,
    ) -> dict[str, Any]:
        """
        One round of critique, then (optionally) one revision.

        This is the public hook Orpheus calls for its polish-pass toggle and
        that ``writing_session`` / ``polish_text`` use. It never raises.

        Returns a dict with:
          ok        False if a step failed (see ``reason``); True otherwise
          polished  the revised text (the input unchanged if nothing to fix)
          changed   whether ``polished`` differs from the input
          issues    the concrete weaknesses the critique found
          verdict   "good" (nothing worth fixing) or "needs_work"

        ``kind="creative"`` tells the model to protect voice, imagery and
        line breaks, and skips the personal style profile: a poem should
        not be edited toward the user's email voice. Only a single round
        is ever run — the critique loop is deliberately not iterative.
        """
        entities = entities or {}
        text = (text or "").strip()
        if not text:
            return {"ok": False, "polished": "", "changed": False,
                    "issues": [], "verdict": "good",
                    "reason": "there was no text to polish"}

        kind = kind if kind in _POLISH_KIND_NOTES else "prose"
        critique_note, revise_note = _POLISH_KIND_NOTES[kind]

        try:
            critique = self._llm_json(
                _POLISH_CRITIQUE_PROMPT.format(
                    kind_note=critique_note, text=text, max_issues=_MAX_POLISH_ISSUES,
                )
            )
        except LLMResponseError:
            logger.exception("polish_pass: critique step failed.")
            return {"ok": False, "polished": text, "changed": False,
                    "issues": [], "verdict": "good",
                    "reason": "the critique step failed"}
        if not isinstance(critique, dict):
            return {"ok": False, "polished": text, "changed": False,
                    "issues": [], "verdict": "good",
                    "reason": "the critique came back malformed"}

        issues = _str_list(critique.get("issues"), _MAX_POLISH_ISSUES)
        # The listed issues are the ground truth: a model that says
        # "needs_work" with nothing concrete gives us nothing to apply.
        verdict = "needs_work" if issues else "good"
        result: dict[str, Any] = {"ok": True, "polished": text, "changed": False,
                                   "issues": issues, "verdict": verdict}
        if not issues or not revise:
            return result

        style_block = self._style_block(entities) if kind == "prose" else ""
        try:
            revised = self._llm_text(
                _POLISH_REVISE_PROMPT.format(
                    style_block=style_block, kind_note=revise_note,
                    issues="\n".join(f"- {i}" for i in issues), text=text,
                )
            )
        except LLMResponseError:
            logger.exception("polish_pass: revision step failed.")
            return {**result, "ok": False, "reason": "the revision step failed"}

        # A revision that gutted or ballooned the piece is the model
        # ignoring "fix only these issues" — keep the original instead.
        before, after = count_words(text), count_words(revised)
        if before >= 10 and not (0.5 * before <= after <= 2.5 * before):
            return {**result, "ok": False,
                    "reason": "the revision drifted too far from the draft, so I kept the original"}

        result["polished"] = revised
        result["changed"] = revised.strip() != text
        return result

    def _polish_text(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        missing = _collect_missing(("text", text, "What text should I polish?"))
        if missing:
            return _clarify(missing)

        kind = str(entities.get("kind", "")).strip().lower()
        kind = kind if kind in _POLISH_KIND_NOTES else "prose"
        result = self.polish_pass(text, kind=kind, entities=entities)
        if not result["ok"]:
            return _err(f"I couldn't polish that: {result.get('reason', 'something went wrong')}.")

        if result["changed"]:
            issues = result["issues"]
            response = result["polished"]
            if issues:
                response += "\n\nWhat I fixed:\n" + "\n".join(f"  • {i}" for i in issues)
        else:
            response = "That already reads well — I'd leave it as it is."

        self._save("polish", result["polished"], "Polish pass", text,
                    {"changed": result["changed"], "num_issues": len(result["issues"]),
                     "kind": kind})
        return _ok(
            response,
            data={"polished": result["polished"], "changed": result["changed"],
                  "issues": result["issues"], "verdict": result["verdict"]},
            confidence=0.9,
        )

    def _writing_session(self, entities: dict) -> dict:
        """
        One command: Orpheus drafts, Metis critiques once and polishes.

        The original Orpheus draft is always kept (as version 1 of the
        creation); the polished text is added as version 2, so the session
        never costs you the draft.
        """
        topic = _extract(entities, "topic", "brief", "subject", "raw_query")
        kind = _session_kind(entities)
        missing = _collect_missing(
            ("topic", topic, "What should it be about?"),
            ("kind", kind, "Should I write a poem, a story, or song lyrics?"),
        )
        if missing:
            return _clarify(missing)

        if self._orpheus is None or not hasattr(self._orpheus, "handle"):
            return _err(
                "The creative writing module isn't available, so I can't start "
                "a writing session right now."
            )

        o_entities: dict[str, Any] = {"topic": topic, "polish": False}
        for key in _SESSION_PASSTHROUGH:
            if entities.get(key):
                o_entities[key] = entities[key]
        try:
            drafted = self._orpheus.handle(_SESSION_KIND_TO_ORPHEUS[kind], o_entities, {})
        except Exception:
            logger.exception("_writing_session: Orpheus draft raised.")
            return _err("I couldn't get a draft started. Please try again.")

        data = drafted.get("data", {}) if isinstance(drafted, dict) else {}
        if data.get("needs_clarification"):
            return drafted
        draft = str(data.get(kind) or "").strip()
        if not draft:
            return _err(
                (drafted.get("response") if isinstance(drafted, dict) else "")
                or "I couldn't get a draft started. Please try again."
            )
        creation_id = data.get("creation_id")
        title = f"Writing session — {kind}: {topic[:40]}"

        do_polish = _truthy(entities.get("polish"), default=True)
        result = self.polish_pass(draft, kind="creative", entities=entities, revise=do_polish)
        issues = result.get("issues", [])
        final = result["polished"] if result.get("ok") and result.get("changed") else draft
        changed = final != draft

        if changed and creation_id is not None:
            try:
                self._orpheus.db.add_version(
                    creation_id, final, note="Metis polish pass (writing session)"
                )
            except Exception:
                logger.exception("_writing_session: could not store polished version.")

        session_id = self._save(
            "session", final, title, topic,
            {"kind": kind, "topic": topic, "draft": draft, "issues": issues,
             "verdict": result.get("verdict", "good"), "polished": changed,
             "orpheus_creation_id": creation_id},
        )

        lines = [title, "", final]
        if issues:
            lines += ["", "Metis's notes:"] + [f"  • {i}" for i in issues]
        lines += ["", _session_status(result, changed, do_polish, creation_id)]
        return _ok(
            "\n".join(lines).strip(),
            data={"session_id": session_id, "title": title, "kind": kind,
                  "topic": topic, "draft": draft, "final": final,
                  "issues": issues, "verdict": result.get("verdict", "good"),
                  "polished": changed, "creation_id": creation_id},
            confidence=0.9,
        )

    # ------------------------------------------------------------------
    # Style profile (#165)
    # ------------------------------------------------------------------

    def _learn_style(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "sample", "raw_query")
        if not text:
            return _clarify([
                "Paste a sample of your own writing — a few paragraphs works "
                "best — and I'll learn your voice from it."
            ])
        text = text[:_MAX_STYLE_SAMPLE_CHARS]
        n_words = count_words(text)
        if n_words < _MIN_STYLE_SAMPLE_WORDS:
            return _ok(
                f"That's only {n_words} words — I need at least "
                f"{_MIN_STYLE_SAMPLE_WORDS} to say anything real about your style. "
                "Paste a longer sample.",
                data={"needs_more": True, "words": n_words}, confidence=0.6,
            )

        try:
            existing = self.db.get_style_samples()
            if len(existing) >= _MAX_STYLE_SAMPLES:
                return _ok(
                    f"I'm already holding {_MAX_STYLE_SAMPLES} samples. Say "
                    "'forget my writing style' to start over.",
                    data={"at_limit": True}, confidence=0.6,
                )
            self.db.add_style_sample(text, label=_extract(entities, "label", "name"),
                                     word_count=n_words)
            samples = self.db.get_style_samples()
        except Exception:
            logger.exception("_learn_style: DB operation failed.")
            return _err("I couldn't save that sample right now.")

        profile, notes_ok = self._build_profile(samples)
        try:
            self.db.save_style_profile(json.dumps(profile), len(samples))
        except Exception:
            logger.exception("_learn_style: could not save profile.")
            return _err("I couldn't save your style profile right now.")

        response = (
            f"Learned from your sample ({n_words} words). "
            f"I now have {len(samples)} sample(s), ~{profile['total_words']} words in total.\n\n"
            + _format_style_profile(profile)
        )
        if not notes_ok:
            response += ("\n\n(I couldn't get a written summary of your voice this "
                         "time, so this is based on measurements only.)")
        return _ok(response, data={"profile": profile, "samples": len(samples),
                                    "summary_available": notes_ok}, confidence=0.9)

    def _build_profile(self, samples: list[dict]) -> tuple[dict, bool]:
        """Measure every sample; add an LLM voice summary from the latest few."""
        texts = [str(x["text"]) for x in samples]
        stats = compute_style_stats(texts)
        profile: dict[str, Any] = {
            "sample_count": len(texts),
            "total_words": stats.get("total_words", 0),
            "stats": stats,
            "traits": describe_style_stats(stats),
            "voice_summary": "",
            "signature_traits": [],
            "avoid": [],
        }
        recent = "\n\n---\n\n".join(texts[-3:])[-_STYLE_NOTES_INPUT_CHARS:]
        try:
            notes = self._llm_json(_STYLE_NOTES_PROMPT.format(samples=recent))
        except LLMResponseError:
            logger.exception("_build_profile: voice-summary call failed.")
            return profile, False
        if not isinstance(notes, dict):
            return profile, False
        profile["voice_summary"] = str(notes.get("voice_summary") or "").strip()[:400]
        profile["signature_traits"] = _str_list(notes.get("signature_traits"), 6)
        profile["avoid"] = _str_list(notes.get("avoid"), 3)
        return profile, bool(profile["voice_summary"] or profile["signature_traits"])

    def _show_style_profile(self) -> dict:
        profile = self._load_profile()
        if not profile:
            return _ok(
                "I haven't learned your writing style yet. Paste a sample of your "
                "own writing and say 'learn my writing style'.",
                data={"has_profile": False}, confidence=0.8,
            )
        return _ok(
            f"Your writing voice (from {profile.get('sample_count', 0)} sample(s), "
            f"~{profile.get('total_words', 0)} words)\n\n" + _format_style_profile(profile)
            + "\n\nI apply this when I correct, rewrite, clarify, draft, expand or "
              "shorten your text. Say 'forget my writing style' to reset it.",
            data={"has_profile": True, "profile": profile}, confidence=0.95,
        )

    def _clear_style_profile(self) -> dict:
        try:
            removed = self.db.clear_style()
        except Exception:
            logger.exception("_clear_style_profile: DB operation failed.")
            return _err("I couldn't clear your style profile right now.")
        if not removed:
            return _ok("There was no style profile to clear.",
                        data={"removed": 0}, confidence=0.8)
        return _ok(f"Done — I've forgotten your writing style and deleted {removed} "
                    "sample(s).", data={"removed": removed}, confidence=0.95)

    # ------------------------------------------------------------------
    # Export (#168)
    # ------------------------------------------------------------------

    def _export_session(self, entities: dict) -> dict:
        """
        Write a writing session (or any saved Metis item) to .md or .txt.

        Picks, in order: the item ``session_id`` / ``id`` names; otherwise
        the latest writing session; otherwise the latest item of any kind.
        """
        raw_id = entities.get("session_id")
        if raw_id in (None, ""):
            raw_id = entities.get("id")
        try:
            if raw_id not in (None, ""):
                item_id = _to_int(raw_id)
                if item_id is None or item_id < 1:
                    return _clarify(["Which session do you mean? Give me its number."])
                item = self.db.get_item(item_id)
                if item is None:
                    return _ok(f"I can't find a saved item numbered {item_id}.",
                                data={"found": False}, confidence=0.7)
            else:
                item = self.db.get_latest("session")
                if item is None:
                    recent = self.db.get_all(limit=1)
                    item = recent[0] if recent else None
                if item is None:
                    return _ok("There's nothing to export yet.",
                                data={"found": False}, confidence=0.7)
        except Exception:
            logger.exception("_export_session: DB read failed.")
            return _err("I couldn't look that up right now.")

        try:
            meta = json.loads(item.get("metadata") or "{}")
            if not isinstance(meta, dict):
                meta = {}
        except (TypeError, ValueError):
            meta = {}

        fmt = normalise_format(entities.get("format") or entities.get("file_format"))
        title = item.get("title") or str(item.get("type", "")).title() or "Metis export"
        kind = str(meta.get("kind") or "")

        sections: list[tuple[str, str]] = []
        if item.get("type") == "session":
            if meta.get("polished") and meta.get("draft"):
                sections.append(("Original draft", str(meta["draft"])))
            issues = _str_list(meta.get("issues"), _MAX_POLISH_ISSUES)
            if issues:
                sections.append(("Metis's notes", "\n".join(f"- {i}" for i in issues)))

        doc = render_document(
            title, str(item.get("content") or ""), fmt,
            meta=[("Type", str(item.get("type", ""))), ("Kind", kind),
                  ("Created", str(item.get("logged_at", "")))],
            sections=sections,
            preserve_lines=kind in VERSE_TYPES or kind == "poem",
        )

        requested = _extract(entities, "filename", "file_name", "name")
        stem = safe_stem(requested, "") if requested else ""
        stem = stem or f"metis-{item['id']}-{slugify(title, 'session')}"
        try:
            path = write_export(self._export_dir, stem, fmt, doc)
        except OSError:
            logger.exception("_export_session: could not write export file.")
            return _err("I couldn't write the export file. Check the export folder is writable.")

        return _ok(f"Exported '{title}' to {path}",
                    data={"path": str(path), "format": fmt, "item_id": item["id"]},
                    confidence=0.92)


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


def _truthy(value: Any, default: bool = False) -> bool:
    """Interpret a loosely-typed flag from NLU entities; unknown -> *default*."""
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
    """Parse 12, "12" or "session 12" to an int; None if there isn't one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    import re
    m = re.search(r"\d+", str(value or ""))
    return int(m.group()) if m else None


def _str_list(value: Any, limit: int) -> list[str]:
    """Keep only non-empty strings from an LLM-supplied list, capped at *limit*."""
    if not isinstance(value, list):
        return []
    out = [str(v).strip() for v in value if isinstance(v, str) and v.strip()]
    return out[:limit]


def _session_kind(entities: dict) -> str:
    """Resolve poem / lyrics / story from entities, or from words in raw_query."""
    for key in ("kind", "type", "form", "medium", "content_type"):
        word = str(entities.get(key, "")).strip().lower()
        if word in _SESSION_KIND_ALIASES:
            return _SESSION_KIND_ALIASES[word]
    import re
    for token in re.findall(r"[a-z]+", str(entities.get("raw_query", "")).lower()):
        if token in _SESSION_KIND_ALIASES:
            return _SESSION_KIND_ALIASES[token]
    return ""


def _session_status(result: dict, changed: bool, do_polish: bool,
                    creation_id: Any) -> str:
    """One-line footer saying what the session did with the draft."""
    ref = f" (creation #{creation_id})" if creation_id is not None else ""
    if not result.get("ok"):
        reason = result.get("reason") or "it couldn't run"
        return f"(Polish pass skipped: {reason}. Draft kept as written{ref}.)"
    if changed:
        return (f"(Polished from the original draft — the draft is saved as "
                f"version 1{ref}.)")
    if not do_polish:
        return f"(Critique only — draft left as written{ref}.)"
    return f"(Metis found nothing worth changing. Draft saved{ref}.)"


def _style_clause(profile: dict) -> str:
    """
    Build the prompt fragment that carries a learned voice profile.

    Starts with a newline and has no trailing newline, so the templates can
    place it right after the role line; returns "" when the profile has
    nothing usable in it.
    """
    summary = str(profile.get("voice_summary") or "").strip()
    traits = _str_list(profile.get("signature_traits"), 6) + _str_list(profile.get("traits"), 8)
    avoid = _str_list(profile.get("avoid"), 3)
    if not (summary or traits):
        return ""
    lines = ["", "Match the user's personal writing voice and do not flatten it "
                 "into a generic style. Only change what the task requires."]
    if summary:
        lines.append(f"Voice: {summary}")
    if traits:
        lines.append("Habits to keep: " + "; ".join(traits) + ".")
    if avoid:
        lines.append("This writer would not: " + "; ".join(avoid) + ".")
    return "\n".join(lines)


def _format_style_profile(profile: dict) -> str:
    lines: list[str] = []
    summary = str(profile.get("voice_summary") or "").strip()
    if summary:
        lines += [summary, ""]
    traits = _str_list(profile.get("traits"), 12)
    if traits:
        lines.append("Measured habits:")
        lines += [f"  • {t}" for t in traits]
    signature = _str_list(profile.get("signature_traits"), 6)
    if signature:
        lines += ["", "Signature traits:"] + [f"  • {t}" for t in signature]
    avoid = _str_list(profile.get("avoid"), 3)
    if avoid:
        lines += ["", "Would not: " + "; ".join(avoid)]
    return "\n".join(lines).strip() or "Not enough text yet to say much."


def _target_block(target: Optional[LengthTarget]) -> str:
    """Prompt fragment for a length target ("" when none)."""
    return target.instructions() if target is not None and target.is_set else ""


def _describe_target(target: LengthTarget) -> str:
    parts: list[str] = []
    if target.target_words is not None:
        parts.append(f"about {target.target_words} words")
    if target.min_words is not None:
        parts.append(f"at least {target.min_words} words")
    if target.max_words is not None:
        parts.append(f"at most {target.max_words} words")
    if target.grade is not None:
        parts.append(f"grade {target.grade:g} reading level")
    return " and ".join(parts)


def _target_footer(text: str, target: Optional[LengthTarget], met: bool = True) -> str:
    """Short, honest note on whether a requested target was hit."""
    if target is None or not target.is_set:
        return ""
    m = text_metrics(text)
    measured = f"{m['words']} words"
    if target.grade is not None:
        measured += f", grade {m['grade']:g}"
    if met:
        return f"\n\n({measured} — target met.)"
    return (f"\n\n({measured} — I couldn't quite reach {_describe_target(target)}; "
            "ask me to try again or loosen the target.)")


def _target_data(output: str, source: str, target: Optional[LengthTarget],
                 met: bool, attempts: int) -> dict[str, Any]:
    """Structured target results for `data` (empty when no target was set)."""
    if target is None or not target.is_set:
        return {}
    m = text_metrics(output)
    return {
        "target": target.as_dict(),
        "target_met": met,
        "attempts": attempts,
        "output_words": m["words"],
        "input_words": count_words(source),
        "reading_grade": m["grade"],
    }


def _format_plagiarism(checked: int, sources: list[dict],
                       matched: set[str]) -> str:
    """Render the spot-check outcome, sources first, with honest caveats."""
    caveat = (
        "This only samples a few passages — it isn't a full scan, so no matches "
        "doesn't prove the text is original, and a match can just be a quotation "
        "or a common phrase. If anything here is borrowed, cite it (I can format "
        "the citation for you)."
    )
    if not sources:
        return (f"I spot-checked {checked} distinctive passage(s) against a web "
                f"search and found no matching pages.\n\n{caveat}")
    lines = [f"I spot-checked {checked} distinctive passage(s) against a web "
             f"search; {len(matched)} of {checked} turned up elsewhere:", ""]
    by_phrase: dict[str, list[dict]] = {}
    for src in sources:
        by_phrase.setdefault(src["phrase"], []).append(src)
    for phrase, hits in by_phrase.items():
        lines.append(f'  “{phrase}”')
        for hit in hits:
            label = hit.get("title") or hit.get("url") or "(untitled)"
            status = "confirmed on the page" if hit["status"] == "confirmed" else \
                "search match, not verified"
            url = f" — {hit['url']}" if hit.get("url") else ""
            lines.append(f"      → {label}{url} ({status})")
    lines += ["", caveat]
    return "\n".join(lines)


def _normalise(value: str, valid: frozenset[str], default: str) -> str:
    """Return *value* if it is in *valid* (case-insensitive), else *default*."""
    stripped = str(value).strip().lower()
    if not stripped:
        return default
    if stripped in valid:
        return stripped
    for v in valid:
        if stripped.startswith(v) or v.startswith(stripped):
            return v
    logger.debug("_normalise: %r not in valid set; using default %r.", value, default)
    return default


def _collect_missing(*checks: tuple[str, str, str]) -> list[str]:
    return [question for _, value, question in checks if not value]


def _clarify(questions: list[str]) -> dict:
    return {
        "response": " ".join(questions),
        "data": {"needs_clarification": True},
        "confidence": 0.6,
    }


def _format_corrections(corrected: str, changes: list[dict]) -> str:
    if not changes:
        return f"No errors found.\n\n{corrected}"
    lines = [corrected, "", f"Changes ({len(changes)}):"]
    for c in changes:
        ctype = c.get("type", "edit")
        original = c.get("original", "")
        suggestion = c.get("suggestion", "")
        lines.append(f"  • [{ctype}] \"{original}\" → \"{suggestion}\"")
    return "\n".join(lines)


def _format_style_suggestions(suggestions: list[dict], revised: str) -> str:
    if not suggestions:
        return revised or "No style issues found."
    lines = ["Style suggestions:"]
    for s in suggestions:
        category = s.get("category", "style")
        issue = s.get("issue", "")
        fix = s.get("suggestion", "")
        lines.append(f"  • [{category}] {issue} → {fix}")
    if revised:
        lines += ["", "Revised:", revised]
    return "\n".join(lines)


def _format_outline(title: str, sections: list[dict]) -> str:
    lines = [title, ""]
    for i, section in enumerate(sections, start=1):
        heading = section.get("heading", f"Section {i}")
        lines.append(f"{i}. {heading}")
        for point in section.get("points") or []:
            lines.append(f"   - {point}")
    return "\n".join(lines).strip()


def _format_consistency(issues: list[dict]) -> str:
    if not issues:
        return "No consistency issues found."
    lines = [f"Consistency issues ({len(issues)}):"]
    for issue in issues:
        category = issue.get("category", "general")
        description = issue.get("description", "")
        examples = issue.get("examples") or []
        lines.append(f"  • [{category}] {description}")
        for ex in examples:
            lines.append(f"      e.g. {ex}")
    return "\n".join(lines)


def _format_readability(result: dict, metrics: Optional[dict] = None) -> str:
    ease = result.get("reading_ease", "unknown")
    variety = result.get("sentence_variety", "unknown")
    issues = [i for i in (result.get("issues") or []) if isinstance(i, str)]
    suggestions = [s for s in (result.get("suggestions") or []) if isinstance(s, str)]

    lines: list[str] = []
    if metrics and metrics.get("words"):
        lines += [
            f"Measured: {metrics['words']} words, {metrics['sentences']} sentence(s), "
            f"{metrics['avg_sentence_words']:g} words per sentence on average",
            f"Flesch reading ease {metrics['reading_ease']:g} "
            f"({ease_label(metrics['reading_ease'])}), grade level ≈ {metrics['grade']:g}",
            "",
        ]
    lines += [f"Reading ease: {ease}", f"Sentence variety: {variety}"]
    if issues:
        lines.append("")
        lines.append("Issues:")
        lines += [f"  • {i}" for i in issues]
    if suggestions:
        lines.append("")
        lines.append("Suggestions:")
        lines += [f"  • {s}" for s in suggestions]
    return "\n".join(lines)


def _format_stats(stats: dict) -> str:
    total = stats.get("total", 0)
    by_type = stats.get("by_type", {})
    if not total:
        return "No writing activity logged yet."
    lines = [f"Total writing items: {total}", ""]
    for type_, counts in sorted(by_type.items()):
        lines.append(
            f"  • {type_}: {counts['count']} "
            f"(≈{counts['input_chars']} chars in / {counts['output_chars']} chars out)"
        )
    return "\n".join(lines)


def _safe_db(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Call a DB function, logging and swallowing any exception.

    Returns the function's result (e.g. the new row id), or None on failure.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception("DB write failed (fn=%s); writing output unaffected.", fn)
        return None


def _ok(response: str, data: Optional[dict[str, Any]] = None,
        confidence: float = 0.9) -> dict[str, Any]:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict[str, Any]:
    return {"response": response, "data": {}, "confidence": 0.0}