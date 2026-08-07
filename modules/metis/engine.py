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
from modules.base import BaseModule
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

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CORRECT_PROMPT = """\
You are Metis, a meticulous copy editor.
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
You are Metis, a clarity editor.
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
You are Metis, a writing-style coach.
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
You are Metis, a tone editor.
Rewrite the text below so it reads as {target_tone}, preserving its \
core meaning and factual content.

Text:
{text}

Write only the rewritten text. No explanation, no preamble."""

_REWRITE_PROMPT = """\
You are Metis, a rewriting assistant.
Rewrite the text below to optimise for: {goal}.

Text:
{text}

Write only the rewritten text. No explanation, no preamble."""

_DRAFT_PROMPT = """\
You are Metis, a professional writing assistant.
Draft a {content_type} based on this brief: {brief}
Desired tone: {tone}
Desired length: {length}

Write only the drafted content. No explanation, no preamble."""

_SUMMARY_PROMPT = """\
You are Metis, a summarisation assistant.
Summarise the text below at {length} length, preserving the key points.

Text:
{text}

Write only the summary. No explanation, no preamble."""

_EXPAND_PROMPT = """\
You are Metis, a writing assistant.
Expand the text below with relevant supporting detail, examples, or \
context, roughly {length} in additional length. Keep the original voice.

Text:
{text}

Write only the expanded text. No explanation, no preamble."""

_SHORTEN_PROMPT = """\
You are Metis, a writing assistant.
Shorten the text below to a {length} length while preserving its key \
meaning and tone.

Text:
{text}

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
               metadata: dict) -> None:
        """Fire-and-forget DB write. Never withholds output on failure."""
        _safe_db(
            self.db.save, type_, content,
            title=title,
            input_preview=input_text[:_INPUT_PREVIEW_MAX_LEN],
            input_chars=len(input_text),
            output_chars=len(content),
            metadata=json.dumps(metadata),
        )
        if type_ in _MEMORY_WORTHY_TYPES:
            self._persist(f"{type_}_{title}", content)

    # ------------------------------------------------------------------
    # Intent handlers (private)
    # ------------------------------------------------------------------

    def _correct_text(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        missing = _collect_missing(("text", text, "What text should I correct?"))
        if missing:
            return _clarify(missing)

        try:
            result = self._llm_json(_CORRECT_PROMPT.format(text=text))
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

        return _ok(response, data={"corrected": corrected, "changes": changes},
                    confidence=0.93)

    def _improve_clarity(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        missing = _collect_missing(("text", text, "What text should I clarify?"))
        if missing:
            return _clarify(missing)

        try:
            result = self._llm_json(_CLARITY_PROMPT.format(text=text))
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

        return _ok(response, data={"revised": revised, "notes": notes}, confidence=0.92)

    def _suggest_style(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        missing = _collect_missing(("text", text, "What text should I review for style?"))
        if missing:
            return _clarify(missing)

        try:
            result = self._llm_json(_STYLE_PROMPT.format(text=text))
        except LLMResponseError:
            logger.exception("_suggest_style: LLM call failed.")
            return _err("I had trouble reviewing that. Please try again.")

        suggestions = [s for s in (result.get("suggestions") or []) if isinstance(s, dict)]
        revised = result.get("revised") if isinstance(result, dict) else ""
        response = _format_style_suggestions(suggestions, revised)

        self._save("style", revised or text, "Style review", text,
                    {"num_suggestions": len(suggestions)})

        return _ok(response, data={"suggestions": suggestions, "revised": revised},
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
                    _TONE_SHIFT_PROMPT.format(target_tone=target, text=text)
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

        try:
            rewritten = self._llm_text(_REWRITE_PROMPT.format(goal=goal, text=text))
        except LLMResponseError:
            logger.exception("_rewrite_text: LLM call failed.")
            return _err("I had trouble rewriting that. Please try again.")

        self._save("rewrite", rewritten, f"Rewrite ({goal})", text, {"goal": goal})

        return _ok(rewritten, data={"rewritten": rewritten, "goal": goal}, confidence=0.93)

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

        try:
            draft = self._llm_text(
                _DRAFT_PROMPT.format(
                    content_type=content_type.replace("_", " "),
                    brief=brief, tone=tone, length=length,
                )
            )
        except LLMResponseError:
            logger.exception("_draft_content: LLM call failed.")
            return _err("I had trouble drafting that. Please try again.")

        title = f"{content_type.replace('_', ' ').title()} — {brief[:40]}"

        self._save("draft", draft, title, brief,
                    {"content_type": content_type, "tone": tone, "length": length})

        return _ok(
            f"{title}\n\n{draft}",
            data={"title": title, "draft": draft, "content_type": content_type},
            confidence=0.92,
        )

    def _summarize_text(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        length = _normalise(entities.get("length", ""), _VALID_LENGTHS, "short")
        missing = _collect_missing(("text", text, "What text should I summarise?"))
        if missing:
            return _clarify(missing)

        try:
            summary = self._llm_text(_SUMMARY_PROMPT.format(length=length, text=text))
        except LLMResponseError:
            logger.exception("_summarize_text: LLM call failed.")
            return _err("I had trouble summarising that. Please try again.")

        self._save("summary", summary, "Summary", text, {"length": length})

        return _ok(summary, data={"summary": summary}, confidence=0.93)

    def _expand_text(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        length = _normalise(entities.get("length", ""), _VALID_LENGTHS, _DEFAULT_LENGTH)
        missing = _collect_missing(("text", text, "What text should I expand?"))
        if missing:
            return _clarify(missing)

        try:
            expanded = self._llm_text(_EXPAND_PROMPT.format(length=length, text=text))
        except LLMResponseError:
            logger.exception("_expand_text: LLM call failed.")
            return _err("I had trouble expanding that. Please try again.")

        self._save("expansion", expanded, "Expanded text", text, {"length": length})

        return _ok(expanded, data={"expanded": expanded}, confidence=0.9)

    def _shorten_text(self, entities: dict) -> dict:
        text = _extract(entities, "text", "content", "raw_query")
        length = _normalise(entities.get("length", ""), _VALID_LENGTHS, "short")
        missing = _collect_missing(("text", text, "What text should I shorten?"))
        if missing:
            return _clarify(missing)

        try:
            shortened = self._llm_text(_SHORTEN_PROMPT.format(length=length, text=text))
        except LLMResponseError:
            logger.exception("_shorten_text: LLM call failed.")
            return _err("I had trouble shortening that. Please try again.")

        self._save("shortened", shortened, "Shortened text", text, {"length": length})

        return _ok(shortened, data={"shortened": shortened}, confidence=0.9)

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
        Best-effort only: this module has no live web index to check
        against, so it's honest about that rather than faking a scan.
        Suggests Hephaestus (search_web) as a manual spot-check path for
        any single distinctive phrase the user is worried about.
        """
        text = _extract(entities, "text", "content", "raw_query")
        missing = _collect_missing(("text", text, "What text should I check?"))
        if missing:
            return _clarify(missing)

        response = (
            "I don't have a live web index to run a real plagiarism scan against, "
            "so I can't give you a reliable originality score. If you're worried "
            "about a specific passage, I can search the web for a distinctive "
            "sentence from it to spot-check for matches — or I can help make sure "
            "any borrowed material is properly cited instead."
        )
        self._save("plagiarism_check", response, "Plagiarism check (limited)", text, {})

        return _ok(response, data={"supported": False}, confidence=0.6)

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

        response = _format_readability(result if isinstance(result, dict) else {})

        self._save("readability", response, "Readability report", text, {})

        return _ok(response, data=result if isinstance(result, dict) else {},
                    confidence=0.88)

    def _writing_stats(self) -> dict:
        try:
            stats = self.db.get_stats()
        except Exception:
            logger.exception("_writing_stats: DB read failed.")
            return _err("I couldn't fetch your writing stats right now.")

        response = _format_stats(stats)
        return _ok(response, data=stats, confidence=0.97)


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


def _format_readability(result: dict) -> str:
    ease = result.get("reading_ease", "unknown")
    variety = result.get("sentence_variety", "unknown")
    issues = [i for i in (result.get("issues") or []) if isinstance(i, str)]
    suggestions = [s for s in (result.get("suggestions") or []) if isinstance(s, str)]

    lines = [f"Reading ease: {ease}", f"Sentence variety: {variety}"]
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


def _safe_db(fn: Any, *args: Any, **kwargs: Any) -> None:
    """Call a DB function, logging and swallowing any exception."""
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.exception("DB write failed (fn=%s); writing output unaffected.", fn)


def _ok(response: str, data: Optional[dict[str, Any]] = None,
        confidence: float = 0.9) -> dict[str, Any]:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _err(response: str) -> dict[str, Any]:
    return {"response": response, "data": {}, "confidence": 0.0}