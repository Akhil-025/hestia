# modules/orpheus/engine.py

import json
import logging
import os
from pathlib import Path
from typing import Optional

from core.ollama_client import generate
from modules.base import BaseModule
from .db import OrpheusDB

log = logging.getLogger(__name__)

_DB_PATH = str(Path(__file__).parent.parent.parent / "data" / "orpheus" / "orpheus.db")

# ── Prompts ───────────────────────────────────────────────────────────────────

_POEM_PROMPT = """You are Orpheus, a master poet.
Write a poem about: {topic}
Style  : {style}
Tone   : {tone}
Length : {length}
Rhyme  : {rhyme}

Write only the poem itself. No title prefix, no explanation.
Start directly with the first line."""

_BRAINSTORM_PROMPT = """You are Orpheus, a creative thinking assistant.
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
  "cross_connections": ["connection between branch 1 and branch 2", "..."],
  "first_action": "the single best idea to start with right now"
}}

Generate 4-5 branches. Be creative and specific. JSON only."""

_PROMPT_PROMPT = """You are Orpheus, a creative catalyst.
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

_LYRICS_PROMPT = """You are Orpheus, a master lyricist.
Write lyrics about: {topic}
Structure : {structure}
Rhyme scheme: {rhyme_scheme}
Tone      : {tone}
Style/Genre: {genre}

Write the full lyrics with clear section labels (VERSE 1, CHORUS, etc.).
Maintain the rhyme scheme consistently.
Write only the lyrics. No explanation."""


class OrpheusEngine(BaseModule):
    name = "orpheus"
    _INTENTS = {
        "write_poem",
        "brainstorm",
        "creative_prompt",
        "generate_lyrics",
    }

    def __init__(self, ollama_cfg: dict = None, memory=None):
        self._ollama = ollama_cfg or {}
        self._memory = memory
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        self.db = OrpheusDB(_DB_PATH)

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        if intent == "write_poem":
            return self._write_poem(entities)
        if intent == "brainstorm":
            return self._brainstorm(entities)
        if intent == "creative_prompt":
            return self._creative_prompt(entities)
        if intent == "generate_lyrics":
            return self._generate_lyrics(entities)
        return {"response": "Unknown Orpheus intent.", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        try:
            recent = self.db.get_all(limit=3)
            return {
                "orpheus_recent_types": [r["type"] for r in recent],
                "orpheus_total_creations": len(self.db.get_all(limit=1000)),
            }
        except Exception:
            return {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ollama_text(self, prompt: str) -> str:
        return generate(
            prompt,
            model=self._ollama.get("model", "mistral"),
            host=self._ollama.get("host", "127.0.0.1"),
            port=self._ollama.get("port", 11434),
        )

    def _ollama_json(self, prompt: str) -> str:
        return generate(
            prompt,
            model=self._ollama.get("model", "mistral"),
            host=self._ollama.get("host", "127.0.0.1"),
            port=self._ollama.get("port", 11434),
            fmt="json",
        )

    def _parse(self, raw: str, intent: str) -> Optional[dict]:
        try:
            return json.loads(raw)
        except Exception:
            log.warning("Orpheus: failed to parse JSON for %s", intent)
            return None

    def _persist_to_memory(self, key: str, content: str) -> None:
        if self._memory:
            safe_key = f"orpheus_{key[:40].replace(' ', '_').lower()}"
            self._memory.learn(safe_key, content[:500])

    # ── write_poem ────────────────────────────────────────────────────────────

    def _write_poem(self, entities: dict) -> dict:
        topic  = entities.get("topic") or entities.get("raw_query", "")
        style  = entities.get("style", "")
        tone   = entities.get("tone", "")
        length = entities.get("length", "")
        rhyme  = entities.get("rhyme", "")

        # clarifying questions if missing
        missing = []
        if not topic:
            missing.append("What should the poem be about?")
        if not style:
            missing.append("Any style preference? (haiku, sonnet, free verse, ballad)")
        if not tone:
            missing.append("What tone? (melancholic, joyful, romantic, dark, playful)")
        if not length:
            missing.append("How long? (short / medium / long)")

        if missing:
            return {
                "response": " ".join(missing),
                "data": {"needs_clarification": True},
                "confidence": 0.6,
            }

        # defaults
        style  = style  or "free verse"
        tone   = tone   or "reflective"
        length = length or "medium"
        rhyme  = rhyme  or "optional"

        poem = self._ollama_text(
            _POEM_PROMPT.format(
                topic=topic, style=style,
                tone=tone, length=length, rhyme=rhyme,
            )
        ).strip()

        if not poem:
            return {"response": "I had trouble writing that poem.", "data": {}, "confidence": 0.3}

        title = f"{style.title()} — {topic.title()}"

        # save to DB and Mnemosyne
        self.db.save("poem", poem, title=title,
                     metadata=json.dumps({"style": style, "tone": tone, "topic": topic}))
        self._persist_to_memory(f"poem_{topic}", poem)

        response = f"{title}\n\n{poem}"

        return {
            "response": response,
            "data": {"title": title, "poem": poem, "style": style, "tone": tone},
            "confidence": 0.95,
        }

    # ── brainstorm ────────────────────────────────────────────────────────────

    def _brainstorm(self, entities: dict) -> dict:
        topic = (
            entities.get("topic")
            or entities.get("raw_query", "")
        )

        if not topic:
            return {
                "response": "What would you like to brainstorm about?",
                "data": {},
                "confidence": 0.5,
            }

        raw    = self._ollama_json(_BRAINSTORM_PROMPT.format(topic=topic))
        result = self._parse(raw, "brainstorm")

        if not result:
            return {"response": "I had trouble brainstorming that.", "data": {}, "confidence": 0.3}

        response = self._format_brainstorm(topic, result)

        self.db.save("brainstorm", response, title=f"Brainstorm — {topic}",
                     metadata=json.dumps({"topic": topic}))
        self._persist_to_memory(f"brainstorm_{topic}", result.get("central_idea", ""))

        return {
            "response": response,
            "data": result,
            "confidence": 0.95,
        }

    @staticmethod
    def _format_brainstorm(topic: str, result: dict) -> str:
        lines = [f"Brainstorm: {topic.title()}", ""]

        central = result.get("central_idea", "")
        if central:
            lines += [f"  ◈ {central}", ""]

        for branch in result.get("branches", []):
            theme = branch.get("theme", "")
            lines.append(f"  ┌─ {theme.upper()}")
            for idea in branch.get("ideas", []):
                lines.append(f"  │   • {idea}")
            unexpected = branch.get("unexpected_angle", "")
            if unexpected:
                lines.append(f"  │   ↯ {unexpected}")
            lines.append("  │")

        connections = result.get("cross_connections", [])
        if connections:
            lines.append("  CONNECTIONS")
            for c in connections:
                lines.append(f"  ↔ {c}")
            lines.append("")

        first = result.get("first_action", "")
        if first:
            lines += ["  START HERE", f"  → {first}"]

        return "\n".join(lines).strip()

    # ── creative_prompt ───────────────────────────────────────────────────────

    def _creative_prompt(self, entities: dict) -> dict:
        medium = (
            entities.get("medium")
            or entities.get("type", "writing")
        )
        theme = (
            entities.get("theme")
            or entities.get("topic")
            or entities.get("raw_query", "anything")
        )
        count = int(entities.get("count", 5))

        # validate medium
        valid_mediums = {"writing", "art", "music"}
        if medium.lower() not in valid_mediums:
            medium = "writing"

        raw    = self._ollama_json(
            _PROMPT_PROMPT.format(
                count=count, medium=medium, theme=theme
            )
        )
        result = self._parse(raw, "creative_prompt")

        if not result:
            return {"response": "I had trouble generating prompts.", "data": {}, "confidence": 0.3}

        prompts = result.get("prompts", [])
        lines   = [f"Creative Prompts — {medium.title()} ({theme})\n"]

        for i, p in enumerate(prompts, 1):
            diff = p.get("difficulty", "")
            med  = p.get("medium", medium)
            lines.append(f"  {i}. [{med.upper()} · {diff}]")
            lines.append(f"     {p.get('prompt', '')}")
            lines.append("")

        response = "\n".join(lines).strip()
        self.db.save("prompt", response, title=f"Prompts — {medium} / {theme}",
                     metadata=json.dumps({"medium": medium, "theme": theme, "count": count}))

        return {
            "response": response,
            "data": {"prompts": prompts},
            "confidence": 0.95,
        }

    # ── generate_lyrics ───────────────────────────────────────────────────────

    def _generate_lyrics(self, entities: dict) -> dict:
        topic   = entities.get("topic") or entities.get("raw_query", "")
        genre   = entities.get("genre", "")
        tone    = entities.get("tone", "")
        rhyme   = entities.get("rhyme_scheme", "")
        structure = entities.get("structure", "")

        # clarifying questions if missing
        missing = []
        if not topic:
            missing.append("What should the song be about?")
        if not genre:
            missing.append("What genre or style? (pop, hip-hop, folk, rock, indie)")
        if not structure:
            missing.append("What structure? (verse-chorus, verse-chorus-bridge, just a verse)")
        if not rhyme:
            missing.append("Rhyme scheme? (ABAB, AABB, free, none)")

        if missing:
            return {
                "response": " ".join(missing),
                "data": {"needs_clarification": True},
                "confidence": 0.6,
            }

        # defaults
        genre     = genre     or "pop"
        tone      = tone      or "emotional"
        rhyme     = rhyme     or "ABAB"
        structure = structure or "verse-chorus-verse-chorus-bridge-chorus"

        lyrics = self._ollama_text(
            _LYRICS_PROMPT.format(
                topic=topic, structure=structure,
                rhyme_scheme=rhyme, tone=tone, genre=genre,
            )
        ).strip()

        if not lyrics:
            return {"response": "I had trouble writing those lyrics.", "data": {}, "confidence": 0.3}

        title = f"{genre.title()} — {topic.title()}"

        self.db.save("lyrics", lyrics, title=title,
                     metadata=json.dumps({
                         "genre": genre, "tone": tone,
                         "rhyme": rhyme, "structure": structure,
                         "topic": topic,
                     }))
        self._persist_to_memory(f"lyrics_{topic}", lyrics[:500])

        response = f"{title}\n\n{lyrics}"

        return {
            "response": response,
            "data": {
                "title": title, "lyrics": lyrics,
                "genre": genre, "structure": structure, "rhyme": rhyme,
            },
            "confidence": 0.95,
        }