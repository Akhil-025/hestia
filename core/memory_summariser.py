import json
import sys
import datetime
from typing import Optional

class MemorySummariser:
    """
    Extracts user facts and mood from recent interactions using the LLM.
    Designed to run as a nightly background job via HestiaHeartbeat.
    """
    def __init__(self, memory, llm_caller):
        """
        Args:
            memory: HestiaMemory instance.
            llm_caller: Callable[[str], Optional[str]] — takes a prompt string,
                        returns LLM response string or None.
                        Pass HestiaNLU._call_llm directly.
        """
        self.memory = memory
        self.llm_caller = llm_caller

    def run(self) -> dict:
        """
        Main entry point. Pulls recent interactions, runs fact extraction
        and mood detection, stores results in memory.
        Returns summary dict: {"facts_extracted": int, "mood": str}
        """
        interactions = self.memory.get_recent(limit=50)
        if not interactions:
            return {"facts_extracted": 0, "mood": "neutral"}

        conversation_text = self._format_interactions(interactions)
        facts_count = self._extract_facts(conversation_text)
        mood = self._detect_mood(conversation_text)

        if mood:
            self.memory.log_mood(mood)

        return {"facts_extracted": facts_count, "mood": mood or "neutral"}

    def _format_interactions(self, interactions: list[dict]) -> str:
        """Format interactions list into a readable conversation string."""
        lines = []
        for item in interactions:
            lines.append(f"User: {item['query']}")
            lines.append(f"Hestia: {item['response']}")
        return "\n".join(lines)

    def _extract_facts(self, conversation_text: str) -> int:
        """
        Ask LLM to extract user facts from conversation.
        Returns count of facts stored.
        """
        prompt = f"""Extract factual information about the user from this conversation.
Return ONLY a JSON array of objects with keys \"key\", \"value\", \"confidence\" (0.0-1.0).
Focus on: name, preferences, location, job, habits, interests, relationships.
If no facts found, return [].
No explanation, no markdown, just the JSON array.

Conversation:
{conversation_text[:2000]}

JSON array:"""

        response = self.llm_caller(prompt)
        if not response:
            return 0

        try:
            # Strip any markdown fences
            clean = response.strip().strip("```json").strip("```").strip()
            start = clean.find("[")
            end = clean.rfind("]") + 1
            if start == -1 or end == 0:
                return 0
            facts = json.loads(clean[start:end])
            count = 0
            for fact in facts:
                if isinstance(fact, dict) and "key" in fact and "value" in fact:
                    self.memory.upsert_fact(
                        key=str(fact["key"]).lower().replace(" ", "_"),
                        value=str(fact["value"]),
                        confidence=float(fact.get("confidence", 0.8)),
                        source="llm_summariser"
                    )
                    count += 1
            return count
        except Exception as e:
            print(f"[MemorySummariser] Fact extraction failed: {e}", file=sys.stderr)
            return 0

    def _detect_mood(self, conversation_text: str) -> Optional[str]:
        """
        Ask LLM to detect user mood from conversation.
        Returns one of: positive, neutral, negative, stressed, happy — or None on failure.
        """
        prompt = f"""Analyse the emotional tone of the USER's messages only (not Hestia's).
Reply with exactly ONE word from this list: positive, neutral, negative, stressed, happy
No explanation. Just the single word.

Conversation:
{conversation_text[:1500]}

Mood:"""

        response = self.llm_caller(prompt)
        if not response:
            return None
        mood = response.strip().lower().split()[0] if response.strip() else None
        valid = {"positive", "neutral", "negative", "stressed", "happy"}
        return mood if mood in valid else "neutral"
