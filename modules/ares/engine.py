# modules/ares/engine.py

import json
import logging
from datetime import datetime, timedelta
from modules.base import BaseModule
from core.ollama_client import generate

log = logging.getLogger(__name__)

_PLAN_PROMPT = """You are Ares, a strategic planning assistant.
The user wants a strategic plan for: {topic}

Respond with ONLY valid JSON in this exact structure:
{{
  "goal": "one sentence stating the core objective",
  "steps": ["step 1", "step 2", "step 3", "step 4", "step 5"],
  "timeline": "realistic timeframe with milestones, 2-3 sentences",
  "risks": ["risk 1", "risk 2", "risk 3"],
  "first_milestone": {{
    "description": "the very first concrete action",
    "due_days": 3
  }}
}}

Be specific to the topic. No preamble. No explanation. JSON only."""


class AresEngine(BaseModule):
    name = "ares"
    _INTENTS = {
        "analyse_risk",
        "strategic_plan",
        "swot_analysis",
        "decision_support",
    }

    def __init__(self, memory=None, ollama_cfg: dict = None):
        self._memory = memory
        self._ollama = ollama_cfg or {}

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        if intent == "strategic_plan":
            return self._strategic_plan(entities, context)
        return {
            "response": f"{intent.replace('_', ' ').title()} is coming soon.",
            "data": {},
            "confidence": 0.0,
        }

    def get_context(self) -> dict:
        return {}

    # ── handlers ────────────────────────────────────────

    def _strategic_plan(self, entities: dict, context: dict) -> dict:
        topic = (
            entities.get("topic")
            or entities.get("raw_query")
            or "your goal"
        )

        raw = generate(
            _PLAN_PROMPT.format(topic=topic),
            model=self._ollama.get("model", "mistral"),
            host=self._ollama.get("host", "127.0.0.1"),
            port=self._ollama.get("port", 11434),
            fmt="json",
        )

        try:
            plan = json.loads(raw)
        except Exception:
            log.warning("Ares: failed to parse plan JSON, returning raw")
            return {
                "response": raw or "I had trouble generating a plan.",
                "data": {},
                "confidence": 0.3,
            }

        response = self._format_plan(topic, plan)

        # ── persist to Mnemosyne ─────────────────────────
        if self._memory:
            fact_key = f"ares_plan_{topic[:30].replace(' ', '_').lower()}"
            self._memory.learn(fact_key, response)

            milestone = plan.get("first_milestone", {})
            if milestone.get("description"):
                due_days = int(milestone.get("due_days", 3))
                due_dt = (datetime.utcnow() + timedelta(days=due_days)).isoformat()
                self._memory.add_reminder(milestone["description"], due_dt)

        return {
            "response": response,
            "data": plan,
            "confidence": 0.9,
        }

    @staticmethod
    def _format_plan(topic: str, plan: dict) -> str:
        lines = [f"Strategic Plan: {topic.title()}", ""]

        if plan.get("goal"):
            lines += ["GOAL", plan["goal"], ""]

        if plan.get("steps"):
            lines.append("STEPS")
            for i, step in enumerate(plan["steps"], 1):
                lines.append(f"  {i}. {step}")
            lines.append("")

        if plan.get("timeline"):
            lines += ["TIMELINE", plan["timeline"], ""]

        if plan.get("risks"):
            lines.append("RISKS")
            for risk in plan["risks"]:
                lines.append(f"  • {risk}")
            lines.append("")

        milestone = plan.get("first_milestone", {})
        if milestone.get("description"):
            due = milestone.get("due_days", 3)
            lines += [
                "FIRST MILESTONE",
                f"  {milestone['description']} (due in {due} days)",
            ]

        return "\n".join(lines).strip()