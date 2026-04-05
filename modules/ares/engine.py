# modules/ares/engine.py

import json
import logging
from datetime import datetime, timedelta
from modules.base import BaseModule
from core.ollama_client import generate

log = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

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

_RISK_PROMPT = """You are Ares, a risk analysis assistant.
Analyse the risks for: {topic}

Respond with ONLY valid JSON in this exact structure:
{{
  "risks": [
    {{
      "risk": "name of the risk",
      "likelihood": "Low | Medium | High",
      "impact": "Low | Medium | High",
      "mitigation": "concrete mitigation strategy"
    }}
  ]
}}

Identify 4-6 distinct risks. Be specific to the topic. JSON only."""

_SWOT_PROMPT = """You are Ares, a strategic analysis assistant.
Perform a SWOT analysis for: {topic}

Respond with ONLY valid JSON in this exact structure:
{{
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "weaknesses": ["weakness 1", "weakness 2", "weakness 3"],
  "opportunities": ["opportunity 1", "opportunity 2", "opportunity 3"],
  "threats": ["threat 1", "threat 2", "threat 3"]
}}

3-4 points per quadrant. Be specific to the topic. JSON only."""

_DECISION_PROMPT = """You are Ares, a decision support assistant.
The user needs help deciding: {topic}
Their options are: {options}

Respond with ONLY valid JSON in this exact structure:
{{
  "recommendation": "which option you recommend and why in one sentence",
  "options_analysis": [
    {{
      "option": "option name",
      "pros": ["pro 1", "pro 2"],
      "cons": ["con 1", "con 2"],
      "score": 7
    }}
  ],
  "key_factors": ["factor 1", "factor 2", "factor 3"],
  "next_step": "the single most important action to take now"
}}

Score each option out of 10. Be specific. JSON only."""


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
        if intent == "analyse_risk":
            return self._analyse_risk(entities, context)
        if intent == "swot_analysis":
            return self._swot_analysis(entities, context)
        if intent == "decision_support":
            return self._decision_support(entities, context)
        return {
            "response": f"{intent.replace('_', ' ').title()} is coming soon.",
            "data": {},
            "confidence": 0.0,
        }

    def get_context(self) -> dict:
        return {}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ollama_call(self, prompt: str) -> str:
        return generate(
            prompt,
            model=self._ollama.get("model", "mistral"),
            host=self._ollama.get("host", "127.0.0.1"),
            port=self._ollama.get("port", 11434),
            fmt="json",
        )

    def _parse(self, raw: str, intent: str) -> dict | None:
        try:
            return json.loads(raw)
        except Exception:
            log.warning("Ares: failed to parse JSON for %s", intent)
            return None

    def _persist(self, key: str, value: str) -> None:
        if self._memory:
            safe_key = f"ares_{key[:40].replace(' ', '_').lower()}"
            self._memory.learn(safe_key, value)

    def _topic(self, entities: dict) -> str:
        return (
            entities.get("topic")
            or entities.get("raw_query")
            or "your topic"
        )

    # ── strategic_plan ───────────────────────────────────────────────────────

    def _strategic_plan(self, entities: dict, context: dict) -> dict:
        topic = self._topic(entities)
        raw   = self._ollama_call(_PLAN_PROMPT.format(topic=topic))
        plan  = self._parse(raw, "strategic_plan")

        if not plan:
            return {"response": raw or "I had trouble generating a plan.",
                    "data": {}, "confidence": 0.3}

        response = self._format_plan(topic, plan)
        self._persist(f"plan_{topic}", response)

        milestone = plan.get("first_milestone", {})
        if milestone.get("description") and self._memory:
            due_days = int(milestone.get("due_days", 3))
            due_dt   = (datetime.utcnow() + timedelta(days=due_days)).isoformat()
            self._memory.add_reminder(milestone["description"], due_dt)

        return {"response": response, "data": plan, "confidence": 0.9}

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

    # ── analyse_risk ─────────────────────────────────────────────────────────

    def _analyse_risk(self, entities: dict, context: dict) -> dict:
        topic  = self._topic(entities)
        raw    = self._ollama_call(_RISK_PROMPT.format(topic=topic))
        result = self._parse(raw, "analyse_risk")

        if not result:
            return {"response": raw or "I had trouble analysing risks.",
                    "data": {}, "confidence": 0.3}

        response = self._format_risk(topic, result)
        self._persist(f"risk_{topic}", response)

        return {"response": response, "data": result, "confidence": 0.9}

    @staticmethod
    def _format_risk(topic: str, result: dict) -> str:
        lines = [f"Risk Analysis: {topic.title()}", ""]

        for r in result.get("risks", []):
            lines.append(f"RISK: {r.get('risk', '')}")
            lines.append(f"  Likelihood : {r.get('likelihood', '?')}")
            lines.append(f"  Impact     : {r.get('impact', '?')}")
            lines.append(f"  Mitigation : {r.get('mitigation', '?')}")
            lines.append("")

        return "\n".join(lines).strip()

    # ── swot_analysis ────────────────────────────────────────────────────────

    def _swot_analysis(self, entities: dict, context: dict) -> dict:
        topic  = self._topic(entities)
        raw    = self._ollama_call(_SWOT_PROMPT.format(topic=topic))
        result = self._parse(raw, "swot_analysis")

        if not result:
            return {"response": raw or "I had trouble running the SWOT analysis.",
                    "data": {}, "confidence": 0.3}

        response = self._format_swot(topic, result)
        self._persist(f"swot_{topic}", response)

        return {"response": response, "data": result, "confidence": 0.9}

    @staticmethod
    def _format_swot(topic: str, result: dict) -> str:
        def col(items: list, width: int = 36) -> list:
            return [f"  • {i}"[:width].ljust(width) for i in items]

        s = result.get("strengths",    [])
        w = result.get("weaknesses",   [])
        o = result.get("opportunities",[])
        t = result.get("threats",      [])

        rows  = max(len(s), len(w), len(o), len(t))
        s    += [""] * (rows - len(s))
        w    += [""] * (rows - len(w))
        o    += [""] * (rows - len(o))
        t    += [""] * (rows - len(t))

        W = 36
        sep = "+" + "-" * W + "+" + "-" * W + "+"

        lines = [f"SWOT Analysis: {topic.title()}", "", sep]
        lines.append("|" + "STRENGTHS".center(W) + "|" + "WEAKNESSES".center(W) + "|")
        lines.append(sep)
        for a, b in zip(col(s, W), col(w, W)):
            lines.append(f"|{a}|{b}|")
        lines.append(sep)
        lines.append("|" + "OPPORTUNITIES".center(W) + "|" + "THREATS".center(W) + "|")
        lines.append(sep)
        for a, b in zip(col(o, W), col(t, W)):
            lines.append(f"|{a}|{b}|")
        lines.append(sep)

        return "\n".join(lines)

    # ── decision_support ─────────────────────────────────────────────────────

    def _decision_support(self, entities: dict, context: dict) -> dict:
        topic   = self._topic(entities)
        options = entities.get("options", "")

        # clarifying question if no options provided
        if not options:
            return {
                "response": (
                    f"I can help you decide on {topic}. "
                    "What are your options? List them and I'll analyse each one."
                ),
                "data": {},
                "confidence": 0.6,
            }

        raw    = self._ollama_call(_DECISION_PROMPT.format(topic=topic, options=options))
        result = self._parse(raw, "decision_support")

        if not result:
            return {"response": raw or "I had trouble analysing that decision.",
                    "data": {}, "confidence": 0.3}

        response = self._format_decision(topic, result)
        self._persist(f"decision_{topic}", response)

        if result.get("next_step") and self._memory:
            due_dt = (datetime.utcnow() + timedelta(days=1)).isoformat()
            self._memory.add_reminder(result["next_step"], due_dt)

        return {"response": response, "data": result, "confidence": 0.9}

    @staticmethod
    def _format_decision(topic: str, result: dict) -> str:
        lines = [f"Decision Support: {topic.title()}", ""]

        if result.get("recommendation"):
            lines += ["RECOMMENDATION", f"  {result['recommendation']}", ""]

        for opt in result.get("options_analysis", []):
            score = opt.get("score", "?")
            lines.append(f"OPTION: {opt.get('option', '')}  [{score}/10]")
            if opt.get("pros"):
                lines.append("  Pros:")
                for p in opt["pros"]:
                    lines.append(f"    + {p}")
            if opt.get("cons"):
                lines.append("  Cons:")
                for c in opt["cons"]:
                    lines.append(f"    - {c}")
            lines.append("")

        if result.get("key_factors"):
            lines.append("KEY FACTORS")
            for f in result["key_factors"]:
                lines.append(f"  • {f}")
            lines.append("")

        if result.get("next_step"):
            lines += ["NEXT STEP", f"  {result['next_step']}"]

        return "\n".join(lines).strip()