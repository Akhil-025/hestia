# modules/ares/engine.py

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from modules.base import BaseModule
from core.ollama_client import generate

log = logging.getLogger(__name__)

# Same "Note saved: " prefix contract as modules/hestia/core_module.py —
# duplicated rather than imported to keep this module's dependency surface
# limited to `modules.base`/`core.ollama_client`, matching the rest of the
# file's import style.
_NOTE_PREFIX_RE = re.compile(r'^Note saved:\s*', re.IGNORECASE)

# Appended to every analysis prompt below. Each _*_PROMPT interpolates a
# {grounding} block built by AresEngine._grounding() from real Mnemosyne
# data (facts/goals/notes). Previously these prompts only ever received a
# bare {topic} string and told the model to "be specific" — with nothing
# true to draw on, a small local model's only way to comply was to invent
# plausible-sounding specifics (fictional job offers, fictional
# competitors, fictional battlefield content). This instruction gives the
# model an explicit, honest way to be specific about what's real and
# general where it isn't, instead of fabricating either way.
_GROUNDING_RULE = (
    "Only state specific facts, names, numbers, or events that appear in "
    "KNOWN CONTEXT above or in the topic/options the user gave you. If "
    "KNOWN CONTEXT is empty or not relevant to the topic, give sound "
    "general strategic reasoning instead — do not invent specific details "
    "to sound concrete."
)

# ── Prompts ──────────────────────────────────────────────────────────────────

_PLAN_PROMPT = """You are Ares, a strategic planning assistant.
The user wants a strategic plan for: {topic}

KNOWN CONTEXT (may be empty):
{grounding}

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

""" + _GROUNDING_RULE + """ No preamble. No explanation. JSON only."""

_RISK_PROMPT = """You are Ares, a risk analysis assistant.
Analyse the risks for: {topic}

KNOWN CONTEXT (may be empty):
{grounding}

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

Identify 4-6 distinct risks. """ + _GROUNDING_RULE + """ JSON only."""

_SWOT_PROMPT = """You are Ares, a strategic analysis assistant.
Perform a SWOT analysis for: {topic}

KNOWN CONTEXT (may be empty):
{grounding}

Respond with ONLY valid JSON in this exact structure:
{{
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "weaknesses": ["weakness 1", "weakness 2", "weakness 3"],
  "opportunities": ["opportunity 1", "opportunity 2", "opportunity 3"],
  "threats": ["threat 1", "threat 2", "threat 3"]
}}

3-4 points per quadrant. """ + _GROUNDING_RULE + """ JSON only."""

_DECISION_PROMPT = """You are Ares, a decision support assistant.
The user needs help deciding: {topic}
Their options are: {options}

KNOWN CONTEXT (may be empty):
{grounding}

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

Score each option out of 10. Only analyse the options the user actually
named above — never invent additional or alternative options. """ + _GROUNDING_RULE + """ JSON only."""

_PREMORTEM_PROMPT = """You are Ares, running a premortem exercise.
Imagine it is some time in the future and the following has already failed, badly: {topic}

KNOWN CONTEXT (may be empty):
{grounding}

Respond with ONLY valid JSON in this exact structure:
{{
  "scenario": "one sentence describing how the failure played out",
  "failure_causes": [
    {{
      "cause": "a specific, plausible reason this failed",
      "warning_sign": "an early signal that would have shown this cause emerging",
      "prevention": "a concrete action to take now to prevent it"
    }}
  ],
  "single_point_of_failure": "the one factor most likely to sink this if nothing else does",
  "confidence_in_success": "Low | Medium | High"
}}

Identify 4-6 distinct, non-overlapping failure causes. This is a hypothetical
exercise, so plausible invented failure *causes* are expected and fine —
but ground them in KNOWN CONTEXT where it's relevant instead of ignoring it.
No preamble. JSON only."""

_COMPETITIVE_PROMPT = """You are Ares, a competitive strategy assistant.
Analyse the competitive landscape for: {topic}
{competitors_line}

KNOWN CONTEXT (may be empty):
{grounding}

Respond with ONLY valid JSON in this exact structure:
{{
  "position_summary": "one sentence on where the user currently stands relative to the field",
  "competitors": [
    {{
      "name": "competitor name",
      "strengths": ["strength 1", "strength 2"],
      "vulnerabilities": ["vulnerability 1", "vulnerability 2"],
      "counter_move": "the single best move to gain ground against this competitor"
    }}
  ],
  "differentiation": "what should set the user apart from all of them",
  "biggest_threat": "which competitor or force poses the greatest risk right now"
}}

If specific competitors were named above, analyse only those. Otherwise,
clearly label inferred rivals as illustrative examples typical for the
topic rather than presenting them as known facts. Analyse 3-5 competitors
total. """ + _GROUNDING_RULE + """ JSON only."""

_CONTINGENCY_PROMPT = """You are Ares, a contingency planning assistant.
Build a fallback plan for: {topic}
{trigger_line}

KNOWN CONTEXT (may be empty):
{grounding}

Respond with ONLY valid JSON in this exact structure:
{{
  "primary_assumption": "the assumption the main plan depends on holding true",
  "trigger_conditions": ["condition 1 that would mean the fallback is needed", "condition 2"],
  "fallback_steps": ["step 1", "step 2", "step 3"],
  "resources_to_preposition": ["resource or preparation to line up now, 1", "resource or preparation 2"],
  "decision_point": "the latest moment by which the switch to the fallback must be made"
}}

""" + _GROUNDING_RULE + """ JSON only."""

_WAR_ROOM_PROMPT = """You are Ares, delivering a consolidated strategic briefing.
Topic: {topic}

KNOWN CONTEXT — the only real information you have about the user's
actual situation (may be empty):
{grounding}

Synthesise KNOWN CONTEXT above into ONE consolidated briefing. Respond
with ONLY valid JSON in this exact structure:
{{
  "situation": "2-3 sentence summary of where things stand right now",
  "priorities": ["top priority 1", "top priority 2", "top priority 3"],
  "open_risks": ["risk still unresolved 1", "risk still unresolved 2"],
  "recommended_next_move": "the single highest-leverage action to take next",
  "confidence": "Low | Medium | High"
}}

If KNOWN CONTEXT is empty, you have no real basis for a briefing: set
"situation" to a one-sentence honest statement that there's nothing on
record yet for this topic, leave "priorities" and "open_risks" as empty
lists, and set "confidence" to "Low". Do NOT invent a scenario, situation,
or details of any kind — including unrelated scenarios like business or
military situations — when KNOWN CONTEXT is empty. JSON only."""


class AresEngine(BaseModule):
    name = "ares"
    _INTENTS = {
        "analyse_risk",
        "strategic_plan",
        "swot_analysis",
        "decision_support",
        "premortem_analysis",
        "competitive_analysis",
        "contingency_plan",
        "war_room_briefing",
    }

    # Low temperature for these calls: every prompt above asks for
    # structured, "be specific" analytical output, which is exactly the
    # kind of task where high sampling temperature (Ollama's ~0.8 default)
    # increases how much a small local model fabricates rather than making
    # the output more useful. This is not passed to _premortem_analysis's
    # hypothetical scenario generation any differently — the "do not
    # invent" instruction there is already scoped to only ground the
    # failure causes, not to suppress the hypothetical itself.
    _ANALYSIS_OPTIONS = {"temperature": 0.2}

    def __init__(self, memory=None, ollama_cfg: dict = None, llm=None):
        self._memory = memory
        self._ollama = ollama_cfg or {}
        self._llm_instance = llm  # HestiaLLM | None — preferred path

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
        if intent == "premortem_analysis":
            return self._premortem_analysis(entities, context)
        if intent == "competitive_analysis":
            return self._competitive_analysis(entities, context)
        if intent == "contingency_plan":
            return self._contingency_plan(entities, context)
        if intent == "war_room_briefing":
            return self._war_room_briefing(entities, context)
        return {
            "response": f"{intent.replace('_', ' ').title()} is coming soon.",
            "data": {},
            "confidence": 0.0,
        }

    def get_context(self) -> dict:
        return {}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ollama_call(self, prompt: str) -> str:
        if self._llm_instance is not None:
            return self._llm_instance.generate(prompt, fmt="json", options=self._ANALYSIS_OPTIONS)
        return generate(
            prompt,
            model=self._ollama.get("model", "mistral"),
            host=self._ollama.get("host", "127.0.0.1"),
            port=self._ollama.get("port", 11434),
            fmt="json",
            options=self._ANALYSIS_OPTIONS,
        )

    def _parse(self, raw: str, intent: str) -> dict | None:
        try:
            parsed = json.loads(raw)
        except Exception:
            log.warning("Ares: failed to parse JSON for %s", intent)
            return None
        if not isinstance(parsed, dict):
            # The prompts always ask for a JSON *object*; an occasional
            # model slip (a bare list/string/number) would otherwise pass
            # json.loads() only to blow up with AttributeError the moment
            # a caller does plan.get(...), which escapes as the generic
            # orchestrator error instead of the friendly "I had trouble..."
            # message every _*() handler below is designed to give.
            log.warning(
                "Ares: expected a JSON object for %s, got %s", intent, type(parsed).__name__
            )
            return None
        return parsed

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

    @staticmethod
    def _note_text(row: dict) -> str:
        resp = (row.get("response") or "").strip()
        if _NOTE_PREFIX_RE.match(resp):
            return _NOTE_PREFIX_RE.sub('', resp).strip()
        return (row.get("query") or "").strip()

    def _grounding(self, context: dict | None = None) -> str:
        """
        Assemble whatever real, verifiable context Mnemosyne has about the
        user — known facts, active goals, recent notes — into a compact
        block every prompt above injects as KNOWN CONTEXT.

        This does not guarantee the topic will actually be covered by any
        of it (a SWOT on "switching jobs" won't magically find real job
        offers if none were ever logged) — but it gives the model
        something true to reason from, and the _GROUNDING_RULE instruction
        on each prompt gives it an explicit, honest fallback ("give sound
        general strategic reasoning instead") rather than fabricating
        specifics to satisfy "be specific to the topic".
        """
        if not self._memory:
            return ""

        parts: list[str] = []

        try:
            facts = self._memory.get_top_facts_for_context(limit=8)
            if facts:
                parts.append("Known facts about the user:\n" + facts)
        except Exception:
            log.warning("Ares: get_top_facts_for_context failed.", exc_info=True)

        try:
            goals = self._memory.db.get_goals(status="active")
            if goals:
                lines = "\n".join(f"- {g['text']}" for g in goals[:8])
                parts.append("Active goals:\n" + lines)
        except Exception:
            log.warning("Ares: get_goals failed.", exc_info=True)

        try:
            notes = self._memory.db.get_by_intent("take_note", 8)
            if notes:
                lines = "\n".join(f"- {self._note_text(n)}" for n in notes)
                parts.append("Recent notes:\n" + lines)
        except Exception:
            log.warning("Ares: get_by_intent(take_note) failed.", exc_info=True)

        # Ares' own past analyses are persisted as facts (see _persist()),
        # so anything already covered by the block above will naturally
        # include prior Ares output too — no separate lookup needed here.

        recent_intents = (context or {}).get("recent_intents")
        if recent_intents:
            parts.append("Recent conversation topics: " + ", ".join(recent_intents[-5:]))

        return "\n\n".join(parts)

    # ── strategic_plan ───────────────────────────────────────────────────────

    def _strategic_plan(self, entities: dict, context: dict) -> dict:
        topic = self._topic(entities)
        raw   = self._ollama_call(_PLAN_PROMPT.format(topic=topic, grounding=self._grounding(context) or "(none)"))
        plan  = self._parse(raw, "strategic_plan")

        if not plan:
            return {"response": raw or "I had trouble generating a plan.",
                    "data": {}, "confidence": 0.3}

        response = self._format_plan(topic, plan)
        self._persist(f"plan_{topic}", response)

        milestone = plan.get("first_milestone", {})
        if milestone.get("description") and self._memory:
            due_days = int(milestone.get("due_days", 3))
            due_dt   = (datetime.now(timezone.utc) + timedelta(days=due_days)).isoformat()
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
        raw    = self._ollama_call(_RISK_PROMPT.format(topic=topic, grounding=self._grounding(context) or "(none)"))
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
        raw    = self._ollama_call(_SWOT_PROMPT.format(topic=topic, grounding=self._grounding(context) or "(none)"))
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

        # Work on copies — result's lists are the same objects returned to
        # the caller as `data`, so padding them in place (to equalise
        # column heights below) would leak spurious "" entries into the
        # structured payload every time the four quadrants differ in size.
        s = list(result.get("strengths",     []))
        w = list(result.get("weaknesses",    []))
        o = list(result.get("opportunities", []))
        t = list(result.get("threats",       []))

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

        # Fallback: try to extract options from the raw query using an "or"/
        # comma split before giving up and asking the user to list them.
        if not options:
            raw_query = entities.get("raw_query", "")
            if raw_query:
                parts = re.split(r'\bor\b|,', raw_query, flags=re.IGNORECASE)
                parts = [p.strip() for p in parts if len(p.strip()) > 3]
                if len(parts) >= 2:
                    options = ", ".join(parts)

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

        raw    = self._ollama_call(_DECISION_PROMPT.format(topic=topic, options=options, grounding=self._grounding(context) or "(none)"))
        result = self._parse(raw, "decision_support")

        if not result:
            return {"response": raw or "I had trouble analysing that decision.",
                    "data": {}, "confidence": 0.3}

        response = self._format_decision(topic, result)
        self._persist(f"decision_{topic}", response)

        if result.get("next_step") and self._memory:
            due_dt = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
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

    # ── premortem_analysis ───────────────────────────────────────────────────

    def _premortem_analysis(self, entities: dict, context: dict) -> dict:
        topic  = self._topic(entities)
        raw    = self._ollama_call(_PREMORTEM_PROMPT.format(topic=topic, grounding=self._grounding(context) or "(none)"))
        result = self._parse(raw, "premortem_analysis")

        if not result:
            return {"response": raw or "I had trouble running that premortem.",
                    "data": {}, "confidence": 0.3}

        response = self._format_premortem(topic, result)
        self._persist(f"premortem_{topic}", response)

        return {"response": response, "data": result, "confidence": 0.9}

    @staticmethod
    def _format_premortem(topic: str, result: dict) -> str:
        lines = [f"Premortem: {topic.title()}", ""]

        if result.get("scenario"):
            lines += ["SCENARIO", f"  {result['scenario']}", ""]

        if result.get("failure_causes"):
            lines.append("FAILURE CAUSES")
            for c in result["failure_causes"]:
                lines.append(f"  • {c.get('cause', '')}")
                if c.get("warning_sign"):
                    lines.append(f"      Warning sign : {c['warning_sign']}")
                if c.get("prevention"):
                    lines.append(f"      Prevention   : {c['prevention']}")
            lines.append("")

        if result.get("single_point_of_failure"):
            lines += ["SINGLE POINT OF FAILURE", f"  {result['single_point_of_failure']}", ""]

        if result.get("confidence_in_success"):
            lines += [f"CONFIDENCE IN SUCCESS (as planned): {result['confidence_in_success']}"]

        return "\n".join(lines).strip()

    # ── competitive_analysis ─────────────────────────────────────────────────

    def _competitive_analysis(self, entities: dict, context: dict) -> dict:
        topic       = self._topic(entities)
        competitors = entities.get("competitors", "")
        competitors_line = (
            f"Known competitors/rivals: {competitors}" if competitors else ""
        )

        raw    = self._ollama_call(
            _COMPETITIVE_PROMPT.format(topic=topic, competitors_line=competitors_line, grounding=self._grounding(context) or "(none)")
        )
        result = self._parse(raw, "competitive_analysis")

        if not result:
            return {"response": raw or "I had trouble analysing the competitive landscape.",
                    "data": {}, "confidence": 0.3}

        response = self._format_competitive(topic, result)
        self._persist(f"competitive_{topic}", response)

        return {"response": response, "data": result, "confidence": 0.9}

    @staticmethod
    def _format_competitive(topic: str, result: dict) -> str:
        lines = [f"Competitive Analysis: {topic.title()}", ""]

        if result.get("position_summary"):
            lines += ["POSITION", f"  {result['position_summary']}", ""]

        for c in result.get("competitors", []):
            lines.append(f"COMPETITOR: {c.get('name', '')}")
            if c.get("strengths"):
                lines.append("  Strengths:")
                for s in c["strengths"]:
                    lines.append(f"    + {s}")
            if c.get("vulnerabilities"):
                lines.append("  Vulnerabilities:")
                for v in c["vulnerabilities"]:
                    lines.append(f"    - {v}")
            if c.get("counter_move"):
                lines.append(f"  Counter-move: {c['counter_move']}")
            lines.append("")

        if result.get("differentiation"):
            lines += ["DIFFERENTIATION", f"  {result['differentiation']}", ""]

        if result.get("biggest_threat"):
            lines += ["BIGGEST THREAT", f"  {result['biggest_threat']}"]

        return "\n".join(lines).strip()

    # ── contingency_plan ─────────────────────────────────────────────────────

    def _contingency_plan(self, entities: dict, context: dict) -> dict:
        topic   = self._topic(entities)
        trigger = entities.get("trigger", "")
        trigger_line = (
            f"The specific risk to plan against: {trigger}" if trigger else ""
        )

        raw    = self._ollama_call(
            _CONTINGENCY_PROMPT.format(topic=topic, trigger_line=trigger_line, grounding=self._grounding(context) or "(none)")
        )
        result = self._parse(raw, "contingency_plan")

        if not result:
            return {"response": raw or "I had trouble building a fallback plan.",
                    "data": {}, "confidence": 0.3}

        response = self._format_contingency(topic, result)
        self._persist(f"contingency_{topic}", response)

        return {"response": response, "data": result, "confidence": 0.9}

    @staticmethod
    def _format_contingency(topic: str, result: dict) -> str:
        lines = [f"Contingency Plan: {topic.title()}", ""]

        if result.get("primary_assumption"):
            lines += ["PRIMARY ASSUMPTION", f"  {result['primary_assumption']}", ""]

        if result.get("trigger_conditions"):
            lines.append("SWITCH TO PLAN B IF")
            for t in result["trigger_conditions"]:
                lines.append(f"  • {t}")
            lines.append("")

        if result.get("fallback_steps"):
            lines.append("FALLBACK STEPS")
            for i, step in enumerate(result["fallback_steps"], 1):
                lines.append(f"  {i}. {step}")
            lines.append("")

        if result.get("resources_to_preposition"):
            lines.append("PRE-POSITION NOW")
            for r in result["resources_to_preposition"]:
                lines.append(f"  • {r}")
            lines.append("")

        if result.get("decision_point"):
            lines += ["DECISION POINT", f"  {result['decision_point']}"]

        return "\n".join(lines).strip()

    # ── war_room_briefing ────────────────────────────────────────────────────
    # Pulls whatever Mnemosyne remembers about this topic (including Ares'
    # own past plans/risks/SWOTs/decisions, since _persist() writes them
    # into memory under "ares_*" keys) and asks the model to synthesise it
    # into one consolidated briefing, rather than starting from a blank
    # page every time the topic comes up again.

    def _war_room_briefing(self, entities: dict, context: dict) -> dict:
        topic = self._topic(entities)

        # Two sources of grounding, combined:
        #  1. memory.remember(topic) — fuzzy semantic recall keyed off the
        #     topic string. Kept because it can surface things the
        #     structured lookups below won't (e.g. summarised history),
        #     but it's an unreliable signal on its own: when `topic` falls
        #     back to the raw command sentence (e.g. "summarize my recent
        #     work and highlight gaps"), it's an instruction, not a search
        #     term, so recall can return weak/irrelevant matches.
        #  2. self._grounding() — the same structured facts/goals/notes
        #     lookup every other Ares prompt now uses, which doesn't
        #     depend on the topic string being a good search query.
        semantic_context = ""
        if self._memory:
            try:
                semantic_context = self._memory.remember(topic, n=6)
            except Exception:
                log.warning("Ares: memory recall failed for war_room_briefing on %r", topic)

        structured_context = self._grounding(context)

        combined = "\n\n".join(p for p in (structured_context, semantic_context) if p)

        raw    = self._ollama_call(
            _WAR_ROOM_PROMPT.format(
                topic=topic,
                grounding=combined or "(nothing on record)",
            )
        )
        result = self._parse(raw, "war_room_briefing")

        if not result:
            return {"response": raw or "I had trouble putting that briefing together.",
                    "data": {}, "confidence": 0.3}

        response = self._format_war_room(topic, result, bool(combined))
        self._persist(f"briefing_{topic}", response)

        # A briefing built from no real context isn't wrong, but it also
        # isn't grounded in anything — reflect that in confidence instead
        # of reporting the same 0.9 as a briefing backed by real data.
        confidence = 0.9 if combined else 0.4

        return {"response": response, "data": result, "confidence": confidence}

    @staticmethod
    def _format_war_room(topic: str, result: dict, had_prior_context: bool) -> str:
        source_note = "drawing on prior analysis" if had_prior_context else "no prior analysis on record"
        lines = [f"War Room Briefing: {topic.title()} ({source_note})", ""]

        if result.get("situation"):
            lines += ["SITUATION", f"  {result['situation']}", ""]

        if result.get("priorities"):
            lines.append("PRIORITIES")
            for i, p in enumerate(result["priorities"], 1):
                lines.append(f"  {i}. {p}")
            lines.append("")

        if result.get("open_risks"):
            lines.append("OPEN RISKS")
            for r in result["open_risks"]:
                lines.append(f"  • {r}")
            lines.append("")

        if result.get("recommended_next_move"):
            lines += ["RECOMMENDED NEXT MOVE", f"  {result['recommended_next_move']}", ""]

        if result.get("confidence"):
            lines += [f"CONFIDENCE: {result['confidence']}"]

        return "\n".join(lines).strip()