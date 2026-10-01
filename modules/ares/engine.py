# modules/ares/engine.py

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from modules.base import BaseModule
from core.ollama_client import generate
from core.free_apis import FreeAPIError, sec_company_search as _fa_sec_company_search
from . import simulate as _sim
from .db import AresDB

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

_CAREER_PROMPT = """You are Ares, a career-decision assistant.
The user is choosing between career/education paths: {topic}
Options to rank: {options}
{scope_line}
Score every option on every criterion below, from 1 (poor) to 10 (excellent),
where a HIGH score is always GOOD for the user (for a cost/effort criterion,
high means cheap/easy).
Criteria: {criteria}

KNOWN CONTEXT (may be empty):
{grounding}

Respond with ONLY valid JSON in this exact structure:
{{
  "options": [
    {{
      "option": "option name exactly as the user gave it",
      "scores": {{"criterion name": 7}},
      "strengths": ["strength 1", "strength 2"],
      "risks": ["risk 1", "risk 2"]
    }}
  ],
  "deciding_factor": "the one thing that should tip the choice",
  "next_step": "the single most useful action to take now",
  "data_to_verify": ["fact the user should confirm from an official source, 1", "fact 2"]
}}

Only rank the options the user actually named — never add alternatives.
Do NOT state specific cutoffs, ranks, salaries, seat counts, fees or dates
unless they appear in KNOWN CONTEXT or in the user's own message; describe
relative differences instead, and put anything you would need to look up in
"data_to_verify". """ + _GROUNDING_RULE + """ JSON only."""

_CAREER_DEFAULT_CRITERIA = {
    "earning potential": 3,
    "long-term growth": 4,
    "job security": 3,
    "learning and skill fit": 4,
    "alignment with my goals": 5,
    "ease of entry": 2,
}
_CAREER_GATE_CRITERIA = {
    "feasibility at my expected score": 5,
    "career outcomes": 4,
    "learning and research value": 3,
    "job security": 3,
    "cost and time": 3,
    "alignment with my goals": 5,
}
_CAREER_GATE_SCOPE = (
    "This is a GATE-exam-related choice (e.g. M.Tech vs PSU vs private job vs "
    "other routes). Judge feasibility only from the score/rank the user has "
    "stated, if any; if they gave none, say feasibility is unknown rather "
    "than guessing."
)

_PLAYBOOK_ANALYSES = (
    # (internal intent, keywords) — checked by earliest match in the text.
    ("premortem_analysis", ("premortem", "pre-mortem", "pre mortem")),
    ("swot_analysis", ("swot",)),
    ("contingency_plan", ("contingency", "fallback", "plan b")),
    ("competitive_analysis", ("competitive", "competitor", "competition")),
    ("career_ranking", ("career ranking", "career", "ranking")),
    ("analyse_risk", ("risk",)),
    ("decision_support", ("decision",)),
    ("strategic_plan", ("strategic plan", "plan")),
)
_PLAYBOOK_LABELS = {
    "premortem_analysis": "premortem", "swot_analysis": "SWOT",
    "contingency_plan": "contingency plan", "competitive_analysis": "competitive analysis",
    "career_ranking": "career ranking", "analyse_risk": "risk analysis",
    "decision_support": "decision support", "strategic_plan": "strategic plan",
}

_OUTCOME_SCORES = {"success": 1.0, "mixed": 0.5, "failure": 0.0}
_MIN_CALIBRATION_SAMPLES = 3

_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12,
}
_UNIT_DAYS = {"day": 1, "week": 7, "fortnight": 14, "month": 30, "year": 365}


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
        # Backlog #153-#157
        "career_ranking",
        "schedule_review",
        "record_outcome",
        "outcome_stats",
        "simulate_outcomes",
        "save_playbook",
        "list_playbooks",
        "run_playbook",
        "delete_playbook",
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

    def __init__(
        self,
        memory=None,
        ollama_cfg: dict = None,
        llm=None,
        db_path: str | None = None,
        auto_review_days: int | None = None,
    ):
        self._memory = memory
        self._ollama = ollama_cfg or {}
        self._llm_instance = llm  # HestiaLLM | None — preferred path
        # Decisions/outcomes/playbooks (#154, #155, #157). Opened lazily so
        # merely constructing the engine never touches disk. With no
        # db_path the store is in-memory and lasts for the process only;
        # main.py passes a real path.
        self._db_path = db_path or ":memory:"
        self._db_instance: AresDB | None = None
        # When set (config `ares.auto_review_days`), every plan/decision/
        # career ranking also schedules a "how did it turn out?" reminder.
        self._auto_review_days = (
            max(1, min(730, int(auto_review_days))) if auto_review_days else None
        )

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
        if intent == "career_ranking":
            return self._career_ranking(entities, context)
        if intent == "schedule_review":
            return self._schedule_review(entities, context)
        if intent == "record_outcome":
            return self._record_outcome(entities, context)
        if intent == "outcome_stats":
            return self._outcome_stats(entities, context)
        if intent == "simulate_outcomes":
            return self._simulate_outcomes(entities, context)
        if intent == "save_playbook":
            return self._save_playbook(entities, context)
        if intent == "list_playbooks":
            return self._list_playbooks(entities, context)
        if intent == "run_playbook":
            return self._run_playbook(entities, context)
        if intent == "delete_playbook":
            return self._delete_playbook(entities, context)
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
        # A saved playbook's standing criteria (#157) arrive via context so
        # every analysis prompt picks them up through this one block.
        focus = ((context or {}).get("_ares_focus") or "").strip()
        focus_part = (
            "Standing criteria for this analysis (from the user's saved "
            "playbook; address each one explicitly):\n" + focus
        ) if focus else ""

        if not self._memory:
            return focus_part

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

        if focus_part:
            parts.append(focus_part)

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

        decision_id, review_note = self._record_decision("plan", topic, response)
        if decision_id is not None:
            plan["decision_id"] = decision_id
        if review_note:
            response += "\n\n" + review_note

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
        options = self._extract_options(entities)

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

        top = self._top_score(result.get("options_analysis"))
        raw_conf = top / 10.0 if top is not None else None
        decision_id, review_note = self._record_decision(
            "decision", topic, response, options, raw_conf
        )
        if decision_id is not None:
            result["decision_id"] = decision_id
        response = self._with_track_record(response, result, raw_conf)
        if review_note:
            response += "\n\n" + review_note

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

        # Best-effort grounding in real public-record data via SEC EDGAR
        # (free, keyless). Only fires for topics that resolve to a
        # US-listed filer — most competitive_analysis topics won't, and
        # that's fine; this is a bonus signal, not a requirement, so a
        # miss or a network failure must never block the analysis.
        sec_line = ""
        try:
            matches = _fa_sec_company_search(topic)
        except FreeAPIError:
            log.debug("_competitive_analysis: SEC EDGAR lookup unavailable.", exc_info=True)
            matches = []
        if matches:
            names = ", ".join(f"{m.get('title')} ({m.get('ticker')})" for m in matches[:3])
            sec_line = f"SEC EDGAR public filers matching {topic!r}: {names}"

        grounding = self._grounding(context) or "(none)"
        if sec_line:
            grounding = f"{grounding}\n{sec_line}" if grounding != "(none)" else sec_line

        raw    = self._ollama_call(
            _COMPETITIVE_PROMPT.format(topic=topic, competitors_line=competitors_line, grounding=grounding)
        )
        result = self._parse(raw, "competitive_analysis")

        if not result:
            return {"response": raw or "I had trouble analysing the competitive landscape.",
                    "data": {}, "confidence": 0.3}

        response = self._format_competitive(topic, result)
        if sec_line:
            response += f"\n\nPUBLIC FILINGS\n  {sec_line}"
            result["sec_matches"] = matches
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

    # ═════════════════════════════════════════════════════════════════════════
    # Backlog #153-#157: career ranking, review reminders, outcome tracking,
    # Monte Carlo, playbooks.
    # ═════════════════════════════════════════════════════════════════════════

    # ── shared helpers ───────────────────────────────────────────────────────

    @property
    def _db(self) -> AresDB:
        if self._db_instance is None:
            self._db_instance = AresDB(self._db_path)
        return self._db_instance

    @staticmethod
    def _extract_options(entities: dict) -> str:
        """
        Options as a comma-separated string: the `options` entity if present,
        else a best-effort "or"/comma split of the raw query. Shared by
        decision_support and career_ranking.
        """
        options = entities.get("options", "")
        if isinstance(options, (list, tuple)):
            options = ", ".join(str(o) for o in options)

        # Fallback: try to extract options from the raw query using an
        # "or"/comma split before giving up and asking the user to list them.
        if not options:
            raw_query = entities.get("raw_query", "")
            if raw_query:
                parts = re.split(r'\bor\b|,', raw_query, flags=re.IGNORECASE)
                parts = [p.strip() for p in parts if len(p.strip()) > 3]
                if len(parts) >= 2:
                    options = ", ".join(parts)
        return options

    @staticmethod
    def _top_score(options_analysis) -> float | None:
        """Highest numeric 0-10 score among decision_support's options."""
        best = None
        for opt in options_analysis or []:
            try:
                score = float(opt.get("score"))
            except (AttributeError, TypeError, ValueError):
                continue
            if 0 <= score <= 10 and (best is None or score > best):
                best = score
        return best

    @staticmethod
    def _parse_delay_days(text: str) -> int | None:
        """'in 2 weeks', 'three months', 'tomorrow', 'next week' -> days (1-730)."""
        t = (text or "").lower()
        m = re.search(
            r"\b(\d+|" + "|".join(_NUM_WORDS) + r")\s+(day|week|fortnight|month|year)s?\b", t
        )
        if m:
            n = int(m.group(1)) if m.group(1).isdigit() else _NUM_WORDS[m.group(1)]
            return max(1, min(730, n * _UNIT_DAYS[m.group(2)]))
        if "tomorrow" in t:
            return 1
        m = re.search(r"\bnext\s+(week|fortnight|month|year)\b", t)
        if m:
            return _UNIT_DAYS[m.group(1)]
        return None

    def _schedule_reminder(self, text: str, days: int) -> str | None:
        """Add a Mnemosyne reminder `days` from now. Returns the ISO due time, or None."""
        if not self._memory:
            return None
        due = datetime.now(timezone.utc) + timedelta(days=days)
        try:
            self._memory.add_reminder(text, due.isoformat())
        except Exception:
            log.warning("Ares: add_reminder failed.", exc_info=True)
            return None
        return due.isoformat()

    def _record_decision(
        self,
        kind: str,
        topic: str,
        summary: str,
        options=None,
        predicted_confidence: float | None = None,
    ) -> tuple[int | None, str]:
        """
        Remember an analysis so it can be revisited and given an outcome.
        Returns (decision_id, note); note is non-empty only when a review
        reminder was auto-scheduled. Never raises: tracking is a bonus and
        must not break the analysis itself.
        """
        try:
            if isinstance(options, str):
                options = [o.strip() for o in options.split(",") if o.strip()]
            decision_id = self._db.add_decision(
                kind, topic, summary, options, predicted_confidence
            )
        except Exception:
            log.warning("Ares: could not record decision for %r.", topic, exc_info=True)
            return None, ""

        note = ""
        if self._auto_review_days:
            due = self._schedule_reminder(
                f"Revisit your {kind} on '{topic}': how did it turn out?",
                self._auto_review_days,
            )
            if due:
                try:
                    self._db.set_review(decision_id, due)
                except Exception:
                    log.warning("Ares: set_review failed.", exc_info=True)
                note = f"Review reminder set for {due[:10]}."
        return decision_id, note

    def _calibration(self) -> dict | None:
        """
        Compare past predicted confidence with how things actually turned out
        (success 1, mixed 0.5, failure 0). None until there are enough
        resolved decisions to say anything.
        """
        try:
            rows = self._db.resolved_with_confidence()
        except Exception:
            log.warning("Ares: calibration lookup failed.", exc_info=True)
            return None
        rows = [r for r in rows if r.get("outcome") in _OUTCOME_SCORES]
        n = len(rows)
        if n < _MIN_CALIBRATION_SAMPLES:
            return None
        mean_pred = sum(r["predicted_confidence"] for r in rows) / n
        actual = sum(_OUTCOME_SCORES[r["outcome"]] for r in rows) / n
        return {
            "n": n,
            "mean_predicted": mean_pred,
            "actual_rate": actual,
            "gap": mean_pred - actual,
            # Shrink toward the raw score until there is a decent sample.
            "weight": min(1.0, n / 10.0),
        }

    @staticmethod
    def _calibrated(raw: float, cal: dict) -> float:
        adjusted = raw + (cal["actual_rate"] - cal["mean_predicted"]) * cal["weight"]
        return max(0.05, min(0.95, adjusted))

    def _with_track_record(self, response: str, data: dict, raw_conf: float | None) -> str:
        """Append a calibrated-confidence note once enough outcomes exist (#155)."""
        if raw_conf is None:
            return response
        cal = self._calibration()
        if not cal:
            return response
        adjusted = self._calibrated(raw_conf, cal)
        data["calibrated_confidence"] = round(adjusted, 2)
        if cal["gap"] > 0.05:
            trend = "your earlier picks scored higher than they turned out"
        elif cal["gap"] < -0.05:
            trend = "your earlier picks turned out better than they scored"
        else:
            trend = "your earlier picks have been well calibrated"
        return (
            f"{response}\n\nTRACK RECORD\n"
            f"  Raw confidence {raw_conf:.2f}, calibrated {adjusted:.2f} "
            f"(based on {cal['n']} past outcomes; {trend})"
        )

    # ── career_ranking (#153) ────────────────────────────────────────────────
    # Same toolkit as decision_support, but with a scoped prompt and weighted
    # criteria. The model only scores each option per criterion; the weighted
    # total and the ranking are computed here, because small local models are
    # unreliable at arithmetic.

    @staticmethod
    def _parse_criteria(raw) -> dict[str, int]:
        """'salary:5, location, growth:4' (or a list) -> {name: weight 1-5}."""
        if not raw:
            return {}
        items = raw if isinstance(raw, (list, tuple)) else re.split(r"[;,\n]", str(raw))
        out: dict[str, int] = {}
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            m = re.match(r"^(.*?)\s*[:=]\s*(\d+)$", text)
            name, weight = (m.group(1).strip(), int(m.group(2))) if m else (text, 3)
            if name:
                out[name[:40]] = max(1, min(5, weight))
            if len(out) >= 8:
                break
        return out

    @staticmethod
    def _match_scores(raw_scores, criteria: dict[str, int]) -> dict[str, float]:
        """Map whatever keys the model used back onto the requested criteria."""
        if not isinstance(raw_scores, dict):
            return {}
        by_lower = {c.lower(): c for c in criteria}
        out: dict[str, float] = {}
        for key, val in raw_scores.items():
            try:
                score = float(val)
            except (TypeError, ValueError):
                continue
            k = str(key).strip().lower()
            target = by_lower.get(k)
            if target is None:
                k_words = set(re.findall(r"[a-z]+", k))
                best, best_overlap = None, 0
                for lower, orig in by_lower.items():
                    overlap = len(k_words & set(re.findall(r"[a-z]+", lower)))
                    if overlap > best_overlap:
                        best, best_overlap = orig, overlap
                target = best
            if target is not None and target not in out:
                out[target] = max(1.0, min(10.0, score))
        return out

    def _career_ranking(self, entities: dict, context: dict) -> dict:
        topic = self._topic(entities)
        options = self._extract_options(entities)

        if not options:
            return {
                "response": (
                    f"I can rank your options for {topic}. "
                    "What are they? List them and I'll score each one."
                ),
                "data": {},
                "confidence": 0.6,
            }

        scan = f"{topic} {options} {entities.get('raw_query', '')}"
        gate = (
            str(entities.get("mode", "")).lower() == "gate"
            or re.search(r"\bgate\b", scan, re.IGNORECASE) is not None
        )
        criteria = self._parse_criteria(entities.get("criteria")) or dict(
            _CAREER_GATE_CRITERIA if gate else _CAREER_DEFAULT_CRITERIA
        )
        criteria_str = ", ".join(f"{name} (weight {w}/5)" for name, w in criteria.items())

        raw = self._ollama_call(
            _CAREER_PROMPT.format(
                topic=topic,
                options=options,
                scope_line=_CAREER_GATE_SCOPE if gate else "",
                criteria=criteria_str,
                grounding=self._grounding(context) or "(none)",
            )
        )
        result = self._parse(raw, "career_ranking")
        if not result:
            return {"response": raw or "I had trouble ranking those options.",
                    "data": {}, "confidence": 0.3}

        def _strs(v) -> list[str]:
            return [str(x) for x in v][:3] if isinstance(v, list) else []

        ranked = []
        for o in result.get("options") or []:
            if not isinstance(o, dict) or not o.get("option"):
                continue
            scores = self._match_scores(o.get("scores"), criteria)
            if not scores:
                continue
            total_w = sum(criteria[c] for c in scores)
            total = sum(criteria[c] * s for c, s in scores.items()) / total_w
            ranked.append({
                "option": str(o["option"]),
                "score": round(total, 1),
                "scores": scores,
                "strengths": _strs(o.get("strengths")),
                "risks": _strs(o.get("risks")),
            })
        ranked.sort(key=lambda r: r["score"], reverse=True)

        if not ranked:
            return {"response": "I couldn't get usable scores for those options. Try again, or "
                                "list the options and the criteria that matter to you.",
                    "data": {}, "confidence": 0.3}

        data = {
            "ranking": ranked,
            "criteria": criteria,
            "gate_mode": gate,
            "deciding_factor": result.get("deciding_factor") or "",
            "next_step": result.get("next_step") or "",
            "data_to_verify": _strs(result.get("data_to_verify")),
        }
        response = self._format_career(topic, data)
        self._persist(f"career_{topic}", response)

        raw_conf = ranked[0]["score"] / 10.0
        decision_id, review_note = self._record_decision(
            "career", topic, response, [r["option"] for r in ranked], raw_conf
        )
        if decision_id is not None:
            data["decision_id"] = decision_id
        response = self._with_track_record(response, data, raw_conf)
        if review_note:
            response += "\n\n" + review_note

        if data["next_step"] and self._memory:
            self._schedule_reminder(data["next_step"], 1)

        return {"response": response, "data": data, "confidence": 0.9}

    @staticmethod
    def _format_career(topic: str, data: dict) -> str:
        scope = " (GATE mode)" if data.get("gate_mode") else ""
        lines = [f"Career Ranking: {topic.title()}{scope}", ""]

        lines.append("WEIGHTS")
        lines.append("  " + ", ".join(f"{c} x{w}" for c, w in data["criteria"].items()))
        lines.append("")

        lines.append("RANKING")
        for i, r in enumerate(data["ranking"], 1):
            lines.append(f"  {i}. {r['option']}  [{r['score']:.1f}/10]")
            lines.append("     Scores: " + ", ".join(f"{c} {s:g}" for c, s in r["scores"].items()))
            for s in r["strengths"]:
                lines.append(f"     + {s}")
            for k in r["risks"]:
                lines.append(f"     - {k}")
        lines.append("")

        if data.get("deciding_factor"):
            lines += ["DECIDING FACTOR", f"  {data['deciding_factor']}", ""]
        if data.get("data_to_verify"):
            lines.append("VERIFY BEFORE DECIDING")
            for d in data["data_to_verify"]:
                lines.append(f"  • {d}")
            lines.append("")
        if data.get("next_step"):
            lines += ["NEXT STEP", f"  {data['next_step']}"]

        return "\n".join(lines).strip()

    # ── schedule_review (#154) ───────────────────────────────────────────────

    def _schedule_review(self, entities: dict, context: dict) -> dict:
        topic = (entities.get("topic") or "").strip()
        when_text = f"{entities.get('when', '')} {entities.get('raw_query', '')}"
        days = self._parse_delay_days(when_text) or 14

        row = self._db.find_decision(topic)
        if not topic and not row:
            return {
                "response": (
                    "Which decision or plan should I remind you to revisit? "
                    "I don't have any saved yet."
                ),
                "data": {}, "confidence": 0.5,
            }
        label = topic or row["topic"]
        # A topic that matches nothing saved still gets a reminder; the user
        # may be asking about something decided outside Ares.
        kind = row["kind"] if row and row["kind"] in ("plan", "decision", "career") else "decision"

        if not self._memory:
            return {"response": "Reminders aren't available right now, so I can't schedule a review.",
                    "data": {}, "confidence": 0.3}
        due = self._schedule_reminder(f"Revisit your {kind} on '{label}': how did it turn out?", days)
        if not due:
            return {"response": "I couldn't schedule that reminder. Please try again.",
                    "data": {}, "confidence": 0.3}

        if row:
            self._db.set_review(row["id"], due)
        due_str = datetime.fromisoformat(due).strftime("%d %b %Y")
        extra = "" if row else " (I don't have a saved analysis for that, but the reminder is set.)"
        return {
            "response": (
                f"Okay, I'll remind you on {due_str} to revisit '{label}'.{extra} "
                "When you know how it went, tell me and I'll log the outcome."
            ),
            "data": {"due": due, "days": days, "decision_id": row["id"] if row else None},
            "confidence": 0.9,
        }

    # ── record_outcome / outcome_stats (#155) ────────────────────────────────

    @staticmethod
    def _parse_outcome(text: str) -> str | None:
        t = (text or "").lower()
        if re.search(r"\b(mixed|partly|partially|so-so|half|kind of|sort of|okay-ish|meh)\b", t):
            return "mixed"
        if re.search(
            r"\b(fail(?:ed|ure)?|didn'?t work|did not work|went badly|backfired|"
            r"wrong call|regret|bad idea|flopped|bombed|went wrong)\b", t
        ):
            return "failure"
        if re.search(
            r"\b(success(?:ful|fully)?|succeeded|worked|went well|paid off|"
            r"right call|good call|great|nailed|worked out)\b", t
        ):
            return "success"
        return None

    def _record_outcome(self, entities: dict, context: dict) -> dict:
        topic = (entities.get("topic") or "").strip()
        note = (entities.get("note") or entities.get("raw_query") or "").strip()
        outcome = self._parse_outcome(str(entities.get("outcome") or "")) \
            or self._parse_outcome(note)

        if outcome is None:
            return {
                "response": "How did it turn out: did it work, was it mixed, or did it fail?",
                "data": {}, "confidence": 0.5,
            }

        row = self._db.find_decision(topic, open_only=True) or self._db.find_decision(topic)
        tracked = row is not None
        if row is None:
            if not topic:
                return {"response": "Which decision or plan was that about?",
                        "data": {}, "confidence": 0.5}
            # No saved analysis: log it anyway so the track record is complete.
            # It has no predicted confidence, so it can't skew calibration.
            decision_id = self._db.add_decision("manual", topic, "")
            row = self._db.get_decision(decision_id)

        previous = row.get("outcome")
        self._db.record_outcome(row["id"], outcome, note)

        label = {"success": "worked", "mixed": "turned out mixed", "failure": "failed"}[outcome]
        lines = [f"Logged: '{row['topic']}' {label}."]
        if previous and previous != outcome:
            lines.append(f"(This replaces the earlier outcome: {previous}.)")
        if not tracked:
            lines.append("I had no saved analysis for it, so it won't affect confidence calibration.")
        cal = self._calibration()
        if cal:
            lines.append(
                f"Across {cal['n']} tracked decisions, your average predicted confidence "
                f"was {cal['mean_predicted']:.0%} and the actual success rate {cal['actual_rate']:.0%}; "
                "future decision confidence will be adjusted to match."
            )
        return {
            "response": " ".join(lines),
            "data": {"decision_id": row["id"], "outcome": outcome, "calibration": cal},
            "confidence": 0.9,
        }

    def _outcome_stats(self, entities: dict, context: dict) -> dict:
        counts = self._db.outcome_counts()
        total = sum(counts.values())
        if total == 0:
            return {
                "response": (
                    "I haven't tracked any plans or decisions yet. They're saved "
                    "automatically when I make a plan, decision or career ranking; "
                    "afterwards you can tell me how each one turned out."
                ),
                "data": {}, "confidence": 0.7,
            }

        resolved = {k: v for k, v in counts.items() if k in _OUTCOME_SCORES}
        n_resolved = sum(resolved.values())
        lines = ["Decision Track Record", ""]
        lines.append(
            f"  Tracked: {total}   Resolved: {n_resolved}   Waiting on an outcome: {counts.get('pending', 0)}"
        )
        if n_resolved:
            rate = sum(_OUTCOME_SCORES[k] * v for k, v in resolved.items()) / n_resolved
            lines.append(
                f"  Worked: {resolved.get('success', 0)}   Mixed: {resolved.get('mixed', 0)}   "
                f"Failed: {resolved.get('failure', 0)}   (success rate {rate:.0%})"
            )

        cal = self._calibration()
        lines.append("")
        if cal:
            lines.append("CALIBRATION")
            lines.append(
                f"  Predicted {cal['mean_predicted']:.0%} on average vs {cal['actual_rate']:.0%} actual "
                f"over {cal['n']} decisions."
            )
        else:
            lines.append(
                f"CALIBRATION\n  Needs at least {_MIN_CALIBRATION_SAMPLES} resolved decisions that had a "
                "confidence score (decisions and career rankings do; plans don't)."
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        due = self._db.due_reviews(now_iso)
        if due:
            lines += ["", "DUE FOR REVIEW"]
            for d in due[:5]:
                lines.append(f"  • {d['topic']} (review date {d['review_at'][:10]})")

        recent = self._db.recent(5)
        if recent:
            lines += ["", "RECENT"]
            for d in recent:
                lines.append(f"  • {d['topic']} [{d['kind']}]: {d['outcome'] or 'no outcome yet'}")

        return {"response": "\n".join(lines),
                "data": {"counts": counts, "calibration": cal, "due_reviews": len(due)},
                "confidence": 0.9}

    # ── simulate_outcomes (#156) ─────────────────────────────────────────────

    def _simulate_outcomes(self, entities: dict, context: dict) -> dict:
        topic = (entities.get("topic") or "this decision").strip()
        raw_query = entities.get("raw_query") or ""

        options = None
        ent_options = entities.get("options")
        if isinstance(ent_options, list) and ent_options and all(isinstance(o, dict) for o in ent_options):
            options = ent_options
        elif isinstance(entities.get("scenarios"), list) and entities["scenarios"]:
            options = [{"name": entities.get("name") or "Option",
                        "scenarios": entities["scenarios"], "cost": entities.get("cost")}]
        elif entities.get("low") is not None and entities.get("high") is not None:
            options = [{"name": entities.get("name") or "Option", "low": entities["low"],
                        "likely": entities.get("likely"), "high": entities["high"],
                        "cost": entities.get("cost")}]

        try:
            if options is None:
                parsed = _sim.parse_text(f"{entities.get('text', '')} {raw_query}")
                if parsed:
                    if entities.get("cost") not in (None, ""):
                        parsed["cost"] = entities["cost"]
                    options = [parsed]

            if not options:
                return {
                    "response": (
                        "To simulate it I need numbers. Give me either scenarios, e.g. "
                        "\"60% chance of 10 lakh, 40% chance of losing 2 lakh\", or a range, e.g. "
                        "\"between 50k and 200k, most likely 100k\". You can add a fixed cost too."
                    ),
                    "data": {}, "confidence": 0.5,
                }

            runs = entities.get("runs") or _sim.DEFAULT_RUNS
            seed = entities.get("seed")
            result = _sim.run(
                options,
                runs=int(runs),
                seed=int(seed) if seed not in (None, "") else None,
            )
        except _sim.SimulationInputError as e:
            return {"response": f"I couldn't run that simulation: {e}",
                    "data": {}, "confidence": 0.4}
        except (TypeError, ValueError):
            return {"response": "I couldn't run that simulation: the numbers weren't in a form I could read.",
                    "data": {}, "confidence": 0.4}

        response = _sim.format_result(topic, result)
        self._persist(f"simulation_{topic}", response)
        return {"response": response, "data": result, "confidence": 0.95}

    # ── playbooks (#157) ─────────────────────────────────────────────────────

    @staticmethod
    def _detect_analysis(text: str) -> str | None:
        """Internal intent named in *text*, by earliest keyword match."""
        t = (text or "").lower().replace("_", " ")
        best, best_pos = None, None
        for intent, words in _PLAYBOOK_ANALYSES:
            if t.strip() == intent.replace("_", " "):
                return intent
            for w in words:
                m = re.search(r"\b" + re.escape(w) + r"\b", t)
                if m and (best_pos is None or m.start() < best_pos):
                    best, best_pos = intent, m.start()
        return best

    @staticmethod
    def _playbook_name_from_text(raw: str) -> str:
        m = re.search(
            r"(?:called|named|name it|titled)\s+[\"'“‘]?(.+?)[\"'”’]?"
            r"(?=\s+(?:that|which|to|with|for|and|runs?|using)\b|[,.;:]|$)",
            raw, re.IGNORECASE,
        )
        if not m:
            m = re.search(
                r"playbook\s+for\s+[\"'“‘]?(.+?)[\"'”’]?"
                r"(?=\s+(?:that|which|with|using|runs?)\b|[,.;:]|$)",
                raw, re.IGNORECASE,
            )
        return m.group(1).strip() if m else ""

    def _save_playbook(self, entities: dict, context: dict) -> dict:
        raw = entities.get("raw_query") or ""
        name = (entities.get("name") or entities.get("playbook") or "").strip() \
            or self._playbook_name_from_text(raw)
        name = re.sub(r"\s+", " ", name)[:60]
        if not name:
            return {"response": "What should I call this playbook?",
                    "data": {}, "confidence": 0.5}

        analysis_src = entities.get("analysis") or entities.get("type") or entities.get("framework")
        if analysis_src:
            analysis = self._detect_analysis(str(analysis_src))
        else:
            analysis = self._detect_analysis(raw.lower().replace(name.lower(), " "))
        if not analysis:
            return {
                "response": (
                    f"Which analysis should the '{name}' playbook run: SWOT, premortem, risk, "
                    "strategic plan, decision support, competitive, contingency or career ranking?"
                ),
                "data": {}, "confidence": 0.5,
            }

        criteria = entities.get("criteria") or entities.get("focus") or ""
        if isinstance(criteria, (list, tuple)):
            criteria = "; ".join(str(c) for c in criteria)
        criteria = str(criteria).strip()
        if not criteria:
            m = re.search(
                r"(?:always\s+)?(?:consider(?:s|ing)?|check(?:s|ing)?|include(?:s)?|cover(?:s)?|"
                r"weigh(?:s)?|criteria(?:\s+(?:are|is))?|focus(?:es)?\s+on)\s*:?\s+(.+?)[.]?$",
                raw, re.IGNORECASE,
            )
            criteria = m.group(1).strip() if m else ""

        options = entities.get("options") or ""
        if isinstance(options, (list, tuple)):
            options = ", ".join(str(o) for o in options)

        existed = self._db.save_playbook(
            name, analysis, criteria[:600], str(entities.get("topic") or "").strip(), str(options)
        )
        label = _PLAYBOOK_LABELS[analysis]
        verb = "Updated" if existed else "Saved"
        crit_note = f" It will always cover: {criteria}." if criteria else ""
        return {
            "response": (
                f"{verb} playbook '{name}' ({label}).{crit_note} "
                f"Run it any time with \"run my {name} playbook on <topic>\"."
            ),
            "data": {"name": name, "analysis": analysis, "criteria": criteria, "updated": existed},
            "confidence": 0.9,
        }

    def _list_playbooks(self, entities: dict, context: dict) -> dict:
        books = self._db.list_playbooks()
        if not books:
            return {
                "response": (
                    "You haven't saved any playbooks yet. Try: \"save a playbook called job offer "
                    "that runs a premortem and always considers salary, growth and commute\"."
                ),
                "data": {"playbooks": []}, "confidence": 0.8,
            }
        lines = ["Playbooks", ""]
        for b in books:
            line = f"  • {b['name']}: {_PLAYBOOK_LABELS.get(b['analysis'], b['analysis'])}"
            if b.get("criteria"):
                crit = b["criteria"] if len(b["criteria"]) <= 80 else b["criteria"][:77] + "..."
                line += f" (covers {crit})"
            if b.get("uses"):
                line += f", used {b['uses']}x"
            lines.append(line)
        return {"response": "\n".join(lines), "data": {"playbooks": books}, "confidence": 0.9}

    def _lookup_playbook(self, entities: dict) -> tuple[dict | None, str]:
        """(playbook, original_text_searched). Exact name first, then name-in-sentence."""
        name = (entities.get("name") or entities.get("playbook") or "").strip()
        raw = entities.get("raw_query") or ""
        pb = self._db.get_playbook(name) if name else None
        if pb is None and name:
            pb = self._db.find_playbook_in_text(name)
        if pb is None and raw:
            pb = self._db.find_playbook_in_text(raw)
        return pb, (name or raw)

    def _run_playbook(self, entities: dict, context: dict) -> dict:
        pb, searched = self._lookup_playbook(entities)
        if pb is None:
            names = [b["name"] for b in self._db.list_playbooks()]
            have = f" You have: {', '.join(names)}." if names else " You haven't saved any yet."
            what = f"'{searched}'" if searched else "that"
            return {"response": f"I don't have a playbook called {what}.{have}",
                    "data": {}, "confidence": 0.5}

        topic = (entities.get("topic") or "").strip()
        if not topic:
            raw = entities.get("raw_query") or ""
            idx = raw.lower().find(pb["name_key"])
            rest = raw[idx + len(pb["name_key"]):] if idx >= 0 else raw
            rest = re.sub(r"^\s*playbook\b", "", rest, flags=re.IGNORECASE)
            m = re.search(r"\b(?:on|for|about|regarding)\s+(.+?)[.?!]*$", rest, re.IGNORECASE)
            topic = m.group(1).strip() if m else ""
        topic = topic or (pb.get("topic") or "").strip()
        if not topic:
            return {"response": f"What should I run the '{pb['name']}' playbook on?",
                    "data": {}, "confidence": 0.5}

        sub_entities = {"topic": topic}
        options = entities.get("options") or pb.get("options") or ""
        if options:
            sub_entities["options"] = options
        sub_context = dict(context or {})
        if pb.get("criteria"):
            sub_context["_ares_focus"] = pb["criteria"]

        result = self.handle(pb["analysis"], sub_entities, sub_context)
        self._db.touch_playbook(pb["name"])

        result = dict(result)
        result["response"] = f"Playbook: {pb['name']}\n\n{result['response']}"
        data = dict(result.get("data") or {})
        data["playbook"] = pb["name"]
        result["data"] = data
        return result

    def _delete_playbook(self, entities: dict, context: dict) -> dict:
        pb, searched = self._lookup_playbook(entities)
        if pb is None:
            return {"response": f"I couldn't find a playbook called '{searched}'." if searched
                    else "Which playbook should I delete?",
                    "data": {}, "confidence": 0.5}
        self._db.delete_playbook(pb["name"])
        return {"response": f"Deleted playbook '{pb['name']}'.",
                "data": {"name": pb["name"]}, "confidence": 0.9}
