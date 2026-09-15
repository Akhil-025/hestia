"""

modules/pluto/advisor_agent.py
A genuinely multi-step financial-advisor agent built on langchain_core,
replacing the previously-unused `langchain` entry in requirements.txt.

Why a hand-rolled ReAct loop instead of langchain's `create_agent`:
`create_agent` (langchain >= 1.0) requires a chat model with native
tool-calling support. Hestia's default local model (Ollama "mistral",
per laptop_config.yaml) does not reliably emit the structured
tool-call format that requires — plugging it into `create_agent` would
either silently fail to call tools or raise on malformed tool-call
JSON. A classic text-based ReAct loop (Thought / Action / Action Input
/ Observation) degrades far more gracefully with a small local model:
worst case it can't parse a step and falls back to a direct answer,
rather than crashing.

This still uses real langchain primitives: `langchain_core.tools.Tool`
for the tool interface, `langchain_core.language_models.llms.LLM` as
the base class for the Ollama-backed model, and
`langchain_core.prompts.PromptTemplate` for the agent scaffold — it is
not a from-scratch reimplementation of langchain's concepts, just of
the outer control loop.

Scope: read-only / informational tools only (budget summary, spending
report, currency conversion, company lookup). `log_expense` is
deliberately NOT exposed to the agent — letting a multi-step LLM loop
autonomously decide to write financial records is a foot-gun; expense
logging stays on Hestia's normal, single-intent path.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from .llm_client import LLMClient, OutputFormat
from .logging_config import get_logger

logger = get_logger(__name__)

MAX_AGENT_STEPS = 4

_REACT_PROMPT = PromptTemplate.from_template(
    """You are Pluto's financial advisor agent. Answer the user's question by \
using the tools below when they help; otherwise answer directly.

Tools:
{tools}

Use exactly this format:

Question: the user's question
Thought: reason about what to do next
Action: the tool name, one of [{tool_names}]
Action Input: the input to the tool
Observation: the tool's result
... (Thought/Action/Action Input/Observation can repeat up to {max_steps} times)
Thought: I now know the final answer
Final Answer: the final answer to the user, under 150 words

Begin.

Question: {question}
{agent_scratchpad}"""
)


class _PlutoOllamaLLM(LLM):
    """
    Adapts Pluto's existing LLMClient (Ollama-or-Kimi, with retry) to
    langchain_core's `LLM` interface, so the ReAct loop below is a real
    langchain LLM call rather than a bespoke HTTP client.
    """

    # Typed as `Any` (not `LLMClient`) deliberately: pydantic would otherwise
    # reject duck-typed test doubles (e.g. a FakeLLMClient with a matching
    # `.generate()` method but no shared base class) via isinstance checks.
    llm_client: Any

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "pluto_ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        text = self.llm_client.generate(prompt, output_format=OutputFormat.TEXT)
        if stop:
            for token in stop:
                idx = text.find(token)
                if idx != -1:
                    text = text[:idx]
        return text


_ACTION_RE = re.compile(r"Action:\s*(.+?)\s*\n", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)


class FinancialAdvisorAgent:
    """
    Runs a bounded ReAct loop over a small toolset backed by
    PersonalFinanceManager, using langchain_core primitives.
    """

    def __init__(self, pf_manager, llm_client: Optional[LLMClient] = None):
        self.pf_manager = pf_manager
        self._llm_client = llm_client or pf_manager.llm_client
        self._llm = _PlutoOllamaLLM(llm_client=self._llm_client)
        self.tools: List[Tool] = self._build_tools()

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _build_tools(self) -> List[Tool]:
        return [
            Tool(
                name="get_budget_summary",
                description=(
                    "Get the user's spending broken down by category with AI advice. "
                    "Takes no meaningful input; pass 'none'."
                ),
                func=lambda _input: self._safe(self.pf_manager.budget_summary),
            ),
            Tool(
                name="get_spending_report",
                description=(
                    "Get a full spending report across all logged transactions. "
                    "Takes no meaningful input; pass 'none'."
                ),
                func=lambda _input: self._safe(self.pf_manager.spending_report),
            ),
            Tool(
                name="convert_currency",
                description=(
                    "Convert an amount between currencies. Input format: "
                    "'AMOUNT FROM_CCY TO_CCY', e.g. '100 USD INR'."
                ),
                func=self._tool_convert_currency,
            ),
            Tool(
                name="company_lookup",
                description=(
                    "Look up public SEC EDGAR / macro (FRED) data for a company name. "
                    "Input: the company name, e.g. 'Apple'."
                ),
                func=self._tool_company_lookup,
            ),
        ]

    def _safe(self, fn) -> str:
        try:
            result = fn()
            return result.get("response", "") or "No data available."
        except Exception as e:
            logger.warning("advisor_agent tool call failed: %s", e)
            return f"Tool call failed: {e}"

    def _tool_convert_currency(self, raw_input: str) -> str:
        parts = raw_input.replace(",", " ").split()
        if len(parts) < 3:
            return "Invalid input — expected 'AMOUNT FROM_CCY TO_CCY', e.g. '100 USD INR'."
        try:
            amount = float(parts[0])
        except ValueError:
            return f"Couldn't parse an amount from {parts[0]!r}."
        entities = {"amount": amount, "from_currency": parts[1], "to_currency": parts[2]}
        return self._safe(lambda: self.pf_manager.convert_currency(entities))

    def _tool_company_lookup(self, raw_input: str) -> str:
        entities = {"company": raw_input.strip()}
        return self._safe(lambda: self.pf_manager.company_lookup(entities))

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    def _tool_by_name(self, name: str) -> Optional[Tool]:
        name = name.strip().lower()
        for tool in self.tools:
            if tool.name.lower() == name:
                return tool
        return None

    def ask(self, question: str) -> dict:
        if not question or not question.strip():
            return {
                "response": "What would you like financial advice about?",
                "data": {},
                "confidence": 0.4,
            }

        tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in self.tools)
        tool_names = ", ".join(t.name for t in self.tools)
        scratchpad = ""
        transcript: list[dict] = []

        for step in range(MAX_AGENT_STEPS):
            prompt = _REACT_PROMPT.format(
                tools=tool_descriptions,
                tool_names=tool_names,
                question=question,
                max_steps=MAX_AGENT_STEPS,
                agent_scratchpad=scratchpad,
            )
            try:
                completion = self._llm.invoke(
                    prompt, stop=["Observation:", "\nQuestion:"]
                )
            except Exception as e:
                logger.exception("advisor_agent: LLM call failed")
                return {
                    "response": "I couldn't reach the language model to answer that.",
                    "data": {"transcript": transcript},
                    "confidence": 0.0,
                }

            final_match = _FINAL_ANSWER_RE.search(completion)
            if final_match:
                answer = final_match.group(1).strip()
                return {
                    "response": answer,
                    "data": {"steps_used": step + 1, "transcript": transcript},
                    "confidence": 0.85,
                }

            action_match = _ACTION_RE.search(completion)
            input_match = _ACTION_INPUT_RE.search(completion)
            if not action_match or not input_match:
                # Model didn't follow the format — fall back to treating its
                # own text as the answer rather than looping forever.
                fallback = completion.strip() or "I wasn't able to work out an answer."
                return {
                    "response": fallback,
                    "data": {"steps_used": step + 1, "transcript": transcript, "unparsed": True},
                    "confidence": 0.4,
                }

            action_name = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            tool = self._tool_by_name(action_name)
            if tool is None:
                observation = f"Unknown tool {action_name!r}. Available: {tool_names}."
            else:
                observation = tool.func(action_input)

            transcript.append(
                {"action": action_name, "input": action_input, "observation": observation}
            )
            scratchpad += (
                f"Thought: (working)\nAction: {action_name}\nAction Input: {action_input}\n"
                f"Observation: {observation}\n"
            )

        return {
            "response": (
                "I looked into that across a few steps but couldn't reach a final "
                "answer in time — try asking a more specific question."
            ),
            "data": {"steps_used": MAX_AGENT_STEPS, "transcript": transcript, "exhausted": True},
            "confidence": 0.3,
        }
