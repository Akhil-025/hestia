"""
core/conference.py  (backlog #158: multi-module "conference")

Some questions don't belong to one module. "Can I afford to cut my hours and
study for GATE?" has a money side (Pluto), a strategy side (Ares), a
workload side (Artemis) and a wellbeing side (Apollo). Answering it from any
single module gives a confident, partial answer.

A conference asks each relevant module for its own read of the situation,
puts them side by side, and says where they agree and where they pull apart.

How it stays inside the architecture
------------------------------------
* **Hecate decides who attends** (``HecateEngine._conference_modules``) and
  marks the decision with ``conference=[...]``. This file never routes.
* **The orchestrator convenes it.** It is the only component that holds every
  module, so it passes in a ``call`` function. Nothing here imports a module,
  and no module calls another. The ``call`` it passes goes through the same
  per-module circuit breakers as any dispatch, so a broken module just
  doesn't get a seat instead of breaking the answer.
* **Each seat uses one read-only "lens".** ``LENSES`` names the intent to ask
  each module. Every one is a pure read: no reminders, no saved decisions, no
  LLM calls inside the module. (Ares's ``decision_support`` would be the
  obvious Ares lens, but it saves the decision and schedules a reminder, so a
  conference asking it would quietly write to your data. ``outcome_stats``
  reads the same track record and writes nothing.)
* **No LLM is required.** If a ``synthesize`` callable is supplied it is used
  to write a short summary; if it is missing, fails or returns nothing, a
  deterministic summary is used. The perspectives themselves are always listed
  verbatim underneath, so nothing the summary says can't be checked.
* **Tension is surfaced, not resolved.** If Apollo and Artemis are both
  seated and ``core/consensus.py`` sees genuine rest-vs-push disagreement,
  that note is added as is. A conference never picks a winner on your behalf.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# module -> (intent to ask, label shown to the user). Every intent here must
# be a pure read. tests/test_conference.py checks the table against each
# module's real _INTENTS so a rename can't silently strand a seat.
LENSES: dict[str, tuple[str, str]] = {
    "pluto":   ("financial_health", "Money (Pluto)"),
    "ares":    ("outcome_stats", "Decision track record (Ares)"),
    "apollo":  ("get_health_summary", "Health (Apollo)"),
    "artemis": ("productivity_summary", "Habits and goals (Artemis)"),
    "chronos": ("weekly_focus", "This week's load (Chronos)"),
}

# A perspective longer than this is trimmed at a sentence/line boundary so one
# chatty module can't bury the others.
MAX_PERSPECTIVE_CHARS = 420
MIN_VOICES = 2

def ollama_synthesizer(ollama_cfg: Optional[dict] = None, timeout: float = 8.0) -> "SynthFn":
    """A ``synthesize`` callable backed by the local Ollama model. Returns
    "" on any failure, which ``Conference`` treats as "use the plain summary"."""
    cfg = ollama_cfg or {}

    def _synth(prompt: str) -> Optional[str]:
        from core.ollama_client import generate  # late import: optional dependency
        return generate(
            prompt,
            model=cfg.get("model", "mistral"),
            host=cfg.get("host", "127.0.0.1"),
            port=cfg.get("port", 11434),
            timeout=timeout,
        )

    return _synth


CallFn = Callable[[str, str, dict, dict], Optional[dict]]
SynthFn = Callable[[str], Optional[str]]


def _trim(text: str, limit: int = MAX_PERSPECTIVE_CHARS) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", "; ", ", "):
        i = cut.rfind(sep)
        if i >= limit // 2:
            return cut[: i + 1].rstrip() + " …"
    return cut.rstrip() + " …"


class Conference:
    """Gather several modules' views of one question and set them side by side."""

    def __init__(
        self,
        call: CallFn,
        *,
        synthesize: Optional[SynthFn] = None,
        consensus: Optional[Any] = None,
        lenses: Optional[dict[str, tuple[str, str]]] = None,
    ) -> None:
        self._call = call
        self._synthesize = synthesize
        self._consensus = consensus
        self.lenses = dict(lenses) if lenses is not None else dict(LENSES)

    # ------------------------------------------------------------------
    # Gathering
    # ------------------------------------------------------------------

    def gather(self, query: str, seats: list[str], context: Optional[dict] = None) -> list[dict]:
        """Ask each seated module for its view. Never raises."""
        out: list[dict] = []
        for name in seats:
            lens = self.lenses.get(name)
            if lens is None:
                out.append({"module": name, "label": name.capitalize(), "text": "",
                            "usable": False, "why": "no read-only view defined for this module"})
                continue
            intent, label = lens
            entry = {"module": name, "label": label, "intent": intent,
                     "text": "", "usable": False, "why": ""}
            try:
                result = self._call(name, intent, {"raw_query": query}, dict(context or {}))
            except Exception:
                logger.exception("conference: %s lens raised.", name)
                result = None
            if not isinstance(result, dict):
                entry["why"] = "didn't answer"
                out.append(entry)
                continue
            text = _trim(str(result.get("response") or ""))
            data = result.get("data")
            try:
                conf = float(result.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            entry["text"] = text
            # A reply with no data behind it ("I haven't tracked anything yet")
            # is an honest answer but not a perspective, so it doesn't count
            # toward having enough voices.
            if text and (data or conf >= 0.85):
                entry["usable"] = True
            else:
                entry["why"] = "nothing recorded yet" if text else "didn't answer"
            out.append(entry)
        return out

    # ------------------------------------------------------------------
    # Tension
    # ------------------------------------------------------------------

    def tension(self, seats: list[str]) -> Optional[str]:
        """The consensus engine's rest-vs-push note, if both sides are seated
        and the signals genuinely disagree."""
        if self._consensus is None or "apollo" not in seats or "artemis" not in seats:
            return None
        try:
            found = self._consensus.evaluate(
                self._consensus.rest_signals() + self._consensus.push_signals()
            )
            return self._consensus.describe(found) if found else None
        except Exception:
            logger.exception("conference: tension check failed; leaving it out.")
            return None

    # ------------------------------------------------------------------
    # Convening
    # ------------------------------------------------------------------

    def convene(self, query: str, seats: list[str], context: Optional[dict] = None) -> dict:
        """Run the conference. Returns ``{"response", "data", "confidence"}``;
        never raises."""
        try:
            return self._convene(query, list(seats or []), context)
        except Exception:
            logger.exception("conference: convene failed.")
            return {
                "response": "I couldn't pull those views together just now.",
                "data": {"convened": False},
                "confidence": 0.2,
            }

    def _convene(self, query: str, seats: list[str], context: Optional[dict]) -> dict:
        views = self.gather(query, seats, context)
        voices = [v for v in views if v["usable"]]
        silent = [v for v in views if not v["usable"]]
        data = {
            "convened": len(voices) >= MIN_VOICES,
            "seats": seats,
            "perspectives": [
                {k: v[k] for k in ("module", "label", "text", "usable", "why") if k in v}
                for v in views
            ],
        }

        if len(voices) < MIN_VOICES:
            who = ", ".join(v["label"] for v in voices) or "none of them"
            gaps = "; ".join(f"{v['label']}: {v['why']}" for v in silent)
            return {
                "response": (
                    "I don't have enough recorded to weigh these against each "
                    f"other \u2014 only {who} had something to say"
                    + (f" ({gaps})." if gaps else ".")
                    + " Log a little in the quiet areas and ask again."
                ),
                "data": data,
                "confidence": 0.5,
            }

        tension = self.tension([v["module"] for v in voices])
        data["tension"] = tension

        summary = self._summary(query, voices, tension)
        parts: list[str] = []
        if summary:
            parts.append(summary)
        parts.append("What each part of my picture says:")
        parts.extend(f"- {v['label']}: {v['text']}" for v in voices)
        if silent:
            parts.append(
                "Nothing to add from: "
                + "; ".join(f"{v['label']} ({v['why']})" for v in silent) + "."
            )
        if tension:
            parts.append(tension)
        return {"response": "\n".join(parts), "data": data, "confidence": 0.85}

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _summary(self, query: str, voices: list[dict], tension: Optional[str]) -> str:
        if self._synthesize is not None:
            try:
                text = self._synthesize(self._prompt(query, voices, tension))
            except Exception:
                logger.exception("conference: synthesis failed; using the plain summary.")
                text = None
            if text and text.strip():
                return text.strip()
        names = [v["label"].split(" (")[0].lower() for v in voices]
        joined = ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else names[0]
        return (
            f"I looked at {joined} together. I haven't merged them into one "
            "verdict, because they're answering different questions; here they "
            "are side by side."
        )

    @staticmethod
    def _prompt(query: str, voices: list[dict], tension: Optional[str]) -> str:
        block = "\n".join(f"[{v['label']}] {v['text']}" for v in voices)
        extra = f"\nKnown disagreement: {tension}\n" if tension else ""
        return (
            f'The user asked: "{query}"\n\n'
            "Several parts of their own data each gave a view:\n"
            f"{block}\n{extra}\n"
            "In at most 80 words, say where these views agree and where they "
            "pull in different directions. Use only facts stated above, "
            "invent nothing, and do not pick a winner: finish by naming the "
            "one trade-off the user has to decide."
        )
