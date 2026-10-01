"""
core/consensus.py  (backlog #159: tension surfacing)

Sometimes two of Hestia's own modules would advise opposite things: Apollo's
data says "rest" (short sleep, low mood) while Artemis says "push" (a habit
streak about to break). Left alone, whichever module answers just answers, and
the user never learns the advice conflicts.

This layer notices that and *says so*. It never overrides or rewrites a
module's answer: it only appends a short note after it, explaining which
signals disagree and why. It is deliberately small:

- a fixed, weighted rule set (no LLM, so it is predictable and testable);
- it only runs on a short whitelist of intents where the conflict matters;
- it needs a real signal on *both* sides before it says anything;
- a config kill switch (``consensus.enabled: false``) stops it being attached.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Intent names as the orchestrator sees them (module prefix already stripped).
DEFAULT_INTENTS: frozenset[str] = frozenset(
    {"complete_habit", "get_motivation", "suggest_exercise", "suggest_activity"}
)

REST_THRESHOLD = 1.0   # total 'rest' weight needed before we speak
PUSH_THRESHOLD = 0.5   # total 'push' weight needed before we speak
_STREAK_MIN = 3        # a streak shorter than this isn't worth defending


class ConsensusEngine:
    """Detect and describe disagreement between Apollo and Artemis signals."""

    def __init__(
        self,
        apollo: Optional[Any] = None,
        artemis: Optional[Any] = None,
        intents: Optional[Iterable[str]] = None,
        enabled: bool = True,
        rest_threshold: float = REST_THRESHOLD,
        push_threshold: float = PUSH_THRESHOLD,
    ) -> None:
        self.apollo = apollo
        self.artemis = artemis
        self.intents = frozenset(intents) if intents else DEFAULT_INTENTS
        self.enabled = enabled
        self.rest_threshold = rest_threshold
        self.push_threshold = push_threshold

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def rest_signals(self, now: Optional[datetime] = None) -> list[dict[str, Any]]:
        if self.apollo is None or not hasattr(self.apollo, "rest_signals"):
            return []
        try:
            return list(self.apollo.rest_signals(now))
        except Exception:
            logger.exception("consensus: apollo.rest_signals failed.")
            return []

    def push_signals(self, now: Optional[datetime] = None) -> list[dict[str, Any]]:
        tracker = getattr(self.artemis, "tracker", None)
        if tracker is None:
            return []
        today = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date()
        yesterday = (today - timedelta(days=1)).isoformat()
        out: list[dict[str, Any]] = []
        try:
            for name, habit in tracker.get_habits().items():
                # An at-risk streak: alive (done yesterday), not done today.
                if habit.streak >= _STREAK_MIN and habit.last_done == yesterday:
                    out.append({
                        "direction": "push", "source": "artemis",
                        "weight": 0.5 + min(habit.streak, 14) / 14 * 0.5,
                        "reason": f"your {habit.streak}-day '{name}' streak ends if it's skipped today",
                    })
        except Exception:
            logger.exception("consensus: reading Artemis habits failed.")
        try:
            at_risk = tracker.get_at_risk_goals(today=today)
            count = len(at_risk) if at_risk else 0
            if count:
                out.append({
                    "direction": "push", "source": "artemis", "weight": 0.5,
                    "reason": f"{count} goal(s) are due soon and behind",
                })
        except Exception:
            pass  # goals are a secondary signal; habits alone are enough
        return out

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, signals: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        rest = [s for s in signals if s["direction"] == "rest"]
        push = [s for s in signals if s["direction"] == "push"]
        rest_w = sum(s["weight"] for s in rest)
        push_w = sum(s["weight"] for s in push)
        if rest_w < self.rest_threshold or push_w < self.push_threshold:
            return None
        return {"rest": rest, "push": push, "rest_weight": rest_w, "push_weight": push_w}

    @staticmethod
    def describe(tension: dict[str, Any]) -> str:
        rest = "; ".join(s["reason"] for s in tension["rest"])
        push = "; ".join(s["reason"] for s in tension["push"])
        rw, pw = tension["rest_weight"], tension["push_weight"]
        if rw >= pw + 1.0:
            lean = f"the rest signals look stronger right now ({rw:.1f} vs {pw:.1f})"
        elif pw >= rw + 1.0:
            lean = f"the push signals look stronger right now ({pw:.1f} vs {rw:.1f})"
        else:
            lean = f"they're fairly evenly matched ({rw:.1f} vs {pw:.1f})"
        return (
            "Heads-up: two things in your own data point in different directions. "
            f"Rest: {rest}. Push: {push}. I'm not quietly picking one — {lean}. "
            "A lighter version of the habit might serve both; it's your call."
        )

    def note_for(self, intent: str, now: Optional[datetime] = None) -> Optional[str]:
        """The disagreement note for *intent*, or None (disabled, not
        whitelisted, or no genuine conflict)."""
        if not self.enabled or intent not in self.intents:
            return None
        try:
            tension = self.evaluate(self.rest_signals(now) + self.push_signals(now))
            return self.describe(tension) if tension else None
        except Exception:
            logger.exception("consensus: note_for failed; staying silent.")
            return None

    def annotate(self, intent: str, text: str, now: Optional[datetime] = None) -> str:
        """Append the note to *text* when there is one. Never alters *text*
        itself, and never raises."""
        note = self.note_for(intent, now)
        if not note or note in text:
            return text
        return f"{text.rstrip()}\n\n{note}"
