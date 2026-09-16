# modules/hestia/core_module.py

import logging
from modules.base import BaseModule
from core.ollama_client import generate, generate_stream
import platform, datetime, re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)


def _safe_int(value, default: int) -> int:
    """Coerce *value* to int, falling back to *default* if that fails."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class CoreModule(BaseModule):
    name = "core"

    _INTENTS = {
        "save_name", "take_note", "get_notes", "delete_notes",
        "get_history", "set_preference",
        "get_system_info", "get_user_info", "chat",
        # Diagnostics (backlog #3, #8, #259). Answered from the injected
        # Diagnostics object (core/observability.py) rather than from any
        # module-specific state, so "are your modules up?" still works
        # when the module being asked about is the broken one.
        "modules_status", "explain_routing", "report_mistake",
    }

    def __init__(self, memory, ollama_cfg: dict, llm=None, timezone_name: str = "UTC",
                 diagnostics=None):
        self._memory = memory
        self._ollama = ollama_cfg
        self._llm_instance = llm  # HestiaLLM | None — preferred path
        # ChronosEngine and HermesEngine both resolve the user's configured
        # IANA timezone (main.py passes chronos.timezone to both) so "what
        # time is it" / calendar events land on local wall-clock time
        # instead of the server's own clock. CoreModule.get_user_info()
        # and get_system_info() need the same thing: they're the recovery
        # path the NLU falls into when it misclassifies date/time questions
        # (see _get_user_info below), so answering with server-local time
        # instead of the user's local time would silently disagree with
        # what Chronos would have said for the exact same question.
        # Injected by main.Hestia after the orchestrator exists (see
        # core/observability.Diagnostics). Optional: when absent, the three
        # diagnostic intents degrade to an honest "diagnostics aren't
        # wired up" reply instead of raising — CoreModule is constructed
        # in tests without it.
        self._diagnostics = diagnostics
        try:
            self._tz = ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, KeyError):
            logger.warning(
                "Unrecognised timezone %r; falling back to UTC.", timezone_name
            )
            self._tz = ZoneInfo("UTC")

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        raw = entities.get("raw_query", "")

        if intent == "chat":
            return self._chat(raw)

        if intent == "get_system_info":
            return self._sys_info()

        if intent == "save_name":
            return self._save_name(entities)

        if intent == "take_note":
            return self._take_note(entities, raw)

        if intent == "get_notes":
            return self._get_notes(entities)

        if intent == "delete_notes":
            return self._delete_notes(entities)

        if intent == "get_history":
            return self._get_history(entities)

        if intent == "get_user_info":
            return self._get_user_info(entities, raw)

        if intent == "set_preference":
            return self._set_preference(entities)

        if intent == "modules_status":
            return self._modules_status()

        if intent == "explain_routing":
            return self._explain_routing()

        if intent == "report_mistake":
            return self._report_mistake(entities, raw)

        return {"response": "", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        return {}

    # ───── handlers ─────

    # -- Diagnostics (backlog #3, #8, #259) --------------------------------
    #
    # All three read from the injected Diagnostics object and touch no
    # other module. They exist as intents (rather than as CLI flags only)
    # because the moment you actually want them is mid-conversation, when
    # something just went to the wrong place — including in voice mode,
    # where there is no CLI to drop to.

    _NO_DIAGNOSTICS = (
        "Diagnostics aren't wired up in this session, so I can't answer that."
    )

    def _modules_status(self) -> dict:
        """One-shot health report across every registered module (#8)."""
        if self._diagnostics is None:
            return {"response": self._NO_DIAGNOSTICS, "data": {}, "confidence": 0.3}
        try:
            summary = self._diagnostics.status_summary()
            data = self._diagnostics.module_status()
        except Exception:
            logger.exception("modules_status failed.")
            return {
                "response": "I couldn't read the module registry just now.",
                "data": {},
                "confidence": 0.2,
            }
        return {"response": summary, "data": {"modules": data}, "confidence": 0.95}

    def _explain_routing(self) -> dict:
        """Explain where the *previous* query was routed, and why (#3)."""
        if self._diagnostics is None:
            return {"response": self._NO_DIAGNOSTICS, "data": {}, "confidence": 0.3}
        try:
            explanation = self._diagnostics.explain_last()
            data = self._diagnostics.last_decision() or {}
        except Exception:
            logger.exception("explain_routing failed.")
            return {
                "response": "I couldn't reconstruct that routing decision.",
                "data": {},
                "confidence": 0.2,
            }
        return {"response": explanation, "data": data, "confidence": 0.95}

    def _report_mistake(self, entities: dict, raw: str) -> dict:
        """
        Log an explicit correction against the previous turn (#259).

        The note is whatever the user said beyond the trigger phrase, so
        "that was wrong, I meant my sleep log" keeps the useful half. The
        record is a labelled data point for the eval set, not just a
        logged complaint — see scripts/eval_intents.py.
        """
        if self._diagnostics is None:
            return {"response": self._NO_DIAGNOSTICS, "data": {}, "confidence": 0.3}
        note = str(entities.get("note") or entities.get("detail") or raw or "")
        try:
            message = self._diagnostics.record_feedback(note)
        except Exception:
            logger.exception("report_mistake failed.")
            return {
                "response": "I couldn't save that feedback, sorry.",
                "data": {},
                "confidence": 0.2,
            }
        return {"response": message, "data": {"note": note}, "confidence": 0.95}


    def _chat(self, query: str) -> dict:
        try:
            prompt = f"You are Hestia. Answer concisely in 1-2 sentences.\n\nQuestion: {query}"
            if self._llm_instance is not None:
                text = self._llm_instance.generate(prompt)
            else:
                text = generate(
                    prompt,
                    model=self._ollama.get("model", "mistral"),
                    host=self._ollama.get("host", "127.0.0.1"),
                    port=self._ollama.get("port", 11434),
                )

            # core.ollama_client.generate() never raises — every failure
            # (Ollama unreachable, timed out, bad response, ...) is caught
            # there, logged to the "hestia.llm_latency" logger, and returned
            # as "". Without this check that silently became a normal-
            # looking chat reply: {"response": "", "confidence": 0.7} — an
            # empty message asserted with 70% confidence, with nothing to
            # tell the user (or the orchestrator) anything had gone wrong.
            # This is the exact "chat responses ... have no way to
            # distinguish 'model said nothing' from 'Ollama was
            # unreachable'" gap from review item 4. Matches the empty-
            # response check already used by Apollo/Orpheus/Metis/Artemis's
            # own `_llm()`/`_llm_text()` helpers — core chat was the one
            # call site still missing it.
            if not text or not text.strip():
                logger.warning(
                    "CoreModule._chat(): LLM returned an empty response "
                    "for query=%r (see hestia.llm_latency log for cause).",
                    query[:80],
                )
                return {
                    "response": (
                        "I couldn't reach my language model just now — "
                        "mind trying that again in a moment?"
                    ),
                    "data": {},
                    "confidence": 0.0,
                }

            return {"response": text, "data": {}, "confidence": 0.7}
        except Exception:
            logger.exception("CoreModule._chat() failed for query=%r", query[:80])
            return {"response": "I'm not sure about that.", "data": {}, "confidence": 0.3}

    def stream_chat(self, query: str, context_block: str = ""):
        """
        Generator counterpart to _chat(), used by the voice loop's
        streaming path (see HestiaOrchestrator.try_stream_chat and
        Hestia.process_voice_turn in main.py) so TTS can start speaking
        the first sentence of the reply while Ollama is still generating
        the rest, instead of waiting for the whole thing.

        *context_block*, when given, is pre-formatted secondary-module
        context (see HestiaOrchestrator._build_context_block) folded
        straight into the prompt. This lets a chat turn that Hecate
        flagged for synthesis with other modules' context (e.g. "should I
        bring an umbrella" pulling in Ares's weather context) still stream
        as a single LLM generation, instead of streaming a plain chat
        reply and only *then* re-synthesizing it with a second, fully
        blocking LLM call the way dispatch()/_synthesize() does — see
        try_stream_chat's docstring for why that second call is what
        synthesis normally costs streaming. Omitted (the default), the
        prompt is identical to the plain-chat case.

        Not used by process_text()'s normal blocking dispatch() path —
        that one calls _chat() as before, since only the voice loop has
        anywhere to send partial output as it arrives (streaming TTS).
        The web UI, Telegram, and heartbeat callers all just want one
        finished string back, same as always.

        Same never-raise contract as _chat(): any failure mid-stream
        (Ollama unreachable, connection dropped) simply stops yielding
        further text rather than raising — the caller is left with
        whatever was spoken/queued so far, the same way a truncated
        network response would look.
        """
        extra = f"Relevant context:\n{context_block}\n\n" if context_block else ""
        prompt = (
            f"You are Hestia. {extra}"
            f"Answer concisely in 1-2 sentences.\n\nQuestion: {query}"
        )
        try:
            if self._llm_instance is not None and hasattr(self._llm_instance, "generate_stream"):
                yield from self._llm_instance.generate_stream(prompt)
            else:
                yield from generate_stream(
                    prompt,
                    model=self._ollama.get("model", "mistral"),
                    host=self._ollama.get("host", "127.0.0.1"),
                    port=self._ollama.get("port", 11434),
                )
        except Exception:
            logger.exception("CoreModule.stream_chat() failed for query=%r", query[:80])
            return

    def _sys_info(self) -> dict:
        return {
            "response": (
                f"Running on {platform.system()} {platform.release()}, "
                f"Python {platform.python_version()}, "
                f"time {datetime.datetime.now(self._tz).strftime('%I:%M %p')}."
            ),
            "data": {},
            "confidence": 1.0,
        }

    def _save_name(self, entities: dict) -> dict:
        # `entities.get("name", "")` only covers a *missing* key; the NLU
        # can also emit the key with an explicit null (entities={"name": None}),
        # and "".strip() on None raises AttributeError. `or ""` covers both,
        # matching the defensive pattern used throughout modules/hermes.
        name = (entities.get("name") or "").strip().title()
        if not name:
            return {"response": "I didn't catch your name.", "data": {}, "confidence": 0.0}

        self._memory.learn("user_name", name)
        return {"response": f"Got it! I'll remember you as {name}.", "data": {}, "confidence": 0.9}

    def _take_note(self, entities: dict, raw: str) -> dict:
        note = (
            entities.get("content")
            or entities.get("text")
            or entities.get("task")
            or ""
        )

        if not note and raw:
            note = re.sub(
                r'^(take a note|note down|jot down|remember|note)\s*[:\-]?\s*',
                '',
                raw,
                flags=re.IGNORECASE
            ).strip()

        if not note:
            return {"response": "What would you like me to note down?", "data": {}, "confidence": 0.0}

        # Do NOT call self._memory.learn() here.
        # The orchestrator's interaction_logged bus event persists this naturally
        # to interaction_log, which is where _get_notes reads from.
        #
        # The response text below (not just entities["data"]) is what actually
        # ends up in interaction_log.hestia_response, so it has to carry the
        # parsed note content — otherwise _get_notes has nothing but the raw
        # user_text (e.g. "take a note: buy toy", "> take note buy milk") to
        # show, which is why the notes list used to echo back raw commands
        # instead of clean note text.
        return {
            "response": f"Note saved: {note}",
            "data": {"note": note},
            "confidence": 0.95
        }

    # Matches the "Note saved: " prefix _take_note stores in hestia_response.
    _NOTE_PREFIX_RE = re.compile(r'^Note saved:\s*', re.IGNORECASE)

    def _get_notes(self, entities: dict | None = None) -> dict:
        entities = entities or {}
        # NLU sometimes extracts a "topic" (e.g. "query my notes on machine
        # learning" -> {"topic": "machine learning"}). Previously this was
        # captured by the NLU but never read here, so every phrasing of
        # "what notes do I have" returned the same unfiltered top-10 list
        # regardless of what was actually asked for.
        topic = (entities.get("topic") or entities.get("query") or "").strip().lower()

        # Pull a larger pool when filtering so a topic match isn't limited
        # to whatever happens to be in the most-recent 10 notes.
        pool_size = 200 if topic else 10
        rows = self._memory.db.get_by_intent("take_note", pool_size)
        notes = [{"query": r["query"], "response": r["response"], "intent": r["intent"]} for r in rows]

        if not notes:
            return {"response": "No notes saved yet.", "data": {}, "confidence": 0.9}

        def _content(n: dict) -> str:
            resp = n.get("response") or ""
            if self._NOTE_PREFIX_RE.match(resp):
                return self._NOTE_PREFIX_RE.sub('', resp).strip()
            # Older rows saved before this fix only have the raw query;
            # fall back to that rather than showing nothing.
            return n.get("query", "").strip()

        contents = [_content(n) for n in notes]

        if topic:
            matched = [c for c in contents if topic in c.lower()]
            if not matched:
                return {
                    "response": f"I don't have any notes about '{topic}'.",
                    "data": {"notes": [], "topic": topic},
                    "confidence": 0.85,
                }
            # Most-recent-first, capped like the unfiltered path.
            matched = matched[:10]
            body = f"Your notes about '{topic}':\n" + "\n".join(f"- {c}" for c in matched)
            return {"response": body, "data": {"notes": matched, "topic": topic}, "confidence": 0.9}

        body = "Your notes:\n" + "\n".join(f"- {c}" for c in contents[:10])
        return {"response": body, "data": {"notes": contents[:10]}, "confidence": 0.9}

    def _delete_notes(self, entities: dict) -> dict:
        """
        Delete every saved note.

        Gated by the orchestrator's confirmation mechanism (see
        HestiaOrchestrator._resolve_pending): this wipes ALL notes in one
        shot with no undo, so the first call only reports how many notes
        would be deleted and asks for confirmation. Only a confirmed
        second call actually deletes anything.
        """
        try:
            note_count = self._memory.db.get_interaction_stats().get("notes", 0)
        except Exception:
            logger.exception("_delete_notes: failed to count notes.")
            return {"response": "I couldn't check your notes right now.", "data": {}, "confidence": 0.0}

        if not note_count:
            return {"response": "You don't have any notes to delete.", "data": {}, "confidence": 0.9}

        if not entities.get("_confirmed"):
            plural = "note" if note_count == 1 else "notes"
            return {
                "response": f"Delete all {note_count} saved {plural}? This can't be undone. Say yes to confirm.",
                "data": {"note_count": note_count},
                "confidence": 0.9,
                "needs_confirmation": True,
                "confirm_intent": "delete_notes",
                "confirm_entities": {},
                "confirm_label": f"delete your {note_count} saved {plural}",
            }

        try:
            deleted = self._memory.db.delete_by_intent("take_note")
        except Exception:
            logger.exception("_delete_notes: DB delete failed.")
            return {"response": "I couldn't delete your notes right now.", "data": {}, "confidence": 0.0}

        if not deleted:
            return {"response": "You don't have any notes to delete.", "data": {}, "confidence": 0.9}

        return {
            "response": f"Deleted {deleted} note(s).",
            "data": {"deleted": deleted},
            "confidence": 0.95,
        }

    def _get_history(self, entities: dict) -> dict:
        # int() raises on a non-numeric NLU extraction (e.g. "a few"); that
        # would otherwise surface to the user as the orchestrator's generic
        # "something went wrong" instead of just falling back to a sane
        # default, so coerce defensively rather than trusting the entity.
        limit = _safe_int(entities.get("limit"), default=5)

        recent = self._memory.db.get_recent_interactions_excluding(
            limit, ["take_note", "set_reminder"]
        )

        if not recent:
            return {"response": "We haven't talked much yet.", "data": {}, "confidence": 0.9}

        body = "Here's what we talked about:\n" + "\n".join(
            f"- {r['query']}" for r in recent
        )

        return {"response": body, "data": {"history": recent}, "confidence": 0.9}


    # Keys the NLU has been observed to send for date/time questions it
    # misclassifies as get_user_info instead of Chronos's get_time/get_date
    # (e.g. "what is todays date" → {"key": "current_date"}). Answered
    # directly here rather than claiming ignorance, since Hestia obviously
    # knows the date/time regardless of which intent name the NLU picked.
    _DATE_KEYS = frozenset({"current_date", "date", "today", "todays_date"})
    _TIME_KEYS = frozenset({"current_time", "time"})

    # Keys/phrasing the NLU has been observed to send (or fail to send) for
    # "what's my name?" — it often emits get_user_info with entities={}
    # rather than {"key": "user_name"}, even though _save_name() always
    # stores the fact under the fixed key "user_name". Without this alias
    # the fact is unreachable from that phrasing even though it was saved
    # correctly — same class of bug the date/time aliasing above already
    # covers, just for identity instead of date/time.
    _NAME_KEYS = frozenset({"name", "user_name", "my_name", "users_name"})
    _NAME_QUERY_RE = re.compile(r"\bname\b", re.IGNORECASE)

    def _get_user_info(self, entities: dict, raw: str = "") -> dict:
        key = (entities.get("key") or "").strip().lower()

        if key in self._DATE_KEYS:
            return {
                "response": f"Today's date is {datetime.datetime.now(self._tz).strftime('%A, %B %d, %Y')}.",
                "data": {"key": key},
                "confidence": 0.95,
            }
        if key in self._TIME_KEYS:
            return {
                "response": f"It's {datetime.datetime.now(self._tz).strftime('%I:%M %p')}.",
                "data": {"key": key},
                "confidence": 0.95,
            }

        if key in self._NAME_KEYS:
            key = "user_name"
        elif not key and raw and self._NAME_QUERY_RE.search(raw):
            # Last-resort recovery: NLU sent no key at all, but the raw
            # question is plainly asking about the user's name (e.g.
            # "what's my name?" / "do you know my name").
            key = "user_name"

        if key:
            try:
                value = self._memory.db.get_fact(key)
            except Exception:
                logger.exception("_get_user_info: get_fact(%r) failed.", key)
                value = None
            if value:
                return {"response": value, "data": {"key": key}, "confidence": 0.85}
            # A specific key was asked for (or aliased from name/date/time)
            # but nothing is stored under it — don't fall through to the
            # general fact dump below, since that would answer a different
            # question than the one asked.
            return {"response": "I don't have that information yet.", "data": {}, "confidence": 0.3}

        # No specific key could be resolved at all — this is a broad
        # "what do you know about me?" style question. Previously this
        # always fell straight to "I don't have that information yet.",
        # even when facts like user_name were already stored, because
        # nothing here ever queried get_all_facts(). Aggregate whatever
        # Mnemosyne actually has instead of claiming ignorance.
        try:
            facts = self._memory.db.get_all_facts(limit=10)
        except Exception:
            logger.exception("_get_user_info: get_all_facts() failed.")
            facts = []

        if not facts:
            return {"response": "I don't have that information yet.", "data": {}, "confidence": 0.3}

        def _label(k: str) -> str:
            return k.replace("_", " ").strip()

        lines = [f"- {_label(f['key'])}: {f['value']}" for f in facts]
        body = "Here's what I know about you:\n" + "\n".join(lines)
        return {"response": body, "data": {"facts": facts}, "confidence": 0.8}

    def _set_preference(self, entities: dict) -> dict:
        key = entities.get("key", "")
        value = entities.get("value", "")

        if not value:
            return {"response": "What should I remember?", "data": {}, "confidence": 0.0}

        if not key or key == "preference":
            # No deterministic key was extracted by NLU. Rather than fabricate one
            # from the value's leading words (which produced unpredictable, hard
            # to look up keys), ask the user to be specific.
            return {
                "response": "What should I call this preference? (e.g. 'set my location preference to Mumbai')",
                "data": {},
                "confidence": 0.3,
            }

        self._memory.learn(key, value)

        return {"response": "Got it, I'll remember that.", "data": {}, "confidence": 0.95}