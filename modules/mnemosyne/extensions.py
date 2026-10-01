"""
modules/mnemosyne/extensions.py

The user-facing half of the remaining Mnemosyne backlog items, kept out of
engine.py (already ~1200 lines) as a mixin that ``MnemosyneEngine`` inherits:

  #31/#32  knowledge graph          graph_connections  (+ /api/mnemosyne/graph)
  #33      spaced repetition        add_study_fact, review_study
  #34/#35  quizzes & weak-spot map  start_quiz, answer_quiz, quiz_performance
  #39      Obsidian vault sync      obsidian_sync
  #43      episodic clustering      recall_episode
  #47      arXiv monitoring         watch_papers

Every component is constructed defensively: if one fails to open, Mnemosyne
still starts and that feature answers with a plain "unavailable" message,
rather than a broken memory module taking Hestia's core down with it.

Multi-turn flows (quiz questions, review cards) reuse the orchestrator's
existing slot-fill mechanism (backlog #29): a handler that wants the user's
next reply returns ``needs_clarification`` + ``missing_slot="answer"``, and
the orchestrator feeds the reply back to the same intent as
``entities["answer"]``. Session state lives on the engine (single user).
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

EXTENSION_INTENTS: frozenset[str] = frozenset({
    "start_quiz", "answer_quiz", "quiz_performance",
    "add_study_fact", "review_study",
    "graph_connections", "recall_episode",
    "obsidian_sync", "watch_papers",
})

_END_WORDS = frozenset({"quit", "exit", "end", "end quiz", "stop quiz", "finish", "done", "end review"})
_MAX_QUIZ_QUESTIONS = 10
_MAX_REVIEW_BATCH = 20
_GRAPH_AUTO_SKIP_PREFIXES = (
    "ares_", "orpheus_", "metis_", "apollo_", "device_location",
)


def _ok(response: str, data: Optional[dict] = None, confidence: float = 0.9) -> dict:
    return {"response": response, "data": data or {}, "confidence": confidence}


def _miss(response: str) -> dict:
    return {"response": response, "data": {}, "confidence": 0.0}


def _ask(question: str, slot_entities: dict) -> dict:
    """Hold the conversation for one more reply (orchestrator slot-fill, #29)."""
    return {
        "response": question,
        "data": {"needs_clarification": True, "missing_slot": "answer",
                 "slot_entities": dict(slot_entities)},
        "confidence": 0.9,
    }


def _readable(key: str) -> str:
    return (key or "").replace("_", " ").strip()


def _text(entities: dict, *names: str) -> str:
    for n in names:
        v = entities.get(n)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# Aliases (config/intent_aliases.yaml) set the intent only and extract no
# entities, so every handler must also be able to read its argument out of
# the raw query ("quiz me on thermodynamics" -> subject "thermodynamics").
_TOPIC_PATTERNS: dict[str, re.Pattern] = {
    "quiz": re.compile(
        r"(?:quiz me|test me|start a quiz|give me (?:\d+ )?questions?)\s+(?:on|about)\s+(?P<t>.+)", re.I),
    "graph": re.compile(
        r"(?:what(?:'s|s| is)?\s+(?:connects?|connected)\s+to|connections?\s+(?:of|for))\s+(?P<t>.+)", re.I),
    "graph_pair": re.compile(
        r"how\s+(?:is|are)\s+(?P<a>.+?)\s+(?:connected|related|linked)\s+to\s+(?P<b>.+)", re.I),
    "papers": re.compile(
        r"(?:watch|follow|stop watching|unwatch)\s+(?:arxiv\s+)?(?:for\s+)?(?P<t>.+)", re.I),
    "study": re.compile(
        r"add\s+(?:this\s+)?to\s+my\s+study(?:\s+material)?\s*[:\-]?\s*(?P<t>.+)", re.I),
    "episode": re.compile(r"(?:about|on|regarding)\s+(?P<t>.+)", re.I),
}
_COUNT_RE = re.compile(r"(?:give me|ask me)?\s*(\d+)\s+questions?", re.I)


def _arg_from_raw(kind: str, raw: str, group: str = "t") -> str:
    m = _TOPIC_PATTERNS[kind].search(raw or "")
    return m.group(group).strip(" .?!,\"'") if m else ""


class MnemosyneExtensionsMixin:
    """Mixed into MnemosyneEngine. Requires: self.db, self.config, self.hestia_llm, self.vector_store."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _init_extensions(self) -> None:
        self._ext_cfg: dict = {}
        self._quiz_session: Optional[dict] = None
        self._study_session: Optional[dict] = None
        self._athena: Any = None
        self._job_last: dict[str, float] = {}
        self.knowledge_graph = self.study_store = self.episode_store = None
        self.quiz_engine = self.obsidian = self.paper_monitor = None

        db_path = self.config.db_path
        for attr, factory in (
            ("knowledge_graph", lambda: _kg(db_path)),
            ("study_store", lambda: _study(db_path)),
            ("episode_store", lambda: _episodes(db_path)),
            ("quiz_engine", lambda: _quiz(self.hestia_llm, db_path)),
        ):
            try:
                setattr(self, attr, factory())
            except Exception:
                logger.exception("Mnemosyne: could not start %s; feature disabled.", attr)

    def configure_extensions(self, cfg: Optional[dict]) -> None:
        """
        Apply the ``mnemosyne:`` block of laptop_config.yaml. Recognised
        keys (all optional; everything risky is off by default)::

            obsidian: {enabled, vault_path, subfolder, writeback}
            papers:   {enabled, max_new_per_query, lookback_days, docs_dir}
            episodes: {use_embeddings}
        """
        self._ext_cfg = dict(cfg or {})
        ob = self._ext_cfg.get("obsidian") or {}
        if ob.get("enabled") and ob.get("vault_path"):
            try:
                from .obsidian import ObsidianVault
                self.obsidian = ObsidianVault(
                    ob["vault_path"], self.config.db_path,
                    subfolder=ob.get("subfolder") or "Hestia",
                )
                if not self.obsidian.exists:
                    logger.warning("Obsidian vault %s does not exist; sync will do nothing.", ob["vault_path"])
            except Exception:
                logger.exception("Mnemosyne: Obsidian sync could not start; disabled.")
                self.obsidian = None
        self._configure_papers()

    def _configure_papers(self) -> None:
        pc = self._ext_cfg.get("papers") or {}
        if not pc.get("enabled"):
            self.paper_monitor = None
            return
        try:
            from .paper_monitor import PaperMonitor
            self.paper_monitor = PaperMonitor(
                self.config.db_path, self._papers_dir(), llm=self.hestia_llm,
                max_new_per_query=int(pc.get("max_new_per_query", 5)),
                lookback_days=int(pc.get("lookback_days", 14)),
            )
        except Exception:
            logger.exception("Mnemosyne: paper monitor could not start; disabled.")
            self.paper_monitor = None

    def _papers_dir(self) -> str:
        explicit = (self._ext_cfg.get("papers") or {}).get("docs_dir")
        if explicit:
            return str(explicit)
        try:   # default: a subfolder of Athena's documents, so its ingest picks them up
            from pathlib import Path
            from modules.athena.config import get_config as athena_config
            return str(Path(athena_config().data_dir) / "arxiv")
        except Exception:
            from pathlib import Path
            return str(Path(self.config.db_path).parent / "papers")

    def attach_athena(self, athena: Any) -> None:
        """Quiz source material and paper ingestion both go through Athena."""
        self._athena = athena

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch_extension(self, intent: str, entities: dict, context: dict) -> Optional[dict]:
        raw = entities.get("raw_query") or context.get("raw_query") or ""
        if intent in ("start_quiz", "answer_quiz"):
            if entities.get("answer") is not None or intent == "answer_quiz":
                return self._quiz_answer(entities, raw)
            return self._quiz_start(entities, raw)
        if intent == "quiz_performance":
            return self._quiz_performance()
        if intent == "add_study_fact":
            return self._study_add(entities, raw)
        if intent == "review_study":
            if entities.get("answer") is not None:
                return self._study_answer(entities)
            return self._study_start()
        if intent == "graph_connections":
            return self._graph_query(entities, raw)
        if intent == "recall_episode":
            return self._episode_recall(entities, raw)
        if intent == "obsidian_sync":
            return self._obsidian_sync_intent()
        if intent == "watch_papers":
            return self._papers_intent(entities, raw)
        return None

    # ==================================================================
    # Quizzes (#34, #35)
    # ==================================================================

    def _quiz_sources(self, subject: str) -> list[str]:
        """Source text for a quiz: Athena documents first, then Obsidian notes, then facts."""
        texts: list[str] = []
        rag = getattr(self._athena, "rag", None)
        if rag is not None and subject:
            try:
                res = rag.search(subject, n_results=6)
                texts += [d for d in getattr(res, "documents", []) if d][:6]
            except Exception:
                logger.exception("Quiz: Athena search failed for %r; continuing.", subject)
        if self.vector_store is not None and subject and len(texts) < 6:
            try:
                for r in self.vector_store.search(
                    subject, n_results=4, where={"type": {"$eq": "note"}}
                ):
                    if r.get("score", 0) >= 0.4 and r.get("text"):
                        texts.append(r["text"])
            except Exception:
                logger.debug("Quiz: note search failed for %r", subject, exc_info=True)
        if len(texts) < 3:
            try:
                keys = self.db.search_fact_keys(subject) if subject else []
                for k in keys[:10]:
                    v = self.db.get_fact(k)
                    if v:
                        texts.append(f"{_readable(k)}: {v}")
            except Exception:
                logger.debug("Quiz: fact lookup failed", exc_info=True)
        return texts

    def _quiz_start(self, entities: dict, raw: str) -> dict:
        if self.quiz_engine is None:
            return _miss("Quizzes aren't available right now.")
        subject = _text(entities, "subject", "topic") or _arg_from_raw("quiz", raw) or "general"
        count = entities.get("count") or entities.get("n")
        if not count:
            m = _COUNT_RE.search(raw or "")
            count = m.group(1) if m else 5
        try:
            n = max(1, min(int(count), _MAX_QUIZ_QUESTIONS))
        except (TypeError, ValueError):
            n = 5

        sources = self._quiz_sources("" if subject == "general" else subject)
        if not sources:
            return _ok(
                f"I couldn't find any notes, documents or facts about {subject} to quiz you on. "
                "Ingest some material first, or tell me facts about it.",
                confidence=0.4,
            )
        questions = self.quiz_engine.generate_quiz(sources, subject, n=n)
        if not questions:
            return _ok("I couldn't generate good questions from that material. Try again?", confidence=0.3)

        self._quiz_session = {
            "subject": subject, "questions": questions, "index": 0, "correct": 0,
        }
        from core.quiz_engine import format_question
        intro = f"Quiz on {subject}: {len(questions)} question(s). Say A, B, C or D — or 'skip', or 'quit'.\n"
        return _ask(intro + format_question(questions[0], 1, len(questions)), {"_quiz": True})

    def _quiz_answer(self, entities: dict, raw: str) -> dict:
        from core.quiz_engine import format_question, is_skip, parse_choice
        session = self._quiz_session
        if session is None or self.quiz_engine is None:
            return _ok("There's no quiz in progress. Say 'quiz me on' a topic to start one.", confidence=0.5)

        reply = _text(entities, "answer", "choice", "query") or raw
        questions, i = session["questions"], session["index"]
        q = questions[i]
        total = len(questions)

        if reply.strip().lower().strip(".!?") in _END_WORDS:
            summary = self._quiz_summary(session, answered=i)
            self._quiz_session = None
            return _ok("Okay, stopping the quiz. " + summary)

        feedback: str
        if is_skip(reply):
            self.quiz_engine.grade_answer(q["id"], -1)         # recorded as incorrect
            feedback = f"Skipped. The answer was {self._answer_label(q)}."
        else:
            idx = parse_choice(reply, q["choices"])
            if idx is None:
                return _ask(
                    "I didn't catch which option you meant — say A, B, C or D.\n"
                    + format_question(q, i + 1, total),
                    {"_quiz": True},
                )
            correct = self.quiz_engine.grade_answer(q["id"], idx)
            if correct:
                session["correct"] += 1
                feedback = "Correct!"
            else:
                feedback = f"Not quite — it's {self._answer_label(q)}."

        session["index"] = i + 1
        if session["index"] < total:
            nxt = questions[session["index"]]
            return _ask(f"{feedback}\n\n" + format_question(nxt, session["index"] + 1, total), {"_quiz": True})

        summary = self._quiz_summary(session, answered=total)
        self._quiz_session = None
        return _ok(f"{feedback}\n\n{summary}", data={"score": session["correct"], "total": total})

    @staticmethod
    def _answer_label(q: dict) -> str:
        ci = q["correct_index"]
        return f"{'ABCD'[ci]}) {q['choices'][ci]}"

    def _quiz_summary(self, session: dict, answered: int) -> str:
        if answered <= 0:
            return "No questions answered."
        pct = round(100 * session["correct"] / answered)
        text = f"You got {session['correct']} of {answered} ({pct}%) on {session['subject']}."
        try:
            weak = self.quiz_engine.get_weakest_subjects(top_n=1)
            if weak and weak[0][1]["accuracy"] < 0.6:
                text += f" Your weakest subject so far is {weak[0][0]}."
        except Exception:
            logger.debug("quiz summary: weakest-subject lookup failed", exc_info=True)
        return text

    def _quiz_performance(self) -> dict:
        if self.quiz_engine is None:
            return _miss("Quiz tracking isn't available right now.")
        try:
            stats = self.quiz_engine.get_strength_weakness_map()
        except Exception:
            logger.exception("quiz_performance failed")
            return _miss("I couldn't read your quiz history.")
        if not stats:
            return _ok("You haven't answered any quiz questions yet. Say 'quiz me on' a topic.", confidence=0.6)

        ranked = sorted(stats.items(), key=lambda kv: kv[1]["accuracy"])
        lines = []
        for subject, s in ranked:
            line = f"{subject}: {s['accuracy']:.0%} ({s['correct']}/{s['attempts']})"
            trend = self.quiz_engine.get_subject_trend(subject)
            if trend and trend["direction"] != "flat":
                line += f", trending {trend['direction']}"
            lines.append(line)
        weakest = self.quiz_engine.get_weakest_subjects(top_n=1)
        head = (
            f"Weakest: {weakest[0][0]}. " if weakest
            else "Not enough attempts per subject yet to call any of them a weakness. "
        )
        return _ok(head + "By subject — " + "; ".join(lines) + ".", data={"subjects": stats})

    # ==================================================================
    # Spaced repetition (#33)
    # ==================================================================

    def _study_add(self, entities: dict, raw: str) -> dict:
        if self.study_store is None:
            return _miss("Study mode isn't available right now.")
        key = _text(entities, "key")
        value = _text(entities, "value")
        pattern = _text(entities, "pattern", "topic", "subject")
        subject = _text(entities, "subject") or None
        if not (key or pattern):
            rest = _arg_from_raw("study", raw)
            # "entropy is a measure of disorder" / "entropy: a measure of disorder"
            m = re.match(r"(?P<k>.+?)\s*(?::|\bis\b|\bare\b|\bmeans\b)\s+(?P<v>.+)", rest, re.I)
            if m:
                key, value = m.group("k").strip(), m.group("v").strip()
            else:
                pattern = rest

        if key and value:
            safe_key = "_".join(key.lower().split())
            self.learn(safe_key, value, source="study")
            keys = [safe_key]
        elif key or pattern:
            needle = key or pattern
            keys = [needle] if self.db.get_fact(needle) else self.find_matching_facts(needle)
        else:
            return _ok("What should I add to your study material? Give me a term and its definition, "
                       "or the name of a fact I already know.", confidence=0.3)

        keys = [k for k in keys if self.db.get_fact(k)]
        if not keys:
            return _ok("I don't have a matching fact to study yet — tell me the fact first, "
                       "e.g. 'study: entropy is a measure of disorder'.", confidence=0.4)

        added = [k for k in keys[:25] if self.study_store.add_card(k, subject)]
        already = len(keys[:25]) - len(added)
        parts = []
        if added:
            parts.append(f"Added {len(added)} to your study material: "
                         + ", ".join(_readable(k) for k in added[:5]) + ("…" if len(added) > 5 else ""))
        if already:
            parts.append(f"{already} already in it")
        return _ok(". ".join(parts) + ".", data={"added": added}, confidence=0.9)

    def _study_start(self) -> dict:
        if self.study_store is None:
            return _miss("Study mode isn't available right now.")
        due = self.study_store.due_cards(limit=_MAX_REVIEW_BATCH)
        if not due:
            stats = self.study_store.stats()
            if not stats["total"]:
                return _ok("You haven't added any study material yet. Say 'add to my study material' "
                           "with a fact.", confidence=0.6)
            upcoming = self.study_store.list_cards(limit=1)
            when = f" Next review: {upcoming[0]['due_date']}." if upcoming else ""
            return _ok(f"Nothing due today — {stats['total']} card(s) on schedule.{when}", confidence=0.8)

        queue = [c["fact_key"] for c in due]
        self._study_session = {"queue": queue, "i": 0, "recalled": 0, "done": 0}
        return self._study_next_prompt(prefix=f"{len(queue)} card(s) due. Say 'skip' if you don't know one.\n")

    def _study_next_prompt(self, prefix: str = "") -> dict:
        s = self._study_session
        while s and s["i"] < len(s["queue"]):
            key = s["queue"][s["i"]]
            if self.db.get_fact(key):
                n, total = s["i"] + 1, len(s["queue"])
                return _ask(f"{prefix}Card {n} of {total}: What is {_readable(key)}?", {"_study": True})
            self.study_store.remove_card(key)      # fact was forgotten; drop the orphan card
            s["i"] += 1
        return self._study_finish(prefix)

    def _study_finish(self, prefix: str = "") -> dict:
        # `prefix` carries the feedback for the card just answered — dropping
        # it would hide the correct answer to the final card from the user.
        s, self._study_session = self._study_session, None
        if not s or not s["done"]:
            return _ok(prefix + "Review finished.", confidence=0.7)
        return _ok(f"{prefix}Review done: you recalled {s['recalled']} of {s['done']} card(s).",
                   data={"recalled": s["recalled"], "done": s["done"]})

    def _study_answer(self, entities: dict) -> dict:
        from .spaced_repetition import grade_recall
        s = self._study_session
        if not s:
            return _ok("No review in progress. Say 'review my study cards' to start.", confidence=0.5)
        reply = _text(entities, "answer") or ""
        if reply.strip().lower().strip(".!?") in _END_WORDS:
            return self._study_finish()

        key = s["queue"][s["i"]]
        expected = self.db.get_fact(key) or ""
        quality = grade_recall(reply, expected)
        card = self.study_store.review(key, quality)
        s["done"] += 1
        if quality >= 3:
            s["recalled"] += 1
        verdict = ("Correct!" if quality >= 4 else "Close enough." if quality == 3
                   else "Not quite.")
        days = card["interval_days"] if card else 1
        feedback = (f"{verdict} The answer: {expected}. "
                    f"See it again in {days} day{'s' if days != 1 else ''}.")
        s["i"] += 1
        return self._study_next_prompt(prefix=feedback + "\n\n")

    def get_study_brief(self, today: Optional[date] = None) -> str:
        """One line for the morning brief, or "" when nothing is due."""
        if self.study_store is None:
            return ""
        try:
            stats = self.study_store.stats(today)
        except Exception:
            logger.exception("get_study_brief failed")
            return ""
        if not stats["due"]:
            return ""
        n = stats["due"]
        return f"You have {n} study card{'s' if n != 1 else ''} due today. Say 'review my study cards' when you're ready."

    # ==================================================================
    # Knowledge graph (#31, #32)
    # ==================================================================

    def _graph_learn_fact(self, key: str, value: str) -> None:
        """Inline extraction when a fact is learned. Never raises."""
        if self.knowledge_graph is None or key.startswith(_GRAPH_AUTO_SKIP_PREFIXES):
            return
        try:
            from .knowledge_graph import extract_from_fact
            ref = f"fact:{key}"
            self.knowledge_graph.remove_source(ref)      # a changed value replaces the old edges
            ents, rels = extract_from_fact(key, value)
            self.knowledge_graph.add_extraction(ref, ents, rels)
        except Exception:
            logger.exception("Knowledge graph update failed for fact %r; continuing.", key)

    def _graph_forget_fact(self, key: str) -> None:
        try:
            if self.knowledge_graph is not None:
                self.knowledge_graph.remove_source(f"fact:{key}")
            if self.study_store is not None:
                self.study_store.remove_card(key)
        except Exception:
            logger.exception("Cleanup after forgetting %r failed; continuing.", key)

    def rebuild_knowledge_graph(self, use_llm: bool = False) -> dict:
        """
        (Re)extract the graph from every fact and conversation summary.
        Incremental by content hash, so running it twice costs nothing the
        second time. ``use_llm`` asks the local model for richer
        subject/relation/object triples for summaries (slow); otherwise the
        rule-based extractor is used.
        """
        kg = self.knowledge_graph
        if kg is None:
            return {"facts": 0, "summaries": 0}
        from .knowledge_graph import (
            content_hash, extract_cooccurrence, extract_from_fact, extract_with_llm,
        )
        done = {"facts": 0, "summaries": 0}
        for f in self.db.get_all_facts(limit=1000):
            key, value = f["key"], f["value"]
            if key.startswith(_GRAPH_AUTO_SKIP_PREFIXES):
                continue
            ref, h = f"fact:{key}", content_hash(f"{key}={value}")
            if kg.is_processed(ref, h):
                continue
            kg.remove_source(ref)
            ents, rels = extract_from_fact(key, value)
            kg.add_extraction(ref, ents, rels)
            kg.mark_processed(ref, h)
            done["facts"] += 1
        for s in self.db.get_recent_summaries(n=200):
            content = s.get("content") or ""
            ref, h = f"summary:{s.get('id')}", content_hash(content)
            if not content or kg.is_processed(ref, h):
                continue
            kg.remove_source(ref)
            ents, rels = (extract_with_llm(self.hestia_llm, content) if use_llm
                          else extract_cooccurrence(content))
            kg.add_extraction(ref, ents, rels)
            kg.mark_processed(ref, h)
            done["summaries"] += 1
        return done

    def _graph_query(self, entities: dict, raw: str) -> dict:
        kg = self.knowledge_graph
        if kg is None:
            return _miss("The knowledge graph isn't available right now.")
        from .knowledge_graph import describe_connections, describe_path

        if any(w in (raw or "").lower() for w in ("rebuild", "refresh the graph", "update the graph")):
            done = self.rebuild_knowledge_graph()
            st = kg.stats()
            return _ok(f"Graph updated from {done['facts']} fact(s) and {done['summaries']} summary(ies): "
                       f"{st['entities']} things, {st['edges']} connections.", data=st)

        subject = _text(entities, "entity", "subject", "topic", "from")
        target = _text(entities, "to", "target", "other")
        if not subject:
            pair = _TOPIC_PATTERNS["graph_pair"].search(raw or "")
            if pair:
                subject, target = pair.group("a").strip(), pair.group("b").strip(" .?!")
            else:
                subject = _arg_from_raw("graph", raw)
        if not subject:
            st = kg.stats()
            if not st["entities"]:
                return _ok("The graph is empty so far. Tell me facts, or ingest notes, and I'll connect them.", confidence=0.5)
            return _ok(f"My graph has {st['entities']} things and {st['edges']} connections. "
                       "Ask 'what connects to' something.", data=st, confidence=0.7)

        a = kg.find_entity(subject)
        if a is None:
            return _ok(f"I don't have anything called {subject} in my knowledge graph.", confidence=0.4)
        if target:
            b = kg.find_entity(target)
            if b is None:
                return _ok(f"I don't have anything called {target} in my knowledge graph.", confidence=0.4)
            path = kg.find_path(a["id"], b["id"])
            return _ok(describe_path(path), data={"path": path or []}, confidence=0.85 if path else 0.5)
        neighbours = kg.neighbors(a["id"])
        return _ok(describe_connections(a, neighbours), data={"entity": a["name"], "connections": neighbours})

    def get_graph_data(self, max_nodes: int = 150) -> dict:
        if self.knowledge_graph is None:
            return {"nodes": [], "links": []}
        return self.knowledge_graph.graph_data(max_nodes=max_nodes)

    # ==================================================================
    # Episodes (#43)
    # ==================================================================

    def cluster_episodes(self) -> dict:
        """Incrementally cluster interactions logged since the last run."""
        if self.episode_store is None:
            return {"processed": 0}
        embed_fn = None
        if (self._ext_cfg.get("episodes") or {}).get("use_embeddings") and self.vector_store is not None:
            embed_fn = getattr(self.vector_store, "_embed", None)
        return self.episode_store.cluster_new(embed_fn=embed_fn)

    def _episode_recall(self, entities: dict, raw: str) -> dict:
        if self.episode_store is None:
            return _miss("Episode recall isn't available right now.")
        try:
            self.cluster_episodes()          # cheap and incremental; keeps recall current
        except Exception:
            logger.exception("recall_episode: clustering failed; using existing episodes.")

        topic = _text(entities, "topic", "subject", "entity") or _arg_from_raw("episode", raw)
        if topic:
            found = self.episode_store.find_episodes(topic, limit=2)
            if not found:
                return _ok(f"I can't find a past stretch of conversation about {topic}.", confidence=0.4)
            parts = []
            for ep in found:
                msgs = self.episode_store.get_members(ep["id"], limit=50)
                asked = [m["query"][:70] for m in msgs if m.get("query")][:3]
                parts.append(
                    f"About {ep['label']} ({ep['size']} messages, {ep['start'][:10]}"
                    + (f" to {ep['end'][:10]}" if ep["end"][:10] != ep["start"][:10] else "")
                    + "): you asked " + "; ".join(f'"{a}"' for a in asked) + "."
                )
            return _ok(" ".join(parts), data={"episodes": found})

        recent = self.episode_store.list_episodes(limit=5, min_size=2)
        if not recent:
            return _ok("I haven't grouped any conversations into episodes yet.", confidence=0.4)
        lines = [f"{e['label']} ({e['size']} messages, {e['end'][:10]})" for e in recent]
        return _ok("Recent topics: " + "; ".join(lines) + ".", data={"episodes": recent})

    # ==================================================================
    # Obsidian (#39)
    # ==================================================================

    def _vector_add_chunks(self, items: list[dict]) -> None:
        if self.vector_store is None:
            return
        for it in items:
            try:
                self.vector_store.add(it["text"], it["metadata"], doc_id=it["id"])
            except Exception:
                logger.exception("Obsidian: vector add failed for %s", it["id"])

    def _vector_delete_chunks(self, ids: list[str]) -> None:
        if self.vector_store is None:
            return
        for i in ids:
            try:
                self.vector_store.delete(i)
            except Exception:
                logger.debug("Obsidian: vector delete failed for %s", i, exc_info=True)

    def sync_obsidian(self) -> Optional[dict]:
        if self.obsidian is None:
            return None
        return self.obsidian.sync(
            add_chunks=self._vector_add_chunks,
            delete_chunks=self._vector_delete_chunks,
            graph=self.knowledge_graph,
        )

    def _obsidian_sync_intent(self) -> dict:
        if self.obsidian is None:
            return _ok("Obsidian sync is off. Set mnemosyne.obsidian.enabled and vault_path in "
                       "laptop_config.yaml to turn it on.", confidence=0.5)
        if not self.obsidian.exists:
            return _ok(f"I can't find your vault at {self.obsidian.root}.", confidence=0.3)
        stats = self.sync_obsidian() or {}
        changed = stats.get("added", 0) + stats.get("updated", 0) + stats.get("removed", 0)
        if not changed and not stats.get("errors"):
            return _ok(f"Your vault is already up to date ({stats.get('unchanged', 0)} notes).", data=stats)
        msg = (f"Synced your vault: {stats.get('added', 0)} new, {stats.get('updated', 0)} updated, "
               f"{stats.get('removed', 0)} removed.")
        if stats.get("errors"):
            msg += f" {stats['errors']} note(s) couldn't be read."
        return _ok(msg, data=stats)

    def write_obsidian_note(self, title: str, body: str, tags=None, links=None) -> Optional[str]:
        """Write a NEW note into the vault's Hestia folder. None if write-back is off."""
        ob = self._ext_cfg.get("obsidian") or {}
        if self.obsidian is None or not ob.get("writeback"):
            return None
        try:
            return str(self.obsidian.write_note(title, body, tags=tags, links=links))
        except Exception:
            logger.exception("Obsidian write-back failed for %r; continuing.", title)
            return None

    def _recall_notes(self, query: str, n: int) -> list[dict]:
        """Vault notes for remember(); empty unless Obsidian sync is on."""
        if self.obsidian is None or self.vector_store is None:
            return []
        try:
            return self.vector_store.search(query, n_results=n, where={"type": {"$eq": "note"}})
        except Exception:
            logger.exception("Obsidian note recall failed for %r; continuing.", query)
            return []

    # ==================================================================
    # Papers (#47)
    # ==================================================================

    def _papers_intent(self, entities: dict, raw: str) -> dict:
        pm = self.paper_monitor
        if pm is None:
            return _ok("Paper monitoring is off. Set mnemosyne.papers.enabled in laptop_config.yaml "
                       "to turn it on.", confidence=0.5)
        low = (raw or "").lower()
        action = _text(entities, "action").lower()
        topic = _text(entities, "topic", "subject")
        if not topic and not action:
            topic = _arg_from_raw("papers", raw)
        if not action:
            if any(w in low for w in ("stop watching", "unwatch", "stop following", "remove")):
                action = "remove"
            elif any(w in low for w in ("check", "any new", "new papers", "look for new", "fetch")):
                action = "check"
            elif any(w in low for w in ("list", "what am i watching", "which topics", "my interests")):
                action = "list"
            else:
                action = "add" if topic else "list"

        if action == "list":
            items = pm.list_interests()
            if not items:
                return _ok("You're not watching any research topics yet. Say 'watch arXiv for' a topic.", confidence=0.6)
            return _ok("Watching: " + "; ".join(i["query"] for i in items) + ".", data={"interests": items})
        if action == "add":
            if not topic:
                return _ok("Which research topic should I watch?", confidence=0.3)
            added = pm.add_interest(topic)
            return _ok(f"Now watching arXiv for {topic}." if added else f"I'm already watching {topic}.")
        if action == "remove":
            if not topic:
                return _ok("Which topic should I stop watching?", confidence=0.3)
            return _ok(f"Stopped watching {topic}." if pm.remove_interest(topic)
                       else f"I wasn't watching {topic}.")

        result = self.check_papers()
        new = result.get("new", [])
        if result.get("errors") and not new:
            return _ok("I couldn't reach arXiv just now. I'll try again later.", confidence=0.3)
        if not new:
            return _ok("No new papers matching your topics.", data={"new": 0})
        titles = "; ".join(p.title for p in new[:5]) + ("…" if len(new) > 5 else "")
        return _ok(f"{len(new)} new paper(s), added to your documents: {titles}.",
                   data={"new": [p.arxiv_id for p in new]})

    def check_papers(self) -> dict:
        """Fetch new papers, then let Athena index them. Never raises."""
        if self.paper_monitor is None:
            return {"new": [], "errors": 0}
        try:
            result = self.paper_monitor.check()
        except Exception:
            logger.exception("Paper check failed.")
            return {"new": [], "errors": 1}
        new = result.get("new", [])
        if new:
            if self._athena is not None and hasattr(self._athena, "_ingest"):
                try:
                    self._athena._ingest()
                except Exception:
                    logger.exception("Athena ingest after paper check failed; files will be picked up next ingest.")
            try:
                from .paper_monitor import digest_markdown
                self.write_obsidian_note(
                    f"New papers {datetime.now():%Y-%m-%d}", digest_markdown(new), tags=["papers", "arxiv"],
                )
            except Exception:
                logger.exception("Paper digest write-back failed; continuing.")
        return result

    # ==================================================================
    # Background jobs (called from the heartbeat tick)
    # ==================================================================

    def run_background_jobs(self, now: Optional[float] = None) -> dict:
        """
        Cadenced upkeep, safe to call on every heartbeat tick: clustering
        runs at most hourly, Obsidian sync every 30 minutes, the paper
        check once per 24h (persisted across restarts via each interest's
        ``last_checked``). Each job is isolated — one failing never blocks
        the others. Returns which jobs ran.
        """
        import time
        now = time.time() if now is None else now
        ran: dict[str, Any] = {}

        def due(name: str, every: float) -> bool:
            if now - self._job_last.get(name, 0.0) >= every:
                self._job_last[name] = now
                return True
            return False

        if self.episode_store is not None and due("episodes", 3600):
            try:
                ran["episodes"] = self.cluster_episodes()
            except Exception:
                logger.exception("Background episode clustering failed.")
        if self.obsidian is not None and due("obsidian", 1800):
            try:
                ran["obsidian"] = self.sync_obsidian()
            except Exception:
                logger.exception("Background Obsidian sync failed.")
        if self.paper_monitor is not None and self._papers_due(now):
            try:
                ran["papers"] = len(self.check_papers().get("new", []))
            except Exception:
                logger.exception("Background paper check failed.")
        return ran

    def _papers_due(self, now_ts: float) -> bool:
        interests = self.paper_monitor.list_interests()
        if not interests:
            return False
        stamps = [i["last_checked"] for i in interests if i.get("last_checked")]
        if len(stamps) < len(interests):
            return True                      # an interest that was never checked
        try:
            oldest = min(datetime.fromisoformat(s) for s in stamps)
        except ValueError:
            return True
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        return now_ts - oldest.timestamp() >= 24 * 3600


# -- lazy factories (kept as functions so the imports stay local) ---------

def _kg(path):
    from .knowledge_graph import KnowledgeGraph
    return KnowledgeGraph(path)


def _study(path):
    from .spaced_repetition import StudyStore
    return StudyStore(path)


def _episodes(path):
    from .episodes import EpisodeStore
    return EpisodeStore(path)


def _quiz(llm, path):
    from core.quiz_engine import QuizEngine
    return QuizEngine(llm, path)
