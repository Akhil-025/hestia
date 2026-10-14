"""
core/nlu_retrieval.py

Per-query example retrieval for the NLU prompt (Hardening Plan, Phase 1.2-1.4).

The legacy prompt prepends ~100k chars (392 examples) to every call. Here the
examples live in ``config/nlu_examples.yaml`` and only the most similar few are
placed in the prompt, along with the disambiguation notes of the groups they
came from and a candidate-intent shortlist used to narrow the schema enum.

Pure Python TF-IDF (no numpy / embeddings needed); loads in milliseconds.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_STOP = frozenset("a an the to of for my me i is it and or in on at with please".split())


def tokenize(text: str) -> List[str]:
    toks = [t for t in _TOKEN_RE.findall((text or "").lower())]
    toks = [t for t in toks if t not in _STOP and len(t) > 1]
    # add bigrams: "take note", "my name" carry intent that unigrams lose
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


class ExampleRetriever:
    def __init__(self, path: str | Path | None = "config/nlu_examples.yaml",
                 examples: Optional[List[dict]] = None,
                 notes: Optional[Dict[str, str]] = None,
                 valid_intents: Optional[Iterable[str]] = None):
        if examples is None:
            examples, notes = self._load(path)
        valid = frozenset(valid_intents) if valid_intents is not None else None
        self.examples = [e for e in examples
                         if e.get("intent") and (valid is None or e["intent"] in valid)]
        self.notes = {k: v for k, v in (notes or {}).items()
                      if k and k != "general"}
        self._vecs: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}
        self._build()

    @staticmethod
    def _load(path):
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            examples = list(data.get("examples") or [])
            notes = dict(data.get("notes") or {})
            # Hand-written additions survive re-running build_nlu_examples.py.
            extra = Path(str(path)).with_name("nlu_examples_extra.yaml")
            if extra.exists():
                with open(extra, "r", encoding="utf-8") as fh:
                    more = yaml.safe_load(fh) or {}
                examples += list(more.get("examples") or [])
                notes.update(more.get("notes") or {})
            return examples, notes
        except Exception as exc:
            logger.warning("nlu_retrieval: could not load %s: %s", path, exc)
            return [], {}

    def __len__(self) -> int:
        return len(self.examples)

    def _build(self) -> None:
        docs = [Counter(tokenize(e["utterance"])) for e in self.examples]
        df: Counter = Counter()
        for d in docs:
            df.update(d.keys())
        n = max(len(docs), 1)
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self._vecs = [self._vec(d) for d in docs]

    def _vec(self, counts: Counter) -> Dict[str, float]:
        v = {t: (1 + math.log(c)) * self._idf.get(t, 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def _scores(self, text: str) -> List[tuple]:
        q = self._vec(Counter(tokenize(text)))
        if not q:
            return []
        out = []
        for i, dv in enumerate(self._vecs):
            s = sum(w * dv.get(t, 0.0) for t, w in q.items())
            if s > 0:
                out.append((s, i))
        out.sort(reverse=True)
        return out

    def top(self, text: str, k: int = 8) -> List[dict]:
        return [self.examples[i] for _, i in self._scores(text)[:k]]

    def candidate_intents(self, text: str, k: int = 12,
                          extra: Sequence[str] = ()) -> List[str]:
        """Top-k distinct intents by best example score; ``extra`` (e.g. the
        classifier's guesses) are always included, as is ``chat``."""
        best: Dict[str, float] = defaultdict(float)
        for s, i in self._scores(text)[:60]:
            it = self.examples[i]["intent"]
            best[it] = max(best[it], s)
        ranked = [it for it, _ in sorted(best.items(), key=lambda kv: -kv[1])]
        out: List[str] = []
        for it in list(extra) + ranked:
            if it not in out:
                out.append(it)
            if len(out) >= k:
                break
        if "chat" not in out:
            out.append("chat")
        return out

    def notes_for(self, examples: Sequence[dict], max_chars: int = 1200) -> str:
        seen, parts, used = set(), [], 0
        for e in examples:
            sec = e.get("section")
            if sec in seen or sec not in self.notes:
                continue
            seen.add(sec)
            note = self.notes[sec].strip()
            if used + len(note) > max_chars:
                continue
            parts.append(note)
            used += len(note)
        return "\n".join(parts)

    @staticmethod
    def format_examples(examples: Sequence[dict], with_response: bool) -> str:
        lines = []
        for e in examples:
            obj = {"intent": e["intent"], "entities": e.get("entities") or {}}
            if with_response:
                obj["response"] = ""
            lines.append(f"User: {e['utterance']}\n{json.dumps(obj, ensure_ascii=False)}")
        return "\n\n".join(lines)
