"""
modules/metis/analysis.py

Deterministic, LLM-free text analysis used by Metis.

Everything here is a pure function of its input, so it is fast, testable
without a model, and gives the same answer every time. Metis uses it for:

  - readability and length measurements (#166): word counts, average
    sentence length, Flesch reading ease and Flesch-Kincaid grade, and the
    ``LengthTarget`` that shorten/expand/summarise are checked against;
  - the writing-voice profile (#165): measurable habits such as sentence
    length, contraction use and punctuation style;
  - distinctive-passage extraction for the plagiarism spot-check (#169).

Readability formulas are heuristic (syllables are estimated, not looked up
in a dictionary). They are good for "is this roughly grade 8 or grade 14?",
not for anything finer, and Metis words its output accordingly.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Tokenising
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")
_VOWEL_GROUPS_RE = re.compile(r"[aeiouy]+")


def words(text: str) -> list[str]:
    """Return the word tokens in *text* (apostrophes/hyphens kept inside a word)."""
    return _WORD_RE.findall(text or "")


def count_words(text: str) -> int:
    return len(words(text))


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation or line breaks; drops empties."""
    parts = _SENTENCE_SPLIT_RE.split(text or "")
    return [p.strip() for p in parts if p and p.strip()]


def count_syllables(word: str) -> int:
    """
    Estimate the syllables in one English word.

    Vowel-group counting with the usual silent-e / "-ed" corrections.
    Always returns at least 1 for a non-empty word.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    # Silent endings: "-es"/"-e" after a consonant, and "-ed" unless it follows
    # t/d ("created", "needed" keep their extra syllable).
    w = re.sub(r"(?:[^laeiouytd]ed|[^laeiouy]es|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(_VOWEL_GROUPS_RE.findall(w)))


# ---------------------------------------------------------------------------
# Readability / length metrics (#166)
# ---------------------------------------------------------------------------

def flesch_reading_ease(n_words: int, n_sentences: int, n_syllables: int) -> float:
    if n_words == 0 or n_sentences == 0:
        return 0.0
    return 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)


def flesch_kincaid_grade(n_words: int, n_sentences: int, n_syllables: int) -> float:
    if n_words == 0 or n_sentences == 0:
        return 0.0
    return 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59


def text_metrics(text: str) -> dict[str, Any]:
    """
    Measure *text*: words, sentences, average sentence length, syllables per
    word, Flesch reading ease and Flesch-Kincaid grade. All zeros when empty.
    """
    toks = words(text)
    n_words = len(toks)
    n_sent = max(1, len(split_sentences(text))) if n_words else 0
    n_syll = sum(count_syllables(t) for t in toks)
    if not n_words:
        return {"words": 0, "sentences": 0, "avg_sentence_words": 0.0,
                "syllables_per_word": 0.0, "reading_ease": 0.0, "grade": 0.0}
    return {
        "words": n_words,
        "sentences": n_sent,
        "avg_sentence_words": round(n_words / n_sent, 1),
        "syllables_per_word": round(n_syll / n_words, 2),
        "reading_ease": round(flesch_reading_ease(n_words, n_sent, n_syll), 1),
        "grade": round(max(0.0, flesch_kincaid_grade(n_words, n_sent, n_syll)), 1),
    }


def ease_label(score: float) -> str:
    """Plain-English band for a Flesch reading-ease score."""
    if score >= 80:
        return "very easy"
    if score >= 60:
        return "plain English"
    if score >= 50:
        return "fairly difficult"
    if score >= 30:
        return "difficult"
    return "very difficult"


# ---------------------------------------------------------------------------
# Length / reading-level targets (#166)
# ---------------------------------------------------------------------------

_MAX_TARGET_WORDS = 50_000
_GRADE_MIN, _GRADE_MAX = 1.0, 18.0
_GRADE_TOLERANCE = 2.0
_MIN_WORDS_FOR_GRADE_CHECK = 20

_TARGET_KEYS = ("target_words", "word_count", "words", "num_words")
_MAX_KEYS = ("max_words", "maximum_words", "at_most_words", "under_words")
_MIN_KEYS = ("min_words", "minimum_words", "at_least_words", "over_words")
_GRADE_KEYS = ("target_grade", "grade_level", "grade", "reading_level")


@dataclass(frozen=True)
class LengthTarget:
    """
    A caller-specified length and/or reading-level requirement.

    ``target_words`` means "about N": it is accepted within +/-10% (at least
    +/-2 words, so tiny targets are not impossible). ``min_words`` and
    ``max_words`` are hard bounds. ``grade`` is a Flesch-Kincaid grade,
    accepted within +/-2 grades because the estimate is coarse.
    """

    target_words: Optional[int] = None
    min_words: Optional[int] = None
    max_words: Optional[int] = None
    grade: Optional[float] = None

    # -- construction helpers -------------------------------------------

    @property
    def is_set(self) -> bool:
        return any(v is not None for v in
                   (self.target_words, self.min_words, self.max_words, self.grade))

    def word_bounds(self) -> tuple[Optional[int], Optional[int]]:
        """Effective (low, high) word limits, combining all word settings."""
        lo, hi = self.min_words, self.max_words
        if self.target_words is not None:
            tol = max(2, round(self.target_words * 0.10))
            t_lo, t_hi = max(1, self.target_words - tol), self.target_words + tol
            lo = t_lo if lo is None else max(lo, t_lo)
            hi = t_hi if hi is None else min(hi, t_hi)
        return lo, hi

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in (
            ("target_words", self.target_words), ("min_words", self.min_words),
            ("max_words", self.max_words), ("grade", self.grade)) if v is not None}

    # -- prompting ------------------------------------------------------

    def instructions(self) -> str:
        """Prompt text stating the target; empty string when nothing is set."""
        if not self.is_set:
            return ""
        lines = ["Hard requirements (these override any general length wording above):"]
        if self.target_words is not None:
            lo, hi = self.word_bounds()
            lines.append(
                f"- Length: about {self.target_words} words (between {lo} and {hi})."
            )
        else:
            if self.min_words is not None:
                lines.append(f"- At least {self.min_words} words.")
            if self.max_words is not None:
                lines.append(f"- No more than {self.max_words} words.")
        if self.grade is not None:
            lines.append(
                f"- Reading level: roughly U.S. grade {self.grade:g} "
                f"(Flesch-Kincaid). Adjust sentence length and word choice to match."
            )
        return "\n".join(lines) + "\n"

    # -- checking -------------------------------------------------------

    def violation(self, text: str) -> str:
        """Describe how *text* misses the target, or "" if it meets it."""
        n = count_words(text)
        lo, hi = self.word_bounds()
        problems: list[str] = []
        if lo is not None and n < lo:
            problems.append(f"It was {n} words; it must be at least {lo}.")
        if hi is not None and n > hi:
            problems.append(f"It was {n} words; it must be at most {hi}.")
        if self.grade is not None and n >= _MIN_WORDS_FOR_GRADE_CHECK:
            got = text_metrics(text)["grade"]
            if abs(got - self.grade) > _GRADE_TOLERANCE:
                direction = "too advanced" if got > self.grade else "too simple"
                problems.append(
                    f"Its reading level was about grade {got:g}, which is "
                    f"{direction} for the target of grade {self.grade:g}."
                )
        return " ".join(problems)

    def score(self, text: str) -> float:
        """0.0 when the target is met; larger means further off. For picking the better attempt."""
        n = count_words(text)
        lo, hi = self.word_bounds()
        miss = 0.0
        if lo is not None and n < lo:
            miss += (lo - n) / max(lo, 1)
        if hi is not None and n > hi:
            miss += (n - hi) / max(hi, 1)
        if self.grade is not None and n >= _MIN_WORDS_FOR_GRADE_CHECK:
            gap = abs(text_metrics(text)["grade"] - self.grade) - _GRADE_TOLERANCE
            if gap > 0:
                miss += gap / _GRADE_MAX
        return miss


def _parse_number(value: Any) -> Optional[float]:
    """Parse 150, "150", "150 words", "grade 8" -> float; None if no number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    m = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    return float(m.group()) if m else None


def _first_present(entities: dict, keys: tuple[str, ...]) -> tuple[bool, Any]:
    for key in keys:
        if key in entities and entities[key] not in (None, ""):
            return True, entities[key]
    return False, None


def parse_length_target(entities: dict) -> tuple[Optional[LengthTarget], str]:
    """
    Read word-count / reading-level targets from NLU entities.

    Returns ``(target, "")`` — with ``target`` None when no target keys are
    present — or ``(None, message)`` when a supplied value can't be used, so
    the caller can ask the user instead of silently ignoring it.
    """
    target_words = min_words = max_words = None
    grade = None

    for keys, name in ((_TARGET_KEYS, "target"), (_MAX_KEYS, "max"), (_MIN_KEYS, "min")):
        present, raw = _first_present(entities, keys)
        if not present:
            continue
        num = _parse_number(raw)
        if num is None or num < 1 or num > _MAX_TARGET_WORDS or int(num) != num:
            return None, (
                "I couldn't read that word count — give me a whole number, "
                "like 'under 100 words'."
            )
        if name == "target":
            target_words = int(num)
        elif name == "max":
            max_words = int(num)
        else:
            min_words = int(num)

    present, raw = _first_present(entities, _GRADE_KEYS)
    if present:
        num = _parse_number(raw)
        if num is None or not (_GRADE_MIN <= num <= _GRADE_MAX):
            return None, (
                "I couldn't read that reading level — give me a U.S. grade "
                f"between {_GRADE_MIN:g} and {_GRADE_MAX:g}, like 'grade 8'."
            )
        grade = float(num)

    if min_words is not None and max_words is not None and min_words > max_words:
        return None, "The minimum word count is higher than the maximum — which did you mean?"

    target = LengthTarget(target_words, min_words, max_words, grade)
    if not target.is_set:
        return None, ""
    lo, hi = target.word_bounds()
    if lo is not None and hi is not None and lo > hi:
        return None, "Those word-count limits contradict each other — could you restate them?"
    return target, ""


# ---------------------------------------------------------------------------
# Writing-voice statistics (#165)
# ---------------------------------------------------------------------------

_CONTRACTION_RE = re.compile(
    r"\b\w+(?:n['\u2019]t|['\u2019](?:s|re|ve|ll|d|m))\b", re.IGNORECASE
)
_FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"})
_SECOND_PERSON = frozenset({"you", "your", "yours", "yourself"})
_PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?\w+(?:ed|en)\b", re.IGNORECASE
)
_EM_DASH_RE = re.compile(r"\u2014|\u2013|--|\s-\s")


def compute_style_stats(texts: list[str]) -> dict[str, Any]:
    """
    Measure the writing habits across one or more samples of the user's text.

    Rates are per 100 words (or per sentence, where noted) so samples of
    different lengths combine sensibly. Returns ``{}`` if there is no text.
    """
    combined = "\n\n".join(t for t in texts if t and t.strip())
    toks = words(combined)
    n_words = len(toks)
    if not n_words:
        return {}

    sentences = split_sentences(combined)
    n_sent = max(1, len(sentences))
    sent_lengths = [count_words(s) for s in sentences] or [n_words]
    mean_len = sum(sent_lengths) / len(sent_lengths)
    variance = sum((x - mean_len) ** 2 for x in sent_lengths) / len(sent_lengths)

    lowered = [t.lower() for t in toks]
    per100 = 100.0 / n_words

    return {
        "sample_count": len([t for t in texts if t and t.strip()]),
        "total_words": n_words,
        "avg_sentence_words": round(mean_len, 1),
        "sentence_length_spread": round(math.sqrt(variance), 1),
        "avg_word_length": round(sum(len(t) for t in toks) / n_words, 2),
        "long_word_ratio": round(sum(1 for t in toks if len(t) >= 7) / n_words, 3),
        "contractions_per_100": round(len(_CONTRACTION_RE.findall(combined)) * per100, 2),
        "exclamation_ratio": round(combined.count("!") / n_sent, 3),
        "question_ratio": round(combined.count("?") / n_sent, 3),
        "commas_per_sentence": round(combined.count(",") / n_sent, 2),
        "semicolons_per_100": round(combined.count(";") * per100, 2),
        "dashes_per_100": round(len(_EM_DASH_RE.findall(combined)) * per100, 2),
        "first_person_per_100": round(sum(1 for t in lowered if t in _FIRST_PERSON) * per100, 2),
        "second_person_per_100": round(sum(1 for t in lowered if t in _SECOND_PERSON) * per100, 2),
        "passive_per_100_sentences": round(len(_PASSIVE_RE.findall(combined)) * 100.0 / n_sent, 1),
    }


def describe_style_stats(stats: dict[str, Any]) -> list[str]:
    """Turn measured habits into short, prompt-ready statements about the voice."""
    if not stats:
        return []
    out: list[str] = []
    n_sent_estimate = stats.get("total_words", 0) / max(stats.get("avg_sentence_words", 1) or 1, 1)

    avg = stats.get("avg_sentence_words", 0)
    spread = stats.get("sentence_length_spread", 0)
    if avg < 10:
        out.append(f"short, punchy sentences (about {avg:g} words on average)")
    elif avg <= 18:
        out.append(f"medium-length sentences (about {avg:g} words on average)")
    else:
        out.append(f"long, flowing sentences (about {avg:g} words on average)")
    if avg and spread / avg >= 0.6:
        out.append("mixes very short and very long sentences")
    elif avg and spread / avg <= 0.3:
        out.append("keeps sentence length steady")

    contractions = stats.get("contractions_per_100", 0)
    if contractions >= 1.5:
        out.append("uses contractions freely (don't, it's)")
    elif contractions < 0.3:
        out.append("rarely uses contractions (writes 'do not', 'it is')")

    if stats.get("exclamation_ratio", 0) >= 0.1:
        out.append("uses exclamation marks")
    elif n_sent_estimate >= 5 and stats.get("exclamation_ratio", 0) == 0:
        out.append("avoids exclamation marks")
    if stats.get("question_ratio", 0) >= 0.1:
        out.append("often asks questions")
    if stats.get("dashes_per_100", 0) >= 0.5:
        out.append("likes dashes for asides")
    if stats.get("semicolons_per_100", 0) >= 0.3:
        out.append("uses semicolons")
    if stats.get("commas_per_sentence", 0) >= 2.0:
        out.append("uses plenty of commas and subordinate clauses")

    if stats.get("first_person_per_100", 0) >= 4:
        out.append("writes in the first person")
    if stats.get("second_person_per_100", 0) >= 3:
        out.append("speaks directly to the reader ('you')")

    long_ratio = stats.get("long_word_ratio", 0)
    if long_ratio >= 0.25:
        out.append("leans toward longer, more formal vocabulary")
    elif long_ratio <= 0.12:
        out.append("prefers plain, everyday words")
    if stats.get("passive_per_100_sentences", 0) >= 15:
        out.append("is comfortable with passive voice")
    return out


# ---------------------------------------------------------------------------
# Distinctive passages for the plagiarism spot-check (#169)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers him his how i if
in into is it its just me more most my no nor not now of off on once only or
other our out over own same she should so some such than that the their them
then there these they this those through to too under until up very was we
were what when where which while who whom why will with would you your
""".split())


def normalise_for_match(text: str) -> str:
    """Lowercase and collapse everything but letters/digits, for phrase lookup in page text."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _window_score(tokens: list[str]) -> float:
    score = 0.0
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        score += len(tok)
        if tok.isdigit():
            score += 3
        elif i > 0 and tok[0].isupper():
            score += 2          # mid-sentence capital: likely a name or coined term
    return score / max(len(tokens), 1)


def distinctive_phrases(text: str, k: int = 4, min_words: int = 8,
                        max_words: int = 12) -> list[str]:
    """
    Pick up to *k* short passages from *text* that are most likely to be
    unique on the web — long, uncommon words and proper nouns — spread
    across different sentences and returned in reading order.

    A search engine can only confirm or rule out a passage that is specific
    enough, and "the fact that the results were mixed" is a useless signal
    for common phrasing, so this deliberately avoids stopword-heavy runs.
    """
    sentences = [s for s in split_sentences(text) if count_words(s) >= min_words]
    if not sentences:
        toks = words(text)
        sentences = [" ".join(toks)] if len(toks) >= min_words else []

    candidates: list[tuple[float, int, str]] = []
    for idx, sentence in enumerate(sentences):
        toks = words(sentence)
        size = min(max_words, len(toks))
        best_score, best_window = -1.0, None
        for start in range(0, len(toks) - size + 1):
            window = toks[start:start + size]
            sc = _window_score(window)
            if sc > best_score:
                best_score, best_window = sc, window
        # A window with no content words at all (score 0) is a useless
        # web query, so it is never a candidate.
        if best_window and best_score > 0:
            candidates.append((best_score, idx, " ".join(best_window)))

    candidates.sort(key=lambda c: (-c[0], c[1]))
    chosen = sorted(candidates[:max(0, k)], key=lambda c: c[1])
    return [phrase for _, _, phrase in chosen]