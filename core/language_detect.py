"""
core/language_detect.py

Script detection to support multi-language input (backlog #27), starting
with Hindi/Hinglish.

What this does and doesn't do
--------------------------------
This does NOT attempt language identification for Romanized Hinglish
("mera sleep log karo") — that's genuine NLP that would need a trained
model or a large lexicon, and guessing wrong would be worse than not
guessing (a misdetected "language" field is actively misleading in a
log). What it DOES reliably and deterministically: classify which
Unicode SCRIPT a query is written in — Devanagari (Hindi/Marathi/etc.
written in their native script), Latin (English, or Hinglish written in
Roman letters, which looks identical to English at the character level),
or a mix of both. That's a real, checkable fact about the bytes, not a
guess.

Where this is used
--------------------
1. `core/observability.py`'s routing log gains a `script` field (Latin-
   only queries dominate for most users, so this is cheap to add and
   free to ignore — but for a bilingual user it's exactly the signal
   `scripts/eval_intents.py` or a future accuracy breakdown would need to
   answer "is classification worse for Devanagari input specifically",
   which prose-only mixed-language logs can't answer after the fact).
2. Devanagari-script queries route straight to `HestiaNLU.understand()`
   the same as any other input — the actual multi-language SUPPORT is in
   the model itself plus the explicit instructions and Hinglish few-shot
   examples added to `config/nlu_prompt.txt`, and the Hinglish phrase
   aliases added to `config/intent_aliases.yaml` (backlog #22's
   mechanism, reused rather than reinvented for this). This module's
   role is purely to make that support OBSERVABLE, not to implement
   translation or classification itself.
"""
from __future__ import annotations

import re

# Devanagari Unicode block (U+0900-U+097F) covers Hindi, Marathi, Sanskrit,
# and several other Indic languages written in that script.
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")


def detect_script(text: str) -> str:
    """
    Classify *text* as "devanagari", "latin", "mixed", or "other".

    - "devanagari": has Devanagari characters, no Latin letters.
    - "latin": has Latin letters, no Devanagari — this is also what
      Hinglish written in Roman script looks like; script detection
      alone cannot and does not try to distinguish "English" from
      "Hinglish" here (see module docstring).
    - "mixed": both scripts present in the same query (a common,
      genuinely useful signal: code-switching mid-sentence, e.g.
      "mujhe 7 baje remind karo about the meeting").
    - "other": neither — numbers only, punctuation only, empty, or a
      script this function doesn't specifically recognise.
    """
    if not text:
        return "other"
    has_devanagari = bool(_DEVANAGARI_RE.search(text))
    has_latin = bool(_LATIN_LETTER_RE.search(text))
    if has_devanagari and has_latin:
        return "mixed"
    if has_devanagari:
        return "devanagari"
    if has_latin:
        return "latin"
    return "other"
