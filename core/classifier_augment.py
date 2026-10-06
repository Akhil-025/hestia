"""
core/classifier_augment.py

Extra training examples for the intent classifier (backlog #25).

Why: about half of the registered intents have one or two training examples.
A classifier cannot generalise from one sentence, and real queries are
messier than the examples: typos, a polite opener, a Hinglish tail.

What it does: for every intent with fewer than ``target_per_intent`` examples
it writes variants of THAT intent's own examples and nothing else:

  * a typo in one long word (swap, drop or double a letter),
  * a conversational opener ("please", "hey hestia", "can you"),
  * a conversational tail ("please", "yaar", "karo", "now").

It invents no new wording or intents, never touches words containing digits,
and never alters an intent that already has enough examples. Variants are
deterministic for a given ``seed``, carry a lower weight than real examples,
and record their ``parent`` so cross-validation can keep a variant in the same
fold as the example it came from.

Leakage guard: ``forbidden`` (the held-out golden prompts) is checked against
every variant, so augmentation can never put a test prompt into training.
Whether it helps is measured: ``python scripts/classifier_bench.py --augment both``.
"""
from __future__ import annotations

import random
import re
from typing import Iterable, Optional, Sequence

from core.intent_classifier import Example, normalise

AUGMENT_SOURCE = "augment"
AUGMENT_WEIGHT = 0.5
DEFAULT_TARGET = 6

OPENERS = ("please ", "hey hestia ", "hestia ", "can you ", "could you please ", "zara ")
TAILS = (" please", " yaar", " karo", " now", " for me")

_WORD = re.compile(r"[^\W\d_]{5,}", re.UNICODE)      # alphabetic words of 5+ letters


def _key(text: str) -> str:
    return " ".join(normalise(text).split())


def typo(text: str, rng: random.Random) -> Optional[str]:
    """One slip in one long alphabetic word, or None if there is none."""
    words = list(_WORD.finditer(text))
    if not words:
        return None
    m = rng.choice(words)
    w = m.group(0)
    kind = rng.choice(("swap", "drop", "double"))
    i = rng.randrange(1, len(w) - 1)                  # never the first or last letter
    if kind == "swap":
        new = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    elif kind == "drop":
        new = w[:i] + w[i + 1:]
    else:
        new = w[:i] + w[i] + w[i:]
    return text[:m.start()] + new + text[m.end():] if new != w else None


def opener(text: str, rng: random.Random) -> Optional[str]:
    low = text.lower().lstrip()
    if low.startswith(("please", "hey", "hestia", "can you", "could you", "zara")):
        return None                                   # already conversational
    return rng.choice(OPENERS) + text


def tail(text: str, rng: random.Random) -> Optional[str]:
    t = text.rstrip(" .!?")
    if t.lower().endswith(("please", "yaar", "karo", "now", "for me")):
        return None
    return t + rng.choice(TAILS)


_TRANSFORMS = (typo, opener, tail)


def augment_examples(
    examples: Sequence[Example],
    *,
    target_per_intent: int = DEFAULT_TARGET,
    seed: int = 0,
    weight: float = AUGMENT_WEIGHT,
    forbidden: Iterable[str] = (),
) -> list[Example]:
    """Variants for under-represented intents. The input is not modified and is
    NOT included in the result (callers concatenate)."""
    seen = {_key(e.text) for e in examples} | {_key(f) for f in forbidden}
    by_intent: dict[str, list[Example]] = {}
    for e in examples:
        if (e.text or "").strip() and e.intent:
            by_intent.setdefault(e.intent, []).append(e)

    out: list[Example] = []
    for intent in sorted(by_intent):
        base = by_intent[intent]
        need = target_per_intent - len(base)
        if need <= 0:
            continue
        rng = random.Random(f"{seed}:{intent}")       # str seed: stable across runs and platforms
        made = attempts = 0
        while made < need and attempts < need * 12:
            attempts += 1
            src = base[attempts % len(base)]
            variant = _TRANSFORMS[rng.randrange(len(_TRANSFORMS))](src.text, rng)
            if not variant:
                continue
            k = _key(variant)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(Example(text=variant, intent=intent, weight=weight,
                               source=AUGMENT_SOURCE, parent=_key(src.text)))
            made += 1
    return out
