#!/usr/bin/env python3
"""
Build tests/golden/eval_set.yaml (Hardening Plan, Phase 6.3): 300-500 labelled
utterances covering clean, typo, Hinglish, ambiguous, conversational and
adversarial (false-write trap) cases. Deterministic (fixed seed) so diffs are
reviewable. Sources: config/nlu_examples.yaml, config/intent_aliases.yaml and
hand-written traps below.

Labels:  intent (exact), optional entities, `must_not_write: true` for traps.
`layer`: deterministic (alias/slots/guards must resolve it) | llm (needs Ollama).
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 1337
WRITE_INTENTS = ["take_note", "learn_fact", "save_name", "remove_habit", "delete_notes",
                 "forget_fact", "add_habit", "pluto_log_expense", "set_reminder"]

TRAPS = [  # (utterance, forbidden write intents) -- must not write
    ("I'm tired", ["save_name"]), ("I am hungry", ["save_name"]),
    ("I'm fine thanks", ["save_name"]), ("i am going home now", ["save_name"]),
    ("I'm Priya's brother", ["save_name"]), ("I am so done with this", ["save_name"]),
    ("What notes do I have?", ["take_note"]), ("what did I note down yesterday", ["take_note"]),
    ("Which notes did I save", ["take_note"]), ("do I have any notes", ["take_note"]),
    ("What's my name?", ["save_name", "learn_fact", "remove_habit"]),
    ("who am i", ["save_name", "learn_fact"]), ("what do you know about me", ["learn_fact", "take_note"]),
    ("what habits do I have", ["add_habit", "remove_habit"]),
    ("how much did I spend on groceries", ["pluto_log_expense"]),
    ("what reminders do I have", ["set_reminder"]),
    ("What's my sister's name?", ["learn_fact"]), ("what is my favourite colour", ["learn_fact"]),
]
HINGLISH = [
    ("mera naam kya hai", "get_user_info"), ("mere notes dikhao", "get_notes"),
    ("mere habits dikhao", "list_habits"),
]
GOLDEN_MORE = [
    ("Take a note: buy new headphones", "take_note", {"content": "buy new headphones"}),
    ("Log an expense: Rs500 for groceries", "pluto_log_expense", {"amount": "500"}),
    ("add a habit: drink water", "add_habit", {"name": "drink water"}),
    ("My name is Akhil", "save_name", {"name": "Akhil"}),
    ("make it 6 instead", "update_last_event", {}),
]


def typo(s: str, rng: random.Random) -> str:
    words = s.split()
    cand = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if not cand:
        return s
    i = rng.choice(cand)
    w = words[i]
    j = rng.randrange(1, len(w) - 1)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]      # transpose two letters
    return " ".join(words)


def main() -> int:
    import yaml
    from core.intent_aliases import IntentAliasResolver
    from modules.hecate.intent_registry import ALL_INTENTS
    rng = random.Random(SEED)
    cases, seen = [], set()

    def add(utt, intent, cat, layer, **extra):
        key = (utt.lower(), intent)
        if key in seen:
            return
        seen.add(key)
        cases.append({"utterance": utt, "intent": intent, "category": cat,
                      "layer": layer, **extra})

    res = IntentAliasResolver(path=ROOT / "config" / "intent_aliases.yaml")
    # clean + typo variants from alias phrases (deterministic layer)
    phrases = sorted(res.phrases())
    rng.shuffle(phrases)
    for ph, it in phrases[:160]:
        add(ph, it, "clean", "deterministic")
        if len(ph) > 8:
            t = typo(ph, rng)
            if t != ph:   # exact-phrase layers rightly miss typos: LLM/classifier cases
                add(t, it, "typo", "llm")
    # LLM-layer cases from the few-shot examples (held for Ollama runs)
    data = yaml.safe_load((ROOT / "config" / "nlu_examples.yaml").read_text(encoding="utf-8"))
    exs = [e for e in data["examples"] if e["intent"] in ALL_INTENTS]
    rng.shuffle(exs)
    for e in exs[:170]:
        add(e["utterance"], e["intent"], "conversational", "llm", entities=e.get("entities") or {})
    for u, it, ents in GOLDEN_MORE:
        add(u, it, "clean", "deterministic", entities=ents)
    for u, it in HINGLISH:
        add(u, it, "hinglish", "llm")
    for u, forbidden in TRAPS:
        cases.append({"utterance": u, "intent": None, "category": "adversarial",
                      "layer": "deterministic", "must_not_write": forbidden})
    out = ROOT / "tests" / "golden" / "eval_set.yaml"
    out.write_text(yaml.safe_dump(cases, allow_unicode=True, sort_keys=False, width=110),
                   encoding="utf-8")
    cats = {}
    for c in cases:
        cats[c["category"]] = cats.get(c["category"], 0) + 1
    print(f"wrote {out}: {len(cases)} cases {cats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
