"""
modules/dionysus/group.py

Group / friends outing coordination (backlog #148).

Hestia is single-user, so there are no friend accounts. You tell it what your
friends said ("Asha is free sat evening, sun afternoon; likes thai, bowling;
hates loud places; budget 800"), it keeps that per group, and works out the
overlap. Everything here is pure (no database, no network) so it is easy to test.

Availability is read as (day, part-of-day) slots. "weekend", "weekdays",
"evenings" and "anytime" are understood; a clause with a day and no time of day
means all day. Commas and semicolons separate clauses, so "sat, sun evening"
is Saturday (all day) plus Sunday evening. Anything that can't be read yields an
empty set and the person is reported as "unreadable" rather than guessed at.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
PARTS = ("morning", "afternoon", "evening", "night")
ALL_SLOTS = frozenset((d, p) for d in DAYS for p in PARTS)

_DAY_WORDS = {
    "mon": "mon", "monday": "mon", "tue": "tue", "tues": "tue", "tuesday": "tue",
    "wed": "wed", "weds": "wed", "wednesday": "wed",
    "thu": "thu", "thur": "thu", "thurs": "thu", "thursday": "thu",
    "fri": "fri", "friday": "fri", "sat": "sat", "saturday": "sat",
    "sun": "sun", "sunday": "sun",
}
_PART_WORDS = {
    "morning": "morning", "mornings": "morning", "am": "morning",
    "afternoon": "afternoon", "afternoons": "afternoon", "noon": "afternoon", "lunch": "afternoon",
    "evening": "evening", "evenings": "evening", "dinner": "evening",
    "night": "night", "nights": "night", "late": "night",
}
_ANY = {"anytime", "any", "flexible", "free", "open", "whenever"}
MAX_PEOPLE = 20
MAX_TEXT = 300


def parse_availability(text: object) -> frozenset:
    """Slots named in ``text`` as a frozenset of (day, part). Never raises."""
    if not isinstance(text, str):
        return frozenset()
    slots: set = set()
    for clause in re.split(r"[;,/\n]+", text.lower()[:MAX_TEXT]):
        words = re.findall(r"[a-z]+", clause)
        if not words:
            continue
        days: set = set()
        for w in words:
            if w in _DAY_WORDS:
                days.add(_DAY_WORDS[w])
            elif w in ("weekend", "weekends"):
                days |= {"sat", "sun"}
            elif w in ("weekday", "weekdays"):
                days |= set(DAYS[:5])
        parts = {_PART_WORDS[w] for w in words if w in _PART_WORDS}
        if "after" in words and "work" in words:       # "after work" = evening
            parts.add("evening")
        if not days and any(w in _ANY for w in words) and not parts:
            days = set(DAYS)
        if not days and parts:                          # "evenings" alone: every day
            days = set(DAYS)
        for d in days:
            for p in (parts or PARTS):
                slots.add((d, p))
    return frozenset(slots)


def format_slot(slot: tuple) -> str:
    return f"{slot[0].capitalize()} {slot[1]}"


def parse_list(text: object) -> list[str]:
    """Comma / 'and' separated words, lower-cased, de-duplicated, order kept."""
    if not isinstance(text, str):
        return []
    out: list[str] = []
    for item in re.split(r"[,;/\n]+|\band\b", text.lower()[:MAX_TEXT]):
        item = item.strip(" .!-")
        if item and item not in out:
            out.append(item[:40])
    return out[:15]


def overlap(people: dict) -> dict:
    """Work out slots, shared likes and things to avoid.

    ``people``: name -> {"slots": set, "likes": list, "dislikes": list, "budget": float|None}.
    """
    known = {n: p for n, p in people.items() if p.get("slots")}
    unreadable = sorted(n for n, p in people.items() if not p.get("slots"))
    counts: dict = {}
    for n, p in known.items():
        for s in p["slots"]:
            counts.setdefault(s, []).append(n)
    order = lambda s: (DAYS.index(s[0]), PARTS.index(s[1]))
    ranked = sorted(counts, key=lambda s: (-len(counts[s]), order(s)))
    everyone = [s for s in ranked if len(counts[s]) == len(known)] if known else []

    like_counts: dict = {}
    for p in people.values():
        for item in p.get("likes", []):
            like_counts[item] = like_counts.get(item, 0) + 1
    avoid = sorted({d for p in people.values() for d in p.get("dislikes", [])})
    # A like somebody else dislikes is not shared, however many people want it.
    liked = sorted((i for i in like_counts if i not in avoid), key=lambda i: (-like_counts[i], i))
    budgets = [p["budget"] for p in people.values() if p.get("budget")]
    return {
        "known": sorted(known), "unreadable": unreadable,
        "everyone": everyone,
        "partial": [(s, counts[s], sorted(set(known) - set(counts[s]))) for s in ranked
                    if len(counts[s]) < len(known)][:3],
        "likes": [(i, like_counts[i]) for i in liked],
        "vetoed": sorted(i for i in like_counts if i in avoid),
        "avoid": avoid,
        "budget_cap": min(budgets) if budgets else None,
    }


def render_plan(group: str, people: dict, result: dict, currency: str = "\u20b9") -> str:
    n = len(people)
    lines = [f"Group '{group}': {n} {'person' if n == 1 else 'people'} ({', '.join(sorted(people))})."]
    if result["unreadable"]:
        lines.append("I couldn't read availability for " + ", ".join(result["unreadable"])
                     + " so they are left out of the timing. Try e.g. 'sat evening, sun afternoon'.")
    if not result["known"]:
        lines.append("Nobody has availability on file yet.")
    elif result["everyone"]:
        slots = [format_slot(s) for s in result["everyone"][:6]]
        more = len(result["everyone"]) - len(slots)
        lines.append("Everyone is free: " + ", ".join(slots) + (f" (+{more} more)" if more > 0 else "") + ".")
    else:
        lines.append("There is no time when everyone is free. Closest:")
        for slot, who, missing in result["partial"]:
            lines.append(f"  {format_slot(slot)}: {', '.join(who)} (missing {', '.join(missing)})")
    if result["likes"]:
        lines.append("Liked by the group: " + ", ".join(
            f"{i} ({c})" if c > 1 else i for i, c in result["likes"][:6]) + ".")
    if result["vetoed"]:
        lines.append("Dropped because someone dislikes it: " + ", ".join(result["vetoed"]) + ".")
    if result["avoid"]:
        lines.append("Avoid: " + ", ".join(result["avoid"]) + ".")
    if result["budget_cap"]:
        lines.append(f"Keep it within {currency}{result['budget_cap']:,.0f} each (the lowest budget given).")
    if n < 2:
        lines.append("Add your friends with 'set_outing_preferences' to compare.")
    return "\n".join(lines)
