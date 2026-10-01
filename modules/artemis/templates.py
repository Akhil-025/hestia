"""
artemis/templates.py

Goal templates (#130): common goal types pre-filled with sensible milestones
and a default length, so "start a 5k plan" doesn't begin with an empty goal.
Pure data plus lookup; no I/O.
"""
from __future__ import annotations

import re
from typing import Optional

# key -> {"title", "days" (suggested length), "priority", "milestones", "aliases"}
GOAL_TEMPLATES: dict[str, dict] = {
    "run_5k": {
        "title": "Run a 5K", "days": 56, "priority": "medium",
        "aliases": ("5k", "run a 5k", "couch to 5k", "first 5k"),
        "milestones": ["Run/walk 20 minutes three times a week", "Run 2 km without stopping",
                       "Run 3 km without stopping", "Run 5 km without stopping", "Do a timed 5K"],
    },
    "read_book": {
        "title": "Read a book", "days": 30, "priority": "low",
        "aliases": ("read a book", "finish a book", "reading goal"),
        "milestones": ["Pick the book and set a daily page target", "Reach 25% of the book",
                       "Reach 50% of the book", "Reach 75% of the book", "Finish and write a short summary"],
    },
    "learn_language": {
        "title": "Learn a language", "days": 90, "priority": "medium",
        "aliases": ("learn a language", "language learning", "learn spanish", "learn french", "learn german"),
        "milestones": ["Learn the 100 most common words", "Hold a 5-minute conversation",
                       "Finish a beginner course", "Read a short article unaided", "Have a 15-minute conversation"],
    },
    "save_money": {
        "title": "Build an emergency fund", "days": 180, "priority": "high",
        "aliases": ("emergency fund", "save money", "savings goal", "build savings"),
        "milestones": ["Set a target amount and open a separate account", "Automate a monthly transfer",
                       "Reach 25% of the target", "Reach 50% of the target", "Reach 100% of the target"],
    },
    "write_paper": {
        "title": "Write a research paper", "days": 120, "priority": "high",
        "aliases": ("write a paper", "research paper", "write my thesis", "publish a paper", "thesis"),
        "milestones": ["Finalise the research question", "Complete the literature review", "Draft the methods",
                       "Draft results and discussion", "Revise with feedback", "Submit"],
    },
    "lose_weight": {
        "title": "Lose weight", "days": 90, "priority": "medium",
        "aliases": ("lose weight", "weight loss", "get fit", "get in shape"),
        "milestones": ["Log your baseline weight and set a target", "Establish a weekly routine",
                       "Reach 25% of the target", "Reach 50% of the target", "Reach the target"],
    },
    "side_project": {
        "title": "Ship a side project", "days": 60, "priority": "medium",
        "aliases": ("side project", "ship a project", "launch a project", "build an app"),
        "milestones": ["Define the smallest useful version", "Build the core feature",
                       "Test with one real user", "Polish and write docs", "Launch"],
    },
    "declutter": {
        "title": "Declutter the home", "days": 14, "priority": "low",
        "aliases": ("declutter", "clean the house", "organise the house", "organize the house"),
        "milestones": ["Sort one room per day: keep / donate / bin", "Clear the wardrobe",
                       "Clear the kitchen", "Clear the desk and papers", "Drop off donations"],
    },
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def find_template(query: str) -> Optional[str]:
    """Template key whose key, title or alias appears in *query* (longest match wins), else None."""
    q = f" {_norm(query)} "
    best, best_len = None, 0
    for key, t in GOAL_TEMPLATES.items():
        for phrase in (key.replace("_", " "), t["title"], *t["aliases"]):
            p = _norm(phrase)
            if p and f" {p} " in q and len(p) > best_len:
                best, best_len = key, len(p)
    return best


def template_names() -> list[str]:
    return [t["title"] for t in GOAL_TEMPLATES.values()]
