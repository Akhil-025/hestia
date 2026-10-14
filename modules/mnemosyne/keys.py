"""Canonical fact-key helpers (Hardening Plan, Phase 4.3)."""
from __future__ import annotations

import re
from typing import Optional, Tuple

_NAME_ALIASES = {"name", "my_name", "username", "user_name", "full_name"}
_JUNK_PREFIXES = {"what", "who", "hi", "hello", "hey", "my_name", "my"}
_REL_RE = re.compile(r"^(?:user_)?(?:my_)?(sister|brother|mother|father|mom|dad|"
                     r"wife|husband|partner|friend|son|daughter)(?:s)?(?:_name)?$")


def slugify_key(key: str) -> str:
    s = re.sub(r"[^a-z0-9\s_]", "", str(key or "").lower())
    return re.sub(r"[\s_]+", "_", s).strip("_")


def canonical_key(key: str) -> str:
    """user_name for any name alias; <relation>_name for relatives; else a slug."""
    slug = slugify_key(key)
    if slug in _NAME_ALIASES:
        return "user_name"
    m = _REL_RE.match(slug)
    if m:
        rel = {"mom": "mother", "dad": "father"}.get(m.group(1), m.group(1))
        return f"{rel}_name"
    return slug


def is_junk(key: str, value: str) -> bool:
    """Entries like 'what: my name' / 'hi my name: akhil' that are parse debris."""
    k = str(key or "").strip().lower()
    v = str(value or "").strip()
    if not k or not v:
        return True
    if ":" in k or "?" in k or "?" in v:
        return True
    first = slugify_key(k).split("_")[0] if slugify_key(k) else ""
    return first in {"what", "who", "hi", "hello", "hey"}


_SALVAGE_RE = re.compile(
    r"^\s*(?:my|the)\s+(?P<k>[\w' ]{1,40}?)\s+(?:is|are|was)\s+(?P<v>\S.*?)\s*[.!]*\s*$",
    re.I)


_STATE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:(?:make a note|note|remember|remember that|note down|"
    r"keep in mind)(?:\s+that)?|i\s+want\s+you\s+to\s+(?:know|remember)(?:\s+that)?)\s+"
    r"(?:i\s+am|i'm|im)\s+(?:a\s+|an\s+)?(?P<v>[\w][\w \-]{0,40}?)\s*[.!]*\s*$", re.I)


def salvage_fact(raw: str) -> Optional[Tuple[str, str]]:
    """'my sister's name is Priya' -> ('sister_name', 'Priya');
    'make a note that I am vegetarian' -> ('user_is_vegetarian', 'yes')."""
    if "?" in (raw or ""):
        return None
    s = _STATE_RE.match(raw or "")
    if s:
        trait = slugify_key(s.group("v"))
        if trait:
            return (f"user_is_{trait}", "yes")
    m = _SALVAGE_RE.match(raw or "")
    if not m or "?" in raw:
        return None
    k = canonical_key(m.group("k").replace("'s", ""))
    return (k, m.group("v").strip()) if k else None
