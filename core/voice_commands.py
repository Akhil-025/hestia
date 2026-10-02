# core/voice_commands.py

"""
Local control phrases for the voice pipeline — handled before NLU.

These are commands *about the assistant's own voice behaviour* rather than
requests for any module, so they are matched here with strict, anchored
patterns instead of being routed through the LLM classifier (which would be
slow, and would occasionally decide "repeat that" is a chat message):

* ``repeat``          — "repeat that", "say it again", "what did you say"  (#171)
* ``dnd_on``          — "do not disturb", "mute notifications for 30 minutes" (#173)
* ``dnd_off``         — "resume notifications", "turn off do not disturb"   (#173)
* ``dnd_status``      — "is do not disturb on"                              (#173)
* ``set_sensitivity`` — "I'm in a noisy room", "wake word sensitivity quiet" (#176)

Matching is deliberately conservative: the *whole* utterance must be the
command (after lower-casing and stripping punctuation and a trailing
"please"). "Can you repeat that recipe for pancakes" is NOT a repeat
command; it goes to the normal pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

REPEAT = "repeat"
DND_ON = "dnd_on"
DND_OFF = "dnd_off"
DND_STATUS = "dnd_status"
SET_SENSITIVITY = "set_sensitivity"


@dataclass(frozen=True)
class VoiceCommand:
    name: str
    minutes: Optional[float] = None      # dnd_on: duration, None = until resumed
    level: Optional[str] = None          # set_sensitivity: quiet | normal | noisy
    unparsed: Optional[str] = None       # dnd_on: a duration phrase we couldn't read


_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "forty five": 45, "sixty": 60,
}

_UNIT_MINUTES = {
    "h": 60, "hr": 60, "hrs": 60, "hour": 60, "hours": 60,
    "m": 1, "min": 1, "mins": 1, "minute": 1, "minutes": 1,
}

_DND_NAMES = r"(?:(?:do not disturb|don'?t disturb(?: me)?|dnd)(?: mode)?|(?:quiet|silent) mode)"
_NOTIF = r"(?:all |my |the )?(?:notifications|reminders|alerts)"

_REPEAT_RE = re.compile(
    r"^(?:(?:can|could|would) you )?(?:please )?"
    r"(?:repeat (?:that|it|yourself)|say (?:that|it) again|say again|"
    r"what did you (?:just )?say|come again|pardon|say that one more time|"
    r"repeat that (?:one )?(?:again|once more))$"
)

_DND_ON_RES = (
    re.compile(
        rf"^(?:please )?(?:(?:turn on|enable|enter|switch to|go into|go to|start|activate) )?"
        rf"{_DND_NAMES}(?: (?:for|in) (?P<dur>.+))?$"
    ),
    re.compile(
        rf"^(?:please )?(?:mute|silence|pause|snooze|hold|stop) {_NOTIF}"
        rf"(?: for (?P<dur>.+))?$"
    ),
)

_DND_OFF_RES = (
    re.compile(
        rf"^(?:please )?(?:turn off|disable|end|stop|exit|cancel|leave|deactivate) {_DND_NAMES}$"
    ),
    re.compile(
        rf"^(?:please )?(?:unmute|resume|restore|enable|turn on|unpause|un-pause) {_NOTIF}$"
    ),
    re.compile(rf"^{_NOTIF} (?:on|back on)$"),
    re.compile(r"^i(?:'m| am|m) back$"),
)

_DND_STATUS_RE = re.compile(
    rf"^(?:is|are) (?:{_DND_NAMES}|{_NOTIF}) (?:on|off|muted|paused|active|enabled)$"
)

_SENS_RES = (
    re.compile(
        r"^(?:set |change |switch )?(?:the )?(?:wake ?word )?sensitivity"
        r"(?: level| mode)?(?: to)? (?P<lvl>quiet|normal|noisy)(?: room)?(?: mode)?$"
    ),
    re.compile(
        r"^(?:i(?:'m| am|m) in|it(?:'s| is)|we(?:'re| are) in) (?:a |an )?"
        r"(?P<lvl>quiet|noisy)(?: room| place| environment| space)?$"
    ),
    re.compile(r"^(?P<lvl>quiet|noisy|normal) room(?: mode| sensitivity)?$"),
)


def _normalise(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s'-]", " ", t)          # drop punctuation, keep ' and -
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(?:hey |ok |okay )?hestia,? ", "", t)
    t = re.sub(r" please$", "", t).strip()
    return t


def parse_duration_minutes(phrase: str) -> Optional[float]:
    """Parse "30 minutes", "an hour", "half an hour", "1 hour 30 minutes",
    "two hours" into minutes. Returns None if nothing usable is found."""
    p = _normalise(phrase)
    if not p:
        return None
    if re.fullmatch(r"(?:an? )?half(?: an| a)? hour", p):
        return 30.0
    if re.fullmatch(r"(?:a )?quarter(?: of)?(?: an| a)? hour", p):
        return 15.0

    total = 0.0
    matched = False
    # "<qty> <unit>" pairs, qty being a number or a small number-word.
    word_alt = "|".join(sorted((re.escape(w) for w in _NUMBER_WORDS), key=len, reverse=True))
    pair_re = re.compile(
        rf"(?P<qty>\d+(?:\.\d+)?|{word_alt})\s*"
        rf"(?P<unit>hours?|hrs?|h|minutes?|mins?|m)\b"
    )
    consumed = pair_re.sub(lambda _m: "", p)
    for m in pair_re.finditer(p):
        qty_s = m.group("qty")
        qty = float(qty_s) if qty_s[0].isdigit() else float(_NUMBER_WORDS[qty_s])
        total += qty * _UNIT_MINUTES[m.group("unit")]
        matched = True
    # Anything left over besides "and" means we misread it — refuse rather
    # than silently guessing a shorter/longer duration.
    leftover = re.sub(r"\band\b", "", consumed).strip()
    if not matched or leftover:
        return None
    return total if total > 0 else None


def parse_voice_command(text: str) -> Optional[VoiceCommand]:
    """Return the VoiceCommand *text* is, or None if it's an ordinary query."""
    t = _normalise(text)
    if not t:
        return None

    if _REPEAT_RE.match(t):
        return VoiceCommand(REPEAT)

    if _DND_STATUS_RE.match(t):
        return VoiceCommand(DND_STATUS)

    for rx in _DND_OFF_RES:
        if rx.match(t):
            return VoiceCommand(DND_OFF)

    for rx in _DND_ON_RES:
        m = rx.match(t)
        if m:
            dur = m.groupdict().get("dur")
            if not dur:
                return VoiceCommand(DND_ON)
            minutes = parse_duration_minutes(dur)
            if minutes is None:
                return VoiceCommand(DND_ON, unparsed=dur.strip())
            return VoiceCommand(DND_ON, minutes=minutes)

    for rx in _SENS_RES:
        m = rx.match(t)
        if m:
            return VoiceCommand(SET_SENSITIVITY, level=m.group("lvl"))

    return None
