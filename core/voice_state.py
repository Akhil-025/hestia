# core/voice_state.py

"""
Shared, thread-safe snapshot of what the voice pipeline is doing right now,
plus the do-not-disturb switch.

Two things live here because both are "state the voice loop owns but other
threads need to read":

* **Listening state (backlog #178).** The voice loop calls ``set_state()``
  each time the microphone's role changes (waiting for the wake word,
  recording an utterance, thinking, speaking). The web UI polls
  ``snapshot()`` via ``/api/voice/state`` and draws an indicator, so a
  headless voice session (no terminal in sight) still shows whether the mic
  is actually open. ``mic_open`` is derived from the state, not guessed by the
  UI: it is True while the wake-word stream or the STT stream is open, and
  while barge-in is armed during speech.

* **Do-not-disturb (backlog #173).** While DND is active, proactive
  notifications (the ``speak`` event-bus event used by reminders, hydration
  nudges, the morning brief...) are *held* instead of spoken, then read out
  together when DND ends. Replies to things you asked for are never held.
  DND can be open-ended or time-boxed ("for 30 minutes"); a timed DND is
  checked lazily (see ``dnd_active``), so no timer thread is needed.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

# --- states ---------------------------------------------------------------
STATE_INACTIVE = "inactive"        # no voice loop running (CLI / web only)
STATE_WAKE = "waiting_for_wake"    # wake-word stream open, mic is hot
STATE_LISTENING = "listening"      # STT stream open, recording an utterance
STATE_THINKING = "thinking"        # utterance captured, mic closed
STATE_SPEAKING = "speaking"        # TTS playing (mic hot only if barge-in armed)
STATE_TYPED = "typed"              # voice unavailable, running on typed input

ALL_STATES = (
    STATE_INACTIVE, STATE_WAKE, STATE_LISTENING,
    STATE_THINKING, STATE_SPEAKING, STATE_TYPED,
)

# States in which the microphone is always open, regardless of anything else.
_MIC_OPEN_STATES = frozenset({STATE_WAKE, STATE_LISTENING})

# Cap on held notifications so a long DND can't grow without bound. When
# exceeded, the oldest are dropped (and counted).
MAX_HELD_NOTIFICATIONS = 50


class VoiceState:
    """Thread-safe voice-pipeline state + DND switch."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._state = STATE_INACTIVE
        self._since = clock()
        self._detail: Optional[str] = None
        self._barge_in_armed = False
        self._dnd = False
        self._dnd_until: Optional[float] = None
        self._held: list[str] = []
        self._dropped = 0

    # -- listening state ----------------------------------------------------

    def set_state(self, state: str, detail: Optional[str] = None) -> None:
        if state not in ALL_STATES:
            raise ValueError(f"unknown voice state {state!r}")
        with self._lock:
            if state != self._state:
                self._since = self._clock()
            self._state = state
            self._detail = detail
            if state != STATE_SPEAKING:
                self._barge_in_armed = False

    def set_barge_in_armed(self, armed: bool) -> None:
        """Mark whether the barge-in listener currently owns an open mic
        stream while Hestia is speaking."""
        with self._lock:
            self._barge_in_armed = bool(armed)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _mic_open_locked(self) -> bool:
        if self._state in _MIC_OPEN_STATES:
            return True
        return self._state == STATE_SPEAKING and self._barge_in_armed

    # -- do-not-disturb -------------------------------------------------------

    def set_dnd(self, on: bool, minutes: Optional[float] = None) -> None:
        """Turn DND on (optionally for *minutes*) or off. Turning it off does
        NOT discard held notifications — call ``release_held()`` to collect
        them."""
        with self._lock:
            self._dnd = bool(on)
            if on and minutes and minutes > 0:
                self._dnd_until = self._clock() + float(minutes) * 60.0
            else:
                self._dnd_until = None

    def dnd_active(self) -> bool:
        """True while DND is on. A timed DND that has run out reads as off
        (its held notifications stay queued for ``release_held()``)."""
        with self._lock:
            return self._dnd_active_locked()

    def _dnd_active_locked(self) -> bool:
        if not self._dnd:
            return False
        if self._dnd_until is not None and self._clock() >= self._dnd_until:
            self._dnd = False
            self._dnd_until = None
            return False
        return True

    def dnd_remaining_minutes(self) -> Optional[float]:
        with self._lock:
            if not self._dnd_active_locked() or self._dnd_until is None:
                return None
            return max(0.0, (self._dnd_until - self._clock()) / 60.0)

    def hold(self, text: str) -> None:
        """Queue a proactive notification instead of speaking it."""
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._held.append(text)
            overflow = len(self._held) - MAX_HELD_NOTIFICATIONS
            if overflow > 0:
                del self._held[:overflow]
                self._dropped += overflow

    def held_count(self) -> int:
        with self._lock:
            return len(self._held)

    def release_held(self) -> list[str]:
        """Return and clear every held notification (oldest first)."""
        with self._lock:
            held, self._held = self._held, []
            self._dropped = 0
            return held

    # -- snapshot for the web UI ------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            dnd = self._dnd_active_locked()
            remaining = None
            if dnd and self._dnd_until is not None:
                remaining = max(0.0, (self._dnd_until - self._clock()) / 60.0)
            return {
                "active": self._state not in (STATE_INACTIVE, STATE_TYPED),
                "state": self._state,
                "detail": self._detail,
                "mic_open": self._mic_open_locked(),
                "since": self._since,
                "dnd": dnd,
                "dnd_remaining_minutes": remaining,
                "held_notifications": len(self._held),
            }
