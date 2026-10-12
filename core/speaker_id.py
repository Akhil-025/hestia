"""Speaker identification for the voice pipeline (backlog #177).

Purpose: let privacy-sensitive modules (Pluto, Mnemosyne, Apollo, Hermes, Iris,
Artemis) behave differently when someone other than you is talking to Hestia
in the room. It is a *convenience gate, not authentication*.

How it works
------------
* Each utterance is reduced to a small voiceprint: the mean and standard
  deviation of MFCC features over its voiced frames (numpy only, no new
  dependency, no model download).
* ``--enroll-voice NAME`` records a few utterances and stores their centroid
  and spread in ``data/voiceprints.json``. Only the numbers are stored, never
  audio, and ``--forget-voice NAME`` deletes them. A voiceprint is biometric
  data: the file is written owner-read-only where the OS allows and is under
  the git-ignored ``data/`` folder.
* A new utterance is accepted as that person when its distance to the centroid
  is within ``margin`` times the spread seen at enrolment.

Honest limits
-------------
MFCC statistics are far weaker than a neural speaker model. Expect a housemate
with a similar voice to sometimes pass, a cold or a different microphone
distance to sometimes make *you* fail, and a recording of you to pass. Tested
here on synthetic voices only, never on real people. Because of that every
doubt resolves to the restrictive side: an utterance that is too short, too
quiet, unreadable or simply not recognised is a **guest**, and a guest gets the
``speaker_id.guest`` policy (by default only Dionysus, Metis and Orpheus; no
memory, no free chat, no "repeat that"). Recognition never *unlocks* anything
beyond what you would have by typing.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
OWNER = "owner"
GUEST = "guest"
DEFAULT_PATH = "data/voiceprints.json"

_N_MFCC = 20
_N_MELS = 26
_FRAME = 400          # 25 ms at 16 kHz
_HOP = 160            # 10 ms
_NFFT = 512
MIN_VOICED_FRAMES = 60  # 0.6 s of actual speech

DEFAULT_GUEST_ALLOW = ("dionysus", "metis", "orpheus")


# ── features ─────────────────────────────────────────────────────────────────

def _mel_filterbank(sr: int) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(80.0), hz_to_mel(sr / 2.0), _N_MELS + 2)
    bins = np.floor((_NFFT + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((_N_MELS, _NFFT // 2 + 1))
    for i in range(1, _N_MELS + 1):
        lo, mid, hi = bins[i - 1], bins[i], bins[i + 1]
        for k in range(lo, mid):
            fb[i - 1, k] = (k - lo) / max(1, mid - lo)
        for k in range(mid, hi):
            fb[i - 1, k] = (hi - k) / max(1, hi - mid)
    return fb


_FB_CACHE: dict = {}


def _dct_matrix() -> np.ndarray:
    n = np.arange(_N_MELS)
    k = np.arange(_N_MFCC)[:, None]
    return np.cos(np.pi * k * (2 * n + 1) / (2 * _N_MELS)) * math.sqrt(2.0 / _N_MELS)


def voiced_frame_count(audio: np.ndarray) -> int:
    return int(_voiced_mask(_frames(np.asarray(audio, dtype=np.float64))).sum())


def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < _FRAME:
        return np.empty((0, _FRAME))
    n = 1 + (len(x) - _FRAME) // _HOP
    idx = np.arange(_FRAME)[None, :] + _HOP * np.arange(n)[:, None]
    return x[idx]


def _voiced_mask(frames: np.ndarray) -> np.ndarray:
    if len(frames) == 0:
        return np.zeros(0, dtype=bool)
    energy = np.log10((frames ** 2).mean(axis=1) + 1e-10)
    floor = np.percentile(energy, 10)
    peak = np.percentile(energy, 95)
    if peak - floor < 1.0:          # no real contrast between speech and silence
        return np.zeros(len(frames), dtype=bool)
    return energy > floor + 0.5 * (peak - floor)


def voiceprint(audio, sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """Feature vector (mean and std of MFCCs 1..19 over voiced frames), or None
    if there is too little speech to say anything."""
    x = np.asarray(audio, dtype=np.float64).ravel()
    if sr != SAMPLE_RATE or len(x) == 0 or not np.isfinite(x).all():
        return None
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])             # pre-emphasis
    frames = _frames(x)
    mask = _voiced_mask(frames)
    if mask.sum() < MIN_VOICED_FRAMES:
        return None
    frames = frames[mask] * np.hamming(_FRAME)
    power = np.abs(np.fft.rfft(frames, n=_NFFT)) ** 2 / _NFFT
    fb = _FB_CACHE.setdefault(sr, _mel_filterbank(sr))
    logmel = np.log(power @ fb.T + 1e-10)
    mfcc = (logmel @ _dct_matrix().T)[:, 1:]               # drop c0 (loudness)
    return np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])


# ── profiles ─────────────────────────────────────────────────────────────────

@dataclass
class Profile:
    name: str
    role: str
    centroid: np.ndarray
    spread: float
    samples: int
    created: str = ""

    def distance(self, vec: np.ndarray) -> float:
        return float(np.linalg.norm(vec - self.centroid))


def build_profile(name: str, vectors: List[np.ndarray], role: str = OWNER) -> Profile:
    if len(vectors) < 3:
        raise ValueError("Enrolment needs at least 3 usable utterances.")
    mat = np.vstack(vectors)
    centroid = mat.mean(axis=0)
    dists = np.linalg.norm(mat - centroid, axis=1)
    # With few samples the spread is optimistic; inflate it by sqrt(n/(n-1)).
    spread = float(dists.mean() * math.sqrt(len(vectors) / (len(vectors) - 1)))
    return Profile(name, role, centroid, spread, len(vectors),
                   datetime.now(timezone.utc).isoformat(timespec="seconds"))


class ProfileStore:
    """Voiceprint numbers on disk. Thread-safe; a damaged file reads as empty."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> List[Profile]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            out = []
            for name, p in (data.get("profiles") or {}).items():
                out.append(Profile(name, str(p.get("role", OWNER)),
                                   np.asarray(p["centroid"], dtype=np.float64),
                                   float(p["spread"]), int(p.get("samples", 0)),
                                   str(p.get("created", ""))))
            return out
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return []

    def save(self, profiles: Iterable[Profile]) -> None:
        payload = {"version": 1, "profiles": {
            p.name: {"role": p.role, "centroid": [round(float(v), 6) for v in p.centroid],
                     "spread": round(p.spread, 6), "samples": p.samples, "created": p.created}
            for p in profiles}}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, self.path)

    def add(self, profile: Profile) -> None:
        keep = [p for p in self.load() if p.name.lower() != profile.name.lower()]
        self.save(keep + [profile])

    def remove(self, name: str) -> bool:
        profiles = self.load()
        keep = [p for p in profiles if p.name.lower() != name.lower()]
        if len(keep) == len(profiles):
            return False
        self.save(keep)
        return True


# ── identification and policy ────────────────────────────────────────────────

@dataclass(frozen=True)
class Identity:
    name: Optional[str]
    role: str                 # OWNER or GUEST (a profile's own role counts as owner-level only if "owner")
    reason: str
    distance: Optional[float] = None

    @property
    def is_owner(self) -> bool:
        return self.role == OWNER


@dataclass(frozen=True)
class GuestPolicy:
    """What an unrecognised speaker may ask for. Fails closed."""

    allow_modules: frozenset = frozenset(DEFAULT_GUEST_ALLOW)
    deny_modules: frozenset = frozenset()
    deny_intents: frozenset = frozenset()
    allow_chat: bool = False
    allow_repeat: bool = False

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "GuestPolicy":
        cfg = cfg or {}

        def s(key):
            v = cfg.get(key)
            if isinstance(v, str):
                v = [v]
            return frozenset(str(x).strip().lower() for x in (v or []) if str(x).strip())

        allow = s("allow_modules") if "allow_modules" in cfg else frozenset(DEFAULT_GUEST_ALLOW)
        return cls(allow_modules=allow, deny_modules=s("deny_modules"),
                   deny_intents=s("deny_intents"),
                   allow_chat=bool(cfg.get("allow_chat", False)),
                   allow_repeat=bool(cfg.get("allow_repeat", False)))

    def allows(self, intent: str, module: Optional[str]) -> bool:
        intent_l = (intent or "").lower()
        if intent_l in self.deny_intents:
            return False
        if intent_l == "chat":
            return self.allow_chat
        if module is None:
            return False                       # unregistered intent: refuse
        module = module.lower()
        return module not in self.deny_modules and module in self.allow_modules


class SpeakerID:
    def __init__(self, store: ProfileStore, *, margin: float = 2.5,
                 min_floor: float = 0.5, guest: Optional[GuestPolicy] = None,
                 module_for_intent=None):
        self.store = store
        self.margin = float(margin)
        self.min_floor = float(min_floor)
        self.guest_policy = guest or GuestPolicy()
        self._module_for_intent = module_for_intent or _registry_module_for_intent

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> Optional["SpeakerID"]:
        cfg = cfg or {}
        if not cfg.get("enabled", False):
            return None
        return cls(ProfileStore(cfg.get("profiles_path", DEFAULT_PATH)),
                   margin=float(cfg.get("margin", 2.5)),
                   guest=GuestPolicy.from_config(cfg.get("guest")))

    def enrolled(self) -> List[Profile]:
        return self.store.load()

    def identify(self, audio, sr: int = SAMPLE_RATE) -> Identity:
        profiles = self.store.load()
        if not profiles:
            return Identity(None, GUEST, "no voice enrolled")
        try:
            vec = voiceprint(audio, sr)
        except Exception:
            logger.exception("Voiceprint extraction failed.")
            return Identity(None, GUEST, "could not analyse the audio")
        if vec is None:
            return Identity(None, GUEST, "too little speech to tell")
        best, best_d = None, float("inf")
        for p in profiles:
            if vec.shape != p.centroid.shape:
                continue
            d = p.distance(vec)
            if d < best_d:
                best, best_d = p, d
        if best is None:
            return Identity(None, GUEST, "enrolled voiceprint is from another version; re-enrol")
        limit = max(self.margin * best.spread, self.min_floor)
        if best_d <= limit:
            role = OWNER if best.role == OWNER else GUEST
            return Identity(best.name, role, "matched", best_d)
        return Identity(None, GUEST, "voice not recognised", best_d)

    def refusal_for(self, identity: Identity, intent: str) -> Optional[str]:
        """A sentence to say instead of acting, or None if this speaker may proceed."""
        if identity.is_owner:
            return None
        try:
            module = self._module_for_intent(intent)
        except Exception:
            logger.exception("Could not consult the intent registry; refusing.")
            module = None
        if self.guest_policy.allows(intent, module):
            return None
        logger.info("Speaker policy refused intent %r (%s).", intent, identity.reason)
        return ("I don't recognise your voice, so I can't help with that one. "
                "Ask me by typing, or let the owner ask.")

    def may_repeat(self, identity: Identity) -> bool:
        return identity.is_owner or self.guest_policy.allow_repeat


def _registry_module_for_intent(intent: str) -> Optional[str]:
    from modules.hecate.intent_registry import module_for_intent
    return module_for_intent(intent)


# ── enrolment (CLI) ──────────────────────────────────────────────────────────

ENROL_PROMPTS = (
    "Say: what is on my calendar for tomorrow",
    "Say: remind me to call my sister this evening",
    "Say: how much did I spend on food this month",
    "Say: tell me something interesting about space",
    "Say: set a timer for twenty five minutes",
)


def record_fixed(seconds: float = 4.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    import sounddevice as sd
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return rec[:, 0]


def enroll_interactive(store: ProfileStore, name: str, role: str = OWNER,
                       record=record_fixed, ask=input, say=print) -> Optional[Profile]:
    """Record the prompts, keep the usable ones, save a profile. ``record``,
    ``ask`` and ``say`` are injectable so tests need no microphone."""
    say(f"Enrolling '{name}'. Use your normal speaking voice, in the room and at the "
        f"distance you usually talk to Hestia. {len(ENROL_PROMPTS)} short phrases.")
    vectors: List[np.ndarray] = []
    for prompt in ENROL_PROMPTS:
        for attempt in range(2):
            ask(f"\n{prompt}\nPress Enter, then speak... ")
            vec = voiceprint(record())
            if vec is not None:
                vectors.append(vec)
                say("  got it")
                break
            say("  I didn't hear enough speech, try again (a bit louder, closer to the mic).")
    if len(vectors) < 3:
        say("Not enough usable recordings; nothing was saved.")
        return None
    profile = build_profile(name, vectors, role)
    store.add(profile)
    say(f"Saved a voiceprint for '{name}' from {len(vectors)} phrases (numbers only, no audio kept). "
        f"Turn it on with `speaker_id.enabled: true` in the config.")
    return profile
