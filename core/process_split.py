"""
core/process_split.py

Split Hestia into three cooperating processes (backlog #20): a middle ground
between the single-process monolith and a container-per-service design.

    voice  - microphone, wake word, speech-to-text, text-to-speech, barge-in.
             Holds NO models for the assistant itself, so a Whisper/PortAudio
             crash or a slow model load cannot stall answering, and restarting
             it doesn't reload the LLM stack.
    core   - the assistant: NLU, Hecate, every module, web UI, Telegram, sync
             API. Answers ``query`` requests from the voice process.
    jobs   - background work: heartbeat (morning brief, reminders, nightly
             reviews, retraining) and the Chronos scheduler. A slow job can
             never delay a reply.

They talk through ``core/event_queue.py`` (a durable SQLite log):

    voice --rpc.query--------------------------> core     (text in, text out)
    core  --voice.say--------------------------> voice    (everything to speak)
    jobs  --speak (notification)---------------> core      (core applies
                                                            do-not-disturb, then
                                                            forwards to voice.say
                                                            and to Telegram)

``python main.py --role supervisor`` starts and babysits all three;
``--role core|jobs|voice`` runs one on its own; ``--role all`` (the default) is
the original single process and is unchanged.

Known limits, stated plainly
----------------------------
* ``core`` and ``jobs`` both load the module stack and open the same SQLite
  databases (WAL mode copes with that) and the same ChromaDB directory.
  ChromaDB does not officially support two processes writing one persistent
  store; if you see lock/corruption errors, use ``--role all``. This has not
  been soak-tested here.
* Voice-turn streaming (speaking the LLM's reply sentence by sentence) is a
  single-process feature; in split mode the reply is spoken once it is ready.
* Conversation state (pending confirmations, session context) lives in
  ``core`` only, which is the right place for it.
* The supervisor restarts a crashed child with back-off and gives up on one
  that crashes repeatedly; it is not a replacement for systemd.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from core.event_queue import EventQueue, QueueRPC, QueueWorker, DEFAULT_PATH

logger = logging.getLogger(__name__)

ROLES = ("all", "core", "voice", "jobs", "supervisor")
SPEAK_TOPIC = "speak"          # notification: jobs -> core
SAY_TOPIC = "voice.say"        # utterance: core -> voice
QUERY_RPC = "query"


@dataclass(frozen=True)
class RoleProfile:
    """Which subsystems a ``Hestia`` instance builds in a given role."""
    role: str
    stt: bool = True
    tts: bool = True
    wake_word: bool = True
    barge_in: bool = True
    heartbeat: bool = True
    chronos_scheduler: bool = True
    web_ui: bool = True
    telegram: bool = True
    sync_api: bool = True
    serve_queries: bool = False        # answer rpc.query from the voice process
    speak_via_queue: bool = False      # send speech to voice.say instead of a local TTS
    export_topics: tuple = ()          # bus topics sent out to the queue
    import_topics: tuple = ()          # queue topics re-emitted on the local bus


_PROFILES = {
    "all": RoleProfile("all"),
    "core": RoleProfile(
        "core", wake_word=False, barge_in=False, heartbeat=False,
        chronos_scheduler=False, serve_queries=True, speak_via_queue=True,
        import_topics=(SPEAK_TOPIC,)),
    "jobs": RoleProfile(
        "jobs", stt=False, tts=False, wake_word=False, barge_in=False,
        web_ui=False, telegram=False, sync_api=False,
        export_topics=(SPEAK_TOPIC,)),
}


def profile_for(role: str) -> RoleProfile:
    """The build profile for *role* (``voice``/``supervisor`` have none: they
    don't construct a ``Hestia``)."""
    if role not in _PROFILES:
        raise ValueError(f"role {role!r} does not build the assistant (known: {sorted(_PROFILES)})")
    return _PROFILES[role]


def queue_path(config: dict) -> Path:
    return Path(((config.get("processes") or {}).get("queue_path")) or DEFAULT_PATH)


# ---------------------------------------------------------------------------
# core side
# ---------------------------------------------------------------------------

class CoreQueryServer:
    """Answers ``rpc.query`` from the voice process with ``hestia.process_text``."""

    def __init__(self, queue: EventQueue, process_text: Callable[[str], str]) -> None:
        self._rpc = QueueRPC(queue, consumer="core.query")
        self._process_text = process_text
        self._worker: Optional[QueueWorker] = None

    def start(self) -> None:
        def _handler(body: dict) -> dict:
            text = str(body.get("text") or "")
            return {"response": self._process_text(text) if text.strip() else ""}
        self._worker = self._rpc.serve(QUERY_RPC, _handler)
        self._worker.start()

    def stop(self) -> None:
        if self._worker:
            self._worker.stop()


# ---------------------------------------------------------------------------
# voice side
# ---------------------------------------------------------------------------

_EXIT_WORDS = frozenset({"bye", "exit", "stop", "shutdown"})


class VoiceFrontend:
    """The voice process: listen, send text to core, speak what comes back.

    Built from the same components ``HestiaBuilder.build_io`` makes for the
    single-process mode, so wake word, STT, TTS and barge-in behave the same.
    """

    def __init__(self, stt, tts, wake_detector, barge_in, queue: EventQueue,
                 stt_max_duration: float = 10, wake_timeout: float = 30,
                 min_input_len: int = 2, query_timeout: float = 120.0,
                 poll_interval: float = 0.1) -> None:
        self.stt, self.tts, self.wake, self.barge_in = stt, tts, wake_detector, barge_in
        self.queue = queue
        self.rpc = QueueRPC(queue, consumer="voice")
        self.stt_max_duration, self.wake_timeout = stt_max_duration, wake_timeout
        self.min_input_len, self.query_timeout = min_input_len, query_timeout
        self.poll_interval = poll_interval
        self._say_lock = threading.Lock()
        self._stop = threading.Event()
        queue.start_from_now("voice.say")

    # -- speaking -----------------------------------------------------------

    def pump(self) -> int:
        """Speak every utterance core has queued. Safe to call from any thread."""
        spoken = 0
        with self._say_lock:
            for ev in self.queue.consume("voice.say", [SAY_TOPIC]):
                text = str((ev["payload"] or {}).get("text") or "").strip()
                if not text:
                    continue
                self._say(text, (ev["payload"] or {}).get("voice"))
                spoken += 1
        return spoken

    def _say(self, text: str, voice: Optional[str]) -> None:
        def _speak() -> None:
            if voice:
                self.tts.speak(text, voice=voice)
            else:
                self.tts.speak(text)
        try:
            if self.barge_in is not None:
                self.barge_in.reset()
                self.barge_in.start(on_barge_in=self.tts.stop)
                try:
                    _speak()
                    self.tts.wait_until_done()
                finally:
                    self.barge_in.stop()
            else:
                _speak()
                self.tts.wait_until_done()
        except Exception:
            logger.exception("Speaking failed.")

    def _background_pump(self) -> None:
        while not self._stop.is_set():
            try:
                self.pump()
            except Exception:
                logger.exception("Voice pump failed.")
            self._stop.wait(self.poll_interval)

    # -- the loop -------------------------------------------------------------

    def handle_text(self, text: str) -> str:
        """Send one utterance to core and speak the answer. Returns the reply text."""
        try:
            result = self.rpc.call(QUERY_RPC, {"text": text}, timeout=self.query_timeout)
            reply = (result or {}).get("response", "")
        except Exception as exc:
            logger.warning("Core did not answer: %s", exc)
            reply = ""
            with self._say_lock:
                self._say("I can't reach the assistant core right now.", None)
        # core queued the spoken reply before answering; speak it now, in order,
        # before going back to listening (so she doesn't hear herself).
        self.pump()
        return reply

    def run(self, max_failures: int = 3) -> None:
        if self.stt is None or self.wake is None:
            raise RuntimeError("voice role needs working speech-to-text and wake-word detection")
        threading.Thread(target=self._background_pump, daemon=True, name="VoicePump").start()
        failures = 0
        logger.info("Voice process listening for the wake word.")
        try:
            while not self._stop.is_set():
                try:
                    if not self.wake.listen_for_wake_word(timeout=self.wake_timeout):
                        failures = 0
                        continue
                    self.tts.speak("Yes?")
                    self.tts.wait_until_done()
                    text = self.stt.listen_once(max_duration=self.stt_max_duration)
                    failures = 0
                except Exception as exc:
                    failures += 1
                    logger.warning("Voice input error (%d/%d): %s", failures, max_failures, exc)
                    if failures >= max_failures:
                        raise RuntimeError(f"the microphone keeps failing ({exc})") from exc
                    time.sleep(0.5)
                    continue
                if not text or len(text.strip()) < self.min_input_len:
                    self.tts.speak("I didn't catch that.")
                    self.tts.wait_until_done()
                    continue
                if text.lower().strip() in _EXIT_WORDS:
                    self.tts.speak("Goodbye.")
                    self.tts.wait_until_done()
                    break
                self.handle_text(text)
                try:
                    self.wake.flush_audio_queue()
                except Exception:
                    pass
        except KeyboardInterrupt:
            logger.info("Voice process interrupted.")
        finally:
            self._stop.set()

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

@dataclass
class _Child:
    role: str
    proc: Any = None
    starts: list = field(default_factory=list)   # monotonic start times
    state: str = "pending"                       # pending|running|stopped|exited|failed
    last_exit: Optional[int] = None


class Supervisor:
    """Starts one child process per role and restarts a crashed one.

    * A child that exits with code 0 is left alone (a deliberate stop).
    * A non-zero exit is restarted after a back-off (1, 2, 5, 10, 30 s).
    * More than ``max_restarts`` starts inside ``window`` seconds marks the
      child ``failed`` and stops restarting it; the others keep running.
    ``popen`` and ``sleep`` are injectable so this is testable without
    launching processes.
    """

    BACKOFF = (1, 2, 5, 10, 30)

    def __init__(self, roles: Sequence[str] = ("core", "jobs", "voice"),
                 base_command: Optional[Sequence[str]] = None,
                 popen: Callable = subprocess.Popen,
                 max_restarts: int = 5, window: float = 300.0,
                 poll_interval: float = 1.0,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        bad = [r for r in roles if r not in ("core", "jobs", "voice")]
        if bad:
            raise ValueError(f"cannot supervise role(s) {bad}; use core, jobs, voice")
        self.children = {r: _Child(r) for r in roles}
        self.base_command = list(base_command or [sys.executable, str(Path(__file__).resolve().parent.parent / "main.py")])
        self._popen, self._clock, self._sleep = popen, clock, sleep
        self.max_restarts, self.window, self.poll_interval = max_restarts, window, poll_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def command_for(self, role: str) -> list[str]:
        return self.base_command + ["--role", role]

    def _spawn(self, child: _Child) -> None:
        now = self._clock()
        child.starts = [t for t in child.starts if now - t < self.window] + [now]
        child.proc = self._popen(self.command_for(child.role))
        child.state = "running"
        logger.info("Started %s process (pid %s).", child.role, getattr(child.proc, "pid", "?"))

    def start(self) -> None:
        for role in self.children:                 # core first: others depend on it
            self._spawn(self.children[role])
        self._thread = threading.Thread(target=self.monitor, daemon=True, name="Supervisor")
        self._thread.start()

    def check_once(self) -> None:
        """One pass over the children: reap exits, restart or give up."""
        for child in self.children.values():
            if child.state != "running" or child.proc is None:
                continue
            code = child.proc.poll()
            if code is None:
                continue
            child.last_exit = code
            if code == 0:
                child.state = "exited"
                logger.info("%s process exited cleanly.", child.role)
                continue
            if len(child.starts) > self.max_restarts:
                child.state = "failed"
                logger.error("%s process keeps crashing (%d starts in %.0fs); giving up on it.",
                             child.role, len(child.starts), self.window)
                continue
            delay = self.BACKOFF[min(len(child.starts) - 1, len(self.BACKOFF) - 1)]
            logger.warning("%s process exited with %s; restarting in %ss.", child.role, code, delay)
            self._sleep(delay)
            if not self._stop.is_set():
                self._spawn(child)

    def monitor(self) -> None:
        while not self._stop.is_set():
            self.check_once()
            self._stop.wait(self.poll_interval)

    def stop(self, grace: float = 10.0) -> None:
        self._stop.set()
        for child in self.children.values():
            if child.proc is not None and child.proc.poll() is None:
                try:
                    child.proc.terminate()
                except Exception:
                    pass
        deadline = self._clock() + grace
        for child in self.children.values():
            if child.proc is None:
                continue
            while child.proc.poll() is None and self._clock() < deadline:
                self._sleep(0.05)
            if child.proc.poll() is None:
                try:
                    child.proc.kill()
                except Exception:
                    pass
            child.state = "stopped"

    def status(self) -> dict[str, dict]:
        return {r: {"state": c.state, "pid": getattr(c.proc, "pid", None),
                    "starts": len(c.starts), "last_exit": c.last_exit}
                for r, c in self.children.items()}

    def run_forever(self) -> int:
        """Start everything and block until SIGINT/SIGTERM, then stop cleanly."""
        done = threading.Event()

        def _handle(signum, _frame):
            logger.info("Supervisor received %s; stopping children.", signal.Signals(signum).name)
            done.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle)
            except (ValueError, OSError, AttributeError):
                pass
        self.start()
        try:
            while not done.is_set():
                if all(c.state in ("failed", "exited", "stopped") for c in self.children.values()):
                    break
                done.wait(1.0)
        finally:
            self.stop()
        return 0
