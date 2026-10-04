# core/tts.py

import os
import re
import sys
import threading
import queue
import subprocess
from typing import Optional

import pyttsx3
import sounddevice as sd


# ----------------------------------------------------------------------
# Streaming sentence splitter (backlog #172)
# ----------------------------------------------------------------------
#
# speak_stream() has to decide, while text is still arriving, how much of it
# is safe to start speaking. Splitting on ". " alone has three failure modes
# that make streaming TTS sound worse than waiting would:
#   * abbreviations ("Dr. Patel", "e.g. this") get cut mid-name,
#   * bullet lists / line-separated output with no end punctuation sit
#     unspoken until the whole reply is done,
#   * a very long first sentence delays the first word of speech, which is
#     exactly the latency streaming exists to remove.

# The lookbehind makes a match start only at the beginning of a run of
# punctuation. Without it the regex retried from every position inside the run,
# so n punctuation characters with no space after them cost O(n^2) (20,000 of
# "?!" took 6.5 s, on the path every spoken reply goes through). A run that
# fails from its start fails from anywhere inside it, so results are unchanged.
_BOUNDARY_RE = re.compile(r"""(?<![.!?])([.!?]+["')\]]*)(\s+)|(\n+)""")

_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "eg", "ie", "no", "approx", "inc", "ltd", "co", "fig", "cf", "al",
    "gen", "col", "lt", "sgt", "capt", "rev", "hon", "mt", "ft",
})

# If this many characters pile up without a sentence boundary, speak up to the
# last clause break (comma / semicolon / colon) instead of waiting.
CLAUSE_FLUSH_CHARS = 140
_CLAUSE_MIN_CHARS = 40
_CLAUSE_BREAK_RE = re.compile(r"[,;:](?=\s)")


def _is_abbreviation(segment: str) -> bool:
    """True if *segment* (text before a lone period) ends in an abbreviation
    or a single-letter initial, so that period isn't a sentence end."""
    words = segment.split()
    if not words:
        return False
    last = words[-1].lstrip("([\"'").lower()
    if last in _ABBREVIATIONS:
        return True
    return len(last) == 1 and last.isalpha()


def split_speakable(buffer: str) -> tuple[list[str], str]:
    """Split streamed *buffer* into (complete sentences, unfinished remainder).

    Boundaries are sentence-ending punctuation followed by whitespace
    (skipping abbreviations and initials) and line breaks. The remainder is
    the trailing text that isn't safe to speak yet, except that an over-long
    remainder is cut at its last clause break so speech can begin.
    """
    sentences: list[str] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(buffer):
        if m.group(1) is not None:
            if m.group(1) == "." and _is_abbreviation(buffer[start:m.start(1)]):
                continue
        piece = buffer[start:m.end()].strip()
        if piece:
            sentences.append(piece)
        start = m.end()

    rest = buffer[start:]
    while len(rest) > CLAUSE_FLUSH_CHARS:
        cut = None
        for cm in _CLAUSE_BREAK_RE.finditer(rest):
            if cm.start() >= _CLAUSE_MIN_CHARS:
                cut = cm.end()
        if cut is None:
            break
        piece = rest[:cut].strip()
        if piece:
            sentences.append(piece)
        rest = rest[cut:].lstrip()
    return sentences, rest


class HestiaTTS:
    """
    Text-to-speech module with a queue-based non-blocking interface.

    Public methods:
        speak(text):
            Queue a complete utterance.

        speak_stream(chunks):
            Consume streamed LLM text and begin speaking completed
            sentences as soon as they are available.

        stop():
            Immediately cancel queued/current speech where possible.
            Intended for barge-in/interruption handling.

    Both speak() and speak_stream() accept an optional ``voice=`` naming one
    of the configured voice profiles (backlog #170), so health replies, money
    replies and casual chat can sound different. An unknown or missing name
    just uses the base voice.
    """

    def __init__(
        self,
        engine: str = "pyttsx3",
        rate: int = 175,
        volume: float = 1.0,
        piper_model_path: str = None,
        voices: Optional[dict] = None,
        echo_reference=None,
    ):
        """
        Initialize TTS engine.

        Args:
            engine: "pyttsx3" or "piper"
            rate: Speech rate in words per minute (pyttsx3)
            volume: Volume level from 0.0 to 1.0
            piper_model_path: Path to Piper TTS model
            voices: Optional named voice profiles, e.g.
                {"health": {"voice_name": "Hazel", "rate": 165,
                            "piper_model_path": "models/piper/x.onnx"}}.
                Every field is optional; a profile only overrides what it sets.
            echo_reference: Optional core.echo_cancel.EchoReference. When given
                (Piper only), every chunk played is also pushed to it so
                barge-in can cancel Hestia's own voice out of the mic (#175).
        """

        self.rate = rate
        self.volume = max(0.0, min(volume, 1.0))

        # Keep the profile config under its own name: the pyttsx3 voice
        # detection below reuses the local name `voices` for the installed
        # system voices, which would otherwise shadow this argument.
        profiles_cfg = voices

        # Speech queue
        self._queue = queue.Queue()

        # Generation counter used for cancellation / barge-in.
        self._gen_lock = threading.Lock()
        self._generation = 0

        # References to currently active engines/processes.
        self._active_pyttsx3_engine = None
        self._active_piper_proc = None

        # Determine engine.
        self._engine = "pyttsx3"
        self._voice_id = None
        self._piper_model_path = None
        self._pyttsx3_voices: list = []
        self._echo_reference = echo_reference

        # --------------------------------------------------------------
        # Detect pyttsx3 voice
        # --------------------------------------------------------------

        try:
            temp_engine = pyttsx3.init()
            voices = temp_engine.getProperty("voices")
            self._pyttsx3_voices = list(voices or [])

            selected_voice = None

            for voice in voices:
                name = voice.name.lower()

                if "zira" in name:
                    self._voice_id = voice.id
                    selected_voice = voice.name
                    break

                elif "hazel" in name:
                    self._voice_id = voice.id
                    selected_voice = voice.name
                    break

                elif "female" in name:
                    self._voice_id = voice.id
                    selected_voice = voice.name
                    break

            if not self._voice_id and voices:
                self._voice_id = voices[0].id
                selected_voice = voices[0].name

            print(f"Selected pyttsx3 voice: {selected_voice}")

        except Exception as e:
            print(
                f"Could not initialize pyttsx3 voice detection: {e}",
                file=sys.stderr,
            )

        # --------------------------------------------------------------
        # Optional Piper engine
        # --------------------------------------------------------------

        if (
            engine == "piper"
            and piper_model_path
            and os.path.exists(piper_model_path)
        ):
            self._engine = "piper"
            self._piper_model_path = piper_model_path

            print(
                f"TTS engine: piper "
                f"(model: {piper_model_path})"
            )

        else:
            if engine == "piper":
                print(
                    "Piper model not found, falling back to pyttsx3",
                    file=sys.stderr,
                )

            print("TTS engine: pyttsx3")

        # --------------------------------------------------------------
        # Named voice profiles (#170)
        # --------------------------------------------------------------

        self._profiles: dict = self._build_profiles(profiles_cfg)

        # --------------------------------------------------------------
        # Start worker
        # --------------------------------------------------------------

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="Hestia-TTS-Worker",
        )

        self._worker_thread.start()

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    available = True

    @property
    def engine(self) -> str:
        """The active engine: "piper" or "pyttsx3"."""
        return self._engine

    @property
    def voice_profiles(self) -> list:
        """Names of the configured voice profiles."""
        return sorted(self._profiles)

    def has_voice(self, name: Optional[str]) -> bool:
        return bool(name) and name in self._profiles

    def _build_profiles(self, voices: Optional[dict]) -> dict:
        """Resolve the ``voices`` config into ready-to-use profile dicts.

        A profile that names a pyttsx3 voice we can't find, or a Piper model
        file that doesn't exist, keeps the rest of its settings and falls back
        to the base voice for the missing part (with a warning) rather than
        failing startup over a cosmetic feature.
        """
        profiles: dict = {}
        for name, spec in (voices or {}).items():
            if not isinstance(spec, dict):
                continue

            voice_id = None
            wanted = spec.get("voice_name")
            if wanted:
                wanted_l = str(wanted).lower()
                for v in self._pyttsx3_voices:
                    if wanted_l in str(getattr(v, "name", "")).lower():
                        voice_id = v.id
                        break
                if voice_id is None:
                    print(
                        f"TTS voice profile {name!r}: no pyttsx3 voice matching "
                        f"{wanted!r}; using the base voice.",
                        file=sys.stderr,
                    )

            piper_path = None
            model = spec.get("piper_model_path")
            if model:
                if os.path.exists(model):
                    piper_path = model
                else:
                    print(
                        f"TTS voice profile {name!r}: Piper model {model!r} not "
                        f"found; using the base model.",
                        file=sys.stderr,
                    )

            rate = spec.get("rate")
            volume = spec.get("volume")
            profiles[str(name)] = {
                "voice_id": voice_id,
                "piper_model_path": piper_path,
                "rate": int(rate) if isinstance(rate, (int, float)) else None,
                "volume": (
                    max(0.0, min(float(volume), 1.0))
                    if isinstance(volume, (int, float)) else None
                ),
            }
        return profiles

    def _profile(self, voice: Optional[str]) -> dict:
        """Settings for *voice*, or {} (= base voice) if unknown/None."""
        if voice and voice in self._profiles:
            return self._profiles[voice]
        return {}

    def speak(self, text: str, voice: Optional[str] = None) -> None:
        """
        Queue text for speech, optionally in a named voice profile.

        This starts a new speech generation, cancelling anything from
        the previous generation.
        """

        if not text or not text.strip():
            return

        gen = self._bump_generation()

        self._drain_queue()

        self._queue.put((gen, text, voice))

    def speak_stream(self, chunks, voice: Optional[str] = None) -> None:
        """
        Speak streamed text as sentences become available.

        Example:

            tts.speak_stream(llm_stream())

        where llm_stream() yields pieces of text such as:

            "Hello "
            "Akhil. "
            "How "
            "can I help?"

        Hestia begins speaking completed sentences without waiting for
        the entire LLM response.

        This method runs in the calling thread. If the source iterator
        blocks on network I/O, call this from a background thread.
        """

        gen = self._bump_generation()

        self._drain_queue()

        buffer = ""

        for chunk in chunks:

            # Stop pulling tokens if this generation was cancelled.
            if gen != self._current_generation():
                return

            if not chunk:
                continue

            buffer += str(chunk)

            # Complete sentences (abbreviation-aware, line-break-aware, with
            # an early clause flush for very long sentences) go to the
            # speech queue immediately; the unfinished tail stays buffered.
            ready, buffer = split_speakable(buffer)

            for sentence in ready:
                self._queue.put((gen, sentence, voice))

        # Speak remaining text.
        buffer = buffer.strip()

        if buffer and gen == self._current_generation():
            self._queue.put((gen, buffer, voice))

    def stop(self) -> None:
        """
        Immediately stop queued/current speech.

        Intended for barge-in.

        Example:

            User starts speaking
                ↓
            STT detects speech
                ↓
            tts.stop()
                ↓
            Hestia stops talking
        """

        # Invalidate the current generation.
        self._bump_generation()

        # Remove anything waiting in queue.
        self._drain_queue()

        # Stop active pyttsx3 speech.
        engine = self._active_pyttsx3_engine

        if engine is not None:

            try:
                engine.stop()
            except Exception:
                pass

        # Kill active Piper process.
        proc = self._active_piper_proc

        if proc is not None and proc.poll() is None:

            try:
                proc.kill()
            except Exception:
                pass

    def wait_until_done(self) -> None:
        """
        Block until all queued speech has completed or been cancelled.
        """

        self._queue.join()

    def set_rate(self, rate: int) -> None:
        """Set pyttsx3 speech rate."""

        self.rate = rate

    def set_volume(self, volume: float) -> None:
        """Set speech volume from 0.0 to 1.0."""

        self.volume = max(0.0, min(volume, 1.0))

    def synthesize_wav_bytes(self, text: str, voice: Optional[str] = None) -> bytes:
        """
        Render `text` to a standalone WAV byte string and return it,
        instead of speaking it through local speakers.

        For callers that want audio *data* back — notably the web UI's
        /api/tts route, for playback in a browser's <audio> element.
        Bypasses the speak()/queue path entirely: nothing here touches
        self._queue or self._generation, so it can't be cancelled by
        stop()/barge-in and won't interrupt (or be interrupted by) the
        normal local voice loop. Safe to call from a different thread
        than the one running the worker loop.
        """

        if not text or not text.strip():
            return b""

        if self._engine == "piper":
            try:
                return self._synthesize_piper_wav(text, voice)
            except Exception as e:
                print(
                    f"Piper synth failed: {e}, falling back to pyttsx3",
                    file=sys.stderr,
                )

        return self._synthesize_pyttsx3_wav(text, voice)

    def _synthesize_pyttsx3_wav(self, text: str, voice: Optional[str] = None) -> bytes:
        """Render via a throwaway pyttsx3 engine using save_to_file(),
        which writes audio to disk instead of playing it — unlike
        say()/runAndWait(), used everywhere else in this class."""

        import tempfile

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        prof = self._profile(voice)

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", prof.get("rate") or self.rate)
            engine.setProperty(
                "volume",
                prof["volume"] if prof.get("volume") is not None else self.volume,
            )

            voice_id = prof.get("voice_id") or self._voice_id
            if voice_id:
                engine.setProperty("voice", voice_id)

            engine.save_to_file(text, path)
            engine.runAndWait()

            with open(path, "rb") as f:
                return f.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _synthesize_piper_wav(self, text: str, voice: Optional[str] = None) -> bytes:
        """Run Piper once, synchronously, and wrap its headerless raw PCM
        output in a WAV container (Piper's --output-raw is 16-bit mono
        PCM at 22050Hz, same as _speak_piper's playback stream)."""

        import io
        import wave

        model = self._profile(voice).get("piper_model_path") or self._piper_model_path

        proc = subprocess.run(
            ["piper", "--model", model, "--output-raw"],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"Piper exited with code {proc.returncode}")

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(22050)
            wf.writeframes(proc.stdout)

        return buf.getvalue()

    # ==================================================================
    # GENERATION / CANCELLATION
    # ==================================================================

    def _bump_generation(self) -> int:

        with self._gen_lock:
            self._generation += 1

            return self._generation

    def _current_generation(self) -> int:

        with self._gen_lock:
            return self._generation

    def _drain_queue(self) -> None:
        """
        Remove all waiting speech items from the queue.
        """

        while True:

            try:
                self._queue.get_nowait()

            except queue.Empty:
                break

            else:
                self._queue.task_done()

    # ==================================================================
    # WORKER
    # ==================================================================

    def _worker_loop(self) -> None:
        """
        Background worker that processes speech sequentially.
        """

        while True:

            item = self._queue.get()

            try:

                # (gen, text) or (gen, text, voice) — tolerate both shapes.
                gen, text = item[0], item[1]
                voice = item[2] if len(item) > 2 else None

                # Only speak current generation. `voice` is only passed
                # along when one was requested, so the default path keeps
                # the original (text, gen) call shape.
                if gen == self._current_generation():
                    if voice:
                        self._speak_blocking(text, gen, voice)
                    else:
                        self._speak_blocking(text, gen)

            except Exception as e:

                # A failure speaking one utterance must never kill the
                # worker thread: it would leave the queue undrained and
                # wait_until_done() would block forever.
                print(f"TTS worker error: {e}", file=sys.stderr)

            finally:

                self._queue.task_done()

    # ==================================================================
    # SPEECH DISPATCH
    # ==================================================================

    def _speak_blocking(self, text: str, gen: int, voice: Optional[str] = None) -> None:
        """
        Execute speech using the selected engine.
        """

        if self._engine == "piper":

            try:

                if voice:
                    self._speak_piper(text, gen, voice)
                else:
                    self._speak_piper(text, gen)

                return

            except Exception as e:

                print(
                    f"Piper TTS failed: {e}, "
                    f"falling back to pyttsx3",
                    file=sys.stderr,
                )

        if voice:
            self._speak_pyttsx3(text, gen, voice)
        else:
            self._speak_pyttsx3(text, gen)

    # ==================================================================
    # PYTTSX3
    # ==================================================================

    def _speak_pyttsx3(self, text: str, gen: int, voice: Optional[str] = None) -> None:
        """
        Speak using pyttsx3.

        A fresh engine is created for each utterance.
        """

        engine = None

        try:

            # Don't initialize speech if already cancelled.
            if gen != self._current_generation():
                return

            engine = pyttsx3.init()

            prof = self._profile(voice)

            engine.setProperty("rate", prof.get("rate") or self.rate)
            engine.setProperty(
                "volume",
                prof["volume"] if prof.get("volume") is not None else self.volume,
            )

            voice_id = prof.get("voice_id") or self._voice_id

            if voice_id:
                engine.setProperty(
                    "voice",
                    voice_id,
                )

            # Check cancellation again after initialization.
            if gen != self._current_generation():
                try:
                    engine.stop()
                except Exception:
                    pass

                return

            self._active_pyttsx3_engine = engine

            try:

                engine.say(text)

                engine.runAndWait()

            finally:

                self._active_pyttsx3_engine = None

        except Exception as e:

            print(
                f"pyttsx3 TTS error: {e}",
                file=sys.stderr,
            )

        finally:

            if engine is not None:

                try:
                    engine.stop()
                except Exception:
                    pass

    # ==================================================================
    # PIPER
    # ==================================================================

    def _speak_piper(self, text: str, gen: int, voice: Optional[str] = None) -> None:
        """
        Speak using Piper TTS.

        Audio is streamed in small chunks so stop() can interrupt
        playback instead of waiting for the entire utterance.
        """

        if gen != self._current_generation():
            return

        model = self._profile(voice).get("piper_model_path") or self._piper_model_path

        proc = subprocess.Popen(
            [
                "piper",
                "--model",
                model,
                "--output-raw",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        self._active_piper_proc = proc

        stream = None
        cancelled = False

        try:

            # Send text to Piper.
            proc.stdin.write(
                text.encode("utf-8")
            )

            proc.stdin.close()

            # Audio output stream.
            stream = sd.RawOutputStream(
                samplerate=22050,
                channels=1,
                dtype="int16",
                blocksize=2048,
            )

            stream.start()

            chunk_size = 4096

            while True:

                # Check for cancellation before every chunk.
                if gen != self._current_generation():

                    cancelled = True

                    break

                chunk = proc.stdout.read(chunk_size)

                if not chunk:
                    break

                stream.write(chunk)

                # Tell the echo canceller what just went to the speakers (#175).
                if self._echo_reference is not None:
                    try:
                        self._echo_reference.push(chunk, 22050)
                    except Exception:
                        pass

        finally:

            self._active_piper_proc = None

            if stream is not None:

                try:
                    stream.stop()
                except Exception:
                    pass

                try:
                    stream.close()
                except Exception:
                    pass

            # Kill Piper if it is still running.
            if proc.poll() is None:

                try:
                    proc.kill()
                except Exception:
                    pass

            try:
                proc.wait()
            except Exception:
                pass

        if not cancelled and proc.returncode != 0:

            raise RuntimeError(
                f"Piper exited with code {proc.returncode}"
            )


class NullTTS:
    """Silent stand-in used when no speech engine could be started
    (backlog #179), so the rest of Hestia can keep calling ``self.tts``
    without None-checks and a broken audio stack never takes the assistant
    down with it. Replies are still logged by the pipeline, so typed use
    sees everything; nothing is spoken.

    ``available`` is False so callers that care (e.g. the web UI's /api/tts)
    can tell this apart from a working engine.
    """

    available = False
    voice_profiles: list = []

    def speak(self, text: str, voice: Optional[str] = None) -> None:
        return None

    def speak_stream(self, chunks, voice: Optional[str] = None) -> None:
        # Must still drain the iterator: callers (main._speak_streaming) tap
        # it to collect the full response text as a side effect.
        for _ in chunks:
            pass

    def stop(self) -> None:
        return None

    def wait_until_done(self) -> None:
        return None

    def set_rate(self, rate: int) -> None:
        return None

    def set_volume(self, volume: float) -> None:
        return None

    def has_voice(self, name: Optional[str]) -> bool:
        return False

    def synthesize_wav_bytes(self, text: str, voice: Optional[str] = None) -> bytes:
        return b""
