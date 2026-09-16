# core/tts.py

import os
import re
import sys
import threading
import queue
import subprocess

import pyttsx3
import sounddevice as sd


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
    """

    _SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        engine: str = "pyttsx3",
        rate: int = 175,
        volume: float = 1.0,
        piper_model_path: str = None,
    ):
        """
        Initialize TTS engine.

        Args:
            engine: "pyttsx3" or "piper"
            rate: Speech rate in words per minute (pyttsx3)
            volume: Volume level from 0.0 to 1.0
            piper_model_path: Path to Piper TTS model
        """

        self.rate = rate
        self.volume = max(0.0, min(volume, 1.0))

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

        # --------------------------------------------------------------
        # Detect pyttsx3 voice
        # --------------------------------------------------------------

        try:
            temp_engine = pyttsx3.init()
            voices = temp_engine.getProperty("voices")

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

    def speak(self, text: str) -> None:
        """
        Queue text for speech.

        This starts a new speech generation, cancelling anything from
        the previous generation.
        """

        if not text or not text.strip():
            return

        gen = self._bump_generation()

        self._drain_queue()

        self._queue.put((gen, text))

    def speak_stream(self, chunks) -> None:
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

            parts = self._SENTENCE_END_RE.split(buffer)

            # All except the final element are complete sentences.
            for sentence in parts[:-1]:

                sentence = sentence.strip()

                if sentence:
                    self._queue.put((gen, sentence))

            # Keep incomplete sentence.
            buffer = parts[-1]

        # Speak remaining text.
        buffer = buffer.strip()

        if buffer and gen == self._current_generation():
            self._queue.put((gen, buffer))

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

    def synthesize_wav_bytes(self, text: str) -> bytes:
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
                return self._synthesize_piper_wav(text)
            except Exception as e:
                print(
                    f"Piper synth failed: {e}, falling back to pyttsx3",
                    file=sys.stderr,
                )

        return self._synthesize_pyttsx3_wav(text)

    def _synthesize_pyttsx3_wav(self, text: str) -> bytes:
        """Render via a throwaway pyttsx3 engine using save_to_file(),
        which writes audio to disk instead of playing it — unlike
        say()/runAndWait(), used everywhere else in this class."""

        import tempfile

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)

            if self._voice_id:
                engine.setProperty("voice", self._voice_id)

            engine.save_to_file(text, path)
            engine.runAndWait()

            with open(path, "rb") as f:
                return f.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _synthesize_piper_wav(self, text: str) -> bytes:
        """Run Piper once, synchronously, and wrap its headerless raw PCM
        output in a WAV container (Piper's --output-raw is 16-bit mono
        PCM at 22050Hz, same as _speak_piper's playback stream)."""

        import io
        import wave

        proc = subprocess.run(
            ["piper", "--model", self._piper_model_path, "--output-raw"],
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

            gen, text = self._queue.get()

            try:

                # Only speak current generation.
                if gen == self._current_generation():
                    self._speak_blocking(text, gen)

            finally:

                self._queue.task_done()

    # ==================================================================
    # SPEECH DISPATCH
    # ==================================================================

    def _speak_blocking(self, text: str, gen: int) -> None:
        """
        Execute speech using the selected engine.
        """

        if self._engine == "piper":

            try:

                self._speak_piper(text, gen)

                return

            except Exception as e:

                print(
                    f"Piper TTS failed: {e}, "
                    f"falling back to pyttsx3",
                    file=sys.stderr,
                )

        self._speak_pyttsx3(text, gen)

    # ==================================================================
    # PYTTSX3
    # ==================================================================

    def _speak_pyttsx3(self, text: str, gen: int) -> None:
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

            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)

            if self._voice_id:
                engine.setProperty(
                    "voice",
                    self._voice_id,
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

    def _speak_piper(self, text: str, gen: int) -> None:
        """
        Speak using Piper TTS.

        Audio is streamed in small chunks so stop() can interrupt
        playback instead of waiting for the entire utterance.
        """

        if gen != self._current_generation():
            return

        proc = subprocess.Popen(
            [
                "piper",
                "--model",
                self._piper_model_path,
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