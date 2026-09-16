# tests/test_main.py
"""
Regression tests for the voice-loop glue code in main.py: Hestia.process_voice_turn,
Hestia._speak_streaming, Hestia._with_barge_in, and Hestia.run_voice_loop.

These four are the newest, most complex orchestration code added for streaming
+ barge-in support (threading, generation counters, barge-in state) and had zero
test coverage before this file — everything else exercised so far sits one level
down (core/tts.py, core/barge_in.py, core/stt.py, modules/hestia's
CoreModule.stream_chat / HestiaOrchestrator.try_stream_chat).

Hestia.__init__ boots the entire app (starts Ollama, hits the network, spins up
a heartbeat thread, etc.) and is intentionally not unit-testable — per its own
docstring, HestiaBuilder is the seam meant for isolated construction, while
Hestia itself only does wiring. So these tests bypass __init__ via
object.__new__() and hand-wire just the attributes each method under test
actually touches, mirroring the "explicit dependencies" style the codebase
already uses (see FakeMemory/FakeDB in test_hestia.py).

modules.pluto pulls in langgraph/langchain/xgboost, which aren't part of this
sandbox's installed set; it is only ever referenced at import time (as a type
for wiring, never touched by the code under test here), so it's faked in
sys.modules before `import main`, the same way conftest.py fakes vosk /
webrtcvad / pyttsx3 / telegram for hardware- and network-bound imports.

Run with:  pytest tests/test_main.py -v
"""
import sys
import os
import types
from unittest.mock import MagicMock, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# The `modules.pluto` stub that used to live here was removed: it was a
# plain module object with no __path__, so after this file imported it,
# every `modules.pluto.<submodule>` import in the same pytest session
# failed with "'modules.pluto' is not a package" and tests/test_pluto.py
# errored at collection depending on file ordering. tests/conftest.py now
# fakes psycopg2 (the dependency that was actually missing) instead, so
# modules.pluto imports for real here and stays a package for everyone.

import main


# ---------------------------------------------------------------------------
# Fixture: a Hestia instance with every collaborator mocked
# ---------------------------------------------------------------------------

def make_hestia(barge_in_enabled: bool = True) -> main.Hestia:
    """
    Build a Hestia instance without running __init__, with mocked
    collaborators standing in for every subsystem process_voice_turn /
    _speak_streaming / _with_barge_in / run_voice_loop touch.
    """
    h = object.__new__(main.Hestia)

    h.mnemosyne = MagicMock()
    h.mnemosyne.get_recent.return_value = []

    h.nlu = MagicMock()
    h.nlu.understand.return_value = {"intent": "chat", "entities": {}, "response": ""}

    h.orchestrator = MagicMock()
    h.orchestrator.try_stream_chat.return_value = None
    h.orchestrator.dispatch.return_value = "a plain response"

    h.tts = MagicMock()
    h.wake_detector = MagicMock()
    h.stt = MagicMock()

    h.barge_in = MagicMock()
    h.barge_in.consume_triggered.return_value = False
    h.barge_in.consume_captured_audio.return_value = None

    h._barge_in_enabled = barge_in_enabled

    return h


@pytest.fixture(autouse=True)
def _stub_bus(monkeypatch):
    """Every process_voice_turn call emits interaction_logged on the real
    module-level bus singleton; stub it out so tests don't depend on (or
    leak state into) the real event bus."""
    fake_bus = MagicMock()
    monkeypatch.setattr(main, "bus", fake_bus)
    return fake_bus


# ---------------------------------------------------------------------------
# process_voice_turn
# ---------------------------------------------------------------------------

def test_process_voice_turn_empty_input_returns_empty_string_and_does_nothing():
    h = make_hestia()
    assert h.process_voice_turn("   ") == ""
    h.nlu.understand.assert_not_called()
    h.orchestrator.try_stream_chat.assert_not_called()
    h.orchestrator.dispatch.assert_not_called()


def test_process_voice_turn_uses_streaming_path_when_stream_available():
    h = make_hestia()
    h.orchestrator.try_stream_chat.return_value = iter(["Hel", "lo "])
    h.tts.speak_stream.side_effect = lambda gen: list(gen)  # simulate real playback consuming it

    response = h.process_voice_turn("hello there")

    assert response == "Hello"
    h.tts.speak_stream.assert_called_once()
    h.orchestrator.dispatch.assert_not_called()  # blocking path skipped
    h.barge_in.reset.assert_called_once()
    h.barge_in.start.assert_called_once()
    assert h.barge_in.start.call_args.kwargs["on_barge_in"] == h.tts.stop
    h.barge_in.stop.assert_called_once()


def test_process_voice_turn_falls_back_to_blocking_dispatch_when_no_stream():
    h = make_hestia()
    h.orchestrator.try_stream_chat.return_value = None
    h.orchestrator.dispatch.return_value = "a plain response"

    response = h.process_voice_turn("what's the weather")

    assert response == "a plain response"
    h.orchestrator.dispatch.assert_called_once()
    h.tts.speak.assert_called_with("a plain response")
    h.tts.speak_stream.assert_not_called()


def test_process_voice_turn_falls_back_to_blocking_dispatch_when_try_stream_chat_raises():
    h = make_hestia()
    h.orchestrator.try_stream_chat.side_effect = RuntimeError("boom")
    h.orchestrator.dispatch.return_value = "recovered response"

    response = h.process_voice_turn("hello")

    assert response == "recovered response"
    h.orchestrator.dispatch.assert_called_once()


def test_process_voice_turn_nlu_failure_falls_back_to_chat_intent():
    h = make_hestia()
    h.nlu.understand.side_effect = RuntimeError("nlu exploded")
    h.orchestrator.try_stream_chat.return_value = None
    h.orchestrator.dispatch.return_value = "still works"

    response = h.process_voice_turn("hello")

    assert response == "still works"
    # dispatch still gets called with the safe fallback nlu_result
    call_args = h.orchestrator.dispatch.call_args.args
    assert call_args[1] == {"intent": "chat", "entities": {}, "response": ""}


def test_process_voice_turn_dispatch_failure_returns_safe_fallback():
    h = make_hestia()
    h.orchestrator.try_stream_chat.return_value = None
    h.orchestrator.dispatch.side_effect = RuntimeError("dispatch exploded")

    response = h.process_voice_turn("hello")

    assert response == "I'm sorry, something went wrong."
    h.tts.speak.assert_called_with("I'm sorry, something went wrong.")


def test_process_voice_turn_unwraps_json_leaked_response_before_speaking():
    h = make_hestia()
    h.orchestrator.try_stream_chat.return_value = None
    h.orchestrator.dispatch.return_value = '{"response": "clean text"}'

    response = h.process_voice_turn("hello")

    assert response == "clean text"
    h.tts.speak.assert_called_with("clean text")


def test_process_voice_turn_flushes_wake_word_audio_queue():
    h = make_hestia()
    h.process_voice_turn("hello")
    h.wake_detector.flush_audio_queue.assert_called_once()


def test_process_voice_turn_flush_audio_queue_failure_is_swallowed():
    h = make_hestia()
    h.wake_detector.flush_audio_queue.side_effect = RuntimeError("mic gone")
    # must not raise
    response = h.process_voice_turn("hello")
    assert response == "a plain response"


def test_process_voice_turn_emits_interaction_logged_event(_stub_bus):
    h = make_hestia()
    h.orchestrator.try_stream_chat.return_value = None
    h.orchestrator.dispatch.return_value = "a plain response"

    h.process_voice_turn("Hello There")

    _stub_bus.emit_sync.assert_called_once_with(
        "interaction_logged",
        {"query": "hello there", "response": "a plain response", "intent": "chat"},
    )


# ---------------------------------------------------------------------------
# _speak_streaming
# ---------------------------------------------------------------------------

def test_speak_streaming_joins_chunks_into_final_response():
    h = make_hestia()
    h.tts.speak_stream.side_effect = lambda gen: list(gen)

    response = h._speak_streaming(iter(["The answer ", "is 42."]))

    assert response == "The answer is 42."


def test_speak_streaming_returns_done_when_stream_yields_nothing():
    h = make_hestia()
    h.tts.speak_stream.side_effect = lambda gen: list(gen)

    response = h._speak_streaming(iter([]))

    assert response == "Done."


def test_speak_streaming_returns_done_when_stream_yields_only_whitespace():
    h = make_hestia()
    h.tts.speak_stream.side_effect = lambda gen: list(gen)

    response = h._speak_streaming(iter(["   ", "\n"]))

    assert response == "Done."


def test_speak_streaming_runs_under_barge_in_and_waits_for_playback():
    h = make_hestia()
    h.tts.speak_stream.side_effect = lambda gen: list(gen)

    h._speak_streaming(iter(["hi"]))

    h.barge_in.reset.assert_called_once()
    h.barge_in.start.assert_called_once()
    h.tts.wait_until_done.assert_called_once()
    h.barge_in.stop.assert_called_once()


# ---------------------------------------------------------------------------
# _speak_with_barge_in
# ---------------------------------------------------------------------------

def test_speak_with_barge_in_speaks_the_full_response_under_barge_in():
    h = make_hestia()
    h._speak_with_barge_in("a full response")
    h.tts.speak.assert_called_once_with("a full response")
    h.barge_in.start.assert_called_once()
    h.barge_in.stop.assert_called_once()


# ---------------------------------------------------------------------------
# _with_barge_in
# ---------------------------------------------------------------------------

def test_with_barge_in_disabled_speaks_directly_without_touching_barge_in():
    h = make_hestia(barge_in_enabled=False)
    speak_fn = MagicMock()

    h._with_barge_in(speak_fn)

    speak_fn.assert_called_once()
    h.tts.wait_until_done.assert_called_once()
    h.barge_in.reset.assert_not_called()
    h.barge_in.start.assert_not_called()
    h.barge_in.stop.assert_not_called()


def test_with_barge_in_enabled_arms_listener_around_speak_fn():
    h = make_hestia(barge_in_enabled=True)
    manager = MagicMock()
    manager.attach_mock(h.barge_in.reset, "reset")
    manager.attach_mock(h.barge_in.start, "start")
    manager.attach_mock(h.barge_in.stop, "stop")
    speak_fn = MagicMock()
    manager.attach_mock(speak_fn, "speak_fn")

    h._with_barge_in(speak_fn)

    # reset -> start -> speak_fn -> stop, in that order
    assert [c[0] for c in manager.mock_calls] == ["reset", "start", "speak_fn", "stop"]
    h.barge_in.start.assert_called_once_with(on_barge_in=h.tts.stop)
    h.tts.wait_until_done.assert_called_once()


def test_with_barge_in_stops_listener_even_if_speak_fn_raises():
    h = make_hestia(barge_in_enabled=True)
    speak_fn = MagicMock(side_effect=RuntimeError("tts blew up"))

    with pytest.raises(RuntimeError):
        h._with_barge_in(speak_fn)

    h.barge_in.stop.assert_called_once()
    h.tts.wait_until_done.assert_not_called()  # never reached — speak_fn raised first


# ---------------------------------------------------------------------------
# run_voice_loop
# ---------------------------------------------------------------------------
#
# process_voice_turn is stubbed out in every run_voice_loop test below: its
# own behaviour is covered exhaustively above, and re-mocking nlu/orchestrator
# for every loop-shape test here would only obscure what run_voice_loop itself
# is actually responsible for (wake-word vs. skip-wake-word vs. pending-audio
# routing, short-input reprompt, exit words, and the post-turn barge-in
# hand-off decision).

def test_run_voice_loop_normal_wake_word_flow_then_exit():
    h = make_hestia()
    h.process_voice_turn = MagicMock()
    h.wake_detector.listen_for_wake_word.return_value = True
    h.stt.listen_once.side_effect = ["hello there", "bye"]

    h.run_voice_loop()

    assert h.tts.speak.call_args_list[0] == call("Yes?")
    assert h.process_voice_turn.call_args_list == [call("hello there")]
    assert h.tts.speak.call_args_list[-1] == call("Goodbye.")


def test_run_voice_loop_continues_when_wake_word_not_detected():
    h = make_hestia()
    h.process_voice_turn = MagicMock()
    h.wake_detector.listen_for_wake_word.side_effect = [False, False, True]
    h.stt.listen_once.return_value = "stop"

    h.run_voice_loop()

    assert h.wake_detector.listen_for_wake_word.call_count == 3
    h.stt.listen_once.assert_called_once()  # only after wake word finally detected


def test_run_voice_loop_short_input_reprompts_without_processing():
    h = make_hestia()
    h.process_voice_turn = MagicMock()
    h.wake_detector.listen_for_wake_word.return_value = True
    h.stt.listen_once.side_effect = ["a", "exit"]  # "a" is below _MIN_VOICE_INPUT_LEN

    h.run_voice_loop()

    h.process_voice_turn.assert_not_called()
    assert call("I didn't catch that.") in h.tts.speak.call_args_list


def test_run_voice_loop_empty_transcription_reprompts():
    h = make_hestia()
    h.process_voice_turn = MagicMock()
    h.wake_detector.listen_for_wake_word.return_value = True
    h.stt.listen_once.side_effect = ["", "shutdown"]

    h.run_voice_loop()

    h.process_voice_turn.assert_not_called()
    assert call("I didn't catch that.") in h.tts.speak.call_args_list


def test_run_voice_loop_skips_wake_word_after_barge_in_without_captured_audio():
    h = make_hestia()
    h.process_voice_turn = MagicMock()
    h.wake_detector.listen_for_wake_word.return_value = True
    h.stt.listen_once.side_effect = ["hello there", "bye"]
    h.barge_in.consume_triggered.side_effect = [True, False]
    h.barge_in.consume_captured_audio.return_value = None  # no usable audio captured

    h.run_voice_loop()

    # wake word only checked once — second turn skipped straight to listen_once
    h.wake_detector.listen_for_wake_word.assert_called_once()
    assert h.stt.listen_once.call_count == 2
    h.stt.transcribe_audio.assert_not_called()


def test_run_voice_loop_uses_captured_audio_directly_after_barge_in():
    h = make_hestia()
    h.process_voice_turn = MagicMock()
    h.wake_detector.listen_for_wake_word.return_value = True
    h.stt.listen_once.return_value = "hello there"
    h.barge_in.consume_triggered.side_effect = [True, False]
    h.barge_in.consume_captured_audio.return_value = b"\x00\x01raw-pcm-audio"
    h.stt.transcribe_audio.return_value = "bye"

    h.run_voice_loop()

    # second turn used the captured audio directly, never re-opened the mic
    # via wake-word detection or a fresh listen_once() call
    h.wake_detector.listen_for_wake_word.assert_called_once()
    h.stt.listen_once.assert_called_once()
    h.stt.transcribe_audio.assert_called_once_with(b"\x00\x01raw-pcm-audio")


def test_run_voice_loop_ignores_barge_in_state_when_disabled():
    h = make_hestia(barge_in_enabled=False)
    h.process_voice_turn = MagicMock()
    h.wake_detector.listen_for_wake_word.return_value = True
    h.stt.listen_once.side_effect = ["hello there", "bye"]

    h.run_voice_loop()

    h.barge_in.consume_triggered.assert_not_called()


def test_run_voice_loop_calls_shutdown_on_exit():
    h = make_hestia()
    h.process_voice_turn = MagicMock()
    h._shutdown = MagicMock()
    h.wake_detector.listen_for_wake_word.return_value = True
    h.stt.listen_once.return_value = "bye"

    h.run_voice_loop()

    h._shutdown.assert_called_once()


def test_run_voice_loop_calls_shutdown_even_on_keyboard_interrupt():
    h = make_hestia()
    h._shutdown = MagicMock()
    h.wake_detector.listen_for_wake_word.side_effect = KeyboardInterrupt()

    h.run_voice_loop()  # must not raise

    h._shutdown.assert_called_once()