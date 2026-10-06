# tests/test_process_split.py
"""Backlog #20: the three-process split (core / voice / jobs) and its supervisor."""
import os
import sys
import threading
import time
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.event_queue import EventQueue
from core.process_split import (
    CoreQueryServer, SAY_TOPIC, Supervisor, VoiceFrontend, profile_for, queue_path,
)


def _wait(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------- profiles

def test_all_role_is_the_original_single_process():
    p = profile_for("all")
    assert (p.stt, p.tts, p.wake_word, p.barge_in, p.heartbeat, p.chronos_scheduler,
            p.web_ui, p.telegram, p.sync_api) == (True,) * 9
    assert not p.speak_via_queue and not p.serve_queries and p.export_topics == ()


def test_core_has_no_microphone_and_no_background_jobs():
    p = profile_for("core")
    assert not p.wake_word and not p.barge_in
    assert not p.heartbeat and not p.chronos_scheduler
    assert p.web_ui and p.telegram and p.sync_api
    assert p.serve_queries and p.speak_via_queue and "speak" in p.import_topics


def test_jobs_has_only_background_work_and_exports_speech():
    p = profile_for("jobs")
    assert p.heartbeat and p.chronos_scheduler
    assert not (p.stt or p.tts or p.wake_word or p.barge_in or p.web_ui or p.telegram or p.sync_api)
    assert p.export_topics == ("speak",)


def test_voice_and_supervisor_do_not_build_an_assistant():
    for role in ("voice", "supervisor", "bogus"):
        with pytest.raises(ValueError):
            profile_for(role)


def test_every_background_job_runs_in_exactly_one_process():
    owners = [r for r in ("core", "jobs") if profile_for(r).heartbeat]
    assert owners == ["jobs"]
    owners = [r for r in ("core", "jobs") if profile_for(r).chronos_scheduler]
    assert owners == ["jobs"]


def test_queue_path_default_and_override():
    assert str(queue_path({})).endswith("events.db")
    assert str(queue_path({"processes": {"queue_path": "x/y.db"}})) == os.path.join("x", "y.db")


# ---------------------------------------------------------------- core server

def test_core_query_server_answers_the_voice_process(tmp_path):
    from core.event_queue import QueueRPC
    path = tmp_path / "e.db"
    server = CoreQueryServer(EventQueue(path, "core"), lambda t: f"you said {t}")
    server._rpc.interval = 0.02
    server.start()
    server._worker.interval = 0.02
    out = QueueRPC(EventQueue(path, "voice"), "voice", interval=0.02).call("query", {"text": "hello"}, timeout=5)
    server.stop()
    assert out == {"response": "you said hello"}


def test_core_query_server_ignores_blank_text(tmp_path):
    from core.event_queue import QueueRPC
    path = tmp_path / "e.db"
    called = []
    server = CoreQueryServer(EventQueue(path, "core"), lambda t: called.append(t) or "x")
    server.start()
    out = QueueRPC(EventQueue(path, "voice"), "voice", interval=0.02).call("query", {"text": "  "}, timeout=5)
    server.stop()
    assert out == {"response": ""} and called == []


# ---------------------------------------------------------------- voice frontend

class FakeTTS:
    def __init__(self): self.said = []
    def speak(self, text, voice=None): self.said.append((text, voice))
    def wait_until_done(self): pass
    def stop(self): pass


class FakeBarge:
    def __init__(self): self.started = self.stopped = 0
    def reset(self): pass
    def start(self, on_barge_in=None): self.started += 1
    def stop(self): self.stopped += 1


def _frontend(tmp_path, barge=None):
    q = EventQueue(tmp_path / "e.db", "voice")
    tts = FakeTTS()
    return VoiceFrontend(stt=object(), tts=tts, wake_detector=object(), barge_in=barge, queue=q,
                         poll_interval=0.02, query_timeout=2.0), tts, EventQueue(tmp_path / "e.db", "core")


def test_pump_speaks_queued_utterances_in_order(tmp_path):
    vf, tts, core_q = _frontend(tmp_path)
    core_q.publish(SAY_TOPIC, {"text": "one"})
    core_q.publish(SAY_TOPIC, {"text": "two", "voice": "calm"})
    core_q.publish(SAY_TOPIC, {"text": "   "})
    assert vf.pump() == 2
    assert tts.said == [("one", None), ("two", "calm")]
    assert vf.pump() == 0


def test_pump_arms_barge_in_around_speech(tmp_path):
    barge = FakeBarge()
    vf, tts, core_q = _frontend(tmp_path, barge)
    core_q.publish(SAY_TOPIC, {"text": "hi"})
    vf.pump()
    assert barge.started == 1 and barge.stopped == 1


def test_handle_text_sends_to_core_then_speaks_the_reply(tmp_path):
    vf, tts, core_q = _frontend(tmp_path)
    server = CoreQueryServer(core_q, lambda t: (core_q.publish(SAY_TOPIC, {"text": f"re: {t}"}), f"re: {t}")[1])
    server.start()
    reply = vf.handle_text("weather")
    server.stop()
    assert reply == "re: weather" and tts.said == [("re: weather", None)]


def test_handle_text_apologises_when_core_is_down(tmp_path):
    vf, tts, _ = _frontend(tmp_path)
    vf.query_timeout = 0.3
    assert vf.handle_text("hello") == ""
    assert "can't reach" in tts.said[0][0]


def test_voice_run_refuses_without_working_stt(tmp_path):
    q = EventQueue(tmp_path / "e.db", "voice")
    vf = VoiceFrontend(None, FakeTTS(), None, None, q)
    with pytest.raises(RuntimeError, match="speech-to-text"):
        vf.run()


class _OneShotWake:
    def __init__(self, stop_after): self.n = 0
    def listen_for_wake_word(self, timeout=None):
        self.n += 1
        return self.n == 1
    def flush_audio_queue(self): pass


class _Stt:
    def __init__(self, phrases): self.phrases = list(phrases)
    def listen_once(self, max_duration=10): return self.phrases.pop(0)


def test_voice_loop_wake_listen_ask_speak_then_exit_word(tmp_path):
    q = EventQueue(tmp_path / "e.db", "voice")
    tts = FakeTTS()
    wake = types.SimpleNamespace(calls=0, flush_audio_queue=lambda: None)

    def listen(timeout=None):
        wake.calls += 1
        return True
    wake.listen_for_wake_word = listen
    vf = VoiceFrontend(_Stt(["what time is it", "bye"]), tts, wake, None, q, poll_interval=0.02, query_timeout=3)
    core_q = EventQueue(tmp_path / "e.db", "core")
    server = CoreQueryServer(core_q, lambda t: (core_q.publish(SAY_TOPIC, {"text": "It is noon."}), "It is noon.")[1])
    server.start()
    t = threading.Thread(target=vf.run); t.start()
    t.join(8)
    server.stop()
    spoken = [s for s, _ in tts.said]
    assert "It is noon." in spoken and spoken[-1] == "Goodbye." and not t.is_alive()


# ---------------------------------------------------------------- supervisor

class FakeProc:
    def __init__(self, cmd, script):
        self.cmd, self.pid, self._codes = cmd, id(self) % 1000, list(script)
        self.terminated = self.killed = False
        self._polls = 0

    def poll(self):
        if self.terminated or self.killed:
            return -15
        return self._codes[0] if self._codes else None

    def terminate(self): self.terminated = True
    def kill(self): self.killed = True


class Launcher:
    def __init__(self, scripts): self.scripts, self.spawned = scripts, []

    def __call__(self, cmd):
        role = cmd[cmd.index("--role") + 1]
        script = self.scripts.get(role, [])
        code = script.pop(0) if isinstance(script, list) and script else None
        p = FakeProc(cmd, [] if code is None else [code])
        self.spawned.append((role, p))
        return p

    def count(self, role): return sum(1 for r, _ in self.spawned if r == role)


def _sup(scripts, **kw):
    launcher = Launcher(scripts)
    clock = types.SimpleNamespace(t=0.0)
    sleeps = []
    sup = Supervisor(base_command=["py", "main.py"], popen=launcher, clock=lambda: clock.t,
                     sleep=lambda s: (sleeps.append(s), setattr(clock, "t", clock.t + s)), **kw)
    return sup, launcher, sleeps


def test_supervisor_commands_carry_the_role():
    sup, _, _ = _sup({})
    assert sup.command_for("jobs") == ["py", "main.py", "--role", "jobs"]


def test_supervisor_rejects_roles_it_cannot_run():
    with pytest.raises(ValueError):
        Supervisor(roles=["core", "all"])


def test_supervisor_starts_every_role_in_order():
    sup, launcher, _ = _sup({})
    for r in sup.children:
        sup._spawn(sup.children[r])
    assert [r for r, _ in launcher.spawned] == ["core", "jobs", "voice"]


def test_crashed_child_restarts_with_backoff_others_untouched():
    sup, launcher, sleeps = _sup({"jobs": [1, 1]})
    for r in sup.children:
        sup._spawn(sup.children[r])
    sup.check_once()                      # jobs crashed -> restart after 1s
    sup.check_once()                      # restarted copy crashed again -> after 2s
    assert launcher.count("jobs") == 3 and launcher.count("core") == 1
    assert sleeps == [1, 2]


def test_clean_exit_is_not_restarted():
    sup, launcher, _ = _sup({"voice": [0]})
    for r in sup.children:
        sup._spawn(sup.children[r])
    sup.check_once()
    assert launcher.count("voice") == 1 and sup.status()["voice"]["state"] == "exited"


def test_crash_loop_gives_up_on_that_child_only():
    sup, launcher, _ = _sup({"voice": [1] * 20}, max_restarts=3)
    for r in sup.children:
        sup._spawn(sup.children[r])
    for _ in range(10):
        sup.check_once()
    st = sup.status()
    assert st["voice"]["state"] == "failed" and st["core"]["state"] == "running"
    assert launcher.count("voice") <= 5


def test_stop_terminates_children_and_kills_stragglers():
    sup, launcher, _ = _sup({})
    for r in sup.children:
        sup._spawn(sup.children[r])
    stubborn = launcher.spawned[1][1]
    stubborn.terminate = lambda: None           # ignores SIGTERM
    sup.stop(grace=0.2)
    assert all(p.terminated for r, p in launcher.spawned if p is not stubborn)
    assert stubborn.killed
    assert all(c["state"] == "stopped" for c in sup.status().values())
