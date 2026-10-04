"""
scripts/voice_latency.py

Latency measurement with a regression budget for the voice round trip
(backlog #211): speech-to-text, NLU, dispatch and text-to-speech.

    python scripts/voice_latency.py                    # measure and compare with the budget
    python scripts/voice_latency.py --runs 20
    python scripts/voice_latency.py --wav sample.wav   # also time STT on a real recording
    python scripts/voice_latency.py --update-baseline  # record this machine's numbers

Two limits are checked per stage, on the 95th percentile (the slow turn you
actually notice, not the average):

* the **hard budget** in ``config/latency_budget.json`` (milliseconds, written
  with starting values on first run: edit them to what you consider acceptable);
* the **baseline** in ``data/latency_baseline.json``, the numbers recorded the
  last time you ran ``--update-baseline``, allowed to grow by ``--tolerance``
  (default 25%) before it counts as a regression.

Exit codes: 0 within both limits, 1 over a limit, 2 could not measure.

Honest scope: this measures the stages that can be driven without a person
speaking. STT is only timed when you give it a WAV file; without one the stage
is reported as skipped, never as passing. A measurement of an idle machine
says nothing about the same turn while a model is loading, so run it the way
you use Hestia. The in-process overhead of Hestia's own code (everything that
is not a model) is checked on every test run by tests/test_voice_latency.py.

The statistics and comparison are pure and tested without any model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
BUDGET_PATH = ROOT / "config" / "latency_budget.json"
BASELINE_PATH = ROOT / "data" / "latency_baseline.json"

STAGES = ("stt", "nlu", "dispatch", "tts")
# Starting values for a typical laptop with a local 7B model; the file is the
# place to change them. "round_trip" is the budget for the whole turn.
DEFAULT_BUDGET_MS = {"stt": 2500, "nlu": 3000, "dispatch": 4000, "tts": 1500, "round_trip": 8000}
DEFAULT_TOLERANCE = 0.25

# Short, read-only, covers both the alias/fast paths and a real model call.
DEFAULT_PROMPTS = ("what time is it", "how are my habits going", "tell me a short joke")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile; 0.0 for no data."""
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return float(data[0])
    rank = (len(data) - 1) * max(0.0, min(100.0, pct)) / 100.0
    lo, hi = math.floor(rank), math.ceil(rank)
    return float(data[lo] + (data[hi] - data[lo]) * (rank - lo))


@dataclass
class StageStats:
    stage: str
    samples: list[float] = field(default_factory=list)
    skipped: str = ""                     # reason, when the stage couldn't run

    @property
    def p50(self) -> float:
        return percentile(self.samples, 50)

    @property
    def p95(self) -> float:
        return percentile(self.samples, 95)


def time_stage(fn: Callable[[], object], runs: int, *,
               clock: Callable[[], float] = time.perf_counter) -> list[float]:
    """Milliseconds for each of *runs* calls to *fn*. One warm-up call is made
    first and discarded: the first call after a model loads is a different
    measurement from the ones you care about."""
    fn()
    out = []
    for _ in range(runs):
        start = clock()
        fn()
        out.append((clock() - start) * 1000.0)
    return out


# ---------------------------------------------------------------------------
# Budget and baseline files
# ---------------------------------------------------------------------------

def load_json_numbers(path: Path, defaults: Optional[dict] = None) -> dict[str, float]:
    """A ``{name: number}`` file. Missing, unreadable or malformed => defaults;
    individual non-numeric or non-positive entries fall back to the default."""
    base = dict(defaults or {})
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return base
    if not isinstance(raw, dict):
        return base
    for key, value in raw.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and math.isfinite(value) and value > 0:
            base[key] = float(value)
    return base


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    stage: str
    p95_ms: Optional[float]
    budget_ms: Optional[float]
    baseline_ms: Optional[float]
    status: str            # "ok" | "over budget" | "regressed" | "skipped"
    note: str = ""


def judge(stats: StageStats, budget: dict[str, float], baseline: dict[str, float],
          tolerance: float = DEFAULT_TOLERANCE) -> Verdict:
    b = budget.get(stats.stage)
    base = baseline.get(stats.stage)
    if stats.skipped or not stats.samples:
        return Verdict(stats.stage, None, b, base, "skipped", stats.skipped or "no samples")
    p95 = stats.p95
    if b is not None and p95 > b:
        return Verdict(stats.stage, p95, b, base, "over budget",
                       f"p95 {p95:.0f} ms is over the {b:.0f} ms budget")
    if base is not None and p95 > base * (1.0 + tolerance):
        return Verdict(stats.stage, p95, b, base, "regressed",
                       f"p95 {p95:.0f} ms is more than {tolerance:.0%} above the recorded {base:.0f} ms")
    return Verdict(stats.stage, p95, b, base, "ok")


def format_report(stats: Sequence[StageStats], verdicts: Sequence[Verdict]) -> str:
    lines = [f"{'stage':<11}{'p50 ms':>9}{'p95 ms':>9}{'budget':>9}{'baseline':>10}  status"]
    by = {s.stage: s for s in stats}
    for v in verdicts:
        s = by.get(v.stage)
        p50 = f"{s.p50:.0f}" if s and s.samples else "-"
        p95 = f"{v.p95_ms:.0f}" if v.p95_ms is not None else "-"
        bud = f"{v.budget_ms:.0f}" if v.budget_ms is not None else "-"
        base = f"{v.baseline_ms:.0f}" if v.baseline_ms is not None else "-"
        lines.append(f"{v.stage:<11}{p50:>9}{p95:>9}{bud:>9}{base:>10}  {v.status}"
                     + (f"  ({v.note})" if v.note and v.status != 'ok' else ""))
    return "\n".join(lines)


def exit_code(verdicts: Sequence[Verdict]) -> int:
    measured = [v for v in verdicts if v.status != "skipped"]
    if not measured:
        return 2
    return 1 if any(v.status in ("over budget", "regressed") for v in measured) else 0


# ---------------------------------------------------------------------------
# Driving the real application
# ---------------------------------------------------------------------------

def read_wav(path: str):
    """16-bit mono PCM WAV -> float32 numpy array in [-1, 1], plus its rate."""
    import wave
    import numpy as np
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM audio")
        frames = w.readframes(w.getnframes())
        channels = w.getnchannels()
        rate = w.getframerate()
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def measure_stages(hestia, prompts: Sequence[str], runs: int, wav: Optional[str] = None
                   ) -> tuple[list[StageStats], list[float]]:
    """Time each stage on a booted Hestia; also each whole text turn."""
    stats = []

    # --- stt
    stt = getattr(hestia, "stt", None)
    if wav and stt is not None:
        try:
            audio, _rate = read_wav(wav)
            stats.append(StageStats("stt", time_stage(lambda: stt.transcribe_audio(audio), runs)))
        except Exception as exc:                       # noqa: BLE001
            stats.append(StageStats("stt", skipped=f"{type(exc).__name__}: {exc}"))
    else:
        stats.append(StageStats("stt", skipped="no --wav given" if stt is not None
                                else "no STT in this build"))

    cycle = list(prompts)
    counter = {"i": 0}

    def next_prompt() -> str:
        counter["i"] += 1
        return cycle[counter["i"] % len(cycle)]

    # --- nlu (cache/alias hits would flatter it, so vary the text slightly)
    nlu = hestia.nlu
    stats.append(StageStats("nlu", time_stage(
        lambda: nlu.understand(f"{next_prompt()} {counter['i']}"), runs)))

    # --- dispatch (nlu result computed outside the timer)
    def dispatch_once():
        text = next_prompt()
        result = nlu.understand(text)
        t0 = time.perf_counter()
        hestia.orchestrator.dispatch(text, result)
        return (time.perf_counter() - t0) * 1000.0

    dispatch_ms = [dispatch_once() for _ in range(runs)]
    stats.append(StageStats("dispatch", dispatch_ms))

    # --- tts (synthesis only, no playback)
    tts = getattr(hestia, "tts", None)
    synth = getattr(tts, "synthesize_wav_bytes", None)
    if callable(synth):
        try:
            stats.append(StageStats("tts", time_stage(lambda: synth("This is a short test sentence."), runs)))
        except Exception as exc:                       # noqa: BLE001
            stats.append(StageStats("tts", skipped=f"{type(exc).__name__}: {exc}"))
    else:
        stats.append(StageStats("tts", skipped="no synthesize_wav_bytes on this TTS"))

    # --- whole text turn
    turns = time_stage(lambda: hestia.process_text(next_prompt()), runs)
    return stats, turns


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Voice round-trip latency against a budget.")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--wav", help="16-bit PCM WAV of a spoken sentence, to time STT")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args(argv)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from scripts.smoke_test import boot_hestia
        hestia = boot_hestia(args.config)
    except Exception as exc:                           # noqa: BLE001
        print(f"Hestia would not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        stats, turns = measure_stages(hestia, DEFAULT_PROMPTS, max(1, args.runs), args.wav)
    finally:
        shutdown = getattr(hestia, "_shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:                          # noqa: BLE001
                pass

    if not BUDGET_PATH.exists():
        write_json(BUDGET_PATH, DEFAULT_BUDGET_MS)
        print(f"Wrote starting budget to {BUDGET_PATH.relative_to(ROOT)}; edit it to taste.")
    budget = load_json_numbers(BUDGET_PATH, DEFAULT_BUDGET_MS)
    baseline = load_json_numbers(BASELINE_PATH)

    stats.append(StageStats("round_trip", turns))
    verdicts = [judge(s, budget, baseline, args.tolerance) for s in stats]
    print(format_report(stats, verdicts))

    if args.update_baseline:
        write_json(BASELINE_PATH, {s.stage: round(s.p95, 1) for s in stats if s.samples})
        print(f"Recorded baseline in {BASELINE_PATH.relative_to(ROOT)}.")
    return exit_code(verdicts)


if __name__ == "__main__":
    raise SystemExit(main())
