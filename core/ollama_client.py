# ollama_client.py

import json
import logging
import time

import requests

# Shared logger name across every Ollama call site (this file and Pluto's
# separate LLMClient — see modules/pluto/llm_client.py) so `grep
# "hestia.llm_latency"` on the log file gives a single, comparable timeline
# of every LLM call Hestia makes, regardless of which module made it. This
# is deliberately just logging, not a profiler/tracer: it's the fastest way
# to get real numbers (which call sites are actually slow, how often NLU's
# retries fire, whether Pluto's ReAct loop is the bottleneck it's suspected
# to be) before committing to a bigger architectural change like splitting
# models or adding concurrency. See that decision written up in
# ollama_client.generate()'s docstring below.
logger = logging.getLogger("hestia.llm_latency")


def generate(prompt, model="mistral", host="127.0.0.1", port=11434,
             fmt=None, timeout=60, options=None):
    """
    Call Ollama's /api/generate and return the response text (or "" on
    failure — see the note in modules/hestia/orchestrator.py and item #4
    of the review for why that's a separate, not-yet-fixed problem).

    Every call is timed and logged (model, elapsed ms, prompt/response
    size) regardless of success or failure. This directly answers "worth
    profiling real end-to-end latency" from the original review: instead
    of guessing whether NLU classification, chat, RAG synthesis, or
    Pluto's multi-step ReAct loop is the actual bottleneck, that's now a
    log grep away. A single "what's my portfolio look like" query will
    show up as several distinct log lines (NLU classification + however
    many ReAct steps Pluto's agent takes), each with its own elapsed_ms —
    which is the real data needed to decide whether splitting NLU onto a
    smaller/faster model (see config/laptop_config.yaml's new `nlu.model`)
    is worth it, or whether the bottleneck is actually somewhere else
    entirely (e.g. a single slow ReAct step, not classification).
    """
    body = {"model": model, "prompt": prompt, "stream": False}
    if fmt:
        body["format"] = fmt
    if options:
        # e.g. {"temperature": 0.2} — Ollama defaults to ~0.8 when this is
        # omitted, which is fine for open-ended chat but is exactly the
        # wrong setting for "be specific to the topic" structured-analysis
        # prompts with thin grounding: high temperature on a prompt the
        # model can't honestly satisfy just increases how much it invents.
        # Callers doing analytical/JSON work should pass a low temperature
        # explicitly rather than relying on Ollama's chat-tuned default.
        body["options"] = options

    t0 = time.perf_counter()
    try:
        response = requests.post(
            f"http://{host}:{port}/api/generate",
            json=body,
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()["response"]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "ollama call ok source=core model=%s elapsed_ms=%.0f "
            "prompt_chars=%d response_chars=%d",
            model, elapsed_ms, len(prompt), len(result),
        )
        return result
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "ollama call failed source=core model=%s elapsed_ms=%.0f error=%s",
            model, elapsed_ms, e,
        )
        return ""


def generate_stream(prompt, model="mistral", host="127.0.0.1", port=11434,
                     timeout=60, options=None):
    """
    Stream tokens from Ollama's /api/generate as they're produced, instead
    of blocking until the whole response is ready like generate() does.

    Ollama's streaming response is one JSON object per line, each shaped
    like ``{"response": "<piece of text>", "done": bool, ...}``. This
    yields just the "response" text pieces, in the order they arrive,
    and stops once a line reports ``done: true``.

    This is what makes streaming TTS possible: the caller (see
    modules/hestia/core_module.py's stream_chat()) can start speaking the
    first sentence while Ollama is still generating the rest, instead of
    waiting for the full reply the way every other call site (chat, NLU,
    RAG synthesis, Pluto's ReAct loop) still does via generate() above —
    which is exactly right for those, since they need the complete text
    before they can do anything with it (parse JSON, decide a next step,
    etc). Streaming is only a win for the one case where the *first*
    partial output is already useful on its own: speaking a reply aloud.

    Same never-raise contract as generate(): any failure (connection
    refused, timeout, malformed line, a dropped connection mid-stream)
    logs a warning to the same "hestia.llm_latency" logger and simply
    stops yielding — callers get whatever text arrived before the
    failure (possibly nothing), never an exception.
    """
    body = {"model": model, "prompt": prompt, "stream": True}
    if options:
        body["options"] = options

    t0 = time.perf_counter()
    chars = 0
    try:
        with requests.post(
            f"http://{host}:{port}/api/generate",
            json=body,
            timeout=timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                piece = payload.get("response", "")
                if piece:
                    chars += len(piece)
                    yield piece
                if payload.get("done"):
                    break

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "ollama stream ok source=core model=%s elapsed_ms=%.0f "
            "prompt_chars=%d response_chars=%d",
            model, elapsed_ms, len(prompt), chars,
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "ollama stream failed source=core model=%s elapsed_ms=%.0f "
            "chars_before_failure=%d error=%s",
            model, elapsed_ms, chars, e,
        )
        return