# ollama_client.py

import requests


def generate(prompt, model="mistral", host="127.0.0.1", port=11434,
             fmt=None, timeout=60, options=None):
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
    try:
        response = requests.post(
            f"http://{host}:{port}/api/generate",
            json=body,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print(f"[OllamaClient] Error: {e}")
        return ""