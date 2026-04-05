# ollama_client.py

import requests


def generate(prompt, model="mistral", host="127.0.0.1", port=11434,
             fmt=None, timeout=60):
    body = {"model": model, "prompt": prompt, "stream": False}
    if fmt:
        body["format"] = fmt
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