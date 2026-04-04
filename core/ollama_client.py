# ollama_client.py

import requests


def generate(prompt, model="mistral", host="127.0.0.1", port=11434):
    """Send a prompt to Ollama and return the response text."""
    response = requests.post(
        f"http://{host}:{port}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60
    )
    response.raise_for_status()
    return response.json()["response"]