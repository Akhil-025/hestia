# ollama_client.py

import requests


def generate(prompt, model="mistral", host="127.0.0.1", port=11434):
    try:
        response = requests.post(
            f"http://{host}:{port}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print(f"[OllamaClient] Error: {e}")
        return ""