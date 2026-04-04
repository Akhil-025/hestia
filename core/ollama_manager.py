# core/ollama_manager.py

import requests
import time


class OllamaManager:
    def __init__(self, host="127.0.0.1", port=11434):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    def is_running(self) -> bool:
        try:
            r = requests.get(self.base_url, timeout=3)
            return r.status_code == 200
        except:
            return False

    def ensure_running(self, retries=5, delay=2) -> bool:
        for i in range(retries):
            if self.is_running():
                print(f"[OllamaManager] Ollama is running at {self.base_url}")
                return True
            print(f"[OllamaManager] Waiting for Ollama... ({i+1}/{retries})")
            time.sleep(delay)
        print("[OllamaManager] Ollama not reachable.")
        return False