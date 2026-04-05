# core/ollama_manager.py

import requests
import time
import subprocess

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
        # Step 1: Check if already running
        if self.is_running():
            print(f"[OllamaManager] Ollama is running at {self.base_url}")
            return True

        print("[OllamaManager] Ollama not running. Starting it...")

        # Step 2: Start Ollama
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"[OllamaManager] Failed to start Ollama: {e}")
            return False

        # Step 3: Wait for it to come up
        for i in range(retries * 2):  # give more time
            if self.is_running():
                print(f"[OllamaManager] Ollama started successfully at {self.base_url}")
                return True
            print(f"[OllamaManager] Waiting for Ollama startup... ({i+1})")
            time.sleep(1)

        print("[OllamaManager] Ollama failed to start.")
        return False