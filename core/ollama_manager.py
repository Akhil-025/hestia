# core/ollama_manager.py

import logging
import subprocess
import time

import requests

logger = logging.getLogger(__name__)


class OllamaManager:
    def __init__(self, host="127.0.0.1", port=11434):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    def is_running(self) -> bool:
        try:
            r = requests.get(self.base_url, timeout=3)
            return r.status_code == 200
        # Narrowed from a bare `except:`, which also swallows
        # KeyboardInterrupt/SystemExit — someone hitting Ctrl+C during
        # startup while Ollama is unreachable would have had the interrupt
        # silently eaten here instead of stopping the process.
        # requests.RequestException covers every network-level failure
        # mode (connection refused, timeout, DNS, ...) this call can raise.
        except requests.RequestException as e:
            logger.debug("Ollama health check failed: %s", e)
            return False

    def ensure_running(self, retries=5, delay=2) -> bool:
        # Step 1: Check if already running
        if self.is_running():
            logger.info("Ollama is running at %s.", self.base_url)
            return True

        logger.info("Ollama not running. Starting it...")

        # Step 2: Start Ollama
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logger.error("Failed to start Ollama: %s", e)
            return False

        # Step 3: Wait for it to come up
        for i in range(retries * 2):  # give more time
            if self.is_running():
                logger.info("Ollama started successfully at %s.", self.base_url)
                return True
            logger.info("Waiting for Ollama startup... (%d)", i + 1)
            time.sleep(1)

        logger.error("Ollama failed to start.")
        return False