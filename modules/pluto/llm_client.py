"""

modules/pluto/llm_client.py
Unified LLM client with fallback and structured output.

"""


import json
import logging
from enum import Enum
from typing import Optional, Union, Dict, Any

from .config import PlutoConfig
from .retry import retry

logger = logging.getLogger(__name__)


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"


class LLMClient:
    """Client for LLM interactions (Ollama or Kimi)."""

    def __init__(
        self,
        config: PlutoConfig,
        fallback_llm: Optional[Any] = None,
    ):
        self.config = config
        self.fallback_llm = fallback_llm
        self._ollama_base_url = f"http://{config.ollama_host}:{config.ollama_port}"

    @retry(max_retries=2, exceptions=(Exception,), delay=0.5)
    def generate(
        self,
        prompt: str,
        output_format: OutputFormat = OutputFormat.TEXT,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The prompt to send.
            output_format: Expected output format.
            **kwargs: Additional parameters.

        Returns:
            Either a string or a dictionary (if JSON format requested).
        """
        # If we have a fallback (e.g., Kimi client), use it
        if self.fallback_llm is not None:
            try:
                raw = self.fallback_llm.generate(prompt, **kwargs)
            except Exception as e:
                logger.error(f"Fallback LLM failed: {e}")
                raw = self._call_ollama(prompt, **kwargs)
        else:
            raw = self._call_ollama(prompt, **kwargs)

        if output_format == OutputFormat.JSON:
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM response: {e}")
                return {}
        return raw

    def _call_ollama(self, prompt: str, **kwargs) -> str:
        """Call Ollama API."""
        import requests
        url = f"{self._ollama_base_url}/api/generate"
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        payload.update(kwargs)
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise