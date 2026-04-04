"""
modules/athena/hestia_llm_adapter.py

Bridges Hestia's LLM interface → Athena's expected interface.
QueryService calls self.ai.generate(prompt) and self.ai.generate_answer(question, sources).
"""
import logging
from typing import Dict, Any, List

from modules.athena.models import SourceDocument
from modules.athena.services.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class HestiaLLMAdapter:
    """
    Wraps any Hestia LLM object so Athena's QueryService can call it.

    Hestia LLM must expose:
        llm.generate(prompt: str) -> str   (plain string response)
    """

    def __init__(self, hestia_llm) -> None:
        self.llm = hestia_llm

    # ── Core interface (used by QueryService._generate_answer) ───────────────

    def generate(self, prompt: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Generate a response and return a normalized dict.
        Athena expects: {"text": str, "error": str|None, "meta": dict}
        """
        try:
            response = self.llm.generate(prompt)
            # Handle both plain-string and dict responses from Hestia LLMs
            if isinstance(response, dict):
                text = response.get("text", "")
            else:
                text = str(response)
            return {"text": text, "error": None, "meta": {}}
        except Exception as e:
            logger.exception("HestiaLLMAdapter.generate failed")
            return {"text": "", "error": str(e), "meta": {}}

    # ── Extended interface (used by fallback path in QueryService) ────────────

    def generate_answer(
        self,
        question: str,
        sources: List[SourceDocument],
        use_cloud: bool = False,
    ) -> str:
        """
        Build a full RAG prompt from sources and generate an answer.
        Returns the answer string (not a dict).
        """
        builder = (
            PromptBuilder.for_cloud_llm() if use_cloud
            else PromptBuilder.for_local_llm()
        )
        prompt = builder.build(question, sources)
        result = self.generate(prompt)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result.get("text", "")

    def has_cloud_llm(self) -> bool:
        """Athena checks this before trying a cloud fallback."""
        return False
