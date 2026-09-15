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

            # core.ollama_client.generate() — what self.llm ultimately calls
            # — never raises. Every failure (Ollama unreachable, timed out,
            # bad HTTP status, ...) is caught there, logged to the
            # "hestia.llm_latency" logger, and turned into "". That meant a
            # fully failed generation call landed here looking identical to
            # success: {"text": "", "error": None, ...}. query_service.py's
            # _generate_answer() only raises LLMError when "error" is set
            # (correctly — see that file), so it never fired for this case:
            # a connectivity failure during RAG synthesis silently produced
            # an empty "answer" with no error at all, rather than surfacing
            # one. Athena's own contract (see class docstring) has no
            # legitimate case where an empty string is a valid answer, so
            # treat it as a failure here explicitly — regardless of which
            # underlying cause produced it (that detail is already in the
            # hestia.llm_latency log if needed).
            if not text or not text.strip():
                logger.warning(
                    "HestiaLLMAdapter.generate: LLM returned an empty "
                    "response (see hestia.llm_latency log for cause)."
                )
                return {
                    "text": "",
                    "error": "LLM returned an empty response.",
                    "meta": {},
                }
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