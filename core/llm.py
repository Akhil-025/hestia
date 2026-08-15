# core/llm.py

from core.ollama_client import generate


class HestiaLLM:
    """
    Thin wrapper around the Ollama generate function.
    Passed as a dependency to modules that need LLM access (Athena, Mnemosyne, Iris).
    Not a BaseModule — this is infrastructure, not a capability.
    """

    def __init__(self, host: str, port: int, model: str):
        self.host  = host
        self.port  = port
        self.model = model

    def generate(self, prompt: str, fmt: str = None, options: dict = None) -> str:
        # Callers must specify fmt explicitly (e.g. fmt="json") when they need
        # structured output. Auto-detecting JSON from a trailing "}" in the
        # prompt text was unreliable and has been removed.
        #
        # `options` (e.g. {"temperature": 0.2}) is forwarded to Ollama as-is.
        # Left as None by default so existing callers (chat, etc.) keep
        # Ollama's normal sampling behaviour; analytical/JSON callers should
        # pass a low temperature explicitly.
        return generate(
            prompt,
            model=self.model,
            host=self.host,
            port=self.port,
            fmt=fmt,
            options=options,
        )