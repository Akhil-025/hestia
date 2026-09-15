# tests/test_llm.py
"""
Regression tests for core/llm.py.

Run with:  pytest tests/test_llm.py -v

HestiaLLM is a thin dependency-injection wrapper around
core.ollama_client.generate — these tests just confirm it forwards its
constructor args and call args through correctly, since several modules
(Athena, Mnemosyne, Iris) depend on that forwarding being exact.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.llm import HestiaLLM


def test_generate_forwards_host_port_model_from_constructor():
    llm = HestiaLLM(host="192.168.1.10", port=11500, model="llama3")

    with patch("core.llm.generate", return_value="hi") as mock_generate:
        result = llm.generate("hello")

    assert result == "hi"
    mock_generate.assert_called_once_with(
        "hello",
        model="llama3",
        host="192.168.1.10",
        port=11500,
        fmt=None,
        options=None,
    )


def test_generate_forwards_fmt_and_options():
    llm = HestiaLLM(host="127.0.0.1", port=11434, model="mistral")
    schema = {"type": "object"}

    with patch("core.llm.generate", return_value="{}") as mock_generate:
        llm.generate("classify", fmt="json", options={"temperature": 0.2})

    mock_generate.assert_called_once_with(
        "classify",
        model="mistral",
        host="127.0.0.1",
        port=11434,
        fmt="json",
        options={"temperature": 0.2},
    )


def test_generate_returns_whatever_the_underlying_client_returns():
    llm = HestiaLLM(host="h", port=1, model="m")
    with patch("core.llm.generate", return_value=""):
        assert llm.generate("anything") == ""
