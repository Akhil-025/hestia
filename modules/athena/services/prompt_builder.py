"""
modules/athena/services/prompt_builder.py

Prompt building — fixed for Hestia (relative imports).
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

from modules.athena.models import SourceDocument
from modules.athena.services.context_assembler import (
    ContextAssembler, ContextConfig, FormattingStrategy,
)

logger = logging.getLogger(__name__)


class PromptMode(Enum):
    LOCAL_INSTRUCT = "local_instruct"
    LOCAL_CHAT     = "local_chat"
    CLOUD_ADVANCED = "cloud_advanced"
    CLOUD_BASIC    = "cloud_basic"


@dataclass
class PromptTemplate:
    system_prompt:   str
    user_template:   str
    context_prefix:  str = "CONTEXT FROM YOUR DOCUMENTS:"
    context_suffix:  str = "END OF CONTEXT"
    answer_prefix:   str = "ANSWER:"

    def render(self, question: str, context: str, **kwargs: Any) -> str:
        variables = {
            "question":       question,
            "context":        context,
            "context_prefix": self.context_prefix,
            "context_suffix": self.context_suffix,
            "answer_prefix":  self.answer_prefix,
            **kwargs,
        }
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt.format(**variables))
        parts.append(self.user_template.format(**variables))
        return "\n\n".join(parts)


class PromptTemplateLibrary:
    LOCAL_INSTRUCT = PromptTemplate(
        system_prompt=(
            "You are a precise academic assistant. Answer ONLY using the provided context. "
            "If information is missing, state: 'The provided documents do not cover this topic.' "
            "Be specific and cite sources."
        ),
        user_template=(
            "=== {context_prefix} ===\n{context}\n=== {context_suffix} ===\n\n"
            "Question: {question}\n\n{answer_prefix}"
        ),
    )

    LOCAL_CHAT = PromptTemplate(
        system_prompt=(
            "You are a helpful study assistant. Use only the context provided to answer questions. "
            "If you don't know based on the context, say so clearly."
        ),
        user_template=(
            "Here's the relevant content from the student's documents:\n\n{context}\n\n"
            "Based on this, please answer: {question}"
        ),
    )

    CLOUD_ADVANCED = PromptTemplate(
        system_prompt=(
            "You are Athena, an expert academic study assistant. "
            "Answer using ONLY the provided sources. Cite file name and page number. "
            "If information is insufficient, state this clearly."
        ),
        user_template=(
            "## Context from Student's Documents\n\n{context}\n\n"
            "## Student's Question\n{question}\n\n"
            "## Your Answer\n"
            "Please provide a comprehensive answer based solely on the context above:"
        ),
    )

    CLOUD_BASIC = PromptTemplate(
        system_prompt=(
            "Answer the question using only the provided context. "
            "Cite your sources. If the answer isn't in the context, say so."
        ),
        user_template="CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:",
    )

    @classmethod
    def get_template(cls, mode: PromptMode) -> PromptTemplate:
        return {
            PromptMode.LOCAL_INSTRUCT: cls.LOCAL_INSTRUCT,
            PromptMode.LOCAL_CHAT:     cls.LOCAL_CHAT,
            PromptMode.CLOUD_ADVANCED: cls.CLOUD_ADVANCED,
            PromptMode.CLOUD_BASIC:    cls.CLOUD_BASIC,
        }[mode]


class PromptBuilder:
    def __init__(
        self,
        mode: Optional[PromptMode] = None,
        template: Optional[PromptTemplate] = None,
        context_config: Optional[ContextConfig] = None,
    ):
        self.template = (
            template
            or (PromptTemplateLibrary.get_template(mode) if mode else PromptTemplateLibrary.CLOUD_ADVANCED)
        )
        self.context_config   = context_config or ContextConfig()
        self.context_assembler = ContextAssembler(self.context_config)

    def build(self, question: str, sources: List[SourceDocument], **template_vars: Any) -> str:
        if not sources:
            logger.warning("Building prompt with no sources")
            context = "No relevant information found in documents."
        else:
            context = self.context_assembler.assemble(sources)

        prompt = self.template.render(
            question=question,
            context=context,
            **template_vars
        )

        MAX_CHARS = 4000
        if len(prompt) > MAX_CHARS:
            prompt = prompt[:MAX_CHARS]

        return prompt

    @classmethod
    def for_local_llm(cls, chat_mode: bool = False) -> "PromptBuilder":
        mode   = PromptMode.LOCAL_CHAT if chat_mode else PromptMode.LOCAL_INSTRUCT
        config = ContextConfig.for_local_llm()
        return cls(mode=mode, context_config=config)

    @classmethod
    def for_cloud_llm(cls, advanced: bool = True) -> "PromptBuilder":
        mode   = PromptMode.CLOUD_ADVANCED if advanced else PromptMode.CLOUD_BASIC
        config = ContextConfig.for_cloud_llm()
        config.strategy = FormattingStrategy.NUMBERED
        return cls(mode=mode, context_config=config)
