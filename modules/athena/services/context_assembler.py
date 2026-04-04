"""
modules/athena/services/context_assembler.py

Context assembly — fixed for Hestia (relative imports).
"""
from typing import List, Optional, Protocol
from dataclasses import dataclass
from enum import Enum
import logging

from modules.athena.models import SourceDocument, SearchResults

logger = logging.getLogger(__name__)


class FormattingStrategy(Enum):
    DETAILED = "detailed"
    COMPACT  = "compact"
    PLAIN    = "plain"
    NUMBERED = "numbered"


@dataclass
class ContextConfig:
    max_sources: int             = 5
    max_chars_per_source: int    = 2000
    include_headers: bool        = True
    strategy: FormattingStrategy = FormattingStrategy.DETAILED
    separator: str               = "\n\n"
    truncation_suffix: str       = "..."

    @classmethod
    def for_local_llm(cls) -> "ContextConfig":
        return cls(
            max_sources=3,
            max_chars_per_source=800,
            strategy=FormattingStrategy.COMPACT
        )

    @classmethod
    def for_cloud_llm(cls) -> "ContextConfig":
        return cls(max_sources=3, max_chars_per_source=2000, strategy=FormattingStrategy.NUMBERED)


class DetailedFormatter:
    @staticmethod
    def format_source(source: SourceDocument, index: int) -> str:
        subject_module = (
            f"{source.subject} → {source.module}" if source.subject and source.module
            else source.subject or "General"
        )
        header = (
            f"--- Excerpt {index}: {source.file_name} | {subject_module} "
            f"(Page {source.page_number}"
            f"{f', Chunk {source.chunk_number}' if source.chunk_number else ''}"
            f"{f', Relevance: {source.score:.2f}' if source.score else ''}) ---"
        )
        return f"{header}\n{source.text}"


class CompactFormatter:
    @staticmethod
    def format_source(source: SourceDocument, index: int) -> str:
        return f"[{index}] {source.file_name} (p.{source.page_number})\n{source.text}"


class PlainFormatter:
    @staticmethod
    def format_source(source: SourceDocument, index: int) -> str:
        return source.text


class NumberedFormatter:
    @staticmethod
    def format_source(source: SourceDocument, index: int) -> str:
        citation = f"[Source {index}: {source.file_name}, Page {source.page_number}]"
        return f"{citation}\n{source.text}"


class ContextAssembler:
    _FORMATTERS = {
        FormattingStrategy.DETAILED: DetailedFormatter(),
        FormattingStrategy.COMPACT:  CompactFormatter(),
        FormattingStrategy.PLAIN:    PlainFormatter(),
        FormattingStrategy.NUMBERED: NumberedFormatter(),
    }

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config     = config or ContextConfig()
        self._formatter = self._FORMATTERS[self.config.strategy]

    def assemble(
        self,
        sources: List[SourceDocument],
        max_sources: Optional[int] = None,
        max_chars_per_source: Optional[int] = None,
    ) -> str:
        if not sources:
            return "No relevant context available."

        max_sources = max_sources or self.config.max_sources
        max_chars   = max_chars_per_source or self.config.max_chars_per_source

        limited   = sources[:max_sources]
        truncated = self._truncate_sources(limited, max_chars)

        formatted_parts = []
        for i, source in enumerate(truncated, start=1):
            try:
                formatted_parts.append(self._formatter.format_source(source, i))
            except Exception as e:
                logger.error(f"Failed to format source {i}: {e}", exc_info=True)
                formatted_parts.append(f"[Source {i}: Error formatting - {source.file_name}]")

        return self.config.separator.join(formatted_parts)

    def _truncate_sources(self, sources: List[SourceDocument], max_chars: int) -> List[SourceDocument]:
        truncated = []
        for source in sources:
            if len(source.text) <= max_chars:
                truncated.append(source)
            else:
                truncated_text = self._smart_truncate(source.text, max_chars)
                truncated.append(SourceDocument(
                    text=truncated_text,
                    file_name=source.file_name,
                    file_path=source.file_path,
                    page_number=source.page_number,
                    subject=source.subject,
                    module=source.module,
                    chunk_number=source.chunk_number,
                    score=source.score,
                ))
        return truncated

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        truncate_point = max_chars - len(self.config.truncation_suffix)
        last_sentence = max(
            text.rfind(". ", 0, truncate_point),
            text.rfind("! ", 0, truncate_point),
            text.rfind("? ", 0, truncate_point),
        )
        if last_sentence > max_chars * 0.7:
            return text[:last_sentence + 1].rstrip() + self.config.truncation_suffix
        last_space = text.rfind(" ", 0, truncate_point)
        if last_space > 0:
            return text[:last_space].rstrip() + self.config.truncation_suffix
        return text[:truncate_point].rstrip() + self.config.truncation_suffix

    @classmethod
    def from_search_results(cls, search_results: SearchResults, config: Optional[ContextConfig] = None) -> str:
        assembler = cls(config)
        return assembler.assemble(search_results.to_source_documents())


# Backward compat
def assemble_context(sources: List[SourceDocument], include_headers: bool = True) -> str:
    config = ContextConfig(
        strategy=FormattingStrategy.DETAILED if include_headers else FormattingStrategy.PLAIN
    )
    return ContextAssembler(config).assemble(sources)
