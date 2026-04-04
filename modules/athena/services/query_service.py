"""
modules/athena/services/query_service.py

Full query pipeline — fixed for Hestia (relative imports).
"""
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from modules.athena.models import QueryResult, SearchResults, SourceDocument
from modules.athena.services.prompt_builder import PromptBuilder
from modules.athena.services.context_assembler import ContextAssembler
from modules.athena.utils.llm_cache import question_hash, load_cached_answer, save_cached_answer
from modules.athena.exceptions import QueryError, LLMError, RAGError
from modules.athena.config import get_config

logger = logging.getLogger(__name__)


class AnswerQuality(Enum):
    HIGH         = "high"
    MEDIUM       = "medium"
    LOW          = "low"
    INSUFFICIENT = "insufficient"


@dataclass
class QueryMetrics:
    search_time_ms:     float                  = 0
    generation_time_ms: float                  = 0
    total_time_ms:      float                  = 0
    sources_found:      int                    = 0
    sources_used:       int                    = 0
    cache_hit:          bool                   = False
    fallback_triggered: bool                   = False
    answer_quality:     Optional[AnswerQuality] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_time_ms":     self.search_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "total_time_ms":      self.total_time_ms,
            "sources_found":      self.sources_found,
            "sources_used":       self.sources_used,
            "cache_hit":          self.cache_hit,
            "fallback_triggered": self.fallback_triggered,
            "answer_quality":     self.answer_quality.value if self.answer_quality else None,
        }


@dataclass
class QueryContext:
    question:       str
    use_cloud:      bool
    subject_filter: Optional[str]    = None
    module_filter:  Optional[str]    = None
    n_results:      Optional[int]    = None
    force_refresh:  bool             = False
    include_metrics: bool            = True
    sources:        List[SourceDocument] = field(default_factory=list)
    search_results: Optional[SearchResults] = None
    cache_key:      Optional[str]    = None
    metrics:        QueryMetrics     = field(default_factory=QueryMetrics)


class AnswerQualityAssessor:
    INSUFFICIENT_PATTERNS = [
        "i don't", "i cannot", "i'm not sure",
        "not in the context", "no information", "cannot find",
        "not mentioned", "does not cover", "insufficient information",
    ]
    MIN_WORD_COUNT = 15

    @classmethod
    def assess(cls, answer: str) -> AnswerQuality:
        answer_lower = answer.lower().strip()
        word_count   = len(answer.split())

        if any(p in answer_lower for p in cls.INSUFFICIENT_PATTERNS):
            return AnswerQuality.INSUFFICIENT
        if word_count < cls.MIN_WORD_COUNT:
            return AnswerQuality.LOW
        if word_count > 100 and "\n" in answer:
            return AnswerQuality.HIGH
        if word_count > 50:
            return AnswerQuality.MEDIUM
        return AnswerQuality.LOW


class CacheManager:
    CACHE_VERSION = "v2"

    @classmethod
    def generate_key(cls, question: str, sources: List[SourceDocument], use_cloud: bool) -> str:
        context_ids = [
            f"{s.file_name}:{s.page_number}:{s.chunk_number or 0}"
            for s in sources
        ]
        mode     = "cloud" if use_cloud else "local"
        base_key = f"{cls.CACHE_VERSION}:{mode}:{question}"
        return question_hash(base_key, context_ids)

    @classmethod
    def load(cls, cache_key: str) -> Optional[Dict[str, Any]]:
        try:
            cached = load_cached_answer(cache_key)
            if cached and cached.get("version") == cls.CACHE_VERSION:
                return cached
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
        return None

    @classmethod
    def save(cls, cache_key, answer, sources, use_cloud, quality) -> bool:
        try:
            payload = {
                "version":   cls.CACHE_VERSION,
                "answer":    answer,
                "sources":   [s.to_dict() for s in sources],
                "mode":      "cloud" if use_cloud else "local",
                "quality":   quality.value,
                "timestamp": time.time(),
            }
            save_cached_answer(cache_key, payload)
            return True
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
            return False

    @classmethod
    def reconstruct_sources(cls, cached_data: Dict[str, Any], fallback: List[SourceDocument]) -> List[SourceDocument]:
        cached_sources = cached_data.get("sources", [])
        if not cached_sources or not isinstance(cached_sources[0], dict):
            return fallback
        try:
            return [
                SourceDocument(
                    text         = s.get("text", ""),
                    file_name    = s.get("file_name", "unknown"),
                    file_path    = s.get("file_path", ""),
                    page_number  = s.get("page", 0),
                    subject      = s.get("subject"),
                    module       = s.get("module"),
                    chunk_number = s.get("chunk_number"),
                    score        = s.get("score", 0.0),
                )
                for s in cached_sources
            ]
        except Exception as e:
            logger.warning(f"Failed to reconstruct cached sources: {e}")
            return fallback


class QueryService:
    """
    Orchestrates the full Athena pipeline:
    search → cache check → LLM generate → quality assess → cache save
    """

    def __init__(self, rag, ai_integration):
        """
        Args:
            rag:            MergedLocalRAG instance
            ai_integration: HestiaLLMAdapter (wraps Hestia's LLM)
        """
        self.rag              = rag
        self.ai               = ai_integration
        self.config           = get_config()
        self.cache            = CacheManager()
        self.quality_assessor = AnswerQualityAssessor()

    def execute(
        self,
        question: str,
        use_cloud: bool = False,
        subject_filter: Optional[str] = None,
        module_filter: Optional[str] = None,
        n_results: Optional[int] = None,
        force_refresh: bool = False,
    ) -> QueryResult:
        start_time = time.time()

        ctx = QueryContext(
            question       = question,
            use_cloud      = use_cloud,
            subject_filter = subject_filter,
            module_filter  = module_filter,
            n_results      = n_results or self.config.default_search_results,
            force_refresh  = force_refresh,
        )

        try:
            self._execute_search(ctx)

            if not force_refresh:
                cached_result = self._try_cache(ctx)
                if cached_result:
                    ctx.metrics.total_time_ms = (time.time() - start_time) * 1000
                    return cached_result

            answer  = self._generate_answer(ctx)
            quality = self.quality_assessor.assess(answer)
            ctx.metrics.answer_quality = quality

            if quality in (AnswerQuality.LOW, AnswerQuality.INSUFFICIENT):
                answer = self._try_fallback(ctx, answer, quality)

            if ctx.cache_key:
                self.cache.save(ctx.cache_key, answer, ctx.sources, ctx.use_cloud, ctx.metrics.answer_quality)

            ctx.metrics.total_time_ms = (time.time() - start_time) * 1000
            return self._build_result(ctx, answer)

        except (RAGError, LLMError):
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in query execution: {e}")
            raise QueryError(f"Query execution failed: {e}") from e

    def _execute_search(self, ctx: QueryContext) -> None:
        logger.info(f"Searching: {ctx.question[:100]}")
        search_start = time.time()
        try:
            rag_response      = self.rag.search(
                ctx.question,
                n_results      = ctx.n_results,
                subject_filter = ctx.subject_filter,
                module_filter  = ctx.module_filter,
            )
            ctx.search_results            = SearchResults.from_rag_response(rag_response)
            ctx.sources                   = ctx.search_results.to_source_documents()
            ctx.metrics.sources_found     = len(ctx.sources)
            ctx.metrics.search_time_ms    = (time.time() - search_start) * 1000
            logger.info(f"Found {len(ctx.sources)} sources in {ctx.metrics.search_time_ms:.0f}ms")
        except Exception as e:
            logger.exception(f"Search failed: {e}")
            raise RAGError(f"Failed to search documents: {e}") from e

    def _try_cache(self, ctx: QueryContext) -> Optional[QueryResult]:
        ctx.cache_key = self.cache.generate_key(ctx.question, ctx.sources, ctx.use_cloud)
        cached = self.cache.load(ctx.cache_key)
        if not cached:
            return None

        ctx.metrics.cache_hit = True
        sources = self.cache.reconstruct_sources(cached, ctx.sources)
        return QueryResult(
            question      = ctx.question,
            answer        = cached.get("answer", ""),
            sources       = sources,
            cached        = True,
            mode          = cached.get("mode", "unknown"),
            total_sources = len(sources),
            metrics       = ctx.metrics.to_dict() if ctx.include_metrics else None,
        )

    def _generate_answer(self, ctx: QueryContext) -> str:
        if not ctx.sources:
            return "No relevant information found in your documents."

        logger.info(f"Generating answer (cloud={ctx.use_cloud})")
        gen_start   = time.time()
        max_sources = (
            self.config.max_chunks_cloud if ctx.use_cloud
            else self.config.max_chunks_local
        )
        sources_to_use       = ctx.sources[:max_sources]
        ctx.metrics.sources_used = len(sources_to_use)

        # Build prompt
        builder = (
            PromptBuilder.for_cloud_llm() if ctx.use_cloud
            else PromptBuilder.for_local_llm()
        )
        prompt = builder.build(ctx.question, sources_to_use)

        # Generate via adapter
        result = self.ai.generate(prompt)
        if result.get("error"):
            raise LLMError(result["error"])

        answer = result.get("text", "")
        ctx.metrics.generation_time_ms = (time.time() - gen_start) * 1000
        logger.info(f"Generated in {ctx.metrics.generation_time_ms:.0f}ms ({len(answer)} chars)")
        return answer

    def _try_fallback(self, ctx, current_answer, current_quality) -> str:
        # Only fallback from local → cloud; adapter doesn't distinguish, so skip
        return current_answer

    def _build_result(self, ctx: QueryContext, answer: str) -> QueryResult:
        return QueryResult(
            question      = ctx.question,
            answer        = answer,
            sources       = ctx.sources,
            cached        = False,
            mode          = "cloud" if ctx.use_cloud else "local",
            total_sources = len(ctx.sources),
            metrics       = ctx.metrics.to_dict() if ctx.include_metrics else None,
        )

    # backward compat
    def execute_query(self, *args, **kwargs) -> QueryResult:
        return self.execute(*args, **kwargs)
