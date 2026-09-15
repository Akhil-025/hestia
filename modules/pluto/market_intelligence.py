"""Market intelligence manager with caching, retries, and optimized data pipelines."""
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import polars as pl
import xgboost as xgb

from .config import PlutoConfig
from .db import DatabaseManager
from .agents import LangGraphOrchestrator
from .retry import retry
from .logging_config import get_logger
from .metrics import MetricsCollector
from .llm_client import LLMClient

logger = get_logger(__name__)


class MarketIntelligenceManager:
    def __init__(
        self,
        config: PlutoConfig,
        db_manager: Optional[DatabaseManager] = None,
        kimi_client: Optional[Any] = None,
        llm_client: Optional[LLMClient] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.config = config
        self.db_manager = db_manager or DatabaseManager(config)
        self.metrics = metrics or MetricsCollector()
        self.redis = self.db_manager.get_redis_client()
        self.vdb = self.db_manager.get_qdrant_client()

        # LLM client for agents
        self.llm_client = llm_client or LLMClient(config, fallback_llm=kimi_client)

        # Initialize Kimi-powered LangGraph Orchestrator
        self.orchestrator = LangGraphOrchestrator(self.llm_client)

        # Load pre-trained model (lazy loading)
        self._model: Optional[xgb.Booster] = None
        self._model_loaded = False

        # Sentence-transformer embedder for Qdrant storage/search — lazily
        # loaded on first actual use (see _get_embedder) rather than here,
        # so constructing this manager never requires a model download.
        self._embedder: Optional[Any] = None

    def _get_model(self) -> Optional[xgb.Booster]:
        """Lazy-load a trained XGBoost model from disk, if one is
        configured and actually present.

        Returns ``None`` — never a freshly-constructed, untrained
        ``xgb.Booster()`` — when no trained model is available.
        ``generate_quant_score`` treats ``None`` as a signal to fall back
        to the documented technical-indicator heuristic rather than
        quietly calling ``.predict()`` on a model that was never fit to
        any data.
        """
        if not self._model_loaded:
            self._model_loaded = True
            model_path = self.config.stock_model_path
            if model_path and Path(model_path).exists():
                try:
                    model = xgb.Booster()
                    model.load_model(str(model_path))
                    self._model = model
                    logger.info("Loaded trained XGBoost model from %s", model_path)
                except Exception as e:
                    logger.warning(
                        "Failed to load XGBoost model from %s (%s); "
                        "generate_quant_score() will use the technical-"
                        "indicator heuristic instead.", model_path, e,
                    )
                    self._model = None
            else:
                logger.info(
                    "No trained stock_predictor model configured (set "
                    "PlutoConfig.stock_model_path); generate_quant_score() "
                    "will use the technical-indicator heuristic instead of "
                    "an ML prediction."
                )
                self._model = None
        return self._model

    @retry(max_retries=2, exceptions=(Exception,), delay=0.5)
    def fetch_market_data(self, ticker: str, days_back: int = 30) -> pl.DataFrame:
        """Fetch market data using parameterized query."""
        # psycopg2 uses "%s" placeholders (not asyncpg-style "$1"/"$2"), and a
        # bind parameter can't be substituted *inside* a quoted string literal
        # like INTERVAL '$2 days' — the driver would send it as a literal
        # string "'$2 days'" rather than interpolating the number. Casting a
        # parameterized text value to ::interval is the safe equivalent.
        query = """
            SELECT
                time_bucket('1 day', time) AS bucket,
                FIRST(open_price) AS open_price,
                LAST(close_price) AS close_price,
                MAX(high) AS high,
                MIN(low) AS low,
                SUM(volume) AS volume
            FROM stock_prices
            WHERE symbol = %s
                AND time > NOW() - (%s || ' days')::interval
            GROUP BY bucket
            ORDER BY bucket DESC
            LIMIT 1000
        """
        with self.db_manager.get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (ticker, days_back))
                data = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
            df = pl.DataFrame(data, schema=columns)
        return df

    def generate_quant_score(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Score an asset on a 0-1 scale from its recent price/volume history.

        Uses a trained XGBoost model when one is configured (see
        ``PlutoConfig.stock_model_path``). Otherwise falls back to
        ``_technical_heuristic_score``: a deterministic composite of
        real, computed technical indicators (momentum, trend vs SMA-20,
        RSI mean-reversion). That heuristic is the same kind of read a
        chartist would eyeball from a price chart — it is NOT a trained
        predictive model, and callers must not present it as one. That's
        why the return value carries an explicit "method" field instead
        of just a bare float: downstream code (analyze_asset, and
        anything that renders this to the user) can and should say
        which kind of score they're looking at.

        Returns:
            {
                "score": float in [0, 1],
                "method": "ml_model" | "technical_heuristic" | "no_data",
                "reliability": float in [0, 1] — reflects how much clean
                    history went into the score (row count after feature
                    engineering), not whether the reading is bullish or
                    bearish.
            }
        """
        if df.is_empty():
            logger.warning("Empty DataFrame for quant score; no data to score from.")
            return {"score": 0.5, "method": "no_data", "reliability": 0.0}

        # Lazy feature engineering
        features = (
            df.lazy()
            .with_columns([
                (pl.col("close_price") / pl.col("open_price") - 1).alias("daily_return"),
                pl.col("close_price").rolling_mean(20).alias("sma_20"),
                pl.col("volume").rolling_mean(10).alias("volume_sma"),
                # RSI calculation (simplified)
                # `lower=` was renamed to `lower_bound=` in Polars >=1.0;
                # requirements.txt pins `polars>=0.19.12` with no upper
                # bound, so a fresh install pulls modern Polars and the
                # old kwarg name raises TypeError at call time, not import
                # time — this was failing silently until generate_quant_score
                # actually ran. See the vectorbt/plotly pin note above for
                # the same class of "confirmed broken empirically" issue.
                (pl.col("close_price").diff().clip(lower_bound=0).rolling_mean(14)
                 / pl.col("close_price").diff().abs().rolling_mean(14)
                 * 100).alias("rsi"),
            ])
            .select([
                "daily_return",
                "sma_20",
                "volume_sma",
                "rsi",
                "volume",
                "close_price",
            ])
            .drop_nulls()
            .collect()
        )

        if features.is_empty():
            logger.warning(
                "No features survived feature engineering (insufficient "
                "price history); no data to score from."
            )
            return {"score": 0.5, "method": "no_data", "reliability": 0.0}

        # Enough rows for a stable 20-day SMA / 14-day RSI to have formed
        # more than a handful of times; below this the heuristic (and any
        # model) is working off too little history to mean much.
        reliability = min(1.0, features.height / 30)

        model = self._get_model()
        if model is not None:
            X = features.to_numpy()
            dmatrix = xgb.DMatrix(X)
            score = float(model.predict(dmatrix)[0])
            return {"score": score, "method": "ml_model", "reliability": reliability}

        return self._technical_heuristic_score(features, reliability)

    def _technical_heuristic_score(
        self, features: pl.DataFrame, reliability: float
    ) -> Dict[str, Any]:
        """Deterministic technical-indicator composite, used whenever no
        trained model is available (the common case out of the box).

        Every input is a real, computed indicator over the fetched price
        history — momentum (recent daily returns), trend (price vs its
        own 20-day SMA), and an RSI mean-reversion read — combined into a
        single 0-1 score where 0.5 is neutral. This is a plain technical
        heuristic, not a statistical or ML prediction, and is labelled as
        such in the returned "method" field on purpose.
        """
        latest = features.row(-1, named=True)

        # Momentum: mean of the last 5 daily returns, scaled so a ~5%
        # average daily move saturates the component at +/-1.
        recent_returns = features["daily_return"].tail(5)
        momentum = recent_returns.mean() if recent_returns.len() else 0.0
        momentum_component = max(-1.0, min(1.0, (momentum or 0.0) * 20))

        # Trend: how far the latest close sits from its 20-day SMA.
        close = latest["close_price"]
        sma = latest["sma_20"]
        trend_component = 0.0
        if sma:
            trend_component = max(-1.0, min(1.0, (close - sma) / sma * 10))

        # RSI mean-reversion: oversold (<30) tilts bullish, overbought
        # (>70) tilts bearish, 40-60 reads as roughly neutral.
        rsi = latest["rsi"]
        rsi_component = 0.0
        if rsi is not None:
            rsi_component = max(-1.0, min(1.0, (50 - rsi) / 30))

        composite = (
            0.40 * momentum_component
            + 0.35 * trend_component
            + 0.25 * rsi_component
        )
        score = max(0.0, min(1.0, 0.5 + 0.5 * composite))
        return {"score": score, "method": "technical_heuristic", "reliability": reliability}

    def analyze_asset(self, ticker: str, force_refresh: bool = False) -> dict:
        """
        Main pipeline: check cache, then fetch data, ML, agent, and cache result.
        """
        ticker = ticker.upper().strip()
        cache_key = f"analysis:{ticker}"

        if not force_refresh:
            cached = self.redis.get(cache_key)
            if cached:
                logger.info("Returning cached analysis for %s", ticker)
                return json.loads(cached)

        # 1. Fetch market data
        market_df = self.fetch_market_data(ticker)

        # 2. Quant score — ML model if one's configured, otherwise the
        # technical-indicator heuristic. See generate_quant_score's
        # docstring for why this is a dict rather than a bare float.
        quant = self.generate_quant_score(market_df)
        quant_score = quant["score"]
        self.metrics.record_quant_score(ticker, quant_score)

        # 3. Agent workflow
        recommendation = self.orchestrator.run_analysis(ticker, quant_score)
        if quant["method"] == "technical_heuristic":
            recommendation += (
                "\n\n(Note: this reading comes from a technical-indicator "
                "heuristic, not a trained ML model — no stock_predictor "
                "model is currently configured.)"
            )
        elif quant["method"] == "no_data":
            recommendation += (
                "\n\n(Note: there wasn't enough price history to score "
                "this — treat the number above as a placeholder.)"
            )

        result = {
            "ticker": ticker,
            "quant_score": quant_score,
            "score_method": quant["method"],
            "reliability": quant["reliability"],
            "explanation": recommendation,
            "timestamp": time.time(),
        }

        # 4. Cache in Redis (5 minutes TTL)
        self.redis.set(cache_key, json.dumps(result), ex=self.config.redis_cache_ttl)

        # 5. Store in Qdrant for future similarity search (best-effort —
        # see _store_in_qdrant; a Qdrant/embedding failure here never fails
        # the analysis itself).
        self._store_in_qdrant(ticker, result)

        logger.info("Analysis completed for %s with score %.3f", ticker, quant_score)
        return result

    def _get_embedder(self):
        """Lazily load the sentence-transformer used to embed analysis text.
        Loaded on first real use rather than in __init__, so constructing a
        MarketIntelligenceManager (including in tests) never triggers a
        model download by itself."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.config.embedding_model)
            logger.info("Embedding model '%s' loaded", self.config.embedding_model)
        return self._embedder

    def _ensure_qdrant_collection(self, vector_size: int) -> None:
        """Create the configured Qdrant collection if it doesn't exist yet."""
        from qdrant_client.models import Distance, VectorParams

        collection = self.config.qdrant_collection
        try:
            # collection_exists() is the modern qdrant-client API; older
            # clients (this repo pins qdrant-client>=1.6.2, which predates
            # it) don't have it, so fall back to a get_collection() probe.
            exists = self.vdb.collection_exists(collection)
        except AttributeError:
            try:
                self.vdb.get_collection(collection)
                exists = True
            except Exception:
                exists = False
        if not exists:
            self.vdb.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s' (size=%d)", collection, vector_size)

    def _store_in_qdrant(self, ticker: str, result: dict) -> None:
        """Embed the analysis explanation and upsert it into Qdrant so past
        analyses can later be retrieved by semantic similarity (see
        find_similar_analyses). Best-effort: any failure here (Qdrant
        unreachable, embedding model unavailable, etc.) is logged and
        swallowed — this is optional enrichment, not the primary analysis
        result, so it should never take down analyze_asset()."""
        try:
            import uuid
            from qdrant_client.models import PointStruct

            text = f"{ticker}: {result.get('explanation', '')}"
            embedder = self._get_embedder()
            vector = embedder.encode(text, normalize_embeddings=True).tolist()

            self._ensure_qdrant_collection(len(vector))

            # Deterministic id from ticker+timestamp so re-storing the same
            # analysis (e.g. a retried request) upserts rather than
            # duplicating the point.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ticker}:{result.get('timestamp')}"))
            self.vdb.upsert(
                collection_name=self.config.qdrant_collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "ticker": ticker,
                            "quant_score": result.get("quant_score"),
                            # Carried through so anything reading past
                            # analyses back out of Qdrant (find_similar_
                            # analyses, or a future dashboard) can tell a
                            # real ML score from the technical-indicator
                            # heuristic without re-deriving it — same
                            # honesty contract as the live analyze_asset()
                            # response, not just the in-the-moment one.
                            "score_method": result.get("score_method"),
                            "reliability": result.get("reliability"),
                            "explanation": result.get("explanation"),
                            "timestamp": result.get("timestamp"),
                        },
                    )
                ],
            )
            logger.debug("Stored analysis for %s in Qdrant collection '%s'", ticker, self.config.qdrant_collection)
        except Exception as e:
            logger.warning("Qdrant storage failed for %s (non-fatal): %s", ticker, e)

    def find_similar_analyses(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search over past analyses stored via _store_in_qdrant.
        E.g. find_similar_analyses("overbought tech stock with slowing
        momentum") surfaces past tickers whose explanation reads similarly,
        regardless of exact wording. Returns [] (with a warning logged)
        rather than raising if Qdrant or the embedding model isn't
        reachable, matching _store_in_qdrant's best-effort behaviour."""
        try:
            embedder = self._get_embedder()
            vector = embedder.encode(query, normalize_embeddings=True).tolist()
            hits = self.vdb.search(
                collection_name=self.config.qdrant_collection,
                query_vector=vector,
                limit=top_k,
            )
            return [{"score": hit.score, **hit.payload} for hit in hits]
        except Exception as e:
            logger.warning("Qdrant similarity search failed (non-fatal): %s", e)
            return []