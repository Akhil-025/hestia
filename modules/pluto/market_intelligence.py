"""Market intelligence manager with caching, retries, and optimized data pipelines."""
import json
import time
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

    def _get_model(self) -> xgb.Booster:
        """Lazy load XGBoost model."""
        if not self._model_loaded:
            # Placeholder: load from file
            # self._model = xgb.Booster()
            # self._model.load_model("models/stock_predictor.json")
            self._model = xgb.Booster()  # dummy
            self._model_loaded = True
            logger.info("XGBoost model loaded")
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

    def generate_quant_score(self, df: pl.DataFrame) -> float:
        """
        Generate quant score using feature engineering and XGBoost.
        Uses lazy evaluation for performance.
        """
        if df.is_empty():
            logger.warning("Empty DataFrame for quant score, returning 0.5")
            return 0.5

        # Lazy feature engineering
        features = (
            df.lazy()
            .with_columns([
                (pl.col("close_price") / pl.col("open_price") - 1).alias("daily_return"),
                pl.col("close_price").rolling_mean(20).alias("sma_20"),
                pl.col("volume").rolling_mean(10).alias("volume_sma"),
                # RSI calculation (simplified)
                (pl.col("close_price").diff().clip(lower=0).rolling_mean(14)
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
            logger.warning("No features after engineering, returning 0.5")
            return 0.5

        # Convert to numpy for XGBoost
        X = features.to_numpy()
        model = self._get_model()
        # Dummy prediction (replace with actual)
        # dmatrix = xgb.DMatrix(X)
        # score = float(model.predict(dmatrix)[0])
        # For now, mock score
        score = 0.75 + 0.2 * (X.mean(axis=0).sum() / 100)  # dummy
        return float(score)

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

        # 2. ML model
        quant_score = self.generate_quant_score(market_df)
        self.metrics.record_quant_score(ticker, quant_score)

        # 3. Agent workflow
        recommendation = self.orchestrator.run_analysis(ticker, quant_score)

        result = {
            "ticker": ticker,
            "quant_score": quant_score,
            "explanation": recommendation,
            "timestamp": time.time(),
        }

        # 4. Cache in Redis (5 minutes TTL)
        self.redis.set(cache_key, json.dumps(result), ex=self.config.redis_cache_ttl)

        # 5. Optionally store in Qdrant for similarity search (not implemented here)
        # self._store_in_qdrant(ticker, result)

        logger.info("Analysis completed for %s with score %.3f", ticker, quant_score)
        return result

    # Placeholder for Qdrant storage (optional)
    def _store_in_qdrant(self, ticker: str, result: dict) -> None:
        # Not implemented, but could store vectors for semantic retrieval
        pass