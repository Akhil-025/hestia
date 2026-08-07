"""

modules/pluto/metrics.py
Prometheus metrics collection.

"""



import time
from functools import wraps
from typing import Callable, Any, Optional

# Optional import; if prometheus_client not installed, metrics are no-ops
try:
    from prometheus_client import Counter, Histogram, Gauge, Registry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Dummy classes
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
    class Registry: pass

from .logging_config import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    def __init__(self, registry: Optional[Registry] = None):
        self.registry = registry or Registry()
        self.enabled = PROMETHEUS_AVAILABLE

        if self.enabled:
            self.request_count = Counter(
                "pluto_requests_total",
                "Total number of requests",
                ["intent", "status"],
                registry=self.registry,
            )
            self.request_duration = Histogram(
                "pluto_request_duration_seconds",
                "Request duration in seconds",
                ["intent"],
                registry=self.registry,
            )
            self.expense_count = Counter(
                "pluto_expenses_logged_total",
                "Total expenses logged",
                registry=self.registry,
            )
            self.investment_value = Gauge(
                "pluto_investment_value",
                "Total investment value",
                registry=self.registry,
            )
            self.quant_score_gauge = Gauge(
                "pluto_quant_score",
                "Latest quant score per ticker",
                ["ticker"],
                registry=self.registry,
            )
        else:
            # No-op attributes
            self.request_count = None
            self.request_duration = None
            self.expense_count = None
            self.investment_value = None
            self.quant_score_gauge = None

    def track_request(self, intent: str) -> Callable:
        """Decorator to track request metrics."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                if not self.enabled:
                    return func(*args, **kwargs)
                start = time.time()
                status = "success"
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception:
                    status = "error"
                    raise
                finally:
                    self.request_count.labels(intent=intent, status=status).inc()
                    self.request_duration.labels(intent=intent).observe(
                        time.time() - start
                    )
            return wrapper
        return decorator

    def record_expense(self) -> None:
        if self.enabled:
            self.expense_count.inc()

    def record_investment_value(self, value: float) -> None:
        if self.enabled:
            self.investment_value.set(value)

    def record_quant_score(self, ticker: str, score: float) -> None:
        if self.enabled:
            self.quant_score_gauge.labels(ticker=ticker).set(score)


def track_metrics(intent: str) -> Callable:
    """
    Decorator factory for tracking request metrics on *instance methods*.

    `MetricsCollector.track_request` is itself an instance method, so it
    can't be used as a class-body decorator (there is no `MetricsCollector`
    instance yet when the class defining `handle()` is being built — e.g.
    `@MetricsCollector.track_request("handle")` above a method definition
    would bind the string "handle" to `self` and blow up at class-creation
    time). This module-level decorator instead defers the metrics lookup
    to call time, reading `self.metrics` off the wrapped method's first
    argument, so it can be applied directly to a module's `handle()` method.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            metrics: Optional["MetricsCollector"] = getattr(self, "metrics", None)
            if metrics is None or not metrics.enabled:
                return func(self, *args, **kwargs)
            start = time.time()
            status = "success"
            try:
                return func(self, *args, **kwargs)
            except Exception:
                status = "error"
                raise
            finally:
                metrics.request_count.labels(intent=intent, status=status).inc()
                metrics.request_duration.labels(intent=intent).observe(
                    time.time() - start
                )
        return wrapper
    return decorator