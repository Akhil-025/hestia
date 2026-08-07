"""

modules/pluto/retry.py
Retry decorator with exponential backoff and jitter.

"""


import time
import random
import functools
from typing import Type, Tuple, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


def retry(
    max_retries: int = 3,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    delay: float = 1.0,
    backoff: float = 2.0,
    jitter: float = 0.5,
) -> Callable:
    """Retry decorator with exponential backoff and jitter."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            f"All retries failed for {func.__name__}: {e}",
                            extra={"function": func.__name__, "attempt": attempt + 1}
                        )
                        raise
                    wait_time = current_delay * (1 + jitter * (random.random() * 2 - 1))
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {wait_time:.2f}s: {e}",
                        extra={"function": func.__name__, "attempt": attempt + 1}
                    )
                    time.sleep(wait_time)
                    current_delay *= backoff
            return func(*args, **kwargs)  # fallback
        return wrapper
    return decorator