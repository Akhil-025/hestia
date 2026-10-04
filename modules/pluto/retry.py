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
    give_up_on: Tuple[Type[Exception], ...] = (),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Callable:
    """Retry decorator with exponential backoff and jitter.

    ``give_up_on``: exception types that propagate at once, without a retry
    (for example a rate-limit error, where retrying only makes it worse).
    ``on_retry(attempt, exc, wait_seconds)``: called just before each sleep,
    so callers can count retries (backlog #144). Both default to the old
    behaviour.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except give_up_on:
                    raise
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
                    if on_retry is not None:
                        try:
                            on_retry(attempt + 1, e, wait_time)
                        except Exception:       # a broken hook must not break the retry
                            logger.exception("retry on_retry hook failed")
                    time.sleep(wait_time)
                    current_delay *= backoff
            return func(*args, **kwargs)  # fallback
        return wrapper
    return decorator