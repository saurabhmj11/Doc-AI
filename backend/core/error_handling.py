"""
Error Handling Utilities

Production-safe circuit breaker, retry logic,
and structured exceptions for API calls.
"""

import time
import logging
import threading
from datetime import datetime
from typing import Callable, Optional, Any
from functools import wraps
from enum import Enum

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)

from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# =====================================================
# EXCEPTIONS
# =====================================================

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class APIError(Exception):
    pass


class APITimeoutError(APIError):
    pass


class APIRateLimitError(APIError):
    pass


class APIAuthenticationError(APIError):
    pass


# =====================================================
# CIRCUIT BREAKER (FIXED VERSION)
# =====================================================

class CircuitBreaker:

    def __init__(
        self,
        failure_threshold: int = None,
        timeout: int = None,
        name: str = "default"
    ):

        self.failure_threshold = failure_threshold or settings.circuit_breaker_threshold
        self.timeout = timeout or settings.circuit_breaker_timeout
        self.name = name

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED

        self._lock = threading.Lock()
        self._half_open_attempted = False

        logger.info(
            f"CircuitBreaker '{name}' initialized "
            f"(threshold={self.failure_threshold}, timeout={self.timeout}s)"
        )

    # =================================================

    def call(self, func: Callable, *args, **kwargs) -> Any:

        with self._lock:

            if self.state == CircuitState.OPEN:

                if self._should_attempt_reset():
                    logger.info(f"{self.name}: OPEN → HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self._half_open_attempted = False
                else:
                    raise APIError("Circuit breaker OPEN")

            if self.state == CircuitState.HALF_OPEN:

                if self._half_open_attempted:
                    raise APIError("HALF_OPEN test already running")

                self._half_open_attempted = True

        try:

            result = func(*args, **kwargs)

            self._on_success()

            return result

        except APIError:

            self._on_failure()
            raise

    # =================================================

    def _on_success(self):

        with self._lock:

            if self.state != CircuitState.CLOSED:
                logger.info(f"{self.name}: Recovery successful → CLOSED")

            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None

    # =================================================

    def _on_failure(self):

        with self._lock:

            self.failure_count += 1
            self.last_failure_time = datetime.now()

            logger.warning(
                f"{self.name}: Failure {self.failure_count}/{self.failure_threshold}"
            )

            if self.failure_count >= self.failure_threshold:
                logger.error(f"{self.name}: Circuit OPEN")
                self.state = CircuitState.OPEN

    # =================================================

    def _should_attempt_reset(self):

        if not self.last_failure_time:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout

    # =================================================

    def reset(self):

        with self._lock:

            logger.info(f"{self.name}: Manual reset")

            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None


# =====================================================
# RETRY DECORATOR (FIXED)
# =====================================================

def api_retry(
    api_name: str = "API",
    max_attempts: int = None,
    base_delay: float = None
):

    max_attempts = max_attempts or settings.retry_attempts
    base_delay = base_delay or settings.retry_base_delay

    def decorator(func: Callable):

        @wraps(func)
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=base_delay,
                min=base_delay,
                max=base_delay * 4
            ),
            retry=retry_if_exception_type(
                (APITimeoutError, ConnectionError, APIError)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.DEBUG)
        )
        def wrapper(*args, **kwargs):

            try:
                return func(*args, **kwargs)

            except Exception as e:

                if settings.log_api_failures:
                    logger.error(
                        f"{api_name} call failed: {type(e).__name__}: {str(e)}",
                        exc_info=True
                    )

                raise

        return wrapper

    return decorator


# =====================================================
# LOGGING SETUP (FIXED)
# =====================================================

def setup_logging():

    from pathlib import Path

    root_logger = logging.getLogger()

    # Prevent duplicate handlers
    if root_logger.handlers:
        return

    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))

    file_handler = logging.FileHandler(settings.log_file)
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter(log_format))

    root_logger.setLevel(getattr(logging, settings.log_level.upper()))

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger.info("Logging configured successfully")


# =====================================================
# FACTORY FUNCTIONS (MISSING)
# =====================================================

_gemini_circuit_breaker = None
_ollama_circuit_breaker = None


def get_gemini_circuit_breaker():
    global _gemini_circuit_breaker
    if _gemini_circuit_breaker is None:
        _gemini_circuit_breaker = CircuitBreaker(name="Gemini")
    return _gemini_circuit_breaker


def get_ollama_circuit_breaker():
    global _ollama_circuit_breaker
    if _ollama_circuit_breaker is None:
        _ollama_circuit_breaker = CircuitBreaker(name="Ollama")
    return _ollama_circuit_breaker
