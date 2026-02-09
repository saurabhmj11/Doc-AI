"""
Error Handling Utilities

Circuit breaker pattern, retry logic, and structured exceptions for API calls.
"""

import time
import logging
from datetime import datetime, timedelta
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


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class APIError(Exception):
    """Base exception for API errors."""
    pass


class APITimeoutError(APIError):
    """API request timeout."""
    pass


class APIRateLimitError(APIError):
    """API rate limit exceeded."""
    pass


class APIAuthenticationError(APIError):
    """API authentication failed."""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Tracks API failures and temporarily disables API calls when
    the failure rate exceeds threshold.
    
    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Too many failures, reject all requests
        - HALF_OPEN: Testing recovery, allow limited requests
    """
    
    def __init__(
        self,
        failure_threshold: int = None,
        timeout: int = None,
        name: str = "default"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds before transitioning to half-open
            name: Circuit breaker identifier for logging
        """
        self.failure_threshold = failure_threshold or settings.circuit_breaker_threshold
        self.timeout = timeout or settings.circuit_breaker_timeout
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        
        logger.info(
            f"CircuitBreaker '{name}' initialized: "
            f"threshold={self.failure_threshold}, timeout={self.timeout}s"
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
            
        Raises:
            APIError: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"CircuitBreaker '{self.name}': Transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
            else:
                time_remaining = self._get_remaining_timeout()
                logger.warning(
                    f"CircuitBreaker '{self.name}': Circuit OPEN, "
                    f"retry in {time_remaining:.0f}s"
                )
                raise APIError(
                    f"Circuit breaker is OPEN. API temporarily unavailable. "
                    f"Retry in {time_remaining:.0f} seconds."
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout
    
    def _get_remaining_timeout(self) -> float:
        """Get remaining timeout in seconds."""
        if self.last_failure_time is None:
            return 0.0
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return max(0.0, self.timeout - elapsed)
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"CircuitBreaker '{self.name}': Transitioning to CLOSED (recovery successful)")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            if self.failure_count > 0:
                logger.debug(f"CircuitBreaker '{self.name}': Resetting failure count")
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        logger.warning(
            f"CircuitBreaker '{self.name}': Failure {self.failure_count}/{self.failure_threshold}"
        )
        
        if self.failure_count >= self.failure_threshold:
            logger.error(
                f"CircuitBreaker '{self.name}': Threshold exceeded, "
                f"transitioning to OPEN for {self.timeout}s"
            )
            self.state = CircuitState.OPEN
    
    def reset(self):
        """Manually reset circuit breaker."""
        logger.info(f"CircuitBreaker '{self.name}': Manual reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None


# Global circuit breaker instances
_gemini_circuit_breaker: Optional[CircuitBreaker] = None
_ollama_circuit_breaker: Optional[CircuitBreaker] = None


def get_gemini_circuit_breaker() -> CircuitBreaker:
    """Get or create Gemini API circuit breaker."""
    global _gemini_circuit_breaker
    if _gemini_circuit_breaker is None:
        _gemini_circuit_breaker = CircuitBreaker(name="gemini_api")
    return _gemini_circuit_breaker


def get_ollama_circuit_breaker() -> CircuitBreaker:
    """Get or create Ollama API circuit breaker."""
    global _ollama_circuit_breaker
    if _ollama_circuit_breaker is None:
        _ollama_circuit_breaker = CircuitBreaker(name="ollama_api")
    return _ollama_circuit_breaker


def api_retry(
    api_name: str = "API",
    max_attempts: int = None,
    base_delay: float = None
):
    """
    Decorator for retrying API calls with exponential backoff.
    
    Args:
        api_name: Name of the API for logging
        max_attempts: Maximum retry attempts (default from settings)
        base_delay: Base delay for exponential backoff (default from settings)
        
    Usage:
        @api_retry(api_name="Gemini")
        def call_gemini_api():
            # API call code
            pass
    """
    max_attempts = max_attempts or settings.retry_attempts
    base_delay = base_delay or settings.retry_base_delay
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=base_delay, min=base_delay, max=base_delay * 4),
            retry=retry_if_exception_type((APITimeoutError, ConnectionError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.DEBUG)
        )
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error with context
                if settings.log_api_failures:
                    logger.error(
                        f"{api_name} call failed: {type(e).__name__}: {str(e)}",
                        exc_info=True
                    )
                raise
        
        return wrapper
    
    return decorator


def setup_logging():
    """Configure logging for error handling."""
    import os
    from pathlib import Path
    
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # File handler for errors
    file_handler = logging.FileHandler(settings.log_file)
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger.info("Logging configured successfully")
