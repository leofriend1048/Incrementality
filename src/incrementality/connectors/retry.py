"""Retry logic with exponential backoff for API connectors.

Provides a resilient HTTP request wrapper that handles:
- HTTP 429 (rate limit) with Retry-After header support
- HTTP 5xx (server errors)
- Connection errors and timeouts
- Exponential backoff with jitter to avoid thundering herd
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# HTTP status codes that should trigger a retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Default configuration
DEFAULT_MAX_RETRIES = 4
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds


class RetryableRequestError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, method: str, url: str, attempts: int, last_error: Exception):
        self.method = method
        self.url = url
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{method} {url} failed after {attempts} attempts: {last_error}"
        )


def _compute_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    retry_after: float | None = None,
) -> float:
    """Compute backoff delay with jitter.

    Uses exponential backoff: base_delay * 2^attempt + random jitter.
    Respects Retry-After header if present.
    """
    if retry_after is not None and retry_after > 0:
        # Respect server's Retry-After, but add small jitter
        return min(retry_after + random.uniform(0, 1), max_delay)
    delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
    return min(delay, max_delay)


def _parse_retry_after(response: requests.Response) -> float | None:
    """Extract Retry-After header value in seconds."""
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        return float(header)
    except (ValueError, TypeError):
        return None


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    **kwargs: Any,
) -> requests.Response:
    """Execute an HTTP request with retry and exponential backoff.

    Args:
        session: The requests.Session to use.
        method: HTTP method ("GET" or "POST").
        url: The request URL.
        max_retries: Maximum number of retry attempts (total attempts = max_retries + 1).
        base_delay: Base delay in seconds for exponential backoff.
        max_delay: Maximum delay cap in seconds.
        **kwargs: Passed through to session.request() (params, json, timeout, etc.).

    Returns:
        The successful requests.Response.

    Raises:
        RetryableRequestError: If all retry attempts are exhausted.
        requests.HTTPError: For non-retryable HTTP errors (4xx except 429).
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            resp = session.request(method, url, **kwargs)

            # Non-retryable client errors (400, 401, 403, 404, etc.)
            if 400 <= resp.status_code < 500 and resp.status_code not in _RETRYABLE_STATUS_CODES:
                resp.raise_for_status()

            # Retryable errors (429, 5xx)
            if resp.status_code in _RETRYABLE_STATUS_CODES:
                retry_after = _parse_retry_after(resp)
                delay = _compute_delay(attempt, base_delay, max_delay, retry_after)
                last_error = requests.HTTPError(
                    f"HTTP {resp.status_code}", response=resp
                )

                if attempt < max_retries:
                    logger.warning(
                        f"Retryable HTTP {resp.status_code} from {method} {url} "
                        f"(attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise RetryableRequestError(method, url, max_retries + 1, last_error)

            # Success
            return resp

        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            delay = _compute_delay(attempt, base_delay, max_delay)

            if attempt < max_retries:
                logger.warning(
                    f"{type(exc).__name__} on {method} {url} "
                    f"(attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                continue
            else:
                raise RetryableRequestError(method, url, max_retries + 1, last_error) from exc

    # Should not reach here, but just in case
    raise RetryableRequestError(method, url, max_retries + 1, last_error or RuntimeError("unknown"))
