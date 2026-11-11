"""The replaceable provider interface.

A provider is a text-in/text-out function and nothing more: it never retrieves, never validates
and never decides what the answer means. Everything it returns is checked downstream.
"""

import logging
import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import httpx

from app.services.generation.errors import ProviderTimeout, ProviderUnavailable
from app.services.generation.types import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    ProviderInfo,
)

logger = logging.getLogger(__name__)


def send_with_retries(
    provider_name: str,
    send: Callable[[float], httpx.Response],
    *,
    retries: int,
    timeout_seconds: float,
) -> tuple[httpx.Response, int]:
    """Send one provider request with bounded retries and one overall time budget."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    attempt = 0

    while attempt <= retries:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderTimeout(f"{provider_name} did not answer within the time budget")

        attempt += 1
        try:
            response = send(remaining)
        except httpx.TimeoutException as exc:
            last_error = exc
            logger.warning(
                "%s request timed out (attempt %d/%d)", provider_name, attempt, retries + 1
            )
        except httpx.RequestError as exc:
            last_error = exc
            logger.warning(
                "%s transport error (attempt %d/%d): %s",
                provider_name,
                attempt,
                retries + 1,
                type(exc).__name__,
            )
        else:
            if not 500 <= response.status_code < 600:
                return response, attempt
            last_error = ProviderUnavailable(
                f"{provider_name} server returned HTTP {response.status_code}"
            )
            logger.warning(
                "%s server returned %d (attempt %d/%d)",
                provider_name,
                response.status_code,
                attempt,
                retries + 1,
            )

        if attempt > retries:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderTimeout(f"{provider_name} did not answer within the time budget")
        backoff = 0.1 * (2 ** (attempt - 1))
        if backoff >= remaining:
            raise ProviderTimeout(f"{provider_name} did not answer within the time budget")
        time.sleep(backoff)

    if time.monotonic() >= deadline or isinstance(last_error, httpx.TimeoutException):
        raise ProviderTimeout(f"{provider_name} did not answer within the time budget")
    raise ProviderUnavailable(f"{provider_name} could not be reached after {attempt} attempt(s)")


@runtime_checkable
class LLMProvider(Protocol):
    """What the answer service needs from a provider.

    ``generate`` raises a ``GenerationError`` subclass (never a bare transport exception) and
    must not log, return or embed prompt text or credentials in any error.
    """

    name: str
    mode: GenerationMode
    model: str

    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def info(self) -> ProviderInfo: ...

    def close(self) -> None: ...
