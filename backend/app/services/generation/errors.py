"""Provider error taxonomy and its mapping to the API error envelope.

Every failure a provider can cause becomes one of these classes, each with a stable machine-
readable code. Messages are short, single-line and secret-free (``sanitize_message``); a provider
response body is never copied into an error, because a model server may echo the prompt.
"""

from app.core.errors import AppError
from app.core.privacy import sanitize_message


class GenerationError(Exception):
    """Base class: something stopped an answer from being produced."""

    code = "generation_error"
    http_status = 503
    retryable = False

    def __init__(self, message: str, *, secrets: tuple[str, ...] = ()) -> None:
        super().__init__(sanitize_message(message, secrets=secrets))


class ProviderNotConfigured(GenerationError):
    """The configured provider cannot be built (missing setting, not available, unsafe URL)."""

    code = "llm_not_configured"


class ProviderUnavailable(GenerationError):
    """The provider could not be reached or answered with a server error."""

    code = "llm_unavailable"
    retryable = True


class ProviderTimeout(GenerationError):
    """The provider did not answer within the time budget."""

    code = "llm_timeout"
    retryable = True


class ProviderBadResponse(GenerationError):
    """The provider answered, but not with something usable (empty, malformed, rejected)."""

    code = "llm_bad_response"


class ContextTooLong(GenerationError):
    """The prompt does not fit the model's context window."""

    code = "llm_context_too_long"


class GenerationBusy(GenerationError):
    """Too many generations are already running (the concurrency limit)."""

    code = "generation_busy"
    retryable = True


PROVIDER_ERRORS: tuple[type[GenerationError], ...] = (
    ProviderNotConfigured,
    ProviderUnavailable,
    ProviderTimeout,
    ProviderBadResponse,
    ContextTooLong,
    GenerationBusy,
)


def to_app_error(exc: GenerationError, details: object | None = None) -> AppError:
    """The API error for a generation failure.

    ``details`` is where the caller puts the retrieved sources, so a failed generation still
    returns the evidence that was found.
    """
    return AppError(exc.http_status, exc.code, str(exc), details)
