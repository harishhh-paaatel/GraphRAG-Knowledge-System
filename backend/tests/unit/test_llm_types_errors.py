"""Generation value types, the error taxonomy and its mapping to API errors."""

import pytest

from app.core.errors import AppError
from app.services.generation.errors import (
    PROVIDER_ERRORS,
    ContextTooLong,
    GenerationBusy,
    GenerationError,
    ProviderBadResponse,
    ProviderNotConfigured,
    ProviderTimeout,
    ProviderUnavailable,
    to_app_error,
)
from app.services.generation.tokens import (
    PER_MESSAGE_OVERHEAD_TOKENS,
    EstimatingCounter,
    TokenCounter,
)
from app.services.generation.types import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    Message,
    ProviderInfo,
    Usage,
)

# ------------------------------------------------------------------------- messages


def test_a_message_needs_a_known_role_and_text():
    assert Message("user", "hi").content == "hi"
    with pytest.raises(ValueError, match="unknown message role"):
        Message("tool", "x")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Message("user", 5)  # type: ignore[arg-type]


def test_repr_never_shows_document_text_or_answers():
    secret = "CONFIDENTIAL document text"
    m = Message("user", secret)
    req = GenerationRequest((Message("system", "rules"), m), max_tokens=64)
    res = GenerationResult(text=secret, model="m")
    for obj in (m, req, res):
        for rendering in (repr(obj), str(obj), f"{obj!r}"):
            assert "CONFIDENTIAL" not in rendering
    assert "chars=" in repr(m) and "messages=2" in repr(req) and "chars=" in repr(res)


# ------------------------------------------------------------------------- request


def req(**kw):
    kw.setdefault("messages", (Message("user", "q"),))
    kw.setdefault("max_tokens", 10)
    return GenerationRequest(**kw)


def test_a_request_is_deterministic_by_default():
    r = req()
    assert (r.temperature, r.seed, r.timeout_seconds) == (0.0, 0, None)
    assert r.prompt_chars == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"messages": ()}, "at least one message"),
        ({"messages": (Message("system", "s"),)}, "last message"),
        ({"messages": (Message("user", "a"), Message("assistant", "b"))}, "last message"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": True}, "max_tokens"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": 2.5}, "temperature"),
        ({"timeout_seconds": 0}, "timeout"),
    ],
)
def test_invalid_requests_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        req(**kwargs)


def test_a_result_reports_truncation_and_requires_text():
    assert GenerationResult("a", "m", finish_reason="length").truncated is True
    assert GenerationResult("a", "m").truncated is False
    with pytest.raises(TypeError):
        GenerationResult(None, "m")  # type: ignore[arg-type]
    assert Usage().prompt_tokens is None


def test_generation_modes_are_the_documented_four():
    assert {m.value for m in GenerationMode} == {"llm", "mock", "extractive", "none"}
    assert max(len(m.value) for m in GenerationMode) <= 16  # fits query_history.generation_mode
    info = ProviderInfo("ollama", GenerationMode.LLM, "qwen")
    assert info.remote is False


# --------------------------------------------------------------------------- errors

EXPECTED = {
    ProviderNotConfigured: ("llm_not_configured", False),
    ProviderUnavailable: ("llm_unavailable", True),
    ProviderTimeout: ("llm_timeout", True),
    ProviderBadResponse: ("llm_bad_response", False),
    ContextTooLong: ("llm_context_too_long", False),
    GenerationBusy: ("generation_busy", True),
}


def test_every_provider_error_has_a_unique_stable_code_and_retry_flag():
    assert set(PROVIDER_ERRORS) == set(EXPECTED)
    for cls, (code, retryable) in EXPECTED.items():
        assert (cls.code, cls.retryable, cls.http_status) == (code, retryable, 503)
        assert issubclass(cls, GenerationError)
    assert len({c.code for c in PROVIDER_ERRORS}) == len(PROVIDER_ERRORS)


def test_transient_failures_are_retryable_and_permanent_ones_are_not():
    retry = {c for c in PROVIDER_ERRORS if c.retryable}
    assert retry == {ProviderUnavailable, ProviderTimeout, GenerationBusy}


def test_messages_are_short_single_line_and_secret_free():
    exc = ProviderBadResponse(
        "bad\nresponse with key sk-abc123 " + "x" * 400, secrets=("sk-abc123",)
    )
    assert "sk-abc123" not in str(exc) and "\n" not in str(exc) and len(str(exc)) <= 200


@pytest.mark.parametrize("cls", list(EXPECTED))
def test_mapping_to_the_api_error_envelope(cls):
    sources = {"sources": [{"chunk_id": "c1", "filename": "a.txt"}]}
    err = to_app_error(cls("something went wrong"), sources)
    assert isinstance(err, AppError)
    assert (err.status_code, err.code) == (503, cls.code)
    assert err.message == "something went wrong" and err.details == sources


def test_mapping_without_details():
    assert to_app_error(ProviderTimeout("slow")).details is None


# ----------------------------------------------------------------------------- tokens


def test_the_estimator_over_estimates_ascii_and_counts_non_ascii_characters_whole():
    c = EstimatingCounter(3.0)
    assert isinstance(c, TokenCounter)
    assert c.count("") == 0
    assert c.count("abc") == 1 and c.count("abcd") == 2  # ceil(4 / 3)
    assert c.count("你好世界") == 4  # one token per non-ASCII character
    assert c.count("ab你") == 2  # ceil(2/3) + 1


def test_the_estimator_is_monotonic_and_configurable():
    c = EstimatingCounter(2.0)
    lengths = [c.count("x" * n) for n in range(0, 40)]
    assert lengths == sorted(lengths)
    assert EstimatingCounter(1.0).count("abcdef") == 6
    assert EstimatingCounter(6.0).count("abcdef") == 1
    for bad in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            EstimatingCounter(bad)


def test_message_counts_include_the_per_message_overhead():
    c = EstimatingCounter(3.0)
    msgs = [Message("system", "abc"), Message("user", "abcdef")]
    assert c.count_messages(msgs) == 1 + 2 + 2 * PER_MESSAGE_OVERHEAD_TOKENS
    assert c.count_messages([]) == 0
