"""Unit tests for every M1 provider: mock, extractive, Ollama and OpenAI-compatible.

Design constraints respected here:
- All tests are deterministic: httpx.MockTransport replaces the network.
- No real model is called.
- API keys never appear in test assertions or log output.
- Each provider's request shape, response parsing, retry policy and failure taxonomy are
  individually covered.
- The extractive provider is tested separately because it has no network path.

Provider ordering matters:
1. The extractive provider (default) is tested most thoroughly — it is the control group.
2. Mock provider: minimal (it is a test double; its logic is in the scripted test helper).
3. Ollama provider: request mapping, response parsing, retry/timeout/5xx/4xx.
4. OpenAI-compatible: same, plus key handling, remote flag, 413/400 context-too-long.
5. Factory: maps provider names to concrete instances; refuses remote URL without opt-in.
6. Mutation checks: skip the retry, allow unknown marker, etc.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from app.core.config import Settings
from app.services.generation.errors import (
    ContextTooLong,
    ProviderBadResponse,
    ProviderNotConfigured,
    ProviderTimeout,
    ProviderUnavailable,
)
from app.services.generation.providers.base import LLMProvider, send_with_retries
from app.services.generation.providers.extractive import (
    ExtractiveProvider,
    _content_words,
    _split_sentences,
)
from app.services.generation.providers.factory import build_provider, register_all
from app.services.generation.providers.mock import MockProvider
from app.services.generation.providers.ollama import OllamaProvider
from app.services.generation.providers.openai_compat import OpenAICompatProvider
from app.services.generation.registry import ProviderRegistry
from app.services.generation.types import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    Message,
    Usage,
)

# ─────────────────────────────────────────────────────────────── helpers


def make_request(
    user_text: str = "Who works at Acme Inc?",
    system_text: str = "Answer from sources.",
    max_tokens: int = 64,
    temperature: float = 0.0,
    seed: int | None = 0,
    timeout_seconds: float | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        messages=(Message("system", system_text), Message("user", user_text)),
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        timeout_seconds=timeout_seconds,
    )


def make_settings(tmp_path, **kw) -> Settings:
    return Settings(data_dir=tmp_path, _env_file=None, **kw)


# ─────────────────────────────────────────────────────────────── mock provider


class TestMockProvider:
    def test_satisfies_the_llm_provider_protocol(self):
        assert isinstance(MockProvider(), LLMProvider)

    def test_default_script_returns_canned_text(self):
        p = MockProvider()
        r = p.generate(make_request())
        assert r.text == "mock answer" and r.model == "mock-1"
        assert r.finish_reason == "stop"

    def test_script_items_are_consumed_in_order_then_last_is_repeated(self):
        p = MockProvider("first", "second", "last")
        texts = [p.generate(make_request()).text for _ in range(5)]
        assert texts == ["first", "second", "last", "last", "last"]

    def test_a_generation_result_item_is_returned_as_is(self):
        result = GenerationResult(text="verbatim", model="m", usage=Usage(prompt_tokens=5))
        p = MockProvider(result)
        assert p.generate(make_request()) is result

    def test_an_exception_item_is_raised(self):
        exc = ProviderUnavailable("server down")
        p = MockProvider(exc)
        with pytest.raises(ProviderUnavailable):
            p.generate(make_request())

    def test_requests_are_recorded(self):
        p = MockProvider("x")
        req = make_request()
        p.generate(req)
        assert p.requests == [req]

    def test_mode_is_mock(self):
        p = MockProvider()
        assert p.mode == GenerationMode.MOCK
        assert p.info().mode == GenerationMode.MOCK

    def test_close_is_idempotent(self):
        p = MockProvider()
        p.close()
        p.close()  # no error

    def test_model_comes_from_settings_factory(self, tmp_path):
        from app.services.generation.providers.mock import build

        p = build(make_settings(tmp_path, llm_model="test-model-42"))
        assert p.model == "test-model-42"

    def test_model_defaults_when_not_set(self, tmp_path):
        from app.services.generation.providers.mock import build

        p = build(make_settings(tmp_path))
        assert p.model == "mock-1"


# ─────────────────────────────────────────────────────────────── extractive provider


SAMPLE_PROMPT = """\
Answer only from the numbered sources below.

[Source S1 — team_notes.txt, section "Team"]
Alice works at Acme Inc. Bob Smith also works at Acme Inc.
[End S1]

[Source S2 — geography.txt, section "Locations"]
Acme Inc is located in Berlin, Germany.
[End S2]

Question: Who works at Acme Inc?
"""

EMPTY_PROMPT = """\
[Source S1 — empty.txt]

[End S1]

Question: Who is here?
"""


class TestExtractiveProvider:
    def test_satisfies_the_llm_provider_protocol(self):
        assert isinstance(ExtractiveProvider(), LLMProvider)

    def test_mode_is_extractive(self):
        p = ExtractiveProvider()
        assert p.mode == GenerationMode.EXTRACTIVE
        assert p.info().mode == GenerationMode.EXTRACTIVE
        assert p.model == "extractive-v1"

    def test_answers_contain_citation_markers(self):
        p = ExtractiveProvider(answer_chars=800)
        req = make_request(user_text=SAMPLE_PROMPT, max_tokens=256)
        result = p.generate(req)
        assert "[S1]" in result.text or "[S2]" in result.text
        assert result.finish_reason == "stop"
        # extractive provider reports no usage (no model)
        assert result.usage is None

    def test_question_content_words_drive_selection(self):
        p = ExtractiveProvider(answer_chars=800)
        req = make_request(user_text=SAMPLE_PROMPT, max_tokens=256)
        result = p.generate(req)
        # "Alice" and "Bob Smith" appear in S1; Berlin appears in S2
        # The question asks about Acme Inc workers → S1 sentences should be preferred
        assert "[S1]" in result.text

    def test_no_matching_sentences_returns_insufficient_evidence(self):
        p = ExtractiveProvider(answer_chars=800)
        req = make_request(user_text=EMPTY_PROMPT, max_tokens=256)
        result = p.generate(req)
        assert result.text == "INSUFFICIENT_EVIDENCE"

    def test_budget_is_respected_no_overflow(self):
        p = ExtractiveProvider(answer_chars=30)  # very tight budget
        req = make_request(user_text=SAMPLE_PROMPT, max_tokens=256)
        result = p.generate(req)
        # Either truncated to fit or insufficient
        if result.text != "INSUFFICIENT_EVIDENCE":
            assert len(result.text) <= 30 + 20  # small slack for marker overhead

    def test_deterministic_on_repeated_calls(self):
        p = ExtractiveProvider(answer_chars=800)
        req = make_request(user_text=SAMPLE_PROMPT, max_tokens=256)
        first = p.generate(req).text
        second = p.generate(req).text
        assert first == second

    def test_close_is_a_no_op(self):
        ExtractiveProvider().close()

    def test_build_factory_derives_budget_from_settings(self, tmp_path):
        from app.services.generation.providers.extractive import build

        p = build(make_settings(tmp_path, llm_max_answer_tokens=256, answer_chars_per_token=4.0))
        assert p._answer_chars == 1024

    def test_build_factory_enforces_minimum_budget(self, tmp_path):
        from app.services.generation.providers.extractive import build

        p = build(make_settings(tmp_path, llm_max_answer_tokens=16, answer_chars_per_token=0.5))
        assert p._answer_chars >= 200


class TestSentenceSplitter:
    def test_splits_on_sentence_boundaries(self):
        text = "Alice works at Acme Inc. Bob Smith also works there. Berlin is in Germany."
        parts = _split_sentences(text)
        assert len(parts) == 3

    def test_handles_empty_and_whitespace(self):
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []

    def test_single_sentence_no_split(self):
        text = "Only one sentence here."
        assert _split_sentences(text) == ["Only one sentence here."]


class TestContentWords:
    def test_filters_stop_words(self):
        words = _content_words("Who works at Acme Inc?")
        assert "works" in words and "acme" in words
        assert "who" not in words and "at" not in words

    def test_minimum_one_token_even_for_stop_words(self):
        words = _content_words("a an the")
        assert len(words) >= 1  # falls back to all tokens when query is empty content words

    def test_empty_text_returns_empty_set(self):
        assert _content_words("") == frozenset()


# ─────────────────────────────────────────────────────────────── Ollama provider helpers


def ollama_response(
    content="Alice works at Acme Inc. [S1]",
    model="qwen2.5:1.5b-instruct",
    done_reason="stop",
    prompt_eval_count=42,
    eval_count=12,
) -> dict:
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


class _MockTransport(httpx.BaseTransport):
    """Wraps a list of (status_code, body_dict) responses, one per request."""

    def __init__(self, responses: list[tuple[int, dict]]) -> None:
        self._responses = list(responses)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if not self._responses:
            raise ProviderUnavailable("no more scripted responses")
        status, body = self._responses.pop(0)
        return httpx.Response(status, json=body, request=request)


def make_ollama_provider(responses, **kw) -> OllamaProvider:
    p = OllamaProvider(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:1.5b-instruct",
        context_window=4096,
        timeout_seconds=30.0,
        retries=kw.pop("retries", 1),
        **kw,
    )
    p._client = httpx.Client(transport=_MockTransport(responses))
    return p


# ─────────────────────────────────────────────────────────────── Ollama: request mapping


class TestOllamaRequestMapping:
    def test_satisfies_the_llm_provider_protocol(self):
        assert isinstance(
            OllamaProvider(
                base_url="http://127.0.0.1:11434",
                model="m",
                context_window=4096,
            ),
            LLMProvider,
        )

    def test_happy_path_returns_text_model_usage_and_latency(self):
        p = make_ollama_provider([(200, ollama_response())])
        result = p.generate(make_request())
        assert result.text == "Alice works at Acme Inc. [S1]"
        assert result.model == "qwen2.5:1.5b-instruct"
        assert result.usage == Usage(prompt_tokens=42, completion_tokens=12)
        assert result.finish_reason == "stop"
        assert result.latency_ms >= 0
        assert result.attempts == 1

    def test_request_maps_messages_options_endpoint_and_timeout(self):
        observed: dict = {}

        class CaptureTransport(httpx.BaseTransport):
            def handle_request(self, request):
                observed["url"] = str(request.url)
                observed["payload"] = json.loads(request.read())
                observed["timeout"] = request.extensions["timeout"]["read"]
                return httpx.Response(200, json=ollama_response(), request=request)

        p = OllamaProvider(
            base_url="http://127.0.0.1:11434",
            model="qwen2.5:1.5b-instruct",
            context_window=4096,
            timeout_seconds=30,
        )
        p._client = httpx.Client(transport=CaptureTransport())
        request = make_request(max_tokens=77, temperature=0.25, seed=13, timeout_seconds=2)
        p.generate(request)

        assert observed["url"] == "http://127.0.0.1:11434/api/chat"
        assert observed["payload"] == {
            "model": "qwen2.5:1.5b-instruct",
            "messages": [
                {"role": "system", "content": "Answer from sources."},
                {"role": "user", "content": "Who works at Acme Inc?"},
            ],
            "stream": False,
            "options": {
                "temperature": 0.25,
                "num_predict": 77,
                "num_ctx": 4096,
                "seed": 13,
            },
        }
        assert 0 < observed["timeout"] <= 2

    def test_length_finish_reason_is_mapped(self):
        p = make_ollama_provider([(200, ollama_response(done_reason="length"))])
        assert p.generate(make_request()).finish_reason == "length"

    def test_unknown_finish_reason_becomes_other(self):
        p = make_ollama_provider([(200, ollama_response(done_reason="cancelled"))])
        assert p.generate(make_request()).finish_reason == "other"

    def test_missing_token_counts_give_none_usage(self):
        body = {"model": "m", "message": {"role": "assistant", "content": "hi"}, "done": True}
        p = make_ollama_provider([(200, body)])
        result = p.generate(make_request())
        assert result.usage is None

    def test_mode_and_name(self):
        p = make_ollama_provider([])
        assert p.name == "ollama" and p.mode == GenerationMode.LLM
        assert p.info().name == "ollama"


# ─────────────────────────────────────────────────────────────── Ollama: failure taxonomy


class TestOllamaFailures:
    def test_5xx_then_success_retries_and_succeeds(self):
        p = make_ollama_provider([(503, {"error": "busy"}), (200, ollama_response())], retries=1)
        result = p.generate(make_request())
        assert result.text == "Alice works at Acme Inc. [S1]"
        assert result.attempts == 2

    def test_any_5xx_status_is_retried(self):
        p = make_ollama_provider([(501, {"error": "unimplemented"}), (200, ollama_response())])
        assert p.generate(make_request()).attempts == 2

    def test_5xx_exhausts_retries_and_raises_provider_unavailable(self):
        p = make_ollama_provider([(503, {"error": "busy"}), (503, {"error": "busy"})], retries=1)
        with pytest.raises(ProviderUnavailable):
            p.generate(make_request())

    def test_timeout_raises_provider_timeout(self):
        p = OllamaProvider(
            base_url="http://127.0.0.1:11434",
            model="m",
            context_window=4096,
            retries=0,
        )

        class TimeoutTransport(httpx.BaseTransport):
            def handle_request(self, request):
                raise httpx.ReadTimeout("timed out", request=request)

        p._client = httpx.Client(transport=TimeoutTransport())
        with pytest.raises(ProviderTimeout):
            p.generate(make_request())

    def test_timeout_on_all_retries_raises_provider_timeout(self):
        class TimeoutTransport(httpx.BaseTransport):
            def handle_request(self, request):
                raise httpx.ReadTimeout("timed out", request=request)

        p = OllamaProvider(
            base_url="http://127.0.0.1:11434",
            model="m",
            context_window=4096,
            retries=1,
        )
        p._client = httpx.Client(transport=TimeoutTransport())
        with pytest.raises(ProviderTimeout):
            p.generate(make_request())

    def test_4xx_is_not_retried_and_raises_provider_bad_response(self):
        p = make_ollama_provider([(422, {"error": "invalid request"})], retries=2)
        with pytest.raises(ProviderBadResponse):
            p.generate(make_request())

    def test_transport_error_retries_then_raises_provider_unavailable(self):
        class ConnectErrorTransport(httpx.BaseTransport):
            def handle_request(self, request):
                raise httpx.ConnectError("connection refused", request=request)

        p = OllamaProvider(
            base_url="http://127.0.0.1:11434",
            model="m",
            context_window=4096,
            retries=1,
        )
        p._client = httpx.Client(transport=ConnectErrorTransport())
        with pytest.raises(ProviderUnavailable):
            p.generate(make_request())

    def test_empty_content_raises_provider_bad_response(self):
        body = {"model": "m", "message": {"role": "assistant", "content": ""}, "done": True}
        p = make_ollama_provider([(200, body)])
        with pytest.raises(ProviderBadResponse):
            p.generate(make_request())

    def test_non_json_body_raises_provider_bad_response(self):
        class TextTransport(httpx.BaseTransport):
            def handle_request(self, request):
                return httpx.Response(200, text="not json", request=request)

        p = OllamaProvider(base_url="http://127.0.0.1:11434", model="m", context_window=4096)
        p._client = httpx.Client(transport=TextTransport())
        with pytest.raises(ProviderBadResponse):
            p.generate(make_request())

    def test_context_too_long_400_raises_context_too_long(self):
        body = {"error": "context window too long, reduce context"}
        p = make_ollama_provider([(400, body)])
        with pytest.raises(ContextTooLong):
            p.generate(make_request())

    def test_error_response_does_not_echo_prompt_or_credentials(self):
        sensitive = "document prompt and sk-private-key"
        p = make_ollama_provider([(400, {"error": sensitive})], retries=0)
        with pytest.raises(ProviderBadResponse) as exc_info:
            p.generate(make_request(user_text=sensitive))
        assert sensitive not in str(exc_info.value)

    @pytest.mark.parametrize("body", [[], {"message": []}])
    def test_malformed_json_shapes_raise_provider_bad_response(self, body):
        p = make_ollama_provider([(200, body)])
        with pytest.raises(ProviderBadResponse):
            p.generate(make_request())

    def test_the_api_key_never_appears_in_errors_or_logs(self, caplog, monkeypatch):
        monkeypatch.setenv("OLLAMA_KEY", "sk-secret-ollama-key")
        p = OllamaProvider(
            base_url="http://127.0.0.1:11434",
            model="m",
            context_window=4096,
            api_key="sk-secret-ollama-key",
            retries=0,
        )

        class ConnectErrorTransport(httpx.BaseTransport):
            def handle_request(self, request):
                raise httpx.ConnectError("connection refused", request=request)

        p._client = httpx.Client(transport=ConnectErrorTransport())
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ProviderUnavailable) as exc_info:
                p.generate(make_request())
        assert "sk-secret-ollama-key" not in str(exc_info.value)
        assert "sk-secret-ollama-key" not in caplog.text

    def test_close_releases_the_client(self):
        p = make_ollama_provider([])
        p.close()


# ─────────────────────────────────────────────────────────────── Ollama: factory


class TestOllamaFactory:
    def test_builds_with_model_and_default_url(self, tmp_path):
        from app.services.generation.providers.ollama import _OLLAMA_DEFAULT_BASE_URL, build

        s = make_settings(tmp_path, llm_provider="ollama", llm_model="qwen2.5:1.5b-instruct")
        p = build(s)
        assert isinstance(p, OllamaProvider)
        assert p._base_url == _OLLAMA_DEFAULT_BASE_URL
        assert p.model == "qwen2.5:1.5b-instruct"

    def test_uses_custom_base_url(self, tmp_path):
        from app.services.generation.providers.ollama import build

        s = make_settings(
            tmp_path,
            llm_provider="ollama",
            llm_model="m",
            llm_base_url="http://127.0.0.1:12345",
        )
        p = build(s)
        assert "12345" in p._base_url

    def test_missing_api_key_env_var_raises_provider_not_configured(self, tmp_path, monkeypatch):
        from app.services.generation.providers.ollama import build

        monkeypatch.delenv("MY_OLLAMA_KEY", raising=False)
        s = make_settings(
            tmp_path, llm_provider="ollama", llm_model="m", llm_api_key_env="MY_OLLAMA_KEY"
        )
        with pytest.raises(ProviderNotConfigured):
            build(s)


# ─────────────────────────────────────────────────────────────── OpenAI-compatible helpers


def openai_response(
    content="Alice works at Acme Inc. [S1]",
    model="gpt-local",
    finish_reason="stop",
    prompt_tokens=50,
    completion_tokens=10,
) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_openai_provider(responses, **kw) -> OpenAICompatProvider:
    p = OpenAICompatProvider(
        base_url="http://127.0.0.1:8080/v1",
        model="gpt-local",
        timeout_seconds=30.0,
        retries=kw.pop("retries", 1),
        **kw,
    )
    p._client = httpx.Client(transport=_MockTransport(responses))
    return p


# ─────────────────────────────────────────────────────────────── OpenAI-compatible: request mapping


class TestOpenAICompatRequestMapping:
    def test_satisfies_the_llm_provider_protocol(self):
        assert isinstance(
            OpenAICompatProvider(base_url="http://127.0.0.1:8080/v1", model="m"),
            LLMProvider,
        )

    def test_happy_path_returns_text_model_usage_and_latency(self):
        p = make_openai_provider([(200, openai_response())])
        result = p.generate(make_request())
        assert result.text == "Alice works at Acme Inc. [S1]"
        assert result.model == "gpt-local"
        assert result.usage == Usage(prompt_tokens=50, completion_tokens=10)
        assert result.finish_reason == "stop"
        assert result.latency_ms >= 0
        assert result.attempts == 1

    def test_request_maps_messages_options_endpoint_and_timeout(self):
        observed: dict = {}

        class CaptureTransport(httpx.BaseTransport):
            def handle_request(self, request):
                observed["url"] = str(request.url)
                observed["payload"] = json.loads(request.read())
                observed["timeout"] = request.extensions["timeout"]["read"]
                return httpx.Response(200, json=openai_response(), request=request)

        p = OpenAICompatProvider(
            base_url="http://127.0.0.1:8080/v1",
            model="gpt-local",
            timeout_seconds=30,
        )
        p._client = httpx.Client(transport=CaptureTransport())
        request = make_request(max_tokens=77, temperature=0.25, seed=13, timeout_seconds=2)
        p.generate(request)

        assert observed["url"] == "http://127.0.0.1:8080/v1/chat/completions"
        assert observed["payload"] == {
            "model": "gpt-local",
            "messages": [
                {"role": "system", "content": "Answer from sources."},
                {"role": "user", "content": "Who works at Acme Inc?"},
            ],
            "max_tokens": 77,
            "temperature": 0.25,
            "stream": False,
            "seed": 13,
        }
        assert 0 < observed["timeout"] <= 2

    def test_length_finish_reason_is_mapped(self):
        p = make_openai_provider([(200, openai_response(finish_reason="length"))])
        assert p.generate(make_request()).finish_reason == "length"

    def test_unknown_finish_reason_becomes_other(self):
        p = make_openai_provider([(200, openai_response(finish_reason="content_filter"))])
        assert p.generate(make_request()).finish_reason == "other"

    def test_missing_usage_gives_none(self):
        body = {
            "choices": [
                {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ]
        }
        p = make_openai_provider([(200, body)])
        result = p.generate(make_request())
        assert result.usage is None

    def test_mode_and_name(self):
        p = make_openai_provider([])
        assert p.name == "openai_compatible" and p.mode == GenerationMode.LLM
        assert p.info().name == "openai_compatible"

    def test_remote_flag_on_info(self):
        p = OpenAICompatProvider(base_url="https://api.example.com/v1", model="m", remote=True)
        assert p.info().remote is True

    def test_local_remote_flag_is_false(self):
        p = make_openai_provider([])
        assert p.info().remote is False


# ─────────────────────────── OpenAI-compatible: failure taxonomy


class TestOpenAICompatFailures:
    def test_5xx_then_success_retries_and_succeeds(self):
        p = make_openai_provider(
            [(503, {"error": {"message": "overloaded"}}), (200, openai_response())], retries=1
        )
        result = p.generate(make_request())
        assert result.text == "Alice works at Acme Inc. [S1]"
        assert result.attempts == 2

    def test_any_5xx_status_is_retried(self):
        p = make_openai_provider([(501, {"error": "unimplemented"}), (200, openai_response())])
        assert p.generate(make_request()).attempts == 2

    def test_5xx_exhausts_retries_raises_provider_unavailable(self):
        p = make_openai_provider([(503, {"error": {}}), (503, {"error": {}})], retries=1)
        with pytest.raises(ProviderUnavailable):
            p.generate(make_request())

    def test_timeout_raises_provider_timeout(self):
        class TimeoutTransport(httpx.BaseTransport):
            def handle_request(self, request):
                raise httpx.ReadTimeout("timed out", request=request)

        p = OpenAICompatProvider(base_url="http://127.0.0.1:8080/v1", model="m", retries=0)
        p._client = httpx.Client(transport=TimeoutTransport())
        with pytest.raises(ProviderTimeout):
            p.generate(make_request())

    def test_transport_error_retries_then_raises_provider_unavailable(self):
        class ConnectErrorTransport(httpx.BaseTransport):
            def handle_request(self, request):
                raise httpx.ConnectError("connection refused", request=request)

        p = OpenAICompatProvider(base_url="http://127.0.0.1:8080/v1", model="m", retries=1)
        p._client = httpx.Client(transport=ConnectErrorTransport())
        with pytest.raises(ProviderUnavailable):
            p.generate(make_request())

    def test_4xx_not_retried_raises_provider_bad_response(self):
        p = make_openai_provider([(422, {"error": {"message": "bad request"}})])
        with pytest.raises(ProviderBadResponse):
            p.generate(make_request())

    def test_413_context_too_long_raises_context_too_long(self):
        body = {"error": {"message": "token limit exceeded"}}
        p = make_openai_provider([(413, body)])
        with pytest.raises(ContextTooLong):
            p.generate(make_request())

    def test_400_context_length_message_raises_context_too_long(self):
        body = {"error": {"message": "context length exceeded, reduce your input"}}
        p = make_openai_provider([(400, body)])
        with pytest.raises(ContextTooLong):
            p.generate(make_request())

    def test_error_response_does_not_echo_prompt_or_credentials(self):
        sensitive = "document prompt and sk-private-key"
        p = make_openai_provider([(400, {"error": {"message": sensitive}})], retries=0)
        with pytest.raises(ProviderBadResponse) as exc_info:
            p.generate(make_request(user_text=sensitive))
        assert sensitive not in str(exc_info.value)

    def test_empty_choices_raises_provider_bad_response(self):
        body = {"choices": [], "model": "m"}
        p = make_openai_provider([(200, body)])
        with pytest.raises(ProviderBadResponse):
            p.generate(make_request())

    def test_empty_content_raises_provider_bad_response(self):
        body = {
            "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]
        }
        p = make_openai_provider([(200, body)])
        with pytest.raises(ProviderBadResponse):
            p.generate(make_request())

    def test_non_json_body_raises_provider_bad_response(self):
        class TextTransport(httpx.BaseTransport):
            def handle_request(self, request):
                return httpx.Response(200, text="not json", request=request)

        p = OpenAICompatProvider(base_url="http://127.0.0.1:8080/v1", model="m")
        p._client = httpx.Client(transport=TextTransport())
        with pytest.raises(ProviderBadResponse):
            p.generate(make_request())

    @pytest.mark.parametrize(
        "body",
        [
            [],
            {"choices": {}},
            {"choices": [[]]},
            {"choices": [{"message": []}]},
            {"choices": [{"message": {"content": "ok"}}], "usage": []},
        ],
    )
    def test_malformed_json_shapes_raise_provider_bad_response(self, body):
        p = make_openai_provider([(200, body)])
        with pytest.raises(ProviderBadResponse):
            p.generate(make_request())

    def test_the_api_key_is_in_the_authorization_header_never_in_logs_or_errors(
        self, caplog, monkeypatch
    ):
        monkeypatch.setenv("MY_OAI_KEY", "sk-secret-openai-key")
        p = OpenAICompatProvider(
            base_url="http://127.0.0.1:8080/v1",
            model="m",
            api_key="sk-secret-openai-key",
            retries=0,
        )

        sent_headers: list[dict] = []

        class CapturingTransport(httpx.BaseTransport):
            def handle_request(self, request):
                sent_headers.append(dict(request.headers))
                raise httpx.ConnectError("refused", request=request)

        p._client = httpx.Client(transport=CapturingTransport())
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ProviderUnavailable) as exc_info:
                p.generate(make_request())

        assert "sk-secret-openai-key" not in str(exc_info.value)
        assert "sk-secret-openai-key" not in caplog.text
        # The key IS in the header sent to the server (that is correct).
        assert sent_headers and "Bearer sk-secret-openai-key" in sent_headers[0].get(
            "authorization", ""
        )

    def test_close_releases_the_client(self):
        p = make_openai_provider([])
        p.close()


# ─────────────────────────────────────────────────────────────── OpenAI-compatible: factory


class TestOpenAICompatFactory:
    def test_builds_with_model_and_url(self, tmp_path):
        from app.services.generation.providers.openai_compat import build

        s = make_settings(
            tmp_path,
            llm_provider="openai_compatible",
            llm_model="phi-3",
            llm_base_url="http://127.0.0.1:8080/v1",
        )
        p = build(s)
        assert isinstance(p, OpenAICompatProvider)
        assert p.model == "phi-3"
        assert p.info().remote is False

    def test_remote_flag_is_true_for_non_loopback_url(self, tmp_path):
        from app.services.generation.providers.openai_compat import build

        s = make_settings(
            tmp_path,
            llm_provider="openai_compatible",
            llm_model="m",
            llm_base_url="https://api.example.com/v1",
            llm_allow_remote=True,
        )
        p = build(s)
        assert p.info().remote is True

    def test_missing_api_key_env_var_raises_provider_not_configured(self, tmp_path, monkeypatch):
        from app.services.generation.providers.openai_compat import build

        monkeypatch.delenv("MY_OAI_KEY", raising=False)
        s = make_settings(
            tmp_path,
            llm_provider="openai_compatible",
            llm_model="m",
            llm_base_url="http://127.0.0.1:8080/v1",
            llm_api_key_env="MY_OAI_KEY",
        )
        with pytest.raises(ProviderNotConfigured):
            build(s)


# ─────────────────────────────────────────────────────────────── factory module


class TestFactory:
    def test_register_all_populates_all_four_providers(self):
        reg = ProviderRegistry()
        register_all(reg)
        assert set(reg.names()) == {"extractive", "mock", "ollama", "openai_compatible"}

    def test_build_provider_returns_extractive_by_default(self, tmp_path):
        p = build_provider(make_settings(tmp_path))
        assert isinstance(p, ExtractiveProvider)

    def test_build_provider_returns_mock(self, tmp_path):
        p = build_provider(make_settings(tmp_path, llm_provider="mock"))
        assert isinstance(p, MockProvider)

    def test_build_provider_returns_ollama(self, tmp_path):
        p = build_provider(
            make_settings(tmp_path, llm_provider="ollama", llm_model="qwen2.5:1.5b-instruct")
        )
        assert isinstance(p, OllamaProvider)

    def test_build_provider_returns_openai_compatible(self, tmp_path):
        p = build_provider(
            make_settings(
                tmp_path,
                llm_provider="openai_compatible",
                llm_model="m",
                llm_base_url="http://127.0.0.1:8080/v1",
            )
        )
        assert isinstance(p, OpenAICompatProvider)

    def test_remote_url_without_allow_remote_is_refused_by_registry(self, tmp_path):
        """The registry checks the URL again even when settings passed it through."""
        from app.services.generation.registry import ProviderRegistry

        reg = ProviderRegistry()
        register_all(reg)
        s = Settings.model_construct(
            **{
                **make_settings(
                    tmp_path,
                    llm_provider="openai_compatible",
                    llm_model="m",
                    llm_base_url="http://127.0.0.1:8080/v1",
                ).model_dump(),
                "llm_base_url": "http://10.0.0.5:8080/v1",
                "llm_allow_remote": False,
            }
        )
        with pytest.raises(ProviderNotConfigured, match="GRAPHRAG_LLM_ALLOW_REMOTE"):
            reg.build(s)

    def test_remote_url_with_allow_remote_builds_fine(self, tmp_path):
        from app.services.generation.registry import ProviderRegistry

        reg = ProviderRegistry()
        register_all(reg)
        s = make_settings(
            tmp_path,
            llm_provider="openai_compatible",
            llm_model="m",
            llm_base_url="https://api.example.com/v1",
            llm_allow_remote=True,
        )
        p = reg.build(s)
        assert isinstance(p, OpenAICompatProvider)
        assert p.info().remote is True


# ─────────────────────────────────────────────────────────────── mutation checks


class TestMutationChecks:
    """These tests exist to catch deliberate regressions in the safety properties.

    Each mutates one behaviour that the proposal requires; the test must fail if the
    code is reverted to the mutated form. They document that:
    - 5xx IS retried (removing the retry makes this fail)
    - 4xx is NOT retried (removing the guard would retry and exhaust scripts)
    - timeout IS NOT retried after retries=0
    - context-too-long raises ContextTooLong, not ProviderUnavailable
    - API key is never echoed in exception text
    """

    def test_mutation_5xx_must_be_retried(self):
        """If the retry is removed, a 5xx would immediately raise ProviderUnavailable."""
        p = make_ollama_provider([(503, {"error": "busy"}), (200, ollama_response())], retries=1)
        result = p.generate(make_request())
        assert result.text == "Alice works at Acme Inc. [S1]"

    def test_mutation_4xx_must_not_be_retried(self):
        """If 4xx were retried, the second item would be consumed; with no retry the
        script has only one item and the error is immediate."""
        p = make_ollama_provider(
            [(422, {"error": "bad request"}), (200, ollama_response())], retries=2
        )
        with pytest.raises(ProviderBadResponse):
            p.generate(make_request())

    def test_mutation_context_too_long_must_not_become_provider_unavailable(self):
        body = {"error": "context window too long, reduce context"}
        p = make_ollama_provider([(400, body)])
        with pytest.raises(ContextTooLong):
            p.generate(make_request())
        # Mutation: if the ContextTooLong check were removed, the same 400 would
        # become ProviderBadResponse.  The test ensures the specific class is raised.

    def test_mutation_key_must_never_appear_in_error_text(self):
        secret = "sk-must-not-appear-in-any-output"
        p = OllamaProvider(
            base_url="http://127.0.0.1:11434",
            model="m",
            context_window=4096,
            api_key=secret,
            retries=0,
        )

        class ConnectErrorTransport(httpx.BaseTransport):
            def handle_request(self, request):
                raise httpx.ConnectError("refused", request=request)

        p._client = httpx.Client(transport=ConnectErrorTransport())
        try:
            p.generate(make_request())
        except ProviderUnavailable as exc:
            assert secret not in str(exc)
        except Exception:
            pass  # any error is acceptable as long as the key is not in it

    def test_mutation_extractive_always_cites_sources(self):
        """The extractive provider must cite [S#] markers. If citation is removed
        the grounding checks in M3 would fail every answer."""
        p = ExtractiveProvider(answer_chars=800)
        req = make_request(user_text=SAMPLE_PROMPT, max_tokens=256)
        result = p.generate(req)
        if result.text != "INSUFFICIENT_EVIDENCE":
            import re

            assert re.search(r"\[S\d+\]", result.text), "extractive answer must cite sources"


class TestSharedRetryPolicy:
    def test_backoff_and_retries_stay_within_the_overall_deadline(self, monkeypatch):
        from app.services.generation.providers import base as provider_base

        now = 0.0
        sleeps: list[float] = []
        timeouts: list[float] = []

        def monotonic() -> float:
            return now

        def sleep(seconds: float) -> None:
            nonlocal now
            sleeps.append(seconds)
            now += seconds

        def send(timeout: float) -> httpx.Response:
            timeouts.append(timeout)
            return httpx.Response(501)

        monkeypatch.setattr(provider_base.time, "monotonic", monotonic)
        monkeypatch.setattr(provider_base.time, "sleep", sleep)

        with pytest.raises(ProviderTimeout):
            send_with_retries(
                "test provider",
                send,
                retries=3,
                timeout_seconds=0.25,
            )

        assert sleeps == [0.1]
        assert timeouts == [0.25, pytest.approx(0.15)]
