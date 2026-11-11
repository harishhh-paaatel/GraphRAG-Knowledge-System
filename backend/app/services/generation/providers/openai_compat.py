"""OpenAI-compatible provider — HTTP client for any OpenAI-style ``/chat/completions`` endpoint.

Works with local servers (llama.cpp, vLLM, LM Studio, text-generation-webui) and remote
OpenAI-compatible APIs when ``GRAPHRAG_LLM_ALLOW_REMOTE=true`` is set explicitly.

POST {base_url}/chat/completions with the standard Chat Completions payload.
Retries transport errors and 5xx; never retries 4xx (including 413 / context-too-long).

Privacy rules: same as ``ollama.py`` — no prompt text, question or document excerpt in
any log message; the API key is passed only in the Authorization header and never echoed.
"""

from __future__ import annotations

import logging
import time

import httpx

from app.core.config import Settings
from app.core.privacy import MissingApiKey, resolve_api_key, sanitize_message
from app.services.generation.errors import (
    ContextTooLong,
    ProviderBadResponse,
)
from app.services.generation.providers.base import LLMProvider, send_with_retries
from app.services.generation.types import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    ProviderInfo,
    Usage,
)

logger = logging.getLogger(__name__)

_FINISH_REASONS = {
    "stop": "stop",
    "length": "length",
}


class OpenAICompatProvider:
    """HTTP wrapper for any OpenAI-compatible ``/chat/completions`` endpoint.

    Parameters
    ----------
    base_url:
        Server base URL, e.g. ``http://127.0.0.1:8080/v1``.  ``/chat/completions``
        is appended automatically.
    model:
        Model identifier passed verbatim to the server.
    timeout_seconds:
        Per-call HTTP timeout.
    retries:
        Number of additional attempts on transport errors and 5xx.
    api_key:
        Bearer key for authenticated endpoints (optional).
    remote:
        Whether the URL is non-loopback (reported in ``info()``; set by the factory).
    """

    name = "openai_compatible"
    mode = GenerationMode.LLM

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        retries: int = 1,
        api_key: str | None = None,
        remote: bool = False,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._retries = retries
        self._remote = remote
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=timeout_seconds)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }
        if request.seed is not None:
            payload["seed"] = request.seed

        url = f"{self._base_url}/chat/completions"
        t0 = time.monotonic()
        timeout = min(
            self._timeout,
            request.timeout_seconds if request.timeout_seconds is not None else self._timeout,
        )
        response, attempts = send_with_retries(
            "OpenAI-compatible",
            lambda remaining: self._client.post(
                url, json=payload, headers=self._headers, timeout=remaining
            ),
            retries=self._retries,
            timeout_seconds=timeout,
        )

        if response.status_code == 200:
            latency_ms = (time.monotonic() - t0) * 1000
            return self._parse(response, latency_ms=latency_ms, attempts=attempts)

        body_text = sanitize_message(response.text)
        if response.status_code in (400, 413):
            lower = body_text.lower()
            if any(kw in lower for kw in ("context", "token", "length", "exceed")):
                raise ContextTooLong("the prompt exceeds the model's context window")
        raise ProviderBadResponse(f"server rejected the request (HTTP {response.status_code})")

    def _parse(
        self, response: httpx.Response, *, latency_ms: float, attempts: int
    ) -> GenerationResult:
        try:
            body = response.json()
        except Exception as exc:
            raise ProviderBadResponse("server returned a non-JSON body") from exc

        if not isinstance(body, dict):
            raise ProviderBadResponse("server returned a malformed response")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderBadResponse("server returned no choices")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise ProviderBadResponse("server returned a malformed choice")
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise ProviderBadResponse("server returned a malformed message")
        text = message.get("content", "")
        if not isinstance(text, str) or not text.strip():
            raise ProviderBadResponse("server returned an empty or missing message content")

        raw_reason = choice.get("finish_reason", "stop")
        finish_reason = (
            _FINISH_REASONS.get(raw_reason, "other") if isinstance(raw_reason, str) else "other"
        )

        usage_data = body.get("usage")
        if usage_data is None:
            usage_data = {}
        if not isinstance(usage_data, dict):
            raise ProviderBadResponse("server returned invalid token usage")
        usage: Usage | None = None
        if usage_data:
            counts = (usage_data.get("prompt_tokens"), usage_data.get("completion_tokens"))
            if any(
                not isinstance(count, int) or isinstance(count, bool) or count < 0
                for count in counts
                if count is not None
            ):
                raise ProviderBadResponse("server returned invalid token usage")
            usage = Usage(
                prompt_tokens=counts[0],
                completion_tokens=counts[1],
            )

        # Use the model name the server reports, falling back to what we sent.
        reported_model = body.get("model", self.model)
        if not isinstance(reported_model, str) or not reported_model:
            raise ProviderBadResponse("server returned an invalid model name")

        return GenerationResult(
            text=text,
            model=reported_model,
            finish_reason=finish_reason,  # type: ignore[arg-type]
            usage=usage,
            latency_ms=latency_ms,
            attempts=attempts,
        )

    def info(self) -> ProviderInfo:
        return ProviderInfo(name=self.name, mode=self.mode, model=self.model, remote=self._remote)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            logger.exception("Closing the OpenAI-compatible HTTP client failed")


def build(settings: Settings) -> LLMProvider:
    """Factory: create an ``OpenAICompatProvider`` from settings.

    ``settings.llm_base_url`` is required (enforced by the settings validator).
    ``settings.llm_allow_remote`` determines whether a non-loopback URL is accepted
    (the registry re-checks this via ``check_provider_url`` before calling us).
    """
    from urllib.parse import urlsplit

    from app.core.privacy import is_loopback_host

    base_url = settings.llm_base_url or ""

    try:
        api_key = resolve_api_key(settings.llm_api_key_env)
    except MissingApiKey as exc:
        from app.services.generation.errors import ProviderNotConfigured

        raise ProviderNotConfigured(str(exc)) from exc

    # Determine if the URL is remote for the ``info()`` call.
    parsed = urlsplit(base_url)
    remote = not is_loopback_host(parsed.hostname or "")

    return OpenAICompatProvider(
        base_url=base_url,
        model=settings.llm_model or "",
        timeout_seconds=settings.llm_timeout_seconds,
        retries=settings.llm_retries,
        api_key=api_key,
        remote=remote,
    )
