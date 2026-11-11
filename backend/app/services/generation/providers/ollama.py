"""Ollama provider — HTTP client for a local Ollama server.

POST {base_url}/api/chat with stream=false, temperature=0, num_predict and num_ctx.
Retries transport errors and 5xx responses; never retries 4xx or context-too-long.

Privacy constraints (enforced; not documented inline to avoid making logs noisy):
- No prompt text, no question, no document excerpt appears in any log message.
- The API key (via Authorization header) is never logged or included in an error message.
- The base URL is validated against the loopback rule by the registry before this module
  is instantiated.

The module imports ``httpx`` at the top level. The generation boundary test allows this
inside ``providers/`` only; non-provider modules must not import ``httpx``.
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

_OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434"

# Ollama finish reasons and their mapping to our vocabulary.
_FINISH_REASONS = {
    "stop": "stop",
    "length": "length",
}


class OllamaProvider:
    """Thin HTTP wrapper around Ollama's ``/api/chat`` endpoint.

    Parameters
    ----------
    base_url:
        Ollama server base URL, e.g. ``http://127.0.0.1:11434``.
    model:
        Model tag passed verbatim to Ollama (e.g. ``qwen2.5:1.5b-instruct``).
    timeout_seconds:
        Per-call HTTP timeout.
    retries:
        Number of additional attempts on transport errors and 5xx.
    api_key:
        Bearer key for Ollama (optional; most local installs need none).
    """

    name = "ollama"
    mode = GenerationMode.LLM

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        context_window: int,
        timeout_seconds: float = 120.0,
        retries: int = 1,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._context_window = context_window
        self._timeout = timeout_seconds
        self._retries = retries
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=timeout_seconds)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "num_ctx": self._context_window,
                **({"seed": request.seed} if request.seed is not None else {}),
            },
        }
        url = f"{self._base_url}/api/chat"
        t0 = time.monotonic()
        timeout = min(
            self._timeout,
            request.timeout_seconds if request.timeout_seconds is not None else self._timeout,
        )
        response, attempts = send_with_retries(
            "Ollama",
            lambda remaining: self._client.post(
                url, json=payload, headers=self._headers, timeout=remaining
            ),
            retries=self._retries,
            timeout_seconds=timeout,
        )

        if response.status_code == 200:
            latency_ms = (time.monotonic() - t0) * 1000
            return self._parse(response, latency_ms=latency_ms, attempts=attempts)

        if response.status_code == 400:
            body_text = sanitize_message(response.text)
            if "context" in body_text.lower() and "long" in body_text.lower():
                raise ContextTooLong("the prompt exceeds the Ollama model's context window")
        raise ProviderBadResponse(f"Ollama rejected the request (HTTP {response.status_code})")

    def _parse(
        self, response: httpx.Response, *, latency_ms: float, attempts: int
    ) -> GenerationResult:
        try:
            body = response.json()
        except Exception as exc:
            raise ProviderBadResponse("Ollama returned a non-JSON body") from exc

        if not isinstance(body, dict):
            raise ProviderBadResponse("Ollama returned a malformed response")
        message = body.get("message") or {}
        if not isinstance(message, dict):
            raise ProviderBadResponse("Ollama returned a malformed message")
        text = message.get("content", "")
        if not isinstance(text, str) or not text.strip():
            raise ProviderBadResponse("Ollama returned an empty or missing message content")

        raw_reason = body.get("done_reason", "stop")
        finish_reason = (
            _FINISH_REASONS.get(raw_reason, "other") if isinstance(raw_reason, str) else "other"
        )

        # Ollama uses "prompt_eval_count" and "eval_count" for token counts.
        usage: Usage | None = None
        prompt_tokens = body.get("prompt_eval_count")
        completion_tokens = body.get("eval_count")
        if prompt_tokens is not None or completion_tokens is not None:
            if any(
                not isinstance(count, int) or isinstance(count, bool) or count < 0
                for count in (prompt_tokens, completion_tokens)
                if count is not None
            ):
                raise ProviderBadResponse("Ollama returned invalid token usage")
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        model = body.get("model", self.model)
        if not isinstance(model, str) or not model:
            raise ProviderBadResponse("Ollama returned an invalid model name")
        return GenerationResult(
            text=text,
            model=model,
            finish_reason=finish_reason,  # type: ignore[arg-type]
            usage=usage,
            latency_ms=latency_ms,
            attempts=attempts,
        )

    def info(self) -> ProviderInfo:
        return ProviderInfo(name=self.name, mode=self.mode, model=self.model)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            logger.exception("Closing the Ollama HTTP client failed")


def build(settings: Settings) -> LLMProvider:
    """Factory: create an ``OllamaProvider`` from settings.

    Raises ``ProviderNotConfigured`` (via the registry) if the model is not set.
    """
    base_url = settings.llm_base_url or _OLLAMA_DEFAULT_BASE_URL

    try:
        api_key = resolve_api_key(settings.llm_api_key_env)
    except MissingApiKey as exc:
        from app.services.generation.errors import ProviderNotConfigured

        raise ProviderNotConfigured(str(exc)) from exc

    return OllamaProvider(
        base_url=base_url,
        model=settings.llm_model or "",
        context_window=settings.llm_context_window,
        timeout_seconds=settings.llm_timeout_seconds,
        retries=settings.llm_retries,
        api_key=api_key,
    )
