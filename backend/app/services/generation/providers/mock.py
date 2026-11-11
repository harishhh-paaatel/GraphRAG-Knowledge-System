"""Scripted mock provider — test-only, never used in production.

Replays a fixed script of text strings, ``GenerationResult`` objects or exceptions,
one item per ``generate()`` call. The last item is repeated when the script is exhausted.

The ``generation_mode`` is ``mock`` so callers and log readers can tell at a glance that the
answer came from a script, not from a real model or the extractive algorithm.
"""

from __future__ import annotations

import logging

from app.core.config import Settings
from app.services.generation.errors import GenerationError
from app.services.generation.providers.base import LLMProvider
from app.services.generation.types import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    ProviderInfo,
    Usage,
)

logger = logging.getLogger(__name__)

# A script item is a canned text, a full result or an exception to raise.
ScriptItem = str | GenerationResult | GenerationError | Exception

_DEFAULT_MODEL = "mock-1"


class MockProvider:
    """Scripted provider that replays ``script`` and records every request it receives.

    Primarily used by tests to avoid a real network or model call. Can also be selected
    via ``GRAPHRAG_LLM_PROVIDER=mock`` for end-to-end demonstrations.

    Parameters
    ----------
    script:
        One or more items. Each ``generate()`` call consumes the next item; the last item
        is repeated indefinitely once the script is exhausted.
    model:
        Reported as ``model`` in every ``GenerationResult``.
    """

    name = "mock"
    mode = GenerationMode.MOCK

    def __init__(
        self,
        *script: ScriptItem,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._script: list[ScriptItem] = list(script) if script else ["mock answer"]
        self.model = model
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        # advance the script; repeat the last item when exhausted
        item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, GenerationResult):
            return item
        # plain text: wrap it into a result with minimal usage info
        return GenerationResult(
            text=str(item),
            model=self.model,
            usage=Usage(prompt_tokens=len(request.messages), completion_tokens=1),
        )

    def info(self) -> ProviderInfo:
        return ProviderInfo(name=self.name, mode=self.mode, model=self.model)

    def close(self) -> None:
        pass  # no resources to release


def build(settings: Settings) -> LLMProvider:
    """Factory: create a ``MockProvider`` from settings.

    Model name is taken from ``settings.llm_model`` when set; otherwise the built-in default.
    The mock provider accepts any model name and never actually uses it.
    """
    model = settings.llm_model or _DEFAULT_MODEL
    return MockProvider(model=model)
