"""Value types shared by every answer-generation component.

Nothing here talks to a network or reads configuration. ``repr()`` of anything that can hold
document text or an answer shows sizes only, so a stray ``logger.debug("%r", request)`` can
never write retrieved text to a log.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

PROMPT_ROLES = ("system", "user", "assistant")
Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "other"]


class GenerationMode(StrEnum):
    """What produced the answer. Reported in every response; never hidden or defaulted."""

    LLM = "llm"  # a language model behind an LLM provider
    MOCK = "mock"  # scripted test provider: not a real answer
    EXTRACTIVE = "extractive"  # sentences copied from the sources; no model involved
    NONE = "none"  # nothing was generated (for example insufficient evidence)


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in PROMPT_ROLES:
            raise ValueError(f"unknown message role {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be text")

    def __repr__(self) -> str:
        return f"Message(role={self.role!r}, chars={len(self.content)})"


@dataclass(frozen=True)
class GenerationRequest:
    """One completion request. Deterministic by default (temperature 0, fixed seed)."""

    messages: tuple[Message, ...]
    max_tokens: int
    temperature: float = 0.0
    seed: int | None = 0
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a generation request needs at least one message")
        if self.messages[-1].role != "user":
            raise ValueError("the last message of a generation request must be a user message")
        if isinstance(self.max_tokens, bool) or self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @property
    def prompt_chars(self) -> int:
        return sum(len(m.content) for m in self.messages)

    def __repr__(self) -> str:
        return (
            f"GenerationRequest(messages={len(self.messages)}, chars={self.prompt_chars}, "
            f"max_tokens={self.max_tokens}, temperature={self.temperature})"
        )


@dataclass(frozen=True)
class Usage:
    """Token counts as REPORTED by the provider (None when it does not report them)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    finish_reason: FinishReason = "stop"
    usage: Usage | None = None
    latency_ms: float = 0.0
    attempts: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("generated text must be a string")

    @property
    def truncated(self) -> bool:
        """The model stopped because it hit the token limit, not because it finished."""
        return self.finish_reason == "length"

    def __repr__(self) -> str:
        return (
            f"GenerationResult(model={self.model!r}, chars={len(self.text)}, "
            f"finish_reason={self.finish_reason!r}, attempts={self.attempts})"
        )


@dataclass(frozen=True)
class ProviderInfo:
    """What a response may say about the provider. Contains no URL parameters and no secrets."""

    name: str
    mode: GenerationMode
    model: str
    remote: bool = False
