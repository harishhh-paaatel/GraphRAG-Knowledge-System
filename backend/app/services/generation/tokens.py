"""Token counting for the context budget.

The model behind a provider is unknown (any OpenAI-style endpoint, any local model), so its real
tokenizer is unavailable. The budget therefore uses a conservative ESTIMATE, and providers that
report real prompt tokens have them recorded next to the estimate so its error can be measured.
"""

import math
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from app.services.generation.types import Message

PER_MESSAGE_OVERHEAD_TOKENS = 4  # role markers and separators, a typical chat-template cost


@runtime_checkable
class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def count_messages(self, messages: Iterable[Message]) -> int: ...


class EstimatingCounter:
    """``ceil(ascii_characters / chars_per_token) + non_ascii_characters``.

    ASCII text costs about one token per 3-4 characters in common tokenizers, so the default of
    3 over-estimates English on purpose. Every non-ASCII character is counted as a whole token,
    which is what scripts such as Chinese or Japanese need and over-estimates accented Latin.
    It is an estimate: it can under-count for unusual text (long random strings, code with many
    symbols), which is why the budget also leaves headroom and providers' real counts are kept.
    """

    name = "estimate"

    def __init__(self, chars_per_token: float = 3.0) -> None:
        if not chars_per_token > 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        ascii_chars = len(text) - non_ascii
        return math.ceil(ascii_chars / self.chars_per_token) + non_ascii

    def count_messages(self, messages: Iterable[Message]) -> int:
        return sum(self.count(m.content) + PER_MESSAGE_OVERHEAD_TOKENS for m in messages)
