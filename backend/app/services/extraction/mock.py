"""Scriptable mock provider for tests. Not an extractor: it returns what it is told to."""

import copy

from app.services.extraction.types import ChunkExtraction, ExtractionError


class MockExtractionProvider:
    """Returns a canned ChunkExtraction when the chunk contains a registered substring.

    Chunks that match nothing yield an empty extraction. Substrings listed in ``fail_on``
    raise ExtractionError, to exercise failure handling. Every call is recorded in ``calls``.
    """

    name = "mock"

    def __init__(
        self,
        responses: dict[str, ChunkExtraction] | None = None,
        fail_on: set[str] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.fail_on = fail_on or set()
        self.calls: list[str] = []

    def extract(self, text: str) -> ChunkExtraction:
        self.calls.append(text)
        if any(needle in text for needle in self.fail_on):
            raise ExtractionError("mock provider failure")
        for needle, response in self.responses.items():
            if needle in text:
                return copy.deepcopy(response)
        return ChunkExtraction()
