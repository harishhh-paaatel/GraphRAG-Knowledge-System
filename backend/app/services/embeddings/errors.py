"""Embedding exceptions (kept separate so policy modules can raise them without import cycles)."""


class EmbeddingError(Exception):
    """The embedding backend failed (model could not load, bad output, ...)."""


class EmbeddingInputError(ValueError):
    """The caller supplied text that cannot be embedded (empty / blank / not a string)."""
