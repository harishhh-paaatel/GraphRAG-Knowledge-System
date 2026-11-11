"""Ingestion exceptions. Messages are safe to show to API clients."""


class IngestionError(Exception):
    """Base class for expected, user-presentable ingestion failures."""


class ParseError(IngestionError):
    """A file could not be parsed or contained no extractable text."""
