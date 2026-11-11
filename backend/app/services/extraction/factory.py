"""Provider selection and the per-chunk extraction runner."""

import logging

from app.core.config import Settings
from app.services.extraction.mock import MockExtractionProvider
from app.services.extraction.persistence import ChunkResult
from app.services.extraction.rule_based import RuleBasedExtractor
from app.services.extraction.types import ExtractionError, ExtractionProvider

logger = logging.getLogger(__name__)


class ConfigurationError(ValueError):
    """The configured provider cannot be constructed."""


def build_extraction_provider(settings: Settings) -> ExtractionProvider:
    """Create the provider named by ``settings.extraction_provider``."""
    name = settings.extraction_provider
    if name == "rule_based":
        return RuleBasedExtractor(settings.entity_types)
    if name == "mock":
        return MockExtractionProvider()  # empty responses: extracts nothing (test/dev only)
    raise ConfigurationError(
        "extraction_provider='llm' needs an LLM client, which is wired up in Phase 7"
    )


def run_provider(
    provider: ExtractionProvider, chunks: list[tuple[str, str]]
) -> tuple[list[ChunkResult], int]:
    """Run ``provider`` over ``(chunk_id, text)`` pairs.

    A failing chunk becomes ``ChunkResult(extraction=None)`` and is counted, never hidden.
    Returns (results, failed_count).
    """
    results: list[ChunkResult] = []
    failed = 0
    for chunk_id, text in chunks:
        try:
            extraction = provider.extract(text)
        except ExtractionError as exc:
            logger.warning("Extraction failed for chunk=%s: %s", chunk_id, exc)
            extraction = None
        except Exception:
            logger.exception("Extractor crashed on chunk=%s", chunk_id)
            extraction = None
        failed += extraction is None
        results.append(ChunkResult(chunk_id, text, extraction))
    return results, failed
