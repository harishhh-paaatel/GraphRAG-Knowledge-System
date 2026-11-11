"""Provider-neutral extraction data types and the relationship type vocabulary."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ExtractedEntity:
    """An entity as reported by a provider, before validation and resolution."""

    name: str
    entity_type: str
    surface_form: str | None = None  # exact text in the chunk; defaults to ``name``
    description: str | None = None
    confidence: float | None = None  # None unless the method really produces one


@dataclass(frozen=True)
class ExtractedRelationship:
    """A directed relationship between two entities named in the same chunk."""

    source: str  # entity name or surface form
    target: str
    relationship_type: str
    supporting_text: str  # must be an exact (whitespace/case-insensitive) span of the chunk
    confidence: float | None = None


@dataclass
class ChunkExtraction:
    """Everything a provider found in one chunk."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)


class ExtractionError(Exception):
    """A provider failed on a chunk (bad model output, unavailable backend, ...)."""


class ExtractionProvider(Protocol):
    """Anything that can turn chunk text into entities and relationships."""

    name: str

    def extract(self, text: str) -> ChunkExtraction:
        """Return raw (unvalidated) findings; raise ExtractionError on failure."""
        ...


# relationship type -> (allowed source entity types, allowed target entity types).
# ``None`` means any type. Types not present in the configured entity vocabulary are
# simply never produced, so custom vocabularies keep working with the unrestricted types.
_ANY = None
RELATIONSHIP_RULES: dict[str, tuple[frozenset[str] | None, frozenset[str] | None]] = {
    "WORKS_AT": (frozenset({"PERSON"}), frozenset({"ORGANIZATION"})),
    "DEVELOPED": (
        frozenset({"PERSON", "ORGANIZATION"}),
        frozenset({"PRODUCT", "TECHNOLOGY", "CONCEPT"}),
    ),
    "USES": (
        frozenset({"PERSON", "ORGANIZATION", "PRODUCT", "TECHNOLOGY"}),
        frozenset({"TECHNOLOGY", "PRODUCT", "CONCEPT"}),
    ),
    "LOCATED_IN": (
        frozenset({"PERSON", "ORGANIZATION", "EVENT", "LOCATION", "PRODUCT"}),
        frozenset({"LOCATION"}),
    ),
    "PART_OF": (_ANY, _ANY),
    "RELATED_TO": (_ANY, _ANY),
    "AUTHORED": (
        frozenset({"PERSON"}),
        frozenset({"PRODUCT", "TECHNOLOGY", "CONCEPT", "DOCUMENT"}),
    ),
    "DEPENDS_ON": (
        frozenset({"PRODUCT", "TECHNOLOGY", "ORGANIZATION"}),
        frozenset({"PRODUCT", "TECHNOLOGY"}),
    ),
}
RELATIONSHIP_TYPES = tuple(RELATIONSHIP_RULES)
