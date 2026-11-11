"""Graph-only retrieval API schemas (Phase 6, M2).

The result shape is the one the hybrid endpoint will reuse (M3): ``vector`` and ``fused_score``
exist already and are always null here because this endpoint never runs the vector path.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.vector import ResultSource
from app.services.extraction.types import RELATIONSHIP_TYPES

_DIRECTION_ALIASES = {"out": "outgoing", "in": "incoming"}


class GraphOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hops: int = Field(default=1, ge=1, description="Hops walked from each entity (capped).")
    direction: Literal["outgoing", "incoming", "both"] = Field(
        default="both",
        description=(
            "Relative to the entity being expanded: outgoing follows edges whose source it is, "
            "incoming follows edges whose target it is. Stored directions are always returned "
            "unchanged. `out` / `in` are accepted as aliases."
        ),
    )
    relationship_types: list[str] | None = Field(default=None, max_length=32)
    max_chunks: int = Field(default=20, ge=1, description="Candidate chunks kept after ranking.")
    include_mentions: bool = Field(
        default=False, description="Also return chunks that merely mention a linked entity."
    )
    link_entities: bool = Field(
        default=True, description="false = use only the explicit entity_ids."
    )
    entity_ids: list[str] | None = Field(
        default=None,
        max_length=50,
        description="Entities to expand regardless of the query text; also resolves ambiguity.",
    )

    @field_validator("direction", mode="before")
    @classmethod
    def _alias(cls, value: object) -> object:
        return _DIRECTION_ALIASES.get(value, value) if isinstance(value, str) else value

    @field_validator("relationship_types")
    @classmethod
    def _known_types(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        unknown = sorted(set(value) - set(RELATIONSHIP_TYPES))
        if unknown:
            raise ValueError(f"unknown relationship types: {', '.join(unknown)}")
        return sorted(set(value))


class GraphSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000, examples=["Who works at Acme Corp?"])
    limit: int = Field(default=10, ge=1, description="Chunks returned (capped).")
    document_ids: list[str] | None = Field(default=None, max_length=100)
    graph: GraphOptions = Field(default_factory=GraphOptions)
    explain: bool = False

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


# ----------------------------------------------------------------------------- response


class EntityRefOut(BaseModel):
    entity_id: str
    name: str
    type: str


class EvidenceOut(BaseModel):
    kind: Literal["relationship", "mention"]
    document_id: str
    provenance: str
    confidence: float | None  # None unless an extractor genuinely reported one
    relationship_type: str | None = None
    source: EntityRefOut | None = None  # stored direction: source -> target
    target: EntityRefOut | None = None
    entity: EntityRefOut | None = None  # mention evidence
    traversal: str | None = None  # outgoing | incoming: how the walk reached this edge
    reached_from: EntityRefOut | None = None
    distance: int | None = None
    edge_id: str | None = None
    relationship_id: str | None = None
    supporting_text: str | None = None
    surface_form: str | None = None


class FeaturesOut(BaseModel):
    seed_coverage: int
    connects_seeds: bool
    relationship_evidence: bool
    distance: int
    support: int
    corroboration: int
    provenance_known: bool
    decided_by: str | None = Field(
        description=(
            "First ranking key on which this chunk differs from the one ranked above it; "
            "'chunk_id' means every earlier key tied. Null for rank 1."
        )
    )


class GraphBlock(BaseModel):
    rank: int
    features: FeaturesOut
    evidence: list[EvidenceOut]


class VectorBlock(BaseModel):  # populated by the hybrid endpoint (M3); null here
    rank: int
    similarity: float


class ExplanationOut(BaseModel):
    summary: str
    ranking_keys: list[str]


class SearchResultOut(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    content: str
    source: ResultSource
    retrieved_by: list[str]
    fused_score: float | None = None
    vector: VectorBlock | None = None
    graph: GraphBlock | None
    explanation: ExplanationOut | None = None


class LinkedEntityOut(BaseModel):
    span: str | None
    entity_id: str
    name: str
    type: str
    match_type: str | None
    resolved_by: Literal["query", "entity_ids"]
    edge_count: int


class CandidateOut(BaseModel):
    entity_id: str
    name: str
    type: str
    match_type: str
    document_count: int


class AmbiguousOut(BaseModel):
    span: str
    reason: Literal["multiple_entities", "overlapping_spans"]
    spans: list[str]
    candidates: list[CandidateOut]
    action: Literal["skipped", "resolved"]
    resolved_to: list[str]
    resolve_with: str = "graph.entity_ids"


class SuppressedOut(BaseModel):
    span: str
    contained_in: str


class GraphPathOut(BaseModel):
    status: Literal["ok", "no_entities", "ambiguous_only"]
    note: str | None
    candidates: int = Field(description="Ranked candidate chunks before max_chunks/limit.")
    graph_revision: int
    database_revision: int | None
    direction: str
    hops: int
    relationship_types: list[str] | None
    truncated: list[str]
    tie_breaking: str = "chunk_id"
    independent_of_vectors: bool = True


class PathSummaryOut(BaseModel):
    mode: str
    complete: bool
    message: str


class DroppedOut(BaseModel):
    missing_chunks: int
    out_of_scope: int
    stale_evidence: int = Field(
        default=0,
        description="Graph relationship evidence the database no longer confirms (dropped).",
    )
    stale_results: int = Field(
        default=0,
        description="Fused results whose chunk vanished or left scope before the response.",
    )


class GraphSearchResponse(BaseModel):
    query: str
    results: list[SearchResultOut]
    linked_entities: list[LinkedEntityOut]
    ambiguous_entities: list[AmbiguousOut]
    suppressed_spans: list[SuppressedOut] | None = None  # only with explain=true
    graph: GraphPathOut
    path_summary: PathSummaryOut
    limit: int
    dropped: DroppedOut
    unknown_document_ids: list[str]
    warnings: list[str]
    ranking_note: str = (
        "Results are ordered by integer graph features (see graph.features and decided_by). "
        "The order is deterministic and reproducible; it is not a relevance or confidence score."
    )
    timings_ms: dict[str, float]
