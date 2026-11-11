"""Hybrid retrieval API schemas (Phase 6, M3)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.graph_search import (
    AmbiguousOut,
    DroppedOut,
    ExplanationOut,
    GraphOptions,
    LinkedEntityOut,
    SearchResultOut,
    SuppressedOut,
)


class VectorOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=20, ge=1, description="Vector candidates kept before fusion.")
    similarity_threshold: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Minimum cosine similarity (inclusive)."
    )


class FusionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["rrf"] = "rrf"
    rrf_k: int | None = Field(
        default=None, ge=1, le=1000, description="Default: GRAPHRAG_HYBRID_RRF_K (60)."
    )
    vector_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    graph_weight: float = Field(default=1.0, ge=0.0, le=10.0)

    @model_validator(mode="after")
    def _one_positive_weight(self) -> "FusionOptions":
        if self.vector_weight == 0 and self.graph_weight == 0:
            raise ValueError("at least one weight must be greater than 0")
        return self


class HybridSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000, examples=["Who works at Acme Corp?"])
    limit: int = Field(default=10, ge=1, description="Fused results returned (capped).")
    document_ids: list[str] | None = Field(default=None, max_length=100)
    vector: VectorOptions = Field(default_factory=VectorOptions)
    graph: GraphOptions = Field(default_factory=GraphOptions)
    fusion: FusionOptions = Field(default_factory=FusionOptions)
    explain: bool = False

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


# ----------------------------------------------------------------------------- response


class FusionTermOut(BaseModel):
    path: Literal["vector", "graph"]
    rank: int
    weight: float
    contribution: float


class HybridExplanationOut(ExplanationOut):
    fusion: list[FusionTermOut]
    graph_ranking: str | None = Field(
        description="Templated description of the graph features, when the graph found the chunk."
    )


class HybridResultOut(SearchResultOut):
    explanation: HybridExplanationOut | None = None


class VectorPathOut(BaseModel):
    status: Literal["ok", "stale", "building", "unavailable"]
    candidates: int
    error_code: str | None = None
    error_message: str | None = None
    top_k: int
    similarity_threshold: float | None = None
    index_stale: bool | None = None
    index_revision: int | None = None
    database_revision: int | None = None
    embedding_model: str | None = None
    embedding_model_revision: str | None = None
    independent_of_graph: bool = True


class GraphPathReportOut(BaseModel):
    status: Literal["ok", "no_entities", "ambiguous_only", "unavailable"]
    candidates: int
    note: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    graph_revision: int | None = None
    database_revision: int | None = None
    direction: str
    hops: int
    relationship_types: list[str] | None = None
    truncated: list[str] = Field(default_factory=list)
    tie_breaking: str = "chunk_id"
    independent_of_vectors: bool = True


class PathsOut(BaseModel):
    vector: VectorPathOut
    graph: GraphPathReportOut


class HybridSummaryOut(BaseModel):
    mode: Literal["hybrid", "vector_only", "graph_only", "partial", "failed"]
    complete: bool
    is_complete_hybrid: bool
    message: str
    succeeded: list[str]
    failed: list[str]
    contributing: list[str] = Field(
        description="Paths that ran successfully and actually returned results."
    )


class FusionOut(BaseModel):
    method: str
    rrf_k: int
    vector_weight: float
    graph_weight: float
    score_meaning: str


class HybridSearchResponse(BaseModel):
    query: str
    results: list[HybridResultOut]
    linked_entities: list[LinkedEntityOut]
    ambiguous_entities: list[AmbiguousOut]
    suppressed_spans: list[SuppressedOut] | None = None  # only with explain=true
    paths: PathsOut
    path_summary: HybridSummaryOut
    fusion: FusionOut
    limit: int
    candidates_fused: int = Field(description="Distinct chunks before the limit was applied.")
    dropped: DroppedOut
    unknown_document_ids: list[str]
    warnings: list[str]
    timings_ms: dict[str, float]
