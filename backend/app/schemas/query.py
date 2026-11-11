"""Request/response schemas of ``POST /api/v1/query`` and the query history (Phase 7).

Shared contract only: the endpoint itself arrives in a later milestone. Design rules encoded
here (see docs/phase-7-proposal.md section 5):

* the provider, model and base URL are NOT request fields (a caller must not be able to aim the
  server at an arbitrary URL); the request may only tighten configured caps;
* there is no confidence/score field for an answer, anywhere;
* ``answer`` is present only when ``status`` is ``answered`` (a withheld, refused or failed
  answer is never smuggled into the body);
* every citation resolves to a source with its chunk, document, page, section and evidence.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.graph_search import DroppedOut, EntityRefOut, EvidenceOut, GraphOptions
from app.schemas.hybrid import FusionOptions, HybridSummaryOut, PathsOut, VectorOptions
from app.schemas.vector import ResultSource

AnswerStatus = Literal["answered", "insufficient_evidence", "ungrounded"]
InsufficientReason = Literal["no_retrieval_results", "no_context_after_budget", "model_declined"]
ExclusionReason = Literal["budget", "overlap", "duplicate", "too_short"]
GroundingPolicy = Literal["withhold", "annotate"]

ANSWER_NOTE = (
    "Answers are grounded by heuristic checks against retrieved text; they are not verified "
    "to be correct, and no confidence score is provided."
)
EXCERPT_CHARS = 300


# --------------------------------------------------------------------------------- request


class RetrievalOptions(BaseModel):
    """Same blocks as ``/search/hybrid`` (minus the query), plus one policy flag."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=8, ge=1, description="Retrieved chunks considered (capped).")
    document_ids: list[str] | None = Field(default=None, max_length=100)
    vector: VectorOptions = Field(default_factory=VectorOptions)
    graph: GraphOptions = Field(default_factory=GraphOptions)
    fusion: FusionOptions = Field(default_factory=FusionOptions)
    require_complete_retrieval: bool = Field(
        default=False,
        description="true = refuse to answer from a partial (one-path) retrieval (503).",
    )


class GenerationOptions(BaseModel):
    """Per-request tightening of the configured generation limits."""

    model_config = ConfigDict(extra="forbid")

    context_tokens: int | None = Field(default=None, ge=128)
    max_answer_tokens: int | None = Field(default=None, ge=16)
    include_graph_facts: bool = True
    grounding_policy: GroundingPolicy | None = None


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000, examples=["Who works at Acme Inc?"])
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions)
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    include_prompt: bool = Field(
        default=False, description="Honoured only when GRAPHRAG_LLM_DEBUG_PROMPT is enabled."
    )

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


# -------------------------------------------------------------------------------- response


class SourceOut(BaseModel):
    """A passage shown to the model. ``cited`` says whether the answer used it."""

    marker: str = Field(pattern=r"^S[1-9][0-9]*$", examples=["S1"])
    chunk_id: str
    document_id: str
    source: ResultSource
    excerpt: str = Field(max_length=EXCERPT_CHARS + 1)
    retrieved_by: list[str]
    fused_rank: int = Field(ge=1)
    evidence: list[EvidenceOut] = Field(default_factory=list)
    cited: bool = False
    cited_in_sentences: list[int] = Field(default_factory=list)
    still_exists: bool = Field(
        default=True, description="false if the chunk was deleted while the answer was generated."
    )


class ConflictClaimOut(BaseModel):
    marker: str = Field(pattern=r"^S[1-9][0-9]*$")
    chunk_id: str
    document_id: str
    target: EntityRefOut
    supporting_text: str | None = None


class ConflictOut(BaseModel):
    detected_by: Literal["structural"] = "structural"
    relationship_type: str
    subject: EntityRefOut
    claims: list[ConflictClaimOut] = Field(min_length=2)


class GroundingOut(BaseModel):
    """Result of the deterministic checks. Heuristic: catches gross failures, proves nothing."""

    method: Literal["heuristic"] = "heuristic"
    policy: GroundingPolicy
    passed: bool
    uncited_sentences: list[int] = Field(default_factory=list)
    invalid_markers: list[str] = Field(default_factory=list)
    unsupported_sentences: list[int] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    retries: int = Field(default=0, ge=0)


class UsageOut(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_prompt_tokens: int | None = None


class GenerationInfoOut(BaseModel):
    mode: Literal["llm", "mock", "extractive", "none"]
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    usage: UsageOut | None = None
    finish_reason: str | None = None
    truncated: bool = False
    latency_ms: float | None = None
    attempts: int = Field(default=0, ge=0)
    fallback_from: str | None = Field(
        default=None, description="Set when the extractive fallback answered instead."
    )


class ExcludedOut(BaseModel):
    chunk_id: str
    reason: ExclusionReason


class ContextReportOut(BaseModel):
    budget_tokens: int
    estimated_tokens: int
    included: int
    excluded: list[ExcludedOut] = Field(default_factory=list)
    graph_facts: int = 0


class RetrievalReportOut(BaseModel):
    path_summary: HybridSummaryOut
    paths: PathsOut
    dropped: DroppedOut
    warnings: list[str] = Field(default_factory=list)
    unknown_document_ids: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    question: str
    status: AnswerStatus
    reason: InsufficientReason | None = None
    answer: str | None = None
    answer_note: str = ANSWER_NOTE
    citations: list[SourceOut] = Field(default_factory=list)
    sources: list[SourceOut] = Field(default_factory=list)
    conflicts: list[ConflictOut] = Field(default_factory=list)
    grounding: GroundingOut
    generation: GenerationInfoOut
    context: ContextReportOut
    retrieval: RetrievalReportOut
    history_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    prompt: str | None = Field(default=None, description="Only in debug mode.")

    @model_validator(mode="after")
    def _consistent(self) -> "QueryResponse":
        if self.status == "answered":
            if not self.answer or not self.answer.strip():
                raise ValueError("an answered response needs a non-empty answer")
            if self.reason is not None:
                raise ValueError("an answered response has no reason")
        else:
            if self.answer is not None:
                raise ValueError(f"a {self.status!r} response must not carry an answer")
        if self.status == "insufficient_evidence" and self.reason is None:
            raise ValueError("insufficient_evidence needs a reason")
        if self.status != "insufficient_evidence" and self.reason is not None:
            raise ValueError("a reason is only valid for insufficient_evidence")
        source_markers = {s.marker for s in self.sources}
        if not {c.marker for c in self.citations} <= source_markers:
            raise ValueError("every citation must also be listed in sources")
        if self.status != "answered" and self.citations:
            raise ValueError("only an answered response has citations")
        return self


# ------------------------------------------------------------------------------- history


class HistoryEntryOut(BaseModel):
    id: str
    question: str
    answer: str | None = Field(description="Null when no answer was returned.")
    status: str
    reason: str | None = None
    generation_mode: str
    retrieval_strategy: str
    execution_time_ms: float
    created_at: datetime


class HistoryDetailOut(HistoryEntryOut):
    metadata: dict


class HistoryListOut(BaseModel):
    items: list[HistoryEntryOut]
    total: int
    limit: int
    offset: int
