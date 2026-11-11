"""Test doubles for answer generation: a scripted provider and a valid response factory."""

from app.schemas.graph_search import DroppedOut
from app.schemas.hybrid import GraphPathReportOut, HybridSummaryOut, PathsOut, VectorPathOut
from app.schemas.query import (
    ContextReportOut,
    GenerationInfoOut,
    GroundingOut,
    QueryResponse,
    RetrievalReportOut,
    SourceOut,
    UsageOut,
)
from app.schemas.vector import ResultSource
from app.services.generation.types import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    ProviderInfo,
    Usage,
)


class ScriptedProvider:
    """A provider that replays a script: each item is a text, a ``GenerationResult``, or an
    exception to raise. Records every request it receives. Test-only: the application's own
    mock provider arrives with M1."""

    def __init__(self, *script, name="scripted", model="scripted-1", mode=GenerationMode.MOCK):
        self.script = list(script) or ["ok"]
        self.name, self.model, self.mode = name, model, mode
        self.requests: list[GenerationRequest] = []
        self.closed = False
        self.close_error: Exception | None = None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, GenerationResult):
            return item
        return GenerationResult(
            text=item, model=self.model, usage=Usage(prompt_tokens=10, completion_tokens=3)
        )

    def info(self) -> ProviderInfo:
        return ProviderInfo(self.name, self.mode, self.model)

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def make_source(marker="S1", chunk_id="c1", cited=True, **kw) -> SourceOut:
    return SourceOut(
        marker=marker,
        chunk_id=chunk_id,
        document_id=kw.pop("document_id", "d1"),
        source=kw.pop(
            "source",
            ResultSource(filename="notes.txt", page_number=None, section="Team", chunk_index=0),
        ),
        excerpt=kw.pop("excerpt", "Alice works at Acme Inc."),
        retrieved_by=kw.pop("retrieved_by", ["vector", "graph"]),
        fused_rank=kw.pop("fused_rank", 1),
        cited=cited,
        cited_in_sentences=kw.pop("cited_in_sentences", [0] if cited else []),
        **kw,
    )


def make_response(**overrides) -> QueryResponse:
    """A valid ``answered`` response; keyword arguments replace top-level fields."""
    source = make_source()
    fields = dict(
        question="Who works at Acme Inc?",
        status="answered",
        answer="Alice works at Acme Inc. [S1]",
        citations=[source],
        sources=[source],
        grounding=GroundingOut(policy="withhold", passed=True),
        generation=GenerationInfoOut(
            mode="mock",
            provider="scripted",
            model="scripted-1",
            prompt_version="v1",
            usage=UsageOut(prompt_tokens=10, completion_tokens=3, estimated_prompt_tokens=12),
            finish_reason="stop",
            latency_ms=1.5,
            attempts=1,
        ),
        context=ContextReportOut(budget_tokens=2048, estimated_tokens=40, included=1),
        retrieval=RetrievalReportOut(
            path_summary=HybridSummaryOut(
                mode="hybrid",
                complete=True,
                is_complete_hybrid=True,
                message="Both the vector and the graph path succeeded.",
                succeeded=["vector", "graph"],
                failed=[],
                contributing=["vector", "graph"],
            ),
            paths=PathsOut(
                vector=VectorPathOut(status="ok", candidates=1, top_k=20),
                graph=GraphPathReportOut(status="ok", candidates=1, direction="both", hops=1),
            ),
            dropped=DroppedOut(missing_chunks=0, out_of_scope=0),
        ),
        timings_ms={"total": 2.0},
    )
    fields.update(overrides)
    return QueryResponse(**fields)
