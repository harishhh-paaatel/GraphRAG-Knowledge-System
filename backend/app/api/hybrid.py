"""POST /search/hybrid: vector retrieval and graph retrieval, fused by rank."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_hybrid, get_settings_dep
from app.api.search import _explanation, _graph_block
from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.graph_search import (
    AmbiguousOut,
    CandidateOut,
    DroppedOut,
    LinkedEntityOut,
    SuppressedOut,
    VectorBlock,
)
from app.schemas.hybrid import (
    FusionOut,
    FusionTermOut,
    GraphPathReportOut,
    HybridExplanationOut,
    HybridResultOut,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSummaryOut,
    PathsOut,
    VectorPathOut,
)
from app.schemas.vector import ResultSource
from app.services.embeddings.errors import EmbeddingInputError
from app.services.retrieval.entity_linker import LinkResult, UnknownEntities
from app.services.retrieval.fusion import FusionParams, rrf_contribution
from app.services.retrieval.graph_search import RANK_KEYS, GraphSearchLimitError
from app.services.retrieval.hybrid import (
    SCORE_MEANING,
    HybridOutcome,
    HybridResult,
    HybridRetriever,
    PathReport,
    RetrievalUnavailable,
)

router = APIRouter(prefix="/search", tags=["search"])

HybridDep = Annotated[HybridRetriever, Depends(get_hybrid)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


def _explanation_for(result: HybridResult, params: FusionParams) -> HybridExplanationOut:
    """A templated, deterministic account of why this chunk is where it is (no generated text)."""
    item = result.fused
    terms: list[FusionTermOut] = []
    if item.vector_rank is not None:
        terms.append(
            FusionTermOut(
                path="vector",
                rank=item.vector_rank,
                weight=params.vector_weight,
                contribution=rrf_contribution(params.vector_weight, params.rrf_k, item.vector_rank),
            )
        )
    if item.graph_rank is not None:
        terms.append(
            FusionTermOut(
                path="graph",
                rank=item.graph_rank,
                weight=params.graph_weight,
                contribution=rrf_contribution(params.graph_weight, params.rrf_k, item.graph_rank),
            )
        )
    found = " and ".join(f"the {t.path} path (rank {t.rank})" for t in terms)
    summary = (
        f"Retrieved by {found}; fused score {item.display_score} = "
        + " + ".join(f"{t.weight:g}/({params.rrf_k}+{t.rank})" for t in terms)
        + f"; fused rank {result.rank}."
    )
    if result.vector is not None:
        summary += f" Vector similarity {result.vector.similarity}."
    return HybridExplanationOut(
        summary=summary,
        ranking_keys=list(RANK_KEYS),
        fusion=terms,
        graph_ranking=_explanation(result.graph).summary if result.graph is not None else None,
    )


def _result(result: HybridResult, params: FusionParams, explain: bool) -> HybridResultOut:
    return HybridResultOut(
        rank=result.rank,
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        content=result.content,
        source=ResultSource(
            filename=result.filename,
            page_number=result.page_number,
            section=result.section,
            chunk_index=result.chunk_index,
        ),
        retrieved_by=list(result.retrieved_by),
        fused_score=result.fused.display_score,
        vector=(
            VectorBlock(rank=result.vector.rank, similarity=result.vector.similarity)
            if result.vector is not None
            else None
        ),
        graph=_graph_block(result.graph) if result.graph is not None else None,
        explanation=_explanation_for(result, params) if explain else None,
    )


def _link_parts(link: LinkResult | None, edge_degree: dict[str, int], explain: bool):
    if link is None:
        return [], [], None
    linked = [
        LinkedEntityOut(
            span=e.span,
            entity_id=e.entity_id,
            name=e.name,
            type=e.entity_type,
            match_type=e.match_type,
            resolved_by=e.resolved_by,
            edge_count=edge_degree.get(e.entity_id, 0),
        )
        for e in link.linked
    ]
    ambiguous = [
        AmbiguousOut(
            span=a.span,
            reason=a.reason,
            spans=list(a.spans),
            candidates=[
                CandidateOut(
                    entity_id=c.entity_id,
                    name=c.name,
                    type=c.entity_type,
                    match_type=c.match_type,
                    document_count=c.document_count,
                )
                for c in a.candidates
            ],
            action=a.action,
            resolved_to=list(a.resolved_to),
        )
        for a in link.ambiguous
    ]
    suppressed = (
        [SuppressedOut(span=s.span, contained_in=s.contained_in) for s in link.suppressed]
        if explain
        else None
    )
    return linked, ambiguous, suppressed


def _vector_path(outcome: HybridOutcome, top_k: int, threshold: float | None) -> VectorPathOut:
    report: PathReport = outcome.vector_report
    base = {
        "status": report.status,
        "candidates": report.candidates,
        "error_code": report.error_code,
        "error_message": report.error_message,
        "top_k": top_k,
        "similarity_threshold": threshold,
    }
    v = outcome.vector_outcome
    if v is None:
        return VectorPathOut(**base)
    return VectorPathOut(
        **base,
        index_stale=v.index_stale,
        index_revision=v.index_revision,
        database_revision=v.database_revision,
        embedding_model=v.embedding_model,
        embedding_model_revision=v.embedding_model_revision,
    )


def _graph_path(outcome: HybridOutcome, request: HybridSearchRequest) -> GraphPathReportOut:
    report = outcome.graph_report
    g = outcome.graph_outcome
    if g is None:
        return GraphPathReportOut(
            status="unavailable",
            candidates=0,
            error_code=report.error_code,
            error_message=report.error_message,
            direction=request.graph.direction,
            hops=request.graph.hops,
            relationship_types=request.graph.relationship_types,
        )
    return GraphPathReportOut(
        status=g.status,
        candidates=report.candidates,
        note=g.note,
        graph_revision=g.revision,
        database_revision=g.database_revision,
        direction=g.direction,
        hops=g.hops,
        relationship_types=g.relationship_types,
        truncated=g.truncated,
    )


def _response(outcome: HybridOutcome, request: HybridSearchRequest) -> HybridSearchResponse:
    g = outcome.graph_outcome
    linked, ambiguous, suppressed = _link_parts(
        g.link if g else None, g.entity_degree if g else {}, request.explain
    )
    summary = outcome.summary
    params = outcome.params
    return HybridSearchResponse(
        query=outcome.query,
        results=[_result(r, params, request.explain) for r in outcome.results],
        linked_entities=linked,
        ambiguous_entities=ambiguous,
        suppressed_spans=suppressed,
        paths=PathsOut(
            vector=_vector_path(outcome, request.vector.top_k, request.vector.similarity_threshold),
            graph=_graph_path(outcome, request),
        ),
        path_summary=HybridSummaryOut(
            mode=summary.mode,
            complete=summary.complete,
            is_complete_hybrid=summary.is_complete_hybrid,
            message=summary.message,
            succeeded=list(summary.succeeded),
            failed=list(summary.failed),
            contributing=list(outcome.contributing),
        ),
        fusion=FusionOut(
            method="rrf",
            rrf_k=params.rrf_k,
            vector_weight=params.vector_weight,
            graph_weight=params.graph_weight,
            score_meaning=SCORE_MEANING,
        ),
        limit=request.limit,
        candidates_fused=len(outcome.fused),
        dropped=DroppedOut(
            missing_chunks=g.dropped_missing_chunks if g else 0,
            out_of_scope=g.dropped_out_of_scope if g else 0,
            stale_evidence=g.dropped_stale_evidence if g else 0,
            stale_results=outcome.dropped_stale_results,
        ),
        unknown_document_ids=outcome.unknown_document_ids,
        warnings=outcome.warnings,
        timings_ms=outcome.timings_ms,
    )


@router.post("/hybrid", response_model=HybridSearchResponse)
def hybrid_search(
    body: HybridSearchRequest, hybrid: HybridDep, settings: SettingsDep
) -> HybridSearchResponse:
    """Hybrid retrieval: semantic (vector) search and knowledge-graph search, fused by rank.

    The two paths run independently and are combined with weighted Reciprocal Rank Fusion, which
    uses only each path's ranking (never its scores). Every result keeps its original vector rank
    and similarity, its original graph rank, features and relationship evidence, and says which
    path(s) found it. `fused_score` is a rank-based retrieval score, not a probability or answer
    confidence.

    `path_summary` states what actually happened. If one path failed the response is HTTP 200 with
    mode `partial`, `complete: false` and a warning: it is NOT a complete hybrid result. If both
    paths fail the response is 503 `retrieval_unavailable` with each path's error code.
    """
    if body.limit > settings.hybrid_max_limit:
        raise AppError(422, "limit_exceeded", f"limit must be <= {settings.hybrid_max_limit}")
    if body.vector.top_k > settings.vector_max_top_k:
        raise AppError(
            422, "limit_exceeded", f"vector.top_k must be <= {settings.vector_max_top_k}"
        )
    g = body.graph
    f = body.fusion
    try:
        outcome = hybrid.search(
            body.query,
            limit=body.limit,
            document_ids=body.document_ids,
            vector_top_k=body.vector.top_k,
            similarity_threshold=body.vector.similarity_threshold,
            hops=g.hops,
            direction=g.direction,
            relationship_types=g.relationship_types,
            max_chunks=g.max_chunks,
            include_mentions=g.include_mentions,
            entity_ids=g.entity_ids,
            link_entities=g.link_entities,
            fusion=FusionParams(
                rrf_k=f.rrf_k if f.rrf_k is not None else settings.hybrid_rrf_k,
                vector_weight=f.vector_weight,
                graph_weight=f.graph_weight,
            ),
        )
    except GraphSearchLimitError as exc:
        raise AppError(422, "limit_exceeded", str(exc)) from exc
    except UnknownEntities as exc:
        raise AppError(
            422,
            "entity_not_found",
            "Some entity_ids are not in the knowledge graph",
            {"unknown_entity_ids": exc.entity_ids},
        ) from exc
    except EmbeddingInputError as exc:
        raise AppError(422, "invalid_query", str(exc)) from exc
    except RetrievalUnavailable as exc:
        raise AppError(
            503,
            "retrieval_unavailable",
            "Neither the vector path nor the graph path is available",
            {
                name: {
                    "status": r.status,
                    "error_code": r.error_code,
                    "error_message": r.error_message,
                }
                for name, r in exc.reports.items()
            },
        ) from exc
    return _response(outcome, body)
