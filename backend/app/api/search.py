"""Search endpoints: semantic (vector) search and graph-only search."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_graph_retriever, get_settings_dep, get_vectors
from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.graph_search import (
    AmbiguousOut,
    CandidateOut,
    DroppedOut,
    EntityRefOut,
    EvidenceOut,
    ExplanationOut,
    FeaturesOut,
    GraphBlock,
    GraphPathOut,
    GraphSearchRequest,
    GraphSearchResponse,
    LinkedEntityOut,
    PathSummaryOut,
    SearchResultOut,
    SuppressedOut,
)
from app.schemas.vector import (
    ResultSource,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorSearchResult,
)
from app.services.embeddings.embedder import EmbeddingError, EmbeddingInputError
from app.services.embeddings.service import VectorIndexBuilding, VectorService
from app.services.embeddings.vector_index import VectorIndexError
from app.services.graph.service import GraphUnavailable
from app.services.retrieval.entity_linker import UnknownEntities
from app.services.retrieval.graph_search import (
    RANK_KEYS,
    EntityRef,
    EvidenceItem,
    GraphHit,
    GraphRetriever,
    GraphSearchLimitError,
    GraphSearchOutcome,
)
from app.services.retrieval.path_status import classify_paths
from app.services.retrieval.vector_search import (
    SIMILARITY_DECIMALS,
    SIMILARITY_METRIC,
    search_vectors,
)

router = APIRouter(prefix="/search", tags=["search"])

VectorsDep = Annotated[VectorService, Depends(get_vectors)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
GraphRetrieverDep = Annotated[GraphRetriever, Depends(get_graph_retriever)]


@router.post("/vector", response_model=VectorSearchResponse)
def vector_search(
    body: VectorSearchRequest, vectors: VectorsDep, settings: SettingsDep
) -> VectorSearchResponse:
    """Semantic search over document chunks (exact cosine similarity, FAISS).

    `similarity` is retrieval similarity between the query and a chunk, NOT answer confidence.
    It is rounded to 6 decimals and `similarity_threshold` (inclusive) is applied to that same
    rounded value. Only chunks of completed documents are searched; `document_ids` restricts the
    search (unknown IDs are reported, not errors). An empty index returns an empty result list.

    While an ingestion is synchronizing the index, searches keep using the previous consistent
    index (`index_stale: true`). 503 `vector_index_building` means no usable index exists yet
    (or the embedding configuration changed and the index is being rebuilt): retry shortly.
    """
    if body.top_k > settings.vector_max_top_k:
        raise AppError(422, "limit_exceeded", f"top_k must be <= {settings.vector_max_top_k}")
    try:
        outcome = search_vectors(
            vectors,
            vectors.session_factory,
            body.query,
            top_k=body.top_k,
            similarity_threshold=body.similarity_threshold,
            document_ids=body.document_ids,
        )
    except EmbeddingInputError as exc:
        raise AppError(422, "invalid_query", str(exc)) from exc
    except VectorIndexBuilding as exc:
        raise AppError(503, "vector_index_building", str(exc)) from exc
    except (EmbeddingError, VectorIndexError) as exc:
        raise AppError(503, "vector_index_unavailable", str(exc)) from exc

    return VectorSearchResponse(
        query=outcome.query,
        results=[
            VectorSearchResult(
                rank=h.rank,
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                content=h.content,
                similarity=h.similarity,
                source=ResultSource(
                    filename=h.filename,
                    page_number=h.page_number,
                    section=h.section,
                    chunk_index=h.chunk_index,
                ),
                embedding_model=h.embedding_model,
            )
            for h in outcome.results
        ],
        embedding_model=outcome.embedding_model,
        embedding_model_revision=outcome.embedding_model_revision,
        similarity_metric=SIMILARITY_METRIC,
        similarity_decimals=SIMILARITY_DECIMALS,
        top_k=body.top_k,
        similarity_threshold=body.similarity_threshold,
        indexed_vectors=outcome.indexed_vectors,
        index_revision=outcome.index_revision,
        database_revision=outcome.database_revision,
        index_stale=outcome.index_stale,
        max_tokens=outcome.max_tokens,
        query_token_count=outcome.query_token_count,
        query_windows=outcome.query_windows,
        unknown_document_ids=outcome.unknown_document_ids,
        timings_ms=outcome.timings_ms,
    )


# ------------------------------------------------------------------------- graph search


def _ref(ref: EntityRef | None) -> EntityRefOut | None:
    if ref is None:
        return None
    return EntityRefOut(entity_id=ref.entity_id, name=ref.name, type=ref.entity_type)


def _evidence(item: EvidenceItem) -> EvidenceOut:
    return EvidenceOut(
        kind=item.kind,
        document_id=item.document_id,
        provenance=item.provenance,
        confidence=item.confidence,
        relationship_type=item.relationship_type,
        source=_ref(item.source),
        target=_ref(item.target),
        entity=_ref(item.entity),
        traversal=item.traversal,
        reached_from=_ref(item.reached_from),
        distance=item.distance,
        edge_id=item.edge_id,
        relationship_id=item.relationship_id,
        supporting_text=item.supporting_text,
        surface_form=item.surface_form,
    )


def _explanation(hit: GraphHit) -> ExplanationOut:
    """A templated, deterministic description of the ranking features (no generated text)."""
    f = hit.features
    kinds = sorted({e.relationship_type for e in hit.evidence if e.relationship_type})
    parts = [
        f"touches {f.seed_coverage} of the query entities",
        (
            f"evidences {f.support} distinct relationship(s) ({', '.join(kinds)}), "
            f"nearest at distance {f.distance}"
            if f.relationship_evidence
            else "only mentions a query entity (no relationship evidence)"
        ),
    ]
    if f.connects_seeds:
        parts.append("connects two query entities")
    if f.corroboration > 1:
        parts.append(f"a relationship here is supported by {f.corroboration} documents")
    if hit.decided_by is None:
        order = "ranked first"
    elif hit.decided_by == "chunk_id":
        order = "ranked below the previous chunk by the stable chunk id (all earlier features tied)"
    else:
        order = f"ranked below the previous chunk because of '{hit.decided_by}'"
    return ExplanationOut(
        summary="; ".join(parts) + "; " + order + ".", ranking_keys=list(RANK_KEYS)
    )


def _graph_block(hit: GraphHit) -> GraphBlock:
    f = hit.features
    return GraphBlock(
        rank=hit.rank,
        features=FeaturesOut(
            seed_coverage=f.seed_coverage,
            connects_seeds=f.connects_seeds,
            relationship_evidence=f.relationship_evidence,
            distance=f.distance,
            support=f.support,
            corroboration=f.corroboration,
            provenance_known=f.provenance_known,
            decided_by=hit.decided_by,
        ),
        evidence=[_evidence(e) for e in hit.evidence],
    )


def _result(hit: GraphHit, explain: bool) -> SearchResultOut:
    return SearchResultOut(
        rank=hit.rank,
        chunk_id=hit.chunk.chunk_id,
        document_id=hit.chunk.document_id,
        content=hit.chunk.content,
        source=ResultSource(
            filename=hit.chunk.filename,
            page_number=hit.chunk.page_number,
            section=hit.chunk.section,
            chunk_index=hit.chunk.chunk_index,
        ),
        retrieved_by=["graph"],
        graph=_graph_block(hit),
        explanation=_explanation(hit) if explain else None,
    )


def _graph_response(outcome: GraphSearchOutcome, limit: int, explain: bool) -> GraphSearchResponse:
    warnings: list[str] = []
    if outcome.unknown_document_ids:
        warnings.append("some document_ids do not exist and were ignored")
    if outcome.truncated:
        warnings.append("traversal limits were reached: " + ", ".join(outcome.truncated))
    if outcome.dropped_missing_chunks:
        warnings.append(
            f"{outcome.dropped_missing_chunks} evidence chunk(s) no longer exist and were dropped"
        )
    if outcome.dropped_stale_evidence:
        warnings.append(
            f"{outcome.dropped_stale_evidence} relationship evidence item(s) are no longer "
            "confirmed by the database and were dropped"
        )
    summary = classify_paths("skipped", "ok")
    link = outcome.link
    return GraphSearchResponse(
        query=outcome.query,
        results=[_result(hit, explain) for hit in outcome.hits[:limit]],
        linked_entities=[
            LinkedEntityOut(
                span=e.span,
                entity_id=e.entity_id,
                name=e.name,
                type=e.entity_type,
                match_type=e.match_type,
                resolved_by=e.resolved_by,
                edge_count=outcome.entity_degree.get(e.entity_id, 0),
            )
            for e in link.linked
        ],
        ambiguous_entities=[
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
        ],
        suppressed_spans=(
            [SuppressedOut(span=s.span, contained_in=s.contained_in) for s in link.suppressed]
            if explain
            else None
        ),
        graph=GraphPathOut(
            status=outcome.status,
            note=outcome.note,
            candidates=outcome.ranked_total,
            graph_revision=outcome.revision,
            database_revision=outcome.database_revision,
            direction=outcome.direction,
            hops=outcome.hops,
            relationship_types=outcome.relationship_types,
            truncated=outcome.truncated,
        ),
        path_summary=PathSummaryOut(
            mode=summary.mode, complete=summary.complete, message=summary.message
        ),
        limit=limit,
        dropped=DroppedOut(
            missing_chunks=outcome.dropped_missing_chunks,
            out_of_scope=outcome.dropped_out_of_scope,
            stale_evidence=outcome.dropped_stale_evidence,
        ),
        unknown_document_ids=outcome.unknown_document_ids,
        warnings=warnings,
        timings_ms=outcome.timings_ms,
    )


@router.post("/graph", response_model=GraphSearchResponse)
def graph_search(
    body: GraphSearchRequest, retriever: GraphRetrieverDep, settings: SettingsDep
) -> GraphSearchResponse:
    """Graph-only retrieval: chunks that evidence relationships of the entities the query names.

    Needs only the knowledge graph and SQLite: it works while the vector index, FAISS and the
    embedding model are unavailable. Entities are linked by exact name/alias matching on token
    boundaries; ambiguous names are reported, never guessed (pass `graph.entity_ids` to choose).
    Every relationship is returned in its stored direction whichever `graph.direction` is used.
    Results are ordered by integer graph features, not by similarity: the order is deterministic,
    not a relevance or confidence score.
    """
    if body.limit > settings.hybrid_max_limit:
        raise AppError(422, "limit_exceeded", f"limit must be <= {settings.hybrid_max_limit}")
    g = body.graph
    try:
        outcome = retriever.search(
            body.query,
            document_ids=body.document_ids,
            hops=g.hops,
            direction=g.direction,
            relationship_types=g.relationship_types,
            max_chunks=g.max_chunks,
            include_mentions=g.include_mentions,
            entity_ids=g.entity_ids,
            link_entities=g.link_entities,
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
    except GraphUnavailable as exc:
        raise AppError(503, "graph_unavailable", str(exc)) from exc
    return _graph_response(outcome, body.limit, body.explain)
