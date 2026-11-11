"""Semantic (vector) retrieval over the vector index.

Pipeline: query -> embedding -> normalization -> index search -> chunk-ID resolution ->
SQLite metadata lookup -> ranked results.

The whole query reads ONE immutable index snapshot, so the ID table, vectors and metadata it
uses always belong together even if a sync publishes a newer snapshot meanwhile.

Similarity policy
-----------------
Scores are *retrieval similarity*: the cosine between the query vector and a chunk vector.
They rank candidates; they are not probabilities and not answer confidence. A score is rounded
to ``SIMILARITY_DECIMALS`` decimals once, and that rounded value is used for BOTH filtering and
presentation (the threshold is rounded the same way), so a result that displays as ``0.800000``
is never rejected by a ``0.8`` threshold. Ranking uses the exact scores, so rounding can create
equal displayed values but never reorders results.

Only chunks of ``completed`` documents are searchable: a document that is still processing
or failed is not yet (or no longer) a trustworthy source.
"""

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Document, DocumentChunk
from app.services.embeddings.service import VectorService
from app.services.retrieval.scope import DocumentScope, resolve_document_scope

logger = logging.getLogger(__name__)

SIMILARITY_DECIMALS = 6
SIMILARITY_METRIC = (
    "cosine similarity between query and chunk embeddings (retrieval similarity, not answer "
    f"confidence), rounded to {SIMILARITY_DECIMALS} decimals; the threshold applies to the "
    "rounded value"
)


def round_similarity(score: float) -> float:
    """The single precision used for filtering and for display."""
    return round(float(score), SIMILARITY_DECIMALS)


def passes_threshold(score: float, threshold: float | None) -> bool:
    """Inclusive threshold test on the same rounded values the caller sees."""
    return threshold is None or round_similarity(score) >= round_similarity(threshold)


@dataclass(frozen=True)
class VectorHit:
    rank: int
    chunk_id: str
    document_id: str
    content: str
    similarity: float  # already rounded by ``round_similarity``
    filename: str
    page_number: int | None
    section: str | None
    chunk_index: int
    embedding_model: str


@dataclass
class VectorSearchOutcome:
    query: str
    results: list[VectorHit]
    embedding_model: str
    embedding_model_revision: str | None
    index_revision: int | None
    database_revision: int | None
    index_stale: bool
    indexed_vectors: int
    max_tokens: int | None = None
    query_token_count: int | None = None
    query_windows: int | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    unknown_document_ids: list[str] = field(default_factory=list)


def search_vectors(
    vectors: VectorService,
    factory: sessionmaker[Session],
    query: str,
    *,
    top_k: int = 5,
    similarity_threshold: float | None = None,
    document_ids: list[str] | None = None,
    scope: DocumentScope | None = None,
) -> VectorSearchOutcome:
    """Run a semantic search. Raises VectorIndexError / EmbeddingError on backend failure.

    ``scope`` lets a caller that already resolved the document scope (the hybrid orchestrator)
    share it, so every path of one request uses the same scope; by default it is resolved here.
    """
    t_start = time.perf_counter()
    query = query.strip()
    view = vectors.ensure_fresh()
    snap = view.snapshot  # everything below uses this one snapshot

    t0 = time.perf_counter()
    info = vectors.embedder.describe([query])[0]
    query_vector = vectors.embedder.embed([query])[0]
    embed_ms = (time.perf_counter() - t0) * 1000

    meta = snap.meta
    outcome = VectorSearchOutcome(
        query=query,
        results=[],
        embedding_model=meta.model_id,
        embedding_model_revision=meta.model_revision,
        index_revision=meta.revision,
        database_revision=view.database_revision,
        index_stale=view.stale,
        indexed_vectors=snap.count,
        max_tokens=meta.max_tokens,
        query_token_count=info.token_count,
        query_windows=info.windows,
    )

    with factory() as session:
        if scope is None:
            scope = resolve_document_scope(session, document_ids)
        outcome.unknown_document_ids = list(scope.unknown)

        t1 = time.perf_counter()
        hits = snap.search(
            query_vector,
            top_k,
            document_ids=None if scope.allowed is None else set(scope.allowed),
            exclude_document_ids=set(scope.not_ready),
        )
        search_ms = (time.perf_counter() - t1) * 1000
        hits = [h for h in hits if passes_threshold(h.score, similarity_threshold)]

        t2 = time.perf_counter()
        rows = {}
        if hits:
            rows = {
                chunk.id: (chunk, filename)
                for chunk, filename in session.execute(
                    select(DocumentChunk, Document.filename)
                    .join(Document, Document.id == DocumentChunk.document_id)
                    .where(DocumentChunk.id.in_([h.entry.chunk_id for h in hits]))
                )
            }
        lookup_ms = (time.perf_counter() - t2) * 1000

    for hit in hits:
        row = rows.get(hit.entry.chunk_id)
        if row is None:  # chunk deleted after the snapshot was built; never invent its metadata
            logger.warning("Dropping search hit for vanished chunk %s", hit.entry.chunk_id)
            continue
        chunk, filename = row
        outcome.results.append(
            VectorHit(
                rank=len(outcome.results) + 1,
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                similarity=round_similarity(hit.score),
                filename=filename,
                page_number=chunk.page_number,
                section=chunk.section,
                chunk_index=chunk.chunk_index,
                embedding_model=meta.model_id,
            )
        )
    outcome.timings_ms = {
        "embed_query": round(embed_ms, 3),
        "faiss_search": round(search_ms, 3),
        "metadata_lookup": round(lookup_ms, 3),
        "total": round((time.perf_counter() - t_start) * 1000, 3),
    }
    return outcome
