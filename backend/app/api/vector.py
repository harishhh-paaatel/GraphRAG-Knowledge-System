"""Vector index health, statistics and maintenance."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_vectors
from app.core.errors import AppError
from app.services.embeddings.service import VectorIndexUnavailable, VectorService

router = APIRouter(prefix="/vector", tags=["vector"])

VectorsDep = Annotated[VectorService, Depends(get_vectors)]


@router.get("/index")
def index_stats(vectors: VectorsDep, check: bool = False) -> dict:
    """Health and statistics of the vector index.

    `check=true` additionally compares SQLite chunks, the in-memory index, the index file and
    the bookkeeping table (reads every chunk once; use for diagnostics, not polling).
    """
    stats = vectors.stats()
    if check:
        stats["consistency"] = vectors.check_consistency()
    return stats


@router.post("/rebuild")
def rebuild_index(vectors: VectorsDep) -> dict:
    """Discard the index and recompute every vector from SQLite-backed chunks (recovery tool).

    Synchronous and CPU-bound: expect it to take as long as embedding the whole corpus.
    """
    try:
        report = vectors.rebuild()
    except VectorIndexUnavailable as exc:
        raise AppError(503, "vector_index_unavailable", str(exc)) from exc
    return {"report": report.as_dict(), "index": vectors.stats()}
