"""The ingestion pipeline: parse -> chunk -> persist chunks -> extract knowledge -> index vectors.

Progress is tracked on a ProcessingJob; the document only becomes ``completed`` once its
knowledge (entities, mentions, relationships) is stored and its chunks are in the
vector index. A vector-indexing failure does NOT fail the document: it is recorded in the
document metadata and repaired by the next sync or rebuild."""

import logging
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.models import ChunkEmbedding, Document, DocumentChunk, ProcessingJob
from app.models.base import utcnow
from app.services.embeddings.revision import bump_vector_revision
from app.services.embeddings.service import VectorService
from app.services.extraction.factory import build_extraction_provider, run_provider
from app.services.extraction.persistence import (
    WRITE_LOCK,
    purge_document_knowledge,
    refresh_entities,
    replace_document_knowledge,
)
from app.services.extraction.types import ExtractionProvider
from app.services.graph.service import GraphService
from app.services.ingestion.chunker import chunk_segments
from app.services.ingestion.errors import IngestionError
from app.services.ingestion.parsers import parse_file

logger = logging.getLogger(__name__)

_INTERNAL_ERROR = "Internal processing error (see server logs)"


def _set_stage(factory: sessionmaker[Session], job_id: str, stage: str) -> None:
    with factory() as session:
        job = session.get(ProcessingJob, job_id)
        if job is not None:
            job.stage = stage
            session.commit()


def _mark_failed(factory: sessionmaker[Session], job_id: str, message: str) -> None:
    with factory() as session:
        job = session.get(ProcessingJob, job_id)
        if job is None:
            return
        job.status = "failed"
        job.error_message = message[:1000]
        job.completed_at = utcnow()
        if job.document is not None:
            job.document.processing_status = "failed"
        session.commit()


def _sync_graph(graph: GraphService | None) -> None:
    """Refresh the derived graph. Never fails the job: SQLite already holds the truth and
    the next graph read (or startup) repairs any gap."""
    if graph is None:
        return
    try:
        graph.sync()
    except Exception:
        logger.exception("Graph sync failed; it will be rebuilt on next access")


def _index_vectors(
    factory: sessionmaker[Session], vectors: VectorService | None, doc_id: str
) -> dict:
    """Embed + index this document's chunks. Never raises; the outcome is recorded."""
    if vectors is None:
        return {"status": "disabled"}
    try:
        report = vectors.sync()
    except Exception as exc:
        logger.exception("Vector indexing failed: document=%s", doc_id)
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]}
    with factory() as session:
        counts = dict(
            session.execute(
                select(ChunkEmbedding.status, func.count())
                .where(ChunkEmbedding.document_id == doc_id)
                .group_by(ChunkEmbedding.status)
            ).all()
        )
        total = session.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.document_id == doc_id)
        )
    indexed, failed = counts.get("indexed", 0), counts.get("failed", 0)
    with factory() as session:
        windowed = session.scalar(
            select(func.count())
            .select_from(ChunkEmbedding)
            .where(ChunkEmbedding.document_id == doc_id, ChunkEmbedding.window_count > 1)
        )
    meta = vectors.snapshot_meta
    return {
        "status": "indexed" if indexed == total else ("partial" if indexed else "failed"),
        "chunks_indexed": indexed,
        "chunks_failed": failed,
        "chunks_over_token_limit_pooled": windowed or 0,
        "chunks_total": total,
        "embedded_now": report.embedded,
        "reused": report.reused,
        "model": meta.model_id if meta else None,
        "model_revision": meta.model_revision if meta else None,
        "dimension": meta.dimension if meta else None,
        "index_revision": meta.revision if meta else None,
    }


def run_ingestion(
    factory: sessionmaker[Session],
    settings: Settings,
    job_id: str,
    provider: ExtractionProvider | None = None,
    graph: GraphService | None = None,
    vectors: VectorService | None = None,
) -> None:
    """Run one job to completion. Safe to call again for the same document (retry/recovery):
    chunks and knowledge are replaced inside transactions, so no duplicates can remain.
    """
    provider = provider or build_extraction_provider(settings)
    with factory() as session:
        job = session.get(ProcessingJob, job_id)
        if job is None or job.status in ("completed", "failed"):
            return
        doc = job.document
        job.status = "running"
        job.stage = "extracting"
        job.started_at = utcnow()
        job.error_message = None
        doc.processing_status = "processing"
        session.commit()
        doc_id, path, file_type, checksum = (
            doc.id,
            Path(doc.stored_path),
            doc.file_type,
            doc.checksum,
        )

    logger.info("Ingestion started: document=%s job=%s", doc_id, job_id)
    try:
        if not path.is_file():
            raise IngestionError("The stored source file is missing")
        parsed = parse_file(path, file_type)

        _set_stage(factory, job_id, "chunking")
        chunks = chunk_segments(
            parsed.segments, checksum, settings.chunk_size, settings.chunk_overlap
        )
        if not chunks:
            raise IngestionError("Chunking produced no content")

        _set_stage(factory, job_id, "persisting")
        with WRITE_LOCK, factory() as session, session.begin():
            doc = session.get(Document, doc_id)
            # A re-run replaces old chunks; first drop the evidence that points at them so
            # entities only this document supported are cleaned up, not orphaned.
            stale = purge_document_knowledge(session, doc_id)
            session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc_id))
            session.add_all(
                DocumentChunk(
                    id=c.id,
                    document_id=doc_id,
                    chunk_index=c.chunk_index,
                    content=c.content,
                    page_number=c.page_number,
                    section=c.section,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    chunk_metadata={},
                )
                for c in chunks
            )
            refresh_entities(session, stale)
            bump_vector_revision(session)  # chunks changed: the vector index must follow
            doc.doc_metadata = {
                **parsed.metadata,
                "chunk_count": len(chunks),
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            }

        _set_stage(factory, job_id, "extracting_knowledge")
        with factory() as session:
            rows = session.execute(
                select(DocumentChunk.id, DocumentChunk.content)
                .where(DocumentChunk.document_id == doc_id)
                .order_by(DocumentChunk.chunk_index)
            ).all()
        # Provider calls (possibly slow LLM requests) happen outside any DB transaction.
        results, failed = run_provider(provider, [(r.id, r.content) for r in rows])
        if results and failed == len(results):
            raise IngestionError("Knowledge extraction failed for every chunk")

        _set_stage(factory, job_id, "indexing_vectors")
        vector_info = _index_vectors(factory, vectors, doc_id)

        with WRITE_LOCK, factory() as session, session.begin():
            doc = session.get(Document, doc_id)
            job = session.get(ProcessingJob, job_id)
            stats = replace_document_knowledge(
                session,
                doc_id,
                results,
                allowed_types=set(settings.entity_types),
                fuzzy_threshold=settings.fuzzy_match_threshold,
                provider_name=provider.name,
            )
            doc.doc_metadata = {
                **doc.doc_metadata,
                "knowledge": stats.as_metadata(provider.name),
                "vector_index": vector_info,
            }
            doc.processing_status = "completed"
            job.status = "completed"
            job.stage = "done"
            job.completed_at = utcnow()
        _sync_graph(graph)
        logger.info(
            "Ingestion completed: document=%s chunks=%d relationships=%d failed_chunks=%d",
            doc_id,
            len(chunks),
            stats.relationships,
            stats.failed_chunks,
        )
    except IngestionError as exc:
        logger.warning("Ingestion failed: document=%s reason=%s", doc_id, exc)
        _mark_failed(factory, job_id, str(exc))
    except Exception:
        logger.exception("Unexpected ingestion error: document=%s", doc_id)
        _mark_failed(factory, job_id, _INTERNAL_ERROR)
