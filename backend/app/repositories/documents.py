"""Data access for documents, chunks and processing jobs (parameterized ORM only)."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk, ProcessingJob

BUSY_STATUSES = ("pending", "processing")


def get_document(session: Session, document_id: str) -> Document | None:
    return session.get(Document, document_id)


def get_by_checksum(session: Session, checksum: str) -> Document | None:
    return session.scalar(select(Document).where(Document.checksum == checksum))


def list_documents(
    session: Session, limit: int, offset: int, status: str | None = None
) -> tuple[list[Document], int]:
    """Newest first. Returns (page, total matching)."""
    query = select(Document)
    count = select(func.count()).select_from(Document)
    if status:
        query = query.where(Document.processing_status == status)
        count = count.where(Document.processing_status == status)
    rows = session.scalars(
        query.order_by(Document.upload_timestamp.desc(), Document.id).limit(limit).offset(offset)
    ).all()
    return list(rows), session.scalar(count) or 0


def latest_job(session: Session, document_id: str) -> ProcessingJob | None:
    return session.scalar(
        select(ProcessingJob)
        .where(ProcessingJob.document_id == document_id)
        .order_by(ProcessingJob.attempt.desc(), ProcessingJob.id.desc())
        .limit(1)
    )


def chunk_count(session: Session, document_id: str) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
        )
        or 0
    )


def jobs_to_recover(session: Session) -> list[ProcessingJob]:
    """Jobs left queued or mid-flight by a previous process."""
    return list(
        session.scalars(
            select(ProcessingJob)
            .where(ProcessingJob.status.in_(("pending", "running")))
            .order_by(ProcessingJob.id)
        )
    )
