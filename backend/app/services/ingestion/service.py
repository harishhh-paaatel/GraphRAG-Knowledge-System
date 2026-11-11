"""Upload handling and document lifecycle (create, retry, delete)."""

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import AppError
from app.models import Document, ProcessingJob
from app.models.base import new_id
from app.repositories import documents as repo
from app.services.embeddings.revision import bump_vector_revision
from app.services.extraction.persistence import (
    WRITE_LOCK,
    purge_document_knowledge,
    refresh_entities,
)
from app.services.ingestion import validation

logger = logging.getLogger(__name__)

_CHUNK = 1024 * 1024


def _stage_upload(stream: BinaryIO, settings: Settings) -> tuple[Path, str, int]:
    """Copy the upload to a temp file, hashing as we go; abort past the size limit."""
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    limit = settings.max_upload_mb * 1024 * 1024
    digest = hashlib.sha256()
    size = 0
    fd, name = tempfile.mkstemp(dir=settings.tmp_dir, suffix=".upload")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as out:
            while block := stream.read(_CHUNK):
                size += len(block)
                if size > limit:
                    raise AppError(
                        413, "file_too_large", f"File exceeds the {settings.max_upload_mb} MB limit"
                    )
                digest.update(block)
                out.write(block)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp, digest.hexdigest(), size


def _queue_job(session: Session, document: Document, attempt: int) -> ProcessingJob:
    job = ProcessingJob(document_id=document.id, status="pending", attempt=attempt)
    document.processing_status = "pending"
    session.add(job)
    session.flush()
    return job


def accept_upload(
    session: Session,
    settings: Settings,
    stream: BinaryIO,
    filename: str | None,
    content_type: str | None,
) -> tuple[Document, ProcessingJob, bool]:
    """Validate and store an upload, create its Document + pending job, and commit.

    Returns ``(document, job, is_retry)``. Raises AppError for every client-side failure.
    A re-upload of a *failed* document re-queues it; any other duplicate is a 409.
    """
    display_name = validation.sanitize_filename(filename)
    file_type = validation.detect_file_type(display_name)
    validation.validate_declared_mime(file_type, content_type)

    tmp, checksum, size = _stage_upload(stream, settings)
    try:
        validation.sniff_content(file_type, tmp)

        existing = repo.get_by_checksum(session, checksum)
        if existing is not None:
            if existing.processing_status != "failed":
                raise AppError(
                    409,
                    "duplicate_document",
                    "An identical document has already been uploaded",
                    {"document_id": existing.id, "processing_status": existing.processing_status},
                )
            last = repo.latest_job(session, existing.id)
            job = _queue_job(session, existing, (last.attempt if last else 0) + 1)
            session.commit()
            logger.info("Retrying failed document=%s job=%s", existing.id, job.id)
            return existing, job, True

        doc_id = new_id()
        target_dir = settings.documents_dir / doc_id
        target = target_dir / f"source.{file_type}"
        target_dir.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, target)
        try:
            doc = Document(
                id=doc_id,
                filename=display_name,
                stored_path=str(target),
                file_type=file_type,
                size_bytes=size,
                checksum=checksum,
                doc_metadata={},
            )
            session.add(doc)
            session.flush()
            job = _queue_job(session, doc, 1)
            session.commit()
        except IntegrityError:  # concurrent identical upload won the unique-checksum race
            session.rollback()
            shutil.rmtree(target_dir, ignore_errors=True)
            raise AppError(
                409, "duplicate_document", "An identical document was just uploaded"
            ) from None
        except BaseException:
            session.rollback()
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        logger.info("Accepted upload: document=%s type=%s bytes=%d", doc.id, file_type, size)
        return doc, job, False
    finally:
        tmp.unlink(missing_ok=True)


def delete_document(session: Session, settings: Settings, document_id: str) -> None:
    """Delete a document, its chunks/jobs (DB cascade) and its stored files.

    The database is the source of truth, so rows go first; leftover files are only logged.
    """
    doc = repo.get_document(session, document_id)
    if doc is None:
        raise AppError(404, "document_not_found", "Document not found")
    if doc.processing_status in repo.BUSY_STATUSES:
        raise AppError(409, "document_busy", "Document is still being processed")
    with WRITE_LOCK:
        # Capture which entities this document evidenced, delete it (DB cascade removes its
        # chunks, mentions, relationships, jobs), then drop entities nothing else supports.
        affected = purge_document_knowledge(session, document_id)
        session.delete(doc)
        session.flush()
        refresh_entities(session, affected)
        bump_vector_revision(session)  # its chunks (and so its vectors) are gone
        session.commit()
    try:
        shutil.rmtree(settings.documents_dir / document_id)
    except FileNotFoundError:
        pass
    except OSError:
        logger.exception("Could not remove files for deleted document=%s", document_id)
    logger.info("Deleted document=%s", document_id)


def require_document(session: Session, document_id: str) -> Document:
    """Fetch a document or raise 404."""
    doc = repo.get_document(session, document_id)
    if doc is None:
        raise AppError(404, "document_not_found", "Document not found")
    return doc
