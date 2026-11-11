"""Document endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_graph, get_session, get_settings_dep, get_vectors, get_worker
from app.core.config import Settings
from app.repositories import documents as repo
from app.schemas.documents import (
    DocumentDetail,
    DocumentList,
    DocumentOut,
    JobOut,
    StatusResponse,
    UploadResponse,
)
from app.services.embeddings.service import VectorService
from app.services.graph.service import GraphService
from app.services.ingestion import service
from app.workers.ingestion_worker import IngestionWorker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
WorkerDep = Annotated[IngestionWorker, Depends(get_worker)]
GraphDep = Annotated[GraphService, Depends(get_graph)]
VectorsDep = Annotated[VectorService, Depends(get_vectors)]


@router.post("/upload", status_code=202, response_model=UploadResponse)
def upload_document(
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT or Markdown")],
    session: SessionDep,
    settings: SettingsDep,
    worker: WorkerDep,
) -> UploadResponse:
    """Validate and store a file, then queue background processing (HTTP 202)."""
    doc, job, retry = service.accept_upload(
        session, settings, file.file, file.filename, file.content_type
    )
    worker.submit(job.id)
    return UploadResponse(
        document=DocumentOut.model_validate(doc), job=JobOut.model_validate(job), retry=retry
    )


@router.get("", response_model=DocumentList)
def list_documents(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: Annotated[str | None, Query(pattern="^(pending|processing|completed|failed)$")] = None,
) -> DocumentList:
    """List documents, newest first, optionally filtered by processing status."""
    items, total = repo.list_documents(session, limit, offset, status)
    return DocumentList(
        items=[DocumentOut.model_validate(d) for d in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(document_id: str, session: SessionDep) -> DocumentDetail:
    """Document details including metadata and chunk count."""
    doc = service.require_document(session, document_id)
    job = repo.latest_job(session, document_id)
    return DocumentDetail(
        **DocumentOut.model_validate(doc).model_dump(),
        metadata=doc.doc_metadata or {},
        chunk_count=repo.chunk_count(session, document_id),
        job=JobOut.model_validate(job) if job else None,
    )


@router.get("/{document_id}/status", response_model=StatusResponse)
def get_status(document_id: str, session: SessionDep) -> StatusResponse:
    """Processing status and latest job (poll this after upload)."""
    doc = service.require_document(session, document_id)
    job = repo.latest_job(session, document_id)
    return StatusResponse(
        document_id=doc.id,
        processing_status=doc.processing_status,
        job=JobOut.model_validate(job) if job else None,
    )


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    session: SessionDep,
    settings: SettingsDep,
    graph: GraphDep,
    vectors: VectorsDep,
) -> Response:
    """Delete a document and everything derived from it (409 while processing)."""
    service.delete_document(session, settings, document_id)
    try:
        graph.sync()
    except Exception:  # SQLite is already correct; the graph self-heals on next read
        logger.exception("Graph sync after delete failed")
    try:
        vectors.sync()  # removal only: needs no embedding
    except Exception:  # the index self-heals on the next sync or search
        logger.exception("Vector sync after delete failed")
    return Response(status_code=204)
