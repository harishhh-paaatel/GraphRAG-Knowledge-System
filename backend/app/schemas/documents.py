"""Document API schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class JobOut(BaseModel):
    """Processing job state."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    stage: str | None
    attempt: int
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None


class DocumentOut(BaseModel):
    """Document summary (no server filesystem paths are exposed)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    size_bytes: int
    checksum: str
    upload_timestamp: datetime
    processing_status: str


class DocumentDetail(DocumentOut):
    """Document with metadata, chunk count and latest job."""

    metadata: dict
    chunk_count: int
    job: JobOut | None


class DocumentList(BaseModel):
    """Paginated document list."""

    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class UploadResponse(BaseModel):
    """Returned with 202 when a document is accepted for processing."""

    document: DocumentOut
    job: JobOut
    retry: bool = False  # true when a previously failed document was re-queued


class StatusResponse(BaseModel):
    """Lightweight processing status."""

    document_id: str
    processing_status: str
    job: JobOut | None
