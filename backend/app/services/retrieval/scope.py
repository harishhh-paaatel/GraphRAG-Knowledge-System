"""SQLite helpers shared by every retrieval path.

This module is deliberately neutral: it knows documents and chunks (the source of truth) and
nothing about FAISS, embeddings or the knowledge graph, so the vector path and the graph path
can both use it without becoming dependent on each other. A test enforces that boundary.

Only chunks of ``completed`` documents are searchable: a document that is still processing or
failed is not (yet, or no longer) a trustworthy source.
"""

from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk

# SQLite caps the number of bound parameters in one statement; stay far below any limit.
_BATCH = 500


@dataclass(frozen=True)
class DocumentScope:
    """Which documents a search may use.

    ``allowed`` is ``None`` when the caller did not restrict documents (everything that is
    not in ``not_ready`` is searchable); otherwise it is the requested documents that exist
    and are completed. ``unknown`` lists requested IDs that do not exist (reported to the
    caller, not an error).
    """

    allowed: frozenset[str] | None
    not_ready: frozenset[str]
    unknown: tuple[str, ...] = field(default_factory=tuple)

    def permits(self, document_id: str) -> bool:
        if document_id in self.not_ready:
            return False
        return self.allowed is None or document_id in self.allowed


def resolve_document_scope(session: Session, document_ids: Collection[str] | None) -> DocumentScope:
    """Resolve the optional ``document_ids`` filter against the documents table."""
    not_ready = frozenset(
        session.scalars(select(Document.id).where(Document.processing_status != "completed"))
    )
    if document_ids is None:
        return DocumentScope(allowed=None, not_ready=not_ready)
    wanted = set(document_ids)
    known = set(session.scalars(select(Document.id).where(Document.id.in_(wanted))))
    return DocumentScope(
        allowed=frozenset(known - not_ready),
        not_ready=not_ready,
        unknown=tuple(sorted(wanted - known)),
    )


@dataclass(frozen=True)
class ChunkRecord:
    """A chunk exactly as stored, with its document's filename."""

    chunk_id: str
    document_id: str
    filename: str
    content: str
    page_number: int | None
    section: str | None
    chunk_index: int
    processing_status: str = "completed"


def lookup_chunks(
    session: Session, chunk_ids: Iterable[str], scope: DocumentScope | None = None
) -> dict[str, ChunkRecord]:
    """Fetch chunks by ID from SQLite.

    IDs that no longer exist (the chunk or its document was deleted after a derived index was
    built) are simply absent from the result: nothing is ever invented for them. With a
    ``scope``, chunks of documents the scope does not permit are absent too.
    """
    ids = list(dict.fromkeys(chunk_ids))
    found: dict[str, ChunkRecord] = {}
    for start in range(0, len(ids), _BATCH):
        batch = ids[start : start + _BATCH]
        rows = session.execute(
            select(DocumentChunk, Document.filename, Document.processing_status)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.id.in_(batch))
        )
        for chunk, filename, status in rows:
            if scope is not None and not scope.permits(chunk.document_id):
                continue
            found[chunk.id] = ChunkRecord(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=filename,
                content=chunk.content,
                page_number=chunk.page_number,
                section=chunk.section,
                chunk_index=chunk.chunk_index,
                processing_status=status,
            )
    return found
