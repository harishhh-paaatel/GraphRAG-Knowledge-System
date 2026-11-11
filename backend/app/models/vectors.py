"""Bookkeeping for the vector index. Vectors themselves live in the FAISS file, not in SQLite."""

from datetime import datetime

from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UTCDateTime, utcnow


class ChunkEmbedding(Base):
    """Identity and synchronization state of one chunk's vector.

    This is a *report* of what the index is believed to contain, not the source of truth:
    ``document_chunks`` decides what must be indexed and the index file holds the vectors.
    ``content_hash`` is the SHA-256 of the chunk text the vector was computed from, so a
    changed chunk is detected as stale without re-embedding.
    """

    __tablename__ = "chunk_embeddings"
    __table_args__ = (Index("ix_chunk_embeddings_status", "status"),)

    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    content_hash: Mapped[str] = mapped_column(String(64))
    model_id: Mapped[str] = mapped_column(String(255))
    model_revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dimension: Mapped[int] = mapped_column(Integer)
    config_fingerprint: Mapped[str] = mapped_column(String(64))
    # Size of the chunk as the model's tokenizer sees it; window_count > 1 means the chunk
    # exceeded the model's token limit and its vector is pooled from several windows.
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    window_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16))  # indexed | failed | stale
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class VectorState(Base):
    """Single-row change counter (row id is always 1), like ``graph_state``.

    Incremented inside every SQLite transaction that adds, replaces or deletes chunks. The
    index file records the revision it was synchronized to, so "SQLite changed since the
    index was built" is one cheap integer comparison.
    """

    __tablename__ = "vector_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
