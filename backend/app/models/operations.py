"""Processing jobs, query history and evaluation runs."""

from datetime import datetime

from sqlalchemy import JSON, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, UTCDateTime, new_id, utcnow
from app.models.document import Document


class ProcessingJob(Base):
    """Tracks one ingestion attempt for a document."""

    __tablename__ = "processing_jobs"
    __table_args__ = (Index("ix_jobs_status", "status"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(16), default="pending")
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    document: Mapped[Document] = relationship(back_populates="jobs")


class QueryHistory(Base):
    """One answered question."""

    __tablename__ = "query_history"
    __table_args__ = (Index("ix_query_history_created", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    question: Mapped[str] = mapped_column(Text)
    retrieval_strategy: Mapped[str] = mapped_column(String(16))
    answer: Mapped[str] = mapped_column(Text)
    execution_time_ms: Mapped[float] = mapped_column(Float)
    generation_mode: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    query_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


class EvaluationRun(Base):
    """A persisted evaluation run; metrics are only ever written by real runs."""

    __tablename__ = "evaluation_runs"
    __table_args__ = (Index("ix_eval_created", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    dataset_name: Mapped[str] = mapped_column(String(128))
    dataset_size: Mapped[int] = mapped_column(Integer)
    strategy: Mapped[str] = mapped_column(String(16))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class GraphState(Base):
    """Single-row change counter for the knowledge graph (row id is always 1).

    ``revision`` is incremented inside every SQLite transaction that changes entities,
    mentions or relationships. The persisted graph file records the revision it was built
    from, so SQLite and the file can never silently disagree: a mismatch means "rebuild".
    """

    __tablename__ = "graph_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
