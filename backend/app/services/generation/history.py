"""Query history on the EXISTING ``query_history`` table (no migration).

Verified before relying on it (see tests/integration/test_query_history_store.py):

* columns: ``id, question, retrieval_strategy(16), answer (NOT NULL), execution_time_ms,
  generation_mode(16), created_at (indexed), metadata (JSON)``;
* there is no status/provider/model column, so those live in ``metadata`` (they cannot be
  filtered in SQL; the list endpoint filters nothing, which is fine for a personal history);
* ``answer`` is NOT NULL, so a question without an answer stores ``""`` and its status in
  ``metadata``;
* VARCHAR lengths are not enforced by SQLite, so this module enforces them itself;
* what is stored: the question, the answer, chunk IDs with a short excerpt snapshot (so an entry
  stays readable after its document is deleted), and run metadata. Never the prompt, the
  retrieved chunk text beyond the excerpt, API keys or URLs.

Recording a query is best-effort: any failure is logged (without the question text) and the
query result is returned anyway.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import QueryHistory
from app.schemas.query import EXCERPT_CHARS, QueryResponse

logger = logging.getLogger(__name__)

MAX_STRATEGY_CHARS = 16  # query_history.retrieval_strategy is VARCHAR(16)
MAX_MODE_CHARS = 16  # query_history.generation_mode is VARCHAR(16)
MAX_METADATA_BYTES = 65_536
MAX_LIST_LIMIT = 200


@dataclass
class QueryRecord:
    """Everything worth remembering about one answered (or unanswered) question."""

    question: str
    strategy: str
    status: str
    generation_mode: str
    execution_time_ms: float
    reason: str | None = None
    answer: str | None = None
    citations: list[dict] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    usage: dict | None = None
    grounding: dict | None = None
    retrieval: dict | None = None
    timings_ms: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def row_values(self) -> dict:
        """The column values for ``QueryHistory``; raises ``ValueError`` for values that would
        not fit the existing columns (a programming error, caught and logged by the store)."""
        if len(self.strategy) > MAX_STRATEGY_CHARS:
            raise ValueError("retrieval_strategy does not fit the column")
        if len(self.generation_mode) > MAX_MODE_CHARS:
            raise ValueError("generation_mode does not fit the column")
        if self.execution_time_ms < 0:
            raise ValueError("execution_time_ms must not be negative")
        citations = [
            {**c, "excerpt": str(c.get("excerpt", ""))[:EXCERPT_CHARS]} for c in self.citations
        ]
        metadata: dict = {
            "status": self.status,
            "reason": self.reason,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "usage": self.usage,
            "grounding": self.grounding,
            "retrieval": self.retrieval,
            "citations": citations,
            "timings_ms": self.timings_ms,
            "warnings": self.warnings[:20],
        }
        metadata = _fit(metadata)
        return {
            "question": self.question,
            "retrieval_strategy": self.strategy,
            "answer": self.answer or "",
            "execution_time_ms": float(self.execution_time_ms),
            "generation_mode": self.generation_mode,
            "query_metadata": metadata,
        }


def _fit(metadata: dict) -> dict:
    """Keep the stored JSON under ``MAX_METADATA_BYTES`` by shedding the bulkiest parts first."""

    def size() -> int:
        return len(json.dumps(metadata, default=str).encode("utf-8"))

    if size() <= MAX_METADATA_BYTES:
        return metadata
    metadata["truncated"] = True
    for key in ("retrieval", "warnings", "grounding"):
        metadata[key] = None if key != "warnings" else []
        if size() <= MAX_METADATA_BYTES:
            return metadata
    while metadata["citations"] and size() > MAX_METADATA_BYTES:
        metadata["citations"].pop()
    return metadata


@dataclass(frozen=True)
class HistoryEntry:
    id: str
    question: str
    answer: str | None
    status: str
    reason: str | None
    generation_mode: str
    retrieval_strategy: str
    execution_time_ms: float
    created_at: datetime
    metadata: dict


def record_from_response(response: QueryResponse, execution_time_ms: float) -> QueryRecord:
    """Build a history record from the API response (one place that knows both shapes)."""
    g = response.generation
    return QueryRecord(
        question=response.question,
        strategy="hybrid",
        status=response.status,
        reason=response.reason,
        answer=response.answer,
        generation_mode=g.mode,
        execution_time_ms=execution_time_ms,
        citations=[
            {
                "marker": c.marker,
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "filename": c.source.filename,
                "page_number": c.source.page_number,
                "section": c.source.section,
                "excerpt": c.excerpt,
                "cited_in_sentences": c.cited_in_sentences,
            }
            for c in response.citations
        ],
        provider=g.provider,
        model=g.model,
        prompt_version=g.prompt_version,
        usage=g.usage.model_dump() if g.usage else None,
        grounding=response.grounding.model_dump(),
        retrieval={
            "mode": response.retrieval.path_summary.mode,
            "complete": response.retrieval.path_summary.complete,
            "contributing": response.retrieval.path_summary.contributing,
        },
        timings_ms=response.timings_ms,
        warnings=response.warnings,
    )


class QueryHistoryStore:
    """Record, list, read and delete history entries. ``record`` never raises."""

    def __init__(self, factory: sessionmaker[Session], enabled: bool = True) -> None:
        self._factory = factory
        self.enabled = enabled

    def record(self, record: QueryRecord) -> str | None:
        """Store ``record``; returns its ID, or None if disabled or if storing failed."""
        if not self.enabled:
            return None
        try:
            values = record.row_values()
            with self._factory() as session:
                row = QueryHistory(**values)
                session.add(row)
                session.commit()
                return row.id
        except Exception:
            # the question text is deliberately not logged: it is the user's data
            logger.exception(
                "Could not record query history (question of %d chars)", len(record.question)
            )
            return None

    def list(self, limit: int = 50, offset: int = 0) -> tuple[list[HistoryEntry], int]:
        limit = max(1, min(int(limit), MAX_LIST_LIMIT))
        offset = max(0, int(offset))
        with self._factory() as session:
            total = session.scalar(select(func.count()).select_from(QueryHistory)) or 0
            rows = session.scalars(
                select(QueryHistory)
                .order_by(QueryHistory.created_at.desc(), QueryHistory.id.desc())
                .limit(limit)
                .offset(offset)
            ).all()
            return [_entry(r) for r in rows], total

    def get(self, history_id: str) -> HistoryEntry | None:
        with self._factory() as session:
            row = session.get(QueryHistory, history_id)
            return _entry(row) if row is not None else None

    def delete(self, history_id: str) -> bool:
        with self._factory() as session:
            result = session.execute(delete(QueryHistory).where(QueryHistory.id == history_id))
            session.commit()
            return (result.rowcount or 0) > 0


def _entry(row: QueryHistory) -> HistoryEntry:
    meta = row.query_metadata or {}
    return HistoryEntry(
        id=row.id,
        question=row.question,
        answer=row.answer or None,
        status=str(meta.get("status", "unknown")),
        reason=meta.get("reason"),
        generation_mode=row.generation_mode,
        retrieval_strategy=row.retrieval_strategy,
        execution_time_ms=row.execution_time_ms,
        created_at=row.created_at,
        metadata=meta,
    )
