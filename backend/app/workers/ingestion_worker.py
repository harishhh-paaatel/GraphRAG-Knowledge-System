"""Lightweight in-process background worker (ThreadPoolExecutor + DB job table)."""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.models.base import utcnow
from app.repositories import documents as repo
from app.services.embeddings.service import VectorService
from app.services.extraction.types import ExtractionProvider
from app.services.graph.service import GraphService
from app.services.ingestion.pipeline import run_ingestion

logger = logging.getLogger(__name__)


class IngestionWorker:
    """Runs ingestion jobs off the request thread. Job state lives in SQLite, not memory."""

    def __init__(
        self,
        factory: sessionmaker[Session],
        settings: Settings,
        provider: ExtractionProvider | None = None,
        graph: GraphService | None = None,
        vectors: VectorService | None = None,
    ) -> None:
        self._factory = factory
        self._settings = settings
        self.provider = provider  # None -> pipeline builds one from settings per job
        self.graph = graph
        self.vectors = vectors
        self._pool = ThreadPoolExecutor(
            max_workers=settings.worker_threads, thread_name_prefix="ingest"
        )
        self._futures: set[Future] = set()
        self._lock = threading.Lock()

    def submit(self, job_id: str) -> Future:
        """Queue a job; returns immediately."""
        future = self._pool.submit(
            run_ingestion,
            self._factory,
            self._settings,
            job_id,
            self.provider,
            self.graph,
            self.vectors,
        )
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._done)
        return future

    def _done(self, future: Future) -> None:
        with self._lock:
            self._futures.discard(future)
        if future.exception() is not None:  # run_ingestion handles its own errors; defensive
            logger.error("Ingestion task crashed: %r", future.exception())

    def drain(self, timeout: float | None = None) -> bool:
        """Block until queued work finishes (used by tests and graceful shutdown)."""
        with self._lock:
            pending = list(self._futures)
        _, not_done = wait(pending, timeout=timeout)
        return not not_done

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)

    def recover_interrupted_jobs(self) -> int:
        """Re-queue jobs a previous process left ``pending`` or ``running``.

        Safe because the pipeline replaces a document's chunks atomically. Jobs whose source
        file has vanished are failed instead of retried. Returns the number re-queued.
        """
        try:
            with self._factory() as session:
                jobs = repo.jobs_to_recover(session)
                queue: list[str] = []
                for job in jobs:
                    if not Path(job.document.stored_path).is_file():
                        job.status = "failed"
                        job.error_message = "Interrupted job could not be recovered: file missing"
                        job.completed_at = utcnow()
                        job.document.processing_status = "failed"
                        continue
                    if job.status == "running":
                        job.attempt += 1
                        job.status = "pending"
                        job.stage = None
                    job.document.processing_status = "pending"
                    queue.append(job.id)
                session.commit()
        except SQLAlchemyError:
            logger.exception("Could not scan for interrupted jobs (is the schema migrated?)")
            return 0
        for job_id in queue:
            self.submit(job_id)
        if queue:
            logger.info("Recovered %d interrupted ingestion job(s)", len(queue))
        return len(queue)
