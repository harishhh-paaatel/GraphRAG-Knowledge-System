"""Idempotent retries and recovery of jobs interrupted by a restart."""

import io

from fastapi.testclient import TestClient

from app.main import create_app
from app.models import Base, Document, DocumentChunk, ProcessingJob
from app.services.ingestion import service
from app.services.ingestion.pipeline import run_ingestion
from tests.helpers import SAMPLE_TEXT


def _accept(session, settings, data=None, name="a.txt"):
    """Accept an upload WITHOUT running the worker, leaving a pending job (as after a crash)."""
    payload = data if data is not None else SAMPLE_TEXT.encode()
    return service.accept_upload(session, settings, io.BytesIO(payload), name, "text/plain")


def _chunk_rows(session, doc_id):
    session.expire_all()
    return [
        (c.id, c.chunk_index, c.content)
        for c in session.query(DocumentChunk)
        .filter_by(document_id=doc_id)
        .order_by(DocumentChunk.chunk_index)
    ]


def test_rerunning_ingestion_never_duplicates_chunks(app, session, settings):
    doc, job, _ = _accept(session, settings)
    run_ingestion(app.state.session_factory, settings, job.id)
    before = _chunk_rows(session, doc.id)
    assert before

    # Simulate a retry / recovery re-run: a fresh pending job on the same document.
    retry = ProcessingJob(document_id=doc.id, status="pending", attempt=2)
    session.add(retry)
    session.commit()
    run_ingestion(app.state.session_factory, settings, retry.id)

    assert _chunk_rows(session, doc.id) == before  # identical IDs, indexes, content, no extras
    session.expire_all()
    assert session.get(ProcessingJob, retry.id).status == "completed"


def test_finished_jobs_are_not_rerun(app, session, settings):
    doc, job, _ = _accept(session, settings)
    run_ingestion(app.state.session_factory, settings, job.id)
    session.expire_all()
    stamp = session.get(ProcessingJob, job.id).completed_at
    run_ingestion(app.state.session_factory, settings, job.id)
    session.expire_all()
    assert session.get(ProcessingJob, job.id).completed_at == stamp


def test_recover_requeues_pending_and_running_jobs(app, session, settings):
    pending_doc, pending_job, _ = _accept(session, settings, b"pending doc text", "p.txt")
    running_doc, running_job, _ = _accept(session, settings, b"running doc text", "r.txt")
    running_job.status, running_job.stage = "running", "chunking"
    running_doc.processing_status = "processing"
    session.commit()

    assert app.state.worker.recover_interrupted_jobs() == 2
    assert app.state.worker.drain(timeout=30)

    session.expire_all()
    for doc, job in ((pending_doc, pending_job), (running_doc, running_job)):
        assert session.get(Document, doc.id).processing_status == "completed"
        assert session.get(ProcessingJob, job.id).status == "completed"
        assert len(_chunk_rows(session, doc.id)) >= 1
    assert session.get(ProcessingJob, pending_job.id).attempt == 1
    assert session.get(ProcessingJob, running_job.id).attempt == 2  # interrupted run counted


def test_recovery_fails_jobs_whose_file_vanished(app, session, settings):
    from pathlib import Path

    doc, job, _ = _accept(session, settings)
    Path(doc.stored_path).unlink()
    assert app.state.worker.recover_interrupted_jobs() == 0
    session.expire_all()
    assert session.get(Document, doc.id).processing_status == "failed"
    assert "file missing" in session.get(ProcessingJob, job.id).error_message


def test_recovery_runs_automatically_on_application_startup(settings):
    # Process 1: accepts an upload, then "crashes" before the worker runs it.
    first = create_app(settings)
    Base.metadata.create_all(first.state.engine)
    with first.state.session_factory() as s:
        doc, job, _ = _accept(s, settings)
        doc_id, job_id = doc.id, job.id
    first.state.worker.shutdown()
    first.state.engine.dispose()

    # Process 2: startup (lifespan) must pick the job up with no client action.
    second = create_app(settings)
    with TestClient(second) as client:
        second.state.worker.drain(timeout=30)
        body = client.get(f"/api/v1/documents/{doc_id}/status").json()
    assert body["processing_status"] == "completed"
    assert body["job"]["id"] == job_id
    second.state.engine.dispose()


def test_startup_recovery_tolerates_unmigrated_database(settings):
    application = create_app(settings)  # no tables created
    assert application.state.worker.recover_interrupted_jobs() == 0
    application.state.worker.shutdown()
