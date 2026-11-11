"""Knowledge extraction as part of the real ingestion pipeline (upload -> worker -> DB)."""

import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import Settings
from app.main import create_app
from app.models import (
    Base,
    Document,
    DocumentChunk,
    Entity,
    EntityMention,
    ProcessingJob,
    Relationship,
)
from app.services.extraction.mock import MockExtractionProvider
from app.services.extraction.types import ChunkExtraction, ExtractedEntity, ExtractedRelationship
from app.services.ingestion import service
from app.services.ingestion.pipeline import run_ingestion

URL = "/api/v1/documents"
TEXT = (
    b"Sebastian Ramirez created FastAPI. FastAPI was created by Sebastian Ramirez.\n\n"
    b"Alice works at Acme Corp. Acme Corp is located in Berlin. FastAPI uses Pydantic."
)


@pytest.fixture
def upload(client, app):
    def _upload(name, data, mime="text/plain"):
        r = client.post(f"{URL}/upload", files={"file": (name, data, mime)})
        assert app.state.worker.drain(timeout=30)
        return r

    return _upload


def rows(session, model):
    session.expire_all()
    return session.scalars(select(model)).all()


def test_end_to_end_default_rule_based_extraction(client, upload, session):
    r = upload("a.txt", TEXT)
    doc_id = r.json()["document"]["id"]
    status = client.get(f"{URL}/{doc_id}/status").json()
    assert status["processing_status"] == "completed" and status["job"]["stage"] == "done"

    ents = {(e.canonical_name, e.entity_type) for e in rows(session, Entity)}
    assert {
        ("Sebastian Ramirez", "PERSON"),
        ("FastAPI", "TECHNOLOGY"),
        ("Alice", "PERSON"),
        ("Acme Corp", "ORGANIZATION"),
        ("Berlin", "LOCATION"),
        ("Pydantic", "TECHNOLOGY"),
    } <= ents

    names = {e.id: e.canonical_name for e in rows(session, Entity)}
    triples = {
        (names[x.source_entity_id], x.relationship_type, names[x.target_entity_id])
        for x in rows(session, Relationship)
    }
    assert triples == {
        ("Sebastian Ramirez", "DEVELOPED", "FastAPI"),
        ("Alice", "WORKS_AT", "Acme Corp"),
        ("Acme Corp", "LOCATED_IN", "Berlin"),
        ("FastAPI", "USES", "Pydantic"),
    }
    chunk = session.scalars(select(DocumentChunk)).first()
    for rel in rows(session, Relationship):
        assert rel.source_document_id == doc_id and rel.provenance == "rule_based"
        assert rel.supporting_text in session.get(DocumentChunk, rel.source_chunk_id).content
    assert all(m.provenance == "rule_based" and m.chunk_id for m in rows(session, EntityMention))
    assert chunk is not None

    knowledge = client.get(f"{URL}/{doc_id}").json()["metadata"]["knowledge"]
    assert knowledge["extractor"] == "rule_based"
    assert knowledge["relationships"] == 4 and knowledge["failed_chunks"] == 0


def test_two_documents_share_entities(client, upload, session):
    upload("a.txt", b"Alice works at Acme Corp. FastAPI uses Pydantic.")
    upload("b.txt", b"Bob Smith joined Acme Corp. Pydantic is part of Python.")
    names = [e.canonical_name for e in rows(session, Entity)]
    assert names.count("Acme Corp") == 1 and names.count("Pydantic") == 1
    acme = session.scalars(select(Entity).where(Entity.canonical_name == "Acme Corp")).one()
    docs = {m.document_id for m in acme.mentions}
    assert len(docs) == 2


def test_concurrent_workers_do_not_duplicate_shared_entities(tmp_path):
    settings = Settings(data_dir=tmp_path, worker_threads=4, environment="test", _env_file=None)
    application = create_app(settings)
    Base.metadata.create_all(application.state.engine)
    with TestClient(application) as c:
        for i in range(8):
            payload = (
                f"Doc {i}. Alice works at Acme Corp. FastAPI uses Pydantic. Note {i}.".encode()
            )
            assert (
                c.post(
                    f"{URL}/upload", files={"file": (f"{i}.txt", payload, "text/plain")}
                ).status_code
                == 202
            )
        assert application.state.worker.drain(timeout=60)
        with application.state.session_factory() as s:
            assert {d.processing_status for d in s.scalars(select(Document))} == {"completed"}
            keys = [(e.normalized_key, e.entity_type) for e in s.scalars(select(Entity))]
            assert len(keys) == len(set(keys))
            assert keys.count(("acme", "ORGANIZATION")) == 1
            acme = s.scalars(select(Entity).where(Entity.normalized_key == "acme")).one()
            assert len({m.document_id for m in acme.mentions}) == 8
    application.state.engine.dispose()


def test_rerunning_the_pipeline_is_idempotent(app, session, settings):
    doc, job, _ = service.accept_upload(session, settings, io.BytesIO(TEXT), "a.txt", "text/plain")
    run_ingestion(app.state.session_factory, settings, job.id)

    def state():
        session.expire_all()
        return (
            sorted(
                (e.id, e.canonical_name, e.entity_type) for e in session.scalars(select(Entity))
            ),
            sorted(
                (m.entity_id, m.chunk_id, m.surface_form)
                for m in session.scalars(select(EntityMention))
            ),
            sorted(
                (r.source_entity_id, r.target_entity_id, r.relationship_type, r.source_chunk_id)
                for r in session.scalars(select(Relationship))
            ),
        )

    before = state()
    assert before[2]
    for attempt in (2, 3):
        retry = ProcessingJob(document_id=doc.id, status="pending", attempt=attempt)
        session.add(retry)
        session.commit()
        run_ingestion(app.state.session_factory, settings, retry.id)
    assert state() == before  # same entity IDs, no duplicate mentions or relationships


# ---- failure handling -------------------------------------------------------------------


def test_total_extraction_failure_fails_the_job_visibly(client, app, upload):
    app.state.worker.provider = MockExtractionProvider(fail_on={"Alice"})
    r = upload("a.txt", b"Alice works at Acme Corp.")
    doc_id = r.json()["document"]["id"]
    status = client.get(f"{URL}/{doc_id}/status").json()
    assert status["processing_status"] == "failed"
    assert "failed for every chunk" in status["job"]["error_message"]
    assert status["job"]["stage"] == "extracting_knowledge"


def test_partial_extraction_failure_is_recorded_in_metadata(app, client, session):
    big = ("Alice works at Acme Corp. " * 40 + "\n\n" + "Poison paragraph. " * 60).encode()
    app.state.settings.chunk_size = 300
    app.state.settings.chunk_overlap = 0
    app.state.worker.provider = MockExtractionProvider(fail_on={"Poison"})
    r = client.post(f"{URL}/upload", files={"file": ("a.txt", big, "text/plain")})
    app.state.worker.drain(30)
    meta = client.get(f"{URL}/{r.json()['document']['id']}").json()
    assert meta["processing_status"] == "completed"
    assert meta["metadata"]["knowledge"]["failed_chunks"] >= 1  # not silently ignored
    assert meta["chunk_count"] > meta["metadata"]["knowledge"]["failed_chunks"]


def test_mock_provider_output_is_validated_before_storage(client, app, upload, session):
    app.state.worker.provider = MockExtractionProvider(
        {
            "Alice": ChunkExtraction(
                [ExtractedEntity("Alice", "PERSON"), ExtractedEntity("Acme Corp", "ORGANIZATION")],
                [
                    ExtractedRelationship("Alice", "Acme Corp", "WORKS_AT", "Alice runs Acme Corp"),
                    ExtractedRelationship(
                        "Alice", "Acme Corp", "WORKS_AT", "Alice works at Acme Corp"
                    ),
                ],
            )
        }
    )
    r = upload("a.txt", b"Alice works at Acme Corp.")
    meta = client.get(f"{URL}/{r.json()['document']['id']}").json()["metadata"]["knowledge"]
    assert meta["extractor"] == "mock"
    assert (meta["relationships"], meta["rejected_relationships"]) == (1, 1)
    assert meta["sample_rejections"][0]["reason"] == "supporting text not found in chunk"
    assert {m.provenance for m in rows(session, EntityMention)} == {"mock"}


def test_reupload_after_extraction_failure_recovers(client, app, upload, session):
    app.state.worker.provider = MockExtractionProvider(fail_on={"Alice"})
    first = upload("a.txt", b"Alice works at Acme Corp.")
    doc_id = first.json()["document"]["id"]
    assert client.get(f"{URL}/{doc_id}/status").json()["processing_status"] == "failed"

    app.state.worker.provider = None  # back to the configured rule-based extractor
    again = upload("a.txt", b"Alice works at Acme Corp.")
    assert again.json()["retry"] is True
    assert client.get(f"{URL}/{doc_id}/status").json()["processing_status"] == "completed"
    assert len(rows(session, Relationship)) == 1


# ---- deletion through the API -----------------------------------------------------------


def test_api_delete_keeps_shared_entities_and_removes_exclusive_ones(client, upload, session):
    a = upload("a.txt", b"Alice works at Acme Corp. FastAPI uses Pydantic.")
    b = upload("b.txt", b"Bob Smith joined Acme Corp.")
    assert client.delete(f"{URL}/{a.json()['document']['id']}").status_code == 204

    names = {e.canonical_name for e in rows(session, Entity)}
    assert names == {"Acme Corp", "Bob Smith"}  # Alice, FastAPI, Pydantic had no other evidence
    assert {m.document_id for m in rows(session, EntityMention)} == {b.json()["document"]["id"]}
    rels = rows(session, Relationship)
    assert [r.relationship_type for r in rels] == ["WORKS_AT"]

    assert client.delete(f"{URL}/{b.json()['document']['id']}").status_code == 204
    for model in (Entity, EntityMention, Relationship):
        assert rows(session, model) == [], model.__name__


def test_extraction_provider_setting_selects_provider(tmp_path):
    application = create_app(
        Settings(data_dir=tmp_path, extraction_provider="mock", environment="test", _env_file=None)
    )
    assert application.state.worker.provider.name == "mock"
    application.state.worker.shutdown()
