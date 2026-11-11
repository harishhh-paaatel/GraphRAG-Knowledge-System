"""SQLite <-> graph consistency: updates, deletion, rebuild, persistence and recovery."""

import io
import json
import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.core.config import Settings
from app.main import create_app
from app.models import Base, Document, ProcessingJob, Relationship
from app.services.extraction.types import ChunkExtraction
from app.services.graph import store as graph_store
from app.services.graph.consistency import logical_form
from app.services.graph.service import GraphService
from app.services.graph.store import GraphPersistenceError, load_graph
from app.services.ingestion import service
from app.services.ingestion.pipeline import run_ingestion
from tests.integration.test_evidence_recompute import E, R, add_doc
from tests.integration.test_evidence_recompute import store as store_knowledge

URL = "/api/v1/documents"
DOC_A = b"Alice works at Acme Corp. FastAPI uses Pydantic."
DOC_B = b"Bob Smith joined Acme Corp. Acme Corp is located in Berlin."


@pytest.fixture
def upload(client, app):
    def _upload(name, data):
        r = client.post(f"{URL}/upload", files={"file": (name, data, "text/plain")})
        assert r.status_code == 202
        assert app.state.worker.drain(timeout=30)
        return r.json()["document"]["id"]

    return _upload


def names(graph):
    return sorted(d["canonical_name"] for _, d in graph.nodes(data=True))


def triples(graph):
    n = {i: d["canonical_name"] for i, d in graph.nodes(data=True)}
    return sorted((n[u], k, n[v]) for u, v, k in graph.edges(keys=True))


def node(graph, name):
    return next(d for _, d in graph.nodes(data=True) if d["canonical_name"] == name)


def file_logical(app):
    graph, _, _ = load_graph(app.state.settings.graph_file)
    return logical_form(graph)


def fresh_service(app):
    return GraphService(app.state.session_factory, app.state.settings.graph_file)


# ---- the graph follows the documents ----------------------------------------------------


def test_adding_a_document_updates_the_graph_and_the_file(client, app, upload):
    graph = app.state.graph
    before = graph.snapshot()
    assert before.graph.number_of_nodes() == 0

    upload("a.txt", DOC_A)

    snap = graph.snapshot()
    assert snap.revision > before.revision
    assert names(snap.graph) == ["Acme Corp", "Alice", "FastAPI", "Pydantic"]
    assert triples(snap.graph) == [
        ("Alice", "WORKS_AT", "Acme Corp"),
        ("FastAPI", "USES", "Pydantic"),
    ]
    assert file_logical(app) == logical_form(snap.graph)  # file written eagerly after ingest
    assert graph.status()["file_in_sync"] is True


def test_second_document_shares_entities_and_aggregates_evidence(client, app, upload):
    a = upload("a.txt", DOC_A)
    b = upload("b.txt", DOC_B)
    g = app.state.graph.snapshot().graph
    acme = node(g, "Acme Corp")
    assert acme["mention_count"] == 2 and acme["source_document_ids"] == sorted([a, b])
    assert names(g).count("Acme Corp") == 1
    assert ("Bob Smith", "WORKS_AT", "Acme Corp") in triples(g)
    assert ("Acme Corp", "LOCATED_IN", "Berlin") in triples(g)


def test_reprocessing_a_document_never_duplicates_nodes_or_edges(client, app, upload, session):
    a = upload("a.txt", DOC_A)
    before = app.state.graph.snapshot().graph
    snapshot = logical_form(before)
    for attempt in (2, 3):
        job = ProcessingJob(document_id=a, status="pending", attempt=attempt)
        session.add(job)
        session.commit()
        run_ingestion(app.state.session_factory, app.state.settings, job.id, None, app.state.graph)
    after = app.state.graph.snapshot().graph
    assert after.number_of_nodes() == before.number_of_nodes()
    assert after.number_of_edges() == before.number_of_edges()
    assert logical_form(after) == snapshot  # same IDs, same evidence, nothing doubled
    for _, _, d in after.edges(data=True):
        assert len(d["supporting_evidence"]) == 1


def test_deleting_a_document_removes_its_unsupported_relationships(client, app, upload):
    a = upload("a.txt", DOC_A)
    upload("b.txt", DOC_B)

    assert client.delete(f"{URL}/{a}").status_code == 204

    g = app.state.graph.snapshot().graph
    # Alice, FastAPI and Pydantic were evidenced only by document A.
    assert names(g) == ["Acme Corp", "Berlin", "Bob Smith"]
    assert triples(g) == [
        ("Acme Corp", "LOCATED_IN", "Berlin"),
        ("Bob Smith", "WORKS_AT", "Acme Corp"),
    ]
    acme = node(g, "Acme Corp")  # shared entity survives, now supported by one document
    assert acme["mention_count"] == 1  # only document B's mention remains
    assert len(acme["source_document_ids"]) == 1
    assert file_logical(app) == logical_form(g)


def test_shared_edge_keeps_remaining_evidence_when_one_document_goes(client, app, upload):
    a = upload("a.txt", b"Alice works at Acme Corp.")
    upload("b.txt", b"Alice works at Acme Corp, said the report. Alice works at Acme Corp again.")
    g = app.state.graph.snapshot().graph
    [edge] = [d for *_, d in g.edges(data=True)]
    assert len(edge["source_document_ids"]) == 2

    client.delete(f"{URL}/{a}")

    g2 = app.state.graph.snapshot().graph
    [edge2] = [d for *_, d in g2.edges(data=True)]
    assert len(edge2["source_document_ids"]) == 1 and a not in edge2["source_document_ids"]
    assert all(e["document_id"] != a for e in edge2["supporting_evidence"])


def test_deleting_the_last_supporting_document_removes_every_graph_record(client, app, upload):
    a = upload("a.txt", DOC_A)
    b = upload("b.txt", DOC_B)
    client.delete(f"{URL}/{a}")
    client.delete(f"{URL}/{b}")
    g = app.state.graph.snapshot().graph
    assert g.number_of_nodes() == 0 and g.number_of_edges() == 0
    assert file_logical(app) == {"nodes": {}, "edges": {}}


def test_entity_metadata_changes_are_reflected_in_graph_nodes(session, settings, app):
    svc = app.state.graph
    d1, c1 = add_doc(session, "OpenAI is an AI lab.", minute=0)
    d2, c2 = add_doc(session, "Open AI is a research company.", minute=1)
    store_knowledge(session, d1, c1, ChunkExtraction([E("OpenAI", desc="AI lab")], []))
    store_knowledge(session, d2, c2, ChunkExtraction([E("Open AI", desc="Research company")], []))
    [n] = [d for _, d in svc.snapshot().graph.nodes(data=True)]
    assert (n["canonical_name"], n["description"], n["aliases"]) == (
        "OpenAI",
        "AI lab",
        ["Open AI"],
    )

    service.delete_document(session, settings, d1.id)  # no explicit sync: reads must self-heal

    [n2] = [d for _, d in svc.snapshot().graph.nodes(data=True)]
    assert n2["entity_id"] == n["entity_id"]  # same node, deterministic ID
    assert (n2["canonical_name"], n2["description"], n2["aliases"]) == (
        "Open AI",
        "Research company",
        [],
    )


# ---- rebuild equals the live graph ------------------------------------------------------


def test_rebuild_from_sqlite_produces_the_same_logical_graph(client, app, upload):
    upload("a.txt", DOC_A)
    upload("b.txt", DOC_B)
    live = logical_form(app.state.graph.snapshot().graph)

    other = fresh_service(app)  # a brand-new process knowing nothing
    other.rebuild()
    assert logical_form(other.snapshot().graph) == live

    report = app.state.graph.check_consistency()
    assert report["memory"]["consistent"] and report["file"]["consistent"]


def test_consistency_check_detects_drift_and_rebuild_repairs_it(client, app, upload, session):
    upload("a.txt", DOC_A)
    svc = app.state.graph
    svc.snapshot()
    # Corrupt SQLite behind the graph's back: no revision bump, so nothing triggers a rebuild.
    session.execute(delete(Relationship))
    session.commit()

    report = svc.check_consistency()
    assert not report["memory"]["consistent"] and len(report["memory"]["extra_edges"]) == 2
    assert not report["file"]["consistent"]

    svc.rebuild()
    repaired = svc.check_consistency()
    assert repaired["memory"]["consistent"] and repaired["file"]["consistent"]
    assert svc.snapshot().graph.number_of_edges() == 0


# ---- persistence and recovery ----------------------------------------------------------


def test_startup_adopts_a_valid_current_file_without_rebuilding(client, app, upload, monkeypatch):
    upload("a.txt", DOC_A)
    expected = logical_form(app.state.graph.snapshot().graph)

    def must_not_run(*_a, **_k):
        raise AssertionError("graph was rebuilt although the file was valid and current")

    monkeypatch.setattr("app.services.graph.service.build_graph", must_not_run)
    fresh = fresh_service(app)
    report = fresh.initialize()
    assert report["source"] == "file"
    assert logical_form(fresh.snapshot().graph) == expected


def test_missing_file_is_recovered_by_rebuilding_from_sqlite(client, app, upload):
    upload("a.txt", DOC_A)
    expected = logical_form(app.state.graph.snapshot().graph)
    path = app.state.settings.graph_file
    path.unlink()

    fresh = fresh_service(app)
    report = fresh.initialize()
    assert report["source"] == "rebuild" and "does not exist" in report["reason"]
    assert logical_form(fresh.snapshot().graph) == expected
    assert path.exists() and logical_form(load_graph(path)[0]) == expected  # file restored


@pytest.mark.parametrize("damage", ["garbage", "truncated", "empty", "tampered", "wrong_format"])
def test_corrupted_file_is_quarantined_and_recovered(client, app, upload, damage):
    upload("a.txt", DOC_A)
    upload("b.txt", DOC_B)
    expected = logical_form(app.state.graph.snapshot().graph)
    path = app.state.settings.graph_file
    raw = path.read_bytes()
    if damage == "garbage":
        bad = b"\x00\x01 definitely not json"
    elif damage == "truncated":
        bad = raw[: len(raw) // 3]
    elif damage == "empty":
        bad = b""
    elif damage == "tampered":
        bad = raw.replace(b"Acme Corp", b"Evil Corp", 1)
    else:
        bad = json.dumps({"hello": "world"}).encode()
    path.write_bytes(bad)

    fresh = fresh_service(app)
    report = fresh.initialize()

    assert report["source"] == "rebuild"
    assert logical_form(fresh.snapshot().graph) == expected  # nothing lost: SQLite was the truth
    assert logical_form(load_graph(path)[0]) == expected  # valid file written back
    assert path.with_name(path.name + ".corrupt").read_bytes() == bad  # evidence kept, not deleted
    assert fresh.check_consistency()["file"]["consistent"]


def test_stale_file_after_a_crash_between_commit_and_graph_update_is_rebuilt(
    client, app, upload, session
):
    upload("a.txt", DOC_A)
    old_revision = load_graph(app.state.settings.graph_file)[1]

    # Simulate a crash: SQLite commits new knowledge, but the process dies before the graph syncs.
    d, c = add_doc(session, "Zed works at Globex.", minute=50)
    store_knowledge(
        session,
        d,
        c,
        ChunkExtraction(
            [E("Zed", "PERSON"), E("Globex")],
            [R("Zed", "Globex", "WORKS_AT", "Zed works at Globex")],
        ),
    )
    assert load_graph(app.state.settings.graph_file)[1] == old_revision  # file is behind

    fresh = fresh_service(app)
    report = fresh.initialize()
    assert report["source"] == "rebuild" and "revision" in report["reason"]
    assert ("Zed", "WORKS_AT", "Globex") in triples(fresh.snapshot().graph)
    assert load_graph(app.state.settings.graph_file)[1] > old_revision


def test_reads_never_serve_a_stale_graph_even_without_an_explicit_sync(app, session):
    svc = app.state.graph
    svc.initialize()
    assert svc.snapshot().graph.number_of_nodes() == 0
    d, c = add_doc(session, "Zed works at Globex.", minute=0)
    store_knowledge(session, d, c, ChunkExtraction([E("Zed", "PERSON"), E("Globex")], []))
    assert names(svc.snapshot().graph) == ["Globex", "Zed"]  # first read after the write


def test_failed_file_write_keeps_memory_current_and_the_old_file_valid(app, session, monkeypatch):
    svc = app.state.graph
    svc.initialize()
    path = app.state.settings.graph_file
    old_revision = load_graph(path)[1]

    def failing_save(*_a, **_k):
        raise GraphPersistenceError("simulated disk failure")

    monkeypatch.setattr("app.services.graph.service.save_graph", failing_save)
    d, c = add_doc(session, "Zed works at Globex.", minute=0)
    store_knowledge(session, d, c, ChunkExtraction([E("Zed", "PERSON"), E("Globex")], []))
    assert svc.sync() is True

    status = svc.status()
    assert status["state"] == "degraded" and not status["file_in_sync"]
    assert "simulated disk failure" in status["persist_error"]
    assert names(svc.snapshot().graph) == ["Globex", "Zed"]  # memory is current
    assert load_graph(path)[1] == old_revision  # previous file untouched and still loadable

    monkeypatch.undo()
    assert svc.sync() is False  # nothing to rebuild, but the pending write is retried
    status = svc.status()
    assert status["state"] == "ok" and status["file_in_sync"] and status["persist_error"] is None
    assert load_graph(path)[1] == svc.snapshot().revision


def test_health_reports_the_graph_and_flags_persistence_failures(client, app, upload, monkeypatch):
    body = client.get("/api/v1/health").json()
    assert body["components"]["graph_index"]["status"] == "ok"
    assert "nodes=0" in body["components"]["graph_index"]["detail"]

    def failing_save(*_a, **_k):
        raise GraphPersistenceError("simulated disk failure")

    monkeypatch.setattr("app.services.graph.service.save_graph", failing_save)
    upload("a.txt", DOC_A)  # ingestion still succeeds; only the file write fails
    body = client.get("/api/v1/health").json()
    assert body["components"]["graph_index"]["status"] == "error"
    assert body["status"] == "degraded"
    assert client.get(f"{URL}").json()["items"][0]["processing_status"] == "completed"


def test_unmigrated_database_makes_the_graph_unavailable_not_the_app(tmp_path):
    application = create_app(Settings(data_dir=tmp_path, environment="test", _env_file=None))
    report = application.state.graph.initialize()  # no tables exist
    assert report["source"] == "unavailable"
    application.state.worker.shutdown()
    application.state.engine.dispose()


def test_application_restart_restores_the_graph_from_its_file(tmp_path):
    settings = Settings(data_dir=tmp_path, environment="test", _env_file=None)
    first = create_app(settings)
    Base.metadata.create_all(first.state.engine)
    with TestClient(first) as c:
        r = c.post(f"{URL}/upload", files={"file": ("a.txt", DOC_A, "text/plain")})
        assert r.status_code == 202
        first.state.worker.drain(30)
        expected = logical_form(first.state.graph.snapshot().graph)
    first.state.engine.dispose()

    second = create_app(settings)
    with TestClient(second) as c:
        body = c.get("/api/v1/graph/overview").json()
        assert second.state.graph.status()["initialized_from"] == "file"
        assert body["stats"]["node_count"] == len(expected["nodes"]) == 4
        assert logical_form(second.state.graph.snapshot().graph) == expected
    second.state.engine.dispose()


def test_concurrent_ingestion_leaves_graph_memory_and_file_consistent(tmp_path):
    settings = Settings(data_dir=tmp_path, worker_threads=4, environment="test", _env_file=None)
    application = create_app(settings)
    Base.metadata.create_all(application.state.engine)
    with TestClient(application) as c:
        for i in range(8):
            body = f"Doc {i}. Alice works at Acme Corp. FastAPI uses Pydantic. Note {i}.".encode()
            assert (
                c.post(
                    f"{URL}/upload", files={"file": (f"{i}.txt", body, "text/plain")}
                ).status_code
                == 202
            )
        assert application.state.worker.drain(60)
        graph = application.state.graph
        graph.sync()
        report = graph.check_consistency()
        assert report["memory"]["consistent"] and report["file"]["consistent"]
        g = graph.snapshot().graph
        assert g.number_of_nodes() == 4 and g.number_of_edges() == 2
        [edge] = [d for u, v, d in g.edges(data=True) if d["relationship_type"] == "WORKS_AT"]
        assert len(edge["source_document_ids"]) == 8 and len(edge["supporting_evidence"]) == 8
    application.state.engine.dispose()


def test_concurrent_readers_and_a_writer_never_see_a_torn_graph(client, app, upload):
    upload("a.txt", DOC_A)
    errors: list[str] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            g = app.state.graph.snapshot().graph
            for u, v in g.edges():  # every edge endpoint must be a node of the same snapshot
                if u not in g or v not in g:
                    errors.append("dangling edge")

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for i in range(4):
        upload(f"x{i}.txt", f"Person{i} X works at Org{i} Corp. Doc {i}.".encode())
    stop.set()
    for t in threads:
        t.join(10)
    assert errors == []
    assert app.state.graph.check_consistency()["memory"]["consistent"]


def test_files_on_disk_do_not_leak_temp_artifacts(client, app, upload):
    upload("a.txt", DOC_A)
    upload("b.txt", DOC_B)
    leftovers = [p.name for p in app.state.settings.graph_file.parent.iterdir()]
    assert leftovers == ["knowledge_graph.json"]


def test_service_function_still_works_with_documents_without_knowledge(
    client, app, session, settings
):
    # A document that never produced entities must not break deletion or the graph.
    r = service.accept_upload(
        session, settings, io.BytesIO(b"nothing notable here"), "n.txt", "text/plain"
    )
    doc = r[0]
    run_ingestion(app.state.session_factory, settings, r[1].id, None, app.state.graph)
    assert app.state.graph.snapshot().graph.number_of_nodes() == 0
    session.expire_all()  # the worker updated the row in another session
    service.delete_document(session, settings, doc.id)
    assert session.scalars(select(Document)).all() == []
    assert graph_store.load_graph(settings.graph_file)[0].number_of_nodes() == 0
