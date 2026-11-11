"""Hybrid and graph retrieval stay valid while the database changes underneath them."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import delete, select

from app.models import Document, DocumentChunk, Relationship
from app.services.embeddings.revision import bump_vector_revision
from app.services.extraction.types import ChunkExtraction
from app.services.retrieval import hybrid as hybrid_module
from tests.graph_search_helpers import add_indexed_doc, seed_corpus
from tests.integration.test_evidence_recompute import E, R, store

HYBRID = "/api/v1/search/hybrid"
GRAPH = "/api/v1/search/graph"
QUERY = "Who works at Acme Corp?"


@pytest.fixture
def corpus(session):
    return seed_corpus(session)


def search(client, url=HYBRID, query=QUERY, **kw):
    r = client.post(url, json={"query": query, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def by_id(body):
    return {r["chunk_id"]: r for r in body["results"]}


def mutate(app, fn):
    """Run ``fn(session)`` in its own committed transaction, as another process would."""
    with app.state.session_factory() as s:
        fn(s)
        s.commit()


def assert_valid(app, body):
    """Every returned chunk exists now, is completed, and its relationship evidence exists."""
    with app.state.session_factory() as s:
        for r in body["results"]:
            chunk = s.get(DocumentChunk, r["chunk_id"])
            assert chunk is not None, f"returned a chunk that does not exist: {r['chunk_id']}"
            assert s.get(Document, chunk.document_id).processing_status == "completed"
            assert chunk.document_id == r["document_id"] and chunk.content == r["content"]
            for ev in (r.get("graph") or {}).get("evidence", []):
                if ev["kind"] == "relationship":
                    row = s.get(Relationship, ev["relationship_id"])
                    assert row is not None, "relationship evidence that does not exist"
                    assert (row.source_chunk_id, row.source_document_id) == (
                        r["chunk_id"],
                        r["document_id"],
                    )
                    assert (row.source_entity_id, row.target_entity_id) == (
                        ev["source"]["entity_id"],
                        ev["target"]["entity_id"],
                    )


# ------------------------------------------------- the graph snapshot is behind SQLite


def test_relationship_evidence_deleted_after_the_graph_was_built_is_dropped(
    client, corpus, app, monkeypatch
):
    stale = app.state.graph.snapshot()
    mutate(
        app,
        lambda s: s.execute(
            delete(Relationship).where(Relationship.source_document_id == corpus["d1"].id)
        ),
    )
    monkeypatch.setattr(app.state.graph, "snapshot", lambda: stale)

    graph = search(client, GRAPH)
    assert graph["dropped"]["stale_evidence"] == 2  # WORKS_AT and LOCATED_IN of document 1
    assert set(by_id(graph)) == {corpus["c2"].id}  # the chunk still exists, its evidence does not
    assert any("no longer confirmed" in w for w in graph["warnings"])
    assert_valid(app, graph)

    hybrid = search(client)
    assert hybrid["dropped"]["stale_evidence"] == 2
    assert by_id(hybrid)[corpus["c1"].id]["retrieved_by"] == ["vector"]  # found, but not by graph
    assert by_id(hybrid)[corpus["c1"].id]["graph"] is None
    assert by_id(hybrid)[corpus["c2"].id]["retrieved_by"] == ["vector", "graph"]
    assert any("behind the database" in w or "no longer confirmed" in w for w in hybrid["warnings"])
    assert_valid(app, hybrid)


def test_evidence_that_now_points_at_a_different_chunk_is_not_trusted(
    client, corpus, app, monkeypatch
):
    stale = app.state.graph.snapshot()

    def repoint(s):
        row = s.scalars(
            select(Relationship).where(
                Relationship.source_document_id == corpus["d1"].id,
                Relationship.relationship_type == "WORKS_AT",
            )
        ).one()
        row.source_chunk_id = corpus["c2"].id  # same row id, different chunk
        row.source_document_id = corpus["d2"].id

    mutate(app, repoint)
    monkeypatch.setattr(app.state.graph, "snapshot", lambda: stale)
    body = search(client, GRAPH)
    assert body["dropped"]["stale_evidence"] == 1
    for r in body["results"]:
        for ev in r["graph"]["evidence"]:
            assert ev["relationship_id"] is not None
    assert_valid(app, body)


def test_a_current_graph_drops_nothing(client, corpus, app):
    body = search(client)
    assert body["dropped"] == {
        "missing_chunks": 0,
        "out_of_scope": 0,
        "stale_evidence": 0,
        "stale_results": 0,
    }
    assert body["warnings"] == []


def test_the_graph_revision_lag_is_visible_in_the_path_report(client, corpus, app, monkeypatch):
    stale = app.state.graph.snapshot()
    doc, chunk = add_indexed_doc(corpus_session(app), "Dan works at Acme Corp.", 40)
    store(
        corpus_session(app),
        doc,
        chunk,
        ChunkExtraction(
            [E("Dan", "PERSON"), E("Acme Corp")],
            [R("Dan", "Acme Corp", "WORKS_AT", "Dan works at Acme Corp")],
        ),
    )
    monkeypatch.setattr(app.state.graph, "snapshot", lambda: stale)
    body = search(client)
    g = body["paths"]["graph"]
    assert g["graph_revision"] < g["database_revision"]
    assert any("graph snapshot used is behind the database" in w for w in body["warnings"])
    assert body["path_summary"]["is_complete_hybrid"] is True  # reported, not a failure
    assert_valid(app, body)


def corpus_session(app):
    """A long-lived session for building fixtures inside a test body."""
    if not hasattr(app.state, "_test_session"):
        app.state._test_session = app.state.session_factory()
    return app.state._test_session


# --------------------------------------------- the database changes between the two paths


def delete_after_vector_path(monkeypatch, app, document_id):
    real = hybrid_module.search_vectors

    def wrapped(*args, **kwargs):
        outcome = real(*args, **kwargs)  # the vector path has finished and hydrated its hits...
        mutate(app, lambda s: s.delete(s.get(Document, document_id)))  # ...then a delete lands
        return outcome

    monkeypatch.setattr(hybrid_module, "search_vectors", wrapped)


def test_a_document_deleted_after_both_paths_answered_is_dropped_before_the_response(
    client, corpus, app, monkeypatch
):
    delete_after_vector_path(monkeypatch, app, corpus["d1"].id)
    body = search(client)
    assert {r["document_id"] for r in body["results"]} == {corpus["d2"].id}
    assert body["dropped"]["stale_results"] == 1  # the chunk both paths had found
    assert [r["rank"] for r in body["results"]] == [1]
    assert any("1 fused result(s) were dropped" in w for w in body["warnings"])
    assert_valid(app, body)


def test_the_next_result_takes_the_place_of_a_dropped_one(
    client, session, corpus, app, monkeypatch
):
    for minute in range(4):  # 2 seeded + 4 more = 6 candidates for a limit of 5
        doc, chunk = add_indexed_doc(session, f"Eve{minute} works at Acme Corp.", 50 + minute)
        store(
            session,
            doc,
            chunk,
            ChunkExtraction(
                [E(f"Eve{minute}", "PERSON"), E("Acme Corp")],
                [R(f"Eve{minute}", "Acme Corp", "WORKS_AT", f"Eve{minute} works at Acme Corp")],
            ),
        )
    full = search(client, limit=5)
    assert len(full["results"]) == 5
    top = full["results"][0]
    delete_after_vector_path(monkeypatch, app, top["document_id"])
    after = search(client, limit=5)
    assert top["chunk_id"] not in by_id(after)
    assert len(after["results"]) == 5  # refilled from the fused candidates below the limit
    assert after["dropped"]["stale_results"] == 1
    assert [r["rank"] for r in after["results"]] == [1, 2, 3, 4, 5]
    assert_valid(app, after)


def test_a_document_that_stops_being_completed_while_the_request_runs_is_dropped(
    client, corpus, app, monkeypatch
):
    real = hybrid_module.search_vectors

    def wrapped(*args, **kwargs):
        outcome = real(*args, **kwargs)
        mutate(
            app,
            lambda s: setattr(s.get(Document, corpus["d1"].id), "processing_status", "processing"),
        )
        return outcome

    monkeypatch.setattr(hybrid_module, "search_vectors", wrapped)
    body = search(client)
    assert {r["document_id"] for r in body["results"]} == {corpus["d2"].id}
    assert body["dropped"]["stale_results"] == 1
    assert_valid(app, body)


def test_the_stored_text_is_what_is_returned_even_if_it_changed_during_the_request(
    client, corpus, app, monkeypatch
):
    real = hybrid_module.search_vectors

    def wrapped(*args, **kwargs):
        outcome = real(*args, **kwargs)
        mutate(
            app,
            lambda s: setattr(s.get(DocumentChunk, corpus["c1"].id), "section", "Moved section"),
        )
        return outcome

    monkeypatch.setattr(hybrid_module, "search_vectors", wrapped)
    body = search(client)
    assert by_id(body)[corpus["c1"].id]["source"]["section"] == "Moved section"


def test_stale_vector_index_and_stale_graph_together_still_return_only_valid_evidence(
    client, corpus, app, monkeypatch
):
    search(client)  # build the vector index
    stale_graph = app.state.graph.snapshot()

    def delete_document(s):
        s.delete(s.get(Document, corpus["d1"].id))
        bump_vector_revision(s)  # what a real delete does: the vector index is now behind

    mutate(app, delete_document)
    monkeypatch.setattr(app.state.graph, "snapshot", lambda: stale_graph)
    lock = app.state.vectors._sync_lock
    lock.acquire()  # the vector index cannot synchronize: it is serving its last good snapshot
    try:
        body = search(client)
    finally:
        lock.release()
    assert body["paths"]["vector"]["status"] == "stale"
    assert {r["document_id"] for r in body["results"]} == {corpus["d2"].id}
    assert_valid(app, body)
    assert body["path_summary"]["mode"] == "hybrid"


def test_an_unfinished_document_is_invisible_to_every_endpoint(client, session, corpus, app):
    doc, chunk = add_indexed_doc(session, "Zed works at Acme Corp.", 70)
    store(
        session,
        doc,
        chunk,
        ChunkExtraction(
            [E("Zed", "PERSON"), E("Acme Corp")],
            [R("Zed", "Acme Corp", "WORKS_AT", "Zed works at Acme Corp")],
        ),
    )
    doc.processing_status = "processing"
    session.commit()
    for url in (HYBRID, GRAPH, "/api/v1/search/vector"):
        body = search(client, url)
        assert chunk.id not in by_id(body), url
    doc.processing_status = "completed"
    session.commit()
    assert chunk.id in by_id(search(client))


# ---------------------------------------------------- concurrent ingestion and deletion


def test_searches_during_concurrent_ingestion_never_return_invalid_evidence(client, app, session):
    seed_corpus(session)
    stop = threading.Event()
    errors: list[str] = []
    bodies: list[dict] = []

    def writer():
        try:
            for i in range(10):
                with app.state.session_factory() as s:
                    doc, chunk = add_indexed_doc(s, f"Writer{i} works at Acme Corp.", 100 + i)
                    store(
                        s,
                        doc,
                        chunk,
                        ChunkExtraction(
                            [E(f"Writer{i}", "PERSON"), E("Acme Corp")],
                            [
                                R(
                                    f"Writer{i}",
                                    "Acme Corp",
                                    "WORKS_AT",
                                    f"Writer{i} works at Acme Corp",
                                )
                            ],
                        ),
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"writer: {exc!r}")
        finally:
            stop.set()

    def reader(n):
        try:
            while not stop.is_set() or n > 0:
                r = client.post(HYBRID, json={"query": QUERY, "limit": 20})
                if r.status_code != 200:
                    errors.append(f"HTTP {r.status_code}: {r.text[:200]}")
                    return
                bodies.append(r.json())
                n -= 1
                if stop.is_set() and n <= 0:
                    return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"reader: {exc!r}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(writer)] + [pool.submit(reader, 6) for _ in range(3)]
        for f in futures:
            f.result(timeout=120)
    assert errors == []
    assert len(bodies) >= 18
    for body in bodies:
        assert body["path_summary"]["mode"] in ("hybrid", "partial")
        assert_valid(app, body)
    final = search(client, limit=20)  # once ingestion stopped, both paths catch up
    assert len(final["results"]) == 12  # 2 seeded + 10 written
    assert all(r["retrieved_by"] == ["vector", "graph"] for r in final["results"])
    assert final["warnings"] == []
    assert final["paths"]["graph"]["graph_revision"] == final["paths"]["graph"]["database_revision"]


def test_searches_during_concurrent_deletion_never_return_a_deleted_chunk(client, app, session):
    seed_corpus(session)
    ids = []
    for i in range(6):
        doc, chunk = add_indexed_doc(session, f"Gone{i} works at Acme Corp.", 200 + i)
        store(
            session,
            doc,
            chunk,
            ChunkExtraction(
                [E(f"Gone{i}", "PERSON"), E("Acme Corp")],
                [R(f"Gone{i}", "Acme Corp", "WORKS_AT", f"Gone{i} works at Acme Corp")],
            ),
        )
        ids.append((doc.id, chunk.id))
    search(client, limit=20)  # indexes everything
    errors: list[str] = []
    bodies: list[dict] = []
    deleted: list[str] = []

    def deleter():
        for doc_id, chunk_id in ids:
            r = client.delete(f"/api/v1/documents/{doc_id}")
            if r.status_code != 204:
                errors.append(f"delete {r.status_code}")
            deleted.append(chunk_id)

    def reader():
        for _ in range(10):
            deleted_before = list(deleted)  # DELETE already answered 204 for these
            r = client.post(HYBRID, json={"query": QUERY, "limit": 20})
            if r.status_code != 200:
                errors.append(f"HTTP {r.status_code}: {r.text[:200]}")
                return
            bodies.append((deleted_before, r.json()))

    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in [pool.submit(deleter), pool.submit(reader), pool.submit(reader)]:
            f.result(timeout=120)
    assert errors == []
    assert len(bodies) == 20
    for deleted_before, body in bodies:
        # a chunk whose document was already deleted when the request started cannot appear in it
        assert not set(deleted_before) & set(by_id(body)), (deleted_before, list(by_id(body)))
        assert body["path_summary"]["mode"] in ("hybrid", "partial")
    final = search(client, limit=20)
    assert not ({c for _, c in ids} & set(by_id(final)))
    assert len(final["results"]) == 2
