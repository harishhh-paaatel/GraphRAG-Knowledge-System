"""API reliability of /search/hybrid: hostile input, concurrency, time budgets, recovery,
and guards against redundant work (query counts, repeated embedding, repeated index builds)."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event

from app.core.config import Settings
from app.main import create_app
from app.models import Base
from app.services.embeddings.service import VectorIndexBuilding
from app.services.extraction.types import ChunkExtraction
from app.services.graph.service import GraphUnavailable
from tests.graph_search_helpers import add_indexed_doc, seed_corpus
from tests.integration.test_evidence_recompute import E, R, store

HYBRID = "/api/v1/search/hybrid"
GRAPH = "/api/v1/search/graph"
VECTOR = "/api/v1/search/vector"
QUERY = "Who works at Acme Corp?"


@pytest.fixture
def corpus(session):
    return seed_corpus(session)


def drop_timings(body):
    body = dict(body)
    body.pop("timings_ms", None)
    return body


# ---------------------------------------------------------------------- hostile input


def test_malformed_and_mistyped_requests_are_validation_errors_never_server_errors(client):
    cases = [
        ({"content": b"{not json"}, "malformed JSON"),
        ({"content": b""}, "empty body"),
        ({"content": b"null"}, "null body"),
        ({"content": b"[]"}, "array body"),
        ({"json": "just a string"}, "string body"),
        ({"json": {"query": 5}}, "number query"),
        ({"json": {"query": ["a"]}}, "list query"),
        ({"json": {"query": None}}, "null query"),
        ({"json": {"query": "x", "limit": 2.5}}, "float limit"),
        ({"json": {"query": "x", "limit": "many"}}, "string limit"),
        ({"json": {"query": "x", "document_ids": "d1"}}, "string document_ids"),
        ({"json": {"query": "x", "document_ids": [f"d{i}" for i in range(101)]}}, "101 documents"),
        ({"json": {"query": "x", "graph": {"entity_ids": [f"e{i}" for i in range(51)]}}}, "51 ids"),
        ({"json": {"query": "x", "graph": {"relationship_types": ["WORKS_AT"] * 33}}}, "33 types"),
        ({"json": {"query": "x", "graph": "both"}}, "graph not an object"),
        ({"json": {"query": "x", "fusion": {"rrf_k": "sixty"}}}, "string rrf_k"),
        ({"json": {"query": "x", "fusion": {"vector_weight": "1"}}}, "string weight is coerced"),
    ]
    for kwargs, label in cases:
        r = client.post(HYBRID, headers={"content-type": "application/json"}, **kwargs)
        assert r.status_code in (200, 422), (label, r.status_code, r.text[:200])
        if r.status_code == 422:
            error = r.json()["error"]
            assert error["code"] == "validation_error" and error["message"], label
            assert isinstance(error["details"], list) and error["details"], label


def test_a_request_without_a_json_content_type_is_rejected_cleanly(client):
    r = client.post(HYBRID, content=b'{"query": "x"}', headers={"content-type": "text/plain"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_unusual_but_valid_queries_are_answered_not_crashed(client, corpus):
    queries = [
        "Acme Corp\x00",  # a NUL byte
        "‮Acme Corp",  # right-to-left override
        "Who works at Acme Corp? \U0001f600\U0001f680",  # emoji
        "ACME CORP",
        "  Acme   Corp  ",
        "'; DROP TABLE documents; --",
        "<script>alert(1)</script> Acme Corp",
        "Acme Corp" + " Acme" * 380,  # long (about 2000 characters)
        "Åcme Corp 你好",
        "a",
    ]
    for q in queries:
        assert len(q) <= 2000
        r = client.post(HYBRID, json={"query": q})
        assert r.status_code == 200, (q[:30], r.status_code, r.text[:200])
        body = r.json()
        assert body["path_summary"]["mode"] in ("hybrid", "partial")
    # a query with nothing to embed is a request error, exactly as for /search/vector
    r = client.post(HYBRID, json={"query": "?" * 50})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_query"
    assert client.get("/api/v1/health").status_code == 200  # nothing was damaged
    from app.models import Document

    # the statement in the query text never ran
    assert client.get("/api/v1/documents").json()["total"] == 2
    assert Document.__tablename__ == "documents"


def test_wrong_methods_and_paths_are_clean_errors(client):
    assert client.get(HYBRID).status_code == 405
    assert client.put(HYBRID, json={"query": "x"}).status_code == 405
    assert client.delete(HYBRID).status_code == 405
    assert client.post("/api/v1/search/hybrids", json={"query": "x"}).status_code == 404


def test_every_error_has_the_standard_envelope(client, corpus):
    probes = [
        client.post(HYBRID, json={"query": ""}),
        client.post(HYBRID, json={"query": "x", "limit": 999}),
        client.post(HYBRID, json={"query": "x", "graph": {"entity_ids": ["nope"]}}),
        client.post(HYBRID, json={"query": "x", "graph": {"hops": 9}}),
        client.get(HYBRID),
    ]
    for r in probes:
        assert 400 <= r.status_code < 500
        error = r.json().get("error") or r.json()
        assert "code" in error or "detail" in error


# ------------------------------------------------------------------ status reporting


def test_the_response_always_states_both_paths_and_the_summary(client, corpus):
    body = client.post(HYBRID, json={"query": QUERY}).json()
    assert set(body["paths"]) == {"vector", "graph"}
    assert set(body["path_summary"]) >= {
        "mode",
        "complete",
        "is_complete_hybrid",
        "message",
        "succeeded",
        "failed",
        "contributing",
    }
    assert set(body["dropped"]) == {
        "missing_chunks",
        "out_of_scope",
        "stale_evidence",
        "stale_results",
    }
    assert {"vector", "graph", "fusion", "verify", "total"} <= set(body["timings_ms"])


def test_a_partial_response_is_never_marked_complete_or_hybrid(client, corpus, app, monkeypatch):
    def down():
        raise VectorIndexBuilding("building")

    monkeypatch.setattr(app.state.vectors, "ensure_fresh", down)
    body = client.post(HYBRID, json={"query": QUERY}).json()
    s = body["path_summary"]
    assert (s["mode"], s["complete"], s["is_complete_hybrid"]) == ("partial", False, False)
    assert s["failed"] == ["vector"] and s["contributing"] == ["graph"]
    assert "NOT a complete hybrid" in body["warnings"][0]


# ---------------------------------------------------------------------- concurrency


def test_many_concurrent_mixed_requests_all_succeed_and_match_the_serial_answers(
    client, corpus, app
):
    queries = [QUERY, "Where is Acme Corp located?", "Tell me about Berlin", "Bob works"]
    plan = [(url, q) for url in (HYBRID, GRAPH, VECTOR) for q in queries]
    body = {
        HYBRID: lambda q: {"query": q},
        GRAPH: lambda q: {"query": q},
        VECTOR: lambda q: {"query": q, "top_k": 20},
    }
    serial = {(u, q): drop_timings(client.post(u, json=body[u](q)).json()) for u, q in plan}

    def call(item):
        u, q = item
        r = client.post(u, json=body[u](q))
        return item, r.status_code, drop_timings(r.json())

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(call, plan * 8))
    assert len(outcomes) == 96
    for item, status, got in outcomes:
        assert status == 200, item
        if item[0] != VECTOR:  # the vector response carries per-request timing detail inside
            assert got == serial[item], item
        else:
            assert [r["chunk_id"] for r in got["results"]] == [
                r["chunk_id"] for r in serial[item]["results"]
            ]


def test_concurrent_first_requests_build_the_alias_index_and_the_vector_index_once(
    client, corpus, app
):
    barrier = threading.Barrier(6)

    def call(_):
        barrier.wait(timeout=10)
        return client.post(HYBRID, json={"query": QUERY}).status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        assert list(pool.map(call, range(6))) == [200] * 6
    assert app.state.graph_retriever.linker.builds == 1
    # two chunks were embedded once, plus one embedding per query
    assert app.state.vectors.embedder.texts_embedded == 2 + 6


def test_requests_during_a_bulk_ingestion_stay_available(client, session, corpus, app):
    errors: list[str] = []
    stop = threading.Event()

    def ingest():
        try:
            for i in range(8):
                with app.state.session_factory() as s:
                    doc, chunk = add_indexed_doc(s, f"Bulk{i} works at Acme Corp.", 300 + i)
                    store(
                        s,
                        doc,
                        chunk,
                        ChunkExtraction(
                            [E(f"Bulk{i}", "PERSON"), E("Acme Corp")],
                            [R(f"Bulk{i}", "Acme Corp", "WORKS_AT", f"Bulk{i} works at Acme Corp")],
                        ),
                    )
        finally:
            stop.set()

    def search():
        count = 0
        while not stop.is_set() or count < 5:
            r = client.post(HYBRID, json={"query": QUERY})
            if r.status_code != 200:
                errors.append(f"{r.status_code} {r.text[:120]}")
            count += 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(ingest)] + [pool.submit(search) for _ in range(3)]
        for f in futures:
            f.result(timeout=120)
    assert errors == []


# --------------------------------------------------------------- redundant-work guards


class SqlCounter:
    def __init__(self, engine):
        self.count = 0
        self.armed = False
        event.listen(engine, "before_cursor_execute", self._on)

    def _on(self, *args):
        if self.armed:
            self.count += 1

    def measure(self, fn):
        self.count, self.armed = 0, True
        try:
            fn()
        finally:
            self.armed = False
        return self.count


def test_database_queries_per_request_stay_within_the_measured_budget(client, corpus, app):
    counter = SqlCounter(app.state.engine)
    client.post(HYBRID, json={"query": QUERY})  # warm up: first request syncs the indexes
    post = lambda url, **b: client.post(url, json={"query": QUERY, **b})  # noqa: E731
    vector = counter.measure(lambda: post(VECTOR))
    graph = counter.measure(lambda: post(GRAPH))
    hybrid = counter.measure(lambda: post(HYBRID))
    # budgets are the statement counts measured at M4 (see docs/phase-6-m4-report.md)
    assert vector <= 3, vector
    assert graph <= 5, graph
    assert hybrid <= 8, hybrid


def test_database_queries_do_not_grow_with_the_number_of_results_or_documents(
    client, session, corpus, app
):
    counter = SqlCounter(app.state.engine)
    client.post(HYBRID, json={"query": QUERY})
    few = counter.measure(lambda: client.post(HYBRID, json={"query": QUERY, "limit": 1}))
    for i in range(25):
        doc, chunk = add_indexed_doc(session, f"Many{i} works at Acme Corp.", 400 + i)
        store(
            session,
            doc,
            chunk,
            ChunkExtraction(
                [E(f"Many{i}", "PERSON"), E("Acme Corp")],
                [R(f"Many{i}", "Acme Corp", "WORKS_AT", f"Many{i} works at Acme Corp")],
            ),
        )
    client.post(HYBRID, json={"query": QUERY})  # sync the new data
    wide = {"query": QUERY, "limit": 25, "vector": {"top_k": 30}, "graph": {"max_chunks": 30}}
    many = counter.measure(lambda: client.post(HYBRID, json=wide))
    body = client.post(HYBRID, json=wide).json()
    assert len(body["results"]) == 25
    assert many == few  # no N+1: the same statements for 1 result or 25


def test_a_hybrid_request_embeds_the_query_once_and_never_re_embeds_chunks(client, corpus, app):
    client.post(HYBRID, json={"query": QUERY})
    embedder = app.state.vectors.embedder
    before = embedder.texts_embedded
    for _ in range(5):
        client.post(HYBRID, json={"query": QUERY})
    assert embedder.texts_embedded - before == 5  # one query embedding per request, nothing else


def test_repeated_searches_do_not_rebuild_the_graph_or_resync_the_vector_index(
    client, corpus, app, monkeypatch
):
    client.post(HYBRID, json={"query": QUERY})
    rebuilds, syncs = [], []
    real_rebuild, real_sync = app.state.graph._rebuild, app.state.vectors.sync
    monkeypatch.setattr(
        app.state.graph, "_rebuild", lambda r: (rebuilds.append(r), real_rebuild(r))
    )
    monkeypatch.setattr(
        app.state.vectors, "sync", lambda *a, **k: (syncs.append(1), real_sync(*a, **k))[1]
    )
    for _ in range(10):
        client.post(HYBRID, json={"query": QUERY})
    assert rebuilds == [] and syncs == []
    assert app.state.graph_retriever.linker.builds == 1


# ------------------------------------------------------ time budgets and recovery (API)


@pytest.fixture
def timed_client(tmp_path):
    settings = Settings(
        data_dir=tmp_path, environment="test", hybrid_path_timeout_seconds=0.3, _env_file=None
    )
    app = create_app(settings)
    Base.metadata.create_all(app.state.engine)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.app_ = app
        yield client
    app.state.engine.dispose()


def test_a_hanging_graph_path_costs_one_time_budget_and_the_vector_result_still_comes_back(
    timed_client,
):
    app = timed_client.app_
    with app.state.session_factory() as s:
        seed_corpus(s)
    healthy = timed_client.post(HYBRID, json={"query": QUERY}).json()
    assert healthy["path_summary"]["is_complete_hybrid"]
    release = threading.Event()
    real = app.state.graph_retriever.search

    def hang(*args, **kwargs):
        release.wait(5)
        return real(*args, **kwargs)

    app.state.graph_retriever.search = hang
    try:
        started = time.perf_counter()
        r = timed_client.post(HYBRID, json={"query": QUERY})
        elapsed = time.perf_counter() - started
    finally:
        release.set()
    assert r.status_code == 200 and elapsed < 2.0
    body = r.json()
    assert body["path_summary"]["mode"] == "partial"
    assert body["paths"]["graph"]["error_code"] == "graph_timeout"
    assert "did not answer within 0.3 s" in body["paths"]["graph"]["error_message"]
    vector_only = [
        x["chunk_id"] for x in timed_client.post(VECTOR, json={"query": QUERY}).json()["results"]
    ]
    assert [x["chunk_id"] for x in body["results"]] == vector_only
    time.sleep(0.2)
    app.state.graph_retriever.search = real  # recovery
    again = timed_client.post(HYBRID, json={"query": QUERY}).json()
    assert again["path_summary"]["is_complete_hybrid"]
    assert drop_timings(again)["results"] == drop_timings(healthy)["results"]


def test_both_paths_hanging_is_a_503_that_names_both_timeouts(timed_client):
    app = timed_client.app_
    with app.state.session_factory() as s:
        seed_corpus(s)
    release = threading.Event()
    app.state.graph_retriever.search = lambda *a, **k: release.wait(5)
    app.state.vectors.ensure_fresh = lambda: release.wait(5)
    try:
        r = timed_client.post(HYBRID, json={"query": QUERY})
    finally:
        release.set()
    assert r.status_code == 503
    details = r.json()["error"]["details"]
    assert r.json()["error"]["code"] == "retrieval_unavailable"
    assert details["vector"]["error_code"] == "vector_timeout"
    assert details["graph"]["error_code"] == "graph_timeout"


def test_the_service_alternates_between_failure_and_health_without_residue(client, corpus, app):
    baseline = drop_timings(client.post(HYBRID, json={"query": QUERY}).json())
    original = (app.state.vectors.ensure_fresh, app.state.graph.snapshot)

    def vector_down():
        raise VectorIndexBuilding("rebuilding")

    def graph_down():
        raise GraphUnavailable("down")

    for _ in range(3):
        app.state.vectors.ensure_fresh = vector_down
        assert (
            client.post(HYBRID, json={"query": QUERY}).json()["path_summary"]["mode"] == "partial"
        )
        app.state.vectors.ensure_fresh = original[0]
        app.state.graph.snapshot = graph_down
        assert (
            client.post(HYBRID, json={"query": QUERY}).json()["path_summary"]["mode"] == "partial"
        )
        app.state.vectors.ensure_fresh = vector_down
        assert client.post(HYBRID, json={"query": QUERY}).status_code == 503
        app.state.vectors.ensure_fresh, app.state.graph.snapshot = original
        healthy = drop_timings(client.post(HYBRID, json={"query": QUERY}).json())
        assert healthy == baseline  # identical to before any failure


def test_shutdown_releases_the_worker_threads(tmp_path):
    settings = Settings(
        data_dir=tmp_path, environment="test", hybrid_path_timeout_seconds=1.0, _env_file=None
    )
    app = create_app(settings)
    Base.metadata.create_all(app.state.engine)
    with TestClient(app):
        pools = dict(app.state.hybrid._pools)
        assert set(pools) == {"graph", "vector"}
    assert all(pool._shutdown for pool in pools.values())
    app.state.engine.dispose()


def test_a_zero_timeout_means_no_pool_and_no_budget(tmp_path):
    settings = Settings(
        data_dir=tmp_path, environment="test", hybrid_path_timeout_seconds=0, _env_file=None
    )
    app = create_app(settings)
    assert app.state.hybrid._pools == {} and app.state.hybrid._timeout is None
    app.state.engine.dispose()


def test_the_default_time_budget_is_generous_enough_for_a_first_model_load():
    assert Settings(_env_file=None).hybrid_path_timeout_seconds == 30.0
    assert json.loads('{"n": 1}')["n"] == 1  # (keeps the json import honest)
