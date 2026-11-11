"""POST /api/v1/search/hybrid against real SQLite data, the real FAISS index (hashing stub
embedder, no model) and the real graph service."""

import pytest

from app.models import Document
from app.services.embeddings.errors import EmbeddingError, EmbeddingInputError
from app.services.embeddings.service import VectorIndexBuilding
from app.services.extraction.normalization import squash
from app.services.extraction.resolver import make_entity_id
from app.services.extraction.types import ChunkExtraction
from app.services.graph.service import GraphUnavailable
from tests.graph_search_helpers import add_indexed_doc, seed_corpus
from tests.integration.test_evidence_recompute import E, R, store

URL = "/api/v1/search/hybrid"
QUERY = "Who works at Acme Corp?"
ALICE = make_entity_id("PERSON", "alice")


def hybrid(client, query=QUERY, *, expect=200, **kw):
    r = client.post(URL, json={"query": query, **kw})
    assert r.status_code == expect, r.text
    return r.json()


def standalone_vector(client, query=QUERY, **kw):
    r = client.post("/api/v1/search/vector", json={"query": query, "top_k": 20, **kw})
    assert r.status_code == 200, r.text
    return r.json()["results"]


def standalone_graph(client, query=QUERY, **kw):
    r = client.post("/api/v1/search/graph", json={"query": query, "graph": kw})
    assert r.status_code == 200, r.text
    return r.json()["results"]


def by_id(results):
    return {r["chunk_id"]: r for r in results}


@pytest.fixture
def corpus(session):
    return seed_corpus(session)


def check_traceable(session, body):
    for r in body["results"]:
        assert r["retrieved_by"], "a result no path claims"
        assert r["vector"] is not None or r["graph"] is not None
        assert session.get(Document, r["document_id"]).filename == r["source"]["filename"]
        if r["graph"] is not None:
            assert r["graph"]["evidence"]
            for ev in r["graph"]["evidence"]:
                assert squash(ev["supporting_text"] or ev["surface_form"]) in squash(r["content"])


# ------------------------------------------------------------------ both paths succeed


def test_a_hybrid_search_fuses_both_paths_and_says_it_is_a_complete_hybrid(client, session, corpus):
    body = hybrid(client)
    s = body["path_summary"]
    assert (s["mode"], s["complete"], s["is_complete_hybrid"]) == ("hybrid", True, True)
    assert s["succeeded"] == ["vector", "graph"] and s["failed"] == []
    assert s["contributing"] == ["vector", "graph"]
    assert body["paths"]["vector"]["status"] == "ok" and body["paths"]["graph"]["status"] == "ok"
    assert body["warnings"] == []
    check_traceable(session, body)


def test_a_chunk_found_by_both_paths_appears_once_with_both_blocks(client, corpus):
    body = hybrid(client)
    ids = [r["chunk_id"] for r in body["results"]]
    assert set(ids) == {corpus["c1"].id, corpus["c2"].id} and len(ids) == len(set(ids))
    for r in body["results"]:
        assert r["retrieved_by"] == ["vector", "graph"]
        assert r["vector"]["similarity"] is not None and r["graph"]["evidence"]
    assert body["candidates_fused"] == 2


def test_the_original_vector_and_graph_ranks_scores_and_evidence_are_preserved_exactly(
    client, corpus
):
    body = by_id(hybrid(client)["results"])
    vectors = by_id(standalone_vector(client))
    graphs = by_id(standalone_graph(client))
    assert set(body) == set(vectors) == set(graphs)
    for chunk_id, result in body.items():
        assert result["vector"] == {
            "rank": vectors[chunk_id]["rank"],
            "similarity": vectors[chunk_id]["similarity"],
        }
        assert result["graph"] == graphs[chunk_id]["graph"]  # rank, features, decided_by, evidence
        assert result["content"] == vectors[chunk_id]["content"]
        assert result["source"] == {
            k: vectors[chunk_id]["source"][k]
            for k in ("filename", "page_number", "section", "chunk_index")
        }


def test_the_fused_score_is_the_rrf_of_the_two_original_ranks(client, corpus):
    body = hybrid(client)
    for r in body["results"]:
        expected = 1 / (60 + r["vector"]["rank"]) + 1 / (60 + r["graph"]["rank"])
        assert r["fused_score"] == pytest.approx(expected, abs=5e-7)
    scores = [r["fused_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in body["results"]] == [1, 2]
    f = body["fusion"]
    assert (f["method"], f["rrf_k"], f["vector_weight"], f["graph_weight"]) == ("rrf", 60, 1.0, 1.0)
    assert "not a probability" in f["score_meaning"]


def test_fusion_parameters_are_applied_and_echoed(client, corpus):
    body = hybrid(client, fusion={"rrf_k": 10, "vector_weight": 2.0, "graph_weight": 0.5})
    assert body["fusion"]["rrf_k"] == 10 and body["fusion"]["vector_weight"] == 2.0
    for r in body["results"]:
        expected = 2.0 / (10 + r["vector"]["rank"]) + 0.5 / (10 + r["graph"]["rank"])
        assert r["fused_score"] == pytest.approx(expected, abs=5e-7)


def test_a_zero_graph_weight_reduces_the_fused_order_to_the_vector_order(client, corpus):
    body = hybrid(client, fusion={"graph_weight": 0.0})
    vector_order = [r["chunk_id"] for r in standalone_vector(client)]
    assert [r["chunk_id"] for r in body["results"]] == vector_order
    assert all(r["graph"] is not None for r in body["results"])  # still reported


def test_limit_truncates_the_fused_list_and_reports_the_pool(client, corpus):
    full = hybrid(client)
    one = hybrid(client, limit=1)
    assert len(one["results"]) == 1 and one["limit"] == 1 and one["candidates_fused"] == 2
    assert one["results"][0]["chunk_id"] == full["results"][0]["chunk_id"]


def test_the_response_has_per_path_status_and_revisions(client, corpus):
    body = hybrid(client)
    v, g = body["paths"]["vector"], body["paths"]["graph"]
    assert v["index_stale"] is False and v["index_revision"] == v["database_revision"]
    assert v["embedding_model"] and v["top_k"] == 20 and v["candidates"] == 2
    assert g["graph_revision"] == g["database_revision"] and g["candidates"] == 2
    assert g["independent_of_vectors"] and v["independent_of_graph"]
    assert body["linked_entities"][0]["name"] == "Acme Corp"
    assert set(body["timings_ms"]) >= {"vector", "graph", "fusion", "total"}


def test_repeated_requests_are_deterministic(client, corpus):
    first = hybrid(client, explain=True)
    first.pop("timings_ms")
    for _ in range(3):
        again = hybrid(client, explain=True)
        again.pop("timings_ms")
        assert again == first


def test_explanations_show_the_fusion_arithmetic_and_the_graph_features(client, corpus):
    body = hybrid(client, explain=True)
    top = body["results"][0]
    ex = top["explanation"]
    assert [t["path"] for t in ex["fusion"]] == ["vector", "graph"]
    assert ex["fusion"][0]["rank"] == top["vector"]["rank"]
    assert ex["fusion"][1]["contribution"] == pytest.approx(1 / (60 + top["graph"]["rank"]))
    assert "Retrieved by the vector path" in ex["summary"] and "fused rank 1" in ex["summary"]
    assert ex["graph_ranking"].startswith("touches 1 of the query entities")
    assert ex["ranking_keys"][-1] == "chunk_id"
    assert body["suppressed_spans"] == [] and hybrid(client)["results"][0]["explanation"] is None


# ------------------------------------------------------------ one path contributes nothing


def test_a_query_without_a_graph_entity_returns_vector_results_and_says_so(client, corpus):
    body = hybrid(client, "tell me about works located")
    assert body["paths"]["graph"]["status"] == "no_entities"
    assert body["path_summary"]["mode"] == "hybrid"  # both paths ran...
    assert body["path_summary"]["contributing"] == ["vector"]  # ...but only one contributed
    assert body["results"] and all(r["retrieved_by"] == ["vector"] for r in body["results"])
    assert all(r["graph"] is None for r in body["results"])
    assert any("graph path ran but contributed no results" in w for w in body["warnings"])


def test_a_query_without_vector_matches_returns_graph_results_and_says_so(client, corpus):
    body = hybrid(client, vector={"similarity_threshold": 0.99})
    assert body["paths"]["vector"]["status"] == "ok" and body["paths"]["vector"]["candidates"] == 0
    assert body["path_summary"]["contributing"] == ["graph"]
    assert body["results"] and all(r["retrieved_by"] == ["graph"] for r in body["results"])
    assert all(r["vector"] is None for r in body["results"])
    assert any("vector path ran but returned no results" in w for w in body["warnings"])
    graph_order = [r["chunk_id"] for r in standalone_graph(client)]
    assert [r["chunk_id"] for r in body["results"]] == graph_order


def test_an_empty_corpus_gives_an_empty_successful_hybrid_result(client):
    body = hybrid(client)
    assert body["results"] == [] and body["path_summary"]["mode"] == "hybrid"
    assert body["path_summary"]["contributing"] == [] and body["candidates_fused"] == 0


def test_an_ambiguous_entity_is_reported_and_the_vector_path_still_answers(client, session, corpus):
    for minute, etype in enumerate(["LOCATION", "PRODUCT"]):
        doc, chunk = add_indexed_doc(session, f"Mercury is a {etype.lower()}.", minute=10 + minute)
        store(session, doc, chunk, ChunkExtraction([E("Mercury", etype)], []))
    body = hybrid(client, "Tell me about Mercury")
    assert body["paths"]["graph"]["status"] == "ambiguous_only"
    assert body["ambiguous_entities"][0]["reason"] == "multiple_entities"
    assert body["path_summary"]["contributing"] == ["vector"]
    pinned = hybrid(
        client,
        "Tell me about Mercury",
        graph={"entity_ids": [make_entity_id("LOCATION", "mercury")]},
    )
    assert pinned["ambiguous_entities"][0]["action"] == "resolved"


# -------------------------------------------------------------------- a path fails: partial


def test_a_failed_vector_path_gives_a_labelled_partial_result_equal_to_graph_only(
    client, corpus, app, monkeypatch
):
    def boom(*a, **k):
        raise EmbeddingError("the embedding model is not available")

    monkeypatch.setattr(app.state.vectors.embedder, "embed", boom)
    body = hybrid(client)
    s = body["path_summary"]
    assert (s["mode"], s["complete"], s["is_complete_hybrid"]) == ("partial", False, False)
    assert s["succeeded"] == ["graph"] and s["failed"] == ["vector"]
    v = body["paths"]["vector"]
    assert v["status"] == "unavailable" and v["error_code"] == "vector_index_unavailable"
    assert "embedding model" in v["error_message"] and v["candidates"] == 0
    assert body["paths"]["graph"]["status"] == "ok"
    assert "NOT a complete hybrid" in body["warnings"][0]
    assert all(r["retrieved_by"] == ["graph"] and r["vector"] is None for r in body["results"])
    graph_only = standalone_graph(client)
    assert [r["chunk_id"] for r in body["results"]] == [r["chunk_id"] for r in graph_only]
    assert [r["graph"] for r in body["results"]] == [r["graph"] for r in graph_only]


def test_a_vector_index_that_is_building_is_a_partial_result_with_its_own_code(
    client, corpus, app, monkeypatch
):
    def building():
        raise VectorIndexBuilding("the vector index is being (re)built")

    monkeypatch.setattr(app.state.vectors, "ensure_fresh", building)
    body = hybrid(client)
    assert body["path_summary"]["mode"] == "partial"
    assert body["paths"]["vector"]["status"] == "building"
    assert body["paths"]["vector"]["error_code"] == "vector_index_building"
    assert body["results"]


def test_a_failed_graph_path_gives_a_labelled_partial_result_equal_to_vector_only(
    client, corpus, app, monkeypatch
):
    vector_only = standalone_vector(client)

    def broken():
        raise GraphUnavailable("database unavailable: OperationalError")

    monkeypatch.setattr(app.state.graph, "snapshot", broken)
    body = hybrid(client)
    s = body["path_summary"]
    assert (s["mode"], s["complete"], s["is_complete_hybrid"]) == ("partial", False, False)
    assert s["succeeded"] == ["vector"] and s["failed"] == ["graph"]
    g = body["paths"]["graph"]
    assert g["status"] == "unavailable" and g["error_code"] == "graph_unavailable"
    assert g["graph_revision"] is None
    assert "NOT a complete hybrid" in body["warnings"][0]
    assert [r["chunk_id"] for r in body["results"]] == [r["chunk_id"] for r in vector_only]
    assert [r["vector"]["similarity"] for r in body["results"]] == [
        r["similarity"] for r in vector_only
    ]
    assert body["linked_entities"] == [] and all(r["graph"] is None for r in body["results"])


@pytest.mark.parametrize("failing", ["graph", "vector"])
def test_an_unexpected_error_in_one_path_is_contained_and_not_leaked(
    client, corpus, app, monkeypatch, failing
):
    def boom(*a, **k):
        raise RuntimeError("secret: /home/user/.ssh/id_rsa")

    if failing == "graph":
        monkeypatch.setattr(app.state.graph_retriever, "search", boom)
    else:
        monkeypatch.setattr(app.state.vectors, "ensure_fresh", boom)
    r = client.post(URL, json={"query": QUERY})
    assert r.status_code == 200 and "secret" not in r.text
    body = r.json()
    assert body["path_summary"]["mode"] == "partial" and body["results"]
    assert body["paths"][failing]["error_code"] == f"{failing}_error"


# ---------------------------------------------------------------------- both paths fail


def test_both_paths_failing_is_a_503_with_each_paths_code(client, corpus, app, monkeypatch):
    def graph_down():
        raise GraphUnavailable("database unavailable")

    def vector_down():
        raise VectorIndexBuilding("being rebuilt")

    monkeypatch.setattr(app.state.graph, "snapshot", graph_down)
    monkeypatch.setattr(app.state.vectors, "ensure_fresh", vector_down)
    r = client.post(URL, json={"query": QUERY})
    assert r.status_code == 503
    error = r.json()["error"]
    assert error["code"] == "retrieval_unavailable"
    assert error["details"]["vector"] == {
        "status": "building",
        "error_code": "vector_index_building",
        "error_message": "being rebuilt",
    }
    assert error["details"]["graph"]["error_code"] == "graph_unavailable"
    assert "results" not in error  # nothing that looks like a result


# ------------------------------------------------------------------ request errors (422)


def test_unknown_entity_ids_are_a_request_error_not_a_partial_result(client, corpus):
    r = client.post(URL, json={"query": QUERY, "graph": {"entity_ids": ["nope", ALICE]}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "entity_not_found"
    assert r.json()["error"]["details"] == {"unknown_entity_ids": ["nope"]}


def test_a_query_the_embedder_rejects_is_a_request_error(client, corpus, app, monkeypatch):
    def reject(*a, **k):
        raise EmbeddingInputError("the query cannot be embedded")

    hybrid(client)  # build the index first: a failing sync would be a path failure, not this
    monkeypatch.setattr(app.state.vectors.embedder, "describe", reject)
    r = client.post(URL, json={"query": QUERY})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_query"


@pytest.mark.parametrize(
    "body, code",
    [
        ({"limit": 51}, "limit_exceeded"),
        ({"vector": {"top_k": 51}}, "limit_exceeded"),
        ({"graph": {"hops": 3}}, "limit_exceeded"),
        ({"graph": {"max_chunks": 51}}, "limit_exceeded"),
    ],
)
def test_parameters_beyond_the_configured_caps_are_refused(client, corpus, body, code):
    r = client.post(URL, json={"query": QUERY, **body})
    assert r.status_code == 422 and r.json()["error"]["code"] == code


@pytest.mark.parametrize(
    "body",
    [
        {"query": "   "},
        {"query": ""},
        {"query": "x" * 2001},
        {"query": "x", "surprise": 1},
        {"query": "x", "limit": 0},
        {"query": "x", "vector": {"top_k": 0}},
        {"query": "x", "vector": {"similarity_threshold": 2}},
        {"query": "x", "vector": {"surprise": 1}},
        {"query": "x", "graph": {"direction": "sideways"}},
        {"query": "x", "fusion": {"method": "linear"}},
        {"query": "x", "fusion": {"rrf_k": 0}},
        {"query": "x", "fusion": {"rrf_k": 1001}},
        {"query": "x", "fusion": {"vector_weight": -1}},
        {"query": "x", "fusion": {"graph_weight": 11}},
        {"query": "x", "fusion": {"vector_weight": 0, "graph_weight": 0}},
        {"query": "x", "fusion": {"surprise": 1}},
        {},
    ],
)
def test_invalid_requests_are_rejected_with_a_validation_error(client, corpus, body):
    r = client.post(URL, json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_direction_aliases_work_in_hybrid_requests_too(client, corpus):
    assert hybrid(client, graph={"direction": "out"})["paths"]["graph"]["direction"] == "outgoing"


# ------------------------------------------------- filters, deletion and stale indexes


def test_the_document_filter_applies_to_both_paths(client, session, corpus):
    body = hybrid(client, document_ids=[corpus["d2"].id, "no-such-document"])
    assert {r["document_id"] for r in body["results"]} == {corpus["d2"].id}
    assert body["unknown_document_ids"] == ["no-such-document"]
    assert body["results"][0]["retrieved_by"] == ["vector", "graph"]
    check_traceable(session, body)


def test_a_deleted_document_disappears_from_both_paths(client, corpus):
    assert hybrid(client)["candidates_fused"] == 2
    assert client.delete(f"/api/v1/documents/{corpus['d1'].id}").status_code == 204
    body = hybrid(client)
    assert {r["document_id"] for r in body["results"]} == {corpus["d2"].id}
    assert body["candidates_fused"] == 1


def test_evidence_for_a_chunk_deleted_after_the_graph_was_built_is_dropped(
    client, session, corpus, app, monkeypatch
):
    hybrid(client)  # builds the vector index too
    stale = app.state.graph.snapshot()
    session.delete(session.get(Document, corpus["d1"].id))
    session.commit()
    monkeypatch.setattr(app.state.graph, "snapshot", lambda: stale)
    body = hybrid(client, vector={"similarity_threshold": 0.99})
    assert body["dropped"]["stale_evidence"] == 2
    assert {r["document_id"] for r in body["results"]} == {corpus["d2"].id}
    assert any("no longer confirmed" in w for w in body["warnings"])


def test_a_stale_vector_index_is_reported_while_the_graph_sees_the_new_data(
    client, session, corpus, app
):
    hybrid(client)  # the vector index now reflects the two documents
    doc, chunk = add_indexed_doc(session, "Cara works at Acme Corp.", minute=30)
    store(
        session,
        doc,
        chunk,
        ChunkExtraction(
            [E("Cara", "PERSON"), E("Acme Corp")],
            [R("Cara", "Acme Corp", "WORKS_AT", "Cara works at Acme Corp")],
        ),
    )
    lock = app.state.vectors._sync_lock
    lock.acquire()  # a synchronization is "running": searches must use the last good index
    try:
        r = client.post(URL, json={"query": QUERY})
    finally:
        lock.release()
    body = r.json()
    assert r.status_code == 200
    assert body["paths"]["vector"]["status"] == "stale" and body["paths"]["vector"]["index_stale"]
    assert body["path_summary"]["is_complete_hybrid"] is True  # last-known-good is still a success
    new = by_id(body["results"])[chunk.id]
    assert new["retrieved_by"] == ["graph"]  # only the always-fresh graph path knows the new chunk
    assert any("behind the database" in w for w in body["warnings"])
    after = hybrid(client)  # once the sync can run, both paths agree again
    assert by_id(after["results"])[chunk.id]["retrieved_by"] == ["vector", "graph"]


def test_the_existing_endpoints_are_unaffected(client, corpus):
    assert client.post("/api/v1/search/vector", json={"query": QUERY}).status_code == 200
    assert client.post("/api/v1/search/graph", json={"query": QUERY}).status_code == 200
    assert client.get("/api/v1/vector/index").status_code == 200
    assert client.get("/api/v1/graph/overview").status_code == 200
