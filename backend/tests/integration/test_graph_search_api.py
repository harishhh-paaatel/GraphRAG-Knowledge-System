"""POST /api/v1/search/graph against real SQLite data and the real graph service."""

import pytest
from sqlalchemy import select

from app.models import Document, DocumentChunk
from app.services.extraction.normalization import squash
from app.services.extraction.resolver import make_entity_id
from app.services.extraction.types import ChunkExtraction
from app.services.graph.service import GraphUnavailable
from tests.graph_search_helpers import seed_corpus
from tests.integration.test_evidence_recompute import E, R, add_doc, store

URL = "/api/v1/search/graph"
ACME = make_entity_id("ORGANIZATION", "acme")
ALICE = make_entity_id("PERSON", "alice")
BOB = make_entity_id("PERSON", "bob")
BERLIN = make_entity_id("LOCATION", "berlin")


def search(client, query="Who works at Acme Corp?", *, expect=200, **kw):
    graph = kw.pop("graph", {})
    body = {"query": query, "graph": graph, **kw}
    r = client.post(URL, json=body)
    assert r.status_code == expect, r.text
    return r.json()


def evidence_of(body):
    return [e for r in body["results"] for e in r["graph"]["evidence"]]


def relationships(body):
    return sorted(
        {
            (e["relationship_type"], e["source"]["name"], e["target"]["name"])
            for e in evidence_of(body)
            if e["kind"] == "relationship"
        }
    )


def check_traceable(session, body):
    """Every returned chunk exists in SQLite and every piece of evidence is visible in it."""
    for r in body["results"]:
        chunk = session.get(DocumentChunk, r["chunk_id"])
        assert chunk is not None and chunk.content == r["content"]
        assert chunk.document_id == r["document_id"]
        assert session.get(Document, r["document_id"]).filename == r["source"]["filename"]
        assert r["retrieved_by"] == ["graph"] and r["vector"] is None and r["fused_score"] is None
        assert r["graph"]["evidence"], "a graph result without evidence"
        for ev in r["graph"]["evidence"]:
            assert ev["document_id"] == r["document_id"]
            text = ev["supporting_text"] or ev["surface_form"]
            assert squash(text) in squash(r["content"])


@pytest.fixture
def corpus(session):
    return seed_corpus(session)


# --------------------------------------------------------------------------- basics


def test_a_question_about_an_entity_returns_the_chunks_that_evidence_its_relationships(
    client, session, corpus
):
    body = search(client)
    assert [e["name"] for e in body["linked_entities"]] == ["Acme Corp"]
    linked = body["linked_entities"][0]
    assert linked["entity_id"] == ACME and linked["span"] == "Acme Corp"
    assert linked["match_type"] == "exact_name" and linked["resolved_by"] == "query"
    assert linked["edge_count"] == 3  # 2 WORKS_AT in + 1 LOCATED_IN out (aggregated edges)
    assert {r["chunk_id"] for r in body["results"]} == {corpus["c1"].id, corpus["c2"].id}
    assert relationships(body) == [
        ("LOCATED_IN", "Acme Corp", "Berlin"),
        ("WORKS_AT", "Alice", "Acme Corp"),
        ("WORKS_AT", "Bob", "Acme Corp"),
    ]
    check_traceable(session, body)


def test_the_response_states_the_path_the_revisions_and_that_it_is_graph_only(client, corpus):
    body = search(client)
    assert body["graph"]["status"] == "ok" and body["graph"]["independent_of_vectors"] is True
    assert body["graph"]["graph_revision"] == body["graph"]["database_revision"]
    assert body["graph"]["direction"] == "both" and body["graph"]["hops"] == 1
    assert body["graph"]["tie_breaking"] == "chunk_id"
    assert body["path_summary"] == {
        "mode": "graph_only",
        "complete": True,
        "message": "Only the graph path was requested, and it succeeded.",
    }
    assert body["limit"] == 10 and body["dropped"] == {
        "missing_chunks": 0,
        "out_of_scope": 0,
        "stale_evidence": 0,
        "stale_results": 0,
    }
    assert "not a relevance or confidence score" in body["ranking_note"]


def test_every_result_carries_its_ranking_features_and_decided_by(client, corpus):
    body = search(client)
    ranks = [r["rank"] for r in body["results"]]
    assert ranks == list(range(1, len(ranks) + 1))
    first, second = body["results"][0]["graph"], body["results"][1]["graph"]
    assert first["rank"] == 1 and first["features"]["decided_by"] is None
    assert second["features"]["decided_by"] in {
        "seed_coverage",
        "connects_seeds",
        "relationship_evidence",
        "distance",
        "support",
        "corroboration",
        "provenance_known",
        "chunk_id",
    }
    f = first["features"]
    assert f["seed_coverage"] == 1 and f["relationship_evidence"] is True
    assert f["distance"] == 1 and f["support"] == 2 and f["corroboration"] == 2
    assert f["provenance_known"] is True and f["connects_seeds"] is False


def test_the_two_symmetric_chunks_are_ordered_by_chunk_id_and_say_so(client, corpus):
    body = search(client)
    a, b = body["results"][0], body["results"][1]
    assert a["chunk_id"] < b["chunk_id"]
    assert b["graph"]["features"]["decided_by"] == "chunk_id"


def test_provenance_and_confidence_are_reported_as_stored(client, corpus):
    body = search(client)
    for ev in evidence_of(body):
        assert ev["provenance"] == "test" and ev["confidence"] is None
        assert ev["relationship_id"] and ev["edge_id"]


# ------------------------------------------------------------------------- directions


def test_outgoing_returns_only_edges_leaving_the_entity(client, corpus):
    body = search(client, graph={"direction": "outgoing"})
    assert relationships(body) == [("LOCATED_IN", "Acme Corp", "Berlin")]
    assert {e["traversal"] for e in evidence_of(body)} == {"outgoing"}
    assert body["graph"]["direction"] == "outgoing"


def test_incoming_returns_only_edges_arriving_at_the_entity(client, corpus):
    body = search(client, graph={"direction": "incoming"})
    assert relationships(body) == [
        ("WORKS_AT", "Alice", "Acme Corp"),
        ("WORKS_AT", "Bob", "Acme Corp"),
    ]
    assert {e["traversal"] for e in evidence_of(body)} == {"incoming"}


@pytest.mark.parametrize("direction", ["outgoing", "incoming", "both"])
def test_the_stored_direction_is_preserved_in_every_traversal_direction(client, corpus, direction):
    body = search(client, graph={"direction": direction})
    for ev in evidence_of(body):
        if ev["relationship_type"] == "WORKS_AT":
            assert ev["source"]["entity_id"] in {ALICE, BOB} and ev["target"]["entity_id"] == ACME
        if ev["relationship_type"] == "LOCATED_IN":
            assert ev["source"]["entity_id"] == ACME and ev["target"]["entity_id"] == BERLIN
        assert ev["reached_from"]["entity_id"] == ACME


def test_both_directions_from_a_target_entity_never_reverse_the_relationship(client, corpus):
    body = search(client, "Where is Berlin?", graph={"direction": "both"})
    assert relationships(body) == [("LOCATED_IN", "Acme Corp", "Berlin")]
    ev = evidence_of(body)[0]
    assert ev["source"]["name"] == "Acme Corp" and ev["target"]["name"] == "Berlin"
    assert ev["traversal"] == "incoming" and ev["reached_from"]["name"] == "Berlin"
    assert search(client, "Berlin", graph={"direction": "outgoing"})["results"] == []


@pytest.mark.parametrize("alias, canonical", [("out", "outgoing"), ("in", "incoming")])
def test_the_short_direction_spellings_are_accepted_and_reported_canonically(
    client, corpus, alias, canonical
):
    assert search(client, graph={"direction": alias})["graph"]["direction"] == canonical


def test_an_unknown_direction_is_rejected(client, corpus):
    search(client, graph={"direction": "sideways"}, expect=422)


# --------------------------------------------------------------------- hops and limits


def test_one_hop_stays_local_and_two_hops_reach_the_next_neighbourhood(client, corpus):
    one = search(client, "Berlin", graph={"hops": 1})
    assert relationships(one) == [("LOCATED_IN", "Acme Corp", "Berlin")]
    two = search(client, "Berlin", graph={"hops": 2})
    assert ("WORKS_AT", "Alice", "Acme Corp") in relationships(two)
    far = [e for e in evidence_of(two) if e["relationship_type"] == "WORKS_AT"]
    assert {e["distance"] for e in far} == {2}
    near = [e for e in evidence_of(two) if e["relationship_type"] == "LOCATED_IN"]
    assert {e["distance"] for e in near} == {1}


def test_an_unrelated_neighbourhood_is_not_returned(client, corpus):
    body = search(client, "Alice", graph={"hops": 1})
    assert relationships(body) == [("WORKS_AT", "Alice", "Acme Corp")]
    assert BOB not in {e["source"]["entity_id"] for e in evidence_of(body)}


def test_hops_above_the_configured_cap_are_rejected(client, corpus):
    r = client.post(URL, json={"query": "Acme Corp", "graph": {"hops": 3}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "limit_exceeded"
    assert client.post(URL, json={"query": "Acme", "graph": {"hops": 0}}).status_code == 422


def test_limit_and_max_chunks_are_capped(client, corpus):
    r = client.post(URL, json={"query": "Acme Corp", "limit": 51})
    assert r.status_code == 422 and r.json()["error"]["code"] == "limit_exceeded"
    r = client.post(URL, json={"query": "Acme Corp", "graph": {"max_chunks": 51}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "limit_exceeded"


def test_limit_returns_the_best_ranked_chunks_only(client, corpus):
    full = search(client)
    one = search(client, limit=1)
    assert len(one["results"]) == 1 and one["limit"] == 1
    assert one["results"][0]["chunk_id"] == full["results"][0]["chunk_id"]
    assert one["graph"]["candidates"] == 2


def test_max_chunks_bounds_the_candidate_pool(client, corpus):
    body = search(client, graph={"max_chunks": 1})
    assert len(body["results"]) == 1 and body["graph"]["candidates"] == 2


def test_relationship_types_restrict_the_walk(client, corpus):
    body = search(client, graph={"relationship_types": ["LOCATED_IN"]})
    assert {r[0] for r in relationships(body)} == {"LOCATED_IN"}
    assert body["graph"]["relationship_types"] == ["LOCATED_IN"]
    search(client, graph={"relationship_types": ["NOT_A_TYPE"]}, expect=422)


# --------------------------------------------------------------------- document filter


def test_the_document_filter_restricts_results_and_evidence(client, session, corpus):
    body = search(client, document_ids=[corpus["d2"].id])
    assert [r["document_id"] for r in body["results"]] == [corpus["d2"].id]
    assert {e["document_id"] for e in evidence_of(body)} == {corpus["d2"].id}
    assert relationships(body) == [
        ("LOCATED_IN", "Acme Corp", "Berlin"),
        ("WORKS_AT", "Bob", "Acme Corp"),
    ]
    check_traceable(session, body)


def test_unknown_document_ids_are_reported_not_fatal(client, corpus):
    body = search(client, document_ids=[corpus["d1"].id, "no-such-document"])
    assert body["unknown_document_ids"] == ["no-such-document"]
    assert {r["document_id"] for r in body["results"]} == {corpus["d1"].id}
    assert any("document_ids" in w for w in body["warnings"])
    only_unknown = search(client, document_ids=["nope"])
    assert only_unknown["results"] == [] and only_unknown["graph"]["status"] == "ok"


def test_documents_that_are_not_completed_are_never_searched(client, session, corpus):
    session.get(Document, corpus["d1"].id).processing_status = "processing"
    session.commit()
    body = search(client)
    assert {r["document_id"] for r in body["results"]} == {corpus["d2"].id}
    assert corpus["d1"].id not in {e["document_id"] for e in evidence_of(body)}


# ------------------------------------------------------------- metadata and provenance


def test_page_section_and_document_metadata_are_preserved(client, session, corpus):
    chunk = session.get(DocumentChunk, corpus["c1"].id)
    chunk.page_number, chunk.section = 7, "Offices"
    session.commit()
    body = search(client, document_ids=[corpus["d1"].id])
    [result] = body["results"]
    assert result["source"] == {
        "filename": corpus["d1"].filename,
        "page_number": 7,
        "section": "Offices",
        "chunk_index": 0,
    }
    assert result["document_id"] == corpus["d1"].id and result["content"] == chunk.content


# ----------------------------------------------------- missing, empty and ambiguous data


def test_a_query_without_a_known_entity_returns_no_results_and_says_why(client, corpus):
    body = search(client, "What is the weather like today?")
    assert body["results"] == [] and body["linked_entities"] == []
    assert body["graph"]["status"] == "no_entities"
    assert "no entity" in body["graph"]["note"]
    assert body["path_summary"]["complete"] is True  # the path ran; it just found nothing


def test_an_empty_knowledge_graph_is_not_an_error(client):
    body = search(client, "Who works at Acme Corp?")
    assert body["results"] == [] and body["graph"]["status"] == "no_entities"
    assert body["graph"]["graph_revision"] >= 0


def test_a_linked_entity_without_relationships_returns_nothing_unless_mentions_are_asked_for(
    client, session, corpus
):
    doc, chunk = add_doc(session, "Zed is a loner.", minute=5)
    store(session, doc, chunk, ChunkExtraction([E("Zed", "PERSON")], []))
    body = search(client, "Tell me about Zed")
    assert [e["name"] for e in body["linked_entities"]] == ["Zed"]
    assert body["linked_entities"][0]["edge_count"] == 0
    assert body["results"] == [] and body["graph"]["status"] == "ok"
    assert "no relationship evidence" in body["graph"]["note"]

    with_mentions = search(client, "Tell me about Zed", graph={"include_mentions": True})
    [result] = with_mentions["results"]
    [ev] = result["graph"]["evidence"]
    assert ev["kind"] == "mention" and ev["surface_form"] == "Zed" and ev["entity"]["name"] == "Zed"
    assert result["graph"]["features"]["relationship_evidence"] is False
    assert result["graph"]["features"]["support"] == 0


@pytest.fixture
def mercury(session):
    for minute, (text, etype) in enumerate(
        [("Mercury orbits the Sun.", "LOCATION"), ("Mercury sells outboard motors.", "PRODUCT")]
    ):
        doc, chunk = add_doc(session, text, minute=10 + minute)
        store(
            session,
            doc,
            chunk,
            ChunkExtraction(
                [E("Mercury", etype), E("Sun" if etype == "LOCATION" else "Brunswick", etype)],
                [
                    R(
                        "Mercury",
                        "Sun" if etype == "LOCATION" else "Brunswick",
                        "PART_OF" if etype == "LOCATION" else "RELATED_TO",
                        "Mercury orbits the Sun" if etype == "LOCATION" else "Mercury sells",
                    )
                ]
                if etype == "LOCATION"
                else [],
            ),
        )
    return {
        "planet": make_entity_id("LOCATION", "mercury"),
        "brand": make_entity_id("PRODUCT", "mercury"),
    }


def test_an_ambiguous_name_is_reported_with_its_candidates_and_nothing_is_guessed(client, mercury):
    body = search(client, "Tell me about Mercury")
    assert body["results"] == [] and body["linked_entities"] == []
    assert body["graph"]["status"] == "ambiguous_only"
    [amb] = body["ambiguous_entities"]
    assert amb["reason"] == "multiple_entities" and amb["action"] == "skipped"
    assert {c["entity_id"] for c in amb["candidates"]} == set(mercury.values())
    assert {c["type"] for c in amb["candidates"]} == {"LOCATION", "PRODUCT"}
    assert amb["resolve_with"] == "graph.entity_ids" and amb["resolved_to"] == []
    assert "entity_ids" in body["graph"]["note"]


def test_explicit_entity_ids_resolve_the_ambiguity(client, mercury):
    body = search(client, "Tell me about Mercury", graph={"entity_ids": [mercury["planet"]]})
    assert body["graph"]["status"] == "ok"
    assert [(e["entity_id"], e["resolved_by"]) for e in body["linked_entities"]] == [
        (mercury["planet"], "entity_ids")
    ]
    [amb] = body["ambiguous_entities"]
    assert amb["action"] == "resolved" and amb["resolved_to"] == [mercury["planet"]]
    assert relationships(body) == [("PART_OF", "Mercury", "Sun")]


def test_entity_ids_can_be_used_without_any_name_in_the_query(client, corpus):
    body = search(client, "tell me more", graph={"entity_ids": [ALICE], "link_entities": False})
    assert body["linked_entities"][0]["span"] is None
    assert relationships(body) == [("WORKS_AT", "Alice", "Acme Corp")]
    no_auto = search(client, "Who works at Acme Corp?", graph={"link_entities": False})
    assert no_auto["results"] == [] and no_auto["graph"]["status"] == "no_entities"


def test_unknown_entity_ids_are_rejected_with_the_offending_ids(client, corpus):
    r = client.post(URL, json={"query": "Acme", "graph": {"entity_ids": ["nope", ALICE]}})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "entity_not_found"
    assert r.json()["error"]["details"] == {"unknown_entity_ids": ["nope"]}


def test_two_names_in_a_query_link_two_entities_and_rank_the_connecting_chunk_first(client, corpus):
    body = search(client, "How are Alice and Acme Corp related?")
    assert {e["name"] for e in body["linked_entities"]} == {"Alice", "Acme Corp"}
    top = body["results"][0]
    assert top["chunk_id"] == corpus["c1"].id
    assert top["graph"]["features"]["connects_seeds"] is True
    assert top["graph"]["features"]["seed_coverage"] == 2


# -------------------------------------------------------------- deleted and vanished data


def test_a_deleted_document_disappears_from_results_and_evidence(client, corpus):
    assert client.delete(f"/api/v1/documents/{corpus['d1'].id}").status_code == 204
    body = search(client)
    assert {r["document_id"] for r in body["results"]} == {corpus["d2"].id}
    assert corpus["d1"].id not in {e["document_id"] for e in evidence_of(body)}
    assert "Alice" not in {n for _, n, _ in relationships(body)}
    assert search(client, "Alice")["linked_entities"] == []


def test_evidence_for_a_chunk_deleted_after_the_graph_was_built_is_dropped_not_invented(
    client, session, corpus, app, monkeypatch
):
    stale = app.state.graph.snapshot()  # built while both documents exist
    session.delete(session.get(Document, corpus["d1"].id))
    session.commit()
    monkeypatch.setattr(app.state.graph, "snapshot", lambda: stale)

    body = search(client)
    assert body["dropped"]["stale_evidence"] == 2  # both relationships of the deleted document
    assert body["dropped"]["missing_chunks"] == 0  # caught earlier, before the chunk lookup
    assert {r["document_id"] for r in body["results"]} == {corpus["d2"].id}
    assert any("no longer confirmed" in w for w in body["warnings"])
    assert [r["rank"] for r in body["results"]] == [1]  # ranks stay contiguous


def test_the_graph_service_failing_gives_a_clear_503(client, corpus, app, monkeypatch):
    def broken():
        raise GraphUnavailable("database unavailable: OperationalError")

    monkeypatch.setattr(app.state.graph, "snapshot", broken)
    r = client.post(URL, json={"query": "Acme Corp"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "graph_unavailable"


# ------------------------------------------------------- independence from the vector side


class _Exploding:
    """Stands in for the vector service: any use at all fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"the graph path touched the vector service ({name})")


def test_graph_search_works_when_the_vector_service_is_unusable(client, corpus, app):
    healthy = search(client, "Who works at Acme Corp?")
    app.state.vectors = _Exploding()
    broken = search(client, "Who works at Acme Corp?")
    for body in (healthy, broken):
        body.pop("timings_ms")
    assert broken == healthy and broken["results"]


def test_graph_search_works_when_the_embedding_model_cannot_load(client, corpus, app, monkeypatch):
    from app.services.embeddings.embedder import EmbeddingError

    def fail(*args, **kwargs):
        raise EmbeddingError("model unavailable")

    monkeypatch.setattr(app.state.vectors.embedder, "embed", fail)
    monkeypatch.setattr(app.state.vectors.embedder, "identity", fail)
    assert search(client)["results"]
    # the vector endpoint reports its own failure; the graph endpoint is unaffected
    assert client.post("/api/v1/search/vector", json={"query": "acme"}).status_code == 503
    assert search(client)["results"]


def test_the_existing_vector_search_endpoint_is_unchanged(client, corpus):
    r = client.post("/api/v1/search/vector", json={"query": "Acme Corp", "top_k": 3})
    assert r.status_code == 200 and "similarity_metric" in r.json()


# --------------------------------------------------------------------- determinism etc.


def test_repeated_identical_requests_return_identical_results(client, corpus):
    first = search(client, "Where is Acme Corp?", graph={"hops": 2})
    for _ in range(3):
        again = search(client, "Where is Acme Corp?", graph={"hops": 2})
        for body in (first, again):
            body.pop("timings_ms", None)
        assert again == first


def test_explain_adds_a_templated_description_and_the_suppressed_spans(client, corpus):
    body = search(client, "Where is Acme Corp?", explain=True)
    first, second = body["results"][0], body["results"][1]
    assert first["explanation"]["summary"].endswith("ranked first.")
    assert "touches 1 of the query entities" in first["explanation"]["summary"]
    assert "stable chunk id" in second["explanation"]["summary"]
    assert first["explanation"]["ranking_keys"][-1] == "chunk_id"
    assert body["suppressed_spans"] == []
    plain = search(client, "Where is Acme Corp?")
    assert plain["suppressed_spans"] is None and plain["results"][0]["explanation"] is None


def test_request_validation(client, corpus):
    assert client.post(URL, json={"query": "   "}).status_code == 422
    assert client.post(URL, json={"query": ""}).status_code == 422
    assert client.post(URL, json={"query": "x" * 2001}).status_code == 422
    assert client.post(URL, json={"query": "x", "surprise": 1}).status_code == 422
    assert client.post(URL, json={"query": "x", "graph": {"surprise": 1}}).status_code == 422
    assert client.post(URL, json={"query": "x", "limit": 0}).status_code == 422
    assert client.post(URL, json={}).status_code == 422


def test_all_results_are_traceable_in_every_mode(client, session, corpus):
    for graph in ({}, {"direction": "outgoing"}, {"direction": "incoming"}, {"hops": 2}):
        check_traceable(session, search(client, "Acme Corp", graph=graph))
    check_traceable(session, search(client, "Zed", graph={"include_mentions": True}))
    assert session.scalars(select(Document)).first() is not None
