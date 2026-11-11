"""Evidence integrity: every result references an existing, in-scope chunk, and every piece of
graph evidence matches the database row it claims to come from, in every retrieval mode."""

import pytest

from app.models import Document, DocumentChunk, EntityMention, Relationship
from app.services.embeddings.service import VectorIndexBuilding
from app.services.extraction.normalization import squash
from app.services.extraction.resolver import make_entity_id
from app.services.graph.service import GraphUnavailable
from tests.graph_search_helpers import seed_corpus

HYBRID = "/api/v1/search/hybrid"
GRAPH = "/api/v1/search/graph"
QUERY = "Who works at Acme Corp?"
ACME = make_entity_id("ORGANIZATION", "acme")


@pytest.fixture
def corpus(session):
    data = seed_corpus(session)
    data["c1"].page_number, data["c1"].section = 4, "Intro"
    data["c2"].page_number, data["c2"].section = None, "Team"
    session.commit()
    return data


def post(client, url, **body):
    body.setdefault("query", QUERY)
    r = client.post(url, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def check_integrity(app, body, *, document_ids=None):
    """Compare every claim in the response with the database."""
    with app.state.session_factory() as s:
        assert body["results"], "an empty result cannot demonstrate integrity"
        seen: set[str] = set()
        for r in body["results"]:
            assert r["chunk_id"] not in seen, "a chunk appears twice"
            seen.add(r["chunk_id"])
            chunk = s.get(DocumentChunk, r["chunk_id"])
            assert chunk is not None
            doc = s.get(Document, chunk.document_id)
            # the chunk is in scope
            assert doc.processing_status == "completed"
            if document_ids is not None:
                assert doc.id in document_ids
            # and every location field is the stored one
            assert r["document_id"] == doc.id and r["content"] == chunk.content
            assert r["source"] == {
                "filename": doc.filename,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
            }
            assert r["retrieved_by"], "a result no path claims"
            assert ("vector" in r["retrieved_by"]) == (r.get("vector") is not None)
            assert ("graph" in r["retrieved_by"]) == (r["graph"] is not None)
            for ev in (r["graph"] or {}).get("evidence", []):
                assert ev["document_id"] == doc.id
                if ev["kind"] == "relationship":
                    row = s.get(Relationship, ev["relationship_id"])
                    assert row is not None
                    assert row.source_chunk_id == chunk.id and row.source_document_id == doc.id
                    assert (row.source_entity_id, row.target_entity_id) == (
                        ev["source"]["entity_id"],
                        ev["target"]["entity_id"],
                    )  # the stored direction, exactly
                    assert row.relationship_type == ev["relationship_type"]
                    assert row.supporting_text == ev["supporting_text"]
                    assert row.provenance == ev["provenance"] and row.confidence == ev["confidence"]
                    assert squash(row.supporting_text) in squash(chunk.content)
                else:
                    mention = (
                        s.query(EntityMention)
                        .filter_by(
                            entity_id=ev["entity"]["entity_id"],
                            chunk_id=chunk.id,
                            surface_form=ev["surface_form"],
                        )
                        .one_or_none()
                    )
                    assert mention is not None and mention.provenance == ev["provenance"]
                    assert squash(ev["surface_form"]) in squash(chunk.content)


MODES = {
    "default": {},
    "outgoing": {"graph": {"direction": "outgoing"}},
    "incoming": {"graph": {"direction": "incoming"}},
    "two hops": {"graph": {"hops": 2}},
    "relationship filter": {"graph": {"relationship_types": ["WORKS_AT"]}},
    "with mentions": {"graph": {"include_mentions": True}},
    "explicit entity ids": {"graph": {"entity_ids": [ACME], "link_entities": False}},
    "weights": {"fusion": {"vector_weight": 0.5, "graph_weight": 3.0, "rrf_k": 5}},
}


@pytest.mark.parametrize("name", list(MODES))
def test_hybrid_results_are_valid_in_every_mode(client, corpus, app, name):
    check_integrity(app, post(client, HYBRID, **MODES[name]))


@pytest.mark.parametrize("name", list(MODES))
def test_graph_results_are_valid_in_every_mode(client, corpus, app, name):
    graph_only = {k: v for k, v in MODES[name].items() if k != "fusion"}
    check_integrity(app, post(client, GRAPH, **graph_only))


def test_page_and_section_are_the_stored_ones(client, corpus, app):
    body = post(client, HYBRID, limit=5)
    by = {r["chunk_id"]: r for r in body["results"]}
    assert by[corpus["c1"].id]["source"]["page_number"] == 4
    assert by[corpus["c1"].id]["source"]["section"] == "Intro"
    assert by[corpus["c2"].id]["source"]["page_number"] is None
    assert by[corpus["c2"].id]["source"]["section"] == "Team"


def test_the_document_filter_is_honoured_by_every_result_and_every_piece_of_evidence(
    client, corpus, app
):
    for ids in ([corpus["d1"].id], [corpus["d2"].id], [corpus["d1"].id, corpus["d2"].id]):
        body = post(client, HYBRID, document_ids=ids)
        check_integrity(app, body, document_ids=set(ids))
        assert {r["document_id"] for r in body["results"]} == set(ids)


def test_results_stay_valid_when_one_path_is_down(client, corpus, app, monkeypatch):
    def vector_down():
        raise VectorIndexBuilding("building")

    monkeypatch.setattr(app.state.vectors, "ensure_fresh", vector_down)
    body = post(client, HYBRID)
    assert body["path_summary"]["mode"] == "partial"
    check_integrity(app, body)
    monkeypatch.undo()

    def graph_down():
        raise GraphUnavailable("down")

    monkeypatch.setattr(app.state.graph, "snapshot", graph_down)
    body = post(client, HYBRID)
    assert body["path_summary"]["mode"] == "partial"
    check_integrity(app, body)


def test_unfinished_and_failed_documents_never_reach_a_result(client, corpus, app, session):
    session.get(Document, corpus["d2"].id).processing_status = "failed"
    session.commit()
    body = post(client, HYBRID)
    check_integrity(app, body)
    assert {r["document_id"] for r in body["results"]} == {corpus["d1"].id}


def test_every_result_has_at_least_one_path_and_unique_contiguous_ranks(client, corpus):
    body = post(client, HYBRID, limit=10)
    assert [r["rank"] for r in body["results"]] == list(range(1, len(body["results"]) + 1))
    assert len({r["chunk_id"] for r in body["results"]}) == len(body["results"])
