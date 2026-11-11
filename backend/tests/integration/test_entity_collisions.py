"""Entities whose names normalise to the same key are never silently merged.

Real ingestion path (resolver + persistence) and the real graph builder. The graph must expose
every colliding entity as its own node, with aliases and types intact, so that the Phase 6 query
linker can *report* an ambiguity instead of guessing (linker tests arrive with M2).
"""

from collections import defaultdict

import pytest
from sqlalchemy import select

from app.models import Entity, EntityCandidate
from app.services.extraction.normalization import normalize_key
from app.services.extraction.resolver import make_entity_id
from app.services.extraction.types import ChunkExtraction
from app.services.graph.builder import build_graph
from tests.integration.test_evidence_recompute import E, add_doc, store


def ingest(session, text, entities, minute):
    doc, chunk = add_doc(session, text, minute=minute)
    store(session, doc, chunk, ChunkExtraction(entities, []))
    return doc


def key_index(graph):
    """What a name lookup sees: normalised key -> every node that carries it (name or alias)."""
    index = defaultdict(set)
    for node_id, attrs in graph.nodes(data=True):
        for name in [attrs["canonical_name"], *attrs["aliases"]]:
            index[normalize_key(name, attrs["entity_type"])].add(node_id)
    return index


@pytest.fixture
def mercury(session):
    ingest(session, "Mercury is the closest planet to the sun.", [E("Mercury", "LOCATION")], 0)
    ingest(session, "Mercury is a popular outboard motor brand.", [E("Mercury", "PRODUCT")], 1)
    return build_graph(session)[0]


def test_two_entities_with_one_normalized_key_stay_two_entities(session, mercury):
    rows = session.scalars(select(Entity).order_by(Entity.entity_type)).all()
    assert [(r.canonical_name, r.entity_type, r.normalized_key) for r in rows] == [
        ("Mercury", "LOCATION", "mercury"),
        ("Mercury", "PRODUCT", "mercury"),
    ]
    assert len({r.id for r in rows}) == 2
    assert rows[0].id == make_entity_id("LOCATION", "mercury")
    assert rows[1].id == make_entity_id("PRODUCT", "mercury")


def test_the_graph_has_one_node_per_colliding_entity_and_no_merged_node(mercury):
    graph = mercury
    assert graph.number_of_nodes() == 2 and graph.number_of_edges() == 0
    assert {d["entity_type"] for _, d in graph.nodes(data=True)} == {"LOCATION", "PRODUCT"}
    assert {d["normalized_key"] for _, d in graph.nodes(data=True)} == {"mercury"}
    for _, d in graph.nodes(data=True):  # each keeps only its own evidence
        assert d["mention_count"] == 1 and len(d["source_document_ids"]) == 1


def test_a_lookup_by_key_sees_both_candidates_so_the_ambiguity_is_detectable(mercury):
    assert key_index(mercury)["mercury"] == {
        make_entity_id("LOCATION", "mercury"),
        make_entity_id("PRODUCT", "mercury"),
    }


def test_the_collision_is_recorded_for_review_and_never_resolved_automatically(session, mercury):
    [cand] = session.scalars(select(EntityCandidate)).all()
    assert cand.reason == "same_name_different_type" and cand.score == 1.0
    assert cand.status == "pending"
    assert {cand.entity_a_id, cand.entity_b_id} == {
        make_entity_id("LOCATION", "mercury"),
        make_entity_id("PRODUCT", "mercury"),
    }


def test_ingesting_in_the_other_order_gives_the_same_identities(session):
    ingest(session, "Mercury is a popular outboard motor brand.", [E("Mercury", "PRODUCT")], 0)
    ingest(session, "Mercury is the closest planet to the sun.", [E("Mercury", "LOCATION")], 1)
    graph, _ = build_graph(session)
    assert set(graph.nodes) == {
        make_entity_id("LOCATION", "mercury"),
        make_entity_id("PRODUCT", "mercury"),
    }


def test_suffix_removal_makes_an_organization_collide_with_a_product_without_merging(session):
    ingest(session, "Acme Inc was founded in 1990.", [E("Acme Inc", "ORGANIZATION")], 0)
    ingest(session, "The Acme is a tool for coyotes.", [E("Acme", "PRODUCT")], 1)
    graph, _ = build_graph(session)
    assert graph.number_of_nodes() == 2
    assert len(key_index(graph)["acme"]) == 2
    [cand] = session.scalars(select(EntityCandidate)).all()
    assert cand.reason == "same_name_different_type"


def test_an_alias_of_one_entity_can_collide_with_the_canonical_name_of_another(session):
    text = "Mercury Corp announced results. Mercury also orbits the sun closely."
    ingest(session, text, [E("Mercury Corp", "ORGANIZATION", surface="Mercury")], 0)
    ingest(session, "Mercury is the closest planet to the sun.", [E("Mercury", "LOCATION")], 1)
    graph, _ = build_graph(session)

    org = next(d for _, d in graph.nodes(data=True) if d["entity_type"] == "ORGANIZATION")
    place = next(d for _, d in graph.nodes(data=True) if d["entity_type"] == "LOCATION")
    assert org["canonical_name"] == "Mercury Corp" and org["aliases"] == ["Mercury"]
    assert place["canonical_name"] == "Mercury" and place["aliases"] == []

    # the organization's canonical name (suffix removed), its alias and the place all say mercury
    assert key_index(graph)["mercury"] == {org["entity_id"], place["entity_id"]}
    assert org["entity_id"] != place["entity_id"]
    assert graph.number_of_nodes() == 2


def test_the_same_name_in_two_documents_with_one_type_is_one_entity_by_ingestion_policy(session):
    """Exact key + same type is the only automatic merge, and it happens at ingestion."""
    ingest(session, "Mercury is the closest planet.", [E("Mercury", "LOCATION")], 0)
    ingest(
        session, "We visited Mercury last year.", [E("MERCURY", "LOCATION", surface="Mercury")], 1
    )
    graph, _ = build_graph(session)
    assert graph.number_of_nodes() == 1
    [node] = [d for _, d in graph.nodes(data=True)]
    assert node["mention_count"] == 2 and len(node["source_document_ids"]) == 2
