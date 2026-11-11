"""Building the graph from SQLite: attributes, direction, aggregation, unsupported records."""

import networkx as nx
import pytest
from sqlalchemy import select

from app.models import Document, DocumentChunk, Entity, Relationship
from app.services.extraction.types import ChunkExtraction
from app.services.graph.builder import build_graph, make_edge_id
from app.services.graph.revision import current_revision
from tests.integration.test_evidence_recompute import E, R, add_doc, store

WORKS = "Alice works at Acme Corp."
PEOPLE = [E("Alice", "PERSON", desc="Engineer"), E("Acme Corp")]


def relation(text=WORKS, rtype="WORKS_AT", src="Alice", tgt="Acme Corp"):
    return ChunkExtraction(PEOPLE, [R(src, tgt, rtype, text.rstrip("."))])


@pytest.fixture
def built(session):
    doc, chunk = add_doc(session, WORKS, minute=0)
    store(session, doc, chunk, relation())
    graph, revision = build_graph(session)
    return graph, revision, doc, chunk


def node_by_name(graph, name):
    return next(d for _, d in graph.nodes(data=True) if d["canonical_name"] == name)


def test_every_entity_is_a_node_with_the_required_attributes(session, built):
    graph, _, doc, _ = built
    assert graph.number_of_nodes() == session.query(Entity).count() == 2
    alice = node_by_name(graph, "Alice")
    entity = session.scalars(select(Entity).where(Entity.canonical_name == "Alice")).one()
    assert (
        alice["entity_id"] == entity.id and entity.id in graph
    )  # SQLite's deterministic ID is the node id
    assert alice["entity_type"] == "PERSON"
    assert alice["description"] == "Engineer"
    assert alice["aliases"] == []
    assert alice["mention_count"] == 1
    assert alice["source_document_ids"] == [doc.id]
    assert alice["normalized_key"] == "alice"


def test_every_relationship_is_a_directed_edge_with_the_required_attributes(session, built):
    graph, _, doc, chunk = built
    assert graph.number_of_edges() == session.query(Relationship).count() == 1
    alice, acme = node_by_name(graph, "Alice"), node_by_name(graph, "Acme Corp")
    edge = graph.edges[alice["entity_id"], acme["entity_id"], "WORKS_AT"]
    assert edge["relationship_type"] == "WORKS_AT"
    assert (edge["source_entity_id"], edge["target_entity_id"]) == (
        alice["entity_id"],
        acme["entity_id"],
    )
    assert edge["edge_id"] == make_edge_id(alice["entity_id"], "WORKS_AT", acme["entity_id"])
    assert edge["source_document_ids"] == [doc.id] and edge["source_chunk_ids"] == [chunk.id]
    assert edge["provenance"] == ["test"]
    [ev] = edge["supporting_evidence"]
    assert ev["supporting_text"] == "Alice works at Acme Corp"
    assert (ev["document_id"], ev["chunk_id"]) == (doc.id, chunk.id)
    assert ev["relationship_id"] == session.scalars(select(Relationship.id)).one()


def test_direction_is_never_reversed(built):
    graph, *_ = built
    alice, acme = (
        node_by_name(graph, "Alice")["entity_id"],
        node_by_name(graph, "Acme Corp")["entity_id"],
    )
    assert graph.has_edge(alice, acme) and not graph.has_edge(acme, alice)
    assert graph.is_directed()


def test_confidence_is_none_unless_an_extractor_reported_one(session, built):
    graph, *_ = built
    [edge] = [d for *_, d in graph.edges(data=True)]
    assert edge["confidence"] is None
    assert edge["supporting_evidence"][0]["confidence"] is None

    session.query(Relationship).update({Relationship.confidence: 0.8})
    session.commit()
    graph2, _ = build_graph(session)
    [edge2] = [d for *_, d in graph2.edges(data=True)]
    assert edge2["confidence"] == 0.8 and edge2["supporting_evidence"][0]["confidence"] == 0.8


def test_evidence_from_several_documents_and_chunks_is_aggregated_not_lost(session):
    docs = []
    for minute, text in enumerate(
        [
            "Alice works at Acme Corp.",
            "Alice is employed by Acme Corp.",
            "Alice works at Acme Corp too.",
        ]
    ):
        d, c = add_doc(session, text, minute)
        docs.append((d, c, text))
        store(session, d, c, relation(text))
    graph, _ = build_graph(session)
    assert graph.number_of_edges() == 1  # one edge...
    [edge] = [d for *_, d in graph.edges(data=True)]
    assert len(edge["supporting_evidence"]) == 3  # ...three pieces of evidence
    assert edge["source_document_ids"] == sorted(d.id for d, *_ in docs)
    assert edge["source_chunk_ids"] == sorted(c.id for _, c, _ in docs)
    assert {e["supporting_text"] for e in edge["supporting_evidence"]} == {
        t.rstrip(".") for *_, t in docs
    }
    alice = node_by_name(graph, "Alice")
    assert alice["mention_count"] == 3 and len(alice["source_document_ids"]) == 3


def test_different_relationship_types_between_one_pair_are_separate_edges(session):
    doc, chunk = add_doc(session, "Alice works at Acme Corp. Alice is related to Acme Corp.", 0)
    ex = ChunkExtraction(
        PEOPLE,
        [
            R("Alice", "Acme Corp", "WORKS_AT", "Alice works at Acme Corp"),
            R("Alice", "Acme Corp", "RELATED_TO", "Alice is related to Acme Corp"),
        ],
    )
    store(session, doc, chunk, ex)
    graph, _ = build_graph(session)
    assert isinstance(graph, nx.MultiDiGraph)  # why MultiDiGraph: this must be representable
    assert graph.number_of_nodes() == 2 and graph.number_of_edges() == 2
    assert sorted(k for *_, k in graph.edges(keys=True)) == ["RELATED_TO", "WORKS_AT"]


def test_aliases_come_from_entity_metadata(session):
    d1, c1 = add_doc(session, "OpenAI one.", 0)
    d2, c2 = add_doc(session, "Open AI two.", 1)
    store(session, d1, c1, ChunkExtraction([E("OpenAI")], []))
    store(session, d2, c2, ChunkExtraction([E("Open AI")], []))
    graph, _ = build_graph(session)
    [node] = [d for _, d in graph.nodes(data=True)]
    assert node["aliases"] == ["Open AI"] and node["mention_count"] == 2


def test_unsupported_records_never_become_graph_elements(session, built):
    _, _, doc, chunk = built
    # An entity nobody mentions, and a relationship that points at it.
    ghost = Entity(
        canonical_name="Ghost", normalized_key="ghost", entity_type="PERSON", id="g" * 32
    )
    session.add(ghost)
    session.flush()
    alice = session.scalars(select(Entity).where(Entity.canonical_name == "Alice")).one()
    session.add(
        Relationship(
            source_entity_id=alice.id,
            target_entity_id=ghost.id,
            relationship_type="RELATED_TO",
            supporting_text="x",
            source_document_id=doc.id,
            source_chunk_id=chunk.id,
            provenance="t",
        )
    )
    session.commit()
    graph, _ = build_graph(session)
    assert ghost.id not in graph
    assert graph.number_of_nodes() == 2 and graph.number_of_edges() == 1
    assert not graph.has_edge(alice.id, ghost.id)


def test_build_is_deterministic_and_graph_is_frozen(session, built):
    first, revision, *_ = built
    second, revision2 = build_graph(session)
    assert revision == revision2 == current_revision(session) > 0
    assert list(first.nodes) == list(second.nodes)
    assert [(u, v, k) for u, v, k in first.edges(keys=True)] == [
        (u, v, k) for u, v, k in second.edges(keys=True)
    ]
    with pytest.raises(nx.NetworkXError):
        first.add_node("intruder")
    assert first.graph["revision"] == revision


def test_empty_database_gives_an_empty_graph(session):
    graph, revision = build_graph(session)
    assert graph.number_of_nodes() == 0 and graph.number_of_edges() == 0 and revision == 0


def test_revision_increases_with_every_evidence_change(session, settings):
    from app.services.ingestion import service

    seen = [current_revision(session)]
    d, c = add_doc(session, WORKS, 0)
    store(session, d, c, relation())
    seen.append(current_revision(session))
    store(session, d, c, relation())  # reprocessing
    seen.append(current_revision(session))
    service.delete_document(session, settings, d.id)
    seen.append(current_revision(session))
    assert seen == sorted(set(seen)) and len(seen) == 4
    assert session.query(Document).count() == 0 and session.query(DocumentChunk).count() == 0
