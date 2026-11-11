"""Builders for graph-retrieval tests: hand-made graph snapshots and a small real corpus."""

import networkx as nx

from app.services.embeddings.revision import bump_vector_revision
from app.services.extraction.types import ChunkExtraction
from app.services.graph.builder import make_edge_id
from app.services.graph.service import GraphSnapshot
from tests.integration.test_evidence_recompute import E, R, add_doc, store


def node(entity_id, name, etype="ORGANIZATION", aliases=(), docs=("d1",)):
    return entity_id, {
        "entity_id": entity_id,
        "canonical_name": name,
        "entity_type": etype,
        "description": None,
        "aliases": list(aliases),
        "mention_count": 1,
        "source_document_ids": list(docs),
        "normalized_key": name.casefold(),
    }


def evidence(chunk_id, text="evidence", doc="d1", provenance="rule_based", n=0):
    return {
        "relationship_id": f"rel-{chunk_id}-{n}",
        "supporting_text": text,
        "document_id": doc,
        "chunk_id": chunk_id,
        "provenance": provenance,
        "confidence": None,
    }


def make_snapshot(nodes, edges=(), revision=1) -> GraphSnapshot:
    """``nodes``: iterable of ``node(...)``; ``edges``: (source, target, type, [evidence...])."""
    graph = nx.MultiDiGraph(revision=revision)
    for entity_id, attrs in nodes:
        graph.add_node(entity_id, **attrs)
    for source, target, rtype, items in edges:
        graph.add_edge(
            source,
            target,
            key=rtype,
            edge_id=make_edge_id(source, rtype, target),
            relationship_type=rtype,
            source_entity_id=source,
            target_entity_id=target,
            supporting_evidence=list(items),
            source_document_ids=sorted({e["document_id"] for e in items}),
            source_chunk_ids=sorted({e["chunk_id"] for e in items}),
            provenance=sorted({e["provenance"] for e in items}),
            confidence=None,
        )
    return GraphSnapshot(nx.freeze(graph), revision, "now", "rebuild")


def add_indexed_doc(session, text, minute):
    """``add_doc`` plus the vector-revision bump that real ingestion performs, so the vector
    index notices the new chunk and synchronizes."""
    doc, chunk = add_doc(session, text, minute=minute)
    bump_vector_revision(session)
    session.commit()
    return doc, chunk


def seed_corpus(session):
    """Two documents about Alice, Bob, Acme Corp and Berlin. Returns the ids by role."""
    d1, c1 = add_indexed_doc(
        session, "Alice works at Acme Corp. Acme Corp is located in Berlin.", minute=0
    )
    store(
        session,
        d1,
        c1,
        ChunkExtraction(
            [E("Alice", "PERSON"), E("Acme Corp"), E("Berlin", "LOCATION")],
            [
                R("Alice", "Acme Corp", "WORKS_AT", "Alice works at Acme Corp"),
                R("Acme Corp", "Berlin", "LOCATED_IN", "Acme Corp is located in Berlin"),
            ],
        ),
    )
    d2, c2 = add_indexed_doc(
        session, "Bob works at Acme Corp. Acme Corp is located in Berlin.", minute=1
    )
    store(
        session,
        d2,
        c2,
        ChunkExtraction(
            [E("Bob", "PERSON"), E("Acme Corp"), E("Berlin", "LOCATION")],
            [
                R("Bob", "Acme Corp", "WORKS_AT", "Bob works at Acme Corp"),
                R("Acme Corp", "Berlin", "LOCATED_IN", "Acme Corp is located in Berlin"),
            ],
        ),
    )
    return {"d1": d1, "c1": c1, "d2": d2, "c2": c2}
