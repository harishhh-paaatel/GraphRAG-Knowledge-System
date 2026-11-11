"""Build the knowledge graph from SQLite.

Graph model
-----------
``networkx.MultiDiGraph`` (not DiGraph): one entity pair can legitimately carry several
relationship types (``A WORKS_AT B`` and ``A RELATED_TO B``). Each *(source, type, target)*
is exactly one edge, keyed by its relationship type, and aggregates **all** evidence rows
that support it, so nothing is lost when several documents or chunks state the same fact.
Direction is exactly the stored direction; the builder never reverses anything.

Only supported records become graph elements: an entity needs at least one mention, and a
relationship needs both endpoint nodes to exist.
"""

import hashlib
import logging
from collections import defaultdict

import networkx as nx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Entity, EntityMention, Relationship
from app.services.graph.revision import current_revision

logger = logging.getLogger(__name__)


def make_edge_id(source_id: str, relationship_type: str, target_id: str) -> str:
    """Stable, deterministic edge identifier."""
    return hashlib.sha1(f"{source_id}:{relationship_type}:{target_id}".encode()).hexdigest()[:32]


def build_graph(session: Session) -> tuple[nx.MultiDiGraph, int]:
    """Read SQLite and return ``(frozen graph, revision it reflects)``.

    The caller must hold the knowledge write lock so the reads see one consistent state.
    """
    revision = current_revision(session)

    mention_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "docs": set()})
    for entity_id, document_id, n in session.execute(
        select(EntityMention.entity_id, EntityMention.document_id, func.count()).group_by(
            EntityMention.entity_id, EntityMention.document_id
        )
    ):
        stats = mention_stats[entity_id]
        stats["count"] += n
        stats["docs"].add(document_id)

    graph = nx.MultiDiGraph(revision=revision)
    for entity in session.scalars(select(Entity).order_by(Entity.id)):
        stats = mention_stats.get(entity.id)
        if stats is None:
            logger.warning("Skipping entity without mentions: %s", entity.id)
            continue
        meta = entity.entity_metadata or {}
        graph.add_node(
            entity.id,
            entity_id=entity.id,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            description=entity.description,
            aliases=list(meta.get("aliases", [])),
            mention_count=stats["count"],
            source_document_ids=sorted(stats["docs"]),
            normalized_key=entity.normalized_key,
        )

    grouped: dict[tuple[str, str, str], list[Relationship]] = defaultdict(list)
    for rel in session.scalars(select(Relationship)):
        grouped[(rel.source_entity_id, rel.relationship_type, rel.target_entity_id)].append(rel)

    for (src, rtype, tgt), rows in sorted(grouped.items()):
        if src not in graph or tgt not in graph:
            logger.warning(
                "Skipping relationship with unsupported endpoint: %s -%s-> %s", src, rtype, tgt
            )
            continue
        evidence = sorted(
            (
                {
                    "relationship_id": r.id,
                    "supporting_text": r.supporting_text,
                    "document_id": r.source_document_id,
                    "chunk_id": r.source_chunk_id,
                    "provenance": r.provenance,
                    "confidence": r.confidence,
                }
                for r in rows
            ),
            key=lambda e: (e["document_id"], e["chunk_id"], e["supporting_text"]),
        )
        confidences = [e["confidence"] for e in evidence if e["confidence"] is not None]
        graph.add_edge(
            src,
            tgt,
            key=rtype,
            edge_id=make_edge_id(src, rtype, tgt),
            relationship_type=rtype,
            source_entity_id=src,
            target_entity_id=tgt,
            supporting_evidence=evidence,
            source_document_ids=sorted({e["document_id"] for e in evidence}),
            source_chunk_ids=sorted({e["chunk_id"] for e in evidence}),
            provenance=sorted({e["provenance"] for e in evidence}),
            # Never fabricated: None unless an extractor actually reported one.
            confidence=max(confidences) if confidences else None,
        )
    return nx.freeze(graph), revision
