"""Persist extraction results and keep entity evidence consistent.

All functions work inside the caller's transaction (they never commit). Anything that writes
entities must hold ``WRITE_LOCK`` for the whole transaction: entity resolution reads the full
entity table, so two concurrent writers could otherwise create the same entity twice.
"""

import hashlib
import logging
import threading
from dataclasses import dataclass, field

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk, Entity, EntityMention, Relationship
from app.services.extraction.normalization import squash
from app.services.extraction.resolver import EntityResolver
from app.services.extraction.types import ChunkExtraction
from app.services.extraction.validation import (
    Rejection,
    ValidEntity,
    validate_entities,
    validate_relationships,
)
from app.services.graph.revision import bump_revision

logger = logging.getLogger(__name__)

WRITE_LOCK = threading.RLock()
_MAX_REPORTED_REJECTIONS = 20


def make_relationship_id(
    source_id: str, relationship_type: str, target_id: str, chunk_id: str
) -> str:
    """Deterministic evidence ID (the row's natural key), stable across reprocessing."""
    key = f"{source_id}:{relationship_type}:{target_id}:{chunk_id}"
    return hashlib.sha1(key.encode()).hexdigest()[:32]


@dataclass
class ChunkResult:
    """A chunk and what the provider returned for it (None if the provider failed on it)."""

    chunk_id: str
    text: str
    extraction: ChunkExtraction | None


@dataclass
class KnowledgeStats:
    """Counters describing one extraction run for a document."""

    entities_created: int = 0
    mentions: int = 0
    relationships: int = 0
    rejected_entities: int = 0
    rejected_relationships: int = 0
    failed_chunks: int = 0
    rejections: list[Rejection] = field(default_factory=list)

    def as_metadata(self, provider: str) -> dict:
        return {
            "extractor": provider,
            "entities_created": self.entities_created,
            "mentions": self.mentions,
            "relationships": self.relationships,
            "rejected_entities": self.rejected_entities,
            "rejected_relationships": self.rejected_relationships,
            "failed_chunks": self.failed_chunks,
            "sample_rejections": [
                {"kind": r.kind, "reason": r.reason, "item": r.item}
                for r in self.rejections[:_MAX_REPORTED_REJECTIONS]
            ],
        }


def purge_document_knowledge(session: Session, document_id: str) -> set[str]:
    """Delete a document's mentions and relationships; return the entity IDs they touched."""
    ids = set(
        session.scalars(
            select(EntityMention.entity_id).where(EntityMention.document_id == document_id)
        )
    )
    rels = session.execute(
        select(Relationship.source_entity_id, Relationship.target_entity_id).where(
            Relationship.source_document_id == document_id
        )
    )
    for src, tgt in rels:
        ids.update((src, tgt))
    session.execute(delete(EntityMention).where(EntityMention.document_id == document_id))
    session.execute(delete(Relationship).where(Relationship.source_document_id == document_id))
    session.flush()
    # Every path that changes graph-relevant evidence goes through here (ingest, re-run,
    # delete), so bumping inside this transaction keeps the graph revision exact.
    bump_revision(session)
    return ids


def _evidence_order(row) -> tuple:
    """Deterministic "earliest evidence first": upload time, document, chunk, then text."""
    mention, uploaded, chunk_index = row
    return (
        uploaded,
        mention.document_id,
        chunk_index,
        mention.extracted_name or "",
        mention.surface_form,
    )


def _source(mention: EntityMention) -> dict:
    return {"document_id": mention.document_id, "chunk_id": mention.chunk_id}


def refresh_entities(session: Session, entity_ids: set[str]) -> int:
    """Recompute entity attributes from the evidence that remains, or delete the entity.

    Everything derived is a pure function of the entity's current mentions:

    * canonical name  - the name reported by the earliest remaining evidence;
    * description     - the first non-empty description among remaining evidence, else None;
    * aliases         - surface forms (other than the canonical name) still present, each with
                        the documents that support it.

    Sources are recorded in ``entity_metadata`` (``canonical_name_source``,
    ``description_source``, ``alias_sources``). Identity (ID, normalized key, type) is never
    touched, so recomputation cannot merge or split entities. An entity with no mentions and
    no relationships is deleted (its candidates cascade). Returns the number deleted.
    """
    removed = 0
    for entity_id in sorted(entity_ids):
        entity = session.get(Entity, entity_id)
        if entity is None:
            continue
        rows = session.execute(
            select(EntityMention, Document.upload_timestamp, DocumentChunk.chunk_index)
            .join(Document, Document.id == EntityMention.document_id)
            .join(DocumentChunk, DocumentChunk.id == EntityMention.chunk_id)
            .where(EntityMention.entity_id == entity_id)
        ).all()
        rel_count = session.scalar(
            select(func.count())
            .select_from(Relationship)
            .where(
                or_(
                    Relationship.source_entity_id == entity_id,
                    Relationship.target_entity_id == entity_id,
                )
            )
        )
        if not rows and not rel_count:
            session.delete(entity)
            removed += 1
            continue

        ordered = [row[0] for row in sorted(rows, key=_evidence_order)]
        meta = dict(entity.entity_metadata or {})

        named = next((m for m in ordered if m.extracted_name), None)
        if named is not None:
            entity.canonical_name = named.extracted_name
            meta["canonical_name_source"] = _source(named)

        described = next((m for m in ordered if m.description), None)
        entity.description = described.description if described else None
        meta["description_source"] = _source(described) if described else None

        alias_docs: dict[str, set[str]] = {}
        for m in ordered:
            if m.surface_form != entity.canonical_name:
                alias_docs.setdefault(m.surface_form, set()).add(m.document_id)
        meta["aliases"] = sorted(alias_docs)
        meta["alias_sources"] = {a: sorted(docs) for a, docs in sorted(alias_docs.items())}

        if meta != (entity.entity_metadata or {}):
            entity.entity_metadata = meta
    session.flush()
    return removed


def replace_document_knowledge(
    session: Session,
    document_id: str,
    results: list[ChunkResult],
    *,
    allowed_types: set[str],
    fuzzy_threshold: float,
    provider_name: str,
) -> KnowledgeStats:
    """Atomically replace everything extracted for ``document_id`` with ``results``.

    Idempotent: running it twice with the same results yields identical rows, because old
    evidence is purged first and mentions/relationships are de-duplicated on their natural keys.
    """
    stats = KnowledgeStats()
    touched = purge_document_knowledge(session, document_id)
    resolver = EntityResolver(session, fuzzy_threshold)
    seen_mentions: dict[tuple[str, str, str], EntityMention] = {}
    seen_relationships: set[tuple[str, str, str, str]] = set()
    seen_rows: set[tuple[str, str, str, str]] = set()

    for result in results:
        if result.extraction is None:
            stats.failed_chunks += 1
            continue
        valid, rejected = validate_entities(result.text, result.extraction.entities, allowed_types)
        resolved: dict[ValidEntity, Entity] = {}
        for item in valid:
            resolution = resolver.resolve(item)
            resolved[item] = resolution.entity
            mention_key = (resolution.entity.id, result.chunk_id, item.surface_form)
            if mention_key in seen_mentions:
                existing = seen_mentions[mention_key]
                if item.description and not existing.description:
                    existing.description = item.description
                continue
            mention = EntityMention(
                entity_id=resolution.entity.id,
                document_id=document_id,
                chunk_id=result.chunk_id,
                surface_form=item.surface_form,
                confidence=item.confidence,
                provenance=provider_name,
                extracted_name=item.name,
                description=item.description,
            )
            session.add(mention)
            seen_mentions[mention_key] = mention
            stats.mentions += 1

        rels, rel_rejected = validate_relationships(
            result.text, valid, result.extraction.relationships
        )
        rejected.extend(rel_rejected)
        for rel in rels:
            src, tgt = resolved[rel.source], resolved[rel.target]
            if src.id == tgt.id:
                rejected.append(
                    Rejection(
                        "relationship", "endpoints resolve to the same entity", rel.supporting_text
                    )
                )
                continue
            # Two guards, both silent because neither is an error:
            #  - overlapping chunks repeat sentences: one record per distinct evidence text;
            #  - the table keeps one row per (source, target, type, chunk): when a chunk states
            #    the same fact twice ("A created B" / "B was created by A") the first wins.
            evidence_key = (src.id, tgt.id, rel.relationship_type, squash(rel.supporting_text))
            row_key = (src.id, tgt.id, rel.relationship_type, result.chunk_id)
            if evidence_key in seen_relationships or row_key in seen_rows:
                continue
            seen_relationships.add(evidence_key)
            seen_rows.add(row_key)
            session.add(
                Relationship(
                    id=make_relationship_id(src.id, rel.relationship_type, tgt.id, result.chunk_id),
                    source_entity_id=src.id,
                    target_entity_id=tgt.id,
                    relationship_type=rel.relationship_type,
                    supporting_text=rel.supporting_text,
                    source_document_id=document_id,
                    source_chunk_id=result.chunk_id,
                    confidence=rel.confidence,
                    provenance=provider_name,
                )
            )
            stats.relationships += 1

        stats.rejected_entities += sum(1 for r in rejected if r.kind == "entity")
        stats.rejected_relationships += sum(1 for r in rejected if r.kind == "relationship")
        stats.rejections.extend(rejected)

    session.flush()
    stats.entities_created = len(resolver.created)
    touched.update(e.id for e in resolver.created)
    touched.update(
        session.scalars(
            select(EntityMention.entity_id).where(EntityMention.document_id == document_id)
        )
    )
    refresh_entities(session, touched)
    logger.info(
        "Knowledge stored: document=%s new_entities=%d mentions=%d relationships=%d rejected=%d",
        document_id,
        stats.entities_created,
        stats.mentions,
        stats.relationships,
        stats.rejected_entities + stats.rejected_relationships,
    )
    return stats
