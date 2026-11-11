"""Entity, mention and relationship models."""

from datetime import datetime

from sqlalchemy import JSON, Float, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, UTCDateTime, new_id, utcnow


class Entity(Base):
    """A canonical entity. ``normalized_key`` drives exact-match deduplication."""

    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("normalized_key", "entity_type", name="uq_entities_key_type"),
        Index("ix_entities_type", "entity_type"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    canonical_name: Mapped[str] = mapped_column(String(255))
    normalized_key: Mapped[str] = mapped_column(String(255))
    entity_type: Mapped[str] = mapped_column(String(64))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    entity_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    mentions: Mapped[list["EntityMention"]] = relationship(
        back_populates="entity", cascade="all, delete-orphan", passive_deletes=True
    )


class EntityMention(Base):
    """Evidence that an entity appears in a specific chunk."""

    __tablename__ = "entity_mentions"
    __table_args__ = (
        Index("ix_mentions_chunk", "chunk_id"),
        UniqueConstraint("entity_id", "chunk_id", "surface_form", name="uq_mention_evidence"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    entity_id: Mapped[str] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_id: Mapped[str] = mapped_column(ForeignKey("document_chunks.id", ondelete="CASCADE"))
    surface_form: Mapped[str] = mapped_column(String(255))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    provenance: Mapped[str] = mapped_column(String(64), default="unknown", server_default="unknown")
    # What the extractor reported for THIS evidence. Entity name/description/aliases are
    # derived from these rows, so deleting a document can never leave stale values behind.
    extracted_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    entity: Mapped[Entity] = relationship(back_populates="mentions")


class Relationship(Base):
    """A directed, evidence-backed relationship between two entities."""

    __tablename__ = "relationships"
    __table_args__ = (
        UniqueConstraint(
            "source_entity_id",
            "target_entity_id",
            "relationship_type",
            "source_chunk_id",
            name="uq_relationship_evidence",
        ),
        Index("ix_relationships_source", "source_entity_id"),
        Index("ix_relationships_target", "target_entity_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    source_entity_id: Mapped[str] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"))
    target_entity_id: Mapped[str] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"))
    relationship_type: Mapped[str] = mapped_column(String(64))
    supporting_text: Mapped[str] = mapped_column(Text)
    source_document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    source_chunk_id: Mapped[str] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE")
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    provenance: Mapped[str] = mapped_column(String(64), default="unknown", server_default="unknown")


class EntityCandidate(Base):
    """A possible-duplicate pair flagged for later review. Never merged automatically.

    ``entity_a_id < entity_b_id`` always, so each unordered pair has one row per reason.
    """

    __tablename__ = "entity_candidates"
    __table_args__ = (
        UniqueConstraint("entity_a_id", "entity_b_id", "reason", name="uq_candidate_pair_reason"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    entity_a_id: Mapped[str] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    entity_b_id: Mapped[str] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    reason: Mapped[str] = mapped_column(String(32))
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", server_default="pending")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
