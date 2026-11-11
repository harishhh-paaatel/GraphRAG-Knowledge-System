"""Validation of raw provider output against the chunk it came from.

Nothing is persisted unless it is traceable to the chunk text: entity surface forms and
relationship supporting text must actually occur in the chunk, and both relationship
endpoints must occur in the supporting text.
"""

from dataclasses import dataclass

from app.services.extraction.normalization import clean_display_name, normalize_key, squash
from app.services.extraction.types import (
    RELATIONSHIP_RULES,
    ExtractedEntity,
    ExtractedRelationship,
)

MAX_NAME = 255
MAX_DESCRIPTION = 1000


@dataclass(frozen=True)
class Rejection:
    """Why an item was dropped (kept for logging and per-document statistics)."""

    kind: str  # "entity" | "relationship"
    reason: str
    item: str


@dataclass(frozen=True)
class ValidEntity:
    """An entity whose surface form was found in the chunk."""

    name: str
    entity_type: str
    surface_form: str
    description: str | None
    confidence: float | None


@dataclass(frozen=True)
class ValidRelationship:
    """A relationship whose evidence and endpoints were verified."""

    source: ValidEntity
    target: ValidEntity
    relationship_type: str
    supporting_text: str
    confidence: float | None


def _bad_confidence(value: float | None) -> bool:
    return value is not None and not (0.0 <= value <= 1.0)


def validate_entities(
    chunk_text: str, entities: list[ExtractedEntity], allowed_types: set[str] | frozenset[str]
) -> tuple[list[ValidEntity], list[Rejection]]:
    """Keep entities with a known type whose text occurs in the chunk; dedupe exact repeats."""
    haystack = squash(chunk_text)
    valid: list[ValidEntity] = []
    rejected: list[Rejection] = []
    seen: set[tuple[str, str, str]] = set()
    for ent in entities:
        label = f"{ent.entity_type}:{ent.name!r}"
        name = clean_display_name(ent.name or "")
        etype = (ent.entity_type or "").strip().upper()
        if not name or len(name) > MAX_NAME:
            rejected.append(Rejection("entity", "invalid name", label))
            continue
        if etype not in allowed_types:
            rejected.append(Rejection("entity", f"type {etype!r} not enabled", label))
            continue
        if _bad_confidence(ent.confidence):
            rejected.append(Rejection("entity", "confidence outside [0, 1]", label))
            continue
        surface = next(
            (
                clean_display_name(c)
                for c in (ent.surface_form, name)
                if c and clean_display_name(c) and squash(c) in haystack
            ),
            None,
        )
        if surface is None:
            rejected.append(Rejection("entity", "text not found in chunk", label))
            continue
        marker = (normalize_key(name, etype), etype, surface)
        if marker in seen:
            continue
        seen.add(marker)
        description = (ent.description or "").strip()[:MAX_DESCRIPTION] or None
        valid.append(ValidEntity(name, etype, surface, description, ent.confidence))
    return valid, rejected


def _identity(ent: ValidEntity) -> tuple[str, str]:
    return normalize_key(ent.name, ent.entity_type), ent.entity_type


def _lookup(entities: list[ValidEntity]) -> dict[str, list[ValidEntity]]:
    index: dict[str, list[ValidEntity]] = {}
    for ent in entities:
        for text in (ent.name, ent.surface_form):
            bucket = index.setdefault(squash(text), [])
            # Different surface forms of the same entity are not ambiguous; different entities are.
            if all(_identity(ent) != _identity(other) for other in bucket):
                bucket.append(ent)
    return index


def validate_relationships(
    chunk_text: str,
    entities: list[ValidEntity],
    relationships: list[ExtractedRelationship],
) -> tuple[list[ValidRelationship], list[Rejection]]:
    """Accept only relationships that are typed correctly and evidenced by the chunk."""
    haystack = squash(chunk_text)
    index = _lookup(entities)
    valid: list[ValidRelationship] = []
    rejected: list[Rejection] = []

    for rel in relationships:
        label = f"{rel.source!r} -{rel.relationship_type}-> {rel.target!r}"

        def reject(reason: str, _label: str = label) -> None:
            rejected.append(Rejection("relationship", reason, _label))

        rtype = (rel.relationship_type or "").strip().upper()
        if rtype not in RELATIONSHIP_RULES:
            reject(f"unknown relationship type {rtype!r}")
            continue
        if _bad_confidence(rel.confidence):
            reject("confidence outside [0, 1]")
            continue
        support = (rel.supporting_text or "").strip()
        if not support:
            reject("missing supporting text")
            continue
        if squash(support) not in haystack:
            reject("supporting text not found in chunk")
            continue

        src_matches = index.get(squash(rel.source or ""), [])
        tgt_matches = index.get(squash(rel.target or ""), [])
        if not src_matches or not tgt_matches:
            reject("endpoint is not an entity extracted from this chunk")
            continue
        if len(src_matches) > 1 or len(tgt_matches) > 1:
            reject("endpoint name is ambiguous within the chunk")
            continue
        src, tgt = src_matches[0], tgt_matches[0]
        if normalize_key(src.name, src.entity_type) == normalize_key(tgt.name, tgt.entity_type):
            reject("self-relationship")
            continue

        squashed_support = squash(support)
        if not all(
            squash(e.surface_form) in squashed_support or squash(e.name) in squashed_support
            for e in (src, tgt)
        ):
            reject("supporting text does not mention both entities")
            continue

        allowed_src, allowed_tgt = RELATIONSHIP_RULES[rtype]
        if (allowed_src is not None and src.entity_type not in allowed_src) or (
            allowed_tgt is not None and tgt.entity_type not in allowed_tgt
        ):
            reject(f"{src.entity_type} -> {tgt.entity_type} is not valid for {rtype}")
            continue

        valid.append(ValidRelationship(src, tgt, rtype, support, rel.confidence))
    return valid, rejected
