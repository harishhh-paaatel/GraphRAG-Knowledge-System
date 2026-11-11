"""Conservative entity resolution.

Policy (strictest first):

1. Same normalized key + same type            -> same entity (merge).
2. Same compact key (spaces/joiners removed)  -> merge ("Open AI" ~ "OpenAI"), but only when
   same type, compact key >= 5 chars, and exactly one entity matches. Several matches means
   the evidence is ambiguous, so a new entity is created and each match is flagged.
3. Everything else stays a separate entity. Possible duplicates are *recorded* in
   ``entity_candidates`` for later review, never merged:
     - ``same_name_different_type``  (e.g. "Mercury" planet vs element)
     - ``fuzzy_name``                (high string similarity, same type, same digits)
     - ``token_subset``              (person/org names: "Ramirez" within "Sebastian Ramirez")
     - ``compact_ambiguous``         (rule 2 matched more than one entity)
"""

import difflib
import hashlib
import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Entity, EntityCandidate
from app.services.extraction.normalization import compact_key, normalize_key
from app.services.extraction.validation import ValidEntity

_MIN_COMPACT_LEN = 5
_MIN_FUZZY_LEN = 6
_NAME_TYPES = frozenset({"PERSON", "ORGANIZATION"})
_DIGITS = re.compile(r"\d+")


def make_entity_id(entity_type: str, normalized_key: str) -> str:
    """Deterministic ID so reprocessing (which may delete and recreate an entity) keeps the
    same identity, which later phases rely on for graph nodes and vector metadata."""
    return hashlib.sha1(f"{entity_type}:{normalized_key}".encode()).hexdigest()[:32]


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving one extracted entity."""

    entity: Entity
    created: bool
    how: str  # "created" | "exact" | "compact"


class EntityResolver:
    """Resolves extracted entities against the whole entity table within one transaction.

    Loads all entities once (fine for the project's scale; see limitations) and keeps its
    indexes up to date as it creates new ones, so results do not depend on flush timing.
    The caller must serialize writers (see ``persistence.WRITE_LOCK``).
    """

    def __init__(self, session: Session, fuzzy_threshold: float) -> None:
        self._session = session
        self._threshold = fuzzy_threshold
        self._by_type_key: dict[tuple[str, str], Entity] = {}
        self._by_key: dict[str, list[Entity]] = {}
        self._by_type_compact: dict[tuple[str, str], list[Entity]] = {}
        self._by_type: dict[str, list[Entity]] = {}
        self._pairs: set[tuple[str, str, str]] = set()
        for ent in session.scalars(select(Entity)):
            self._index(ent)
        for c in session.scalars(select(EntityCandidate)):
            self._pairs.add((c.entity_a_id, c.entity_b_id, c.reason))
        self.created: list[Entity] = []

    def _index(self, ent: Entity) -> None:
        self._by_type_key[(ent.entity_type, ent.normalized_key)] = ent
        self._by_key.setdefault(ent.normalized_key, []).append(ent)
        self._by_type_compact.setdefault(
            (ent.entity_type, compact_key(ent.normalized_key)), []
        ).append(ent)
        self._by_type.setdefault(ent.entity_type, []).append(ent)

    def resolve(self, item: ValidEntity) -> Resolution:
        """Return the entity this mention belongs to, creating one if nothing safely matches."""
        key = normalize_key(item.name, item.entity_type)
        compact = compact_key(key)

        existing = self._by_type_key.get((item.entity_type, key))
        if existing is not None:
            return Resolution(existing, False, "exact")

        compact_hits = (
            self._by_type_compact.get((item.entity_type, compact), [])
            if len(compact) >= _MIN_COMPACT_LEN
            else []
        )
        if len(compact_hits) == 1:
            return Resolution(compact_hits[0], False, "compact")

        entity = Entity(
            id=make_entity_id(item.entity_type, key),
            canonical_name=item.name,
            normalized_key=key,
            entity_type=item.entity_type,
            description=item.description,
            entity_metadata={},
        )
        self._session.add(entity)
        self.created.append(entity)

        for hit in compact_hits:  # ambiguous: several existing entities share the compact key
            self._flag(entity, hit, "compact_ambiguous", None)
        for other in self._by_key.get(key, []):
            self._flag(entity, other, "same_name_different_type", 1.0)
        self._flag_similar(entity, key)
        self._index(entity)
        return Resolution(entity, True, "created")

    def _flag_similar(self, entity: Entity, key: str) -> None:
        digits = _DIGITS.findall(key)
        tokens = set(key.split())
        for other in self._by_type.get(entity.entity_type, []):
            okey = other.normalized_key
            if okey == key:
                continue
            if len(key) >= _MIN_FUZZY_LEN and len(okey) >= _MIN_FUZZY_LEN:
                matcher = difflib.SequenceMatcher(None, key, okey)
                if (
                    matcher.real_quick_ratio() >= self._threshold
                    and matcher.quick_ratio() >= self._threshold
                    and _DIGITS.findall(okey) == digits
                ):
                    ratio = matcher.ratio()
                    if ratio >= self._threshold:
                        self._flag(entity, other, "fuzzy_name", round(ratio, 4))
                        continue
            if entity.entity_type in _NAME_TYPES:
                otokens = set(okey.split())
                small, large = sorted((tokens, otokens), key=len)
                if small < large and any(len(t) >= 3 for t in small):
                    self._flag(entity, other, "token_subset", None)

    def _flag(self, new: Entity, other: Entity, reason: str, score: float | None) -> None:
        a, b = sorted((new.id, other.id))
        if (a, b, reason) in self._pairs:
            return
        self._pairs.add((a, b, reason))
        self._session.add(EntityCandidate(entity_a_id=a, entity_b_id=b, reason=reason, score=score))
