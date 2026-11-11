"""Conservative exact entity linking: find graph entities that a query explicitly names.

Only exact matching of normalised names and aliases, on whole-token boundaries. There is no
fuzzy matching, no stemming, no substring or prefix search and no LLM extraction. When the
evidence is not unambiguous the linker reports the ambiguity with every candidate and links
nothing: it never selects an entity on the caller's behalf and never merges entities.

Rules (docs/phase-6-proposal.md, section 4.2):

* Both sides use ``normalize_key``: Unicode NFKC, case-folded, whitespace-insensitive, edge
  punctuation stripped. A name matches only as a whole sequence of consecutive query tokens, so
  "Acme" is not found inside "Acmeville" and "Al" not inside "Alice".
* Names and aliases are both indexed. ``match_type`` is ``exact_name`` when the canonical name
  produced the key, otherwise ``exact_alias``.
* Corporate suffixes ("Inc", "Corp", "Ltd", ...) are removed from a query n-gram only to look up
  ORGANIZATION entities, mirroring how organisation keys are built at ingestion.
* A trailing possessive ('s) on the last token of a name is accepted as the one morphological
  allowance: "Alice's manager" links Alice. Plurals and stems are not.
* Names of a single token shorter than 4 characters ("Go", "AI", "IBM") link only when the query
  spells them exactly as stored (case-sensitive), so "where do I go" never links an entity "Go".
  Single-character names are not linkable at all.
* Sentence and clause punctuation (. , ; : ! ?) ends a name: "I like Acme. Corp is big" does not
  link "Acme Corp". The price is that a stored name with an internal ". " (such as "St. Louis")
  is not matched.
* Longest match wins: a match fully inside a longer match is suppressed (and reported). Two
  matches that overlap without one containing the other are ambiguous and neither is linked.
* One span matching several entities is ambiguous. Explicit ``entity_ids`` resolve it.

This module depends on the graph snapshot and on name normalisation only; it must not import
anything from the vector side (a test enforces that).
"""

import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx

from app.services.extraction.normalization import clean_display_name, normalize_key
from app.services.graph.service import GraphSnapshot

MATCH_NAME = "exact_name"
MATCH_ALIAS = "exact_alias"
SHORT_NAME_LENGTH = 4  # single-token keys shorter than this need an exact-case match
_SUFFIX_SLACK = 4  # a query n-gram may carry up to this many extra corporate-suffix tokens
_SENTENCE_END = ".,;:!?"
_CLOSERS = "\"')]}»”’"
_EDGE_PUNCT = ".,;:!?\"'()[]{}<>/-‘’“”"
_POSSESSIVE = re.compile(r"(?:'s|’s)$", re.IGNORECASE)

AmbiguousReason = Literal["multiple_entities", "overlapping_spans"]


class UnknownEntities(KeyError):
    """Explicit ``entity_ids`` that are not in the graph."""

    def __init__(self, entity_ids: list[str]) -> None:
        super().__init__(", ".join(entity_ids))
        self.entity_ids = entity_ids


# ----------------------------------------------------------------------------- results


@dataclass(frozen=True)
class Candidate:
    """One entity a span could refer to."""

    entity_id: str
    name: str
    entity_type: str
    match_type: str
    document_count: int


@dataclass(frozen=True)
class LinkedEntity:
    entity_id: str
    name: str
    entity_type: str
    span: str | None  # the query words that matched; None for an entity given only by ID
    match_type: str | None
    resolved_by: Literal["query", "entity_ids"]


@dataclass(frozen=True)
class AmbiguousMention:
    """A mention that could not be linked without guessing."""

    span: str
    reason: AmbiguousReason
    candidates: tuple[Candidate, ...]
    spans: tuple[str, ...]  # every overlapping span involved (one entry unless overlapping)
    action: Literal["skipped", "resolved"]
    resolved_to: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuppressedSpan:
    span: str
    contained_in: str


@dataclass(frozen=True)
class LinkResult:
    linked: tuple[LinkedEntity, ...] = ()
    ambiguous: tuple[AmbiguousMention, ...] = ()
    suppressed: tuple[SuppressedSpan, ...] = ()
    seeds: tuple[str, ...] = ()  # entity IDs to expand, sorted


# ------------------------------------------------------------------------------- index


@dataclass(frozen=True)
class _Entry:
    entity_id: str
    entity_type: str
    match_type: str
    forms: frozenset[str]  # stored spellings that produce this key, whitespace-collapsed


@dataclass(frozen=True)
class AliasIndex:
    revision: int
    by_key: dict[str, tuple[_Entry, ...]]
    max_key_tokens: int
    skipped_too_long: int = 0

    @classmethod
    def build(cls, graph: nx.MultiDiGraph, revision: int, max_name_tokens: int) -> "AliasIndex":
        gathered: dict[tuple[str, str], dict] = {}
        longest, skipped = 1, 0
        for node_id in sorted(graph.nodes):
            attrs = graph.nodes[node_id]
            etype = attrs["entity_type"]
            names = [(attrs["canonical_name"], MATCH_NAME)] + [
                (alias, MATCH_ALIAS) for alias in attrs["aliases"]
            ]
            for text, kind in names:
                key = normalize_key(text, etype)
                tokens = len(key.split())
                if not key or len(key) < 2:
                    continue
                if tokens > max_name_tokens:
                    skipped += 1
                    continue
                longest = max(longest, tokens)
                slot = gathered.setdefault(
                    (key, node_id), {"type": etype, "kind": MATCH_ALIAS, "forms": set()}
                )
                if kind == MATCH_NAME:
                    slot["kind"] = MATCH_NAME
                cleaned = clean_display_name(text)
                slot["forms"].add(cleaned)
                # the spelling of the name proper, as written, without a trailing corporate suffix
                words = cleaned.split()
                for size in range(1, len(words)):
                    if normalize_key(" ".join(words[:size]), etype) == key:
                        slot["forms"].add(" ".join(words[:size]))
        by_key: dict[str, list[_Entry]] = defaultdict(list)
        for (key, node_id), slot in sorted(gathered.items()):
            by_key[key].append(
                _Entry(node_id, slot["type"], slot["kind"], frozenset(slot["forms"]))
            )
        return cls(revision, {k: tuple(v) for k, v in by_key.items()}, longest, skipped)


# ----------------------------------------------------------------------------- tokens


@dataclass(frozen=True)
class _Token:
    norm: str
    raw: str  # as written, edge punctuation removed
    alt: str | None  # the token without a trailing possessive
    alt_raw: str | None
    boundary_after: bool


def tokenize_query(query: str) -> list[_Token]:
    """Split a query into normalised tokens, keeping the original spelling for the case guard."""
    tokens: list[_Token] = []
    for word in clean_display_name(query).split():
        parts = normalize_key(word).split()
        stripped_end = word.rstrip(_CLOSERS)
        boundary = stripped_end[-1:] in _SENTENCE_END if stripped_end else False
        if not parts:  # punctuation only ("-", "?", an em dash): never joins its neighbours
            if tokens and not tokens[-1].boundary_after:
                last = tokens[-1]
                tokens[-1] = _Token(last.norm, last.raw, last.alt, last.alt_raw, True)
            continue
        raw = word.strip(_EDGE_PUNCT)
        for index, part in enumerate(parts):
            is_last = index == len(parts) - 1
            alt = part[:-2] if part.endswith("'s") and len(part) > 2 else None
            alt_raw = _POSSESSIVE.sub("", raw) if alt is not None and is_last else None
            tokens.append(
                _Token(
                    part,
                    raw if len(parts) == 1 else part,
                    alt,
                    alt_raw,
                    boundary and is_last,
                )
            )
    return tokens


# --------------------------------------------------------------------------- the linker


@dataclass
class _Span:
    start: int  # token index, inclusive
    end: int  # token index, exclusive
    text: str
    candidates: dict[str, Candidate] = field(default_factory=dict)

    def contains(self, other: "_Span") -> bool:
        return self.start <= other.start and other.end <= self.end and self != other

    def overlaps(self, other: "_Span") -> bool:
        return self.start < other.end and other.start < self.end


class EntityLinker:
    """Links query text to graph entities. The alias index is cached per graph revision."""

    def __init__(self, max_name_tokens: int = 8) -> None:
        self._max_name_tokens = max_name_tokens
        self._lock = threading.Lock()
        self._index: AliasIndex | None = None
        self.builds = 0  # how many times an index was built (observable in tests)

    def index_for(self, snapshot: GraphSnapshot) -> AliasIndex:
        with self._lock:
            if self._index is None or self._index.revision != snapshot.revision:
                self._index = AliasIndex.build(
                    snapshot.graph, snapshot.revision, self._max_name_tokens
                )
                self.builds += 1
            return self._index

    def link(
        self,
        snapshot: GraphSnapshot,
        query: str,
        *,
        entity_ids: list[str] | tuple[str, ...] = (),
        auto_link: bool = True,
    ) -> LinkResult:
        graph = snapshot.graph
        pinned = list(dict.fromkeys(entity_ids))
        unknown = sorted(i for i in pinned if i not in graph)
        if unknown:
            raise UnknownEntities(unknown)

        linked: list[LinkedEntity] = []
        ambiguous: list[AmbiguousMention] = []
        suppressed: list[SuppressedSpan] = []
        if auto_link:
            index = self.index_for(snapshot)
            spans = self._find_spans(index, tokenize_query(query), graph)
            linked, ambiguous, suppressed = self._resolve(spans, graph, set(pinned))

        seeds = {entity.entity_id for entity in linked}
        for entity_id in pinned:
            if entity_id not in seeds:
                attrs = graph.nodes[entity_id]
                linked.append(
                    LinkedEntity(
                        entity_id,
                        attrs["canonical_name"],
                        attrs["entity_type"],
                        None,
                        None,
                        "entity_ids",
                    )
                )
                seeds.add(entity_id)
        return LinkResult(
            linked=tuple(linked),
            ambiguous=tuple(ambiguous),
            suppressed=tuple(suppressed),
            seeds=tuple(sorted(seeds)),
        )

    # -- span discovery --------------------------------------------------------------

    def _find_spans(
        self, index: AliasIndex, tokens: list[_Token], graph: nx.MultiDiGraph
    ) -> list[_Span]:
        longest = index.max_key_tokens + _SUFFIX_SLACK
        spans: list[_Span] = []
        for start in range(len(tokens)):
            for end in range(start + 1, min(len(tokens), start + longest) + 1):
                if any(tokens[k].boundary_after for k in range(start, end - 1)):
                    break  # punctuation ends a name: longer n-grams would cross it
                found = self._lookup(index, tokens[start:end], graph)
                if found:
                    text = " ".join(t.raw for t in tokens[start:end])
                    spans.append(_Span(start, end, text, found))
        return spans

    def _lookup(
        self, index: AliasIndex, window: list[_Token], graph: nx.MultiDiGraph
    ) -> dict[str, Candidate]:
        found: dict[str, Candidate] = {}
        variants = [([t.norm for t in window], False)]
        if window[-1].alt is not None:
            variants.append(([t.norm for t in window[:-1]] + [window[-1].alt], True))
        for norms, possessive in variants:
            key = " ".join(norms)
            raws = [t.raw for t in window]
            if possessive and window[-1].alt_raw is not None:
                raws[-1] = window[-1].alt_raw
            self._collect(index, key, raws, len(norms), None, graph, found)
            stripped = normalize_key(key, "ORGANIZATION")
            if stripped and stripped != key:
                count = len(stripped.split())
                self._collect(index, stripped, raws, count, "ORGANIZATION", graph, found)
        return found

    @staticmethod
    def _collect(
        index: AliasIndex,
        key: str,
        raws: list[str],
        core_tokens: int,
        only_type: str | None,
        graph: nx.MultiDiGraph,
        found: dict[str, Candidate],
    ) -> None:
        short = " " not in key and len(key) < SHORT_NAME_LENGTH
        core_raw = " ".join(raws[:core_tokens])
        for entry in index.by_key.get(key, ()):
            if only_type is not None and entry.entity_type != only_type:
                continue
            if short and core_raw not in entry.forms:
                continue  # short names need the exact capitalisation that was stored
            previous = found.get(entry.entity_id)
            if previous is not None and previous.match_type == MATCH_NAME:
                continue
            attrs = graph.nodes[entry.entity_id]
            found[entry.entity_id] = Candidate(
                entry.entity_id,
                attrs["canonical_name"],
                attrs["entity_type"],
                entry.match_type,
                len(attrs["source_document_ids"]),
            )

    # -- resolution -----------------------------------------------------------------

    @staticmethod
    def _resolve(
        spans: list[_Span], graph: nx.MultiDiGraph, pinned: set[str]
    ) -> tuple[list[LinkedEntity], list[AmbiguousMention], list[SuppressedSpan]]:
        suppressed: list[SuppressedSpan] = []
        maximal: list[_Span] = []
        for span in spans:
            container = next(
                (
                    other
                    for other in sorted(spans, key=lambda s: (s.start - s.end, s.start))
                    if other.contains(span)
                ),
                None,
            )
            if container is None:
                maximal.append(span)
            elif not set(span.candidates) <= set(container.candidates):
                # only worth reporting when the shorter match meant a different entity
                suppressed.append(SuppressedSpan(span.text, container.text))

        # connected groups of maximal spans that overlap each other
        groups: list[list[_Span]] = []
        for span in sorted(maximal, key=lambda s: (s.start, s.end)):
            if groups and any(span.overlaps(member) for member in groups[-1]):
                groups[-1].append(span)
            else:
                groups.append([span])

        linked: list[LinkedEntity] = []
        ambiguous: list[AmbiguousMention] = []
        for group in groups:
            if len(group) == 1 and len(group[0].candidates) == 1:
                span = group[0]
                (cand,) = span.candidates.values()
                linked.append(_linked(cand, span.text, "query"))
                continue
            reason: AmbiguousReason = "overlapping_spans" if len(group) > 1 else "multiple_entities"
            merged: dict[str, Candidate] = {}
            for span in group:
                for cid, cand in span.candidates.items():
                    merged.setdefault(cid, cand)
            candidates = tuple(sorted(merged.values(), key=lambda c: c.entity_id))
            chosen = tuple(c.entity_id for c in candidates if c.entity_id in pinned)
            label = " / ".join(span.text for span in group)
            if chosen:
                for span in group:
                    for cid in span.candidates:
                        if cid in chosen:
                            linked.append(_linked(merged[cid], span.text, "entity_ids"))
            ambiguous.append(
                AmbiguousMention(
                    span=label,
                    reason=reason,
                    candidates=candidates,
                    spans=tuple(span.text for span in group),
                    action="resolved" if chosen else "skipped",
                    resolved_to=chosen,
                )
            )
        return linked, ambiguous, suppressed


def _linked(
    cand: Candidate, span: str, resolved_by: Literal["query", "entity_ids"]
) -> LinkedEntity:
    return LinkedEntity(
        cand.entity_id, cand.name, cand.entity_type, span, cand.match_type, resolved_by
    )
