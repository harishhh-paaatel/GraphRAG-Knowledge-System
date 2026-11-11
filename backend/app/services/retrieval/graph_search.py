"""Graph retrieval: query entities -> bounded directed neighbourhood -> evidence chunks.

The graph path is independent of the vector side. It uses the knowledge-graph snapshot, SQLite
and nothing else: no FAISS, no embeddings, no similarity score. A test enforces the import
boundary, and another runs the whole path with the vector service replaced by one that raises.

Pipeline
--------
1. ``EntityLinker`` finds the entities the query explicitly names (exact matching only).
2. ``traverse`` walks from those seeds ``hops`` hops in the requested direction. Edges are
   returned with their stored direction; the walk label only says how they were reached.
3. Every supporting-evidence item of every walked edge nominates its chunk as a candidate. With
   ``include_mentions`` the seeds' mention chunks are added as weaker, mention-only candidates.
4. Candidates are ranked by the lexicographic key below. 5. The ranked chunks are checked
   against SQLite (a chunk or document deleted after the graph was built is dropped and
   counted, never invented) and cut to ``max_chunks``.

Ranking (all integers or booleans; no floats, no similarity)
-----------------------------------------------------------
For a candidate chunk ``c`` let ``E(c)`` be the walked edges with evidence in ``c``::

    sort key = ( -seed_coverage, -connects_seeds, -relationship_evidence, distance,
                 -support, -corroboration, -provenance_known, chunk_id )

* seed_coverage          distinct seeds that are an endpoint of an edge in E(c) or are mentioned
                         in c (when mentions are enabled)
* connects_seeds         some edge in E(c) has both endpoints among the seeds
* relationship_evidence  E(c) is not empty (mention-only chunks rank after relationship chunks
                         with the same coverage)
* distance               smallest edge distance in E(c); edge distance is
                         ``min(depth(source), depth(target)) + 1`` (0 for mention-only chunks)
* support                number of distinct relationships evidenced in c
* corroboration          the largest number of distinct documents supporting one edge of E(c)
* provenance_known       some evidence in c has a provenance other than ``"unknown"``
* chunk_id               ascending; unique, so the order is total and ties cannot survive

``decided_by`` names the first key on which a chunk differs from the chunk ranked just above it
(``chunk_id`` means every earlier key tied). Confidence is deliberately not a key (it is unset
for every row today) and neither is degree centrality.

A deterministic order is reproducible, not necessarily relevant: this ranking says "close to the
entities the query names, with more distinct supporting relationships", never "answers the
question". Whether it is useful has to be measured against labelled data (Phase 8).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.models import EntityMention, Relationship
from app.services.graph.queries import EdgeRef, Traversal, traverse
from app.services.graph.service import GraphService, GraphSnapshot
from app.services.retrieval.entity_linker import EntityLinker, LinkResult
from app.services.retrieval.scope import (
    ChunkRecord,
    DocumentScope,
    lookup_chunks,
    resolve_document_scope,
)

logger = logging.getLogger(__name__)

Direction = Literal["outgoing", "incoming", "both"]
DIRECTIONS: tuple[str, ...] = ("outgoing", "incoming", "both")
_TRAVERSE_DIRECTION = {"outgoing": "out", "incoming": "in", "both": "both"}
RANK_KEYS = (
    "seed_coverage",
    "connects_seeds",
    "relationship_evidence",
    "distance",
    "support",
    "corroboration",
    "provenance_known",
    "chunk_id",
)
DEFAULT_MENTIONS_PER_ENTITY = 5


class GraphSearchLimitError(ValueError):
    """A request parameter exceeds a configured cap."""


# ------------------------------------------------------------------------------ values


@dataclass(frozen=True)
class EntityRef:
    entity_id: str
    name: str
    entity_type: str


@dataclass(frozen=True)
class EvidenceItem:
    """One piece of graph evidence for a chunk, exactly as stored."""

    kind: Literal["relationship", "mention"]
    document_id: str
    provenance: str
    confidence: float | None
    relationship_type: str | None = None
    source: EntityRef | None = None  # stored direction: source -> target
    target: EntityRef | None = None
    entity: EntityRef | None = None  # mention evidence
    traversal: str | None = None  # outgoing | incoming: how the walk reached the edge
    reached_from: EntityRef | None = None
    distance: int | None = None
    edge_id: str | None = None
    relationship_id: str | None = None
    supporting_text: str | None = None
    surface_form: str | None = None


@dataclass(frozen=True)
class RankFeatures:
    seed_coverage: int
    connects_seeds: bool
    relationship_evidence: bool
    distance: int
    support: int
    corroboration: int
    provenance_known: bool


@dataclass
class Candidate:
    """A chunk nominated by graph evidence, before ranking."""

    chunk_id: str
    document_id: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    seeds_touched: set[str] = field(default_factory=set)
    edge_ids: set[str] = field(default_factory=set)
    connects_seeds: bool = False
    distance: int | None = None
    corroboration: int = 0

    def features(self) -> RankFeatures:
        relationship = bool(self.edge_ids)
        return RankFeatures(
            seed_coverage=len(self.seeds_touched),
            connects_seeds=self.connects_seeds,
            relationship_evidence=relationship,
            distance=self.distance if relationship and self.distance is not None else 0,
            support=len(self.edge_ids),
            corroboration=self.corroboration,
            provenance_known=any(e.provenance != "unknown" for e in self.evidence),
        )


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    candidate: Candidate
    features: RankFeatures
    decided_by: str | None  # None for the first chunk: nothing ranks above it


@dataclass(frozen=True)
class GraphHit:
    rank: int
    chunk: ChunkRecord
    evidence: tuple[EvidenceItem, ...]
    features: RankFeatures
    decided_by: str | None


@dataclass
class GraphSearchOutcome:
    query: str
    hits: list[GraphHit]
    link: LinkResult
    status: Literal["ok", "no_entities", "ambiguous_only"]
    note: str | None
    revision: int
    database_revision: int | None
    direction: str
    hops: int
    relationship_types: list[str] | None
    truncated: list[str]
    ranked_total: int  # candidates that survived scope and existence checks, before max_chunks
    dropped_missing_chunks: int
    dropped_out_of_scope: int
    unknown_document_ids: list[str]
    entity_degree: dict[str, int]
    dropped_stale_evidence: int = 0  # relationship evidence SQLite no longer confirms
    timings_ms: dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------- pure graph functions


def edge_distances(traversal: Traversal) -> dict[EdgeRef, int]:
    """Distance of each walked edge: one more than its nearer endpoint's distance to a seed."""
    depth = traversal.nodes
    return {ref: min(depth[ref[0]], depth[ref[1]]) + 1 for ref, _ in traversal.edges}


def _entity_ref(graph: nx.MultiDiGraph, entity_id: str) -> EntityRef:
    attrs = graph.nodes[entity_id]
    return EntityRef(entity_id, attrs["canonical_name"], attrs["entity_type"])


def collect_relationship_evidence(
    graph: nx.MultiDiGraph,
    traversal: Traversal,
    seeds: set[str],
    permits_document,
    permits_evidence=None,
) -> dict[str, Candidate]:
    """Chunks nominated by the evidence of the walked edges.

    Source and target are copied from the stored edge, whichever way the walk went.
    """
    distances = edge_distances(traversal)
    candidates: dict[str, Candidate] = {}
    for ref, label in traversal.edges:
        source_id, target_id, rel_type = ref
        data = graph.edges[ref]
        distance = distances[ref]
        reached_from = source_id if label == "outgoing" else target_id
        for ev in data["supporting_evidence"]:
            if not permits_document(ev["document_id"]):
                continue
            if permits_evidence is not None and not permits_evidence(ev):
                continue
            cand = candidates.setdefault(
                ev["chunk_id"], Candidate(ev["chunk_id"], ev["document_id"])
            )
            cand.evidence.append(
                EvidenceItem(
                    kind="relationship",
                    document_id=ev["document_id"],
                    provenance=ev["provenance"],
                    confidence=ev["confidence"],
                    relationship_type=rel_type,
                    source=_entity_ref(graph, source_id),
                    target=_entity_ref(graph, target_id),
                    traversal=label,
                    reached_from=_entity_ref(graph, reached_from),
                    distance=distance,
                    edge_id=data["edge_id"],
                    relationship_id=ev["relationship_id"],
                    supporting_text=ev["supporting_text"],
                )
            )
            cand.edge_ids.add(data["edge_id"])
            cand.seeds_touched |= {source_id, target_id} & seeds
            cand.connects_seeds |= source_id in seeds and target_id in seeds
            cand.distance = distance if cand.distance is None else min(cand.distance, distance)
            cand.corroboration = max(cand.corroboration, len(data["source_document_ids"]))
    return candidates


def add_mention_evidence(
    graph: nx.MultiDiGraph, candidates: dict[str, Candidate], mentions: list[dict]
) -> None:
    """Add the seeds' mention chunks as mention evidence (weaker than relationship evidence)."""
    for row in mentions:
        cand = candidates.setdefault(
            row["chunk_id"], Candidate(row["chunk_id"], row["document_id"])
        )
        cand.evidence.append(
            EvidenceItem(
                kind="mention",
                document_id=row["document_id"],
                provenance=row["provenance"],
                confidence=row["confidence"],
                entity=_entity_ref(graph, row["entity_id"]),
                surface_form=row["surface_form"],
            )
        )
        cand.seeds_touched.add(row["entity_id"])


def sort_key(features: RankFeatures, chunk_id: str) -> tuple:
    """The lexicographic ranking key (see the module docstring). Compare left to right."""
    return (
        -features.seed_coverage,
        -int(features.connects_seeds),
        -int(features.relationship_evidence),
        features.distance,
        -features.support,
        -features.corroboration,
        -int(features.provenance_known),
        chunk_id,
    )


def decided_by(previous: tuple | None, key: tuple) -> str | None:
    """The first ranking key on which ``key`` differs from the key ranked just above it."""
    if previous is None:
        return None
    return next(
        (RANK_KEYS[i] for i, (a, b) in enumerate(zip(previous, key, strict=True)) if a != b),
        RANK_KEYS[-1],
    )


def rank_candidates(candidates: list[Candidate]) -> list[RankedCandidate]:
    """Order candidates by ``sort_key`` and record which key separated each from its predecessor."""
    keyed = sorted(
        ((sort_key(c.features(), c.chunk_id), c) for c in candidates), key=lambda kc: kc[0]
    )
    ranked: list[RankedCandidate] = []
    previous: tuple | None = None
    for position, (key, cand) in enumerate(keyed, start=1):
        ranked.append(RankedCandidate(position, cand, cand.features(), decided_by(previous, key)))
        previous = key
    return ranked


# -------------------------------------------------------------------------- the retriever


class GraphRetriever:
    """Graph-only retrieval over the graph service and SQLite."""

    def __init__(
        self, graph: GraphService, factory: sessionmaker[Session], settings: Settings
    ) -> None:
        self._graph = graph
        self._factory = factory
        self._settings = settings
        self.linker = EntityLinker(settings.hybrid_max_name_tokens)

    def search(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        scope: DocumentScope | None = None,
        hops: int = 1,
        direction: Direction = "both",
        relationship_types: list[str] | None = None,
        max_chunks: int = 20,
        include_mentions: bool = False,
        mentions_per_entity: int = DEFAULT_MENTIONS_PER_ENTITY,
        entity_ids: list[str] | None = None,
        link_entities: bool = True,
    ) -> GraphSearchOutcome:
        """Run a graph search.

        Raises ``GraphUnavailable`` (graph cannot be provided), ``UnknownEntities`` (explicit
        IDs not in the graph) and ``GraphSearchLimitError`` (a parameter exceeds its cap).
        """
        started = time.perf_counter()
        self._check_limits(hops, max_chunks, direction)
        query = query.strip()

        snapshot: GraphSnapshot = self._graph.snapshot()
        graph = snapshot.graph
        t0 = time.perf_counter()
        link = self.linker.link(
            snapshot, query, entity_ids=entity_ids or (), auto_link=link_entities
        )
        link_ms = (time.perf_counter() - t0) * 1000
        degree = {
            e.entity_id: graph.in_degree(e.entity_id) + graph.out_degree(e.entity_id)
            for e in link.linked
        }

        with self._factory() as session:
            if scope is None:
                scope = resolve_document_scope(session, document_ids)
            outcome = GraphSearchOutcome(
                query=query,
                hits=[],
                link=link,
                status="ok",
                note=None,
                revision=snapshot.revision,
                database_revision=self._graph.status()["database_revision"],
                direction=direction,
                hops=hops,
                relationship_types=sorted(relationship_types) if relationship_types else None,
                truncated=[],
                ranked_total=0,
                dropped_missing_chunks=0,
                dropped_out_of_scope=0,
                unknown_document_ids=list(scope.unknown),
                entity_degree=degree,
            )
            if not link.seeds:
                outcome.status = "ambiguous_only" if link.ambiguous else "no_entities"
                outcome.note = (
                    "the query names no unambiguous graph entity; pass graph.entity_ids to "
                    "resolve the ambiguity"
                    if link.ambiguous
                    else "the query names no entity that exists in the knowledge graph"
                )
                outcome.timings_ms = {
                    "link": round(link_ms, 3),
                    "total": round((time.perf_counter() - started) * 1000, 3),
                }
                return outcome

            seeds = set(link.seeds)
            t1 = time.perf_counter()
            walk = traverse(
                graph,
                link.seeds,
                hops=hops,
                direction=_TRAVERSE_DIRECTION[direction],
                rel_types=frozenset(relationship_types) if relationship_types else None,
                max_nodes=self._settings.graph_max_nodes,
                max_edges=self._settings.graph_max_edges,
            )
            outcome.truncated = list(walk.truncated)
            candidates = collect_relationship_evidence(graph, walk, seeds, scope.permits)
            candidates, outcome.dropped_stale_evidence = self._verified(
                session, graph, walk, seeds, scope, candidates
            )
            if include_mentions:
                add_mention_evidence(
                    graph,
                    candidates,
                    self._mentions(session, graph, link.seeds, scope, mentions_per_entity),
                )
            walk_ms = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            ranked = rank_candidates(list(candidates.values()))
            records = lookup_chunks(session, [r.candidate.chunk_id for r in ranked])
            rank_ms = (time.perf_counter() - t2) * 1000

        kept: list[RankedCandidate] = []
        for item in ranked:
            record = records.get(item.candidate.chunk_id)
            if record is None:  # the chunk or its document vanished after the graph was built
                outcome.dropped_missing_chunks += 1
                logger.warning(
                    "Dropping graph evidence for vanished chunk %s", item.candidate.chunk_id
                )
            elif not scope.permits(record.document_id):
                outcome.dropped_out_of_scope += 1
            else:
                kept.append(item)
        outcome.ranked_total = len(kept)
        # filtering preserves order, so renumbering keeps ranks contiguous and the order intact
        previous_key: tuple | None = None
        for position, item in enumerate(kept[:max_chunks], start=1):
            key = sort_key(item.features, item.candidate.chunk_id)
            outcome.hits.append(
                GraphHit(
                    position,
                    records[item.candidate.chunk_id],
                    tuple(item.candidate.evidence),
                    item.features,
                    decided_by(previous_key, key),
                )
            )
            previous_key = key
        if not outcome.hits:
            outcome.note = (
                "the linked entities have no relationship evidence in the requested scope"
            )
        outcome.timings_ms = {
            "link": round(link_ms, 3),
            "walk": round(walk_ms, 3),
            "rank_and_lookup": round(rank_ms, 3),
            "total": round((time.perf_counter() - started) * 1000, 3),
        }
        return outcome

    # -- helpers --------------------------------------------------------------------

    def _check_limits(self, hops: int, max_chunks: int, direction: str) -> None:
        s = self._settings
        if direction not in DIRECTIONS:
            raise GraphSearchLimitError(f"direction must be one of {', '.join(DIRECTIONS)}")
        if not 1 <= hops <= s.effective_hybrid_graph_max_hops:
            raise GraphSearchLimitError(
                f"hops must be between 1 and {s.effective_hybrid_graph_max_hops}"
            )
        if not 1 <= max_chunks <= s.hybrid_graph_max_chunks:
            raise GraphSearchLimitError(
                f"max_chunks must be between 1 and {s.hybrid_graph_max_chunks}"
            )

    @staticmethod
    def _verified(
        session: Session,
        graph: nx.MultiDiGraph,
        walk: Traversal,
        seeds: set[str],
        scope: DocumentScope,
        candidates: dict[str, Candidate],
    ) -> tuple[dict[str, Candidate], int]:
        """Drop relationship evidence that SQLite no longer confirms.

        The graph is a derived snapshot: a document re-processed or deleted after it was built
        can leave evidence rows that no longer exist. Each evidence item is checked against the
        ``relationships`` table (same row id, same chunk, same document) in the same read
        transaction as the chunk lookup. Returns the (possibly re-collected) candidates and the
        number of evidence items dropped. Costs one query, and only re-collects on a mismatch.
        """
        wanted: dict[str, tuple[str, str]] = {}
        for cand in candidates.values():
            for ev in cand.evidence:
                if ev.kind == "relationship" and ev.relationship_id:
                    wanted[ev.relationship_id] = (cand.chunk_id, ev.document_id)
        if not wanted:
            return candidates, 0
        ids = list(wanted)
        confirmed: set[str] = set()
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            for rid, chunk_id, doc_id in session.execute(
                select(
                    Relationship.id, Relationship.source_chunk_id, Relationship.source_document_id
                ).where(Relationship.id.in_(batch))
            ):
                if wanted[rid] == (chunk_id, doc_id):
                    confirmed.add(rid)
        if len(confirmed) == len(wanted):
            return candidates, 0
        rebuilt = collect_relationship_evidence(
            graph,
            walk,
            seeds,
            scope.permits,
            lambda ev: ev["relationship_id"] in confirmed,
        )
        return rebuilt, len(wanted) - len(confirmed)

    @staticmethod
    def _mentions(
        session: Session,
        graph: nx.MultiDiGraph,
        seeds: tuple[str, ...],
        scope: DocumentScope,
        per_entity: int,
    ) -> list[dict]:
        rows = session.execute(
            select(
                EntityMention.entity_id,
                EntityMention.document_id,
                EntityMention.chunk_id,
                EntityMention.surface_form,
                EntityMention.provenance,
                EntityMention.confidence,
            )
            .where(EntityMention.entity_id.in_(seeds))
            .order_by(
                EntityMention.entity_id,
                EntityMention.document_id,
                EntityMention.chunk_id,
                EntityMention.surface_form,
            )
        )
        taken: dict[str, int] = {}
        out: list[dict] = []
        for entity_id, document_id, chunk_id, surface, provenance, confidence in rows:
            if entity_id not in graph or not scope.permits(document_id):
                continue
            if taken.get(entity_id, 0) >= per_entity:
                continue
            taken[entity_id] = taken.get(entity_id, 0) + 1
            out.append(
                {
                    "entity_id": entity_id,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "surface_form": surface,
                    "provenance": provenance,
                    "confidence": confidence,
                }
            )
        return out
