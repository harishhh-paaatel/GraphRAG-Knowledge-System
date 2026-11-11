"""Read-only graph queries. All functions treat the graph as immutable and enforce limits.

Direction semantics: ``out`` follows edges source->target, ``in`` follows them
target->source, ``both`` does either. Returned edges always keep their stored direction
(``source_entity_id`` -> ``target_entity_id``); the ``traversal`` field only says how the
walk reached them. Default everywhere is ``out`` so a walk never fans out through every
relationship in both directions unless the caller asks for it.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx

Direction = Literal["out", "in", "both"]
EdgeRef = tuple[str, str, str]  # (source, target, relationship_type)


class EntityNotFound(KeyError):
    """The requested entity is not in the graph."""


@dataclass
class Traversal:
    """Result of a bounded breadth-first walk."""

    nodes: dict[str, int] = field(default_factory=dict)  # node id -> hop depth
    edges: list[tuple[EdgeRef, str]] = field(default_factory=list)  # (edge, traversal label)
    truncated: list[str] = field(default_factory=list)  # which limits were hit


def require_node(graph: nx.MultiDiGraph, node_id: str) -> None:
    if node_id not in graph:
        raise EntityNotFound(node_id)


def node_view(graph: nx.MultiDiGraph, node_id: str) -> dict:
    """Node attributes plus in/out degree (aggregated edges, i.e. distinct source-type-target)."""
    return {
        **graph.nodes[node_id],
        "in_degree": graph.in_degree(node_id),
        "out_degree": graph.out_degree(node_id),
    }


def edge_view(graph: nx.MultiDiGraph, ref: EdgeRef) -> dict:
    src, tgt, key = ref
    return dict(graph.edges[src, tgt, key])


def _incident(
    graph: nx.MultiDiGraph, node: str, direction: Direction, rel_types: frozenset[str] | None
) -> list[tuple[str, EdgeRef, str]]:
    """Edges at ``node`` as (neighbor, edge ref, label), deterministically ordered."""
    items: list[tuple[str, EdgeRef, str]] = []
    if direction in ("out", "both"):
        items += [(v, (u, v, k), "outgoing") for u, v, k in graph.out_edges(node, keys=True)]
    if direction in ("in", "both"):
        items += [(u, (u, v, k), "incoming") for u, v, k in graph.in_edges(node, keys=True)]
    if rel_types is not None:
        items = [i for i in items if i[1][2] in rel_types]
    return sorted(items, key=lambda i: (i[1][2], i[0], i[2]))


def neighbors(
    graph: nx.MultiDiGraph,
    node_id: str,
    direction: Direction = "out",
    rel_types: frozenset[str] | None = None,
) -> list[tuple[str, EdgeRef, str]]:
    """Direct neighbors of one entity (1 hop, no limits needed: bounded by its degree)."""
    require_node(graph, node_id)
    return _incident(graph, node_id, direction, rel_types)


def traverse(
    graph: nx.MultiDiGraph,
    seeds: Iterable[str],
    *,
    hops: int,
    direction: Direction = "out",
    rel_types: frozenset[str] | None = None,
    max_nodes: int,
    max_edges: int,
    induced: bool = False,
) -> Traversal:
    """Breadth-first walk from ``seeds`` up to ``hops`` hops, never exceeding the limits.

    Nodes are admitted in BFS order, so truncation keeps the closest ones. An edge is only
    returned if both endpoints are admitted. With ``induced=True`` every edge between the
    admitted nodes is added (a true subgraph), otherwise only the edges that were walked.
    """
    seed_list = sorted(set(seeds))
    for s in seed_list:
        require_node(graph, s)
    result = Traversal()
    for s in seed_list[:max_nodes]:
        result.nodes[s] = 0
    if len(seed_list) > max_nodes:
        result.truncated.append("max_nodes")

    seen_edges: set[EdgeRef] = set()
    frontier = list(result.nodes)
    for depth in range(1, hops + 1):
        next_frontier: list[str] = []
        for node in frontier:
            for other, ref, label in _incident(graph, node, direction, rel_types):
                if ref in seen_edges:
                    continue
                if other not in result.nodes:
                    if len(result.nodes) >= max_nodes:
                        _flag(result, "max_nodes")
                        continue
                    result.nodes[other] = depth
                    next_frontier.append(other)
                if len(result.edges) >= max_edges:
                    _flag(result, "max_edges")
                    continue
                seen_edges.add(ref)
                result.edges.append((ref, label))
        frontier = next_frontier
        if not frontier:
            break

    if induced:
        members = set(result.nodes)
        for u in sorted(members):
            for _, v, k in sorted(graph.out_edges(u, keys=True)):
                ref = (u, v, k)
                if v not in members or ref in seen_edges:
                    continue
                if rel_types is not None and k not in rel_types:
                    continue
                if len(result.edges) >= max_edges:
                    _flag(result, "max_edges")
                    break
                seen_edges.add(ref)
                result.edges.append((ref, "induced"))
    return result


def _flag(result: Traversal, reason: str) -> None:
    if reason not in result.truncated:
        result.truncated.append(reason)


@dataclass
class PathResult:
    nodes: list[str]
    steps: list[tuple[int, EdgeRef, str]]  # (step index, edge, "forward" | "reverse")


def shortest_path(
    graph: nx.MultiDiGraph, source: str, target: str, *, directed: bool = True, max_hops: int
) -> PathResult | None:
    """Fewest-hop path, or None if unreachable or longer than ``max_hops``.

    ``directed=True`` only follows edges forward. With ``directed=False`` edges may be walked
    against their direction; such steps are labelled ``reverse`` and keep their true direction.
    """
    require_node(graph, source)
    require_node(graph, target)
    view = graph if directed else graph.to_undirected(as_view=True)
    try:
        nodes = nx.shortest_path(view, source, target)
    except nx.NetworkXNoPath:
        return None
    if len(nodes) - 1 > max_hops:
        return None
    steps: list[tuple[int, EdgeRef, str]] = []
    for i, (a, b) in enumerate(zip(nodes, nodes[1:], strict=False)):
        for key in sorted(graph.get_edge_data(a, b) or {}):
            steps.append((i, (a, b, key), "forward"))
        if not directed:
            for key in sorted(graph.get_edge_data(b, a) or {}):
                steps.append((i, (b, a, key), "reverse"))
    return PathResult(nodes, steps)


def find_nodes(
    graph: nx.MultiDiGraph,
    *,
    search: str | None = None,
    entity_type: str | None = None,
    document_id: str | None = None,
    sort: Literal["name", "degree"] = "name",
) -> list[str]:
    """Entity lookup. ``search`` matches canonical names and aliases (case-insensitive);
    exact matches rank before prefix matches before substring matches."""
    q = search.casefold().strip() if search else None
    scored: list[tuple[int, str, str]] = []
    for node_id, attrs in graph.nodes(data=True):
        if entity_type and attrs["entity_type"] != entity_type:
            continue
        if document_id and document_id not in attrs["source_document_ids"]:
            continue
        rank = 0
        if q:
            names = [attrs["canonical_name"], *attrs["aliases"]]
            folded = [n.casefold() for n in names]
            if q in folded:
                rank = 0
            elif any(n.startswith(q) for n in folded):
                rank = 1
            elif any(q in n for n in folded):
                rank = 2
            else:
                continue
        scored.append((rank, attrs["canonical_name"].casefold(), node_id))
    if sort == "degree":
        return [
            n for _, _, n in sorted(scored, key=lambda t: (t[0], -graph.degree(t[2]), t[1], t[2]))
        ]
    return [n for _, _, n in sorted(scored)]


def find_edges(
    graph: nx.MultiDiGraph,
    *,
    relationship_type: str | None = None,
    source_id: str | None = None,
    target_id: str | None = None,
    document_id: str | None = None,
) -> list[EdgeRef]:
    """Edges matching all given filters, in a stable order."""
    out: list[EdgeRef] = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        if relationship_type and k != relationship_type:
            continue
        if source_id and u != source_id:
            continue
        if target_id and v != target_id:
            continue
        if document_id and document_id not in data["source_document_ids"]:
            continue
        out.append((u, v, k))
    return sorted(out, key=lambda r: (graph.nodes[r[0]]["canonical_name"].casefold(), r[2], r[1]))
