"""Knowledge graph endpoints (read-only; the graph is derived from SQLite)."""

from typing import Annotated, Literal

import networkx as nx
from fastapi import APIRouter, Depends, Query
from pydantic import StringConstraints
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_graph, get_session, get_settings_dep
from app.core.config import Settings
from app.core.errors import AppError
from app.models import Document, EntityMention
from app.schemas.graph import (
    EdgeList,
    EdgeOut,
    EntityDetail,
    GraphMeta,
    Limits,
    MentionOut,
    NeighborsResponse,
    NodeDepthOut,
    NodeList,
    NodeOut,
    OverviewResponse,
    PathResponse,
    PathStep,
    TraversedEdgeOut,
)
from app.services.graph import queries
from app.services.graph.queries import EdgeRef
from app.services.graph.service import GraphService, GraphSnapshot, GraphUnavailable

router = APIRouter(prefix="/graph", tags=["graph"])

SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
GraphDep = Annotated[GraphService, Depends(get_graph)]
_TYPE = r"^[A-Z][A-Z_]*$"
RelType = Annotated[str, StringConstraints(pattern=_TYPE, max_length=64)]


def _snapshot(graph: GraphService) -> GraphSnapshot:
    try:
        return graph.snapshot()
    except GraphUnavailable as exc:
        raise AppError(503, "graph_unavailable", str(exc)) from exc


def _meta(snap: GraphSnapshot) -> GraphMeta:
    return GraphMeta(revision=snap.revision, built_at=snap.built_at)


def _require(snap: GraphSnapshot, entity_id: str) -> None:
    if entity_id not in snap.graph:
        raise AppError(404, "entity_not_found", "Entity not found in the knowledge graph")


def _edge(graph: nx.MultiDiGraph, ref: EdgeRef) -> dict:
    data = queries.edge_view(graph, ref)
    data["source_name"] = graph.nodes[ref[0]]["canonical_name"]
    data["target_name"] = graph.nodes[ref[1]]["canonical_name"]
    return data


def _doc_names(
    session: Session, edges: list[dict], nodes: list[dict] | None = None
) -> dict[str, str]:
    ids: set[str] = set()
    for e in edges:
        ids.update(e["source_document_ids"])
    for n in nodes or []:
        ids.update(n["source_document_ids"])
    if not ids:
        return {}
    rows = session.execute(select(Document.id, Document.filename).where(Document.id.in_(ids)))
    return {doc_id: name for doc_id, name in rows}


@router.get("/overview", response_model=OverviewResponse)
def overview(graph: GraphDep) -> OverviewResponse:
    """Counts, type distributions, degree centrality and connected components.

    Statistics are computed once per graph revision and cached.
    """
    snap = _snapshot(graph)
    return OverviewResponse(stats=graph.stats(), graph=_meta(snap), state=graph.status())


@router.get("/nodes", response_model=NodeList)
def list_nodes(
    graph: GraphDep,
    entity_type: Annotated[str | None, Query(pattern=_TYPE)] = None,
    search: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
    document_id: Annotated[str | None, Query(max_length=64)] = None,
    sort: Literal["name", "degree"] = "name",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> NodeList:
    """Entities in the graph. `search` matches names and aliases (exact > prefix > substring)."""
    snap = _snapshot(graph)
    ids = queries.find_nodes(
        snap.graph, search=search, entity_type=entity_type, document_id=document_id, sort=sort
    )
    return NodeList(
        items=[NodeOut(**queries.node_view(snap.graph, n)) for n in ids[offset : offset + limit]],
        total=len(ids),
        limit=limit,
        offset=offset,
        graph=_meta(snap),
    )


@router.get("/edges", response_model=EdgeList)
def list_edges(
    graph: GraphDep,
    session: SessionDep,
    relationship_type: Annotated[str | None, Query(pattern=_TYPE)] = None,
    source_id: Annotated[str | None, Query(max_length=64)] = None,
    target_id: Annotated[str | None, Query(max_length=64)] = None,
    document_id: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> EdgeList:
    """Relationships with all supporting evidence and source references."""
    snap = _snapshot(graph)
    refs = queries.find_edges(
        snap.graph,
        relationship_type=relationship_type,
        source_id=source_id,
        target_id=target_id,
        document_id=document_id,
    )
    page = [_edge(snap.graph, r) for r in refs[offset : offset + limit]]
    return EdgeList(
        items=[EdgeOut(**e) for e in page],
        total=len(refs),
        limit=limit,
        offset=offset,
        documents=_doc_names(session, page),
        graph=_meta(snap),
    )


@router.get("/entities/{entity_id}", response_model=EntityDetail)
def get_entity(
    entity_id: str,
    graph: GraphDep,
    session: SessionDep,
    mention_limit: Annotated[int, Query(ge=0, le=500)] = 100,
) -> EntityDetail:
    """One entity with its incoming/outgoing relationships and the mentions behind it."""
    snap = _snapshot(graph)
    _require(snap, entity_id)
    g = snap.graph
    out_refs = [r for _, r, _ in queries.neighbors(g, entity_id, "out")]
    in_refs = [r for _, r, _ in queries.neighbors(g, entity_id, "in")]
    outgoing = [_edge(g, r) for r in out_refs]
    incoming = [_edge(g, r) for r in in_refs]
    node = queries.node_view(g, entity_id)
    mentions = session.scalars(
        select(EntityMention)
        .where(EntityMention.entity_id == entity_id)
        .order_by(EntityMention.document_id, EntityMention.chunk_id, EntityMention.surface_form)
        .limit(mention_limit)
    ).all()
    return EntityDetail(
        **node,
        outgoing=[EdgeOut(**e) for e in outgoing],
        incoming=[EdgeOut(**e) for e in incoming],
        mentions=[
            MentionOut(
                document_id=m.document_id,
                chunk_id=m.chunk_id,
                surface_form=m.surface_form,
                extracted_name=m.extracted_name,
                provenance=m.provenance,
                confidence=m.confidence,
            )
            for m in mentions
        ],
        documents=_doc_names(session, outgoing + incoming, [node]),
        graph=_meta(snap),
    )


@router.get("/entities/{entity_id}/neighbors", response_model=NeighborsResponse)
def entity_neighbors(
    entity_id: str,
    graph: GraphDep,
    session: SessionDep,
    settings: SettingsDep,
    direction: Literal["out", "in", "both"] = "out",
    hops: Annotated[int, Query(ge=1)] = 1,
    relationship_type: Annotated[list[RelType] | None, Query()] = None,
    max_nodes: Annotated[int | None, Query(ge=1)] = None,
    max_edges: Annotated[int | None, Query(ge=1)] = None,
    induced: bool = False,
) -> NeighborsResponse:
    """Bounded multi-hop neighborhood (a subgraph around the entity).

    Defaults to outgoing edges only. `direction=both` explores in both directions.
    `induced=true` adds every edge between the returned nodes, not only the walked ones.
    """
    if hops > settings.graph_max_hops:
        raise AppError(422, "limit_exceeded", f"hops must be <= {settings.graph_max_hops}")
    node_cap = min(max_nodes or 100, settings.graph_max_nodes)
    edge_cap = min(max_edges or 200, settings.graph_max_edges)
    if (max_nodes or 0) > settings.graph_max_nodes or (max_edges or 0) > settings.graph_max_edges:
        raise AppError(
            422,
            "limit_exceeded",
            f"max_nodes <= {settings.graph_max_nodes} and max_edges <= {settings.graph_max_edges}",
        )

    snap = _snapshot(graph)
    _require(snap, entity_id)
    g = snap.graph
    types = frozenset(relationship_type) if relationship_type else None
    result = queries.traverse(
        g,
        [entity_id],
        hops=hops,
        direction=direction,
        rel_types=types,
        max_nodes=node_cap,
        max_edges=edge_cap,
        induced=induced,
    )
    nodes = [
        {**queries.node_view(g, n), "depth": d}
        for n, d in sorted(result.nodes.items(), key=lambda kv: (kv[1], kv[0]))
    ]
    edges = [{**_edge(g, ref), "traversal": label} for ref, label in result.edges]
    return NeighborsResponse(
        center=entity_id,
        direction=direction,
        hops=hops,
        relationship_types=sorted(types) if types else None,
        nodes=[NodeDepthOut(**n) for n in nodes],
        edges=[TraversedEdgeOut(**e) for e in edges],
        truncated=result.truncated,
        limits=Limits(max_hops=hops, max_nodes=node_cap, max_edges=edge_cap),
        documents=_doc_names(session, edges, nodes),
        graph=_meta(snap),
    )


@router.get("/path", response_model=PathResponse)
def shortest_path(
    graph: GraphDep,
    session: SessionDep,
    settings: SettingsDep,
    source: Annotated[str, Query(min_length=1, max_length=64)],
    target: Annotated[str, Query(min_length=1, max_length=64)],
    directed: bool = True,
    max_hops: Annotated[int | None, Query(ge=1)] = None,
) -> PathResponse:
    """Fewest-hop path between two entities (follows edge direction unless `directed=false`)."""
    if (max_hops or 0) > settings.graph_max_hops:
        raise AppError(422, "limit_exceeded", f"max_hops must be <= {settings.graph_max_hops}")
    snap = _snapshot(graph)
    _require(snap, source)
    _require(snap, target)
    limit = max_hops or settings.graph_max_hops
    path = queries.shortest_path(snap.graph, source, target, directed=directed, max_hops=limit)
    if path is None:
        raise AppError(
            404,
            "no_path",
            f"No {'directed ' if directed else ''}path from source to target within {limit} hops",
        )
    g = snap.graph
    steps = [{"step": i, "traversed": label, "edge": _edge(g, ref)} for i, ref, label in path.steps]
    nodes = [queries.node_view(g, n) for n in path.nodes]
    return PathResponse(
        source=source,
        target=target,
        directed=directed,
        length=len(path.nodes) - 1,
        nodes=[NodeOut(**n) for n in nodes],
        steps=[PathStep(**s) for s in steps],
        documents=_doc_names(session, [s["edge"] for s in steps], nodes),
        graph=_meta(snap),
    )
