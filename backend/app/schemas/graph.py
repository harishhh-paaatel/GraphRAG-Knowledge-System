"""Graph API schemas. Every edge carries its supporting evidence and source references."""

from pydantic import BaseModel


class EvidenceOut(BaseModel):
    """One piece of evidence behind an edge, traceable to a document chunk."""

    relationship_id: str
    supporting_text: str
    document_id: str
    chunk_id: str
    provenance: str
    confidence: float | None  # None unless an extractor genuinely reported one


class NodeOut(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str
    description: str | None
    aliases: list[str]
    mention_count: int
    source_document_ids: list[str]
    normalized_key: str
    in_degree: int
    out_degree: int


class NodeDepthOut(NodeOut):
    depth: int


class EdgeOut(BaseModel):
    """Aggregated (source, relationship_type, target) with all of its evidence."""

    edge_id: str
    relationship_type: str
    source_entity_id: str
    target_entity_id: str
    source_name: str
    target_name: str
    supporting_evidence: list[EvidenceOut]
    source_document_ids: list[str]
    source_chunk_ids: list[str]
    provenance: list[str]
    confidence: float | None


class TraversedEdgeOut(EdgeOut):
    traversal: str  # outgoing | incoming | induced (how the walk reached it)


class GraphMeta(BaseModel):
    revision: int
    built_at: str


class NodeList(BaseModel):
    items: list[NodeOut]
    total: int
    limit: int
    offset: int
    graph: GraphMeta


class EdgeList(BaseModel):
    items: list[EdgeOut]
    total: int
    limit: int
    offset: int
    documents: dict[str, str]  # document id -> filename
    graph: GraphMeta


class MentionOut(BaseModel):
    document_id: str
    chunk_id: str
    surface_form: str
    extracted_name: str | None
    provenance: str
    confidence: float | None


class EntityDetail(NodeOut):
    outgoing: list[EdgeOut]
    incoming: list[EdgeOut]
    mentions: list[MentionOut]
    documents: dict[str, str]
    graph: GraphMeta


class Limits(BaseModel):
    max_hops: int
    max_nodes: int
    max_edges: int


class NeighborsResponse(BaseModel):
    center: str
    direction: str
    hops: int
    relationship_types: list[str] | None
    nodes: list[NodeDepthOut]
    edges: list[TraversedEdgeOut]
    truncated: list[str]  # which limits cut the result short (empty = complete)
    limits: Limits
    documents: dict[str, str]
    graph: GraphMeta


class PathStep(BaseModel):
    step: int  # 0-based index of the hop along the path
    traversed: str  # forward | reverse (reverse only when directed=false)
    edge: EdgeOut


class PathResponse(BaseModel):
    source: str
    target: str
    directed: bool
    length: int
    nodes: list[NodeOut]
    steps: list[PathStep]
    documents: dict[str, str]
    graph: GraphMeta


class OverviewResponse(BaseModel):
    stats: dict
    graph: GraphMeta
    state: dict
