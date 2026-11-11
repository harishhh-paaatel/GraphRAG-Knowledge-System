"""Graph statistics. Pure functions; callers cache the result per graph revision.

Counting convention: an "edge" is one aggregated *(source, relationship type, target)*;
a pair of entities joined by two different relationship types has two edges. Degrees count
those edges (so they are multigraph degrees), and ``degree_centrality`` is
``degree / (n - 1)`` over the same counts, which can exceed 1 for densely multi-linked nodes.
"""

from collections import Counter

import networkx as nx

TOP_COMPONENTS = 10


def _centrality(graph: nx.MultiDiGraph) -> dict[str, float]:
    n = graph.number_of_nodes()
    if n <= 1:
        return dict.fromkeys(graph, 0.0)
    return nx.degree_centrality(graph)


def compute_stats(graph: nx.MultiDiGraph, top_k: int = 10) -> dict:
    """Counts, distributions, connected components and the most central entities."""
    n = graph.number_of_nodes()
    centrality = _centrality(graph)

    weak = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
    strong_components = list(nx.strongly_connected_components(graph))
    strong = sorted((len(c) for c in strong_components), reverse=True)

    top = sorted(graph.nodes, key=lambda x: (-centrality[x], graph.nodes[x]["canonical_name"], x))
    return {
        "node_count": n,
        "edge_count": graph.number_of_edges(),
        "entity_type_distribution": dict(
            sorted(Counter(d["entity_type"] for _, d in graph.nodes(data=True)).items())
        ),
        "relationship_type_distribution": dict(
            sorted(Counter(k for _, _, k in graph.edges(keys=True)).items())
        ),
        "density": nx.density(graph) if n > 1 else 0.0,
        "weakly_connected_components": {
            "count": len(weak),
            "largest_size": weak[0] if weak else 0,
            "sizes": weak[:TOP_COMPONENTS],
            "isolated_nodes": sum(1 for s in weak if s == 1),
        },
        "strongly_connected_components": {
            "count": len(strong),
            "largest_size": strong[0] if strong else 0,
            "sizes": strong[:TOP_COMPONENTS],
            "non_trivial_count": sum(1 for s in strong if s > 1),
        },
        "top_entities_by_degree_centrality": [
            {
                "entity_id": node,
                "canonical_name": graph.nodes[node]["canonical_name"],
                "entity_type": graph.nodes[node]["entity_type"],
                "in_degree": graph.in_degree(node),
                "out_degree": graph.out_degree(node),
                "degree_centrality": round(centrality[node], 6),
            }
            for node in top[:top_k]
        ],
    }
