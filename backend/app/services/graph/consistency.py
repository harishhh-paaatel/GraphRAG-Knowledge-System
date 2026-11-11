"""Logical comparison of graphs, used to prove "rebuild == persisted == SQLite"."""

from dataclasses import dataclass, field

import networkx as nx


def logical_form(graph: nx.MultiDiGraph) -> dict:
    """Order-independent, JSON-comparable content of a graph (ignores build timestamps)."""
    return {
        "nodes": {n: dict(d) for n, d in graph.nodes(data=True)},
        "edges": {
            d["edge_id"]: {"key": k, **d} for _, _, k, d in graph.edges(keys=True, data=True)
        },
    }


@dataclass
class ConsistencyReport:
    """Differences between an expected graph (built from SQLite) and an actual one."""

    missing_nodes: list[str] = field(default_factory=list)
    extra_nodes: list[str] = field(default_factory=list)
    changed_nodes: list[str] = field(default_factory=list)
    missing_edges: list[str] = field(default_factory=list)
    extra_edges: list[str] = field(default_factory=list)
    changed_edges: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return not any(
            (
                self.missing_nodes,
                self.extra_nodes,
                self.changed_nodes,
                self.missing_edges,
                self.extra_edges,
                self.changed_edges,
            )
        )

    def as_dict(self) -> dict:
        return {"consistent": self.consistent, **{k: v for k, v in vars(self).items()}}


def compare(expected: nx.MultiDiGraph, actual: nx.MultiDiGraph) -> ConsistencyReport:
    """Diff ``actual`` against ``expected`` (what SQLite says the graph should be)."""
    want, have = logical_form(expected), logical_form(actual)
    report = ConsistencyReport()
    for kind in ("nodes", "edges"):
        w, h = want[kind], have[kind]
        setattr(report, f"missing_{kind}", sorted(set(w) - set(h)))
        setattr(report, f"extra_{kind}", sorted(set(h) - set(w)))
        setattr(report, f"changed_{kind}", sorted(i for i in set(w) & set(h) if w[i] != h[i]))
    return report
