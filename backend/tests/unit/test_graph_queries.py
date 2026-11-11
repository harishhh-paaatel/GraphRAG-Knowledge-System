"""Traversal, paths, subgraphs, lookup and analytics on small hand-built graphs."""

import networkx as nx
import pytest

from app.services.graph import analytics, queries
from app.services.graph.builder import make_edge_id
from app.services.graph.queries import EntityNotFound


def make_graph(nodes, edges, aliases=None):
    """nodes: {id: type}; edges: [(src, rel_type, tgt)]. Same attribute schema as the builder."""
    g = nx.MultiDiGraph()
    for nid, etype in nodes.items():
        g.add_node(
            nid,
            entity_id=nid,
            canonical_name=nid.title(),
            entity_type=etype,
            description=None,
            aliases=(aliases or {}).get(nid, []),
            mention_count=1,
            source_document_ids=["d1"],
            normalized_key=nid,
        )
    for src, rtype, tgt in edges:
        g.add_edge(
            src,
            tgt,
            key=rtype,
            edge_id=make_edge_id(src, rtype, tgt),
            relationship_type=rtype,
            source_entity_id=src,
            target_entity_id=tgt,
            supporting_evidence=[{"supporting_text": f"{src} {rtype} {tgt}"}],
            source_document_ids=["d1"],
            source_chunk_ids=["c1"],
            provenance=["test"],
            confidence=None,
        )
    return nx.freeze(g)


NODES = {
    "alice": "PERSON",
    "bob": "PERSON",
    "acme": "ORGANIZATION",
    "berlin": "LOCATION",
    "group": "ORGANIZATION",
    "dave": "PERSON",
}
EDGES = [
    ("alice", "WORKS_AT", "acme"),
    ("bob", "WORKS_AT", "acme"),
    ("acme", "LOCATED_IN", "berlin"),
    ("acme", "PART_OF", "group"),
    ("group", "RELATED_TO", "acme"),  # makes acme <-> group a cycle
]


@pytest.fixture
def g():
    return make_graph(NODES, EDGES, aliases={"acme": ["Acme Corp"]})


# ---- direct neighbors --------------------------------------------------------------------


def test_outgoing_and_incoming_are_distinct(g):
    out = queries.neighbors(g, "acme", "out")
    inc = queries.neighbors(g, "acme", "in")
    assert sorted((n, ref[2]) for n, ref, _ in out) == [
        ("berlin", "LOCATED_IN"),
        ("group", "PART_OF"),
    ]
    assert sorted((n, ref[2]) for n, ref, _ in inc) == [
        ("alice", "WORKS_AT"),
        ("bob", "WORKS_AT"),
        ("group", "RELATED_TO"),
    ]
    assert {label for *_, label in out} == {"outgoing"} and {label for *_, label in inc} == {
        "incoming"
    }


def test_both_directions_and_relationship_filter(g):
    both = queries.neighbors(g, "acme", "both")
    assert len(both) == 5
    only = queries.neighbors(g, "acme", "both", frozenset({"WORKS_AT"}))
    assert sorted(n for n, *_ in only) == ["alice", "bob"]


def test_returned_edges_always_keep_stored_direction(g):
    for _, (u, v, k), label in queries.neighbors(g, "acme", "in"):
        assert v == "acme" and g.edges[u, v, k]["source_entity_id"] == u
        assert g.edges[u, v, k]["target_entity_id"] == "acme"
        assert label == "incoming"


def test_unknown_entity_raises(g):
    with pytest.raises(EntityNotFound):
        queries.neighbors(g, "nobody")
    with pytest.raises(EntityNotFound):
        queries.traverse(g, ["nobody"], hops=1, max_nodes=10, max_edges=10)


# ---- multi-hop traversal -----------------------------------------------------------------


def trav(g, seeds, **kw):
    kw.setdefault("hops", 2)
    kw.setdefault("max_nodes", 100)
    kw.setdefault("max_edges", 100)
    return queries.traverse(g, seeds, **kw)


def test_default_outgoing_walk_does_not_go_backwards(g):
    r = trav(g, ["alice"], hops=3)
    assert r.nodes == {"alice": 0, "acme": 1, "berlin": 2, "group": 2}
    assert "bob" not in r.nodes  # reachable only against edge direction
    assert not r.truncated


def test_hop_limit_is_respected(g):
    assert trav(g, ["alice"], hops=1).nodes == {"alice": 0, "acme": 1}


def test_both_directions_reaches_against_the_grain(g):
    r = trav(g, ["alice"], hops=2, direction="both")
    assert r.nodes["bob"] == 2
    labels = {ref: label for ref, label in r.edges}
    assert labels[("bob", "acme", "WORKS_AT")] == "incoming"  # reached against its direction
    assert labels[("alice", "acme", "WORKS_AT")] == "outgoing"


def test_incoming_walk(g):
    r = trav(g, ["berlin"], hops=2, direction="in")
    assert r.nodes == {"berlin": 0, "acme": 1, "alice": 2, "bob": 2, "group": 2}


def test_relationship_type_filter_applies_to_walk(g):
    r = trav(g, ["acme"], hops=2, rel_types=frozenset({"LOCATED_IN"}))
    assert r.nodes == {"acme": 0, "berlin": 1}


def test_cycles_do_not_loop(g):
    r = trav(g, ["acme"], hops=4)
    assert set(r.nodes) == {"acme", "berlin", "group"}
    assert len(r.edges) == len(set(ref for ref, _ in r.edges))


def test_multiple_seeds(g):
    r = trav(g, ["alice", "bob"], hops=1)
    assert r.nodes == {"alice": 0, "bob": 0, "acme": 1}
    assert len(r.edges) == 2


def test_max_nodes_truncates_closest_first_and_reports_it(g):
    r = trav(g, ["acme"], hops=2, direction="both", max_nodes=3)
    assert len(r.nodes) == 3 and r.nodes["acme"] == 0
    assert "max_nodes" in r.truncated
    for (u, v, _), _ in r.edges:  # no edge to an excluded node
        assert u in r.nodes and v in r.nodes


def test_max_edges_truncates_and_reports_it(g):
    r = trav(g, ["acme"], hops=2, direction="both", max_edges=2)
    assert len(r.edges) == 2 and "max_edges" in r.truncated


def test_induced_subgraph_adds_edges_between_returned_nodes(g):
    plain = trav(g, ["alice"], hops=2, direction="out")
    assert ("group", "acme", "RELATED_TO") not in [ref for ref, _ in plain.edges]
    induced = trav(g, ["alice"], hops=2, direction="out", induced=True)
    labels = dict(induced.edges)
    assert labels[("group", "acme", "RELATED_TO")] == "induced"
    assert set(induced.nodes) == set(plain.nodes)
    for (u, v, _), _ in induced.edges:
        assert u in induced.nodes and v in induced.nodes


def test_traversal_is_deterministic(g):
    assert trav(g, ["acme"], direction="both").edges == trav(g, ["acme"], direction="both").edges


# ---- shortest paths ----------------------------------------------------------------------


def test_directed_shortest_path_follows_direction(g):
    p = queries.shortest_path(g, "alice", "berlin", max_hops=5)
    assert p.nodes == ["alice", "acme", "berlin"]
    assert [(ref, label) for _, ref, label in p.steps] == [
        (("alice", "acme", "WORKS_AT"), "forward"),
        (("acme", "berlin", "LOCATED_IN"), "forward"),
    ]


def test_directed_path_against_the_edges_does_not_exist(g):
    assert queries.shortest_path(g, "berlin", "alice", max_hops=5) is None
    assert queries.shortest_path(g, "alice", "bob", max_hops=5) is None
    assert queries.shortest_path(g, "alice", "dave", max_hops=5) is None


def test_undirected_path_labels_reverse_steps_and_keeps_true_direction(g):
    p = queries.shortest_path(g, "alice", "bob", directed=False, max_hops=5)
    assert p.nodes == ["alice", "acme", "bob"]
    (_, e1, l1), (_, e2, l2) = p.steps
    assert (e1, l1) == (("alice", "acme", "WORKS_AT"), "forward")
    assert (e2, l2) == (("bob", "acme", "WORKS_AT"), "reverse")  # stored direction bob -> acme


def test_path_respects_max_hops(g):
    assert queries.shortest_path(g, "alice", "berlin", max_hops=2) is not None
    assert queries.shortest_path(g, "alice", "berlin", max_hops=1) is None


def test_path_to_self_has_zero_hops(g):
    p = queries.shortest_path(g, "alice", "alice", max_hops=3)
    assert p.nodes == ["alice"] and p.steps == []


def test_parallel_relationship_types_are_all_reported_on_the_step():
    g2 = make_graph(
        {"a": "PERSON", "b": "ORGANIZATION"}, [("a", "WORKS_AT", "b"), ("a", "RELATED_TO", "b")]
    )
    p = queries.shortest_path(g2, "a", "b", max_hops=2)
    assert sorted(ref[2] for _, ref, _ in p.steps) == ["RELATED_TO", "WORKS_AT"]
    assert {i for i, *_ in p.steps} == {0}  # both on the same hop


def test_path_unknown_nodes(g):
    with pytest.raises(EntityNotFound):
        queries.shortest_path(g, "alice", "nobody", max_hops=3)


# ---- lookup ------------------------------------------------------------------------------


def test_search_ranks_exact_then_prefix_then_substring_and_uses_aliases():
    g2 = make_graph(
        {"openai": "ORGANIZATION", "open_labs": "ORGANIZATION", "reopen": "CONCEPT"},
        [],
        aliases={"openai": ["Open AI"]},
    )
    assert queries.find_nodes(g2, search="openai")[0] == "openai"
    ranked = queries.find_nodes(g2, search="open")
    assert set(ranked) == {"openai", "open_labs", "reopen"}
    assert ranked[-1] == "reopen"  # substring match ranks last
    assert queries.find_nodes(g2, search="open ai") == ["openai"]  # alias match


def test_filters_and_sorting(g):
    assert sorted(queries.find_nodes(g, entity_type="PERSON")) == ["alice", "bob", "dave"]
    assert queries.find_nodes(g, document_id="d2") == []
    assert queries.find_nodes(g, sort="degree")[0] == "acme"
    assert queries.find_nodes(g, search="zzz") == []
    assert queries.find_nodes(g, search="acme corp") == ["acme"]


def test_find_edges_filters(g):
    assert len(queries.find_edges(g)) == 5
    assert len(queries.find_edges(g, relationship_type="WORKS_AT")) == 2
    assert queries.find_edges(g, source_id="acme", target_id="berlin") == [
        ("acme", "berlin", "LOCATED_IN")
    ]
    assert queries.find_edges(g, document_id="nope") == []


# ---- analytics ---------------------------------------------------------------------------


def test_stats_on_known_graph(g):
    s = analytics.compute_stats(g, top_k=3)
    assert (s["node_count"], s["edge_count"]) == (6, 5)
    assert s["entity_type_distribution"] == {"LOCATION": 1, "ORGANIZATION": 2, "PERSON": 3}
    assert s["relationship_type_distribution"] == {
        "LOCATED_IN": 1,
        "PART_OF": 1,
        "RELATED_TO": 1,
        "WORKS_AT": 2,
    }
    weak = s["weakly_connected_components"]
    assert (weak["count"], weak["largest_size"], weak["sizes"], weak["isolated_nodes"]) == (
        2,
        5,
        [5, 1],
        1,
    )
    strong = s["strongly_connected_components"]
    assert strong["count"] == 5  # {acme, group} + 4 singletons
    assert (strong["largest_size"], strong["non_trivial_count"]) == (2, 1)
    top = s["top_entities_by_degree_centrality"]
    assert [t["entity_id"] for t in top][0] == "acme" and len(top) == 3
    # acme touches 5 edges; n-1 = 5  ->  centrality 1.0
    assert top[0]["degree_centrality"] == 1.0
    assert (top[0]["in_degree"], top[0]["out_degree"]) == (3, 2)
    assert s["density"] == pytest.approx(5 / (6 * 5))


def test_weak_and_strong_components_differ_on_a_directed_chain():
    chain = make_graph({"a": "X", "b": "X", "c": "X"}, [("a", "R", "b"), ("b", "R", "c")])
    s = analytics.compute_stats(chain)
    assert s["weakly_connected_components"]["count"] == 1
    assert s["strongly_connected_components"]["count"] == 3  # no cycles


def test_stats_on_empty_and_single_node_graphs():
    empty = analytics.compute_stats(make_graph({}, []))
    assert empty["node_count"] == 0 and empty["weakly_connected_components"]["count"] == 0
    assert empty["top_entities_by_degree_centrality"] == []
    one = analytics.compute_stats(make_graph({"a": "X"}, []))
    assert one["top_entities_by_degree_centrality"][0]["degree_centrality"] == 0.0
    assert one["density"] == 0.0
