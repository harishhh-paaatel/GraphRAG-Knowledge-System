"""Graph ranking, evidence collection and directed traversal on hand-built graphs.

No database, no embedding model, no vector index: the whole graph path is exercised without
them, which is also the point (the ranking must not depend on vector similarity).
"""

import itertools

import pytest

from app.services.graph.queries import traverse
from app.services.retrieval.graph_search import (
    RANK_KEYS,
    Candidate,
    EvidenceItem,
    add_mention_evidence,
    collect_relationship_evidence,
    decided_by,
    edge_distances,
    rank_candidates,
    sort_key,
)
from tests.graph_search_helpers import evidence, make_snapshot, node

# --------------------------------------------------------------------- ranking features


def cand(
    chunk_id,
    *,
    coverage=1,
    connects=False,
    edges=1,
    distance=1,
    corroboration=1,
    provenance="rule_based",
    mention_only=False,
):
    c = Candidate(chunk_id, "doc")
    c.seeds_touched = {f"s{i}" for i in range(coverage)}
    c.connects_seeds = connects
    c.corroboration = corroboration
    if mention_only:
        c.evidence.append(EvidenceItem("mention", "doc", provenance, None))
        return c
    c.edge_ids = {f"e{i}" for i in range(edges)}
    c.distance = distance
    c.evidence.append(EvidenceItem("relationship", "doc", provenance, None))
    return c


def order(*candidates):
    return [r.candidate.chunk_id for r in rank_candidates(list(candidates))]


def test_the_rank_keys_are_the_documented_lexicographic_order():
    assert RANK_KEYS == (
        "seed_coverage",
        "connects_seeds",
        "relationship_evidence",
        "distance",
        "support",
        "corroboration",
        "provenance_known",
        "chunk_id",
    )


# each case: (name of the deciding key, the better candidate, the worse candidate); both are
# otherwise identical, so that key alone must decide, and `decided_by` must name it
DECIDING = [
    ("seed_coverage", cand("z", coverage=2), cand("a", coverage=1)),
    ("connects_seeds", cand("z", connects=True), cand("a", connects=False)),
    ("relationship_evidence", cand("z"), cand("a", mention_only=True)),
    ("distance", cand("z", distance=1), cand("a", distance=2)),
    ("support", cand("z", edges=3), cand("a", edges=2)),
    ("corroboration", cand("z", corroboration=3), cand("a", corroboration=2)),
    ("provenance_known", cand("z"), cand("a", provenance="unknown")),
    ("chunk_id", cand("a"), cand("b")),
]


@pytest.mark.parametrize("key, better, worse", DECIDING, ids=[d[0] for d in DECIDING])
def test_each_key_decides_alone_when_all_earlier_keys_tie(key, better, worse):
    ranked = rank_candidates([worse, better])
    assert [r.candidate.chunk_id for r in ranked] == [better.chunk_id, worse.chunk_id]
    assert ranked[0].decided_by is None
    assert ranked[1].decided_by == key


@pytest.mark.parametrize("key, better, worse", DECIDING, ids=[d[0] for d in DECIDING])
def test_the_input_order_never_changes_the_result(key, better, worse):
    extra = cand("m", coverage=1)
    results = {
        tuple(r.candidate.chunk_id for r in rank_candidates(list(p)))
        for p in itertools.permutations([better, worse, extra])
    }
    assert len(results) == 1


def test_an_earlier_key_always_beats_a_later_one():
    close_but_thin = cand("a", coverage=1, distance=1, edges=1)
    far_but_broad = cand("b", coverage=2, distance=2, edges=1)
    assert order(close_but_thin, far_but_broad) == ["b", "a"]  # coverage outranks distance
    many_edges_far = cand("c", edges=9, distance=3)
    one_edge_near = cand("d", edges=1, distance=1)
    assert order(many_edges_far, one_edge_near) == ["d", "c"]  # distance outranks support
    corroborated = cand("e", corroboration=5, provenance="unknown")
    known = cand("f", corroboration=1)
    assert order(corroborated, known) == ["e", "f"]  # corroboration outranks provenance


def test_identical_features_are_ordered_by_chunk_id_and_the_ranking_says_so():
    ranked = rank_candidates([cand("c"), cand("a"), cand("b")])
    assert [r.candidate.chunk_id for r in ranked] == ["a", "b", "c"]
    assert [r.decided_by for r in ranked] == [None, "chunk_id", "chunk_id"]
    assert [r.rank for r in ranked] == [1, 2, 3]


def test_a_mention_only_chunk_ranks_after_relationship_evidence_with_the_same_coverage():
    assert order(cand("a", mention_only=True), cand("z")) == ["z", "a"]
    assert order(cand("a", mention_only=True, coverage=2), cand("z", coverage=1)) == ["a", "z"]


def test_the_features_report_exactly_what_was_ranked():
    c = cand("x", coverage=2, connects=True, edges=3, distance=2, corroboration=4)
    f = c.features()
    assert (f.seed_coverage, f.connects_seeds, f.relationship_evidence) == (2, True, True)
    assert (f.distance, f.support, f.corroboration, f.provenance_known) == (2, 3, 4, True)
    m = cand("y", mention_only=True).features()
    assert (m.relationship_evidence, m.distance, m.support, m.corroboration) == (False, 0, 0, 1)


def test_provenance_is_known_when_any_evidence_has_a_real_provenance():
    c = cand("x", provenance="unknown")
    assert not c.features().provenance_known
    c.evidence.append(EvidenceItem("relationship", "doc", "llm", None))
    assert c.features().provenance_known


def test_the_sort_key_is_made_of_integers_and_the_chunk_id_only():
    key = sort_key(cand("x").features(), "x")
    assert all(isinstance(part, int) for part in key[:-1]) and key[-1] == "x"
    assert not any(isinstance(part, float) for part in key)


def test_nothing_in_the_ranking_depends_on_similarity_or_vectors():
    """No identifier or import of the module mentions vectors, embeddings or similarity."""
    import ast

    import app.services.retrieval.graph_search as module

    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    names: set[str] = set()
    for item in ast.walk(tree):
        if isinstance(item, ast.Name):
            names.add(item.id)
        elif isinstance(item, ast.Attribute):
            names.add(item.attr)
        elif isinstance(item, ast.arg):
            names.add(item.arg)
        elif isinstance(item, ast.Import):
            names |= {a.name for a in item.names}
        elif isinstance(item, ast.ImportFrom):
            names |= {item.module or ""} | {a.name for a in item.names}
    for forbidden in ("faiss", "numpy", "embed", "similarity", "cosine", "vector"):
        assert not [n for n in names if forbidden in n.lower()], forbidden


def test_decided_by_names_the_first_differing_key():
    base = sort_key(cand("a").features(), "a")
    assert decided_by(None, base) is None
    assert decided_by(base, sort_key(cand("a", edges=2).features(), "a")) == "support"
    assert decided_by(base, base) == "chunk_id"


def test_ranking_nothing_gives_nothing():
    assert rank_candidates([]) == []


# ------------------------------------------------------- directed traversal and evidence

NODES = [
    node("alice", "Alice", "PERSON"),
    node("bob", "Bob", "PERSON"),
    node("acme", "Acme Corp"),
    node("berlin", "Berlin", "LOCATION"),
    node("germany", "Germany", "LOCATION"),
]
EDGES = [
    ("alice", "acme", "WORKS_AT", [evidence("c-alice", "Alice works at Acme Corp", "d1")]),
    ("bob", "acme", "WORKS_AT", [evidence("c-bob", "Bob works at Acme Corp", "d2")]),
    (
        "acme",
        "berlin",
        "LOCATED_IN",
        [
            evidence("c-alice", "Acme Corp is in Berlin", "d1"),
            evidence("c-bob", "Acme Corp is in Berlin", "d2"),
        ],
    ),
    ("berlin", "germany", "PART_OF", [evidence("c-geo", "Berlin is in Germany", "d3")]),
]


@pytest.fixture(scope="module")
def graph():
    return make_snapshot(NODES, EDGES).graph


def walk(graph, seeds, direction="both", hops=1, rel_types=None):
    mapping = {"outgoing": "out", "incoming": "in", "both": "both"}
    return traverse(
        graph,
        sorted(seeds),
        hops=hops,
        direction=mapping[direction],
        rel_types=rel_types,
        max_nodes=100,
        max_edges=100,
    )


def collect(graph, seeds, direction="both", hops=1, permits=lambda d: True, rel_types=None):
    t = walk(graph, seeds, direction, hops, rel_types)
    return collect_relationship_evidence(graph, t, set(seeds), permits), t


def edges_of(cands):
    return sorted(
        (e.relationship_type, e.source.entity_id, e.target.entity_id)
        for c in cands.values()
        for e in c.evidence
    )


def test_outgoing_follows_edges_whose_source_is_the_entity(graph):
    cands, _ = collect(graph, {"acme"}, "outgoing")
    assert edges_of(cands) == [("LOCATED_IN", "acme", "berlin")] * 2
    assert {e.traversal for c in cands.values() for e in c.evidence} == {"outgoing"}


def test_incoming_follows_edges_whose_target_is_the_entity(graph):
    cands, _ = collect(graph, {"acme"}, "incoming")
    assert edges_of(cands) == [("WORKS_AT", "alice", "acme"), ("WORKS_AT", "bob", "acme")]
    assert {e.traversal for c in cands.values() for e in c.evidence} == {"incoming"}


def test_both_follows_either_direction(graph):
    cands, _ = collect(graph, {"acme"}, "both")
    assert len(edges_of(cands)) == 4
    assert {e.traversal for c in cands.values() for e in c.evidence} == {"outgoing", "incoming"}


@pytest.mark.parametrize("direction", ["outgoing", "incoming", "both"])
def test_the_stored_direction_is_returned_unchanged_in_every_mode(graph, direction):
    for seed in ("acme", "berlin", "alice"):
        cands, _ = collect(graph, {seed}, direction)
        for cand_ in cands.values():
            for ev in cand_.evidence:
                stored = graph.edges[ev.source.entity_id, ev.target.entity_id, ev.relationship_type]
                assert stored["source_entity_id"] == ev.source.entity_id
                assert stored["target_entity_id"] == ev.target.entity_id
                assert ev.edge_id == stored["edge_id"]


def test_an_entity_with_only_incoming_edges_has_nothing_outgoing(graph):
    cands, _ = collect(graph, {"germany"}, "outgoing")
    assert cands == {}
    cands, _ = collect(graph, {"germany"}, "incoming")
    assert edges_of(cands) == [("PART_OF", "berlin", "germany")]


def test_both_does_not_create_reverse_relationships(graph):
    cands, _ = collect(graph, {"berlin"}, "both")
    pairs = {(s, t) for _, s, t in edges_of(cands)}
    assert pairs == {("acme", "berlin"), ("berlin", "germany")}
    assert ("berlin", "acme") not in pairs and ("germany", "berlin") not in pairs


def test_reached_from_names_the_entity_the_edge_was_expanded_from(graph):
    cands, _ = collect(graph, {"acme"}, "incoming")
    for c in cands.values():
        for ev in c.evidence:
            assert ev.reached_from.entity_id == "acme" and ev.target.entity_id == "acme"
    cands, _ = collect(graph, {"acme"}, "outgoing")
    for c in cands.values():
        for ev in c.evidence:
            assert ev.reached_from.entity_id == "acme" and ev.source.entity_id == "acme"


def test_edge_distance_is_one_more_than_the_nearer_endpoint(graph):
    t = walk(graph, {"berlin"}, "both", hops=2)
    distances = edge_distances(t)
    assert distances[("acme", "berlin", "LOCATED_IN")] == 1
    assert distances[("berlin", "germany", "PART_OF")] == 1
    assert distances[("alice", "acme", "WORKS_AT")] == 2
    assert distances[("bob", "acme", "WORKS_AT")] == 2


def test_hops_limit_the_neighbourhood_and_one_hop_stays_local(graph):
    one, _ = collect(graph, {"berlin"}, "both", hops=1)
    assert {e.relationship_type for c in one.values() for e in c.evidence} == {
        "LOCATED_IN",
        "PART_OF",
    }
    two, _ = collect(graph, {"berlin"}, "both", hops=2)
    assert "WORKS_AT" in {e.relationship_type for c in two.values() for e in c.evidence}


def test_an_unrelated_neighbourhood_is_never_expanded(graph):
    cands, t = collect(graph, {"alice"}, "both", hops=1)
    assert set(t.nodes) == {"alice", "acme"}
    assert edges_of(cands) == [("WORKS_AT", "alice", "acme")]  # not Bob, not Berlin


def test_relationship_types_restrict_the_walk(graph):
    cands, _ = collect(graph, {"acme"}, "both", rel_types=frozenset({"LOCATED_IN"}))
    assert {e.relationship_type for c in cands.values() for e in c.evidence} == {"LOCATED_IN"}


def test_corroboration_counts_distinct_documents_not_evidence_rows():
    snapshot = make_snapshot(
        [node("a", "A"), node("b", "B")],
        [
            (
                "a",
                "b",
                "RELATED_TO",
                [
                    evidence("c1", "A b", "d1", n=0),
                    evidence("c2", "A b", "d1", n=1),
                    evidence("c3", "A b", "d1", n=2),
                    evidence("c4", "A b", "d2", n=3),
                ],
            )
        ],
    )
    cands, _ = collect(snapshot.graph, {"a"}, "both")
    assert len(cands) == 4  # one candidate per evidence chunk
    assert {c.features().corroboration for c in cands.values()} == {2}  # d1 and d2


def test_a_chunk_supporting_several_walked_edges_is_one_candidate_with_all_of_them(graph):
    cands, _ = collect(graph, {"acme"}, "both")
    alice = cands["c-alice"]
    assert alice.edge_ids and len(alice.edge_ids) == 2  # WORKS_AT and LOCATED_IN
    assert len(alice.evidence) == 2 and alice.features().support == 2
    assert alice.features().corroboration == 2  # LOCATED_IN is supported by two documents
    assert set(cands) == {"c-alice", "c-bob"}


def test_two_linked_entities_make_the_connecting_edge_rank_first(graph):
    cands, _ = collect(graph, {"alice", "acme"}, "both")
    ranked = rank_candidates(list(cands.values()))
    top = ranked[0]
    assert top.candidate.chunk_id == "c-alice"
    assert top.features.connects_seeds and top.features.seed_coverage == 2


def test_the_document_filter_removes_evidence_of_other_documents(graph):
    cands, _ = collect(graph, {"acme"}, "both", permits=lambda doc: doc == "d1")
    assert set(cands) == {"c-alice"}
    assert {e.document_id for e in cands["c-alice"].evidence} == {"d1"}
    none, _ = collect(graph, {"acme"}, "both", permits=lambda doc: False)
    assert none == {}


def test_mention_evidence_adds_weaker_candidates_and_merges_into_existing_ones(graph):
    cands, _ = collect(graph, {"alice"}, "both")
    add_mention_evidence(
        graph,
        cands,
        [
            {
                "entity_id": "alice",
                "document_id": "d1",
                "chunk_id": "c-alice",
                "surface_form": "Alice",
                "provenance": "rule_based",
                "confidence": None,
            },
            {
                "entity_id": "alice",
                "document_id": "d9",
                "chunk_id": "c-only-mention",
                "surface_form": "Alice",
                "provenance": "rule_based",
                "confidence": None,
            },
        ],
    )
    assert {e.kind for e in cands["c-alice"].evidence} == {"relationship", "mention"}
    assert [e.kind for e in cands["c-only-mention"].evidence] == ["mention"]
    ranked = [r.candidate.chunk_id for r in rank_candidates(list(cands.values()))]
    assert ranked == ["c-alice", "c-only-mention"]
    mention = cands["c-only-mention"].evidence[0]
    assert mention.entity.name == "Alice" and mention.surface_form == "Alice"


def test_entities_that_are_not_linked_produce_no_candidates(graph):
    cands, _ = collect(graph, set())
    assert cands == {}


def test_walks_are_deterministic(graph):
    a, _ = collect(graph, {"acme", "berlin"}, "both", hops=2)
    b, _ = collect(graph, {"berlin", "acme"}, "both", hops=2)
    assert [r.candidate.chunk_id for r in rank_candidates(list(a.values()))] == [
        r.candidate.chunk_id for r in rank_candidates(list(b.values()))
    ]
    assert edges_of(a) == edges_of(b)
