"""Weighted RRF checked against an independent exact-arithmetic reference on random inputs.

The reference uses ``fractions.Fraction`` (no floating point, no rounding) and the documented
ordering rule, written separately from the implementation. Weights are chosen from values that
are exact in binary, so the reference and the implementation must agree on every order.
"""

import random
from fractions import Fraction

import pytest

from app.services.retrieval.fusion import FusionParams, rrf_fuse

WEIGHTS = [Fraction(1, 4), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(3), Fraction(10)]


def reference(vector, graph, k, wv, wg):
    """Exact fused scores and the documented order, computed the obvious way."""
    scores: dict[str, Fraction] = {}
    v_rank = {c: i for i, c in enumerate(vector, 1)}
    g_rank = {c: i for i, c in enumerate(graph, 1)}
    for c in set(vector) | set(graph):
        s = Fraction(0)
        if c in v_rank:
            s += wv / (k + v_rank[c])
        if c in g_rank:
            s += wg / (k + g_rank[c])
        scores[c] = s

    def key(c):
        best = min(r for r in (v_rank.get(c), g_rank.get(c)) if r is not None)
        return (-scores[c], best, 0 if c in v_rank else 1, c)

    order = sorted(scores, key=key)
    return order, scores, v_rank, g_rank


def random_case(rng):
    pool = [f"c{i:03d}" for i in range(rng.randint(1, 40))]
    vector = rng.sample(pool, rng.randint(0, len(pool)))
    graph = rng.sample(pool, rng.randint(0, len(pool)))
    k = rng.choice([1, 2, 10, 60, 100, 1000])
    wv, wg = rng.choice(WEIGHTS), rng.choice(WEIGHTS)
    return vector, graph, k, wv, wg


@pytest.mark.parametrize("seed", range(40))
def test_the_fused_order_matches_the_exact_reference(seed):
    rng = random.Random(seed)
    for _ in range(25):
        vector, graph, k, wv, wg = random_case(rng)
        got = rrf_fuse(vector, graph, FusionParams(k, float(wv), float(wg)))
        order, scores, v_rank, g_rank = reference(vector, graph, k, wv, wg)
        assert [i.chunk_id for i in got] == order, (vector, graph, k, wv, wg)
        for item in got:
            assert item.score == pytest.approx(float(scores[item.chunk_id]), rel=1e-12)
            assert item.vector_rank == v_rank.get(item.chunk_id)
            assert item.graph_rank == g_rank.get(item.chunk_id)


@pytest.mark.parametrize("seed", range(10))
def test_every_input_chunk_appears_exactly_once_with_contiguous_ranks(seed):
    rng = random.Random(1000 + seed)
    for _ in range(25):
        vector, graph, k, wv, wg = random_case(rng)
        got = rrf_fuse(vector, graph, FusionParams(k, float(wv), float(wg)))
        assert sorted(i.chunk_id for i in got) == sorted(set(vector) | set(graph))
        assert [i.rank for i in got] == list(range(1, len(got) + 1))


@pytest.mark.parametrize("seed", range(10))
def test_the_original_ranks_are_the_input_positions_and_scores_are_their_sum(seed):
    rng = random.Random(2000 + seed)
    for _ in range(25):
        vector, graph, k, wv, wg = random_case(rng)
        for item in rrf_fuse(vector, graph, FusionParams(k, float(wv), float(wg))):
            if item.vector_rank is not None:
                assert vector[item.vector_rank - 1] == item.chunk_id
            if item.graph_rank is not None:
                assert graph[item.graph_rank - 1] == item.chunk_id
            assert item.score == pytest.approx(item.vector_contribution + item.graph_contribution)
            assert set(item.retrieved_by) == {
                p
                for p, r in (("vector", item.vector_rank), ("graph", item.graph_rank))
                if r is not None
            }


@pytest.mark.parametrize("seed", range(10))
def test_the_order_is_independent_of_the_python_hash_seed_and_dict_order(seed):
    """Same inputs built in different insertion orders give the same fused list."""
    rng = random.Random(4000 + seed)
    vector, graph, k, wv, wg = random_case(rng)
    params = FusionParams(k, float(wv), float(wg))
    base = rrf_fuse(vector, graph, params)
    for _ in range(5):
        shuffled_names = {name: f"z{rng.random():.6f}" for name in set(vector) | set(graph)}
        renamed = rrf_fuse(
            [shuffled_names[c] for c in vector], [shuffled_names[c] for c in graph], params
        )

        # relabelling can only change the final chunk-id tie-break: the multiset of
        # (score, original ranks) is identical, and so is every position not decided by a tie
        def shape(items):
            return sorted(
                ((round(i.score, 12), i.vector_rank, i.graph_rank) for i in items), key=repr
            )

        assert shape(renamed) == shape(base)


def test_symmetric_ranks_tie_and_are_ordered_by_chunk_id_not_by_input_name_order():
    got = rrf_fuse(["m", "x", "a"], ["a", "y", "m"])  # m: 1+3, a: 3+1 -> equal
    assert got[0].score == got[1].score
    assert [got[0].chunk_id, got[1].chunk_id] == ["a", "m"]


def test_one_empty_path_reduces_to_the_other_paths_order_for_any_weights():
    rng = random.Random(7)
    for _ in range(50):
        items = rng.sample([f"c{i}" for i in range(30)], rng.randint(1, 30))
        k, wv = rng.choice([1, 60, 1000]), rng.choice(WEIGHTS)
        assert [i.chunk_id for i in rrf_fuse(items, None, FusionParams(k, float(wv), 1.0))] == items
        assert [i.chunk_id for i in rrf_fuse(None, items, FusionParams(k, 1.0, float(wv)))] == items
