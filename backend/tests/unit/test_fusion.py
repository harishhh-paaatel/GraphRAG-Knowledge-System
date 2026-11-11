"""Weighted RRF: hand-computed values, ties, determinism, validation, purity."""

import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.services.retrieval.fusion import (
    FusedItem,
    FusionError,
    FusionParams,
    rrf_contribution,
    rrf_fuse,
)

APP = Path(__file__).resolve().parents[2] / "app"


def ids(items):
    return [i.chunk_id for i in items]


def by_id(items):
    return {i.chunk_id: i for i in items}


# ----------------------------------------------------------------- hand-computed values


def test_one_list_keeps_its_order_with_the_textbook_scores():
    items = rrf_fuse(["a", "b", "c"], None)
    assert ids(items) == ["a", "b", "c"]
    assert [i.rank for i in items] == [1, 2, 3]
    assert [i.score for i in items] == pytest.approx([1 / 61, 1 / 62, 1 / 63])
    assert all(i.graph_rank is None and i.retrieved_by == ("vector",) for i in items)


def test_two_lists_sum_the_contributions_of_each_path():
    # vector: a b c        graph: c a d        k = 60, weights 1
    items = by_id(rrf_fuse(["a", "b", "c"], ["c", "a", "d"]))
    assert items["a"].score == pytest.approx(1 / 61 + 1 / 62)  # vector 1, graph 2
    assert items["c"].score == pytest.approx(1 / 63 + 1 / 61)  # vector 3, graph 1
    assert items["b"].score == pytest.approx(1 / 62)  # vector 2 only
    assert items["d"].score == pytest.approx(1 / 63)  # graph 3 only
    assert ids(rrf_fuse(["a", "b", "c"], ["c", "a", "d"])) == ["a", "c", "b", "d"]


def test_a_chunk_found_by_both_paths_is_merged_once_and_keeps_both_original_ranks():
    items = rrf_fuse(["a", "b", "c"], ["c", "a", "d"])
    assert len(items) == 4 and len(set(ids(items))) == 4
    c = by_id(items)["c"]
    assert (c.vector_rank, c.graph_rank) == (3, 1)
    assert c.retrieved_by == ("vector", "graph")
    assert c.score == pytest.approx(c.vector_contribution + c.graph_contribution)
    assert c.vector_contribution == pytest.approx(1 / 63)
    assert c.graph_contribution == pytest.approx(1 / 61)


def test_original_ranks_are_the_positions_in_the_input_lists_not_the_fused_positions():
    items = by_id(rrf_fuse(["x", "y"], ["z", "y"]))
    assert items["y"].vector_rank == 2 and items["y"].graph_rank == 2
    assert items["z"].vector_rank is None and items["z"].graph_rank == 1
    assert items["x"].graph_rank is None


def test_k_changes_the_scores_exactly():
    items = by_id(rrf_fuse(["a"], ["a"], FusionParams(rrf_k=1)))
    assert items["a"].score == pytest.approx(1 / 2 + 1 / 2)
    items = by_id(rrf_fuse(["a"], None, FusionParams(rrf_k=1000)))
    assert items["a"].score == pytest.approx(1 / 1001)


def test_weights_scale_each_path_independently():
    params = FusionParams(vector_weight=2.0, graph_weight=0.5)
    items = by_id(rrf_fuse(["a"], ["b"], params))
    assert items["a"].score == pytest.approx(2.0 / 61)
    assert items["b"].score == pytest.approx(0.5 / 61)
    assert ids(rrf_fuse(["a"], ["b"], params)) == ["a", "b"]
    flipped = FusionParams(vector_weight=0.5, graph_weight=2.0)
    assert ids(rrf_fuse(["a"], ["b"], flipped)) == ["b", "a"]


def test_a_zero_weight_path_contributes_nothing_but_keeps_its_ranks_and_sorts_last():
    items = rrf_fuse(["a", "b"], ["z", "a"], FusionParams(graph_weight=0.0))
    by = by_id(items)
    assert by["z"].score == 0.0 and by["z"].graph_rank == 1 and by["z"].vector_rank is None
    assert by["a"].graph_contribution == 0.0 and by["a"].graph_rank == 2
    assert ids(items) == ["a", "b", "z"]


def test_rrf_contribution_is_weight_over_k_plus_rank():
    assert rrf_contribution(1.0, 60, 1) == pytest.approx(1 / 61)
    assert rrf_contribution(3.0, 10, 5) == pytest.approx(3 / 15)
    with pytest.raises(FusionError):
        rrf_contribution(1.0, 60, 0)


def test_the_display_score_is_rounded_but_the_score_is_exact():
    [item] = rrf_fuse(["a"], None)
    assert item.score == 1 / 61
    assert item.display_score == round(1 / 61, 6) == 0.016393


# ------------------------------------------------------------------ empty and missing paths


@pytest.mark.parametrize("vector, graph", [(None, None), ([], []), (None, []), ([], None)])
def test_nothing_in_means_nothing_out(vector, graph):
    assert rrf_fuse(vector, graph) == []


def test_none_and_an_empty_list_mean_the_same_thing():
    assert rrf_fuse(["a", "b"], None) == rrf_fuse(["a", "b"], [])
    assert rrf_fuse(None, ["a", "b"]) == rrf_fuse([], ["a", "b"])


def test_graph_only_input_is_ordered_by_graph_rank():
    items = rrf_fuse(None, ["g1", "g2", "g3"])
    assert ids(items) == ["g1", "g2", "g3"]
    assert all(i.retrieved_by == ("graph",) for i in items)


def test_the_inputs_are_not_modified():
    vector, graph = ["a", "b"], ["b", "c"]
    rrf_fuse(vector, graph)
    assert vector == ["a", "b"] and graph == ["b", "c"]


# ----------------------------------------------------------------------- ties and ordering


def test_a_full_tie_is_decided_by_chunk_id_ascending_whatever_the_names():
    items = rrf_fuse(["b", "m", "a"], ["a", "n", "b"])
    assert by_id(items)["a"].score == by_id(items)["b"].score
    assert ids(items)[:2] == ["a", "b"]
    items = rrf_fuse(["y", "m", "x"], ["x", "n", "y"])
    assert ids(items)[:2] == ["x", "y"]


def test_equal_scores_with_different_best_ranks_prefer_the_lower_rank():
    # k=1, weights 0.1 / 0.3:  A = vector 3 + graph 3 = 0.1/4 + 0.3/4,  B = graph 2 = 0.3/3.
    # Mathematically both are exactly 0.1; in floating point A is 0.1 and B is 0.0999...9.
    params = FusionParams(rrf_k=1, vector_weight=0.1, graph_weight=0.3)
    items = rrf_fuse(["v1", "v2", "A"], ["g1", "B", "A"], params)
    by = by_id(items)
    assert by["A"].score > by["B"].score  # the float noise that must NOT decide the order
    assert by["B"].rank < by["A"].rank  # the tie rule (better path rank) does
    assert by["B"].graph_rank == 2 and by["A"].graph_rank == 3


def test_a_vector_hit_beats_a_graph_only_hit_when_everything_else_is_equal():
    items = rrf_fuse(["v"], ["g"])
    assert ids(items) == ["v", "g"] and items[0].score == items[1].score


def test_ranks_are_contiguous_and_scores_never_increase():
    items = rrf_fuse([f"v{i}" for i in range(20)], [f"g{i}" for i in range(15)] + ["v3", "v1"])
    assert [i.rank for i in items] == list(range(1, len(items) + 1))
    scores = [round(i.score, 12) for i in items]
    assert scores == sorted(scores, reverse=True)


# ----------------------------------------------------------------------------- determinism


def test_repeated_calls_give_identical_results():
    vector, graph = [f"c{i}" for i in range(30)], [f"c{i}" for i in range(29, -1, -1)]
    first = rrf_fuse(vector, graph)
    assert all(rrf_fuse(vector, graph) == first for _ in range(5))


def test_the_result_does_not_depend_on_hash_seed_or_set_iteration_order():
    """The implementation unions two key sets; string hashing is randomised per process."""
    code = (
        "import json;"
        "from app.services.retrieval.fusion import rrf_fuse;"
        "v=['k%d'%i for i in range(25)];g=['k%d'%i for i in range(24,-1,-1)]+['only_g'];"
        "print(json.dumps([[i.chunk_id,i.rank,i.score] for i in rrf_fuse(v,g)]))"
    )
    outputs = set()
    for seed in ("0", "1", "2", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=APP.parent,
            check=True,
            timeout=60,
        )
        outputs.add(proc.stdout.strip())
    assert len(outputs) == 1
    assert json.loads(outputs.pop())[0][0].startswith(("k", "only_g"))


def test_fusion_looks_only_at_ranks_not_at_which_path_is_which_name():
    a = rrf_fuse(["p", "q"], ["q", "r"])
    swapped = rrf_fuse(["q", "r"], ["p", "q"])
    assert {i.chunk_id: i.score for i in a} == pytest.approx({i.chunk_id: i.score for i in swapped})


# ------------------------------------------------------------------------------ validation


def test_a_chunk_repeated_within_one_list_is_a_caller_bug():
    with pytest.raises(FusionError, match="vector list contains chunk 'a' more than once"):
        rrf_fuse(["a", "b", "a"], None)
    with pytest.raises(FusionError, match="graph list contains chunk"):
        rrf_fuse(None, ["a", "a"])


@pytest.mark.parametrize("k", [0, -1, 1001, 10**6])
def test_rrf_k_out_of_range_is_rejected(k):
    with pytest.raises(FusionError, match="rrf_k must be between 1 and 1000"):
        FusionParams(rrf_k=k)


@pytest.mark.parametrize("k", [60.0, "60", None, True])
def test_rrf_k_must_be_an_integer(k):
    with pytest.raises(FusionError, match="rrf_k must be an integer"):
        FusionParams(rrf_k=k)


@pytest.mark.parametrize("w", [-0.1, 10.5, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["vector_weight", "graph_weight"])
def test_weights_must_be_finite_and_between_zero_and_ten(field, w):
    with pytest.raises(FusionError, match="between 0 and 10"):
        FusionParams(**{field: w})


@pytest.mark.parametrize("w", ["1", None, True])
def test_weights_must_be_numbers(w):
    with pytest.raises(FusionError, match="must be a number"):
        FusionParams(vector_weight=w)


def test_at_least_one_weight_must_be_positive():
    with pytest.raises(FusionError, match="at least one weight"):
        FusionParams(vector_weight=0, graph_weight=0)
    FusionParams(vector_weight=0, graph_weight=1)
    FusionParams(vector_weight=10, graph_weight=0)


def test_integer_weights_and_the_default_parameters_are_accepted():
    p = FusionParams()
    assert (p.rrf_k, p.vector_weight, p.graph_weight) == (60, 1.0, 1.0)
    assert FusionParams(vector_weight=2, graph_weight=3).graph_weight == 3


def test_the_parameters_and_items_are_immutable():
    with pytest.raises(AttributeError):
        FusionParams().rrf_k = 5  # type: ignore[misc]
    [item] = rrf_fuse(["a"], None)
    assert isinstance(item, FusedItem)
    with pytest.raises(AttributeError):
        item.rank = 9  # type: ignore[misc]


# ------------------------------------------------------------------------------- purity


def test_the_module_is_pure_and_imports_nothing_from_the_application():
    tree = ast.parse((APP / "services/retrieval/fusion.py").read_text("utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"math", "collections", "dataclasses"}, imported
