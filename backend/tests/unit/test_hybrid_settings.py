"""Hybrid retrieval settings: defaults, bounds and the hop cap."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def make(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_defaults_match_the_approved_proposal(tmp_path):
    s = make(data_dir=tmp_path)
    assert (s.hybrid_rrf_k, s.hybrid_max_limit, s.hybrid_graph_max_chunks) == (60, 50, 50)
    assert (s.hybrid_graph_max_hops, s.hybrid_max_name_tokens) == (2, 8)


@pytest.mark.parametrize(
    "field, bad",
    [
        ("hybrid_rrf_k", 0),
        ("hybrid_rrf_k", 1001),
        ("hybrid_max_limit", 0),
        ("hybrid_graph_max_chunks", 501),
        ("hybrid_graph_max_hops", 0),
        ("hybrid_max_name_tokens", 17),
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, field, bad):
    with pytest.raises(ValidationError):
        make(data_dir=tmp_path, **{field: bad})


def test_environment_variables_are_read_with_the_graphrag_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHRAG_HYBRID_RRF_K", "25")
    monkeypatch.setenv("GRAPHRAG_HYBRID_MAX_LIMIT", "7")
    s = make(data_dir=tmp_path)
    assert (s.hybrid_rrf_k, s.hybrid_max_limit) == (25, 7)


def test_hybrid_hops_never_exceed_the_graph_api_cap(tmp_path):
    assert make(data_dir=tmp_path, graph_max_hops=1).effective_hybrid_graph_max_hops == 1
    assert make(data_dir=tmp_path, graph_max_hops=4).effective_hybrid_graph_max_hops == 2
    both = make(data_dir=tmp_path, graph_max_hops=6, hybrid_graph_max_hops=5)
    assert both.effective_hybrid_graph_max_hops == 5


def test_the_path_time_budget_defaults_and_bounds(tmp_path):
    assert make(data_dir=tmp_path).hybrid_path_timeout_seconds == 30.0
    assert make(data_dir=tmp_path, hybrid_path_timeout_seconds=0).hybrid_path_timeout_seconds == 0
    for bad in (-1, 601):
        with pytest.raises(ValidationError):
            make(data_dir=tmp_path, hybrid_path_timeout_seconds=bad)
