"""What a retrieval response may call itself, given the state of each path."""

import pytest

from app.services.retrieval.path_status import classify_paths

# (vector, graph) -> (mode, complete)
TABLE = [
    # both requested
    (("ok", "ok"), ("hybrid", True)),
    (("ok", "unavailable"), ("partial", False)),
    (("unavailable", "ok"), ("partial", False)),
    (("unavailable", "unavailable"), ("failed", False)),
    # only one requested
    (("ok", "skipped"), ("vector_only", True)),
    (("skipped", "ok"), ("graph_only", True)),
    (("unavailable", "skipped"), ("failed", False)),
    (("skipped", "unavailable"), ("failed", False)),
]


@pytest.mark.parametrize("states, expected", TABLE)
def test_every_combination_has_one_explicit_label(states, expected):
    summary = classify_paths(*states)
    assert (summary.mode, summary.complete) == expected


def test_both_paths_successful_is_the_only_complete_hybrid():
    flags = {s: classify_paths(*s).is_complete_hybrid for s, _ in TABLE}
    assert [s for s, v in flags.items() if v] == [("ok", "ok")]


def test_a_partial_result_is_never_presented_as_a_complete_hybrid():
    for states in (("ok", "unavailable"), ("unavailable", "ok")):
        summary = classify_paths(*states)
        assert not summary.is_complete_hybrid and not summary.complete
        assert summary.mode == "partial"
        assert "NOT a complete hybrid" in summary.message
        assert len(summary.failed) == 1 and len(summary.succeeded) == 1


def test_the_failed_and_succeeding_paths_are_named():
    s = classify_paths("ok", "unavailable")
    assert (s.succeeded, s.failed, s.skipped) == (("vector",), ("graph",), ())
    assert "graph path failed" in s.message and "vector path only" in s.message
    s = classify_paths("unavailable", "ok")
    assert (s.succeeded, s.failed) == (("graph",), ("vector",))


def test_both_failed_says_so_and_has_no_succeeding_path():
    s = classify_paths("unavailable", "unavailable")
    assert s.mode == "failed" and s.succeeded == () and set(s.failed) == {"vector", "graph"}
    assert s.message.startswith("Retrieval failed")


def test_a_path_that_was_not_requested_is_skipped_not_failed():
    s = classify_paths("ok", "skipped")
    assert s.skipped == ("graph",) and s.failed == () and s.requested == ("vector",)
    assert "complete hybrid" not in s.message.lower().replace("not a complete hybrid", "")


def test_nothing_requested_is_a_caller_bug():
    with pytest.raises(ValueError, match="at least one retrieval path"):
        classify_paths("skipped", "skipped")


@pytest.mark.parametrize("bad", ["failed", "OK", "", None])
def test_unknown_states_are_rejected(bad):
    with pytest.raises(ValueError, match="unknown state"):
        classify_paths(bad, "ok")
    with pytest.raises(ValueError, match="unknown state"):
        classify_paths("ok", bad)
