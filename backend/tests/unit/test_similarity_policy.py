"""One precision for filtering and presentation: boundaries around the threshold."""

import random

import pytest

from app.services.retrieval.vector_search import (
    SIMILARITY_DECIMALS,
    SIMILARITY_METRIC,
    passes_threshold,
    round_similarity,
)


def test_policy_uses_six_decimals_and_says_so():
    assert SIMILARITY_DECIMALS == 6
    assert "rounded to 6 decimals" in SIMILARITY_METRIC
    assert "not answer confidence" in SIMILARITY_METRIC


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.7999999, 0.8), (0.7999994, 0.799999), (0.1234567, 0.123457), (-0.25, -0.25), (1.0, 1.0)],
)
def test_rounding(score, expected):
    assert round_similarity(score) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("score", "threshold", "expected"),
    [
        (0.8, 0.8, True),  # exactly on the boundary
        (0.7999999, 0.8, True),  # displays as 0.800000, so it must pass a 0.8 threshold
        (0.8000001, 0.8, True),
        (0.7999994, 0.8, False),  # displays as 0.799999: genuinely below
        (0.799999, 0.799999, True),
        (0.7999994, 0.799999, True),
        (0.7999996, 0.799999, True),  # rounds up to 0.8, above the threshold
        (0.8, 0.8000004, True),  # finer than 6 decimals: the threshold is rounded the same way
        (0.8, 0.8000006, False),  # rounds to 0.800001, above 0.8
        (-1.0, -1.0, True),
        (1.0, 1.0, True),
        (0.5, None, True),  # no threshold: everything passes
        (-0.9, 0.0, False),
    ],
)
def test_threshold_boundaries(score, threshold, expected):
    assert passes_threshold(score, threshold) is expected


def test_a_result_always_passes_a_threshold_equal_to_the_value_it_displays():
    rng = random.Random(7)
    for _ in range(2000):
        score = rng.uniform(-1, 1)
        shown = round_similarity(score)
        assert passes_threshold(score, shown), (score, shown)


def test_ordering_by_displayed_value_never_contradicts_filtering():
    rng = random.Random(11)
    scores = sorted((rng.uniform(0.2, 0.9) for _ in range(500)), reverse=True)
    for threshold in (0.3, 0.55, 0.8):
        kept = [s for s in scores if passes_threshold(s, threshold)]
        assert kept == scores[: len(kept)]  # filtering a ranked list keeps a prefix
        assert all(round_similarity(s) >= round_similarity(threshold) for s in kept)
        assert all(round_similarity(s) < round_similarity(threshold) for s in scores[len(kept) :])
