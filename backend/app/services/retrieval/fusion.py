"""Weighted Reciprocal Rank Fusion (RRF) of the vector ranking and the graph ranking.

Pure and deterministic: no I/O, no clock, no randomness, no content or similarity values, and
no imports from the rest of the application. The input is two lists of chunk IDs, each ordered
best first, and a few validated parameters; the output is a fully ordered list.

    fused(c) = sum over the paths containing c of   weight_path / (rrf_k + rank_path(c))

Why ranks only: a cosine similarity and the graph's integer ranking features live on unrelated
scales, so adding or averaging them would invent a meaning they do not have. RRF consumes only
the position each path gave a chunk, and each path's ranking is produced independently of the
other before this module sees it.

What the fused score is NOT: it is a rank-based retrieval score. It is not a probability, not
relevance and not answer confidence, and it is not comparable across queries or configurations.
Deterministic ordering is also not evidence of relevance: it only means the same inputs always
give the same list. Whether fusion improves retrieval has to be measured against labelled data.

Ordering of the output (a total order, so ties cannot survive):

1. higher fused score first, compared after rounding to ``SCORE_COMPARE_DECIMALS`` so that
   floating-point noise cannot decide an order;
2. the better (lower) of the chunk's path ranks;
3. chunks the vector path found before chunks only the graph path found;
4. ``chunk_id`` ascending.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_RRF_K = 60
MAX_RRF_K = 1000
MAX_WEIGHT = 10.0
SCORE_COMPARE_DECIMALS = 12  # used only to compare scores; never shown
SCORE_DISPLAY_DECIMALS = 6


class FusionError(ValueError):
    """Invalid fusion parameters or inputs (a caller bug, not a runtime retrieval failure)."""


@dataclass(frozen=True)
class FusionParams:
    """Validated fusion configuration."""

    rrf_k: int = DEFAULT_RRF_K
    vector_weight: float = 1.0
    graph_weight: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.rrf_k, bool) or not isinstance(self.rrf_k, int):
            raise FusionError("rrf_k must be an integer")
        if not 1 <= self.rrf_k <= MAX_RRF_K:
            raise FusionError(f"rrf_k must be between 1 and {MAX_RRF_K}")
        for name in ("vector_weight", "graph_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise FusionError(f"{name} must be a number")
            if not math.isfinite(value) or not 0 <= value <= MAX_WEIGHT:
                raise FusionError(f"{name} must be between 0 and {MAX_WEIGHT:g}")
        if self.vector_weight == 0 and self.graph_weight == 0:
            raise FusionError("at least one weight must be greater than 0")


@dataclass(frozen=True)
class FusedItem:
    """One chunk after fusion, with everything needed to reproduce and explain its score."""

    chunk_id: str
    rank: int  # 1-based position in the fused list
    score: float  # exact fused score (use ``display_score`` to show it)
    vector_rank: int | None  # original 1-based rank in the vector list, None if absent
    graph_rank: int | None  # original 1-based rank in the graph list, None if absent
    vector_contribution: float
    graph_contribution: float

    @property
    def retrieved_by(self) -> tuple[str, ...]:
        found = []
        if self.vector_rank is not None:
            found.append("vector")
        if self.graph_rank is not None:
            found.append("graph")
        return tuple(found)

    @property
    def display_score(self) -> float:
        return round(self.score, SCORE_DISPLAY_DECIMALS)


def rrf_contribution(weight: float, rrf_k: int, rank: int) -> float:
    """What one path adds for a chunk at ``rank`` (1-based)."""
    if rank < 1:
        raise FusionError("ranks are 1-based")
    return weight / (rrf_k + rank)


def _ranks(ids: Sequence[str] | None, label: str) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for position, chunk_id in enumerate(ids or (), start=1):
        if chunk_id in ranks:
            raise FusionError(f"{label} list contains chunk {chunk_id!r} more than once")
        ranks[chunk_id] = position
    return ranks


def rrf_fuse(
    vector_ids: Sequence[str] | None,
    graph_ids: Sequence[str] | None,
    params: FusionParams | None = None,
) -> list[FusedItem]:
    """Fuse two ranked chunk-ID lists (best first). ``None`` or empty means "contributes nothing".

    A path that did not run or failed must be passed as ``None`` or empty; the caller (not this
    function) is responsible for reporting that the result is then not a complete hybrid.
    The same chunk appearing in both lists is merged into one item that keeps both original
    ranks. A chunk repeated *within* one list is a caller bug and raises ``FusionError``.
    """
    params = params or FusionParams()
    vector = _ranks(vector_ids, "vector")
    graph = _ranks(graph_ids, "graph")

    scored: list[tuple[tuple, FusedItem]] = []
    for chunk_id in vector.keys() | graph.keys():
        v_rank, g_rank = vector.get(chunk_id), graph.get(chunk_id)
        v_part = rrf_contribution(params.vector_weight, params.rrf_k, v_rank) if v_rank else 0.0
        g_part = rrf_contribution(params.graph_weight, params.rrf_k, g_rank) if g_rank else 0.0
        score = v_part + g_part
        best_rank = min(r for r in (v_rank, g_rank) if r is not None)
        key = (
            -round(score, SCORE_COMPARE_DECIMALS),
            best_rank,
            0 if v_rank is not None else 1,
            chunk_id,
        )
        item = FusedItem(chunk_id, 0, score, v_rank, g_rank, v_part, g_part)
        scored.append((key, item))

    scored.sort(key=lambda pair: pair[0])
    return [
        FusedItem(
            it.chunk_id,
            position,
            it.score,
            it.vector_rank,
            it.graph_rank,
            it.vector_contribution,
            it.graph_contribution,
        )
        for position, (_, it) in enumerate(scored, start=1)
    ]
