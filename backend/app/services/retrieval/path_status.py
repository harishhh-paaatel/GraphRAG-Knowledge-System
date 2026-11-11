"""Explicit status of the two retrieval paths, and what a response may call itself.

A hybrid search runs a vector path and a graph path. Every response must say what actually
happened, because a result built from one surviving path is *not* a complete hybrid result and
must never be presented as one. This module is the single place that decides the label, from the
state of each path alone:

* ``ok``          the path ran and succeeded (it may legitimately have found nothing, for
                  example a query that names no graph entity; that is reported by the path's own
                  detail, it is not a failure);
* ``unavailable`` the path was requested and failed (index building, backend error, ...);
* ``skipped``     the path was not requested (for example ``/search/graph`` never runs vectors).

Resulting ``mode``:

=============  ==========================================================  ========
mode           meaning                                                     complete
=============  ==========================================================  ========
hybrid         both paths requested and both succeeded                     yes
vector_only    only the vector path was requested, and it succeeded        yes
graph_only     only the graph path was requested, and it succeeded         yes
partial        both requested, exactly one failed: results are one path's  no
failed         at least one requested, none succeeded: nothing to return   no
=============  ==========================================================  ========

Pure module: no I/O and no imports from the vector, graph or fusion code.
"""

from dataclasses import dataclass
from typing import Literal

PathState = Literal["ok", "unavailable", "skipped"]
Mode = Literal["hybrid", "vector_only", "graph_only", "partial", "failed"]

PATHS = ("vector", "graph")
_STATES = frozenset({"ok", "unavailable", "skipped"})


@dataclass(frozen=True)
class PathSummary:
    mode: Mode
    complete: bool
    requested: tuple[str, ...]
    succeeded: tuple[str, ...]
    failed: tuple[str, ...]
    skipped: tuple[str, ...]
    message: str

    @property
    def is_complete_hybrid(self) -> bool:
        """True only when both paths ran and succeeded."""
        return self.mode == "hybrid"


def classify_paths(vector: PathState, graph: PathState) -> PathSummary:
    """Label a retrieval from the state of its two paths."""
    states = {"vector": vector, "graph": graph}
    for name, state in states.items():
        if state not in _STATES:
            raise ValueError(f"unknown state {state!r} for the {name} path")
    requested = tuple(p for p in PATHS if states[p] != "skipped")
    if not requested:
        raise ValueError("at least one retrieval path must be requested")
    succeeded = tuple(p for p in requested if states[p] == "ok")
    failed = tuple(p for p in requested if states[p] == "unavailable")
    skipped = tuple(p for p in PATHS if states[p] == "skipped")

    mode: Mode
    if not succeeded:
        mode = "failed"
        message = "Retrieval failed: no requested path (" + ", ".join(failed) + ") is available."
    elif failed:
        mode = "partial"
        message = (
            f"Partial result: the {failed[0]} path failed; these results come from the "
            f"{succeeded[0]} path only and are NOT a complete hybrid result."
        )
    elif len(requested) == 2:
        mode = "hybrid"
        message = "Both the vector and the graph path succeeded."
    else:
        mode = "vector_only" if requested == ("vector",) else "graph_only"
        message = f"Only the {requested[0]} path was requested, and it succeeded."
    return PathSummary(
        mode=mode,
        complete=mode in ("hybrid", "vector_only", "graph_only"),
        requested=requested,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        message=message,
    )
