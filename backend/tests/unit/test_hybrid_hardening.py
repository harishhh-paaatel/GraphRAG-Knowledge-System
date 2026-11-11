"""M4 hardening of the hybrid orchestrator, with both paths faked.

Final verification of the fused results, one shared document scope, per-path time budgets, a
bounded worker pool, and recovery after failures.
"""

import threading
import time
from contextlib import nullcontext

import pytest

from app.services.embeddings.service import VectorIndexBuilding
from app.services.graph.service import GraphUnavailable
from app.services.retrieval import hybrid as hybrid_module
from app.services.retrieval.entity_linker import UnknownEntities
from app.services.retrieval.hybrid import HybridRetriever, RetrievalUnavailable
from app.services.retrieval.scope import ChunkRecord, DocumentScope
from tests.unit.test_hybrid_orchestration import (
    FakeGraph,
    ghit,
    graph_outcome,
    vector_outcome,
    vhit,
)


class World:
    """A fake SQLite plus fake paths, so each test states exactly what is true."""

    def __init__(self, monkeypatch, *, allowed=None):
        self.vanished: set[str] = set()
        self.status: dict[str, str] = {}
        self.document: dict[str, str] = {}
        self.content: dict[str, str] = {}
        self.lookups: list[list[str]] = []
        self.scope_calls = 0
        self.vector_scopes: list[object] = []
        self.scope = DocumentScope(allowed=allowed, not_ready=frozenset())
        monkeypatch.setattr(hybrid_module, "lookup_chunks", self._lookup)
        monkeypatch.setattr(hybrid_module, "resolve_document_scope", self._resolve)
        self.monkeypatch = monkeypatch

    def _resolve(self, session, document_ids):
        self.scope_calls += 1
        return self.scope

    def _lookup(self, session, chunk_ids, scope=None):
        ids = list(chunk_ids)
        self.lookups.append(ids)
        return {
            cid: ChunkRecord(
                cid,
                self.document.get(cid, "d1"),
                "f.txt",
                self.content.get(cid, f"content of {cid}"),
                3,
                "Sec",
                7,
                self.status.get(cid, "completed"),
            )
            for cid in ids
            if cid not in self.vanished
        }

    def retriever(self, vector, graph, **kwargs):
        def fake_vectors(vectors, factory, query, **kw):
            self.vector_scopes.append(kw.get("scope"))
            if callable(vector):
                return vector()
            if isinstance(vector, BaseException):
                raise vector
            return vector

        self.monkeypatch.setattr(hybrid_module, "search_vectors", fake_vectors)
        fake_graph = (
            graph
            if hasattr(graph, "search")
            else FakeGraph(
                result=None if isinstance(graph, BaseException) else graph,
                error=graph if isinstance(graph, BaseException) else None,
            )
        )
        self.graph = fake_graph
        return HybridRetriever(object(), fake_graph, lambda: nullcontext(object()), **kwargs)


@pytest.fixture
def world(monkeypatch):
    return World(monkeypatch)


# ------------------------------------------------------------ final verification


def test_a_chunk_that_vanished_during_the_request_is_dropped_and_the_next_one_takes_its_place(
    world,
):
    world.vanished = {"a"}
    r = world.retriever(vector_outcome(vhit(1, "a"), vhit(2, "b"), vhit(3, "c")), graph_outcome())
    outcome = r.search("q", limit=2)
    assert [x.chunk_id for x in outcome.results] == ["b", "c"]
    assert [x.rank for x in outcome.results] == [1, 2]  # contiguous
    assert outcome.dropped_stale_results == 1
    assert [x.fused.vector_rank for x in outcome.results] == [2, 3]  # original ranks untouched
    assert len(outcome.fused) == 3  # the complete fused list is unchanged
    assert any("1 fused result(s) were dropped" in w for w in outcome.warnings)
    assert world.lookups == [["a", "b"], ["c"]]  # the refill needed one more lookup


def test_with_nothing_vanished_the_verification_is_a_single_lookup(world):
    r = world.retriever(vector_outcome(vhit(1, "a"), vhit(2, "b")), graph_outcome(ghit(1, "g")))
    outcome = r.search("q", limit=2)
    assert len(world.lookups) == 1 and outcome.dropped_stale_results == 0


def test_every_chunk_vanishing_gives_an_empty_but_honest_result(world):
    world.vanished = {"a", "b"}
    r = world.retriever(vector_outcome(vhit(1, "a"), vhit(2, "b")), graph_outcome())
    outcome = r.search("q", limit=5)
    assert outcome.results == [] and outcome.dropped_stale_results == 2
    assert outcome.summary.mode == "hybrid"  # both paths ran; nothing survived verification


def test_a_document_that_left_the_completed_state_is_dropped(world):
    world.status = {"a": "processing", "b": "failed"}
    r = world.retriever(vector_outcome(vhit(1, "a"), vhit(2, "b"), vhit(3, "c")), graph_outcome())
    outcome = r.search("q", limit=5)
    assert [x.chunk_id for x in outcome.results] == ["c"] and outcome.dropped_stale_results == 2


def test_a_chunk_outside_the_requested_documents_is_dropped(monkeypatch):
    world = World(monkeypatch, allowed=frozenset({"d1"}))
    world.document = {"b": "d2"}
    r = world.retriever(vector_outcome(vhit(1, "a"), vhit(2, "b")), graph_outcome(ghit(1, "g")))
    outcome = r.search("q", limit=5, document_ids=["d1"])
    assert sorted(x.chunk_id for x in outcome.results) == ["a", "g"]
    assert outcome.dropped_stale_results == 1


def test_what_is_returned_is_what_is_stored_at_verification_time(world):
    world.content = {"a": "freshly stored text"}
    r = world.retriever(vector_outcome(vhit(1, "a")), graph_outcome(ghit(1, "a")))
    [result] = r.search("q").results
    assert result.content == "freshly stored text"
    assert (result.filename, result.page_number, result.section, result.chunk_index) == (
        "f.txt",
        3,
        "Sec",
        7,
    )
    assert result.vector.chunk_id == "a" and result.graph.chunk.chunk_id == "a"


def test_both_paths_share_one_document_scope_resolved_once(world):
    r = world.retriever(vector_outcome(vhit(1, "a")), graph_outcome(ghit(1, "g")))
    r.search("q", document_ids=["d1"])
    assert world.scope_calls == 1
    assert world.vector_scopes == [world.scope]
    assert world.graph.calls[0][1]["scope"] is world.scope


def test_a_graph_snapshot_behind_the_database_is_reported(world):
    behind = graph_outcome(ghit(1, "g"))
    behind.database_revision = behind.revision + 2
    outcome = world.retriever(vector_outcome(vhit(1, "a")), behind).search("q")
    assert any("graph snapshot used is behind the database" in w for w in outcome.warnings)


def test_stale_graph_evidence_is_reported_in_the_warnings(world):
    stale = graph_outcome(ghit(1, "g"))
    stale.dropped_stale_evidence = 3
    outcome = world.retriever(vector_outcome(), stale).search("q")
    assert any("3 relationship evidence item(s)" in w for w in outcome.warnings)


# --------------------------------------------------------------------- time budgets


def slow(seconds, result):
    def run():
        time.sleep(seconds)
        return result

    return run


class SlowGraph:
    def __init__(self, seconds, result, record=None):
        self.seconds, self.result, self.record = seconds, result, record

    def search(self, query, **kwargs):
        if self.record is not None:
            self.record.append(threading.current_thread().name)
        time.sleep(self.seconds)
        return self.result


def test_a_slow_vector_path_times_out_and_the_graph_result_is_still_returned(world):
    r = world.retriever(
        slow(0.5, vector_outcome(vhit(1, "a"))),
        graph_outcome(ghit(1, "g")),
        path_timeout_seconds=0.1,
    )
    try:
        started = time.perf_counter()
        outcome = r.search("q")
        elapsed = time.perf_counter() - started
    finally:
        r.close()
    assert elapsed < 0.4  # it did not wait for the slow path
    assert outcome.summary.mode == "partial" and outcome.summary.failed == ("vector",)
    assert outcome.vector_report.error_code == "vector_timeout"
    assert "did not answer within 0.1 s" in outcome.vector_report.error_message
    assert [x.chunk_id for x in outcome.results] == ["g"]
    assert "NOT a complete hybrid" in outcome.warnings[0]


def test_a_slow_graph_path_times_out_and_the_vector_result_is_still_returned(world):
    r = world.retriever(
        vector_outcome(vhit(1, "a")),
        SlowGraph(0.5, graph_outcome(ghit(1, "g"))),
        path_timeout_seconds=0.1,
    )
    try:
        outcome = r.search("q")
    finally:
        r.close()
    assert outcome.graph_report.error_code == "graph_timeout"
    assert outcome.summary.mode == "partial" and [x.chunk_id for x in outcome.results] == ["a"]


def test_both_paths_timing_out_is_retrieval_unavailable(world):
    r = world.retriever(
        slow(0.5, vector_outcome(vhit(1, "a"))),
        SlowGraph(0.5, graph_outcome(ghit(1, "g"))),
        path_timeout_seconds=0.1,
    )
    try:
        with pytest.raises(RetrievalUnavailable) as caught:
            r.search("q")
    finally:
        r.close()
    assert caught.value.reports["vector"].error_code == "vector_timeout"
    assert caught.value.reports["graph"].error_code == "graph_timeout"


def test_paths_that_answer_in_time_are_unaffected(world):
    r = world.retriever(
        slow(0.02, vector_outcome(vhit(1, "a"))),
        SlowGraph(0.02, graph_outcome(ghit(1, "g"))),
        path_timeout_seconds=2.0,
    )
    try:
        outcome = r.search("q")
    finally:
        r.close()
    assert outcome.summary.is_complete_hybrid and sorted(x.chunk_id for x in outcome.results) == [
        "a",
        "g",
    ]


def test_without_a_time_budget_the_paths_run_in_the_calling_thread(world):
    threads: list[str] = []
    r = world.retriever(vector_outcome(), SlowGraph(0, graph_outcome(), record=threads))
    r.search("q")
    assert threads == [threading.current_thread().name]
    assert r._pools == {}
    r.close()  # harmless without a pool


def test_with_a_time_budget_the_paths_run_in_the_worker_pool(world):
    threads: list[str] = []
    r = world.retriever(
        vector_outcome(), SlowGraph(0, graph_outcome(), record=threads), path_timeout_seconds=2.0
    )
    try:
        r.search("q")
    finally:
        r.close()
    assert threads[0].startswith("hybrid-graph")


def test_request_errors_raised_inside_the_pool_reach_the_caller_unchanged(world):
    r = world.retriever(vector_outcome(), UnknownEntities(["nope"]), path_timeout_seconds=2.0)
    try:
        with pytest.raises(UnknownEntities) as caught:
            r.search("q")
    finally:
        r.close()
    assert caught.value.entity_ids == ["nope"]


def test_a_timeout_error_raised_by_a_path_itself_is_not_mistaken_for_the_time_budget(world):
    r = world.retriever(
        vector_outcome(vhit(1, "a")), TimeoutError("socket timed out"), path_timeout_seconds=2.0
    )
    try:
        outcome = r.search("q")
    finally:
        r.close()
    assert outcome.graph_report.error_code == "graph_error"  # an unexpected error, not a timeout
    assert "socket" not in outcome.graph_report.error_message


def test_stuck_paths_cannot_pile_up_threads_or_starve_the_other_path(world):
    stuck = threading.Event()

    class StuckGraph:
        def __init__(self):
            self.started = 0

        def search(self, query, **kwargs):
            self.started += 1
            stuck.wait(5)  # released by the test
            return graph_outcome(ghit(1, "g"))

    graph = StuckGraph()

    def graph_workers():  # threads of earlier tests may still be finishing: count the change
        return len([t for t in threading.enumerate() if t.name.startswith("hybrid-graph")])

    before = graph_workers()
    r = world.retriever(vector_outcome(vhit(1, "a")), graph, path_timeout_seconds=0.05)
    try:
        for _ in range(2):  # two searches abandon the graph pool's two workers
            first = r.search("q")
            assert first.graph_report.error_code == "graph_timeout"
            assert [x.chunk_id for x in first.results] == ["a"]  # the vector path is unaffected
        third = r.search("q")  # the graph pool is saturated: queued, timed out, cancelled
        assert third.graph_report.error_code == "graph_timeout"
        assert graph.started == 2  # the third call never started
        assert [x.chunk_id for x in third.results] == ["a"]  # the vector path still answers
        assert third.summary.mode == "partial"
        assert graph_workers() - before <= 2  # bounded
    finally:
        stuck.set()
        r.close()


def test_a_stuck_vector_path_cannot_starve_the_graph_path_either(world):
    stuck = threading.Event()

    def stuck_vector():
        stuck.wait(5)
        return vector_outcome(vhit(1, "a"))

    r = world.retriever(stuck_vector, graph_outcome(ghit(1, "g")), path_timeout_seconds=0.05)
    try:
        for _ in range(3):
            outcome = r.search("q")
            assert outcome.vector_report.error_code == "vector_timeout"
            assert [x.chunk_id for x in outcome.results] == ["g"]
    finally:
        stuck.set()
        r.close()


def test_the_retriever_recovers_once_the_slow_path_is_fast_again(world):
    state = {"delay": 0.4}

    class Flaky:
        def search(self, query, **kwargs):
            time.sleep(state["delay"])
            return graph_outcome(ghit(1, "g"))

    r = world.retriever(vector_outcome(vhit(1, "a")), Flaky(), path_timeout_seconds=0.1)
    try:
        assert r.search("q").summary.mode == "partial"
        state["delay"] = 0.0
        time.sleep(0.5)  # let the abandoned call finish
        healthy = r.search("q")
        assert healthy.summary.is_complete_hybrid
        assert sorted(x.chunk_id for x in healthy.results) == ["a", "g"]
    finally:
        r.close()


def test_a_slow_path_that_finishes_late_does_not_change_a_later_response(world):
    r = world.retriever(
        vector_outcome(vhit(1, "a")),
        SlowGraph(0.3, graph_outcome(ghit(1, "late"))),
        path_timeout_seconds=0.05,
    )
    try:
        first = r.search("q")
        time.sleep(0.5)  # the abandoned graph call completes in the background
        assert [x.chunk_id for x in first.results] == ["a"]  # the returned result is unchanged
        assert first.graph_outcome is None
    finally:
        r.close()


# ----------------------------------------------------------------------- recovery


def test_failures_and_recoveries_alternate_without_leaving_state_behind(world):
    modes = ["ok", "vector_down", "ok", "graph_down", "ok", "both_down", "ok"]
    expected = ["hybrid", "partial", "hybrid", "partial", "hybrid", None, "hybrid"]
    for mode, want in zip(modes, expected, strict=True):
        vector = (
            VectorIndexBuilding("building")
            if mode in ("vector_down", "both_down")
            else vector_outcome(vhit(1, "a"))
        )
        graph = (
            GraphUnavailable("down")
            if mode in ("graph_down", "both_down")
            else graph_outcome(ghit(1, "g"))
        )
        r = world.retriever(vector, graph)
        if want is None:
            with pytest.raises(RetrievalUnavailable):
                r.search("q")
            continue
        outcome = r.search("q")
        assert outcome.summary.mode == want
        if want == "hybrid":
            assert sorted(x.chunk_id for x in outcome.results) == ["a", "g"]
            assert outcome.warnings == []
