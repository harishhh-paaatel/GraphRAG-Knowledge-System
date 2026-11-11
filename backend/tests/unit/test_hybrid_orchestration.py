"""The hybrid orchestrator with both paths faked: every outcome, exactly and in isolation.

``search_vectors`` and the graph retriever are replaced by fakes that return hand-built outcomes
or raise, so the fusion, duplicate handling, provenance and failure rules are tested without a
database, an index or a model.
"""

from contextlib import nullcontext

import pytest

from app.services.embeddings.errors import EmbeddingError, EmbeddingInputError
from app.services.embeddings.service import VectorIndexBuilding, VectorIndexUnavailable
from app.services.embeddings.vector_index import IndexCorruptError
from app.services.graph.service import GraphUnavailable
from app.services.retrieval import hybrid as hybrid_module
from app.services.retrieval.entity_linker import LinkResult, UnknownEntities
from app.services.retrieval.fusion import FusionParams, rrf_fuse
from app.services.retrieval.graph_search import (
    GraphHit,
    GraphSearchLimitError,
    GraphSearchOutcome,
    RankFeatures,
)
from app.services.retrieval.hybrid import HybridRetriever, RetrievalUnavailable
from app.services.retrieval.scope import ChunkRecord, DocumentScope
from app.services.retrieval.vector_search import VectorHit, VectorSearchOutcome

FEATURES = RankFeatures(1, False, True, 1, 1, 1, True)
SCOPE = DocumentScope(allowed=None, not_ready=frozenset())


def vhit(rank, chunk_id, similarity=0.5, doc="d1"):
    return VectorHit(
        rank=rank,
        chunk_id=chunk_id,
        document_id=doc,
        content=f"content of {chunk_id}",
        similarity=similarity,
        filename=f"{doc}.txt",
        page_number=None,
        section=None,
        chunk_index=rank,
        embedding_model="stub",
    )


def ghit(rank, chunk_id, doc="d1", decided_by=None):
    chunk = ChunkRecord(chunk_id, doc, f"{doc}.txt", f"content of {chunk_id}", None, None, rank)
    return GraphHit(rank, chunk, (), FEATURES, decided_by)


def vector_outcome(*hits, stale=False, unknown=()):
    return VectorSearchOutcome(
        query="q",
        results=list(hits),
        embedding_model="stub",
        embedding_model_revision=None,
        index_revision=3,
        database_revision=3,
        index_stale=stale,
        indexed_vectors=10,
        unknown_document_ids=list(unknown),
    )


def graph_outcome(*hits, status="ok", note=None, unknown=(), truncated=(), dropped=0):
    return GraphSearchOutcome(
        query="q",
        hits=list(hits),
        link=LinkResult(),
        status=status,
        note=note,
        revision=3,
        database_revision=3,
        direction="both",
        hops=1,
        relationship_types=None,
        truncated=list(truncated),
        ranked_total=len(hits),
        dropped_missing_chunks=dropped,
        dropped_out_of_scope=0,
        unknown_document_ids=list(unknown),
        entity_degree={},
    )


class FakeGraph:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def run(monkeypatch):
    """run(vector, graph, **search_kwargs): each is an outcome to return or an exception to raise.

    Returns (outcome, vector_calls, graph_calls); ``run.vector_calls`` is also kept so a test
    that expects an exception can still see whether the vector path was invoked.
    """

    def go(vector, graph, *, vanished=(), retriever_kwargs=None, **kwargs):
        go.vector_calls = vector_calls = []

        def fake_lookup(session, chunk_ids, scope=None):
            """Stands in for SQLite: every chunk exists unless the test says it vanished."""
            return {
                cid: ChunkRecord(cid, "d1", "d1.txt", f"content of {cid}", None, None, 0)
                for cid in chunk_ids
                if cid not in vanished
            }

        monkeypatch.setattr(hybrid_module, "lookup_chunks", fake_lookup)
        monkeypatch.setattr(hybrid_module, "resolve_document_scope", lambda s, ids: SCOPE)

        def fake_search_vectors(vectors, factory, query, **kw):
            vector_calls.append((query, kw))
            if isinstance(vector, BaseException):
                raise vector
            return vector

        monkeypatch.setattr(hybrid_module, "search_vectors", fake_search_vectors)
        if isinstance(graph, BaseException):
            fake_graph = FakeGraph(error=graph)
        else:
            fake_graph = FakeGraph(result=graph)
        retriever = HybridRetriever(
            object(), fake_graph, lambda: nullcontext(object()), **(retriever_kwargs or {})
        )
        outcome = retriever.search("  some query  ", **kwargs)
        return outcome, vector_calls, fake_graph.calls

    go.vector_calls = []
    return go


# ------------------------------------------------------------------------ both succeed


def test_both_paths_succeeding_is_a_complete_hybrid(run):
    outcome, vector_calls, graph_calls = run(
        vector_outcome(vhit(1, "a"), vhit(2, "b"), vhit(3, "c")),
        graph_outcome(ghit(1, "c"), ghit(2, "a"), ghit(3, "d")),
    )
    s = outcome.summary
    assert (s.mode, s.complete, s.is_complete_hybrid) == ("hybrid", True, True)
    assert outcome.contributing == ("vector", "graph")
    assert [r.chunk_id for r in outcome.results] == ["a", "c", "b", "d"]
    assert outcome.warnings == []
    assert vector_calls[0][0] == "some query" and graph_calls[0][0] == "some query"


def test_a_chunk_found_by_both_paths_is_merged_once_with_both_original_ranks(run):
    outcome, *_ = run(
        vector_outcome(vhit(1, "a", 0.9), vhit(2, "b", 0.4)),
        graph_outcome(ghit(1, "b"), ghit(2, "a")),
    )
    ids = [r.chunk_id for r in outcome.results]
    assert sorted(ids) == ["a", "b"] and len(ids) == len(set(ids))
    by = {r.chunk_id: r for r in outcome.results}
    assert by["a"].retrieved_by == ("vector", "graph")
    assert (by["a"].vector.rank, by["a"].vector.similarity) == (1, 0.9)
    assert by["a"].graph.rank == 2 and by["a"].graph.features == FEATURES
    assert (by["b"].vector.rank, by["b"].graph.rank) == (2, 1)
    assert by["a"].fused.vector_rank == 1 and by["a"].fused.graph_rank == 2


def test_the_fused_order_is_exactly_what_rrf_gives_for_the_two_independent_rankings(run):
    vector = [vhit(i + 1, f"v{i}") for i in range(6)]
    graph = [ghit(i + 1, f"v{(i * 2) % 6}") for i in range(3)] + [ghit(4, "g-only")]
    outcome, *_ = run(vector_outcome(*vector), graph_outcome(*graph), limit=50)
    expected = rrf_fuse([h.chunk_id for h in vector], [h.chunk.chunk_id for h in graph])
    assert [r.chunk_id for r in outcome.results] == [i.chunk_id for i in expected]
    assert [r.fused for r in outcome.results] == expected


def test_the_fusion_parameters_are_applied_and_reported(run):
    params = FusionParams(rrf_k=10, vector_weight=3.0, graph_weight=0.5)
    outcome, *_ = run(vector_outcome(vhit(1, "a")), graph_outcome(ghit(1, "b")), fusion=params)
    assert outcome.params == params
    by = {r.chunk_id: r.fused.score for r in outcome.results}
    assert by["a"] == pytest.approx(3.0 / 11) and by["b"] == pytest.approx(0.5 / 11)


def test_the_full_lists_survive_the_limit_for_later_evaluation(run):
    vector = [vhit(i + 1, f"v{i}") for i in range(8)]
    graph = [ghit(i + 1, f"g{i}") for i in range(5)]
    outcome, *_ = run(vector_outcome(*vector), graph_outcome(*graph), limit=3)
    assert len(outcome.results) == 3 and len(outcome.fused) == 13
    assert [h.chunk_id for h in outcome.vector_outcome.results] == [h.chunk_id for h in vector]
    assert [h.chunk.chunk_id for h in outcome.graph_outcome.hits] == [
        h.chunk.chunk_id for h in graph
    ]
    assert outcome.fused[:3] == [r.fused for r in outcome.results]


def test_content_and_location_come_from_the_stored_chunk_whichever_path_found_it(run):
    outcome, *_ = run(vector_outcome(vhit(1, "a")), graph_outcome(ghit(1, "g")))
    by = {r.chunk_id: r for r in outcome.results}
    assert by["a"].content == "content of a" and by["a"].filename == "d1.txt"
    assert by["g"].content == "content of g" and by["g"].document_id == "d1"
    assert by["a"].vector is not None and by["a"].graph is None
    assert by["g"].graph is not None and by["g"].vector is None


def test_ranking_is_deterministic(run):
    args = (vector_outcome(vhit(1, "a"), vhit(2, "b")), graph_outcome(ghit(1, "b"), ghit(2, "c")))
    first, *_ = run(*args)
    for _ in range(3):
        again, *_ = run(*args)
        assert [r.fused for r in again.results] == [r.fused for r in first.results]


# ------------------------------------------------- a path that ran but found nothing


def test_a_graph_path_without_entities_still_counts_as_run_but_does_not_contribute(run):
    outcome, *_ = run(
        vector_outcome(vhit(1, "a")),
        graph_outcome(status="no_entities", note="the query names no entity"),
    )
    assert outcome.summary.mode == "hybrid" and outcome.summary.complete
    assert outcome.contributing == ("vector",)
    assert outcome.graph_report.status == "no_entities" and outcome.graph_report.state == "ok"
    assert [r.retrieved_by for r in outcome.results] == [("vector",)]
    assert any("graph path ran but contributed no results" in w for w in outcome.warnings)


def test_a_vector_path_without_results_still_counts_as_run_but_does_not_contribute(run):
    outcome, *_ = run(vector_outcome(), graph_outcome(ghit(1, "g")))
    assert outcome.summary.mode == "hybrid" and outcome.contributing == ("graph",)
    assert [r.retrieved_by for r in outcome.results] == [("graph",)]
    assert any("vector path ran but returned no results" in w for w in outcome.warnings)


def test_both_paths_empty_is_an_empty_but_successful_result(run):
    outcome, *_ = run(vector_outcome(), graph_outcome(status="no_entities"))
    assert outcome.results == [] and outcome.fused == [] and outcome.contributing == ()
    assert outcome.summary.mode == "hybrid"


def test_a_stale_vector_index_is_reported_and_still_counts_as_success(run):
    outcome, *_ = run(vector_outcome(vhit(1, "a"), stale=True), graph_outcome(ghit(1, "g")))
    assert outcome.vector_report.status == "stale" and outcome.summary.is_complete_hybrid
    assert any("behind the database" in w for w in outcome.warnings)


def test_unknown_documents_and_graph_notes_are_collected(run):
    outcome, *_ = run(
        vector_outcome(vhit(1, "a"), unknown=["x", "y"]),
        graph_outcome(ghit(1, "g"), unknown=["y", "z"], truncated=["max_edges"], dropped=2),
    )
    assert outcome.unknown_document_ids == ["x", "y", "z"]
    text = " | ".join(outcome.warnings)
    assert "traversal limits" in text and "2 graph evidence chunk(s)" in text


# ------------------------------------------------------------------- one path fails

VECTOR_FAILURES = [
    (VectorIndexBuilding("building"), "building", "vector_index_building"),
    (VectorIndexUnavailable("down"), "unavailable", "vector_index_unavailable"),
    (IndexCorruptError("corrupt"), "unavailable", "vector_index_unavailable"),
    (EmbeddingError("model missing"), "unavailable", "vector_index_unavailable"),
]


@pytest.mark.parametrize("error, status, code", VECTOR_FAILURES)
def test_a_failed_vector_path_leaves_a_clearly_labelled_partial_graph_result(
    run, error, status, code
):
    graph = graph_outcome(ghit(1, "g1"), ghit(2, "g2"))
    outcome, *_ = run(error, graph)
    s = outcome.summary
    assert (s.mode, s.complete, s.is_complete_hybrid) == ("partial", False, False)
    assert (s.succeeded, s.failed) == (("graph",), ("vector",))
    assert (outcome.vector_report.status, outcome.vector_report.error_code) == (status, code)
    assert outcome.vector_report.error_message == str(error)
    assert outcome.vector_outcome is None and outcome.graph_outcome is graph
    assert [r.chunk_id for r in outcome.results] == ["g1", "g2"]
    assert [r.retrieved_by for r in outcome.results] == [("graph",), ("graph",)]
    assert "NOT a complete hybrid" in outcome.warnings[0]
    assert any(code in w for w in outcome.warnings)


def test_a_failed_graph_path_leaves_a_clearly_labelled_partial_vector_result(run):
    vector = vector_outcome(vhit(1, "a"), vhit(2, "b"))
    outcome, *_ = run(vector, GraphUnavailable("database unavailable"))
    s = outcome.summary
    assert (s.mode, s.complete) == ("partial", False) and s.failed == ("graph",)
    assert outcome.graph_report.error_code == "graph_unavailable"
    assert outcome.graph_outcome is None and outcome.vector_outcome is vector
    assert [r.chunk_id for r in outcome.results] == ["a", "b"]
    assert "NOT a complete hybrid" in outcome.warnings[0]


def test_the_surviving_path_is_returned_exactly_as_it_would_be_alone(run):
    vector = [vhit(1, "a", 0.8), vhit(2, "b", 0.6)]
    outcome, *_ = run(vector_outcome(*vector), GraphUnavailable("x"))
    assert [(r.vector.rank, r.vector.similarity) for r in outcome.results] == [(1, 0.8), (2, 0.6)]
    assert all(r.graph is None for r in outcome.results)
    graph = [ghit(1, "g1"), ghit(2, "g2")]
    outcome, *_ = run(VectorIndexUnavailable("x"), graph_outcome(*graph))
    assert [r.graph for r in outcome.results] == graph and all(
        r.vector is None for r in outcome.results
    )


@pytest.mark.parametrize("failing", ["vector", "graph"])
def test_unexpected_errors_in_one_path_are_contained_logged_and_not_leaked(run, failing, caplog):
    boom = RuntimeError("secret internal detail /etc/passwd")
    ok_vector, ok_graph = vector_outcome(vhit(1, "a")), graph_outcome(ghit(1, "g"))
    with caplog.at_level("ERROR"):
        outcome, *_ = run(
            boom if failing == "vector" else ok_vector, boom if failing == "graph" else ok_graph
        )
    report = outcome.vector_report if failing == "vector" else outcome.graph_report
    assert outcome.summary.mode == "partial"
    assert report.error_code == f"{failing}_error"
    assert "secret" not in report.error_message and "see the server log" in report.error_message
    assert "secret internal detail" in caplog.text  # the detail is in the log, not the response
    assert not any("secret" in w for w in outcome.warnings)


# ---------------------------------------------------------------------- both paths fail


@pytest.mark.parametrize("vector_error, _status, vector_code", VECTOR_FAILURES)
def test_both_paths_failing_raises_with_each_paths_code(run, vector_error, _status, vector_code):
    with pytest.raises(RetrievalUnavailable) as caught:
        run(vector_error, GraphUnavailable("no graph"))
    reports = caught.value.reports
    assert reports["vector"].error_code == vector_code
    assert reports["graph"].error_code == "graph_unavailable"
    assert all(r.state == "unavailable" for r in reports.values())


def test_both_paths_failing_unexpectedly_is_also_unavailable(run):
    with pytest.raises(RetrievalUnavailable) as caught:
        run(RuntimeError("a"), ValueError("b"))
    assert caught.value.reports["vector"].error_code == "vector_error"
    assert caught.value.reports["graph"].error_code == "graph_error"


# --------------------------------------------------------------- request errors pass through


def test_a_request_error_is_not_a_path_failure(run):
    with pytest.raises(UnknownEntities):
        run(vector_outcome(vhit(1, "a")), UnknownEntities(["nope"]))
    with pytest.raises(GraphSearchLimitError):
        run(vector_outcome(vhit(1, "a")), GraphSearchLimitError("hops"))
    with pytest.raises(EmbeddingInputError):
        run(EmbeddingInputError("too long"), graph_outcome(ghit(1, "g")))


def test_the_graph_path_runs_first_so_a_refused_request_costs_no_embedding(run):
    with pytest.raises(UnknownEntities):
        run(vector_outcome(vhit(1, "a")), UnknownEntities(["nope"]))
    assert run.vector_calls == []  # the vector path was never invoked


def test_the_parameters_reach_the_right_path(run):
    _, vector_calls, graph_calls = run(
        vector_outcome(),
        graph_outcome(status="no_entities"),
        document_ids=["d1"],
        vector_top_k=7,
        similarity_threshold=0.25,
        hops=2,
        direction="incoming",
        relationship_types=["WORKS_AT"],
        max_chunks=9,
        include_mentions=True,
        entity_ids=["e1"],
        link_entities=False,
    )
    assert vector_calls[0][1] == {
        "top_k": 7,
        "similarity_threshold": 0.25,
        "document_ids": ["d1"],
        "scope": SCOPE,
    }
    assert graph_calls[0][1] == {
        "scope": SCOPE,
        "hops": 2,
        "direction": "incoming",
        "relationship_types": ["WORKS_AT"],
        "max_chunks": 9,
        "include_mentions": True,
        "entity_ids": ["e1"],
        "link_entities": False,
    }
