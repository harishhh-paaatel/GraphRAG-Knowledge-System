"""Hybrid retrieval: run the vector path and the graph path independently, then fuse their ranks.

This module is the only place that knows both paths. Each path is a separate, self-contained
call (``search_vectors`` and ``GraphRetriever.search``); neither receives anything from the
other, and fusion sees only the two ranked chunk-ID lists. A path that fails is reported with a
machine-readable code and contributes nothing; the other path's result is returned unchanged.

What a response may call itself is decided by ``classify_paths`` from the state of the two
paths:

* both succeeded            -> ``hybrid`` (complete);
* exactly one failed        -> ``partial``: the results come from one path and are NOT a
                               complete hybrid result (the response says so);
* both failed               -> ``RetrievalUnavailable`` (HTTP 503 ``retrieval_unavailable``).

A path that succeeded but found nothing (for example a query that names no graph entity) still
counts as having run; the response lists which paths actually contributed results.

Request errors (unknown ``entity_ids``, a parameter over its cap, a query the embedder rejects)
are not path failures and propagate to the caller unchanged.

Consistency guarantees (M4)
---------------------------
* Both paths use ONE document scope, resolved once at the start of the request.
* The graph path confirms every piece of relationship evidence against SQLite before ranking.
* After fusion, the chunks about to be returned are looked up once more in a single read
  transaction: a chunk that vanished or left scope while the request ran is dropped (the next
  fused result takes its place) and counted, and the content, filename, page and section that
  are returned are the ones stored at that moment.
* Each path has an optional time budget; a path that exceeds it is reported unavailable
  (``vector_timeout`` / ``graph_timeout``) and the other path's result is still returned.

The full per-path lists and the full fused list are kept on the outcome so that later
evaluation can cut each of them at any K; the API returns only the fused top ``limit``.
"""

import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy.orm import Session, sessionmaker

from app.services.embeddings.errors import EmbeddingError, EmbeddingInputError
from app.services.embeddings.service import VectorIndexBuilding, VectorService
from app.services.embeddings.vector_index import VectorIndexError
from app.services.graph.service import GraphUnavailable
from app.services.retrieval.entity_linker import UnknownEntities
from app.services.retrieval.fusion import FusedItem, FusionParams, rrf_fuse
from app.services.retrieval.graph_search import (
    Direction,
    GraphHit,
    GraphRetriever,
    GraphSearchLimitError,
    GraphSearchOutcome,
)
from app.services.retrieval.path_status import PathState, PathSummary, classify_paths
from app.services.retrieval.scope import DocumentScope, lookup_chunks, resolve_document_scope
from app.services.retrieval.vector_search import (
    VectorHit,
    VectorSearchOutcome,
    search_vectors,
)

logger = logging.getLogger(__name__)

SCORE_MEANING = (
    "rank-based retrieval score (weighted reciprocal rank fusion); not a probability, not "
    "relevance and not answer confidence, and not comparable across queries or configurations"
)
INTERNAL_ERROR_MESSAGE = "unexpected internal error; see the server log"


class PathTimeout(Exception):
    """One path did not answer within its time budget."""

    def __init__(self, path: str, seconds: float) -> None:
        super().__init__(f"the {path} path did not answer within {seconds:g} s")
        self.path = path


class RetrievalUnavailable(Exception):
    """Every requested path failed; ``reports`` explains each one."""

    def __init__(self, reports: dict[str, "PathReport"]) -> None:
        super().__init__("no retrieval path is available")
        self.reports = reports


@dataclass(frozen=True)
class PathReport:
    """What happened to one path."""

    name: Literal["vector", "graph"]
    state: PathState  # ok | unavailable (| skipped, never produced here)
    status: str  # ok | stale | building | unavailable | no_entities | ambiguous_only
    candidates: int = 0  # chunks the path contributed to fusion
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class HybridResult:
    """One fused chunk with everything both paths knew about it."""

    rank: int
    chunk_id: str
    document_id: str
    content: str
    filename: str
    page_number: int | None
    section: str | None
    chunk_index: int
    fused: FusedItem
    vector: VectorHit | None  # original vector rank/similarity; None if the path did not find it
    graph: GraphHit | None  # original graph rank/features/evidence; None if not found

    @property
    def retrieved_by(self) -> tuple[str, ...]:
        return self.fused.retrieved_by


@dataclass
class HybridOutcome:
    query: str
    results: list[HybridResult]  # the fused top ``limit``
    fused: list[FusedItem]  # the complete fused list
    params: FusionParams
    summary: PathSummary
    vector_report: PathReport
    graph_report: PathReport
    vector_outcome: VectorSearchOutcome | None  # full vector list (None if the path failed)
    graph_outcome: GraphSearchOutcome | None  # full graph list (None if the path failed)
    warnings: list[str] = field(default_factory=list)
    unknown_document_ids: list[str] = field(default_factory=list)
    dropped_stale_results: int = 0  # fused results whose chunk vanished or left scope
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def contributing(self) -> tuple[str, ...]:
        """Paths that ran successfully and actually returned results."""
        out = []
        if self.vector_outcome is not None and self.vector_outcome.results:
            out.append("vector")
        if self.graph_outcome is not None and self.graph_outcome.hits:
            out.append("graph")
        return tuple(out)


class HybridRetriever:
    """Orchestrates one hybrid search over the vector service and the graph retriever."""

    def __init__(
        self,
        vectors: VectorService,
        graph: GraphRetriever,
        factory: sessionmaker[Session],
        *,
        path_timeout_seconds: float | None = None,
    ) -> None:
        self._vectors = vectors
        self._graph = graph
        self._factory = factory
        self._timeout = path_timeout_seconds if path_timeout_seconds else None
        # One small dedicated pool PER PATH, used only when a time budget is set, so that a path
        # which hangs cannot starve the other one. A path that overruns its budget keeps running
        # in the background (Python cannot interrupt it); each pool is bounded, so a path that
        # hangs for good cannot accumulate threads: its later requests time out visibly instead.
        self._pools: dict[str, ThreadPoolExecutor] = (
            {
                name: ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"hybrid-{name}")
                for name in ("graph", "vector")
            }
            if self._timeout
            else {}
        )

    def close(self) -> None:
        """Release the worker threads (application shutdown)."""
        for pool in self._pools.values():
            pool.shutdown(wait=False, cancel_futures=True)

    def _call(self, path: str, fn: Callable[[], object]):
        """Run one path, within the time budget if there is one."""
        if not self._pools:
            return fn()
        future: Future = self._pools[path].submit(fn)
        try:
            return future.result(timeout=self._timeout)
        except FutureTimeout:
            if future.done():  # the path itself raised a TimeoutError: not our timeout
                raise
            future.cancel()  # only effective while still queued behind stuck paths
            raise PathTimeout(path, self._timeout or 0.0) from None

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        document_ids: list[str] | None = None,
        vector_top_k: int = 20,
        similarity_threshold: float | None = None,
        hops: int = 1,
        direction: Direction = "both",
        relationship_types: list[str] | None = None,
        max_chunks: int = 20,
        include_mentions: bool = False,
        entity_ids: list[str] | None = None,
        link_entities: bool = True,
        fusion: FusionParams | None = None,
    ) -> HybridOutcome:
        """Run both paths and fuse them.

        Raises ``RetrievalUnavailable`` when every path failed, and lets request errors
        (``UnknownEntities``, ``GraphSearchLimitError``, ``EmbeddingInputError``) through.
        """
        started = time.perf_counter()
        params = fusion or FusionParams()
        query = query.strip()
        timings: dict[str, float] = {}

        with self._factory() as session:
            scope = resolve_document_scope(session, document_ids)

        # graph first: it is in memory, and it validates the request's graph parameters
        # before any embedding work is spent on a request that is going to be refused
        t = time.perf_counter()
        graph_outcome, graph_report = self._run_graph(
            query,
            scope=scope,
            hops=hops,
            direction=direction,
            relationship_types=relationship_types,
            max_chunks=max_chunks,
            include_mentions=include_mentions,
            entity_ids=entity_ids,
            link_entities=link_entities,
        )
        timings["graph"] = round((time.perf_counter() - t) * 1000, 3)

        t = time.perf_counter()
        vector_outcome, vector_report = self._run_vector(
            query, vector_top_k, similarity_threshold, document_ids, scope
        )
        timings["vector"] = round((time.perf_counter() - t) * 1000, 3)

        summary = classify_paths(vector_report.state, graph_report.state)
        if summary.mode == "failed":
            raise RetrievalUnavailable({"vector": vector_report, "graph": graph_report})

        t = time.perf_counter()
        vector_hits = {h.chunk_id: h for h in (vector_outcome.results if vector_outcome else [])}
        graph_hits = {h.chunk.chunk_id: h for h in (graph_outcome.hits if graph_outcome else [])}
        fused = rrf_fuse(
            list(vector_hits) if vector_outcome is not None else None,
            list(graph_hits) if graph_outcome is not None else None,
            params,
        )
        timings["fusion"] = round((time.perf_counter() - t) * 1000, 3)

        t = time.perf_counter()
        results, stale = self._verified_results(fused, vector_hits, graph_hits, limit, scope)
        timings["verify"] = round((time.perf_counter() - t) * 1000, 3)
        outcome = HybridOutcome(
            query=query,
            results=results,
            fused=fused,
            params=params,
            summary=summary,
            vector_report=vector_report,
            graph_report=graph_report,
            vector_outcome=vector_outcome,
            graph_outcome=graph_outcome,
            dropped_stale_results=stale,
            unknown_document_ids=sorted(
                set(vector_outcome.unknown_document_ids if vector_outcome else [])
                | set(graph_outcome.unknown_document_ids if graph_outcome else [])
            ),
        )
        outcome.warnings = self._warnings(outcome)
        timings["total"] = round((time.perf_counter() - started) * 1000, 3)
        outcome.timings_ms = timings
        return outcome

    # -- the two paths, each isolated from the other ------------------------------------

    def _run_graph(self, query: str, **kwargs) -> tuple[GraphSearchOutcome | None, PathReport]:
        try:
            outcome = self._call("graph", lambda: self._graph.search(query, **kwargs))
        except (UnknownEntities, GraphSearchLimitError):
            raise  # the request is wrong; that is not a path failure
        except PathTimeout as exc:
            return None, PathReport(
                "graph", "unavailable", "unavailable", 0, "graph_timeout", str(exc)
            )
        except GraphUnavailable as exc:
            return None, PathReport(
                "graph", "unavailable", "unavailable", 0, "graph_unavailable", str(exc)
            )
        except Exception:
            logger.exception("Graph retrieval path failed unexpectedly")
            return None, PathReport(
                "graph", "unavailable", "unavailable", 0, "graph_error", INTERNAL_ERROR_MESSAGE
            )
        return outcome, PathReport("graph", "ok", outcome.status, len(outcome.hits))

    def _run_vector(
        self,
        query: str,
        top_k: int,
        threshold: float | None,
        document_ids: list[str] | None,
        scope: DocumentScope,
    ) -> tuple[VectorSearchOutcome | None, PathReport]:
        try:
            outcome = self._call(
                "vector",
                lambda: search_vectors(
                    self._vectors,
                    self._factory,
                    query,
                    top_k=top_k,
                    similarity_threshold=threshold,
                    document_ids=document_ids,
                    scope=scope,
                ),
            )
        except EmbeddingInputError:
            raise  # the query itself is unacceptable to the embedder: a request error
        except PathTimeout as exc:
            return None, PathReport(
                "vector", "unavailable", "unavailable", 0, "vector_timeout", str(exc)
            )
        except VectorIndexBuilding as exc:
            return None, PathReport(
                "vector", "unavailable", "building", 0, "vector_index_building", str(exc)
            )
        except (EmbeddingError, VectorIndexError) as exc:
            return None, PathReport(
                "vector", "unavailable", "unavailable", 0, "vector_index_unavailable", str(exc)
            )
        except Exception:
            logger.exception("Vector retrieval path failed unexpectedly")
            return None, PathReport(
                "vector", "unavailable", "unavailable", 0, "vector_error", INTERNAL_ERROR_MESSAGE
            )
        status = "stale" if outcome.index_stale else "ok"
        return outcome, PathReport("vector", "ok", status, len(outcome.results))

    # -- assembly -----------------------------------------------------------------------

    def _verified_results(
        self,
        fused: list[FusedItem],
        vector_hits: dict[str, VectorHit],
        graph_hits: dict[str, GraphHit],
        limit: int,
        scope: DocumentScope,
    ) -> tuple[list[HybridResult], int]:
        """The fused top ``limit``, re-checked against SQLite in one read transaction.

        A path can lag the database by a few milliseconds (or much more for a stale index), so a
        chunk it found may have been deleted, or its document may have left ``completed``, while
        the request ran. Such chunks are dropped and the next fused result takes their place.
        What is returned (content, filename, page, section, document) is what is stored now.
        """
        results: list[HybridResult] = []
        dropped, cursor = 0, 0
        with self._factory() as session:
            while len(results) < limit and cursor < len(fused):
                batch = fused[cursor : cursor + (limit - len(results))]
                cursor += len(batch)
                records = lookup_chunks(session, [item.chunk_id for item in batch])
                for item in batch:
                    record = records.get(item.chunk_id)
                    if (
                        record is None
                        or record.processing_status != "completed"
                        or not scope.permits(record.document_id)
                    ):
                        dropped += 1
                        logger.warning("Dropping fused result for vanished chunk %s", item.chunk_id)
                        continue
                    results.append(
                        HybridResult(
                            rank=len(results) + 1,
                            chunk_id=record.chunk_id,
                            document_id=record.document_id,
                            content=record.content,
                            filename=record.filename,
                            page_number=record.page_number,
                            section=record.section,
                            chunk_index=record.chunk_index,
                            fused=item,
                            vector=vector_hits.get(item.chunk_id),
                            graph=graph_hits.get(item.chunk_id),
                        )
                    )
        return results, dropped

    @staticmethod
    def _warnings(outcome: HybridOutcome) -> list[str]:
        warnings: list[str] = []
        if not outcome.summary.complete:
            warnings.append(outcome.summary.message)
        for report in (outcome.vector_report, outcome.graph_report):
            if report.state == "unavailable":
                warnings.append(
                    f"{report.name} path unavailable ({report.error_code}): {report.error_message}"
                )
        if outcome.vector_report.status == "stale":
            warnings.append(
                "the vector index is behind the database (an ingestion is synchronizing it); "
                "the last consistent index was used"
            )
        vector, graph = outcome.vector_outcome, outcome.graph_outcome
        if vector is not None and not vector.results:
            warnings.append("the vector path ran but returned no results")
        if graph is not None:
            if not graph.hits:
                reason = graph.note or "no graph evidence matched"
                warnings.append(f"the graph path ran but contributed no results: {reason}")
            if graph.truncated:
                warnings.append(
                    "graph traversal limits were reached: " + ", ".join(graph.truncated)
                )
            if graph.dropped_missing_chunks:
                warnings.append(
                    f"{graph.dropped_missing_chunks} graph evidence chunk(s) no longer exist "
                    "and were dropped"
                )
            if graph.dropped_stale_evidence:
                warnings.append(
                    f"{graph.dropped_stale_evidence} relationship evidence item(s) are no longer "
                    "confirmed by the database and were dropped"
                )
            if graph.revision != graph.database_revision:
                warnings.append(
                    "the graph snapshot used is behind the database; concurrent changes are "
                    "not reflected in the graph results"
                )
        if outcome.dropped_stale_results:
            warnings.append(
                f"{outcome.dropped_stale_results} fused result(s) were dropped because their "
                "chunk no longer exists or left the searchable scope"
            )
        if outcome.unknown_document_ids:
            warnings.append("some document_ids do not exist and were ignored")
        return warnings
