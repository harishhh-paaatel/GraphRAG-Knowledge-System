"""GraphService: the in-memory graph, its file, and the rules that keep them honest.

SQLite and the graph file cannot share a transaction, so consistency is *detected and
repaired* instead of assumed:

* every evidence change bumps ``graph_state.revision`` in the same SQLite transaction;
* the file stores the revision it was built from;
* any reader first compares the in-memory revision with SQLite's (one tiny query) and
  rebuilds from SQLite when they differ, so a query can never serve a stale graph;
* startup loads the file only if it validates *and* its revision matches SQLite, otherwise
  it rebuilds (missing, corrupt, truncated, wrong-format and stale files all take this path);
* a failed file write keeps the new graph in memory, leaves the previous file untouched, and
  is retried on the next sync.

Build happens under the knowledge write lock so it reads one committed state.
"""

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.services.extraction.persistence import WRITE_LOCK
from app.services.graph import analytics, consistency
from app.services.graph.builder import build_graph
from app.services.graph.revision import current_revision
from app.services.graph.store import (
    GraphLoadError,
    GraphPersistenceError,
    load_graph,
    save_graph,
)

logger = logging.getLogger(__name__)


class GraphUnavailable(Exception):
    """The graph cannot be provided right now (for example the database is not migrated)."""


@dataclass(frozen=True)
class GraphSnapshot:
    """An immutable graph plus the SQLite revision it reflects."""

    graph: nx.MultiDiGraph
    revision: int
    built_at: str
    source: str  # "rebuild" | "file"


@dataclass
class _State:
    snapshot: GraphSnapshot | None = None
    persisted_revision: int | None = None
    persist_error: str | None = None
    init_report: dict = field(default_factory=dict)
    stats: tuple[int, dict] | None = None  # (revision, stats)


class GraphService:
    """Owns the graph snapshot. Queries read ``snapshot()``; writers just bump SQLite."""

    def __init__(self, factory: sessionmaker[Session], path: Path) -> None:
        self._factory = factory
        self._path = path
        self._state = _State()
        self._lock = threading.RLock()

    # -- lifecycle ----------------------------------------------------------------------

    def initialize(self) -> dict:
        """Startup: adopt the file if it is valid and current, otherwise rebuild from SQLite."""
        with self._lock:
            try:
                db_revision = self._db_revision()
            except GraphUnavailable as exc:
                self._state.init_report = {"source": "unavailable", "reason": str(exc)}
                logger.error("Graph not initialized: %s", exc)
                return self._state.init_report

            try:
                graph, revision, meta = load_graph(self._path)
            except GraphLoadError as exc:
                reason = str(exc)
                if self._path.exists():
                    self._quarantine()
            else:
                if revision == db_revision:
                    self._state.snapshot = GraphSnapshot(
                        graph, revision, meta.get("built_at", ""), "file"
                    )
                    self._state.persisted_revision = revision
                    self._state.init_report = {"source": "file", "reason": "file valid and current"}
                    logger.info("Graph loaded from file: revision=%d", revision)
                    return self._state.init_report
                reason = f"file revision {revision} != database revision {db_revision}"

            self._rebuild(reason)
            self._state.init_report = {"source": "rebuild", "reason": reason}
            return self._state.init_report

    def _quarantine(self) -> None:
        """Keep a bad file for inspection (one copy, overwritten) instead of deleting it."""
        target = self._path.with_name(self._path.name + ".corrupt")
        try:
            os.replace(self._path, target)
            logger.warning("Moved unusable graph file to %s", target.name)
        except OSError:
            logger.exception("Could not quarantine graph file")

    # -- synchronization --------------------------------------------------------------

    def _db_revision(self) -> int:
        try:
            with self._factory() as session:
                return current_revision(session)
        except SQLAlchemyError as exc:
            raise GraphUnavailable(f"database unavailable: {type(exc).__name__}") from exc

    def sync(self, *, force: bool = False) -> bool:
        """Bring memory and file up to SQLite's revision. Returns True if a rebuild ran."""
        with self._lock:
            db_revision = self._db_revision()
            snap = self._state.snapshot
            if force or snap is None or snap.revision != db_revision:
                self._rebuild("forced" if force else "stale or missing graph")
                return True
            if self._state.persisted_revision != snap.revision:
                self._persist(snap)  # retry a previously failed write
            return False

    def rebuild(self) -> None:
        """Unconditionally rebuild from SQLite and rewrite the file."""
        self.sync(force=True)

    def _rebuild(self, reason: str) -> None:
        with self._lock, WRITE_LOCK:
            try:
                with self._factory() as session:
                    graph, revision = build_graph(session)
            except SQLAlchemyError as exc:
                raise GraphUnavailable(f"database unavailable: {type(exc).__name__}") from exc
            snap = GraphSnapshot(graph, revision, datetime.now(UTC).isoformat(), "rebuild")
            self._state.snapshot = snap
            self._state.stats = None
            logger.info(
                "Graph rebuilt (%s): revision=%d nodes=%d edges=%d",
                reason,
                revision,
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
            self._persist(snap)

    def _persist(self, snap: GraphSnapshot) -> None:
        try:
            save_graph(snap.graph, snap.revision, self._path)
        except GraphPersistenceError as exc:
            self._state.persist_error = str(exc)
            logger.error("Graph file not updated (memory graph is current): %s", exc)
        else:
            self._state.persisted_revision = snap.revision
            self._state.persist_error = None

    # -- reads ------------------------------------------------------------------------

    def snapshot(self) -> GraphSnapshot:
        """A current, immutable snapshot (rebuilds first if SQLite has moved on)."""
        snap = self._state.snapshot
        if snap is None or snap.revision != self._db_revision():
            self.sync()
            snap = self._state.snapshot
        if snap is None:
            raise GraphUnavailable("graph has not been built")
        return snap

    def stats(self) -> dict:
        """Analytics for the current revision; computed once per revision, then cached."""
        snap = self.snapshot()
        cached = self._state.stats
        if cached is not None and cached[0] == snap.revision:
            return cached[1]
        result = analytics.compute_stats(snap.graph)
        self._state.stats = (snap.revision, result)
        return result

    def status(self) -> dict:
        """Cheap health summary (no graph rebuild)."""
        snap = self._state.snapshot
        try:
            db_revision: int | None = self._db_revision()
        except GraphUnavailable:
            db_revision = None
        if snap is None:
            state = "error"
        elif db_revision is not None and snap.revision != db_revision:
            state = "stale"
        elif self._state.persist_error:
            state = "degraded"
        else:
            state = "ok"
        return {
            "state": state,
            "revision": snap.revision if snap else None,
            "database_revision": db_revision,
            "node_count": snap.graph.number_of_nodes() if snap else 0,
            "edge_count": snap.graph.number_of_edges() if snap else 0,
            "built_at": snap.built_at if snap else None,
            "file_in_sync": bool(snap) and self._state.persisted_revision == snap.revision,
            "persist_error": self._state.persist_error,
            "initialized_from": self._state.init_report.get("source"),
        }

    # -- verification -----------------------------------------------------------------

    def check_consistency(self) -> dict:
        """Compare memory and the file against a graph freshly built from SQLite."""
        snap = self.snapshot()
        with self._lock, WRITE_LOCK:
            with self._factory() as session:
                expected, revision = build_graph(session)
        memory = consistency.compare(expected, snap.graph).as_dict()
        try:
            file_graph, file_revision, _ = load_graph(self._path)
        except GraphLoadError as exc:
            file_report: dict = {"consistent": False, "error": str(exc)}
        else:
            file_report = consistency.compare(expected, file_graph).as_dict()
            file_report["revision_matches"] = file_revision == revision
            file_report["consistent"] = (
                file_report["consistent"] and file_report["revision_matches"]
            )
        return {"revision": revision, "memory": memory, "file": file_report}
