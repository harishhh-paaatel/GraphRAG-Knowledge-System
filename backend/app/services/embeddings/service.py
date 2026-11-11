"""VectorService: keeps the vector index equal to "the chunks SQLite says exist".

Truth model (mirrors the Phase 4 graph design)
-----------------------------------------------
* ``document_chunks`` (SQLite) is the only authority on *what must be indexed*.
* The index file holds the vectors plus an ID table ``(chunk_id, document_id, content_hash,
  token_count, windows)``.
* ``chunk_embeddings`` (SQLite) is a bookkeeping report of what the index contains.
* ``vector_state.revision`` is bumped in the same transaction as any chunk change; the index
  file records the revision it was synced to, so "SQLite moved on" is one integer comparison.

``sync()`` reconciles the index with SQLite *incrementally*: vectors whose chunk disappeared or
whose text changed are dropped, vectors for chunks that already match are kept untouched (no
re-embedding), and only genuinely missing chunks are embedded. An embedding-configuration
change (different model, revision, dimension, token limit, ...) discards the whole index
first: vectors from different models are never mixed.

Snapshots and availability
--------------------------
Readers hold one immutable ``IndexSnapshot`` (index + metadata together). A sync builds the
next snapshot privately (copy-on-write), validates it against the chunks it was built for, and
publishes it with a single attribute assignment. Consequences:

* a search never sees a partially updated index or an index paired with the wrong metadata;
* while a sync is running, searches keep using the previous snapshot (last known good) instead
  of waiting. They are marked ``stale`` in the response. This is only done when the old
  snapshot is still *compatible* with the current embedder; vectors from a different model are
  never served, so in that case the caller gets a retryable "rebuilding" error instead;
* if no snapshot exists yet (first build still running) searches get a retryable "building"
  error; if nothing is running the first search builds the index itself;
* a sync that fails (embedding failure, validation failure, ...) leaves the previous snapshot
  and the index file untouched and records the failure.

Other safety properties
-----------------------
* Embedding runs outside the database write lock; the revision is re-checked before applying.
* A failed file write keeps the new snapshot in memory, leaves the previous file intact, and is
  retried on the next sync.
* Corrupt / missing / stale files are rebuilt from SQLite-backed chunks.
"""

import hashlib
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.models import ChunkEmbedding, DocumentChunk
from app.models.base import utcnow
from app.services.embeddings.embedder import Embedder, EmbeddingIdentity, TextInfo
from app.services.embeddings.revision import current_vector_revision
from app.services.embeddings.snapshot import IndexSnapshot
from app.services.embeddings.vector_index import (
    FAISS_ENGINE,
    IndexCorruptError,
    IndexEngine,
    IndexEntry,
    SearchHit,
    VectorIndex,
    VectorIndexError,
)
from app.services.embeddings.vector_store import (
    IndexMeta,
    IndexPersistenceError,
    load_index,
    now_iso,
    save_index,
)
from app.services.embeddings.windowing import POLICY_ID
from app.services.extraction.persistence import WRITE_LOCK

logger = logging.getLogger(__name__)

_EMBED_GROUP = 256
_CONTENT_BATCH = 500
_MAX_ATTEMPTS = 3
_MAX_ERROR = 500


class VectorIndexUnavailable(VectorIndexError):
    """The index cannot be provided right now (database down, embedding failure, ...)."""


class VectorIndexBuilding(VectorIndexUnavailable):
    """No usable snapshot exists and another thread is (re)building it: retry shortly."""


@dataclass(frozen=True)
class ReadView:
    """What a query gets: ONE snapshot plus whether it is behind SQLite."""

    snapshot: IndexSnapshot
    stale: bool
    database_revision: int | None


@dataclass
class SyncReport:
    """What one ``sync()`` did."""

    reason: str = "up to date"
    rebuilt: bool = False
    embedded: int = 0
    removed: int = 0
    reused: int = 0
    total: int = 0
    revision: int = 0
    persisted: bool = True
    duration_ms: float = 0.0
    embed_ms: float = 0.0
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class VectorService:
    """Owns the published snapshot, its file and the rules keeping them consistent with SQLite."""

    def __init__(
        self,
        factory: sessionmaker[Session],
        path: Path,
        embedder: Embedder,
        *,
        sync_on_startup: bool = True,
        engine: IndexEngine = FAISS_ENGINE,
    ) -> None:
        self._factory = factory
        self._path = path
        self.embedder = embedder
        self.engine = engine
        self._sync_on_startup = sync_on_startup
        self._sync_lock = threading.RLock()  # serializes syncs; searches never block on it
        self._snapshot: IndexSnapshot | None = None  # replaced only by one atomic assignment
        self._syncing = False
        self._persisted_revision: int | None = None
        self._persist_error: str | None = None
        self._last_error: str | None = None
        self._invalid_reason: str | None = None
        self._identity_verified = False
        self._init_report: dict = {}
        self._last_sync: dict | None = None

    # -- lifecycle ----------------------------------------------------------------------

    def initialize(self) -> dict:
        """Startup: adopt the file if valid and compatible, then reconcile with SQLite.

        Never raises: a problem becomes a recorded error and the app keeps running.
        """
        with self._sync_lock:
            try:
                self._db_revision()
            except VectorIndexUnavailable as exc:
                self._init_report = {"source": "unavailable", "reason": str(exc)}
                logger.error("Vector index not initialized: %s", exc)
                return self._init_report

            source, reason = "rebuild", "no index file yet"
            try:
                index, meta = load_index(self._path, self.engine)
                candidate = IndexSnapshot(index, meta)
                candidate.validate()  # a snapshot is adopted only if it is self-consistent
            except IndexCorruptError as exc:
                source, reason = "rebuild", str(exc)
                if self._path.exists():
                    self._quarantine()
            else:
                if meta.config_fingerprint != self.embedder.config_fingerprint:
                    source = "rebuild"
                    reason = "embedding configuration changed since the index was built"
                    self._invalid_reason = reason
                else:
                    self._snapshot = candidate
                    self._persisted_revision = meta.revision
                    source, reason = "file", "index file valid and compatible"
            self._init_report = {"source": source, "reason": reason}

            if self._sync_on_startup:
                try:
                    report = self.sync()
                    self._init_report["sync"] = report.as_dict()
                except Exception as exc:
                    self._last_error = self._short(exc)
                    self._init_report["sync_error"] = self._last_error
                    logger.error("Vector sync at startup failed: %s", self._last_error)
            return self._init_report

    def _quarantine(self) -> None:
        """Keep an unusable file for inspection (one copy, overwritten) rather than deleting it."""
        target = self._path.with_name(self._path.name + ".corrupt")
        try:
            os.replace(self._path, target)
            logger.warning("Moved unusable vector index to %s", target.name)
        except OSError:
            logger.exception("Could not quarantine vector index file")

    # -- helpers ------------------------------------------------------------------------

    @staticmethod
    def _short(exc: BaseException) -> str:
        return f"{type(exc).__name__}: {exc}"[:_MAX_ERROR]

    def _db_revision(self) -> int:
        try:
            with self._factory() as session:
                return current_vector_revision(session)
        except SQLAlchemyError as exc:
            raise VectorIndexUnavailable(f"database unavailable: {type(exc).__name__}") from exc

    def _load_desired(self) -> dict[str, tuple[str, str]]:
        """``{chunk_id: (document_id, content_hash)}`` for every chunk in SQLite (streamed)."""
        desired: dict[str, tuple[str, str]] = {}
        try:
            with self._factory() as session:
                rows = session.execute(
                    select(DocumentChunk.id, DocumentChunk.document_id, DocumentChunk.content)
                    .order_by(DocumentChunk.id)
                    .execution_options(yield_per=500)
                )
                for chunk_id, document_id, text in rows:
                    desired[chunk_id] = (document_id, content_hash(text))
        except SQLAlchemyError as exc:
            raise VectorIndexUnavailable(f"database unavailable: {type(exc).__name__}") from exc
        return desired

    def _load_contents(self, chunk_ids: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        with self._factory() as session:
            for i in range(0, len(chunk_ids), _CONTENT_BATCH):
                batch = chunk_ids[i : i + _CONTENT_BATCH]
                for chunk_id, text in session.execute(
                    select(DocumentChunk.id, DocumentChunk.content).where(
                        DocumentChunk.id.in_(batch)
                    )
                ):
                    out[chunk_id] = text
        return out

    # -- synchronization ----------------------------------------------------------------

    def sync(self, *, force: bool = False) -> SyncReport:
        """Make the index equal to SQLite's chunks. ``force`` recomputes every vector."""
        with self._sync_lock:
            self._syncing = True
            try:
                for _ in range(_MAX_ATTEMPTS):
                    report = self._sync_once(force)
                    if report is not None:
                        return report
                raise VectorIndexUnavailable("chunks kept changing during synchronization; retry")
            finally:
                self._syncing = False

    def rebuild(self) -> SyncReport:
        """Discard the index and recompute every vector from SQLite-backed chunks."""
        return self.sync(force=True)

    def _incompatibility(self, snap: IndexSnapshot | None) -> str | None:
        """Why this snapshot may not be used with the current embedder (None = compatible)."""
        if snap is None:
            return None
        meta = snap.meta
        if meta.config_fingerprint != self.embedder.config_fingerprint:
            return "embedding configuration changed"
        if snap.index is not None and self.embedder.is_loaded:
            ident = self.embedder.identity()
            if ident.dimension != snap.index.dimension:
                return "embedding dimension changed"
            if ident.revision and meta.model_revision and ident.revision != meta.model_revision:
                return "embedding model revision changed"
            if (
                ident.max_tokens is not None
                and meta.max_tokens is not None
                and ident.max_tokens != meta.max_tokens
            ):
                return "embedding token limit changed"
        return None

    def _still_current(self, entries: list[IndexEntry]) -> list[int]:
        """Positions of ``entries`` whose chunk still exists with the same content hash."""
        contents = self._load_contents([e.chunk_id for e in entries])
        return [
            i
            for i, e in enumerate(entries)
            if e.chunk_id in contents and content_hash(contents[e.chunk_id]) == e.content_hash
        ]

    def _sync_once(self, force: bool) -> SyncReport | None:
        started = time.perf_counter()
        db_revision = self._db_revision()
        desired = self._load_desired()
        snap = self._snapshot
        index = snap.index if snap else None
        meta = snap.meta if snap else None

        reason = "forced rebuild" if force else (self._invalid_reason or None)
        reason = reason or self._incompatibility(snap)
        discard = reason is not None
        base = {} if (discard or index is None) else {e.chunk_id: e for e in index.entries()}
        remove = [
            cid for cid, e in base.items() if desired.get(cid) != (e.document_id, e.content_hash)
        ]
        removed_set = set(remove)
        kept = {cid for cid in base if cid not in removed_set}
        missing = [cid for cid in desired if cid not in kept]

        report = SyncReport(
            reason=reason or "reconciled with SQLite",
            rebuilt=discard,
            removed=len(remove),
            reused=len(kept),
            total=len(desired),
            revision=db_revision,
        )

        if not discard and not remove and not missing and meta is not None:
            if meta.revision == db_revision and self._persisted_revision == db_revision:
                report.reason = "up to date"
                report.duration_ms = (time.perf_counter() - started) * 1000
                self._write_bookkeeping(desired, snap)  # repairs lost bookkeeping rows
                return report
            # Chunks were rewritten with identical content (e.g. a reprocess): nothing to embed.
            report.reason = "revision advanced; content unchanged"

        vectors: np.ndarray | None = None
        identity: EmbeddingIdentity | None = None
        entries: list[IndexEntry] = []
        failures: dict[str, str] = {}
        if missing:
            contents = self._load_contents(missing)
            missing = [cid for cid in missing if cid in contents]  # vanished chunks: next retry
            e0 = time.perf_counter()
            try:
                ok, vectors, infos, failures = self._embed_isolating_failures(
                    missing, [contents[cid] for cid in missing]
                )
                identity = self.embedder.identity() if ok else None
            except Exception as exc:  # the embedding backend itself is unavailable
                msg = self._short(exc)
                self._record_failure(dict.fromkeys(missing, msg), desired)
                self._last_error = msg
                raise VectorIndexUnavailable(f"embedding failed: {msg}") from exc
            if not ok:  # every chunk failed individually
                self._record_failure(failures, desired)
                self._last_error = self._failure_summary(failures)
                raise VectorIndexUnavailable(f"embedding failed: {self._last_error}")
            entries = [
                IndexEntry(missing[i], *desired[missing[i]], info.token_count, info.windows)
                for i, info in zip(ok, infos, strict=True)
            ]
            report.embed_ms = (time.perf_counter() - e0) * 1000
            report.embedded = len(entries)
            oversized = sum(1 for e in entries if e.windows > 1)
            if oversized:
                report.details["oversized_chunks_pooled"] = oversized
            if failures:
                self._record_failure(failures, desired)
                report.details["failed_chunks"] = len(failures)
            if not discard and snap is not None:
                late = self._incompatibility(snap)  # model identity became knowable now
                if late:
                    self._invalid_reason = late
                    return None
            if self._db_revision() != db_revision:
                # SQLite moved while we were embedding. Never throw finished (expensive) work
                # away: keep only the vectors whose chunk text is still exactly what we embedded.
                # The snapshot is recorded at the OLDER revision, so the next sync reconciles
                # whatever changed in the meantime.
                keep_rows = self._still_current(entries)
                entries = [entries[i] for i in keep_rows]
                vectors = vectors[keep_rows] if vectors is not None else vectors
                report.embedded = len(entries)
                report.details["raced_with_writes"] = True

        new_index = self._apply(index, discard, remove, entries, vectors)
        model = identity or (self.embedder.identity() if self.embedder.is_loaded else None)
        new_meta = IndexMeta(
            revision=db_revision,
            model_id=model.model_id if model else self.embedder.model_id,
            model_revision=model.revision
            if model
            else (meta.model_revision if meta and not discard else None),
            dimension=new_index.dimension if new_index else None,
            config_fingerprint=self.embedder.config_fingerprint,
            built_at=now_iso(),
            max_tokens=model.max_tokens
            if model
            else (meta.max_tokens if meta and not discard else None),
        )

        # Validate privately, publish atomically. A snapshot that is not exactly "the chunks
        # we meant to index" is never visible to readers; the previous one stays in service.
        expected = {cid: desired[cid] for cid in kept}
        expected.update({e.chunk_id: (e.document_id, e.content_hash) for e in entries})
        candidate = IndexSnapshot(new_index, new_meta)
        try:
            candidate.validate(expected)
        except IndexCorruptError as exc:
            self._last_error = f"new index failed validation and was NOT published: {exc}"[
                :_MAX_ERROR
            ]
            logger.error(self._last_error)
            raise VectorIndexUnavailable(self._last_error) from exc
        self._snapshot = candidate
        self._invalid_reason = None
        self._identity_verified = self._identity_verified or identity is not None
        self._last_error = self._failure_summary(failures) if failures else None

        try:
            save_index(new_index, new_meta, self._path, self.engine)
        except IndexPersistenceError as exc:
            self._persist_error = str(exc)
            report.persisted = False
            logger.error("Vector index file not updated (memory snapshot is current): %s", exc)
        else:
            self._persisted_revision = db_revision
            self._persist_error = None
        self._write_bookkeeping(desired, candidate)

        report.duration_ms = (time.perf_counter() - started) * 1000
        self._last_sync = report.as_dict()
        logger.info(
            "Vector index synced: %s embedded=%d removed=%d reused=%d total=%d revision=%d",
            report.reason,
            report.embedded,
            report.removed,
            report.reused,
            report.total,
            db_revision,
        )
        return report

    def _apply(
        self,
        index: VectorIndex | None,
        discard: bool,
        remove: list[str],
        entries: list[IndexEntry],
        vectors: np.ndarray | None,
    ) -> VectorIndex | None:
        """Build the next index without touching the published one (copy-on-write)."""
        if discard or index is None:
            if not entries:
                return None
            assert vectors is not None
            fresh = self.engine.create(vectors.shape[1])
            fresh.add(entries, vectors)
            return fresh
        if not remove and not entries:
            return index
        nxt = index.clone()
        nxt.remove(remove)
        if entries:
            assert vectors is not None
            nxt.add(entries, vectors)
        return nxt if nxt.count else None

    # -- bookkeeping --------------------------------------------------------------------

    def _write_bookkeeping(self, desired: dict[str, tuple[str, str]], snap: IndexSnapshot) -> None:
        """Make ``chunk_embeddings`` describe the snapshot. Best effort: it is a report."""
        indexed = {e.chunk_id: e for e in snap.entries()}
        meta = snap.meta
        try:
            with WRITE_LOCK, self._factory() as session, session.begin():
                present = set(session.scalars(select(DocumentChunk.id)))
                rows = {r.chunk_id: r for r in session.scalars(select(ChunkEmbedding))}
                now = utcnow()
                for cid, (doc_id, h) in desired.items():
                    entry = indexed.get(cid)
                    if cid not in present or entry is None:
                        continue
                    want = {
                        "document_id": doc_id,
                        "content_hash": h,
                        "model_id": meta.model_id,
                        "model_revision": meta.model_revision,
                        "dimension": meta.dimension or 0,
                        "config_fingerprint": meta.config_fingerprint,
                        "token_count": entry.token_count or None,
                        "window_count": entry.windows,
                        "status": "indexed",
                        "error_message": None,
                    }
                    row = rows.get(cid)
                    if row is None:
                        session.add(
                            ChunkEmbedding(chunk_id=cid, created_at=now, updated_at=now, **want)
                        )
                    elif any(getattr(row, k) != v for k, v in want.items()):
                        for k, v in want.items():
                            setattr(row, k, v)
                        row.updated_at = now
                for cid, row in rows.items():
                    if cid not in desired:
                        session.delete(row)
        except SQLAlchemyError:
            logger.exception("Could not update chunk_embeddings bookkeeping")

    def _embed_isolating_failures(
        self, ids: list[str], texts: list[str]
    ) -> tuple[list[int], np.ndarray | None, list[TextInfo], dict[str, str]]:
        """Embed in groups. If a group fails while the backend is healthy, retry its chunks one
        by one so a single bad chunk cannot block every other chunk from being indexed.

        Returns (positions that succeeded, their vectors in the same order, their token/window
        info, {chunk_id: error}). A failure while the backend is not even loaded (e.g. the model
        cannot be fetched) is re-raised: retrying chunk by chunk would only repeat it.
        """
        parts: list[np.ndarray] = []
        infos: list[TextInfo] = []
        ok: list[int] = []
        failures: dict[str, str] = {}
        for start in range(0, len(texts), _EMBED_GROUP):
            group = texts[start : start + _EMBED_GROUP]
            try:
                group_infos = self.embedder.describe(group)
                parts.append(self.embedder.embed(group))
                infos.extend(group_infos)
                ok.extend(range(start, start + len(group)))
            except Exception:
                if not self.embedder.is_loaded:
                    raise
                for offset, text in enumerate(group):
                    try:
                        one_info = self.embedder.describe([text])
                        parts.append(self.embedder.embed([text]))
                        infos.extend(one_info)
                        ok.append(start + offset)
                    except Exception as exc:
                        failures[ids[start + offset]] = self._short(exc)
        return ok, (np.vstack(parts) if parts else None), infos, failures

    @staticmethod
    def _failure_summary(failures: dict[str, str]) -> str:
        first = next(iter(failures.values()))
        return f"{len(failures)} chunk(s) failed to embed; first error: {first}"[:_MAX_ERROR]

    def _record_failure(
        self, failures: dict[str, str], desired: dict[str, tuple[str, str]]
    ) -> None:
        """Remember which chunks could not be embedded; the index and its file are not touched."""
        logger.error("Embedding failed for %d chunk(s)", len(failures))
        chunk_ids = list(failures)
        try:
            with WRITE_LOCK, self._factory() as session, session.begin():
                present = set(session.scalars(select(DocumentChunk.id)))
                rows = {r.chunk_id: r for r in session.scalars(select(ChunkEmbedding))}
                now = utcnow()
                for cid in chunk_ids:
                    if cid not in present or cid not in desired:
                        continue
                    row = rows.get(cid)
                    if row is None:
                        session.add(
                            ChunkEmbedding(
                                chunk_id=cid,
                                document_id=desired[cid][0],
                                content_hash=desired[cid][1],
                                model_id=self.embedder.model_id,
                                model_revision=None,
                                dimension=0,
                                config_fingerprint=self.embedder.config_fingerprint,
                                status="failed",
                                error_message=failures[cid],
                                created_at=now,
                                updated_at=now,
                            )
                        )
                    elif row.status != "indexed" or row.content_hash != desired[cid][1]:
                        row.status, row.error_message, row.updated_at = "failed", failures[cid], now
                        row.content_hash = desired[cid][1]
                for row in rows.values():  # vectors from another configuration are stale
                    if row.config_fingerprint != self.embedder.config_fingerprint:
                        row.status, row.updated_at = "stale", now
        except SQLAlchemyError:
            logger.exception("Could not record embedding failure")

    # -- reads --------------------------------------------------------------------------

    def ensure_fresh(self) -> ReadView:
        """Return the snapshot a query should use, synchronizing first when that is cheap/safe.

        * current and compatible              -> that snapshot;
        * behind SQLite, no sync running      -> this call syncs, then returns the new snapshot;
        * behind SQLite, a sync is running    -> the previous snapshot (``stale=True``), but only
          if it is still compatible with the current embedder (last known good);
        * incompatible / none, a sync running -> ``VectorIndexBuilding`` (retryable);
        * database or embedding failure       -> ``VectorIndexUnavailable``.
        """
        db_rev = self._db_revision()
        snap = self._snapshot
        usable = (
            snap is not None
            and self._invalid_reason is None
            and snap.meta.config_fingerprint == self.embedder.config_fingerprint
        )
        if usable and snap.meta.revision == db_rev:
            if self._identity_verified or snap.index is None:
                return ReadView(snap, False, db_rev)
            # First use in this process: loading the model reveals dimension/revision drift.
            self.embedder.identity()
            self._identity_verified = True
            reason = self._incompatibility(snap)
            if reason is None:
                return ReadView(snap, False, db_rev)
            self._invalid_reason = reason
            usable = False

        if self._sync_lock.acquire(blocking=False):
            try:
                self.sync()
            finally:
                self._sync_lock.release()
            fresh = self._snapshot
            if fresh is None:  # pragma: no cover - sync either publishes or raises
                raise VectorIndexUnavailable("index has not been built")
            return ReadView(fresh, fresh.meta.revision != self._db_revision(), db_rev)

        if usable:
            assert snap is not None
            return ReadView(snap, True, db_rev)
        raise VectorIndexBuilding(
            "the vector index is being (re)built; searches resume as soon as it is ready"
        )

    def snapshot(self) -> IndexSnapshot | None:
        """The currently published snapshot (``None`` before the first build)."""
        return self._snapshot

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        *,
        document_ids=None,
        exclude_document_ids=None,
    ) -> list[SearchHit]:
        """Search the current snapshot. No snapshot or an empty index -> empty list."""
        snap = self._snapshot
        if snap is None:
            return []
        return snap.search(
            query_vector,
            top_k,
            document_ids=document_ids,
            exclude_document_ids=exclude_document_ids,
        )

    @property
    def session_factory(self) -> sessionmaker[Session]:
        return self._factory

    @property
    def index(self) -> VectorIndex | None:
        snap = self._snapshot
        return snap.index if snap else None

    @property
    def snapshot_meta(self) -> IndexMeta | None:
        snap = self._snapshot
        return snap.meta if snap else None

    @property
    def index_count(self) -> int:
        snap = self._snapshot
        return snap.count if snap else 0

    def indexed_entries(self) -> list[IndexEntry]:
        snap = self._snapshot
        return snap.entries() if snap else []

    # -- introspection ------------------------------------------------------------------

    def status(self) -> dict:
        """Cheap health summary (no sync, no model load). Reads one snapshot."""
        try:
            db_revision: int | None = self._db_revision()
        except VectorIndexUnavailable:
            db_revision = None
        snap = self._snapshot
        meta = snap.meta if snap else None
        if meta is None and db_revision is None:
            state = "error"
        elif meta is None and self._syncing:
            state = "building"
        elif self._last_error or self._persist_error:
            state = "degraded"
        elif (
            meta is None
            or self._invalid_reason
            or (db_revision is not None and meta.revision != db_revision)
            or meta.config_fingerprint != self.embedder.config_fingerprint
        ):
            state = "stale"
        else:
            state = "ok"
        return {
            "state": state,
            "vector_count": snap.count if snap else 0,
            "dimension": meta.dimension if meta else None,
            "model_id": meta.model_id if meta else self.embedder.model_id,
            "model_revision": meta.model_revision if meta else None,
            "max_tokens": meta.max_tokens if meta else None,
            "oversize_policy": POLICY_ID,
            "engine": self.engine.name,
            "config_fingerprint": meta.config_fingerprint
            if meta
            else self.embedder.config_fingerprint,
            "current_config_fingerprint": self.embedder.config_fingerprint,
            "index_revision": meta.revision if meta else None,
            "database_revision": db_revision,
            "built_at": meta.built_at if meta else None,
            "file_in_sync": bool(meta) and self._persisted_revision == meta.revision,
            "sync_in_progress": self._syncing,
            "persist_error": self._persist_error,
            "last_error": self._last_error,
            "invalid_reason": self._invalid_reason,
            "embedder_loaded": self.embedder.is_loaded,
            "initialized_from": self._init_report.get("source"),
        }

    def stats(self) -> dict:
        """``status()`` plus bookkeeping counts and the index file size."""
        info = self.status()
        try:
            with self._factory() as session:
                counts = dict(
                    session.execute(
                        select(ChunkEmbedding.status, func.count()).group_by(ChunkEmbedding.status)
                    ).all()
                )
                total_chunks = session.scalar(select(func.count()).select_from(DocumentChunk)) or 0
                oversized = (
                    session.scalar(
                        select(func.count())
                        .select_from(ChunkEmbedding)
                        .where(ChunkEmbedding.window_count > 1)
                    )
                    or 0
                )
        except SQLAlchemyError:
            counts, total_chunks, oversized = {}, None, None
        info["chunks_in_database"] = total_chunks
        info["embedding_status_counts"] = {
            s: counts.get(s, 0) for s in ("indexed", "failed", "stale")
        }
        info["oversized_chunks_pooled"] = oversized
        info["file_size_bytes"] = self._path.stat().st_size if self._path.exists() else None
        info["last_sync"] = self._last_sync
        return info

    def check_consistency(self) -> dict:
        """Compare SQLite chunks, the published snapshot, the file and the bookkeeping table."""
        snap = self._snapshot
        desired = self._load_desired()
        entries = {e.chunk_id: e for e in (snap.entries() if snap else [])}
        with self._factory() as session:
            rows = {r.chunk_id: r for r in session.scalars(select(ChunkEmbedding))}
            db_revision = current_vector_revision(session)
        mismatch = sorted(
            cid
            for cid, e in entries.items()
            if cid in desired and desired[cid] != (e.document_id, e.content_hash)
        )
        memory = {
            "missing_from_index": sorted(set(desired) - set(entries)),
            "extra_in_index": sorted(set(entries) - set(desired)),
            "content_mismatch": mismatch,
            "bookkeeping_missing": sorted(c for c in entries if c in desired and c not in rows),
            "bookkeeping_unindexed": sorted(
                c for c, r in rows.items() if r.status == "indexed" and c not in entries
            ),
            "model_mismatch": bool(
                snap and snap.meta.config_fingerprint != self.embedder.config_fingerprint
            ),
        }
        memory["consistent"] = not any(memory[k] for k in memory if k != "consistent")
        try:
            file_index, file_meta = load_index(self._path, self.engine)
        except IndexCorruptError as exc:
            file_report: dict = {"consistent": False, "error": str(exc)}
        else:
            file_entries = {e.chunk_id: e for e in (file_index.entries() if file_index else [])}
            file_report = {
                "matches_memory": file_entries == entries,
                "revision_matches": file_meta.revision == db_revision,
                "fingerprint_matches": file_meta.config_fingerprint
                == self.embedder.config_fingerprint,
            }
            file_report["consistent"] = all(file_report.values())
        return {"database_revision": db_revision, "memory": memory, "file": file_report}
