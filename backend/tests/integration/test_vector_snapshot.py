"""Atomic snapshot publication and last-known-good availability during a sync."""

import dataclasses
import threading
import time

import numpy as np
import pytest

from app.models import ChunkEmbedding, DocumentChunk
from app.services.embeddings import vector_store
from app.services.embeddings.embedder import HashingEmbedder
from app.services.embeddings.service import (
    VectorIndexBuilding,
    VectorIndexUnavailable,
    VectorService,
)
from app.services.embeddings.snapshot import IndexSnapshot
from app.services.embeddings.vector_index import IndexCorruptError
from app.services.retrieval.vector_search import search_vectors
from tests.vector_helpers import CORPUS, BlockingEmbedder, ControlledEmbedder, add_doc


@pytest.fixture
def make_service(app, tmp_path):
    def _make(embedder=None, sync_on_startup=True):
        return VectorService(
            app.state.session_factory,
            tmp_path / "vec" / "index.bin",
            embedder or HashingEmbedder(),
            sync_on_startup=sync_on_startup,
        )

    return _make


def assert_consistent(snap: IndexSnapshot) -> None:
    """The invariants every published snapshot must satisfy."""
    assert snap.meta.dimension == (snap.index.dimension if snap.index else None)
    assert len(snap.entries()) == snap.count
    ids = [e.chunk_id for e in snap.entries()]
    assert len(ids) == len(set(ids))


def run_in_thread(fn):
    t = threading.Thread(target=fn)
    t.start()
    return t


# ---- the snapshot object --------------------------------------------------------------------


def test_published_snapshot_is_immutable_and_consistent(session, make_service):
    add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    snap = svc.snapshot()
    assert_consistent(snap)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.index = None
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.meta = None


def test_every_sync_publishes_a_new_object_and_never_mutates_the_old_one(session, make_service):
    add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    old = svc.snapshot()
    old_entries, old_revision = old.entries(), old.meta.revision
    add_doc(session, ["Kubernetes schedules containers."])
    svc.sync()
    new = svc.snapshot()
    assert new is not old and new.count == 5
    assert (old.entries(), old.meta.revision, old.count) == (old_entries, old_revision, 4)


# ---- last-known-good availability -----------------------------------------------------------


def test_searches_use_the_previous_snapshot_while_a_sync_is_embedding(session, make_service):
    _, old_ids = add_doc(session, CORPUS)
    embedder = BlockingEmbedder()
    svc = make_service(embedder)
    svc.sync()
    before = svc.snapshot()
    _, slow_ids = add_doc(session, ["SLOWDOC quantum gardening for beginners"])

    syncing = run_in_thread(svc.sync)  # blocks inside embed() for the new chunk
    try:
        assert embedder.entered.wait(10), "the sync never reached the embedder"
        assert svc.status()["sync_in_progress"] is True

        started = time.perf_counter()
        outcome = search_vectors(
            svc, svc.session_factory, "web framework for building APIs", top_k=3
        )
        elapsed = time.perf_counter() - started

        assert elapsed < 5, "search waited for the sync instead of using the previous index"
        assert outcome.index_stale is True
        assert outcome.index_revision == before.meta.revision < outcome.database_revision
        assert outcome.indexed_vectors == 4
        assert outcome.results and {r.chunk_id for r in outcome.results} <= set(old_ids)
        assert not set(slow_ids) & {r.chunk_id for r in outcome.results}
        assert svc.snapshot() is before  # nothing was published mid-sync
    finally:
        embedder.release.set()
        syncing.join(15)

    after = svc.snapshot()
    assert after is not before and after.count == 5
    fresh = search_vectors(svc, svc.session_factory, "quantum gardening", top_k=1)
    assert fresh.index_stale is False and fresh.results[0].chunk_id == slow_ids[0]


def test_many_concurrent_searches_all_succeed_during_a_sync(session, make_service):
    add_doc(session, CORPUS)
    embedder = BlockingEmbedder()
    svc = make_service(embedder)
    svc.sync()
    add_doc(session, ["SLOWDOC another brand new document"])
    results, errors = [], []

    def searcher():
        try:
            out = search_vectors(svc, svc.session_factory, "capital city of Germany", top_k=2)
            results.append(out)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    syncing = run_in_thread(svc.sync)
    try:
        assert embedder.entered.wait(10)
        threads = [run_in_thread(searcher) for _ in range(8)]
        for t in threads:
            t.join(15)
    finally:
        embedder.release.set()
        syncing.join(15)
    assert errors == [] and len(results) == 8
    assert all(r.index_stale and r.results for r in results)


def test_an_incompatible_snapshot_is_never_served_while_the_index_rebuilds(session, make_service):
    add_doc(session, CORPUS)
    svc = make_service(HashingEmbedder(256))
    svc.sync()
    old = svc.snapshot()

    new_embedder = BlockingEmbedder(64)  # a different embedding configuration
    svc.embedder = new_embedder
    add_doc(session, ["SLOWDOC forces the rebuild to wait"])
    syncing = run_in_thread(svc.sync)
    try:
        assert new_embedder.entered.wait(10)
        with pytest.raises(VectorIndexBuilding, match="being"):
            search_vectors(svc, svc.session_factory, "web framework", top_k=3)
        # The old, now-incompatible snapshot stays published until the rebuild is validated...
        assert svc.snapshot() is old
    finally:
        new_embedder.release.set()
        syncing.join(15)
    # ...and is replaced as a whole by a snapshot of the new model.
    assert svc.snapshot() is not old and svc.snapshot().index.dimension == 64
    assert search_vectors(svc, svc.session_factory, "web framework", top_k=3).results


def test_first_build_in_progress_is_reported_as_building_not_as_an_empty_index(
    session, make_service
):
    add_doc(session, ["SLOWDOC the very first chunk"])
    embedder = BlockingEmbedder()
    svc = make_service(embedder, sync_on_startup=False)
    assert svc.snapshot() is None

    syncing = run_in_thread(svc.sync)
    try:
        assert embedder.entered.wait(10)
        assert svc.status()["state"] == "building"
        with pytest.raises(VectorIndexBuilding):
            search_vectors(svc, svc.session_factory, "first chunk", top_k=1)
    finally:
        embedder.release.set()
        syncing.join(15)
    assert search_vectors(svc, svc.session_factory, "first chunk", top_k=1).results


def test_with_no_snapshot_and_nothing_running_the_first_search_builds_the_index(
    session, make_service
):
    add_doc(session, CORPUS)
    svc = make_service(HashingEmbedder(), sync_on_startup=False)
    assert svc.snapshot() is None
    out = search_vectors(svc, svc.session_factory, "capital of Germany", top_k=2)
    assert svc.snapshot() is not None and out.results and out.index_stale is False


# ---- failed or invalid builds never replace the published snapshot -------------------------


def test_a_failed_embedding_leaves_the_published_snapshot_and_file_untouched(
    session, make_service, tmp_path
):
    add_doc(session, CORPUS)
    embedder = ControlledEmbedder()
    svc = make_service(embedder)
    svc.sync()
    before, file_before = svc.snapshot(), (tmp_path / "vec" / "index.bin").read_bytes()

    add_doc(session, ["POISON one.", "POISON two."])
    embedder.fail_on = {"POISON"}
    with pytest.raises(VectorIndexUnavailable):
        svc.sync()

    assert svc.snapshot() is before
    assert (tmp_path / "vec" / "index.bin").read_bytes() == file_before
    assert_consistent(svc.snapshot())


def test_a_snapshot_that_fails_validation_is_never_published(
    session, make_service, tmp_path, monkeypatch
):
    add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    before, file_before = svc.snapshot(), (tmp_path / "vec" / "index.bin").read_bytes()
    original = IndexSnapshot.validate

    def corrupting_validate(self, expected=None):
        if expected is not None:  # the candidate being published, not a loaded file
            raise IndexCorruptError("simulated internal inconsistency")
        return original(self, expected)

    monkeypatch.setattr(IndexSnapshot, "validate", corrupting_validate)
    add_doc(session, ["Redis stores keys in memory."])
    with pytest.raises(VectorIndexUnavailable, match="NOT published"):
        svc.sync()

    assert svc.snapshot() is before  # previous consistent snapshot still serving
    assert (tmp_path / "vec" / "index.bin").read_bytes() == file_before
    status = svc.status()
    assert status["state"] == "degraded" and "NOT published" in status["last_error"]


def test_startup_rejects_a_file_whose_snapshot_is_inconsistent(
    session, make_service, tmp_path, monkeypatch
):
    add_doc(session, CORPUS)
    make_service().sync()
    original = IndexSnapshot.validate

    def rejecting_validate(self, expected=None):
        if expected is None:  # the snapshot assembled from the file
            raise IndexCorruptError("simulated: metadata disagrees with the index")
        return original(self, expected)

    monkeypatch.setattr(IndexSnapshot, "validate", rejecting_validate)
    embedder = HashingEmbedder()
    svc = make_service(embedder)
    report = svc.initialize()

    assert report["source"] == "rebuild" and "disagrees" in report["reason"]
    assert (tmp_path / "vec" / "index.bin.corrupt").exists()  # kept for inspection
    assert embedder.texts_embedded == 4  # rebuilt from SQLite-backed chunks
    assert_consistent(svc.snapshot())


# ---- regression: index and metadata always belong together -----------------------------------


def test_readers_always_see_an_index_paired_with_its_own_metadata_during_churn(
    session, make_service
):
    add_doc(session, CORPUS)
    svc = make_service(HashingEmbedder(256))
    svc.sync()
    fingerprints = {d: HashingEmbedder(d).config_fingerprint for d in (256, 64)}
    stop, errors, dims_seen = threading.Event(), [], set()

    def reader():
        while not stop.is_set():
            snap = svc.snapshot()  # ONE reference, then only that reference is used
            try:
                assert_consistent(snap)
                assert snap.meta.config_fingerprint == fingerprints[snap.meta.dimension]
                dims_seen.add(snap.meta.dimension)
            except AssertionError as exc:
                errors.append(repr(exc))
            time.sleep(0.001)  # yield the GIL so the writer is not starved

    readers = [run_in_thread(reader) for _ in range(3)]
    try:
        for i in range(12):  # alternate the embedding model while documents keep arriving
            svc.embedder = HashingEmbedder(64 if i % 2 == 0 else 256)
            add_doc(session, [f"churn document number {i} about topic {i}"])
            svc.sync()
    finally:
        stop.set()
        for t in readers:
            t.join(10)
    assert errors == []
    assert dims_seen == {256, 64}


def test_snapshot_never_references_a_chunk_that_sqlite_does_not_know(session, make_service):
    doc, ids = add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    assert {e.chunk_id for e in svc.snapshot().entries()} == set(ids)
    assert session.query(DocumentChunk).count() == 4 and session.query(ChunkEmbedding).count() == 4
    loaded, meta = vector_store.load_index(svc._path)
    assert IndexSnapshot(loaded, meta).count == 4
    assert np.isfinite(loaded.vector(ids[0])).all()
