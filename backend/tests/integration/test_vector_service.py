"""VectorService: SQLite <-> FAISS consistency, incremental sync, recovery, model changes."""

import threading

import numpy as np
import pytest
from sqlalchemy import delete, select, update

from app.models import ChunkEmbedding, DocumentChunk
from app.services.embeddings import vector_store
from app.services.embeddings.embedder import EmbeddingError, HashingEmbedder
from app.services.embeddings.revision import current_vector_revision
from app.services.embeddings.service import (
    VectorIndexUnavailable,
    VectorService,
    content_hash,
)
from app.services.embeddings.vector_store import IndexPersistenceError
from tests.vector_helpers import (
    CORPUS,
    ControlledEmbedder,
    add_doc,
    remove_doc,
    set_chunk_text,
)


@pytest.fixture
def make_service(app, tmp_path):
    def _make(embedder=None, path=None, sync_on_startup=True):
        return VectorService(
            app.state.session_factory,
            path or tmp_path / "vec" / "index.bin",
            embedder or HashingEmbedder(),
            sync_on_startup=sync_on_startup,
        )

    return _make


def rows(session):
    session.expire_all()
    return {r.chunk_id: r for r in session.scalars(select(ChunkEmbedding))}


def entry_ids(svc):
    return sorted(e.chunk_id for e in svc.indexed_entries())


# ---- initial indexing ---------------------------------------------------------------------


def test_sync_embeds_every_chunk_and_records_identity_metadata(session, make_service):
    doc, ids = add_doc(session, CORPUS)
    svc = make_service()
    report = svc.sync()

    assert (report.embedded, report.removed, report.reused, report.total) == (4, 0, 0, 4)
    assert entry_ids(svc) == sorted(ids)
    got = rows(session)
    assert set(got) == set(ids)
    emb = svc.embedder.identity()
    for i, cid in enumerate(ids):
        r = got[cid]
        assert r.document_id == doc.id and r.status == "indexed"
        assert r.content_hash == content_hash(CORPUS[i])
        assert (r.model_id, r.dimension, r.config_fingerprint) == (
            emb.model_id,
            emb.dimension,
            emb.config_fingerprint,
        )
        assert r.created_at and r.updated_at and r.error_message is None
    assert svc.snapshot_meta.revision == current_vector_revision(session)


def test_vectors_are_not_stored_in_sqlite(session, make_service):
    add_doc(session, CORPUS)
    make_service().sync()
    assert {c.name for c in ChunkEmbedding.__table__.columns}.isdisjoint({"vector", "embedding"})


def test_file_is_written_and_matches_memory(session, make_service, tmp_path):
    add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    loaded, meta = vector_store.load_index(tmp_path / "vec" / "index.bin")
    assert [e.chunk_id for e in loaded.entries()] == [e.chunk_id for e in svc.indexed_entries()]
    assert meta.revision == svc.snapshot_meta.revision and meta.model_id == svc.embedder.model_id
    assert svc.status()["file_in_sync"] is True


def test_empty_corpus_persists_a_valid_empty_index(session, make_service, tmp_path):
    svc = make_service()
    report = svc.sync()
    assert report.total == 0 and svc.index_count == 0
    loaded, meta = vector_store.load_index(tmp_path / "vec" / "index.bin")
    assert loaded is None and meta.dimension is None
    assert svc.status()["state"] == "ok"


# ---- incremental behaviour: nothing is re-embedded needlessly ---------------------------------


def test_second_sync_is_a_noop_and_embeds_nothing(session, make_service):
    add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    before = svc.embedder.texts_embedded
    report = svc.sync()
    assert report.reason == "up to date" and report.embedded == 0
    assert svc.embedder.texts_embedded == before


def test_adding_a_document_embeds_only_its_chunks(session, make_service):
    add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    before = svc.embedder.texts_embedded
    new_doc, new_ids = add_doc(
        session, ["Kubernetes schedules containers.", "Docker packages apps."]
    )
    report = svc.sync()
    assert (report.embedded, report.reused) == (2, 4)
    assert svc.embedder.texts_embedded == before + 2
    assert set(new_ids) <= set(entry_ids(svc)) and svc.index_count == 6


def test_changed_chunk_text_is_detected_and_only_that_chunk_is_reembedded(session, make_service):
    _, ids = add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    before = svc.embedder.texts_embedded
    set_chunk_text(session, ids[2], "Munich is the capital of Bavaria.")
    report = svc.sync()
    assert (report.embedded, report.removed, report.reused) == (1, 1, 3)
    assert svc.embedder.texts_embedded == before + 1
    assert rows(session)[ids[2]].content_hash == content_hash("Munich is the capital of Bavaria.")
    # the index now answers for the new text, not the old one
    q = svc.embedder.embed(["Munich Bavaria capital"])[0]
    assert svc.search(q, 1)[0].entry.chunk_id == ids[2]


def test_reprocessing_identical_content_preserves_vectors_and_mappings(session, make_service):
    doc, ids = add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    before_vectors = {c: svc.index.vector(c).copy() for c in ids}
    embedded_before = svc.embedder.texts_embedded

    # What the pipeline does on a re-run: replace the document's chunks with identical ones
    # (same deterministic IDs), bumping the revision. The cascade wipes chunk_embeddings rows.
    from app.models import Document
    from app.services.embeddings.revision import bump_vector_revision

    session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
    for i, text in enumerate(CORPUS):
        session.add(
            DocumentChunk(
                id=ids[i], document_id=doc.id, chunk_index=i, content=text, chunk_metadata={}
            )
        )
    bump_vector_revision(session)
    session.commit()
    assert rows(session) == {}  # bookkeeping was cascaded away

    report = svc.sync()
    assert report.embedded == 0 and report.removed == 0 and report.reused == 4
    assert svc.embedder.texts_embedded == embedded_before  # nothing recomputed
    assert set(rows(session)) == set(ids)  # bookkeeping restored
    for c in ids:
        assert np.array_equal(svc.index.vector(c), before_vectors[c])
    assert svc.snapshot_meta.revision == current_vector_revision(session)
    assert session.get(Document, doc.id) is not None


def test_no_duplicate_vectors_after_repeated_syncs_and_reprocessing(session, make_service):
    doc, ids = add_doc(session, CORPUS)
    svc = make_service()
    for _ in range(3):
        svc.sync()
        set_chunk_text(session, ids[0], CORPUS[0])  # revision bump, same content
    svc.sync()
    assert svc.index_count == 4 and len(set(entry_ids(svc))) == 4


# ---- deletion ----------------------------------------------------------------------------


def test_deleting_a_document_removes_its_vectors_without_embedding(session, make_service, tmp_path):
    keep, keep_ids = add_doc(session, CORPUS[:2])
    drop, drop_ids = add_doc(session, CORPUS[2:])
    svc = make_service()
    svc.sync()
    before = svc.embedder.calls
    remove_doc(session, drop)
    report = svc.sync()
    assert (report.removed, report.embedded) == (2, 0)
    assert svc.embedder.calls == before  # removal needs no model
    assert entry_ids(svc) == sorted(keep_ids)
    assert set(rows(session)) == set(keep_ids)
    loaded, _ = vector_store.load_index(tmp_path / "vec" / "index.bin")
    assert sorted(e.chunk_id for e in loaded.entries()) == sorted(keep_ids)
    q = svc.embedder.embed(["capital of Germany"])[0]
    assert all(h.entry.document_id == keep.id for h in svc.search(q, 10))


def test_deleting_every_document_empties_the_index(session, make_service):
    doc, _ = add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    remove_doc(session, doc)
    svc.sync()
    assert svc.index_count == 0 and rows(session) == {}
    assert svc.status()["state"] == "ok"


# ---- embedding model changes ------------------------------------------------------------------


def test_changed_model_invalidates_the_whole_index_and_never_mixes_vectors(
    session, make_service, tmp_path
):
    add_doc(session, CORPUS)
    add_doc(session, ["Kubernetes schedules containers."])
    old = make_service(HashingEmbedder(256))
    old.sync()
    assert old.index.dimension == 256

    new_embedder = HashingEmbedder(64)  # a different model: other dimension and fingerprint
    new = make_service(new_embedder)
    report = new.initialize()  # loads the old file, notices the mismatch, rebuilds
    assert report["source"] == "rebuild" and "configuration changed" in report["reason"]
    assert new.index.dimension == 64
    assert new_embedder.texts_embedded == 5  # every chunk was recomputed with the new model
    assert new.snapshot_meta.model_id == "hashing-bow-v1/dim64"
    assert new.snapshot_meta.config_fingerprint == new_embedder.config_fingerprint
    assert {r.dimension for r in rows(session).values()} == {64}
    assert {r.model_id for r in rows(session).values()} == {"hashing-bow-v1/dim64"}
    loaded, meta = vector_store.load_index(tmp_path / "vec" / "index.bin")
    assert loaded.dimension == 64 and meta.config_fingerprint == new_embedder.config_fingerprint
    assert new.check_consistency()["memory"]["consistent"]


def test_model_change_detected_at_query_time_not_only_at_startup(session, make_service):
    add_doc(session, CORPUS)
    svc = make_service(HashingEmbedder(256))
    svc.sync()
    svc.embedder = HashingEmbedder(64)  # configuration swapped under a running service
    assert svc.status()["state"] == "stale"
    svc.ensure_fresh()
    assert svc.index.dimension == 64 and svc.status()["state"] == "ok"


def test_dimension_drift_with_unchanged_configuration_is_still_caught(session, make_service):
    add_doc(session, CORPUS)
    a = HashingEmbedder(256)
    svc = make_service(a)
    svc.sync()
    sneaky = HashingEmbedder(64)
    sneaky.config_fingerprint = a.config_fingerprint  # same declared config, different output size
    svc.embedder = sneaky
    svc.sync()
    assert svc.index.dimension == 64  # old 256-d vectors were discarded, not mixed


def test_model_revision_drift_is_detected(session, make_service):
    class Revisioned(HashingEmbedder):
        def __init__(self, revision):
            super().__init__(128)
            self._revision = revision

        def identity(self):
            base = super().identity()
            return type(base)(
                base.kind, base.model_id, self._revision, base.dimension, base.config_fingerprint
            )

    add_doc(session, CORPUS)
    svc = make_service(Revisioned("a" * 40))
    svc.sync()
    assert svc.snapshot_meta.model_revision == "a" * 40
    emb = Revisioned("b" * 40)  # same settings fingerprint, but the model files changed
    svc.embedder = emb
    report = svc.sync()
    assert report.rebuilt and "revision" in report.reason
    assert emb.texts_embedded == 4 and svc.snapshot_meta.model_revision == "b" * 40


def test_rebuild_with_same_model_and_content_is_logically_equivalent(session, make_service):
    add_doc(session, CORPUS)
    add_doc(session, ["Docker packages apps.", "Redis stores keys in memory."])
    svc = make_service()
    svc.sync()
    before_entries = svc.indexed_entries()
    before_vecs = {e.chunk_id: svc.index.vector(e.chunk_id).copy() for e in before_entries}

    report = svc.rebuild()
    assert report.rebuilt and report.embedded == 6 and report.reused == 0
    assert sorted(svc.indexed_entries(), key=lambda e: e.chunk_id) == sorted(
        before_entries, key=lambda e: e.chunk_id
    )
    for e in before_entries:
        assert np.array_equal(svc.index.vector(e.chunk_id), before_vecs[e.chunk_id])
    q = svc.embedder.embed(["web framework"])[0]
    assert [h.entry.chunk_id for h in svc.search(q, 6)] == [
        h.entry.chunk_id for h in svc.index.clone().search(q, 6)
    ]


# ---- startup, missing / corrupted / stale files --------------------------------------------------


def test_startup_adopts_a_valid_current_file_without_embedding(session, make_service):
    add_doc(session, CORPUS)
    make_service().sync()
    fresh_embedder = HashingEmbedder()
    fresh = make_service(fresh_embedder)
    report = fresh.initialize()
    assert report["source"] == "file" and fresh_embedder.calls == 0
    assert fresh.index_count == 4 and fresh.status()["state"] == "ok"


def test_missing_file_is_recovered_from_sqlite_chunks(session, make_service, tmp_path):
    add_doc(session, CORPUS)
    original = make_service()
    original.sync()
    expected = {
        e.chunk_id: original.index.vector(e.chunk_id).copy() for e in original.indexed_entries()
    }
    (tmp_path / "vec" / "index.bin").unlink()

    recovered = make_service()
    report = recovered.initialize()
    assert report["source"] == "rebuild" and "does not exist" in report["reason"]
    assert report["sync"]["embedded"] == 4
    assert (tmp_path / "vec" / "index.bin").exists()
    assert sorted(entry_ids(recovered)) == sorted(expected)
    for cid, vec in expected.items():
        assert np.array_equal(recovered.index.vector(cid), vec)


@pytest.mark.parametrize("damage", ["garbage", "truncated", "empty", "bitflip", "tampered_ids"])
def test_corrupted_file_is_quarantined_and_rebuilt(session, make_service, tmp_path, damage):
    add_doc(session, CORPUS)
    make_service().sync()
    path = tmp_path / "vec" / "index.bin"
    raw = bytearray(path.read_bytes())
    if damage == "garbage":
        bad = b"\x00garbage" * 20
    elif damage == "truncated":
        bad = bytes(raw[: len(raw) // 2])
    elif damage == "empty":
        bad = b""
    elif damage == "bitflip":
        raw[-5] ^= 0xFF
        bad = bytes(raw)
    else:
        bad = bytes(raw).replace(b"-c1", b"-cX", 1)
    path.write_bytes(bad)

    svc = make_service()
    report = svc.initialize()
    assert report["source"] == "rebuild" and report["sync"]["embedded"] == 4
    assert path.with_name(path.name + ".corrupt").read_bytes() == bad  # kept, not deleted
    assert vector_store.load_index(path)[0].count == 4
    assert svc.check_consistency()["file"]["consistent"]


def test_stale_file_is_brought_up_to_date_incrementally(session, make_service):
    add_doc(session, CORPUS)
    make_service().sync()
    add_doc(session, ["Redis stores keys in memory."])  # file is now one chunk behind
    embedder = HashingEmbedder()
    svc = make_service(embedder)
    report = svc.initialize()
    assert report["source"] == "file" and report["sync"]["embedded"] == 1
    assert embedder.texts_embedded == 1 and svc.index_count == 5


def test_startup_sync_can_be_disabled(session, make_service):
    add_doc(session, CORPUS)
    embedder = HashingEmbedder()
    svc = make_service(embedder, sync_on_startup=False)
    svc.initialize()
    assert embedder.calls == 0 and svc.status()["state"] == "stale"


def test_unmigrated_database_makes_the_index_unavailable_not_the_app(tmp_path):
    from app.core.database import make_engine, make_session_factory

    factory = make_session_factory(make_engine(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"))
    svc = VectorService(factory, tmp_path / "i.bin", HashingEmbedder())
    assert svc.initialize()["source"] == "unavailable"
    assert svc.status()["state"] == "error"
    with pytest.raises(VectorIndexUnavailable):
        svc.sync()


# ---- failures never damage the existing index --------------------------------------------------


def test_embedding_failure_leaves_index_and_file_untouched_and_is_recorded(
    session, make_service, tmp_path
):
    _, good_ids = add_doc(session, CORPUS)
    embedder = ControlledEmbedder()
    svc = make_service(embedder)
    svc.sync()
    path = tmp_path / "vec" / "index.bin"
    file_before, entries_before = path.read_bytes(), svc.indexed_entries()

    _, bad_ids = add_doc(session, ["POISON one.", "POISON two."])
    embedder.fail_on = {"POISON"}
    with pytest.raises(VectorIndexUnavailable, match="embedding failed"):
        svc.sync()

    assert svc.indexed_entries() == entries_before  # in-memory index unchanged
    assert path.read_bytes() == file_before  # file byte-identical
    got = rows(session)
    assert {got[c].status for c in bad_ids} == {"failed"}
    assert "simulated embedding failure" in got[bad_ids[0]].error_message
    assert {got[c].status for c in good_ids} == {"indexed"}
    status = svc.status()
    assert status["state"] == "degraded" and "simulated embedding failure" in status["last_error"]
    # search over the old index still works
    q = embedder.embed(["capital of Germany"])[0]
    assert svc.search(q, 1)[0].entry.chunk_id == good_ids[2]


def test_recovery_after_a_failure(session, make_service):
    add_doc(session, CORPUS)
    embedder = ControlledEmbedder()
    svc = make_service(embedder)
    svc.sync()
    _, bad_ids = add_doc(session, ["POISON text here."])
    embedder.fail_on = {"POISON"}
    with pytest.raises(VectorIndexUnavailable):
        svc.sync()
    embedder.fail_on = set()  # the cause is fixed
    report = svc.sync()
    assert report.embedded == 1 and svc.status()["state"] == "ok"
    assert (
        rows(session)[bad_ids[0]].status == "indexed"
        and rows(session)[bad_ids[0]].error_message is None
    )


def test_failed_file_write_keeps_the_memory_index_current_and_retries(
    session, make_service, tmp_path, monkeypatch
):
    add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    path = tmp_path / "vec" / "index.bin"
    file_before = path.read_bytes()

    def failing(*_a, **_k):
        raise IndexPersistenceError("simulated disk failure")

    monkeypatch.setattr("app.services.embeddings.service.save_index", failing)
    add_doc(session, ["Redis stores keys in memory."])
    report = svc.sync()
    assert report.persisted is False and svc.index_count == 5
    status = svc.status()
    assert status["state"] == "degraded" and not status["file_in_sync"]
    assert path.read_bytes() == file_before  # old file intact and loadable
    assert vector_store.load_index(path)[0].count == 4

    monkeypatch.undo()
    assert svc.sync().embedded == 0  # nothing to embed, but the pending write is retried
    assert svc.status()["state"] == "ok" and svc.status()["file_in_sync"]
    assert vector_store.load_index(path)[0].count == 5


def test_stale_embeddings_are_marked_when_a_new_model_cannot_embed(session, make_service):
    add_doc(session, CORPUS)
    make_service(HashingEmbedder(256)).sync()
    broken = ControlledEmbedder(64, fail_on={"FastAPI", "Pydantic", "Berlin", "SQLAlchemy"})
    svc = make_service(broken)
    svc.initialize()  # startup sync fails, but must not raise
    statuses = {r.status for r in rows(session).values()}
    assert "stale" in statuses or "failed" in statuses
    assert svc.status()["state"] == "degraded"


# ---- consistency check ------------------------------------------------------------------


def test_consistency_check_reports_a_healthy_index(session, make_service):
    add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    report = svc.check_consistency()
    assert report["memory"]["consistent"] and report["file"]["consistent"]


def test_consistency_check_detects_drift_and_rebuild_repairs_it(session, make_service):
    doc, ids = add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    # Tamper behind the index's back (no revision bump, so nothing triggers a sync).
    session.execute(delete(DocumentChunk).where(DocumentChunk.id == ids[0]))
    session.execute(
        update(DocumentChunk).where(DocumentChunk.id == ids[1]).values(content="changed")
    )
    session.execute(delete(ChunkEmbedding).where(ChunkEmbedding.chunk_id == ids[3]))
    session.commit()

    memory = svc.check_consistency()["memory"]
    assert not memory["consistent"]
    assert memory["extra_in_index"] == [ids[0]]
    assert memory["content_mismatch"] == [ids[1]]
    assert memory["bookkeeping_missing"] == [ids[3]]

    svc.rebuild()
    assert svc.check_consistency()["memory"]["consistent"]
    assert svc.index_count == 3


# ---- concurrency ------------------------------------------------------------------------


def test_concurrent_syncs_and_writes_end_consistent(session, make_service):
    add_doc(session, CORPUS)
    svc = make_service()
    svc.sync()
    errors: list[BaseException] = []

    def syncer():
        try:
            for _ in range(5):
                svc.sync()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=syncer) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(5):
        add_doc(session, [f"extra document number {i} about topic {i}"])
    for t in threads:
        t.join(30)
    svc.sync()
    assert errors == []
    assert svc.index_count == 9 and svc.check_consistency()["memory"]["consistent"]


def test_embedder_exceptions_are_embedding_errors_not_crashes(session, make_service):
    class Exploding(HashingEmbedder):
        def embed(self, texts):
            raise RuntimeError("boom")

    add_doc(session, CORPUS)
    svc = make_service(Exploding())
    with pytest.raises(VectorIndexUnavailable):
        svc.sync()
    assert svc.index_count == 0 and isinstance(EmbeddingError("x"), Exception)


def test_writes_that_race_with_embedding_never_poison_the_index(session, make_service, app):
    """SQLite changes while embedding is in flight: keep valid work, drop outdated vectors,
    record the older revision, and let the next sync finish the job."""
    _, ids = add_doc(session, CORPUS)

    class RacingEmbedder(HashingEmbedder):
        fired = False

        def embed(self, texts):
            out = super().embed(texts)
            if not RacingEmbedder.fired:
                RacingEmbedder.fired = True
                with app.state.session_factory() as other:  # a concurrent writer
                    set_chunk_text(other, ids[1], "Pydantic was rewritten while we were embedding.")
                    add_doc(other, ["A brand new document appeared meanwhile."])
            return out

    svc = make_service(RacingEmbedder())
    report = svc.sync()

    assert report.details.get("raced_with_writes") is True
    indexed = {e.chunk_id: e for e in svc.indexed_entries()}
    assert ids[1] not in indexed  # its vector was computed from outdated text: not stored
    assert report.embedded == 3 and len(indexed) == 3
    assert svc.snapshot_meta.revision < current_vector_revision(session)  # recorded as behind
    assert svc.status()["state"] == "stale"

    followup = svc.sync()
    assert followup.embedded == 2 and svc.index_count == 5  # the changed chunk + the new one
    assert svc.status()["state"] == "ok" and svc.check_consistency()["memory"]["consistent"]


def test_one_bad_chunk_does_not_block_the_rest_from_being_indexed(session, make_service, tmp_path):
    add_doc(session, CORPUS)
    embedder = ControlledEmbedder()
    svc = make_service(embedder)
    svc.sync()
    _, ids = add_doc(
        session, ["This text is fine.", "POISON triggers a failure.", "So is this one."]
    )
    embedder.fail_on = {"POISON"}

    report = svc.sync()  # does not raise: partial success is applied

    assert report.embedded == 2 and report.details["failed_chunks"] == 1
    assert {ids[0], ids[2]} <= set(entry_ids(svc)) and ids[1] not in entry_ids(svc)
    got = rows(session)
    assert (
        got[ids[1]].status == "failed"
        and "simulated embedding failure" in got[ids[1]].error_message
    )
    assert got[ids[0]].status == got[ids[2]].status == "indexed"
    status = svc.status()
    assert status["state"] == "degraded" and "1 chunk(s) failed to embed" in status["last_error"]
    assert vector_store.load_index(tmp_path / "vec" / "index.bin")[0].count == 6  # persisted

    embedder.fail_on = set()  # fixed: the failed chunk is retried by the next sync
    assert svc.sync().embedded == 1
    assert svc.status()["state"] == "ok" and rows(session)[ids[1]].status == "indexed"


def test_a_backend_that_cannot_load_is_not_retried_chunk_by_chunk(session, make_service):
    class NeverLoads(HashingEmbedder):
        attempts = 0

        @property
        def is_loaded(self):
            return False

        def embed(self, texts):
            NeverLoads.attempts += 1
            raise EmbeddingError("model download failed")

    add_doc(session, CORPUS)
    svc = make_service(NeverLoads())
    with pytest.raises(VectorIndexUnavailable, match="model download failed"):
        svc.sync()
    assert NeverLoads.attempts == 1  # one attempt for the whole group, not one per chunk
