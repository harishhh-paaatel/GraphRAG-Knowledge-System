"""Phase 5A reliability behaviours exercised through the service and the HTTP API."""

import threading

import pytest
from sqlalchemy import delete, select

from app.models import ChunkEmbedding
from app.services.embeddings.embedder import HashingEmbedder, SentenceTransformerEmbedder
from app.services.embeddings.service import VectorIndexUnavailable, VectorService
from app.services.embeddings.snapshot import IndexSnapshot
from app.services.embeddings.vector_index import SearchHit
from tests.vector_helpers import (
    CORPUS,
    BlockingEmbedder,
    add_doc,
    install_fake_sentence_transformers,
)

DOCS = "/api/v1/documents"
SEARCH = "/api/v1/search/vector"
INDEX = "/api/v1/vector/index"


@pytest.fixture
def upload(client, app):
    def _upload(name, data, mime="text/plain"):
        r = client.post(f"{DOCS}/upload", files={"file": (name, data, mime)})
        assert r.status_code == 202, r.text
        assert app.state.worker.drain(timeout=30)
        return r.json()["document"]["id"]

    return _upload


def search(client, query, **kw):
    r = client.post(SEARCH, json={"query": query, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def rows(session):
    session.expire_all()
    return {r.chunk_id: r for r in session.scalars(select(ChunkEmbedding))}


# ---- token limit: detected, recorded, and the tail of a long chunk is still searchable -----------


def test_oversize_chunk_is_pooled_from_windows_recorded_and_its_tail_is_searchable(
    client, app, upload, session
):
    app.state.vectors.embedder = HashingEmbedder(256, max_tokens=12)  # 12-token model limit
    head = " ".join(f"alpha{i}" for i in range(12))
    tail = " ".join(f"omega{i}" for i in range(12))
    doc = upload("long.txt", f"{head} {tail}".encode())

    detail = client.get(f"{DOCS}/{doc}").json()
    info = detail["metadata"]["vector_index"]
    assert detail["processing_status"] == "completed" and info["status"] == "indexed"
    assert info["chunks_over_token_limit_pooled"] == 1

    [row] = rows(session).values()
    assert (row.token_count, row.window_count) == (24, 2)  # counted by the embedder's tokenizer

    stats = client.get(INDEX).json()
    assert stats["oversized_chunks_pooled"] == 1 and stats["max_tokens"] == 12
    assert stats["oversize_policy"] == "window-mean-v1" and stats["engine"] == "faiss-flat-ip"

    body = search(client, tail)  # words that exist ONLY in the second window
    assert body["results"][0]["document_id"] == doc and body["results"][0]["similarity"] > 0.5
    assert (body["query_token_count"], body["query_windows"], body["max_tokens"]) == (12, 1, 12)

    long_query = search(client, f"{head} {tail}")
    assert (long_query["query_token_count"], long_query["query_windows"]) == (24, 2)
    assert long_query["results"][0]["document_id"] == doc


def test_chunks_that_fit_are_not_marked_as_pooled(client, app, upload, session):
    app.state.vectors.embedder = HashingEmbedder(256, max_tokens=50)
    doc = upload("short.txt", b"FastAPI is a web framework for building APIs with Python.")
    info = client.get(f"{DOCS}/{doc}").json()["metadata"]["vector_index"]
    assert info["chunks_over_token_limit_pooled"] == 0
    [row] = rows(session).values()
    assert row.window_count == 1 and row.token_count == 10


def test_token_bookkeeping_is_restored_after_reprocessing_without_re_embedding(
    session, app, tmp_path
):
    embedder = HashingEmbedder(64, max_tokens=10)
    svc = VectorService(app.state.session_factory, tmp_path / "v.bin", embedder)
    _, ids = add_doc(session, ["short text here", " ".join(f"w{i}" for i in range(25))])
    svc.sync()
    got = rows(session)
    assert [(got[i].token_count, got[i].window_count) for i in ids] == [(3, 1), (25, 3)]
    assert [(e.token_count, e.windows) for e in svc.indexed_entries()] == [(3, 1), (25, 3)]

    session.execute(delete(ChunkEmbedding))  # what deleting + re-inserting chunks does (cascade)
    session.commit()
    embedded_before = embedder.texts_embedded
    svc.sync()
    assert embedder.texts_embedded == embedded_before  # nothing recomputed
    got = rows(session)
    assert [(got[i].token_count, got[i].window_count) for i in ids] == [(3, 1), (25, 3)]


# ---- availability through the HTTP API --------------------------------------------------------


def test_api_serves_the_previous_index_and_says_so_while_a_sync_is_running(
    client, app, upload, session
):
    upload("a.txt", b"Berlin is the capital city of Germany and a large European city.")
    blocker = BlockingEmbedder(256)  # same configuration fingerprint as the default stub
    assert blocker.config_fingerprint == app.state.vectors.embedder.config_fingerprint
    app.state.vectors.embedder = blocker
    add_doc(session, ["SLOWDOC a chunk whose embedding takes a long time"])

    syncing = threading.Thread(target=app.state.vectors.sync)
    syncing.start()
    try:
        assert blocker.entered.wait(10)
        body = search(client, "capital of Germany")
        assert body["index_stale"] is True and body["results"]
        assert body["index_revision"] < body["database_revision"]
        assert client.get(INDEX).json()["sync_in_progress"] is True
    finally:
        blocker.release.set()
        syncing.join(15)
    assert search(client, "capital of Germany")["index_stale"] is False


def test_api_reports_503_building_instead_of_serving_vectors_from_another_model(
    client, app, upload, session
):
    upload("a.txt", b"Berlin is the capital city of Germany and a large European city.")
    blocker = BlockingEmbedder(64)  # different model: the existing 256-d vectors are unusable
    app.state.vectors.embedder = blocker
    add_doc(session, ["SLOWDOC forces the rebuild to take a while"])

    syncing = threading.Thread(target=app.state.vectors.sync)
    syncing.start()
    try:
        assert blocker.entered.wait(10)
        r = client.post(SEARCH, json={"query": "capital of Germany"})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "vector_index_building"
    finally:
        blocker.release.set()
        syncing.join(15)
    assert client.post(SEARCH, json={"query": "capital of Germany"}).status_code == 200


# ---- threshold policy through the API --------------------------------------------------------


def test_threshold_and_displayed_similarity_use_the_same_rounded_value(
    client, app, upload, monkeypatch
):
    upload("a.txt", b"Berlin is the capital city of Germany.")
    upload("b.txt", b"Kubernetes schedules containers across a cluster.")
    first, second = app.state.vectors.indexed_entries()[:2]

    def fake_search(self, query, top_k, *, document_ids=None, exclude_document_ids=None):
        # Raw scores straddling 0.8 by amounts far below the 6-decimal display precision.
        return [SearchHit(first, 0.7999999), SearchHit(second, 0.7999994)]

    monkeypatch.setattr(IndexSnapshot, "search", fake_search)

    at_08 = search(client, "anything", similarity_threshold=0.8, top_k=5)
    assert [r["chunk_id"] for r in at_08["results"]] == [first.chunk_id]
    assert at_08["results"][0]["similarity"] == 0.8  # displays exactly the boundary it passed

    both = search(client, "anything", similarity_threshold=0.799999, top_k=5)
    assert [r["similarity"] for r in both["results"]] == [0.8, 0.799999]

    none = search(client, "anything", similarity_threshold=0.800001, top_k=5)
    assert none["results"] == []

    for body in (at_08, both):
        floor = round(body["similarity_threshold"], body["similarity_decimals"])
        assert all(
            r["similarity"] >= floor for r in body["results"]
        )  # display never contradicts filter
    assert (
        at_08["similarity_decimals"] == 6 and "not answer confidence" in at_08["similarity_metric"]
    )


# ---- revision / configuration mismatch and offline, at the service level ----------------------


def test_changing_the_pinned_revision_rebuilds_the_index_without_any_model_update(
    session, app, tmp_path, monkeypatch
):
    install_fake_sentence_transformers(monkeypatch)
    _, ids = add_doc(session, CORPUS)
    path = tmp_path / "idx.bin"

    def service(revision):
        embedder = SentenceTransformerEmbedder(
            "org/model", revision=revision, cache_dir=tmp_path / "models"
        )
        return VectorService(app.state.session_factory, path, embedder), embedder

    first, emb_a = service("a" * 40)
    first.sync()
    assert first.snapshot().meta.model_revision == "a" * 40

    second, emb_b = service("b" * 40)
    report = second.initialize()  # same file, different pinned revision
    assert report["source"] == "rebuild" and "configuration changed" in report["reason"]
    assert report["sync"]["rebuilt"] is True and report["sync"]["embedded"] == len(ids)
    assert second.snapshot().meta.model_revision == "b" * 40
    assert second.snapshot().meta.config_fingerprint == emb_b.config_fingerprint
    assert emb_a.config_fingerprint != emb_b.config_fingerprint
    assert {r.model_revision for r in rows(session).values()} == {"b" * 40}  # nothing mixed


def test_unavailable_model_offline_fails_the_sync_with_a_clear_message_and_changes_nothing(
    session, app, tmp_path, monkeypatch
):
    install_fake_sentence_transformers(monkeypatch, fail=OSError("not in the local cache"))
    add_doc(session, CORPUS)
    embedder = SentenceTransformerEmbedder(
        "org/model", revision="a" * 40, cache_dir=tmp_path / "models", offline=True
    )
    svc = VectorService(app.state.session_factory, tmp_path / "idx.bin", embedder)
    with pytest.raises(VectorIndexUnavailable) as exc:
        svc.sync()
    assert "offline mode is on" in str(exc.value) and "No other model is substituted" in str(
        exc.value
    )
    assert svc.snapshot() is None and not (tmp_path / "idx.bin").exists()
    status = svc.status()
    assert status["state"] in {"degraded", "stale"} and "offline mode is on" in status["last_error"]
    assert {r.status for r in rows(session).values()} == {"failed"}
