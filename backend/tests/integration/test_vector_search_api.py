"""Vector indexing wired into ingestion, plus the search and index-stats endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import Settings
from app.main import create_app
from app.models import Base, ChunkEmbedding, Document, DocumentChunk, ProcessingJob
from app.services.embeddings import vector_store
from app.services.embeddings.embedder import HashingEmbedder
from app.services.ingestion.pipeline import run_ingestion
from tests.helpers import make_pdf
from tests.vector_helpers import ControlledEmbedder

DOCS = "/api/v1/documents"
SEARCH = "/api/v1/search/vector"
INDEX = "/api/v1/vector/index"

FASTAPI = b"FastAPI is a web framework for building APIs with Python."
BERLIN = b"Berlin is the capital city of Germany and a large European city."


@pytest.fixture
def upload(client, app):
    def _upload(name, data, mime="text/plain"):
        r = client.post(f"{DOCS}/upload", files={"file": (name, data, mime)})
        assert r.status_code == 202, r.text
        assert app.state.worker.drain(timeout=30)
        return r.json()["document"]["id"]

    return _upload


@pytest.fixture
def corpus(upload):
    return {
        "fastapi": upload("fastapi.txt", FASTAPI),
        "berlin": upload("berlin.txt", BERLIN),
        "pdf": upload(
            "db.pdf",
            make_pdf(
                [
                    "SQLAlchemy maps Python objects to relational database tables.",
                    "Alembic handles database schema migrations for SQLAlchemy.",
                ]
            ),
            "application/pdf",
        ),
    }


def search(client, query, **kw):
    r = client.post(SEARCH, json={"query": query, **kw})
    assert r.status_code == 200, r.text
    return r.json()


# ---- ingestion -> embeddings -> index ------------------------------------------------------


def test_ingestion_generates_embeddings_and_indexes_chunks(client, app, session, corpus):
    n_chunks = session.query(DocumentChunk).count()
    assert n_chunks == 4  # 1 + 1 + 2 pdf pages
    assert app.state.vectors.index_count == n_chunks
    session.expire_all()
    rows = session.scalars(select(ChunkEmbedding)).all()
    assert len(rows) == n_chunks and {r.status for r in rows} == {"indexed"}

    detail = client.get(f"{DOCS}/{corpus['pdf']}").json()
    vi = detail["metadata"]["vector_index"]
    assert vi["status"] == "indexed" and vi["chunks_indexed"] == vi["chunks_total"] == 2
    assert vi["model"] == "hashing-bow-v1/dim256" and vi["dimension"] == 256
    status = client.get(f"{DOCS}/{corpus['pdf']}/status").json()
    assert status["processing_status"] == "completed" and status["job"]["stage"] == "done"


def test_index_file_is_persisted_and_matches_sqlite(client, app, corpus):
    path = app.state.settings.vector_index_file
    index, meta = vector_store.load_index(path)
    assert index.count == 4 and meta.model_id == "hashing-bow-v1/dim256"
    assert client.get(f"{INDEX}?check=true").json()["consistency"]["file"]["consistent"] is True


# ---- search ---------------------------------------------------------------------------------


def test_search_returns_the_relevant_chunk_first_with_full_metadata(client, corpus):
    body = search(client, "web framework for building APIs", top_k=3)
    first = body["results"][0]
    assert first["document_id"] == corpus["fastapi"]
    assert first["content"] == FASTAPI.decode()
    assert first["source"]["filename"] == "fastapi.txt"
    assert first["rank"] == 1 and first["chunk_id"] and first["source"]["chunk_index"] == 0
    assert first["embedding_model"] == body["embedding_model"] == "hashing-bow-v1/dim256"
    assert 0.0 < first["similarity"] <= 1.0
    sims = [r["similarity"] for r in body["results"]]
    assert sims == sorted(sims, reverse=True) and len(sims) == 3
    assert "not answer confidence" in body["similarity_metric"]
    assert body["indexed_vectors"] == 4 and body["top_k"] == 3 and body["index_revision"]
    assert set(body["timings_ms"]) == {"embed_query", "faiss_search", "metadata_lookup", "total"}


def test_similarity_is_the_real_cosine_between_query_and_stored_vector(client, app, corpus):
    body = search(client, "database schema migrations", top_k=4)
    embedder = app.state.vectors.embedder
    q = embedder.embed(["database schema migrations"])[0]
    index = app.state.vectors.index
    for r in body["results"]:
        stored = index.vector(r["chunk_id"])
        assert r["similarity"] == pytest.approx(float(q @ stored), abs=1e-5)
        assert r["similarity"] == pytest.approx(
            float(q @ stored / (np.linalg.norm(q) * np.linalg.norm(stored))), abs=1e-5
        )


def test_page_numbers_are_returned_for_pdf_chunks(client, corpus):
    body = search(client, "Alembic schema migrations", top_k=2)
    top = body["results"][0]
    assert top["document_id"] == corpus["pdf"] and top["source"]["filename"] == "db.pdf"
    assert top["source"]["page_number"] == 2
    assert "Alembic" in top["content"]


def test_top_k_limits_results_and_k_larger_than_corpus_is_fine(client, corpus):
    assert len(search(client, "python", top_k=1)["results"]) == 1
    assert len(search(client, "python", top_k=50)["results"]) == 4


def test_document_filter_restricts_results_and_reports_unknown_ids(client, corpus):
    body = search(client, "capital city", document_ids=[corpus["pdf"], "no-such-document"])
    assert {r["document_id"] for r in body["results"]} == {corpus["pdf"]}
    assert body["unknown_document_ids"] == ["no-such-document"]
    assert search(client, "capital city", document_ids=["no-such-document"])["results"] == []
    assert search(client, "capital city", document_ids=[])["results"] == []


def test_similarity_threshold_filters_low_scores(client, corpus):
    everything = search(client, "web framework for building APIs", top_k=4)["results"]
    cutoff = everything[0]["similarity"] - 1e-5  # responses are rounded to 6 decimals
    kept = search(client, "web framework for building APIs", top_k=4, similarity_threshold=cutoff)
    assert [r["chunk_id"] for r in kept["results"]] == [everything[0]["chunk_id"]]
    assert search(client, "web framework", similarity_threshold=1.0)["results"] == []
    assert len(search(client, "web framework", similarity_threshold=-1.0, top_k=4)["results"]) == 4


def test_empty_index_returns_an_empty_result_list(client):
    body = search(client, "anything at all")
    assert body["results"] == [] and body["indexed_vectors"] == 0


def test_only_completed_documents_are_searchable(client, corpus, session):
    session.get(Document, corpus["fastapi"]).processing_status = "failed"
    session.commit()
    ids = {r["document_id"] for r in search(client, "FastAPI web framework", top_k=10)["results"]}
    assert corpus["fastapi"] not in ids and ids


def test_equal_scores_are_returned_in_a_deterministic_order(client, upload):
    same = b"Redis stores keys in memory."
    upload("a.txt", same)
    upload("b.txt", same + b" ")  # different bytes (so not a duplicate upload), same chunk text
    runs = [
        [r["chunk_id"] for r in search(client, "redis keys memory", top_k=2)["results"]]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2] == sorted(runs[0])


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 2001},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "top_k": -3},
        {"query": "ok", "similarity_threshold": 1.5},
        {"query": "ok", "similarity_threshold": -2},
        {"query": "ok", "document_ids": ["d"] * 101},
        {},
        {"query": 5},
    ],
)
def test_search_validation(client, payload):
    r = client.post(SEARCH, json=payload)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_top_k_above_the_configured_maximum_is_rejected(client, app):
    app.state.settings.vector_max_top_k = 10
    r = client.post(SEARCH, json={"query": "ok", "top_k": 11})
    assert r.status_code == 422 and r.json()["error"]["code"] == "limit_exceeded"


# ---- reprocessing -------------------------------------------------------------------------


def test_reprocessing_a_document_does_not_duplicate_or_recompute_vectors(
    client, app, corpus, session
):
    embedder = app.state.vectors.embedder
    before_count = app.state.vectors.index_count
    before_chunks = sorted(e.chunk_id for e in app.state.vectors.indexed_entries())
    embedded_before = embedder.texts_embedded

    job = ProcessingJob(document_id=corpus["pdf"], status="pending", attempt=2)
    session.add(job)
    session.commit()
    run_ingestion(
        app.state.session_factory,
        app.state.settings,
        job.id,
        None,
        app.state.graph,
        app.state.vectors,
    )

    assert app.state.vectors.index_count == before_count
    assert sorted(e.chunk_id for e in app.state.vectors.indexed_entries()) == before_chunks
    assert embedder.texts_embedded == embedded_before  # identical chunks: nothing re-embedded
    session.expire_all()
    assert session.query(ChunkEmbedding).count() == before_count  # bookkeeping restored
    assert app.state.vectors.check_consistency()["memory"]["consistent"]
    assert search(client, "Alembic migrations")["results"][0]["document_id"] == corpus["pdf"]


# ---- deletion -----------------------------------------------------------------------------


def test_deleting_a_document_removes_its_vectors_and_persists_the_change(
    client, app, corpus, session
):
    assert client.delete(f"{DOCS}/{corpus['pdf']}").status_code == 204

    assert app.state.vectors.index_count == 2
    assert all(e.document_id != corpus["pdf"] for e in app.state.vectors.indexed_entries())
    session.expire_all()
    assert session.query(ChunkEmbedding).count() == 2
    index, _ = vector_store.load_index(app.state.settings.vector_index_file)
    assert index.count == 2  # the file was rewritten
    body = search(client, "Alembic database schema migrations", top_k=10)
    assert corpus["pdf"] not in {r["document_id"] for r in body["results"]}
    assert client.get(INDEX).json()["vector_count"] == 2


def test_deleting_the_last_document_empties_the_index(client, app, upload):
    doc = upload("a.txt", FASTAPI)
    assert client.delete(f"{DOCS}/{doc}").status_code == 204
    assert app.state.vectors.index_count == 0
    assert vector_store.load_index(app.state.settings.vector_index_file)[0] is None
    assert search(client, "FastAPI")["results"] == []


# ---- failures ------------------------------------------------------------------------------


def test_embedding_outage_does_not_fail_ingestion_and_is_recoverable(client, app, upload):
    app.state.vectors.embedder = ControlledEmbedder(fail_on={"POISON"})
    doc = upload("p.txt", b"POISON cannot be embedded right now.")

    detail = client.get(f"{DOCS}/{doc}").json()
    assert detail["processing_status"] == "completed"  # the rest of the pipeline succeeded
    assert detail["metadata"]["vector_index"]["status"] == "failed"
    health = client.get("/api/v1/health").json()
    assert (
        health["components"]["vector_index"]["status"] == "error" and health["status"] == "degraded"
    )
    stats = client.get(INDEX).json()
    assert stats["state"] == "degraded" and stats["embedding_status_counts"]["failed"] == 1

    app.state.vectors.embedder.fail_on = set()  # the cause is fixed
    r = client.post("/api/v1/vector/rebuild")
    assert r.status_code == 200 and r.json()["index"]["state"] == "ok"
    assert search(client, "POISON cannot be embedded")["results"][0]["document_id"] == doc
    assert client.get("/api/v1/health").json()["components"]["vector_index"]["status"] == "ok"


def test_search_reports_503_when_the_index_cannot_be_synchronized(client, app, upload):
    upload("a.txt", FASTAPI)
    app.state.vectors.embedder = ControlledEmbedder(fail_on={"FastAPI", "Berlin", "Python"})
    # make the index stale so the search must re-embed, and the embedder fails
    upload_fail = client.post(f"{DOCS}/upload", files={"file": ("b.txt", BERLIN, "text/plain")})
    app.state.worker.drain(30)
    assert upload_fail.status_code == 202
    r = client.post(SEARCH, json={"query": "anything"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "vector_index_unavailable"


# ---- model changes ---------------------------------------------------------------------------


def test_model_change_is_detected_and_the_index_rebuilt_on_the_next_search(client, app, corpus):
    assert app.state.vectors.snapshot_meta.dimension == 256
    app.state.vectors.embedder = HashingEmbedder(64)  # operator switched the embedding model

    stale = client.get(INDEX).json()
    assert stale["state"] == "stale"
    assert stale["config_fingerprint"] != stale["current_config_fingerprint"]

    body = search(client, "web framework for building APIs")
    assert body["embedding_model"] == "hashing-bow-v1/dim64"
    assert body["results"][0]["document_id"] == corpus["fastapi"]
    after = client.get(INDEX).json()
    assert after["state"] == "ok" and after["dimension"] == 64 and after["vector_count"] == 4
    assert {m for m in {e.chunk_id for e in app.state.vectors.indexed_entries()}} == {
        r["chunk_id"] for r in search(client, "python", top_k=10)["results"]
    }


# ---- stats, rebuild, restart ------------------------------------------------------------------


def test_index_stats_endpoint(client, app, corpus):
    s = client.get(INDEX).json()
    assert s["state"] == "ok" and s["vector_count"] == 4 and s["dimension"] == 256
    assert s["model_id"] == "hashing-bow-v1/dim256" and s["chunks_in_database"] == 4
    assert s["embedding_status_counts"] == {"indexed": 4, "failed": 0, "stale": 0}
    assert s["index_revision"] == s["database_revision"] and s["file_in_sync"] is True
    assert s["file_size_bytes"] > 0 and s["last_sync"]["total"] == 4
    assert client.get(f"{INDEX}?check=true").json()["consistency"]["memory"]["consistent"]


def test_rebuild_endpoint_recomputes_everything_with_identical_results(client, app, corpus):
    before = search(client, "web framework for building APIs", top_k=4)["results"]
    r = client.post("/api/v1/vector/rebuild")
    assert r.status_code == 200
    report = r.json()["report"]
    assert report["rebuilt"] is True and report["embedded"] == 4 and report["reused"] == 0
    after = search(client, "web framework for building APIs", top_k=4)["results"]
    assert [(x["chunk_id"], x["similarity"]) for x in after] == [
        (x["chunk_id"], x["similarity"]) for x in before
    ]


def test_restart_loads_the_persisted_index_without_re_embedding(tmp_path):
    settings = Settings(data_dir=tmp_path, environment="test", _env_file=None)
    first = create_app(settings)
    Base.metadata.create_all(first.state.engine)
    with TestClient(first) as c:
        r = c.post(f"{DOCS}/upload", files={"file": ("a.txt", FASTAPI, "text/plain")})
        first.state.worker.drain(30)
        doc = r.json()["document"]["id"]
    first.state.engine.dispose()

    second = create_app(settings)
    embedder = second.state.vectors.embedder
    with TestClient(second) as c:
        assert second.state.vectors.status()["initialized_from"] == "file"
        assert embedder.texts_embedded == 0  # startup embedded nothing
        body = c.post(SEARCH, json={"query": "FastAPI web framework"}).json()
        assert body["results"][0]["document_id"] == doc
    second.state.engine.dispose()


def test_health_reports_the_vector_index(client, corpus):
    comp = client.get("/api/v1/health").json()["components"]["vector_index"]
    assert comp["status"] == "ok" and "vectors=4" in comp["detail"]


def test_reads_self_heal_after_a_delete_that_was_never_followed_by_a_sync(
    client, app, corpus, session, settings
):
    """The revision bump (not an explicit sync) is what makes the next query correct."""
    from app.services.ingestion import service

    session.expire_all()
    service.delete_document(session, settings, corpus["fastapi"])  # no vectors.sync() here
    assert app.state.vectors.index_count == 4  # the in-memory index has not noticed yet
    body = search(client, "web framework for building APIs", top_k=10)
    assert corpus["fastapi"] not in {r["document_id"] for r in body["results"]}
    assert app.state.vectors.index_count == 3 and body["indexed_vectors"] == 3
