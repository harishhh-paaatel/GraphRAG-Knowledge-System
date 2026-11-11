"""Real-model checks for token limits, the pinned revision and offline behaviour (opt-in).

Run with: GRAPHRAG_RUN_SLOW=1 pytest -m slow tests/slow
All offline tests also make any network connection fail, so "works offline" is proven rather
than assumed. They use the real model from the project cache (data/models, on the project drive).
"""

import socket
import time

import numpy as np
import pytest
from sqlalchemy import select

from app.core.config import DEFAULT_EMBEDDING_REVISION, REPO_ROOT
from app.models import ChunkEmbedding
from app.services.embeddings.embedder import EmbeddingError, SentenceTransformerEmbedder
from app.services.embeddings.service import VectorService
from app.services.embeddings.windowing import split_into_windows
from tests.vector_helpers import add_doc

pytestmark = pytest.mark.slow

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = REPO_ROOT / "data" / "models"
LIMIT = 256

PROSE = ("The ingestion service stores source documents and tracks processing jobs. " * 11)[:800]
URLISH = ("https://example.com/a/very/long/path/segment?query=1234567890&x=abcdefghij " * 11)[:800]
LONG = " ".join(f"alpha{i} beta{i} gamma{i} delta{i}" for i in range(120))


@pytest.fixture
def no_network(monkeypatch):
    def deny(*_a, **_k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


@pytest.fixture(scope="module")
def embedder():
    return SentenceTransformerEmbedder(
        MODEL, revision=DEFAULT_EMBEDDING_REVISION, cache_dir=CACHE, offline=True
    )


# ---- token limit, measured by the real tokenizer ---------------------------------------------


def test_the_real_tokenizer_not_the_character_count_decides_what_is_oversize(embedder):
    assert len(PROSE) == len(URLISH) == 800  # same length in characters...
    prose, urlish = embedder.describe([PROSE, URLISH])
    assert prose.windows == 1 and prose.token_count < LIMIT
    assert urlish.token_count > LIMIT and urlish.windows >= 2  # ...very different in tokens
    assert embedder.identity().max_tokens == LIMIT


def test_every_window_of_an_oversize_chunk_fits_the_real_limit_and_loses_nothing(embedder):
    windows = split_into_windows(URLISH, embedder._count, LIMIT)
    assert len(windows) >= 2
    assert all(c <= LIMIT for c in embedder._count(windows))
    assert "".join(w.replace(" ", "") for w in windows) == URLISH.replace(" ", "")  # lossless


def test_silent_truncation_is_real_and_the_policy_removes_it(embedder):
    tok = embedder._model.tokenizer
    ids = tok(LONG, add_special_tokens=True, truncation=False)["input_ids"]
    assert len(ids) > 4 * LIMIT
    head = tok.decode(ids[1 : LIMIT - 1])
    tail = tok.decode(ids[-(LIMIT - 2) : -1])
    v_head, v_tail = embedder.embed([head])[0], embedder.embed([tail])[0]

    # What the model alone does with the long text: it reads only the first LIMIT tokens.
    with embedder._encode_lock:
        raw = embedder._model.encode([LONG], normalize_embeddings=True, convert_to_numpy=True)[0]
    assert float(raw @ v_head) > 0.9999  # the tail contributed nothing

    pooled = embedder.embed([LONG])[0]
    assert float(np.linalg.norm(pooled)) == pytest.approx(1.0, abs=1e-5)
    assert float(pooled @ v_tail) > float(raw @ v_tail) + 0.02  # the tail now counts
    assert float(pooled @ v_head) < 0.9999
    assert embedder.describe([LONG])[0].windows >= 5


def test_an_oversize_chunk_flows_through_the_service_with_its_size_recorded(
    embedder, app, session, tmp_path
):
    _, ids = add_doc(session, [PROSE, URLISH])
    svc = VectorService(app.state.session_factory, tmp_path / "idx.bin", embedder)
    svc.sync()
    session.expire_all()
    got = {r.chunk_id: r for r in session.scalars(select(ChunkEmbedding))}
    assert got[ids[0]].window_count == 1 and got[ids[0]].token_count < LIMIT
    assert got[ids[1]].window_count >= 2 and got[ids[1]].token_count > LIMIT
    assert svc.snapshot().meta.max_tokens == LIMIT
    entry = {e.chunk_id: e for e in svc.indexed_entries()}[ids[1]]
    assert (entry.token_count, entry.windows) == (got[ids[1]].token_count, got[ids[1]].window_count)


# ---- pinned revision and offline -----------------------------------------------------------


def test_the_pinned_revision_loads_offline_from_the_local_cache(embedder, no_network):
    ident = embedder.identity()
    assert (
        ident.revision == DEFAULT_EMBEDDING_REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    assert ident.dimension == 384 and ident.max_tokens == LIMIT
    assert embedder.embed(["offline and pinned"]).shape == (1, 384)


def test_a_missing_model_cache_in_offline_mode_fails_fast_and_clearly(no_network, tmp_path):
    empty = SentenceTransformerEmbedder(
        MODEL, revision=DEFAULT_EMBEDDING_REVISION, cache_dir=tmp_path / "empty", offline=True
    )
    started = time.perf_counter()
    with pytest.raises(EmbeddingError) as exc:
        empty.embed(["anything"])
    assert time.perf_counter() - started < 60
    message = str(exc.value)
    assert MODEL in message and "offline mode is on" in message
    assert "No other model is substituted" in message and str(tmp_path) not in message
    assert not empty.is_loaded
    assert not list((tmp_path / "empty").rglob("models--*"))  # nothing was downloaded


def test_an_unknown_pinned_revision_offline_is_refused_not_replaced_by_the_cached_one(
    no_network,
):
    wrong = SentenceTransformerEmbedder(MODEL, revision="0" * 40, cache_dir=CACHE, offline=True)
    with pytest.raises(EmbeddingError, match="No other model is substituted"):
        wrong.embed(["anything"])
    assert not wrong.is_loaded
