"""Real-model smoke tests (opt-in): GRAPHRAG_RUN_SLOW=1 pytest -m slow tests/slow

These load the actual sentence-transformers/all-MiniLM-L6-v2 model (downloaded once into
data/models on the project drive) and check properties a stub cannot: true semantic ranking,
the real embedding dimension, and revision capture. The default suite never runs them.
"""

from pathlib import Path

import numpy as np
import pytest

from app.core.config import REPO_ROOT
from app.services.embeddings import vector_store
from app.services.embeddings.embedder import SentenceTransformerEmbedder
from app.services.embeddings.service import VectorService
from tests.vector_helpers import add_doc

pytestmark = pytest.mark.slow

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = REPO_ROOT / "data" / "models"  # kept on the project drive, never on C:

PASSAGES = {
    "fastapi": (
        "FastAPI is a modern web framework for building APIs with Python, based on type hints."
    ),
    "pydantic": "Pydantic validates data by checking values against Python type annotations.",
    "berlin": "Berlin is the capital and largest city of Germany, located on the river Spree.",
    "sqlalchemy": (
        "SQLAlchemy is an object-relational mapper that lets Python code talk to SQL databases."
    ),
    "alembic": "Alembic tracks and applies changes to a database schema over time as migrations.",
    "faiss": "FAISS is a library for fast similarity search over dense vectors.",
    "volcano": "A volcano is a rupture in the crust of a planet that lets lava and gas escape.",
    "bread": "To bake bread, mix flour, water, yeast and salt, let the dough rise, then bake it.",
}


@pytest.fixture(scope="module")
def embedder():
    return SentenceTransformerEmbedder(MODEL, cache_dir=CACHE, batch_size=16)


def test_real_model_loads_once_with_the_expected_dimension_and_revision(embedder):
    assert not embedder.is_loaded
    ident = embedder.identity()
    assert embedder.is_loaded
    assert ident.dimension == 384 and ident.kind == "sentence_transformers"
    assert ident.model_id == MODEL
    assert ident.revision is not None and len(ident.revision) == 40  # resolved HF commit
    assert embedder.identity() is ident  # loaded exactly once


def test_real_vectors_are_normalized_and_reproducible(embedder):
    texts = list(PASSAGES.values())
    a, b = embedder.embed(texts), embedder.embed(texts)
    assert a.shape == (8, 384) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.allclose(a, b, atol=1e-6)  # same model + text -> same vector
    assert np.allclose(embedder.embed(texts[:1]), a[:1], atol=1e-5)  # batch-size independent


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Which technologies are used to build Python APIs?", "fastapi"),
        ("What is the capital of Germany?", "berlin"),
        ("tool that checks data types in Python", "pydantic"),
        ("how do I version my database tables", "alembic"),
        ("searching quickly through embeddings", "faiss"),
        ("how is a loaf made", "bread"),
    ],
)
def test_real_semantic_search_ranks_the_right_passage_first(
    embedder, app, session, tmp_path, query, expected
):
    names = list(PASSAGES)
    _, ids = add_doc(session, [PASSAGES[n] for n in names])
    svc = VectorService(app.state.session_factory, tmp_path / "idx.bin", embedder)
    svc.sync()
    q = embedder.embed([query])[0]
    top = svc.search(q, 3)
    assert names[ids.index(top[0].entry.chunk_id)] == expected
    assert top[0].score > top[1].score and top[0].score > 0.2


def test_real_rebuild_is_logically_equivalent(embedder, app, session, tmp_path):
    """Same model + same chunks => same ranking and scores equal up to float32 noise."""
    names = list(PASSAGES)
    _, ids = add_doc(session, [PASSAGES[n] for n in names])
    svc = VectorService(app.state.session_factory, tmp_path / "idx.bin", embedder)
    svc.sync()
    q = embedder.embed(["Which technologies are used to build Python APIs?"])[0]
    before = svc.search(q, 8)
    before_vecs = {e.chunk_id: svc.index.vector(e.chunk_id).copy() for e in svc.indexed_entries()}

    svc.rebuild()
    after = svc.search(q, 8)
    assert [h.entry.chunk_id for h in before] == [h.entry.chunk_id for h in after]
    assert max(abs(a.score - b.score) for a, b in zip(before, after, strict=True)) < 1e-5
    for cid, vec in before_vecs.items():
        assert np.allclose(svc.index.vector(cid), vec, atol=1e-5)
    assert sorted(e.chunk_id for e in svc.indexed_entries()) == sorted(ids)


def test_real_index_persists_reloads_and_detects_a_config_change(embedder, app, session, tmp_path):
    names = list(PASSAGES)
    add_doc(session, [PASSAGES[n] for n in names])
    path = tmp_path / "idx.bin"
    svc = VectorService(app.state.session_factory, path, embedder)
    svc.sync()
    assert svc.snapshot_meta.model_revision == embedder.identity().revision

    reloaded = VectorService(app.state.session_factory, path, embedder)
    report = reloaded.initialize()
    assert report["source"] == "file" and report["sync"]["embedded"] == 0
    index, meta = vector_store.load_index(path)
    assert index.dimension == 384 and meta.model_id == MODEL

    # Same model files, different embedding configuration (shorter max sequence length):
    # vectors can differ, so the old index must be discarded rather than reused.
    changed = SentenceTransformerEmbedder(MODEL, cache_dir=CACHE, max_seq_length=32)
    assert changed.config_fingerprint != embedder.config_fingerprint
    svc2 = VectorService(app.state.session_factory, path, changed)
    rep = svc2.initialize()
    assert rep["sync"]["rebuilt"] is True and rep["sync"]["embedded"] == 8
    assert (
        Path(path).exists() and svc2.snapshot_meta.config_fingerprint == changed.config_fingerprint
    )
