"""Pinned model revision, revision/config mismatch handling, and offline behaviour.

Nothing here needs the internet or a real model: a fake ``sentence_transformers`` stands in,
and ``no_network`` makes any attempted connection fail the test.
"""

import socket

import numpy as np
import pytest

from app.core.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_REVISION, Settings
from app.services.embeddings.embedder import EmbeddingError, SentenceTransformerEmbedder
from app.services.embeddings.factory import build_embedder
from tests.vector_helpers import install_fake_sentence_transformers

PIN_A = "a" * 40
PIN_B = "b" * 40


@pytest.fixture
def no_network(monkeypatch):
    def deny(*_a, **_k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def make(tmp_path, **kw):
    return SentenceTransformerEmbedder("org/model", cache_dir=tmp_path / "m", **kw)


def fake_cache_lookup(monkeypatch, snapshot_hash):
    """Make huggingface_hub report that the cached snapshot is ``snapshot_hash``."""
    import huggingface_hub

    path = None if snapshot_hash is None else f"/cache/snapshots/{snapshot_hash}/config.json"
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: path)


# ---- settings: the verified revision is pinned by default -------------------------------------


def test_the_default_model_is_pinned_to_the_verified_commit(tmp_path):
    s = Settings(data_dir=tmp_path, _env_file=None)
    assert s.embedding_model == DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.embedding_model_revision == DEFAULT_EMBEDDING_REVISION
    assert DEFAULT_EMBEDDING_REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def test_the_default_pin_never_leaks_onto_a_different_model(tmp_path):
    other = Settings(data_dir=tmp_path, embedding_model="org/other", _env_file=None)
    assert other.embedding_model_revision is None
    explicit = Settings(
        data_dir=tmp_path,
        embedding_model="org/other",
        embedding_model_revision=PIN_A,
        _env_file=None,
    )
    assert explicit.embedding_model_revision == PIN_A


@pytest.mark.parametrize("bad", ["main", "abc123", "A" * 40, "g" * 40, "a" * 39, "a" * 41])
def test_only_exact_commit_hashes_are_accepted_as_a_pin(tmp_path, bad):
    with pytest.raises(ValueError, match="40-character commit hash"):
        Settings(data_dir=tmp_path, embedding_model_revision=bad, _env_file=None)


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_an_empty_revision_means_explicitly_unpinned(tmp_path, empty):
    assert (
        Settings(
            data_dir=tmp_path, embedding_model_revision=empty, _env_file=None
        ).embedding_model_revision
        is None
    )


def test_revision_can_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_MODEL_REVISION", PIN_B)
    assert Settings(data_dir=tmp_path, _env_file=None).embedding_model_revision == PIN_B
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_MODEL_REVISION", "")
    assert Settings(data_dir=tmp_path, _env_file=None).embedding_model_revision is None


def test_the_factory_hands_the_pin_to_the_loader(monkeypatch, tmp_path):
    fake = install_fake_sentence_transformers(monkeypatch)
    fake_cache_lookup(monkeypatch, DEFAULT_EMBEDDING_REVISION)
    settings = Settings(
        data_dir=tmp_path, embedding_provider="sentence_transformers", _env_file=None
    )
    embedder = build_embedder(settings)
    embedder.embed(["hello there"])
    name, kwargs = fake.created[0]
    assert name == DEFAULT_EMBEDDING_MODEL and kwargs["revision"] == DEFAULT_EMBEDDING_REVISION
    assert embedder.identity().revision == DEFAULT_EMBEDDING_REVISION


# ---- the fingerprint covers everything that changes vectors ----------------------------------


def test_fingerprint_covers_model_revision_length_and_policy(tmp_path):
    base = make(tmp_path, revision=PIN_A).config_fingerprint
    assert make(tmp_path, revision=PIN_A).config_fingerprint == base
    assert make(tmp_path, revision=PIN_B).config_fingerprint != base  # revision
    assert make(tmp_path, revision=None).config_fingerprint != base  # pinned vs floating
    assert make(tmp_path, revision=PIN_A, max_seq_length=64).config_fingerprint != base
    assert SentenceTransformerEmbedder("org/other", revision=PIN_A).config_fingerprint != base
    # batch size / device / offline do not change vector values: they must not invalidate an index
    same = make(tmp_path, revision=PIN_A, batch_size=1, device="cpu", offline=True)
    assert same.config_fingerprint == base


# ---- a pinned revision is verified, never silently substituted -----------------------------


def test_loaded_snapshot_matching_the_pin_is_accepted(monkeypatch, tmp_path):
    install_fake_sentence_transformers(monkeypatch)
    fake_cache_lookup(monkeypatch, PIN_A)
    assert make(tmp_path, revision=PIN_A).identity().revision == PIN_A


def test_a_different_cached_snapshot_than_the_pin_is_refused(monkeypatch, tmp_path):
    install_fake_sentence_transformers(monkeypatch)
    fake_cache_lookup(monkeypatch, PIN_B)  # the cache says a different commit is what loaded
    with pytest.raises(EmbeddingError, match="does not match the pinned revision"):
        make(tmp_path, revision=PIN_A).embed(["text"])


def test_a_pin_is_trusted_when_the_cache_layout_cannot_be_inspected(monkeypatch, tmp_path):
    install_fake_sentence_transformers(monkeypatch)
    fake_cache_lookup(monkeypatch, None)
    assert make(tmp_path, revision=PIN_A).identity().revision == PIN_A  # the loader got the pin


def test_without_a_pin_the_resolved_revision_is_reported_or_unknown(monkeypatch, tmp_path):
    install_fake_sentence_transformers(monkeypatch)
    fake_cache_lookup(monkeypatch, PIN_B)
    assert make(tmp_path).identity().revision == PIN_B
    fake_cache_lookup(monkeypatch, None)
    assert make(tmp_path).identity().revision is None


# ---- offline behaviour -------------------------------------------------------------------------


def test_a_cached_model_loads_offline_without_touching_the_network(
    monkeypatch, tmp_path, no_network
):
    fake = install_fake_sentence_transformers(monkeypatch)
    fake_cache_lookup(monkeypatch, PIN_A)
    embedder = make(tmp_path, revision=PIN_A, offline=True)
    out = embedder.embed(["offline works", "and so does a second text"])
    assert out.shape == (2, 8) and np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert fake.created[0][1]["local_files_only"] is True
    assert fake.created[0][1]["revision"] == PIN_A


def test_a_missing_cache_in_offline_mode_gives_a_clear_actionable_error(monkeypatch, tmp_path):
    fake = install_fake_sentence_transformers(
        monkeypatch, fail=OSError("we couldn't find the model files in the local cache")
    )
    embedder = make(tmp_path, revision=PIN_A, offline=True)
    with pytest.raises(EmbeddingError) as exc:
        embedder.embed(["anything"])
    message = str(exc.value)
    assert "org/model" in message and PIN_A[:12] in message
    assert "offline mode is on" in message and "GRAPHRAG_EMBEDDING_OFFLINE" in message
    assert "GRAPHRAG_EMBEDDING_CACHE_DIR" in message
    assert "No other model is substituted" in message
    assert str(tmp_path) not in message  # no filesystem paths leak into API-facing messages
    assert not embedder.is_loaded
    assert {name for name, _ in fake.created} == {"org/model"}  # never tried a fallback model


def test_a_failure_to_load_online_is_worded_differently_and_also_never_substitutes(
    monkeypatch, tmp_path
):
    fake = install_fake_sentence_transformers(monkeypatch, fail=ConnectionError("unreachable"))
    with pytest.raises(EmbeddingError) as exc:
        make(tmp_path).embed(["anything"])
    assert "could not load embedding model" in str(exc.value)
    assert "network access" in str(exc.value) and "No other model is substituted" in str(exc.value)
    assert "offline mode is on" not in str(exc.value)
    assert {name for name, _ in fake.created} == {"org/model"}


def test_every_embedding_call_fails_the_same_way_until_the_model_is_available(
    monkeypatch, tmp_path
):
    install_fake_sentence_transformers(monkeypatch, fail=OSError("missing"))
    embedder = make(tmp_path, offline=True)
    for _ in range(3):
        with pytest.raises(EmbeddingError, match="offline mode is on"):
            embedder.embed(["text"])
    assert not embedder.is_loaded  # a failed load is never cached as a loaded model
