"""Embedding generation: normalization, dimensions, batching, validation, lazy loading."""

import sys
import threading
import types

import numpy as np
import pytest

from app.core.config import Settings
from app.services.embeddings.embedder import (
    EmbeddingError,
    EmbeddingInputError,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    normalize_rows,
    validate_texts,
)
from app.services.embeddings.factory import build_embedder

TEXTS = [
    "FastAPI is a web framework for building APIs with Python.",
    "Pydantic validates data using Python type hints.",
    "Berlin is the capital city of Germany.",
]


# ---- normalization, dimensions, batching (stub) ------------------------------------------


def test_vectors_are_unit_length_float32_with_configured_dimension():
    for dim in (64, 256):
        v = HashingEmbedder(dim).embed(TEXTS)
        assert v.shape == (3, dim) and v.dtype == np.float32
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)


def test_inner_product_of_normalized_vectors_is_cosine_similarity():
    v = HashingEmbedder().embed(TEXTS)
    cosine = (v[0] @ v[1]) / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]))
    assert float(v[0] @ v[1]) == pytest.approx(float(cosine), abs=1e-6)
    assert float(v[0] @ v[0]) == pytest.approx(1.0, abs=1e-6)


def test_deterministic_across_instances():
    assert np.array_equal(HashingEmbedder().embed(TEXTS), HashingEmbedder().embed(TEXTS))


def test_batch_result_equals_one_by_one_result():
    e = HashingEmbedder()
    batch = e.embed(TEXTS)
    singles = np.vstack([e.embed([t]) for t in TEXTS])
    assert np.array_equal(batch, singles)


def test_large_batch_preserves_order():
    e = HashingEmbedder()
    texts = [f"document number {i} about topic {i % 7}" for i in range(500)]
    out = e.embed(texts)
    assert out.shape[0] == 500
    assert np.array_equal(out[123], e.embed([texts[123]])[0])


def test_related_texts_score_higher_than_unrelated_texts():
    v = HashingEmbedder().embed(
        ["python web framework", "a python framework for the web", "capital of germany"]
    )
    assert v[0] @ v[1] > v[0] @ v[2] + 0.2


def test_empty_batch_returns_zero_rows():
    assert HashingEmbedder(32).embed([]).shape == (0, 32)


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_blank_text_is_rejected_not_embedded(bad):
    with pytest.raises(EmbeddingInputError, match="index"):
        HashingEmbedder().embed(["fine text", bad])


def test_all_offending_indexes_are_reported_and_non_strings_rejected():
    with pytest.raises(EmbeddingInputError) as exc:
        validate_texts(["ok", "", None, "ok", "  "])
    assert "[1, 2, 4]" in str(exc.value)
    with pytest.raises(EmbeddingInputError):
        validate_texts("a single string is not a batch")


def test_text_without_embeddable_tokens_is_rejected():
    with pytest.raises(EmbeddingInputError, match="no embeddable tokens"):
        HashingEmbedder().embed(["!!! ???"])


def test_counters_expose_how_much_was_embedded():
    e = HashingEmbedder()
    e.embed(TEXTS)
    e.embed(TEXTS[:1])
    assert (e.calls, e.texts_embedded) == (2, 4)


def test_fingerprint_changes_with_configuration_not_with_instance():
    assert HashingEmbedder(64).config_fingerprint == HashingEmbedder(64).config_fingerprint
    assert HashingEmbedder(64).config_fingerprint != HashingEmbedder(128).config_fingerprint
    ident = HashingEmbedder(64).identity()
    assert (ident.kind, ident.dimension) == ("hashing", 64)


def test_tiny_dimension_rejected():
    with pytest.raises(ValueError):
        HashingEmbedder(4)


# ---- normalize_rows ----------------------------------------------------------------------


def test_normalize_rows_accepts_and_fixes_scale():
    out = normalize_rows(np.array([[3.0, 4.0]]), expected_dim=2)
    assert np.allclose(out, [[0.6, 0.8]])


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
        np.array([[0.0, 0.0]]),
        np.array([1.0, 2.0]),
    ],
)
def test_normalize_rows_rejects_invalid_output(bad):
    with pytest.raises(EmbeddingError):
        normalize_rows(bad)


def test_normalize_rows_rejects_dimension_mismatch():
    with pytest.raises(EmbeddingError, match="dimension"):
        normalize_rows(np.ones((2, 3)), expected_dim=4)


# ---- real-model wrapper with a fake sentence_transformers (no download) -------------------


class FakeTokenizer:
    """Whitespace "tokenizer": one token per word, plus two special tokens (like [CLS]/[SEP])."""

    def __call__(self, texts, add_special_tokens=True, truncation=False, verbose=True):
        extra = 2 if add_special_tokens else 0
        return {"input_ids": [[0] * (len(t.split()) + extra) for t in texts]}


class FakeModel:
    loads = 0

    def __init__(self, name, **kwargs):
        FakeModel.loads += 1
        self.name, self.kwargs = name, kwargs
        self.max_seq_length = 128
        self.tokenizer = FakeTokenizer()
        self.encode_calls = []
        self.out_dim = 6

    def get_sentence_embedding_dimension(self):
        return self.out_dim

    def encode(self, texts, **kwargs):
        self.encode_calls.append((list(texts), kwargs))
        rng = np.random.default_rng(len(texts))
        return rng.normal(size=(len(texts), self.out_dim)).astype("float32")


@pytest.fixture
def fake_st(monkeypatch):
    FakeModel.loads = 0
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return FakeModel


def make_st(tmp_path, **kw):
    return SentenceTransformerEmbedder("org/model", cache_dir=tmp_path / "models", **kw)


def test_model_is_not_loaded_until_first_use_and_fingerprint_needs_no_load(fake_st, tmp_path):
    e = make_st(tmp_path)
    assert not e.is_loaded and fake_st.loads == 0
    assert len(e.config_fingerprint) == 64  # available without loading
    e.embed(["hello world"])
    assert e.is_loaded and fake_st.loads == 1


def test_model_loads_exactly_once_across_calls_and_threads(fake_st, tmp_path):
    e = make_st(tmp_path)
    threads = [threading.Thread(target=lambda: e.embed(["some text here"])) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    e.embed(["again"])
    assert fake_st.loads == 1


def test_wrapper_passes_configuration_and_normalizes_output(fake_st, tmp_path):
    e = make_st(
        tmp_path, revision="a" * 40, batch_size=7, device="cpu", max_seq_length=64, offline=True
    )
    out = e.embed(["one", "two", "three"])
    assert out.shape == (3, 6) and np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    model = e._model
    assert model.max_seq_length == 64
    assert model.kwargs["device"] == "cpu" and model.kwargs["revision"] == "a" * 40
    assert model.kwargs["local_files_only"] is True
    _, enc_kwargs = model.encode_calls[0]
    assert enc_kwargs["batch_size"] == 7 and enc_kwargs["normalize_embeddings"] is True
    assert enc_kwargs["show_progress_bar"] is False
    assert e.identity().dimension == 6 and e.identity().kind == "sentence_transformers"


def test_model_cache_is_kept_under_the_configured_directory(fake_st, tmp_path):
    e = make_st(tmp_path)
    e.embed(["x y"])
    assert e._model.kwargs["cache_folder"] == str(tmp_path / "models" / "hf" / "hub")
    assert (tmp_path / "models").is_dir()


def test_blank_input_rejected_before_the_model_is_even_loaded(fake_st, tmp_path):
    e = make_st(tmp_path)
    with pytest.raises(EmbeddingInputError):
        e.embed(["ok", " "])
    assert fake_st.loads == 0


def test_load_failure_is_an_embedding_error_and_is_retried(monkeypatch, tmp_path):
    attempts = {"n": 0}

    class Flaky(FakeModel):
        def __init__(self, *a, **k):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OSError("network down")
            super().__init__(*a, **k)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = Flaky
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    e = make_st(tmp_path)
    with pytest.raises(EmbeddingError, match="could not load"):
        e.embed(["text one"])
    assert not e.is_loaded
    assert e.embed(["text one"]).shape == (1, 6)  # a transient failure is not cached forever


def test_encode_failure_and_dimension_drift_are_embedding_errors(fake_st, tmp_path):
    e = make_st(tmp_path)
    e.embed(["warm up"])
    e._model.out_dim = 9  # model suddenly returns a different width than it announced
    with pytest.raises(EmbeddingError, match="dimension"):
        e.embed(["text"])

    def boom(*a, **k):
        raise RuntimeError("cuda oom")

    e._model.encode = boom
    with pytest.raises(EmbeddingError, match="generation failed"):
        e.embed(["text"])


def test_fingerprint_depends_on_model_revision_and_sequence_length(fake_st, tmp_path):
    base = make_st(tmp_path).config_fingerprint
    assert make_st(tmp_path).config_fingerprint == base
    assert make_st(tmp_path, revision="b" * 40).config_fingerprint != base
    assert make_st(tmp_path, max_seq_length=64).config_fingerprint != base
    assert SentenceTransformerEmbedder("org/other").config_fingerprint != base
    # batch size and device do not change vector values, so they must not invalidate an index
    assert make_st(tmp_path, batch_size=1, device="cpu").config_fingerprint == base


def test_factory_builds_the_configured_embedder(tmp_path):
    hashing = Settings(data_dir=tmp_path, embedding_provider="hashing", _env_file=None)
    assert isinstance(build_embedder(hashing), HashingEmbedder)
    st = Settings(
        data_dir=tmp_path,
        embedding_provider="sentence_transformers",
        embedding_model="org/m",
        embedding_batch_size=16,
        _env_file=None,
    )
    emb = build_embedder(st)
    assert isinstance(emb, SentenceTransformerEmbedder) and emb.model_id == "org/m"
    assert not emb.is_loaded  # building the service never downloads anything
