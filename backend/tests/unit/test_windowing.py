"""Token-limit policy: detection, windowing at safe boundaries, pooling, no silent truncation."""

import re

import numpy as np
import pytest

from app.services.embeddings import windowing
from app.services.embeddings.embedder import (
    EmbeddingError,
    EmbeddingInputError,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)
from app.services.embeddings.windowing import (
    describe_texts,
    embed_with_policy,
    pool,
    split_into_windows,
)
from tests.vector_helpers import install_fake_sentence_transformers


def words(n, prefix="w"):
    """n distinct single-token words."""
    return " ".join(f"{prefix}{i}" for i in range(n))


def count_words(texts):
    return [len(t.split()) for t in texts]


def count_chars(texts):  # one token per character: forces the hard-split path
    return [len(t) for t in texts]


# ---- boundary lengths --------------------------------------------------------------------


def test_text_exactly_at_the_limit_is_one_window():
    text = words(10)
    assert split_into_windows(text, count_words, 10) == [text]
    assert describe_texts([text], count=count_words, limit=10)[0].windows == 1


def test_text_one_token_over_the_limit_needs_more_than_one_window():
    text = words(11)
    windows = split_into_windows(text, count_words, 10)
    assert len(windows) == 2
    assert all(n <= 10 for n in count_words(windows))
    info = describe_texts([text], count=count_words, limit=10)[0]
    assert (info.token_count, info.windows) == (11, 2)


@pytest.mark.parametrize("n", [1, 9, 10, 11, 19, 20, 21, 95])
def test_every_window_fits_and_no_content_is_lost(n):
    text = words(n)
    windows = split_into_windows(text, count_words, 10)
    assert all(c <= 10 for c in count_words(windows))
    assert " ".join(windows).split() == text.split()  # lossless, in order, no overlap


def test_unlimited_embedder_never_windows():
    assert describe_texts([words(500)], count=count_words, limit=None)[0].windows == 1


def test_limit_must_be_usable():
    with pytest.raises(ValueError):
        split_into_windows("a b c", count_words, 2)


# ---- safe boundaries -------------------------------------------------------------------------


def test_windows_break_at_sentence_ends_when_possible():
    text = "Alpha one two three. Beta four five six. Gamma seven eight nine."
    windows = split_into_windows(text, count_words, 8)  # fits two sentences (3+... words each)
    assert windows == ["Alpha one two three. Beta four five six.", "Gamma seven eight nine."]


def test_a_long_sentence_is_split_between_words_not_inside_them():
    text = " ".join(f"word{i}" for i in range(30)) + "."
    windows = split_into_windows(text, count_words, 7)
    assert all(re.fullmatch(r"(word\d+ )*word\d+\.?", w) for w in windows)


def test_an_oversize_single_word_is_hard_split_by_characters():
    token = "".join(f"{i:03d}" for i in range(20))  # 60 chars, no whitespace
    windows = split_into_windows(token, count_chars, 25)
    assert [len(w) for w in windows] == [25, 25, 10]
    assert "".join(windows) == token


def test_splitting_is_deterministic():
    text = words(77)
    assert split_into_windows(text, count_words, 10) == split_into_windows(text, count_words, 10)


# ---- pooling ---------------------------------------------------------------------------------


def test_pooled_vector_is_the_token_weighted_mean_normalized():
    e1, e2 = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    vecs = np.vstack([e1, e2]).astype("float32")
    pooled = pool(vecs, [3, 1])
    expected = np.array([3.0, 1.0]) / np.linalg.norm([3.0, 1.0])
    assert np.allclose(pooled, expected, atol=1e-6)
    assert float(np.linalg.norm(pooled)) == pytest.approx(1.0, abs=1e-6)


def test_cancelling_windows_cannot_be_pooled():
    with pytest.raises(EmbeddingError, match="cancel"):
        pool(np.array([[1.0, 0.0], [-1.0, 0.0]], dtype="float32"), [2, 2])


def test_embed_with_policy_batches_everything_into_one_encode_call():
    calls = []

    def encode(inputs):
        calls.append(list(inputs))
        v = np.arange(1, 4 * len(inputs) + 1, dtype="float32").reshape(len(inputs), 4)
        return v / np.linalg.norm(v, axis=1, keepdims=True)  # unit rows, like a real encoder

    texts = [words(3), words(25, "x"), words(2, "y")]
    out = embed_with_policy(texts, count=count_words, limit=10, encode=encode)
    assert len(calls) == 1 and out.shape == (3, 4)
    assert all(n <= 10 for n in count_words(calls[0]))  # nothing handed to the model is oversize
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_encoder_returning_the_wrong_number_of_vectors_is_an_error():
    with pytest.raises(EmbeddingError, match="returned"):
        embed_with_policy(
            [words(3)],
            count=count_words,
            limit=10,
            encode=lambda inputs: np.ones((len(inputs) + 1, 4), dtype="float32"),
        )


# ---- the stub embedder end to end: the tail of a long text is NOT ignored -------------------


def test_tail_of_an_oversize_text_still_influences_its_vector():
    head = words(8, "head")
    tail = words(8, "tail")
    text = f"{head} {tail}"

    limited = HashingEmbedder(256, max_tokens=8)
    pooled = limited.embed([text])[0]
    only_tail = limited.embed([tail])[0]
    only_head = limited.embed([head])[0]
    assert float(pooled @ only_tail) > 0.5 and float(pooled @ only_head) > 0.5

    # What a silently truncating model would have produced: the head only.
    truncated = limited.embed([" ".join(text.split()[:8])])[0]
    assert float(truncated @ only_tail) < 0.2
    assert float(pooled @ only_tail) > float(truncated @ only_tail) + 0.4


def test_oversize_input_does_not_raise_and_is_normalized():
    e = HashingEmbedder(64, max_tokens=5)
    out = e.embed([words(40), "short text"])
    assert out.shape == (2, 64) and np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_describe_reports_tokens_and_windows_from_the_embedders_own_counter():
    e = HashingEmbedder(64, max_tokens=10)
    infos = e.describe(["one two three", words(25), "x"])
    assert [(i.token_count, i.windows) for i in infos] == [(3, 1), (25, 3), (1, 1)]


def test_token_count_is_not_character_count():
    e = HashingEmbedder(64, max_tokens=10)
    one_token = "a" * 40  # 40 characters, a single token
    many_tokens = "a b c d e f g h i j k l"  # 23 characters, 12 tokens
    first, second = e.describe([one_token, many_tokens])
    assert (first.token_count, first.windows) == (1, 1)
    assert second.token_count == 12 and second.windows == 2


def test_blank_text_is_still_rejected_before_any_windowing():
    with pytest.raises(EmbeddingInputError):
        HashingEmbedder(64, max_tokens=5).embed(["fine", "  "])


def test_policy_is_part_of_the_embedding_fingerprint():
    assert (
        HashingEmbedder(64).config_fingerprint
        != HashingEmbedder(64, max_tokens=5).config_fingerprint
    )
    assert windowing.POLICY_ID == "window-mean-v1"


# ---- the real-model wrapper (fake loader): counts come from the tokenizer --------------------


def test_sentence_transformer_embedder_never_sends_an_oversize_input_to_the_model(
    monkeypatch, tmp_path
):
    fake = install_fake_sentence_transformers(monkeypatch, max_seq_length=12)
    embedder = SentenceTransformerEmbedder("org/model", cache_dir=tmp_path / "m")
    out = embedder.embed([words(50), "tiny text"])
    assert out.shape == (2, 8) and np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    assert fake.encoded_inputs and all(len(t.split()) + 2 <= 12 for t in fake.encoded_inputs)
    assert embedder.identity().max_tokens == 12
    info = embedder.describe([words(50)])[0]
    assert info.token_count == 52 and info.windows > 1  # 50 words + 2 special tokens


def test_model_without_a_tokenizer_is_refused_instead_of_truncating_silently(monkeypatch, tmp_path):
    fake = install_fake_sentence_transformers(monkeypatch)

    original_init = fake.__init__

    def init_without_tokenizer(self, name, **kwargs):
        original_init(self, name, **kwargs)
        self.tokenizer = None

    monkeypatch.setattr(fake, "__init__", init_without_tokenizer)
    embedder = SentenceTransformerEmbedder("org/model", cache_dir=tmp_path / "m")
    with pytest.raises(EmbeddingError, match="truncated silently"):
        embedder.embed(["anything"])
