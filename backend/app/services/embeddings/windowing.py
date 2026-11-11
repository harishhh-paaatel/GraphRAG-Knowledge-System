"""Oversize-text policy: never let the model silently drop the tail of a chunk.

Embedding models read a fixed number of tokens (``all-MiniLM-L6-v2``: 256, special tokens
included) and silently ignore the rest. Measured on the real model, the vector of a 1158-token
text had cosine 1.0000 with the vector of its first 256 tokens: the tail contributed nothing.
Token counts also cannot be derived from character counts (the same 800 characters were 143
tokens as prose but 361-489 tokens as URLs, digits or code), so this module only ever works
with counts produced by the model's own tokenizer.

Policy ``window-mean-v1`` (deterministic, documented in docs/architecture.md):

1. A text whose token count is <= the limit is embedded as is.
2. Otherwise it is split into *windows*, each <= the limit, at the safest available boundary:
   sentence ends first, then whitespace between words, and only for a single "word" that is
   itself over the limit, at the largest character prefix that fits. The split is lossless
   (the pieces concatenate back to the original text) and has no overlap.
3. Each window is embedded; the chunk's vector is the token-count-weighted mean of the window
   vectors, L2-normalized. One chunk still maps to exactly one vector, so the stable chunk-ID
   mapping, filtering and provenance (document, page, section) are unchanged.
4. The chunk text in SQLite is never modified. The index records ``token_count`` and
   ``windows`` per chunk so oversize handling is visible, never silent.
"""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from app.services.embeddings.errors import EmbeddingError

POLICY_ID = "window-mean-v1"

CountFn = Callable[[Sequence[str]], list[int]]
EncodeFn = Callable[[list[str]], np.ndarray]

_SENTENCE_END = re.compile(r"[.!?]+\s+")
_WORD_END = re.compile(r"\S+\s+")


@dataclass(frozen=True)
class TextInfo:
    """How the model sees one text."""

    token_count: int  # tokens of the whole text, counted by the model's own tokenizer
    windows: int  # model inputs needed: 1 = fits, >1 = pooled from several windows


def _cut(text: str, ends: Iterable[int]) -> list[str]:
    """Lossless pieces of ``text`` ending at the given offsets."""
    pieces, prev = [], 0
    for end in ends:
        if end > prev:
            pieces.append(text[prev:end])
            prev = end
    if prev < len(text):
        pieces.append(text[prev:])
    return pieces


def _hard_split(piece: str, count: CountFn, limit: int) -> list[str]:
    """Split one over-limit "word" (a URL, a long number) at the largest prefix that fits."""
    out, rest = [], piece
    while rest:
        lo, hi = 1, len(rest)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count([rest[:mid]])[0] <= limit:
                lo = mid
            else:
                hi = mid - 1
        if count([rest[:lo]])[0] > limit:
            raise ValueError(f"token limit {limit} is too small to hold even one character")
        out.append(rest[:lo])
        rest = rest[lo:]
    return out


def split_into_windows(text: str, count: CountFn, limit: int) -> list[str]:
    """Split ``text`` into windows of at most ``limit`` tokens (a single window if it fits)."""
    if limit < 3:
        raise ValueError("token limit must be >= 3")
    if count([text])[0] <= limit:
        return [text]

    atoms: list[str] = []
    for sentence in _cut(text, (m.end() for m in _SENTENCE_END.finditer(text))):
        if count([sentence])[0] <= limit:
            atoms.append(sentence)
            continue
        for word in _cut(sentence, (m.end() for m in _WORD_END.finditer(sentence))):
            if count([word])[0] <= limit:
                atoms.append(word)
            else:
                atoms.extend(_hard_split(word, count, limit))

    windows: list[str] = []
    current = ""
    for atom in atoms:
        if current and count([current + atom])[0] > limit:
            windows.append(current)
            current = atom
        else:
            current += atom
    if current:
        windows.append(current)

    windows = [w.strip() for w in windows if w.strip()]
    if not windows or any(c > limit for c in count(windows)):
        raise EmbeddingError("could not split oversized text into windows within the token limit")
    return windows


def describe_texts(texts: Sequence[str], *, count: CountFn, limit: int | None) -> list[TextInfo]:
    """Token count and window count per text, without embedding anything."""
    counts = count(texts)
    infos = []
    for text, n in zip(texts, counts, strict=True):
        windows = 1 if limit is None or n <= limit else len(split_into_windows(text, count, limit))
        infos.append(TextInfo(n, windows))
    return infos


def pool(vectors: np.ndarray, weights: Sequence[int]) -> np.ndarray:
    """Token-weighted mean of unit vectors, re-normalized to unit length."""
    w = np.asarray(weights, dtype=np.float64)
    mean = (vectors.astype(np.float64) * w[:, None]).sum(axis=0) / w.sum()
    norm = float(np.linalg.norm(mean))
    if norm < 1e-9:
        raise EmbeddingError("window vectors cancel out; cannot pool them into one vector")
    return (mean / norm).astype(np.float32)


def embed_with_policy(
    texts: Sequence[str], *, count: CountFn, limit: int | None, encode: EncodeFn
) -> np.ndarray:
    """Embed ``texts`` applying the oversize policy; one normalized vector per text.

    All model inputs (whole texts that fit plus the windows of oversize ones) go through a single
    ``encode`` call so batching stays efficient.
    """
    counts = count(texts)
    inputs: list[str] = []
    owner: list[int] = []
    weight: list[int] = []
    for i, (text, n) in enumerate(zip(texts, counts, strict=True)):
        if limit is None or n <= limit:
            inputs.append(text)
            owner.append(i)
            weight.append(n)
            continue
        windows = split_into_windows(text, count, limit)
        for window, c in zip(windows, count(windows), strict=True):
            inputs.append(window)
            owner.append(i)
            weight.append(c)

    raw = encode(inputs)
    if raw.shape[0] != len(inputs):
        raise EmbeddingError(f"encoder returned {raw.shape[0]} vectors for {len(inputs)} inputs")
    out = np.empty((len(texts), raw.shape[1]), dtype=np.float32)
    rows_of: dict[int, list[int]] = {}
    for row, i in enumerate(owner):
        rows_of.setdefault(i, []).append(row)
    for i, rows in rows_of.items():
        out[i] = raw[rows[0]] if len(rows) == 1 else pool(raw[rows], [weight[r] for r in rows])
    return out
