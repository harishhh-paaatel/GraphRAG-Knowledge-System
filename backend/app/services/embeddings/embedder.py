"""Embedding generation.

Two implementations behind one small interface:

* ``SentenceTransformerEmbedder`` - the real model. CPU-first, loaded lazily and exactly once.
* ``HashingEmbedder`` - a deterministic feature-hashing bag-of-words encoder used by the
  default test suite so no model download is needed. It is a genuine (lexical) embedder with
  real similarity structure, but it is NOT semantic; it is never selected by default.

Every embedder returns L2-normalized float32 vectors, so FAISS inner product equals cosine
similarity. Empty or blank input is rejected, never embedded: there is nothing to represent
and inventing a vector would be fabrication.

Over-long input is never silently truncated: see ``windowing`` for the documented policy.
Token counts always come from the model's own tokenizer (``HashingEmbedder`` counts its own
regex tokens), never from character counts.
"""

import hashlib
import json
import logging
import os
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.services.embeddings.errors import EmbeddingError, EmbeddingInputError
from app.services.embeddings.windowing import (
    POLICY_ID,
    TextInfo,
    describe_texts,
    embed_with_policy,
)

__all__ = [
    "Embedder",
    "EmbeddingError",
    "EmbeddingIdentity",
    "EmbeddingInputError",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "TextInfo",
    "fingerprint",
    "normalize_rows",
    "validate_texts",
]

logger = logging.getLogger(__name__)

_NORM_TOLERANCE = 1e-3
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class EmbeddingIdentity:
    """What produced a vector. Vectors from different identities must never share an index."""

    kind: str
    model_id: str
    revision: str | None  # resolved model revision (HF commit) when known
    dimension: int
    config_fingerprint: str
    max_tokens: int | None = None  # the model's real input limit (special tokens included)


class Embedder(Protocol):
    """Anything that turns texts into normalized vectors."""

    model_id: str
    config_fingerprint: str  # computable WITHOUT loading the model

    @property
    def is_loaded(self) -> bool: ...

    def identity(self) -> EmbeddingIdentity:
        """Full identity (may load the model to learn the dimension, revision and token limit)."""
        ...

    def describe(self, texts: Sequence[str]) -> list[TextInfo]:
        """Token count and window count per text, using the model's own tokenizer."""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dimension), rows L2-normalized."""
        ...


def fingerprint(config: dict) -> str:
    """Stable digest of the settings that determine vector values."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_texts(texts: Sequence[str]) -> list[str]:
    """Reject non-strings and blank strings (reporting every offending index)."""
    if isinstance(texts, str):
        raise EmbeddingInputError("texts must be a sequence of strings, not a single string")
    bad = [i for i, t in enumerate(texts) if not isinstance(t, str) or not t.strip()]
    if bad:
        raise EmbeddingInputError(f"empty or non-string text at index(es) {bad[:10]}")
    return list(texts)


def normalize_rows(vectors: np.ndarray, expected_dim: int | None = None) -> np.ndarray:
    """Validate shape/finiteness and L2-normalize rows (guards against float drift)."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise EmbeddingError(f"embedding output must be 2-D, got shape {arr.shape}")
    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise EmbeddingError(f"embedding dimension {arr.shape[1]} != expected {expected_dim}")
    if not np.isfinite(arr).all():
        raise EmbeddingError("embedding output contains NaN or infinity")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if (norms < 1e-12).any():
        raise EmbeddingError("embedding output contains a zero vector")
    out = (arr / norms).astype(np.float32)
    if not np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=_NORM_TOLERANCE):
        raise EmbeddingError("embeddings could not be normalized")
    return out


# --- deterministic stub ------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def _hashing_count(texts: Sequence[str]) -> list[int]:
    return [len(_TOKEN.findall(t.casefold())) for t in texts]


class HashingEmbedder:
    """Signed feature-hashing bag-of-words. Deterministic across processes and platforms.

    ``max_tokens`` (default: unlimited) gives the stub a token limit so the oversize policy can
    be exercised without a model; its "tokens" are the lowercase alphanumeric runs it hashes.
    """

    def __init__(self, dimension: int = 256, max_tokens: int | None = None) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        if max_tokens is not None and max_tokens < 3:
            raise ValueError("max_tokens must be >= 3")
        self._dim = dimension
        self._max_tokens = max_tokens
        self.model_id = f"hashing-bow-v1/dim{dimension}" + (
            f"/max{max_tokens}" if max_tokens else ""
        )
        self.config_fingerprint = fingerprint(
            {
                "kind": "hashing",
                "version": 1,
                "dimension": dimension,
                "max_tokens": max_tokens,
                "normalize": True,
                "oversize_policy": POLICY_ID,
            }
        )
        self.calls = 0  # number of embed() invocations (lets tests prove nothing was recomputed)
        self.texts_embedded = 0

    @property
    def is_loaded(self) -> bool:
        return True

    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            "hashing", self.model_id, "1", self._dim, self.config_fingerprint, self._max_tokens
        )

    def describe(self, texts: Sequence[str]) -> list[TextInfo]:
        items = validate_texts(texts)
        return describe_texts(items, count=_hashing_count, limit=self._max_tokens)

    def _encode(self, inputs: list[str]) -> np.ndarray:
        out = np.zeros((len(inputs), self._dim), dtype=np.float32)
        for row, text in enumerate(inputs):
            tokens = _TOKEN.findall(text.casefold())
            if not tokens:
                raise EmbeddingInputError(f"text at index {row} has no embeddable tokens")
            for token in tokens:
                h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
                out[row, h % self._dim] += 1.0 if (h >> 40) & 1 else -1.0
        return normalize_rows(out, self._dim)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        items = validate_texts(texts)
        self.calls += 1
        self.texts_embedded += len(items)
        if not items:
            return np.zeros((0, self._dim), dtype=np.float32)
        return embed_with_policy(
            items, count=_hashing_count, limit=self._max_tokens, encode=self._encode
        )


# --- real model --------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    """Sentence Transformers model on CPU (or the configured device), loaded once, lazily.

    ``revision`` pins an exact Hugging Face commit. When it is a 40-character commit hash the
    loaded snapshot is verified to be exactly that commit; a different snapshot is an error,
    never a silent substitution. ``offline`` forbids network access (cache only).
    """

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        batch_size: int = 32,
        device: str = "cpu",
        max_seq_length: int | None = None,
        cache_dir: Path | None = None,
        offline: bool = False,
    ) -> None:
        self.model_id = model_name
        self._revision = revision
        self._batch_size = batch_size
        self._device = device
        self._max_seq_length = max_seq_length
        self._cache_dir = cache_dir
        self._offline = offline
        self._model = None
        self._identity: EmbeddingIdentity | None = None
        self._load_lock = threading.Lock()
        # Torch modules and HF fast tokenizers are not safe for concurrent use.
        self._encode_lock = threading.Lock()
        # Computed from settings only, so the index can be validated without loading the model.
        self.config_fingerprint = fingerprint(
            {
                "kind": "sentence_transformers",
                "model": model_name,
                "requested_revision": revision,
                "max_seq_length": max_seq_length,
                "normalize": True,
                "oversize_policy": POLICY_ID,
            }
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_error(self, exc: BaseException) -> EmbeddingError:
        rev = f" at revision {self._revision[:12]}" if self._revision else ""
        if self._offline:
            return EmbeddingError(
                f"embedding model {self.model_id!r}{rev} is not available (or is incomplete) in "
                "the local model cache and offline mode is on (GRAPHRAG_EMBEDDING_OFFLINE=true). "
                "Download it once with network access, or point GRAPHRAG_EMBEDDING_CACHE_DIR at a "
                f"cache that contains it. No other model is substituted. [{type(exc).__name__}]"
            )
        return EmbeddingError(
            f"could not load embedding model {self.model_id!r}{rev}: {type(exc).__name__}. "
            "Check the model name/revision and network access, or enable offline mode with a "
            "populated cache. No other model is substituted."
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:  # another thread finished first
                return
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            if self._cache_dir is not None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("HF_HOME", str(self._cache_dir / "hf"))
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(
                    self.model_id,
                    device=self._device,
                    revision=self._revision,
                    cache_folder=str(self._cache_dir / "hf" / "hub") if self._cache_dir else None,
                    local_files_only=self._offline,
                )
                if self._max_seq_length:
                    model.max_seq_length = self._max_seq_length
                dimension = int(model.get_sentence_embedding_dimension())
            except Exception as exc:
                logger.error(
                    "Embedding model failed to load: %s (%s)", self.model_id, type(exc).__name__
                )
                raise self._load_error(exc) from exc

            limit = getattr(model, "max_seq_length", None)
            if getattr(model, "tokenizer", None) is None or not isinstance(limit, int) or limit < 3:
                raise EmbeddingError(
                    f"cannot determine the token limit of {self.model_id!r} (no tokenizer or "
                    "max_seq_length); refusing to embed because input could be truncated silently"
                )
            revision = self._verified_revision()
            self._identity = EmbeddingIdentity(
                "sentence_transformers",
                self.model_id,
                revision,
                dimension,
                self.config_fingerprint,
                limit,
            )
            self._model = model
            logger.info(
                "Embedding model loaded: %s revision=%s dim=%d max_tokens=%d device=%s",
                self.model_id,
                revision,
                dimension,
                limit,
                self._device,
            )

    def _resolve_revision(self) -> str | None:
        """The commit hash of the cached snapshot (the snapshot folder name), if findable."""
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(
                self.model_id,
                "config.json",
                cache_dir=str(self._cache_dir / "hf" / "hub") if self._cache_dir else None,
                revision=self._revision,
            )
            if isinstance(cached, str):
                name = Path(cached).parent.name
                if _COMMIT.fullmatch(name):
                    return name
        except Exception:
            logger.debug("Could not resolve model revision", exc_info=True)
        return None

    def _verified_revision(self) -> str | None:
        """The loaded revision. A pinned commit must match exactly or loading fails."""
        resolved = self._resolve_revision()
        pinned = self._revision if self._revision and _COMMIT.fullmatch(self._revision) else None
        if pinned is None:
            return resolved
        if resolved is not None and resolved != pinned:
            raise EmbeddingError(
                f"loaded snapshot {resolved[:12]} of {self.model_id!r} does not match the pinned "
                f"revision {pinned[:12]}; refusing to use a different model snapshot"
            )
        return pinned

    def identity(self) -> EmbeddingIdentity:
        self._load()
        assert self._identity is not None
        return self._identity

    # -- tokenization / encoding (both under the encode lock) -------------------------------

    def _count(self, texts: Sequence[str]) -> list[int]:
        """Token counts from the model's own tokenizer, special tokens included."""
        try:
            with self._encode_lock:
                enc = self._model.tokenizer(
                    list(texts), add_special_tokens=True, truncation=False, verbose=False
                )
        except Exception as exc:
            raise EmbeddingError(f"tokenization failed: {type(exc).__name__}") from exc
        return [len(ids) for ids in enc["input_ids"]]

    def _encode(self, inputs: list[str]) -> np.ndarray:
        try:
            with self._encode_lock:
                raw = self._model.encode(
                    inputs,
                    batch_size=self._batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
        except Exception as exc:
            logger.exception("Embedding generation failed")
            raise EmbeddingError(f"embedding generation failed: {type(exc).__name__}") from exc
        assert self._identity is not None
        return normalize_rows(raw, self._identity.dimension)

    def describe(self, texts: Sequence[str]) -> list[TextInfo]:
        items = validate_texts(texts)
        self._load()
        assert self._identity is not None
        return describe_texts(items, count=self._count, limit=self._identity.max_tokens)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        items = validate_texts(texts)
        self._load()
        assert self._identity is not None
        if not items:
            return np.zeros((0, self._identity.dimension), dtype=np.float32)
        return embed_with_policy(
            items, count=self._count, limit=self._identity.max_tokens, encode=self._encode
        )
