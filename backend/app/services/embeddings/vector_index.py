"""Exact cosine-similarity vector index (FAISS ``IndexFlatIP`` over L2-normalized vectors).

FAISS positions are an implementation detail: removing vectors shifts every later position
down, so a position is never an identity. Identity is the chunk ID, kept in a parallel,
persisted entry list, and every public method speaks chunk IDs only.

Two small protocols describe the engine boundary and are used as the types everywhere above
this module (the service, the file store and the snapshot never name FAISS):

* ``VectorIndex`` - one in-memory index (add / remove / search / clone / serialize ...).
* ``IndexEngine`` - creates and deserializes indexes and has a ``name`` that is written into
  the index file, so a file is only ever loaded by the engine that produced it.

FAISS is the only implementation. There is deliberately no second one: the protocols document
the boundary the code really uses, they do not claim a pluggability that was never built.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import faiss
import numpy as np

_NORM_TOLERANCE = 1e-3


class VectorIndexError(Exception):
    """Base class for index problems."""


class DuplicateChunkError(VectorIndexError):
    """A chunk ID was added twice."""


class DimensionMismatchError(VectorIndexError):
    """Vectors do not match the index dimension."""


class IndexCorruptError(VectorIndexError):
    """An index or its file failed validation."""


@dataclass(frozen=True)
class IndexEntry:
    """Identity of one stored vector."""

    chunk_id: str
    document_id: str
    content_hash: str
    token_count: int = 0  # tokens of the whole chunk per the model tokenizer (0 = unknown)
    windows: int = 1  # >1: the chunk exceeded the token limit and its vector is pooled

    def validate(self) -> None:
        if not (self.chunk_id and self.document_id and self.content_hash):
            raise VectorIndexError(f"index entry is missing metadata: {self!r}")
        if self.token_count < 0 or self.windows < 1:
            raise VectorIndexError(f"index entry has invalid size information: {self!r}")


@dataclass(frozen=True)
class SearchHit:
    entry: IndexEntry
    score: float  # cosine similarity in [-1, 1]


@runtime_checkable
class VectorIndex(Protocol):
    """Everything the service, store and snapshot need from an index."""

    @property
    def dimension(self) -> int: ...
    @property
    def count(self) -> int: ...
    def entries(self) -> list[IndexEntry]: ...
    def vector(self, chunk_id: str) -> np.ndarray: ...
    def add(self, entries: Sequence[IndexEntry], vectors: np.ndarray) -> None: ...
    def remove(self, chunk_ids: Collection[str]) -> int: ...
    def search(
        self,
        query: np.ndarray,
        top_k: int,
        *,
        document_ids: Collection[str] | None = None,
        exclude_document_ids: Collection[str] | None = None,
    ) -> list[SearchHit]: ...
    def clone(self) -> "VectorIndex": ...
    def serialize(self) -> bytes: ...


@runtime_checkable
class IndexEngine(Protocol):
    """Creates indexes and restores them from their serialized form."""

    name: str

    def create(self, dimension: int) -> VectorIndex: ...
    def deserialize(
        self, dimension: int, entries: Sequence[IndexEntry], payload: bytes
    ) -> VectorIndex: ...


class FaissFlatIPIndex:
    """FAISS exact inner-product index with a stable chunk-ID mapping."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self._d = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._entries: list[IndexEntry] = []  # parallel to FAISS positions (internal only)
        self._pos: dict[str, int] = {}

    # -- introspection ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._d

    @property
    def count(self) -> int:
        return len(self._entries)

    def entries(self) -> list[IndexEntry]:
        return list(self._entries)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._pos

    def entry(self, chunk_id: str) -> IndexEntry:
        return self._entries[self._pos[chunk_id]]

    def vector(self, chunk_id: str) -> np.ndarray:
        """The stored vector for ``chunk_id`` (used by tests and diagnostics)."""
        return self._index.reconstruct(self._pos[chunk_id])

    # -- mutation -----------------------------------------------------------------------

    def add(self, entries: Sequence[IndexEntry], vectors: np.ndarray) -> None:
        """Add vectors. Validates everything first so a failed call changes nothing."""
        arr = np.asarray(vectors)
        if arr.ndim != 2 or arr.shape[0] != len(entries):
            raise DimensionMismatchError(
                f"expected {len(entries)} vectors as a 2-D array, got shape {arr.shape}"
            )
        if arr.shape[1] != self._d:
            raise DimensionMismatchError(f"vector dimension {arr.shape[1]} != index {self._d}")
        if len(entries) == 0:
            return
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise VectorIndexError("vectors contain NaN or infinity")
        if not np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=_NORM_TOLERANCE):
            raise VectorIndexError("vectors must be L2-normalized for cosine similarity")
        seen: set[str] = set()
        for e in entries:
            e.validate()
            if e.chunk_id in self._pos or e.chunk_id in seen:
                raise DuplicateChunkError(f"chunk already indexed: {e.chunk_id}")
            seen.add(e.chunk_id)
        self._index.add(arr)
        for e in entries:
            self._pos[e.chunk_id] = len(self._entries)
            self._entries.append(e)

    def remove(self, chunk_ids: Collection[str]) -> int:
        """Remove vectors by chunk ID; unknown IDs are ignored. Returns how many were removed."""
        positions = sorted({self._pos[c] for c in chunk_ids if c in self._pos})
        if not positions:
            return 0
        self._index.remove_ids(np.array(positions, dtype="int64"))
        drop = set(positions)
        self._entries = [e for i, e in enumerate(self._entries) if i not in drop]
        self._pos = {e.chunk_id: i for i, e in enumerate(self._entries)}
        if self._index.ntotal != len(self._entries):  # pragma: no cover - would be a FAISS bug
            raise IndexCorruptError("FAISS and ID mapping diverged after removal")
        return len(positions)

    # -- search -------------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        *,
        document_ids: Collection[str] | None = None,
        exclude_document_ids: Collection[str] | None = None,
    ) -> list[SearchHit]:
        """Exact top-K by cosine similarity, optionally restricted to / excluding documents.

        Ordering is deterministic: score descending, ties broken by chunk ID. The tie rule is
        exact even at the cut-off: if the K-th and (K+1)-th scores are equal the search widens
        until the whole tie group is seen, so which tied chunk survives never depends on FAISS
        internals.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._d:
            raise DimensionMismatchError(f"query dimension {q.shape[0]} != index {self._d}")
        if not np.isfinite(q).all() or abs(float(np.linalg.norm(q)) - 1.0) > _NORM_TOLERANCE:
            raise VectorIndexError("query vector must be finite and L2-normalized")
        if not self._entries:
            return []

        allowed: np.ndarray | None = None
        if document_ids is not None or exclude_document_ids is not None:
            wanted = set(document_ids) if document_ids is not None else None
            banned = set(exclude_document_ids or ())
            positions = [
                i
                for i, e in enumerate(self._entries)
                if (wanted is None or e.document_id in wanted) and e.document_id not in banned
            ]
            if not positions:
                return []
            allowed = np.array(positions, dtype="int64")

        eligible = len(self._entries) if allowed is None else len(allowed)
        k = min(top_k + 1, eligible)  # one extra to see whether a tie straddles the cut-off
        while True:
            hits = self._raw_search(q, k, allowed)
            tie_at_cutoff = len(hits) > top_k and hits[top_k][1] == hits[top_k - 1][1]
            if not tie_at_cutoff or k >= eligible:
                break
            k = min(eligible, k * 2)
        hits.sort(key=lambda h: (-h[1], self._entries[h[0]].chunk_id))
        return [SearchHit(self._entries[pos], float(score)) for pos, score in hits[:top_k]]

    def _raw_search(
        self, q: np.ndarray, k: int, allowed: np.ndarray | None
    ) -> list[tuple[int, float]]:
        row = q.reshape(1, -1)
        if allowed is None:
            scores, positions = self._index.search(row, k)
        else:
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(allowed)
            scores, positions = self._index.search(row, k, params=params)
        return [(int(p), float(s)) for p, s in zip(positions[0], scores[0], strict=True) if p >= 0]

    # -- serialization ------------------------------------------------------------------

    def serialize(self) -> bytes:
        return faiss.serialize_index(self._index).tobytes()

    @classmethod
    def deserialize(
        cls, dimension: int, entries: Sequence[IndexEntry], payload: bytes
    ) -> "FaissFlatIPIndex":
        """Rebuild from persisted parts, validating that they agree with each other."""
        try:
            raw = faiss.deserialize_index(np.frombuffer(payload, dtype="uint8"))
        except Exception as exc:
            raise IndexCorruptError("FAISS payload could not be deserialized") from exc
        if not isinstance(raw, faiss.IndexFlatIP):
            raise IndexCorruptError(f"unexpected FAISS index type {type(raw).__name__}")
        if raw.d != dimension:
            raise IndexCorruptError(f"payload dimension {raw.d} != header dimension {dimension}")
        if raw.ntotal != len(entries):
            raise IndexCorruptError(
                f"payload holds {raw.ntotal} vectors but the ID table has {len(entries)}"
            )
        ids = [e.chunk_id for e in entries]
        if len(set(ids)) != len(ids):
            raise IndexCorruptError("duplicate chunk IDs in the ID table")
        for e in entries:
            try:
                e.validate()
            except VectorIndexError as exc:
                raise IndexCorruptError(str(exc)) from exc
        index = cls(dimension)
        index._index = raw
        index._entries = list(entries)
        index._pos = {e.chunk_id: i for i, e in enumerate(index._entries)}
        return index

    def clone(self) -> "FaissFlatIPIndex":
        return FaissFlatIPIndex.deserialize(self._d, self._entries, self.serialize())


class FaissFlatIPEngine:
    """The FAISS exact inner-product engine (the only engine)."""

    name = "faiss-flat-ip"

    def create(self, dimension: int) -> FaissFlatIPIndex:
        return FaissFlatIPIndex(dimension)

    def deserialize(
        self, dimension: int, entries: Sequence[IndexEntry], payload: bytes
    ) -> FaissFlatIPIndex:
        return FaissFlatIPIndex.deserialize(dimension, entries, payload)


FAISS_ENGINE = FaissFlatIPEngine()
