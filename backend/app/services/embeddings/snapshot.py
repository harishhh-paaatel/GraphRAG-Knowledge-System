"""An immutable, self-consistent index snapshot.

A reader takes ONE reference to an ``IndexSnapshot`` and uses it for its whole query, so the
index, its chunk-ID table and its metadata (revision, model, dimension, fingerprint) always
belong together. The service publishes a new snapshot with a single attribute assignment,
and only after ``validate`` has passed; a snapshot that fails validation is never published,
so a half-built or mismatched index cannot be observed.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np

from app.services.embeddings.vector_index import (
    IndexCorruptError,
    IndexEntry,
    SearchHit,
    VectorIndex,
)
from app.services.embeddings.vector_store import IndexMeta


@dataclass(frozen=True)
class IndexSnapshot:
    """An index (``None`` = valid empty index) together with the metadata describing it."""

    index: VectorIndex | None
    meta: IndexMeta

    @property
    def count(self) -> int:
        return self.index.count if self.index is not None else 0

    def entries(self) -> list[IndexEntry]:
        return self.index.entries() if self.index is not None else []

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        *,
        document_ids: Collection[str] | None = None,
        exclude_document_ids: Collection[str] | None = None,
    ) -> list[SearchHit]:
        if self.index is None:
            return []
        return self.index.search(
            query, top_k, document_ids=document_ids, exclude_document_ids=exclude_document_ids
        )

    def validate(self, expected: Mapping[str, tuple[str, str]] | None = None) -> None:
        """Raise ``IndexCorruptError`` unless index and metadata agree with each other.

        With ``expected`` (``{chunk_id: (document_id, content_hash)}``) the snapshot must also
        contain exactly those chunks: nothing missing, nothing extra, no stale content.
        """
        entries = self.entries()
        if self.index is None:
            if self.meta.dimension is not None:
                raise IndexCorruptError("empty snapshot declares a vector dimension")
        else:
            if self.index.count == 0:
                raise IndexCorruptError("an empty index must be represented as None")
            if self.meta.dimension != self.index.dimension:
                raise IndexCorruptError(
                    f"metadata dimension {self.meta.dimension} != index dimension "
                    f"{self.index.dimension}"
                )
            if len(entries) != self.index.count:
                raise IndexCorruptError(
                    f"ID table has {len(entries)} entries but the index holds {self.index.count}"
                )
        ids = [e.chunk_id for e in entries]
        if len(set(ids)) != len(ids):
            raise IndexCorruptError("duplicate chunk IDs in the snapshot")
        if expected is not None:
            actual = {e.chunk_id: (e.document_id, e.content_hash) for e in entries}
            if actual != dict(expected):
                missing = sorted(set(expected) - set(actual))[:3]
                extra = sorted(set(actual) - set(expected))[:3]
                raise IndexCorruptError(
                    "snapshot does not match the chunks it was built for "
                    f"(missing e.g. {missing}, unexpected e.g. {extra})"
                )
