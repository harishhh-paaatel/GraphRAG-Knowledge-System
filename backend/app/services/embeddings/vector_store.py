"""Vector index file persistence.

One file, so a single ``os.replace`` swaps vectors and ID table together (two files could
never be replaced atomically as a pair):

    b"GRVEC001"                      8-byte magic
    <uint64 big-endian>              header length
    <header JSON, UTF-8>             metadata + the chunk ID table + checksums
    <payload>                        serialized FAISS index (empty when the index is empty)

Header fields: ``format``, ``format_version`` (2), ``engine``, ``revision`` (the SQLite
``vector_state.revision`` this index was synchronized to), ``built_at``, ``model_id``,
``model_revision``,
``dimension`` (null for an empty index), ``max_tokens``, ``config_fingerprint``, ``count``,
``entries`` (``[chunk_id, document_id, content_hash, token_count, windows]`` per vector, in
index order), ``entries_sha256``,
``payload_sha256``, ``payload_size``.

Loading validates magic, JSON, sizes, both checksums, FAISS type/dimension/count and ID
uniqueness; any failure raises ``IndexCorruptError`` so callers rebuild from SQLite.
"""

import hashlib
import json
import os
import struct
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.services.embeddings.vector_index import (
    FAISS_ENGINE,
    IndexCorruptError,
    IndexEngine,
    IndexEntry,
    VectorIndex,
    VectorIndexError,
)

MAGIC = b"GRVEC001"
FORMAT = "graphrag-vector-index"
FORMAT_VERSION = 2
_MAX_HEADER = 512 * 1024 * 1024


class IndexPersistenceError(VectorIndexError):
    """The index could not be written; the previous file (if any) is untouched."""


@dataclass(frozen=True)
class IndexMeta:
    """Everything needed to decide whether a persisted index is still valid."""

    revision: int
    model_id: str
    model_revision: str | None
    dimension: int | None
    config_fingerprint: str
    built_at: str
    max_tokens: int | None = None  # the model's input limit this index was built under


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entries_digest(entries: list[list[str]]) -> str:
    return _sha256(json.dumps(entries, separators=(",", ":"), ensure_ascii=False).encode())


def save_index(
    index: VectorIndex | None,
    meta: IndexMeta,
    path: Path,
    engine: IndexEngine = FAISS_ENGINE,
) -> int:
    """Atomically write the index (``None`` = empty). Returns the file size in bytes."""
    entries = (
        [
            [e.chunk_id, e.document_id, e.content_hash, e.token_count, e.windows]
            for e in index.entries()
        ]
        if index
        else []
    )
    payload = index.serialize() if index and index.count else b""
    header = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "engine": engine.name,
        **asdict(meta),
        "count": len(entries),
        "entries": entries,
        "entries_sha256": _entries_digest(entries),
        "payload_sha256": _sha256(payload),
        "payload_size": len(payload),
    }
    header_bytes = json.dumps(header, ensure_ascii=False, sort_keys=True).encode()
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            fh.write(MAGIC)
            fh.write(struct.pack(">Q", len(header_bytes)))
            fh.write(header_bytes)
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except (OSError, ValueError, TypeError) as exc:
        tmp.unlink(missing_ok=True)
        raise IndexPersistenceError(f"could not write vector index: {type(exc).__name__}") from exc
    return path.stat().st_size


def load_index(
    path: Path, engine: IndexEngine = FAISS_ENGINE
) -> tuple[VectorIndex | None, IndexMeta]:
    """Load and fully validate an index file. ``None`` index means a valid empty index.

    The file is only accepted by the engine that wrote it (``header["engine"]``).
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise IndexCorruptError("vector index file does not exist") from None
    except OSError as exc:
        raise IndexCorruptError(f"vector index unreadable: {type(exc).__name__}") from exc

    if len(raw) < len(MAGIC) + 8 or raw[: len(MAGIC)] != MAGIC:
        raise IndexCorruptError("not a vector index file (bad magic)")
    (header_len,) = struct.unpack(">Q", raw[len(MAGIC) : len(MAGIC) + 8])
    start = len(MAGIC) + 8
    if header_len > _MAX_HEADER or start + header_len > len(raw):
        raise IndexCorruptError("vector index header is truncated")
    try:
        header = json.loads(raw[start : start + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IndexCorruptError("vector index header is not valid JSON") from exc
    if not isinstance(header, dict) or header.get("format") != FORMAT:
        raise IndexCorruptError("not a vector index file (bad format)")
    if header.get("format_version") != FORMAT_VERSION:
        raise IndexCorruptError(
            f"unsupported index format version {header.get('format_version')!r}"
        )

    if header.get("engine") != engine.name:
        raise IndexCorruptError(
            f"index was written by engine {header.get('engine')!r} but this server uses "
            f"{engine.name!r}"
        )

    payload = raw[start + header_len :]
    try:
        entries_raw = header["entries"]
        meta = IndexMeta(
            revision=int(header["revision"]),
            model_id=str(header["model_id"]),
            model_revision=header["model_revision"],
            dimension=header["dimension"],
            config_fingerprint=str(header["config_fingerprint"]),
            built_at=str(header["built_at"]),
            max_tokens=header.get("max_tokens"),
        )
        if header["count"] != len(entries_raw):
            raise IndexCorruptError("entry count does not match the header")
        if header["payload_size"] != len(payload):
            raise IndexCorruptError("vector payload is truncated or has trailing data")
        if header["entries_sha256"] != _entries_digest(entries_raw):
            raise IndexCorruptError("ID table checksum mismatch")
        if header["payload_sha256"] != _sha256(payload):
            raise IndexCorruptError("vector payload checksum mismatch")
        entries = [
            IndexEntry(str(c), str(d), str(h), int(t), int(w)) for c, d, h, t, w in entries_raw
        ]
    except IndexCorruptError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise IndexCorruptError("vector index header is missing required fields") from exc

    if not entries:
        return None, meta
    if not isinstance(meta.dimension, int) or meta.dimension < 1:
        raise IndexCorruptError("non-empty index without a valid dimension")
    return engine.deserialize(meta.dimension, entries, payload), meta


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
