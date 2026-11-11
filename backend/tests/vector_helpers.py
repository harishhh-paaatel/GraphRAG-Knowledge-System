"""Helpers for vector-index tests: documents with chunks, and misbehaving embedders."""

import itertools

import numpy as np

from app.models import Document, DocumentChunk
from app.services.embeddings.embedder import EmbeddingError, HashingEmbedder
from app.services.embeddings.revision import bump_vector_revision

_counter = itertools.count(1)

CORPUS = [
    "FastAPI is a web framework for building APIs with Python.",
    "Pydantic validates data using Python type hints.",
    "Berlin is the capital city of Germany.",
    "SQLAlchemy maps Python objects to relational database tables.",
]


def add_doc(session, texts, *, status="completed", filename=None, pages=None):
    """Insert a document with one chunk per text (and bump the vector revision)."""
    n = next(_counter)
    doc = Document(
        id=f"d{n:04d}",
        filename=filename or f"doc{n}.txt",
        stored_path="x",
        file_type="txt",
        size_bytes=1,
        checksum=f"{n + 90000:064d}",
        processing_status=status,
    )
    session.add(doc)
    session.flush()
    ids = []
    for i, text in enumerate(texts):
        cid = f"{doc.id}-c{i}"
        session.add(
            DocumentChunk(
                id=cid,
                document_id=doc.id,
                chunk_index=i,
                content=text,
                page_number=(pages[i] if pages else None),
                chunk_metadata={},
            )
        )
        ids.append(cid)
    bump_vector_revision(session)
    session.commit()
    return doc, ids


def set_chunk_text(session, chunk_id, text):
    session.get(DocumentChunk, chunk_id).content = text
    bump_vector_revision(session)
    session.commit()


def remove_doc(session, doc):
    session.expire_all()
    session.delete(session.get(Document, doc.id))
    bump_vector_revision(session)
    session.commit()


class ControlledEmbedder(HashingEmbedder):
    """Hashing embedder that can be told to fail on certain texts."""

    def __init__(self, dimension=256, fail_on=()):
        super().__init__(dimension)
        self.fail_on = set(fail_on)

    def embed(self, texts):
        for t in texts:
            if any(f in t for f in self.fail_on):
                raise EmbeddingError("simulated embedding failure")
        return super().embed(texts)


def unit_vectors(n, d, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, d)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---- helpers added for the Phase 5A reliability work ------------------------------------------

import sys  # noqa: E402
import threading  # noqa: E402
import types  # noqa: E402

from app.services.embeddings.vector_index import FAISS_ENGINE  # noqa: E402


class BlockingEmbedder(HashingEmbedder):
    """Holds a sync open: embed() blocks on texts containing ``marker`` until released."""

    def __init__(self, dimension=256, marker="SLOWDOC", max_tokens=None):
        super().__init__(dimension, max_tokens)
        self.marker = marker
        self.entered = threading.Event()
        self.release = threading.Event()

    def embed(self, texts):
        if any(self.marker in t for t in texts):
            self.entered.set()
            assert self.release.wait(5), "test forgot to release the blocked embedder"
        return super().embed(texts)


class SpyEngine:
    """Delegates to the FAISS engine but under another name, counting calls (a test double)."""

    name = "spy-faiss"

    def __init__(self):
        self.created = 0
        self.deserialized = 0

    def create(self, dimension):
        self.created += 1
        return FAISS_ENGINE.create(dimension)

    def deserialize(self, dimension, entries, payload):
        self.deserialized += 1
        return FAISS_ENGINE.deserialize(dimension, entries, payload)


class _WordTokenizer:
    """One token per whitespace-separated word plus two special tokens, like [CLS]/[SEP]."""

    def __call__(self, texts, add_special_tokens=True, truncation=False, verbose=True):
        extra = 2 if add_special_tokens else 0
        return {"input_ids": [[0] * (len(t.split()) + extra) for t in texts]}


def install_fake_sentence_transformers(monkeypatch, *, max_seq_length=128, fail=None):
    """Replace ``sentence_transformers`` with a fake. No network, no download, no torch.

    ``fail`` is an exception instance raised by the constructor (to simulate a missing cache).
    Returns the fake class; every construction is recorded in ``.created`` as (name, kwargs).
    """

    class FakeModel:
        created: list = []
        encoded_inputs: list = []

        def __init__(self, name, **kwargs):
            FakeModel.created.append((name, kwargs))
            if fail is not None:
                raise fail
            self.max_seq_length = max_seq_length
            self.tokenizer = _WordTokenizer()

        def get_sentence_embedding_dimension(self):
            return 8

        def encode(self, texts, **kwargs):
            FakeModel.encoded_inputs.extend(list(texts))
            rng = np.random.default_rng(abs(hash(tuple(texts))) % 2**32)
            return rng.normal(size=(len(texts), 8)).astype("float32")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return FakeModel
