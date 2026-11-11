"""FAISS index: stable ID mapping, exact cosine search, filtering, ties, persistence format."""

import os
import struct

import numpy as np
import pytest

from app.services.embeddings import vector_store
from app.services.embeddings.vector_index import (
    DimensionMismatchError,
    DuplicateChunkError,
    FaissFlatIPIndex,
    IndexCorruptError,
    IndexEntry,
    VectorIndexError,
)
from app.services.embeddings.vector_store import IndexMeta, IndexPersistenceError
from tests.vector_helpers import unit_vectors

D = 16


def entry(i, doc="doc1"):
    return IndexEntry(f"chunk{i:03d}", doc, f"hash{i}")


def build(n=20, docs=("docA", "docB"), seed=1):
    vecs = unit_vectors(n, D, seed)
    idx = FaissFlatIPIndex(D)
    idx.add([entry(i, docs[i % len(docs)]) for i in range(n)], vecs)
    return idx, vecs


def brute_force(vecs, q, k, ids):
    scores = vecs @ q
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
    return [(ids[i], float(scores[i])) for i in order[:k]]


# ---- insertion and search ---------------------------------------------------------------


def test_add_then_search_returns_the_vector_itself_first():
    idx, vecs = build()
    hits = idx.search(vecs[7], 3)
    assert hits[0].entry.chunk_id == "chunk007" and hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert idx.count == 20 and idx.dimension == D


def test_search_matches_numpy_brute_force_exactly():
    idx, vecs = build(n=200, seed=3)
    ids = [f"chunk{i:03d}" for i in range(200)]
    rng = np.random.default_rng(99)
    for _ in range(20):
        q = rng.normal(size=D).astype("float32")
        q /= np.linalg.norm(q)
        for k in (1, 5, 17):
            got = [(h.entry.chunk_id, h.score) for h in idx.search(q, k)]
            want = brute_force(vecs, q, k, ids)
            assert [g[0] for g in got] == [w[0] for w in want]
            assert np.allclose([g[1] for g in got], [w[1] for w in want], atol=1e-5)


def test_cosine_similarity_equals_normalized_dot_product():
    idx = FaissFlatIPIndex(2)
    idx.add([entry(0), entry(1), entry(2)], np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype="float32"))
    q = np.array([0.8, 0.6], dtype="float32")
    scores = {h.entry.chunk_id: h.score for h in idx.search(q, 3)}
    assert scores["chunk000"] == pytest.approx(0.8, abs=1e-6)  # cos(angle) hand-computed
    assert scores["chunk001"] == pytest.approx(0.6, abs=1e-6)
    assert scores["chunk002"] == pytest.approx(0.96, abs=1e-6)


def test_results_are_ordered_by_descending_score_and_top_k_is_honoured():
    idx, vecs = build()
    hits = idx.search(vecs[0], 5)
    assert len(hits) == 5
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert len(idx.search(vecs[0], 1000)) == 20  # k larger than the index just returns everything


def test_top_k_validation():
    idx, vecs = build()
    with pytest.raises(ValueError):
        idx.search(vecs[0], 0)


def test_empty_index_search_returns_nothing():
    assert FaissFlatIPIndex(D).search(unit_vectors(1, D)[0], 5) == []


# ---- document filtering -----------------------------------------------------------------


def test_document_filter_restricts_and_still_returns_top_k_within_it():
    idx, vecs = build(n=40, docs=("docA", "docB", "docC"))
    hits = idx.search(vecs[3], 5, document_ids={"docB"})
    assert len(hits) == 5 and all(h.entry.document_id == "docB" for h in hits)
    ids = [e.chunk_id for e in idx.entries() if e.document_id == "docB"]
    sub = np.array([vecs[int(i[5:])] for i in ids])
    assert [h.entry.chunk_id for h in hits] == [c for c, _ in brute_force(sub, vecs[3], 5, ids)]


def test_exclude_filter_and_combination_and_empty_allow_list():
    idx, vecs = build(n=30, docs=("docA", "docB", "docC"))
    assert {
        h.entry.document_id for h in idx.search(vecs[0], 30, exclude_document_ids={"docA"})
    } == {
        "docB",
        "docC",
    }
    both = idx.search(vecs[0], 30, document_ids={"docA", "docB"}, exclude_document_ids={"docB"})
    assert {h.entry.document_id for h in both} == {"docA"}
    assert idx.search(vecs[0], 5, document_ids=set()) == []  # explicit empty allow-list
    assert idx.search(vecs[0], 5, document_ids={"nope"}) == []


# ---- stable ID mapping --------------------------------------------------------------------


def test_removal_shifts_faiss_positions_but_ids_stay_correct():
    idx, vecs = build(n=10)
    removed = idx.remove(["chunk001", "chunk004", "nonexistent"])
    assert removed == 2 and idx.count == 8
    assert "chunk001" not in idx and "chunk005" in idx
    for i in (0, 2, 3, 5, 6, 7, 8, 9):
        top = idx.search(vecs[i], 1)[0]
        assert top.entry.chunk_id == f"chunk{i:03d}"  # every survivor still finds itself
        assert np.allclose(idx.vector(f"chunk{i:03d}"), vecs[i], atol=1e-6)


def test_remove_everything_leaves_a_usable_empty_index():
    idx, vecs = build(n=5)
    assert idx.remove([e.chunk_id for e in idx.entries()]) == 5
    assert idx.count == 0 and idx.search(vecs[0], 3) == []
    idx.add([entry(99)], vecs[:1])
    assert idx.count == 1


def test_remove_nothing_is_a_noop():
    idx, _ = build(n=3)
    assert idx.remove([]) == 0 and idx.remove(["ghost"]) == 0 and idx.count == 3


# ---- validation and atomic failure --------------------------------------------------------


def test_duplicate_ids_are_detected_against_existing_and_within_a_batch():
    idx, vecs = build(n=3)
    with pytest.raises(DuplicateChunkError):
        idx.add([entry(1)], vecs[:1])
    with pytest.raises(DuplicateChunkError):
        idx.add([entry(50), entry(50)], unit_vectors(2, D, 9))
    assert idx.count == 3 and idx.search(vecs[0], 10).__len__() == 3  # nothing partially added


def test_dimension_mismatch_and_shape_errors():
    idx, _ = build(n=2)
    with pytest.raises(DimensionMismatchError):
        idx.add([entry(10)], unit_vectors(1, D + 1))
    with pytest.raises(DimensionMismatchError):
        idx.add([entry(10), entry(11)], unit_vectors(1, D))  # count mismatch
    with pytest.raises(DimensionMismatchError):
        idx.add([entry(10)], np.ones(D, dtype="float32"))  # 1-D
    with pytest.raises(DimensionMismatchError):
        idx.search(unit_vectors(1, D + 1)[0], 3)
    assert idx.count == 2


def test_non_normalized_and_non_finite_vectors_are_rejected():
    idx = FaissFlatIPIndex(D)
    with pytest.raises(VectorIndexError, match="normalized"):
        idx.add([entry(0)], unit_vectors(1, D) * 2)
    bad = unit_vectors(1, D)
    bad[0, 0] = np.nan
    with pytest.raises(VectorIndexError, match="NaN"):
        idx.add([entry(0)], bad)
    with pytest.raises(VectorIndexError, match="normalized"):
        build()[0].search(unit_vectors(1, D)[0] * 3, 3)
    assert idx.count == 0


@pytest.mark.parametrize(
    "bad", [IndexEntry("", "d", "h"), IndexEntry("c", "", "h"), IndexEntry("c", "d", "")]
)
def test_entries_with_missing_metadata_are_rejected(bad):
    idx = FaissFlatIPIndex(D)
    with pytest.raises(VectorIndexError, match="missing metadata"):
        idx.add([bad], unit_vectors(1, D))
    assert idx.count == 0


# ---- deterministic ordering of ties -------------------------------------------------------


def test_equal_scores_are_ordered_by_chunk_id_regardless_of_insertion_order():
    v = unit_vectors(1, D, 5)
    ids = ["zeta", "alpha", "mu", "beta"]
    for order in (ids, list(reversed(ids)), ["mu", "zeta", "beta", "alpha"]):
        idx = FaissFlatIPIndex(D)
        idx.add([IndexEntry(c, "d", "h") for c in order], np.repeat(v, 4, axis=0))
        assert [h.entry.chunk_id for h in idx.search(v[0], 4)] == ["alpha", "beta", "mu", "zeta"]


def test_a_tie_straddling_the_cutoff_is_resolved_deterministically():
    v = unit_vectors(1, D, 6)
    other = unit_vectors(3, D, 7)
    ids = [f"tie{i}" for i in range(6)]
    results = set()
    for order in (ids, list(reversed(ids))):
        idx = FaissFlatIPIndex(D)
        idx.add([IndexEntry(c, "d", "h") for c in order], np.repeat(v, 6, axis=0))
        idx.add([IndexEntry(f"other{i}", "d", "h") for i in range(3)], other)
        results.add(tuple(h.entry.chunk_id for h in idx.search(v[0], 3)))
    assert results == {("tie0", "tie1", "tie2")}  # the 3 smallest IDs of the 6-way tie


# ---- serialization -----------------------------------------------------------------------


def test_serialize_roundtrip_preserves_results():
    idx, vecs = build()
    back = FaissFlatIPIndex.deserialize(D, idx.entries(), idx.serialize())
    assert back.entries() == idx.entries()
    assert [h.entry.chunk_id for h in back.search(vecs[4], 5)] == [
        h.entry.chunk_id for h in idx.search(vecs[4], 5)
    ]


def test_clone_is_independent():
    idx, vecs = build(n=5)
    clone = idx.clone()
    clone.remove(["chunk000"])
    assert idx.count == 5 and clone.count == 4 and "chunk000" in idx


def test_deserialize_validates_consistency():
    idx, _ = build(n=4)
    payload = idx.serialize()
    with pytest.raises(IndexCorruptError, match="deserialized"):
        FaissFlatIPIndex.deserialize(D, idx.entries(), b"not faiss data" * 5)
    with pytest.raises(IndexCorruptError, match="4 vectors"):
        FaissFlatIPIndex.deserialize(D, idx.entries()[:3], payload)
    with pytest.raises(IndexCorruptError, match="dimension"):
        FaissFlatIPIndex.deserialize(D + 1, idx.entries(), payload)
    dup = idx.entries()[:3] + [idx.entries()[0]]
    with pytest.raises(IndexCorruptError, match="duplicate"):
        FaissFlatIPIndex.deserialize(D, dup, payload)
    with pytest.raises(IndexCorruptError, match="missing metadata"):
        FaissFlatIPIndex.deserialize(D, idx.entries()[:3] + [IndexEntry("x", "", "h")], payload)


# ---- file persistence ---------------------------------------------------------------------


META = IndexMeta(
    revision=7,
    model_id="m",
    model_revision="abc",
    dimension=D,
    config_fingerprint="f" * 64,
    built_at="2026-01-01T00:00:00+00:00",
)


@pytest.fixture
def path(tmp_path):
    return tmp_path / "embeddings" / "vector_index.bin"


def test_save_and_load_roundtrip(path):
    idx, vecs = build()
    size = vector_store.save_index(idx, META, path)
    assert size == path.stat().st_size and path.read_bytes()[:8] == vector_store.MAGIC
    loaded, meta = vector_store.load_index(path)
    assert meta == META and loaded.entries() == idx.entries()
    assert [h.entry.chunk_id for h in loaded.search(vecs[9], 4)] == [
        h.entry.chunk_id for h in idx.search(vecs[9], 4)
    ]


def test_empty_index_roundtrips_as_none(path):
    meta = IndexMeta(3, "m", None, None, "f" * 64, "t")
    vector_store.save_index(None, meta, path)
    loaded, got = vector_store.load_index(path)
    assert loaded is None and got == meta


def test_overwriting_leaves_no_temp_files(path):
    idx, _ = build(n=3)
    vector_store.save_index(idx, META, path)
    vector_store.save_index(idx, META, path)
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_failed_write_keeps_the_previous_file_intact(path, monkeypatch):
    idx, _ = build(n=3)
    vector_store.save_index(idx, META, path)
    before = path.read_bytes()

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(IndexPersistenceError):
        vector_store.save_index(build(n=9)[0], META, path)
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == [path.name]


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "empty",
        "bad_magic",
        "truncated_header",
        "truncated_payload",
        "trailing_garbage",
        "payload_bitflip",
        "idtable_tamper",
        "wrong_format",
        "wrong_version",
        "header_not_json",
    ],
)
def test_corrupted_files_are_detected(path, damage):
    idx, _ = build()
    vector_store.save_index(idx, META, path)
    raw = bytearray(path.read_bytes())
    (hlen,) = struct.unpack(">Q", raw[8:16])
    header_end = 16 + hlen
    if damage == "missing":
        path.unlink()
    elif damage == "empty":
        path.write_bytes(b"")
    elif damage == "bad_magic":
        path.write_bytes(b"NOTAVEC1" + bytes(raw[8:]))
    elif damage == "truncated_header":
        path.write_bytes(bytes(raw[: 16 + hlen // 2]))
    elif damage == "truncated_payload":
        path.write_bytes(bytes(raw[:-10]))
    elif damage == "trailing_garbage":
        path.write_bytes(bytes(raw) + b"junk")
    elif damage == "payload_bitflip":
        raw[header_end + 40] ^= 0xFF
        path.write_bytes(bytes(raw))
    elif damage == "idtable_tamper":
        path.write_bytes(bytes(raw).replace(b"chunk003", b"chunkXXX", 1))
    elif damage == "wrong_format":
        path.write_bytes(bytes(raw).replace(b"graphrag-vector-index", b"graphrag-other-thing--", 1))
    elif damage == "wrong_version":
        assert b'"format_version": 2' in raw  # current format is version 2
        path.write_bytes(bytes(raw).replace(b'"format_version": 2', b'"format_version": 9', 1))
    else:
        path.write_bytes(bytes(raw[:16]) + b"{not json" + bytes(raw[16 + 9 :]))
    with pytest.raises(IndexCorruptError):
        vector_store.load_index(path)
