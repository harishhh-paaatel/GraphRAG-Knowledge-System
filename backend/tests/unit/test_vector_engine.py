"""The VectorIndex / IndexEngine protocols describe the boundary the code really uses."""

from pathlib import Path

import pytest

from app.services.embeddings import vector_store
from app.services.embeddings.vector_index import (
    FAISS_ENGINE,
    FaissFlatIPIndex,
    IndexCorruptError,
    IndexEngine,
    IndexEntry,
    VectorIndex,
)
from app.services.embeddings.vector_store import IndexMeta
from tests.vector_helpers import SpyEngine, unit_vectors

APP = Path(__file__).resolve().parents[2] / "app"
LAYERS_ABOVE_THE_ENGINE = [
    "services/embeddings/service.py",
    "services/embeddings/vector_store.py",
    "services/embeddings/snapshot.py",
    "services/retrieval/vector_search.py",
    "api/search.py",
    "api/vector.py",
]


def test_the_faiss_implementation_satisfies_both_protocols():
    assert isinstance(FaissFlatIPIndex(4), VectorIndex)
    assert isinstance(FAISS_ENGINE, IndexEngine)
    assert FAISS_ENGINE.name == "faiss-flat-ip"
    assert isinstance(FAISS_ENGINE.create(4), VectorIndex)


@pytest.mark.parametrize("relative", LAYERS_ABOVE_THE_ENGINE)
def test_nothing_above_the_engine_names_faiss_directly(relative):
    """If this fails, the abstraction has silently turned back into a concrete dependency."""
    source = (APP / relative).read_text(encoding="utf-8")
    assert "FaissFlatIPIndex" not in source, relative
    assert "import faiss" not in source, relative


def test_there_is_exactly_one_engine_no_pretend_pluggability():
    from app.services.embeddings import vector_index as module

    classes = {name for name, obj in vars(module).items() if isinstance(obj, type)}
    assert {n for n in classes if n.endswith("Engine")} == {"FaissFlatIPEngine", "IndexEngine"}
    assert {n for n in classes if n.endswith("Index")} == {"FaissFlatIPIndex", "VectorIndex"}


def test_the_index_file_records_which_engine_wrote_it(tmp_path):
    idx = FAISS_ENGINE.create(8)
    idx.add([IndexEntry("c1", "d", "h")], unit_vectors(1, 8))
    meta = IndexMeta(1, "m", None, 8, "f" * 64, "t")
    path = tmp_path / "i.bin"
    vector_store.save_index(idx, meta, path)
    assert b'"engine": "faiss-flat-ip"' in path.read_bytes()
    loaded, _ = vector_store.load_index(path)
    assert loaded.count == 1


def test_a_file_is_only_loaded_by_the_engine_that_wrote_it(tmp_path):
    idx = FAISS_ENGINE.create(8)
    idx.add([IndexEntry("c1", "d", "h")], unit_vectors(1, 8))
    path = tmp_path / "i.bin"
    spy = SpyEngine()
    vector_store.save_index(idx, IndexMeta(1, "m", None, 8, "f" * 64, "t"), path, spy)
    with pytest.raises(IndexCorruptError, match="spy-faiss.*faiss-flat-ip"):
        vector_store.load_index(path)  # default FAISS engine refuses a foreign file
    loaded, _ = vector_store.load_index(path, spy)
    assert loaded.count == 1 and spy.deserialized == 1


def test_entry_size_information_survives_a_save_and_load(tmp_path):
    idx = FAISS_ENGINE.create(8)
    entries = [IndexEntry("a", "d", "h1", 300, 2), IndexEntry("b", "d", "h2", 40, 1)]
    idx.add(entries, unit_vectors(2, 8))
    path = tmp_path / "i.bin"
    vector_store.save_index(idx, IndexMeta(1, "m", None, 8, "f" * 64, "t", max_tokens=256), path)
    loaded, meta = vector_store.load_index(path)
    assert loaded.entries() == entries
    assert meta.max_tokens == 256


def test_entries_with_impossible_size_information_are_rejected():
    idx = FAISS_ENGINE.create(8)
    for bad in (IndexEntry("a", "d", "h", -1, 1), IndexEntry("a", "d", "h", 5, 0)):
        with pytest.raises(Exception, match="invalid size"):
            idx.add([bad], unit_vectors(1, 8))
    assert idx.count == 0
