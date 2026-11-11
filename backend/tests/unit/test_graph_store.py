"""Graph file format, atomic replacement and corruption detection."""

import json
import os

import networkx as nx
import pytest

from app.services.graph import store
from app.services.graph.consistency import logical_form
from app.services.graph.store import GraphLoadError, GraphPersistenceError
from tests.unit.test_graph_queries import EDGES, NODES, make_graph


@pytest.fixture
def graph():
    return make_graph(NODES, EDGES, aliases={"acme": ["Acme Corp"]})


@pytest.fixture
def path(tmp_path):
    return tmp_path / "graphs" / "knowledge_graph.json"


def test_roundtrip_preserves_every_attribute(graph, path):
    meta = store.save_graph(graph, 7, path)
    loaded, revision, loaded_meta = store.load_graph(path)
    assert revision == 7 and meta["revision"] == 7
    assert (meta["node_count"], meta["edge_count"]) == (6, 5)
    assert logical_form(loaded) == logical_form(graph)
    assert loaded.is_directed() and loaded.is_multigraph()
    assert loaded_meta["format"] == store.FORMAT
    assert loaded.edges["alice", "acme", "WORKS_AT"]["supporting_evidence"][0]["supporting_text"]
    assert loaded.edges["alice", "acme", "WORKS_AT"]["confidence"] is None  # None survives


def test_direction_survives_the_roundtrip(graph, path):
    store.save_graph(graph, 1, path)
    loaded, _, _ = store.load_graph(path)
    assert loaded.has_edge("alice", "acme") and not loaded.has_edge("acme", "alice")


def test_loaded_graph_is_frozen(graph, path):
    store.save_graph(graph, 1, path)
    loaded, _, _ = store.load_graph(path)
    with pytest.raises(nx.NetworkXError):
        loaded.add_node("x")


def test_file_is_documented_json_envelope(graph, path):
    store.save_graph(graph, 3, path)
    env = json.loads(path.read_text(encoding="utf-8"))
    assert {
        "format",
        "format_version",
        "revision",
        "built_at",
        "node_count",
        "edge_count",
        "sha256",
        "graph",
    } <= set(env)
    assert env["format"] == "graphrag-knowledge-graph" and env["format_version"] == 1
    assert env["graph"]["directed"] is True and env["graph"]["multigraph"] is True


def test_save_creates_directories_and_leaves_no_temp_files(graph, path):
    store.save_graph(graph, 1, path)
    store.save_graph(graph, 2, path)
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_failed_write_keeps_the_previous_file_intact(graph, path, monkeypatch):
    store.save_graph(graph, 1, path)
    before = path.read_bytes()

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(GraphPersistenceError):
        store.save_graph(make_graph({"x": "T"}, []), 2, path)
    monkeypatch.undo()

    assert path.read_bytes() == before  # old file untouched
    assert store.load_graph(path)[1] == 1
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # temp file cleaned up


def test_failed_write_when_directory_unusable(graph, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    with pytest.raises(GraphPersistenceError):
        store.save_graph(graph, 1, blocker / "graph.json")


@pytest.mark.parametrize("damage", ["missing", "garbage", "truncated", "empty", "binary"])
def test_unreadable_files_raise_load_error(graph, path, damage):
    store.save_graph(graph, 1, path)
    raw = path.read_bytes()
    if damage == "missing":
        path.unlink()
    elif damage == "garbage":
        path.write_text("not json at all")
    elif damage == "truncated":
        path.write_bytes(raw[: len(raw) // 2])
    elif damage == "empty":
        path.write_bytes(b"")
    else:
        path.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(GraphLoadError):
        store.load_graph(path)


def _rewrite(path, mutate, keep_checksum=True):
    env = json.loads(path.read_text(encoding="utf-8"))
    mutate(env)
    if not keep_checksum:
        env["sha256"] = store.checksum(env["graph"])
    path.write_text(json.dumps(env), encoding="utf-8")


def test_tampered_content_is_detected_by_checksum(graph, path):
    store.save_graph(graph, 1, path)
    _rewrite(path, lambda e: e["graph"]["nodes"][0].update(canonical_name="HACKED"))
    with pytest.raises(GraphLoadError, match="checksum"):
        store.load_graph(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda e: e.update(format="other"), "not a knowledge graph"),
        (lambda e: e.update(format_version=99), "unsupported"),
        (lambda e: e.pop("graph"), "missing required"),
        (lambda e: e.update(revision="x"), "missing required"),
        (lambda e: e.update(node_count=999), "counts"),
    ],
)
def test_structurally_wrong_files_are_rejected(graph, path, mutate, message):
    store.save_graph(graph, 1, path)
    _rewrite(path, mutate, keep_checksum=True)
    with pytest.raises(GraphLoadError, match=message):
        store.load_graph(path)


def test_non_directed_or_non_multigraph_payloads_are_rejected(graph, path):
    store.save_graph(graph, 1, path)

    def undirect(env):
        env["graph"]["directed"] = False

    _rewrite(path, undirect, keep_checksum=False)
    with pytest.raises(GraphLoadError, match="directed multigraph"):
        store.load_graph(path)
