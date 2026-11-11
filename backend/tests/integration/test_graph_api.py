"""Graph REST endpoints against data produced by the real ingestion pipeline."""

import pytest

from app.services.graph import analytics
from app.services.graph.service import GraphUnavailable

G = "/api/v1/graph"
DOC_A = b"Sebastian Ramirez created FastAPI. FastAPI uses Pydantic. FastAPI depends on Starlette."
DOC_B = b"Alice works at Acme Corp. Acme Corp is located in Berlin. Bob Smith joined Acme Corp."


@pytest.fixture
def docs(client, app):
    ids = {}
    for name, body in (("a.txt", DOC_A), ("b.txt", DOC_B)):
        r = client.post("/api/v1/documents/upload", files={"file": (name, body, "text/plain")})
        assert r.status_code == 202
        ids[name] = r.json()["document"]["id"]
    assert app.state.worker.drain(30)
    return ids


@pytest.fixture
def eid(client, docs):
    def _eid(name):
        items = client.get(f"{G}/nodes", params={"search": name}).json()["items"]
        return next(i["entity_id"] for i in items if i["canonical_name"] == name)

    return _eid


# ---- overview ---------------------------------------------------------------------------


def test_overview_reports_real_statistics(client, docs):
    body = client.get(f"{G}/overview").json()
    s = body["stats"]
    assert (s["node_count"], s["edge_count"]) == (8, 6)
    assert s["entity_type_distribution"] == {
        "LOCATION": 1,
        "ORGANIZATION": 1,
        "PERSON": 3,
        "TECHNOLOGY": 3,
    }
    assert s["relationship_type_distribution"] == {
        "DEPENDS_ON": 1,
        "DEVELOPED": 1,
        "LOCATED_IN": 1,
        "USES": 1,
        "WORKS_AT": 2,
    }
    assert s["weakly_connected_components"]["count"] == 2
    assert s["weakly_connected_components"]["sizes"] == [4, 4]
    assert s["strongly_connected_components"]["count"] == 8  # acyclic: every node alone
    top = s["top_entities_by_degree_centrality"][0]
    assert top["canonical_name"] == "Acme Corp"  # ties broken by name; degree 3 of n-1 = 7
    assert (top["in_degree"], top["out_degree"], top["degree_centrality"]) == (
        2,
        1,
        round(3 / 7, 6),
    )
    assert body["graph"]["revision"] > 0 and body["state"]["state"] == "ok"


def test_overview_statistics_are_cached_per_revision(client, app, docs, monkeypatch):
    calls = []
    real = analytics.compute_stats
    monkeypatch.setattr(
        analytics, "compute_stats", lambda g, *a, **k: calls.append(1) or real(g, *a, **k)
    )
    first = client.get(f"{G}/overview").json()
    client.get(f"{G}/overview")
    client.get(f"{G}/overview")
    assert len(calls) == 1  # computed once, served from cache

    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("c.txt", b"Carol Diaz joined Acme Corp.", "text/plain")},
    )
    app.state.worker.drain(30)
    second = client.get(f"{G}/overview").json()
    assert len(calls) == 2  # new revision -> recomputed
    assert second["stats"]["node_count"] == first["stats"]["node_count"] + 1
    assert r.status_code == 202


def test_graph_unavailable_returns_503(client, app, monkeypatch):
    def broken():
        raise GraphUnavailable("database unavailable")

    monkeypatch.setattr(app.state.graph, "snapshot", broken)
    r = client.get(f"{G}/overview")
    assert r.status_code == 503 and r.json()["error"]["code"] == "graph_unavailable"


# ---- nodes ------------------------------------------------------------------------------


def test_nodes_pagination_filters_and_search(client, docs):
    page1 = client.get(f"{G}/nodes", params={"limit": 3}).json()
    page2 = client.get(f"{G}/nodes", params={"limit": 3, "offset": 3}).json()
    assert page1["total"] == 8 and len(page1["items"]) == 3
    assert not {n["entity_id"] for n in page1["items"]} & {n["entity_id"] for n in page2["items"]}

    persons = client.get(f"{G}/nodes", params={"entity_type": "PERSON"}).json()
    assert persons["total"] == 3 and {n["entity_type"] for n in persons["items"]} == {"PERSON"}

    hit = client.get(f"{G}/nodes", params={"search": "acme"}).json()
    assert [n["canonical_name"] for n in hit["items"]] == ["Acme Corp"]

    by_degree = client.get(f"{G}/nodes", params={"sort": "degree", "limit": 2}).json()["items"]
    assert by_degree[0]["in_degree"] + by_degree[0]["out_degree"] == 3

    in_doc = client.get(f"{G}/nodes", params={"document_id": docs["a.txt"]}).json()
    assert in_doc["total"] == 4 and all(
        docs["a.txt"] in n["source_document_ids"] for n in in_doc["items"]
    )


def test_node_payload_has_the_required_fields(client, docs):
    n = client.get(f"{G}/nodes", params={"search": "FastAPI"}).json()["items"][0]
    assert {
        "entity_id",
        "canonical_name",
        "entity_type",
        "description",
        "aliases",
        "mention_count",
        "source_document_ids",
        "in_degree",
        "out_degree",
    } <= set(n)
    assert (n["in_degree"], n["out_degree"], n["mention_count"]) == (1, 2, 1)  # 1 chunk


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 201},
        {"offset": -1},
        {"entity_type": "person"},
        {"sort": "bogus"},
        {"search": ""},
    ],
)
def test_nodes_validation(client, params):
    r = client.get(f"{G}/nodes", params=params)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


# ---- edges ------------------------------------------------------------------------------


def test_edges_carry_evidence_and_source_references(client, docs, eid):
    body = client.get(f"{G}/edges", params={"relationship_type": "DEVELOPED"}).json()
    assert body["total"] == 1
    e = body["items"][0]
    assert (e["source_name"], e["target_name"]) == (
        "Sebastian Ramirez",
        "FastAPI",
    )  # direction kept
    assert e["source_entity_id"] == eid("Sebastian Ramirez")
    [ev] = e["supporting_evidence"]
    assert ev["supporting_text"] == "Sebastian Ramirez created FastAPI"
    assert (
        ev["document_id"] == docs["a.txt"] and ev["chunk_id"] and ev["provenance"] == "rule_based"
    )
    assert ev["confidence"] is None and e["confidence"] is None
    assert e["source_document_ids"] == [docs["a.txt"]] and e["source_chunk_ids"] == [ev["chunk_id"]]
    assert body["documents"] == {docs["a.txt"]: "a.txt"}  # ids resolved to filenames


def test_edge_filters_and_pagination(client, docs, eid):
    assert client.get(f"{G}/edges").json()["total"] == 6
    assert client.get(f"{G}/edges", params={"limit": 4}).json()["items"].__len__() == 4
    by_src = client.get(f"{G}/edges", params={"source_id": eid("FastAPI")}).json()
    assert {e["relationship_type"] for e in by_src["items"]} == {"USES", "DEPENDS_ON"}
    by_tgt = client.get(f"{G}/edges", params={"target_id": eid("Acme Corp")}).json()
    assert by_tgt["total"] == 2
    by_doc = client.get(f"{G}/edges", params={"document_id": docs["b.txt"]}).json()
    assert by_doc["total"] == 3
    assert client.get(f"{G}/edges", params={"relationship_type": "bad type"}).status_code == 422


# ---- entity detail ----------------------------------------------------------------------


def test_entity_detail_has_relationships_mentions_and_documents(client, docs, eid):
    d = client.get(f"{G}/entities/{eid('Acme Corp')}").json()
    assert d["canonical_name"] == "Acme Corp" and d["entity_type"] == "ORGANIZATION"
    assert sorted(e["source_name"] for e in d["incoming"]) == ["Alice", "Bob Smith"]
    assert [(e["relationship_type"], e["target_name"]) for e in d["outgoing"]] == [
        ("LOCATED_IN", "Berlin")
    ]
    assert all(e["supporting_evidence"] for e in d["incoming"] + d["outgoing"])
    assert len(d["mentions"]) == 1  # one chunk, one surface form
    assert {m["document_id"] for m in d["mentions"]} == {docs["b.txt"]}
    assert all(m["chunk_id"] and m["provenance"] == "rule_based" for m in d["mentions"])
    assert d["documents"] == {docs["b.txt"]: "b.txt"}
    assert d["graph"]["revision"] > 0


def test_entity_detail_mention_limit_and_404(client, app, docs, eid):
    client.post(
        "/api/v1/documents/upload",
        files={"file": ("c.txt", b"Carol Diaz joined Acme Corp.", "text/plain")},
    )
    app.state.worker.drain(30)
    url = f"{G}/entities/{eid('Acme Corp')}"
    assert len(client.get(url).json()["mentions"]) == 2  # one per document/chunk
    assert len(client.get(url, params={"mention_limit": 1}).json()["mentions"]) == 1
    assert client.get(url, params={"mention_limit": 0}).json()["mentions"] == []
    assert client.get(url, params={"mention_limit": 501}).status_code == 422
    r = client.get(f"{G}/entities/does-not-exist")
    assert r.status_code == 404 and r.json()["error"]["code"] == "entity_not_found"


# ---- neighbors / multi-hop / subgraph -------------------------------------------------


def nb(client, entity_id, **params):
    r = client.get(f"{G}/entities/{entity_id}/neighbors", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_default_neighbors_are_outgoing_only(client, docs, eid):
    body = nb(client, eid("Acme Corp"))
    assert body["direction"] == "out" and body["hops"] == 1
    assert [n["canonical_name"] for n in body["nodes"] if n["depth"] == 1] == ["Berlin"]
    assert [(e["relationship_type"], e["traversal"]) for e in body["edges"]] == [
        ("LOCATED_IN", "outgoing")
    ]
    assert body["truncated"] == []


def test_incoming_and_both_directions(client, docs, eid):
    inc = nb(client, eid("Acme Corp"), direction="in")
    assert sorted(n["canonical_name"] for n in inc["nodes"] if n["depth"] == 1) == [
        "Alice",
        "Bob Smith",
    ]
    assert {e["traversal"] for e in inc["edges"]} == {"incoming"}
    for e in inc["edges"]:  # stored direction is preserved even when walked backwards
        assert e["target_name"] == "Acme Corp" and e["relationship_type"] == "WORKS_AT"
    both = nb(client, eid("Acme Corp"), direction="both")
    assert len(both["nodes"]) == 4 and {e["traversal"] for e in both["edges"]} == {
        "incoming",
        "outgoing",
    }


def test_multi_hop_traversal_reports_depths_and_evidence(client, docs, eid):
    body = nb(client, eid("Sebastian Ramirez"), hops=2)
    depth = {n["canonical_name"]: n["depth"] for n in body["nodes"]}
    assert depth == {"Sebastian Ramirez": 0, "FastAPI": 1, "Pydantic": 2, "Starlette": 2}
    assert len(body["edges"]) == 3 and all(e["supporting_evidence"] for e in body["edges"])
    assert body["documents"] == {docs["a.txt"]: "a.txt"}
    one = nb(client, eid("Sebastian Ramirez"), hops=1)
    assert {n["canonical_name"] for n in one["nodes"]} == {"Sebastian Ramirez", "FastAPI"}


def test_relationship_type_filter(client, docs, eid):
    body = nb(client, eid("FastAPI"), relationship_type="USES")
    assert [n["canonical_name"] for n in body["nodes"]] == ["FastAPI", "Pydantic"]
    assert body["relationship_types"] == ["USES"]
    two = nb(client, eid("FastAPI"), relationship_type=["USES", "DEPENDS_ON"])
    assert len(two["nodes"]) == 3


def test_subgraph_induced_edges(client, docs, eid):
    plain = nb(client, eid("Alice"), hops=2, direction="both")
    induced = nb(client, eid("Alice"), hops=2, direction="both", induced=True)
    assert {n["entity_id"] for n in induced["nodes"]} == {n["entity_id"] for n in plain["nodes"]}
    assert len(induced["edges"]) >= len(plain["edges"])
    ids = {n["entity_id"] for n in induced["nodes"]}
    assert all(
        e["source_entity_id"] in ids and e["target_entity_id"] in ids for e in induced["edges"]
    )


def test_traversal_limits_truncate_and_say_so(client, docs, eid):
    body = nb(client, eid("Acme Corp"), direction="both", max_nodes=2)
    assert len(body["nodes"]) == 2 and "max_nodes" in body["truncated"]
    body = nb(client, eid("Acme Corp"), direction="both", max_edges=1)
    assert len(body["edges"]) == 1 and "max_edges" in body["truncated"]
    assert body["limits"]["max_edges"] == 1


def test_traversal_hard_caps_are_enforced(client, app, docs, eid):
    app.state.settings.graph_max_hops = 2
    app.state.settings.graph_max_nodes = 5
    app.state.settings.graph_max_edges = 7
    base = f"{G}/entities/{eid('Acme Corp')}/neighbors"
    for params in ({"hops": 3}, {"max_nodes": 6}, {"max_edges": 8}):
        r = client.get(base, params=params)
        assert r.status_code == 422 and r.json()["error"]["code"] == "limit_exceeded"
    default = client.get(base, params={"direction": "both"}).json()
    assert (
        default["limits"]["max_nodes"] <= 5 and default["limits"]["max_edges"] <= 7
    )  # defaults clamp too


@pytest.mark.parametrize(
    "params",
    [{"direction": "sideways"}, {"hops": 0}, {"max_nodes": 0}, {"relationship_type": "lower"}],
)
def test_neighbors_validation(client, docs, eid, params):
    r = client.get(f"{G}/entities/{eid('Acme Corp')}/neighbors", params=params)
    assert r.status_code == 422


def test_neighbors_unknown_entity(client, docs):
    r = client.get(f"{G}/entities/nope/neighbors")
    assert r.status_code == 404 and r.json()["error"]["code"] == "entity_not_found"


# ---- shortest path ----------------------------------------------------------------------


def path(client, src, tgt, **params):
    return client.get(f"{G}/path", params={"source": src, "target": tgt, **params})


def test_directed_shortest_path_with_evidence(client, docs, eid):
    r = path(client, eid("Sebastian Ramirez"), eid("Pydantic"))
    assert r.status_code == 200
    body = r.json()
    assert body["length"] == 2 and body["directed"] is True
    assert [n["canonical_name"] for n in body["nodes"]] == [
        "Sebastian Ramirez",
        "FastAPI",
        "Pydantic",
    ]
    assert [(s["step"], s["traversed"], s["edge"]["relationship_type"]) for s in body["steps"]] == [
        (0, "forward", "DEVELOPED"),
        (1, "forward", "USES"),
    ]
    assert all(s["edge"]["supporting_evidence"] for s in body["steps"])
    assert body["documents"] == {docs["a.txt"]: "a.txt"}


def test_path_against_edge_direction_is_not_found_unless_undirected(client, docs, eid):
    pydantic, ramirez = eid("Pydantic"), eid("Sebastian Ramirez")
    r = path(client, pydantic, ramirez)
    assert r.status_code == 404 and r.json()["error"]["code"] == "no_path"
    r = path(client, pydantic, ramirez, directed=False)
    assert r.status_code == 200
    steps = r.json()["steps"]
    assert [s["traversed"] for s in steps] == ["reverse", "reverse"]
    # The edge keeps its true direction even though the walk went backwards.
    assert (
        steps[0]["edge"]["source_name"] == "FastAPI"
        and steps[0]["edge"]["target_name"] == "Pydantic"
    )


def test_path_between_disconnected_components(client, docs, eid):
    r = path(client, eid("Alice"), eid("FastAPI"), directed=False)
    assert r.status_code == 404 and r.json()["error"]["code"] == "no_path"


def test_path_hop_limits_and_unknown_entities(client, app, docs, eid):
    alice, berlin = eid("Alice"), eid("Berlin")
    assert path(client, alice, berlin, max_hops=2).status_code == 200
    assert path(client, alice, berlin, max_hops=1).status_code == 404
    app.state.settings.graph_max_hops = 2
    assert path(client, alice, berlin, max_hops=3).status_code == 422
    r = path(client, "ghost", berlin)
    assert r.status_code == 404 and r.json()["error"]["code"] == "entity_not_found"
    assert client.get(f"{G}/path", params={"source": alice}).status_code == 422


def test_graph_endpoints_follow_document_deletion(client, docs, eid):
    pydantic = eid("Pydantic")
    assert client.delete(f"/api/v1/documents/{docs['a.txt']}").status_code == 204
    assert client.get(f"{G}/entities/{pydantic}").status_code == 404
    assert client.get(f"{G}/overview").json()["stats"]["node_count"] == 4
    assert client.get(f"{G}/edges", params={"relationship_type": "USES"}).json()["total"] == 0
