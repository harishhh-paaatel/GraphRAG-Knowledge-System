r"""Hybrid retrieval demo: vector, graph and fused views of the same queries, then the failure modes.

Usage (from the repo root):
    backend\.venv\Scripts\python scripts\hybrid_demo.py
    backend\.venv\Scripts\python scripts\hybrid_demo.py --provider hashing    # no model needed

Runs in a throwaway data directory with the real ingestion pipeline (rule-based extraction), the
real graph, the real FAISS index and the real API (in-process TestClient). Everything printed is
read from the responses of that run; nothing is hard-coded.

This shows MECHANICS only. The corpus has 8 chunks, so the output says nothing about retrieval
quality, and no claim is made that hybrid retrieval beats vector-only retrieval: that needs the
labelled evaluation of Phase 8.
"""

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--provider", default="sentence_transformers", choices=["sentence_transformers", "hashing"]
)
args = parser.parse_args()
os.environ["GRAPHRAG_EMBEDDING_PROVIDER"] = args.provider
if args.provider == "sentence_transformers":
    os.environ["GRAPHRAG_EMBEDDING_CACHE_DIR"] = str(ROOT / "data" / "models")
    os.environ["GRAPHRAG_EMBEDDING_OFFLINE"] = "true"  # the model is cached; never use the network

import logging  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Base  # noqa: E402
from app.services.embeddings.service import VectorIndexBuilding  # noqa: E402
from app.services.graph.service import GraphUnavailable  # noqa: E402

logging.disable(logging.CRITICAL)
SAMPLES = [
    "backend_stack.md",
    "team_notes.txt",
    "geography.txt",
    "vector_search_notes.md",
    "kitchen_notes.txt",
]
API = "/api/v1"


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def short(text: str, n: int = 52) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="graphrag-hd-", dir="E:/graphrag-cache/tmp"))
    app = create_app(
        Settings(data_dir=tmp, environment="test", hybrid_path_timeout_seconds=2.0, _env_file=None)
    )
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as c:
        post = lambda path, **body: c.post(f"{API}{path}", json=body)  # noqa: E731

        banner("1. INGEST the sample documents (parse -> chunk -> entities -> graph -> vectors)")
        ids = {}
        t = time.perf_counter()
        for name in SAMPLES:
            data = (ROOT / "scripts" / "samples" / name).read_bytes()
            r = c.post(f"{API}/documents/upload", files={"file": (name, data, "text/plain")})
            ids[name] = r.json()["document"]["id"]
        for _ in range(600):
            docs = c.get(f"{API}/documents").json()["items"]
            if all(d["processing_status"] in ("completed", "failed") for d in docs):
                break
            time.sleep(0.2)
        ing = time.perf_counter() - t
        print("documents:", {d["filename"]: d["processing_status"] for d in docs})
        idx = c.get(f"{API}/vector/index").json()
        ov = c.get(f"{API}/graph/overview").json()
        print(
            f"vectors indexed: {idx['vector_count']} ({idx['model_id']}, {idx['dimension']} dims)"
        )
        print(f"graph: {ov['state']['node_count']} nodes, {ov['state']['edge_count']} edges")
        print(f"ingestion took {ing:.1f} s (includes the one-off model load)")

        def views(query: str, **graph_opts) -> None:
            v = post("/search/vector", query=query, top_k=3).json()
            g = post("/search/graph", query=query, limit=3, graph=graph_opts).json()
            h = post("/search/hybrid", query=query, limit=4, graph=graph_opts).json()
            print(f"\nQ: {query!r}  graph options: {graph_opts or 'defaults'}")
            print(
                f"  entities linked: {[e['name'] for e in g['linked_entities']] or 'none'}"
                f"   graph status: {g['graph']['status']}"
            )
            print(
                "  vector only :",
                [(r["source"]["filename"], r["similarity"]) for r in v["results"]],
            )
            print(
                "  graph only  :",
                [
                    (
                        r["source"]["filename"],
                        f"rank {r['rank']} ({r['graph']['features']['decided_by']})",
                    )
                    for r in g["results"]
                ],
            )
            print(
                f"  HYBRID      : mode={h['path_summary']['mode']} "
                f"contributing={h['path_summary']['contributing']}"
            )
            for r in h["results"]:
                vec = (
                    f"vec#{r['vector']['rank']} sim {r['vector']['similarity']:.3f}"
                    if r["vector"]
                    else "vec -"
                )
                gr = f"graph#{r['graph']['rank']}" if r["graph"] else "graph -"
                print(
                    f"     #{r['rank']} fused {r['fused_score']:.6f} {r['source']['filename']:<22}"
                    f" {vec:<22} {gr}"
                )
            for w in h["warnings"]:
                print("     warning:", w)

        banner("2. THE SAME QUERY THROUGH THE THREE ENDPOINTS")
        views("Who works at Acme Inc?")
        views("How do I make bread?")
        views("Which technologies does FastAPI use?")
        views("How are Alice and Acme Inc related?")

        banner("3. DIRECTION: stored relationships are never reversed")
        for direction in ("outgoing", "incoming", "both"):
            body = post("/search/graph", query="Acme Inc", graph={"direction": direction}).json()
            seen = sorted(
                {
                    (
                        e["source"]["name"],
                        e["relationship_type"],
                        e["target"]["name"],
                        e["traversal"],
                    )
                    for r in body["results"]
                    for e in r["graph"]["evidence"]
                }
            )
            print(f"  direction={direction:<8} -> {seen}")

        banner("4. EXPLANATION of one fused result (explain=true)")
        h = post("/search/hybrid", query="Who works at Acme Inc?", limit=1, explain=True).json()
        top = h["results"][0]
        print("  fusion:", top["explanation"]["summary"])
        print("  graph :", top["explanation"]["graph_ranking"])
        for ev in top["graph"]["evidence"]:
            print(
                f"  evidence: {ev['source']['name']} -{ev['relationship_type']}-> "
                f"{ev['target']['name']}  (reached {ev['traversal']} from "
                f"{ev['reached_from']['name']}, distance {ev['distance']}, "
                f"provenance {ev['provenance']}): {short(ev['supporting_text'], 40)!r}"
            )

        banner("5. EXPLICIT entity_ids and the no-entity case")
        acme = post("/search/graph", query="Acme Inc").json()["linked_entities"][0]["entity_id"]
        g = post(
            "/search/graph",
            query="tell me more",
            graph={"entity_ids": [acme], "link_entities": False},
        ).json()
        print(
            f"  entity_ids=[Acme Inc], link_entities=false -> {len(g['results'])} chunks,"
            f" linked_by={g['linked_entities'][0]['resolved_by']}"
        )
        h = post("/search/hybrid", query="completely unrelated gardening question").json()
        print(
            f"  no graph entity -> graph status {h['paths']['graph']['status']!r},"
            f" contributing {h['path_summary']['contributing']}"
        )

        banner("6. FAILURE MODES (injected the way the tests inject them)")
        original = (app.state.graph.snapshot, app.state.vectors.ensure_fresh)

        def graph_down():
            raise GraphUnavailable("database unavailable")

        def vector_down():
            raise VectorIndexBuilding("the vector index is being (re)built")

        def show(label: str) -> None:
            r = c.post(f"{API}/search/hybrid", json={"query": "Who works at Acme Inc?", "limit": 3})
            body = r.json()
            if r.status_code == 200:
                s = body["path_summary"]
                print(
                    f"  {label:<34} HTTP 200 mode={s['mode']:<8} complete={s['complete']!s:<5}"
                    f" chunks={[x['source']['filename'] for x in body['results']]}"
                )
                print(f"     {body['warnings'][0] if body['warnings'] else '(no warnings)'}")
            else:
                e = body["error"]
                print(
                    f"  {label:<34} HTTP {r.status_code} {e['code']}: "
                    f"{ {k: v['error_code'] for k, v in e['details'].items()} }"
                )

        show("both paths healthy")
        app.state.graph.snapshot = graph_down
        show("graph path down")
        app.state.graph.snapshot = original[0]
        app.state.vectors.ensure_fresh = vector_down
        show("vector path down")
        app.state.graph.snapshot = graph_down
        show("both paths down")
        app.state.graph.snapshot, app.state.vectors.ensure_fresh = original
        show("recovered")

        release = threading.Event()
        real_search = app.state.graph_retriever.search

        def hang(*a, **k):
            release.wait(10)
            return real_search(*a, **k)

        app.state.graph_retriever.search = hang
        t = time.perf_counter()
        show("graph path hangs (2 s budget)")
        print(f"     answered in {time.perf_counter() - t:.1f} s instead of waiting for the hang")
        release.set()
        app.state.graph_retriever.search = real_search
        time.sleep(0.3)

        banner("7. DELETION: a deleted document leaves both paths")
        before = post("/search/hybrid", query="Who works at Acme Inc?", limit=10).json()
        r = c.delete(f"{API}/documents/{ids['team_notes.txt']}")
        after = post("/search/hybrid", query="Who works at Acme Inc?", limit=10).json()
        print(f"  DELETE team_notes.txt -> HTTP {r.status_code}")
        print("  before:", [x["source"]["filename"] for x in before["results"]])
        print("  after :", [x["source"]["filename"] for x in after["results"]])
        print(f"  mode {after['path_summary']['mode']}, warnings {after['warnings']}")

        banner("8. LATENCY of this run (service-reported, one machine, not a benchmark)")
        totals = [
            post("/search/hybrid", query="Who works at Acme Inc?").json()["timings_ms"]
            for _ in range(20)
        ]
        for key in ("graph", "vector", "fusion", "verify", "total"):
            vals = sorted(t[key] for t in totals)
            print(f"  {key:<7} median {vals[len(vals) // 2]:7.2f} ms   max {vals[-1]:7.2f} ms")
    app.state.engine.dispose()


if __name__ == "__main__":
    main()
