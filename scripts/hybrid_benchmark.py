r"""Hybrid retrieval benchmark: latency and SQL statements per request, measured.

Usage (from the repo root):
    backend\.venv\Scripts\python scripts\hybrid_benchmark.py                  # hashing stub, fast
    backend\.venv\Scripts\python scripts\hybrid_benchmark.py --provider sentence_transformers
    backend\.venv\Scripts\python scripts\hybrid_benchmark.py --out evaluation\reports\x.json

Everything runs in a throwaway data directory with the real migrations' schema, the real graph
service, the real FAISS index and the real API (in-process TestClient). The corpus is synthetic
("PersonN works at OrgM. OrgM is located in CityK."), built through the real knowledge-persistence
code. These are LATENCY and QUERY-COUNT measurements on one machine, not retrieval-quality
measurements, and a single run is preliminary.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

parser = argparse.ArgumentParser()
parser.add_argument("--provider", default="hashing", choices=["hashing", "sentence_transformers"])
parser.add_argument("--sizes", default="50,200,800", help="documents per corpus")
parser.add_argument("--requests", type=int, default=60)
parser.add_argument("--out", default=None)
args = parser.parse_args()
os.environ["GRAPHRAG_EMBEDDING_PROVIDER"] = args.provider
if args.provider == "sentence_transformers":
    os.environ["GRAPHRAG_EMBEDDING_CACHE_DIR"] = str(ROOT / "data" / "models")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import event  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Base, Document, DocumentChunk  # noqa: E402
from app.services.embeddings.revision import bump_vector_revision  # noqa: E402
from app.services.extraction.persistence import ChunkResult, replace_document_knowledge  # noqa: E402
from app.services.extraction.types import (  # noqa: E402
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)

TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY", "PRODUCT", "CONCEPT", "EVENT"}
T0 = datetime(2026, 1, 1, 12, 0, 0)


def populate(app, documents: int) -> None:
    orgs, cities = max(documents // 10, 3), 10
    with app.state.session_factory() as session:
        for i in range(documents):
            org, city = i % orgs, (i % orgs) % cities
            text = f"Person{i} works at Org{org}. Org{org} is located in City{city}."
            doc = Document(
                filename=f"doc{i}.txt",
                stored_path="x",
                file_type="txt",
                size_bytes=len(text),
                checksum=f"{i + 1:064d}",
                processing_status="completed",
                upload_timestamp=T0 + timedelta(seconds=i),
            )
            session.add(doc)
            session.flush()
            chunk = DocumentChunk(id=f"chunk-{i}", document_id=doc.id, chunk_index=0, content=text)
            session.add(chunk)
            extraction = ChunkExtraction(
                [
                    ExtractedEntity(f"Person{i}", "PERSON", surface_form=f"Person{i}"),
                    ExtractedEntity(f"Org{org}", "ORGANIZATION", surface_form=f"Org{org}"),
                    ExtractedEntity(f"City{city}", "LOCATION", surface_form=f"City{city}"),
                ],
                [
                    ExtractedRelationship(
                        f"Person{i}", f"Org{org}", "WORKS_AT", f"Person{i} works at Org{org}"
                    ),
                    ExtractedRelationship(
                        f"Org{org}",
                        f"City{city}",
                        "LOCATED_IN",
                        f"Org{org} is located in City{city}",
                    ),
                ],
            )
            session.flush()
            replace_document_knowledge(
                session,
                doc.id,
                [ChunkResult(chunk.id, text, extraction)],
                allowed_types=TYPES,
                fuzzy_threshold=0.88,
                provider_name="benchmark",
            )
            bump_vector_revision(session)
            session.commit()


class SqlCounter:
    """Counts statements the engine executes while armed."""

    def __init__(self, engine) -> None:
        self.count = 0
        self.statements: list[str] = []
        self._armed = False
        self._lock = threading.Lock()
        event.listen(engine, "before_cursor_execute", self._on)

    def _on(self, conn, cursor, statement, params, context, executemany) -> None:  # noqa: ANN001
        if self._armed:
            with self._lock:
                self.count += 1
                self.statements.append(statement.split("\n")[0][:90])

    def arm(self) -> None:
        self.count, self.statements, self._armed = 0, [], True

    def disarm(self) -> int:
        self._armed = False
        return self.count


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(p * len(ordered)))], 3)


def summarize(ms: list[float]) -> dict:
    return {
        "n": len(ms),
        "median_ms": round(statistics.median(ms), 3),
        "p95_ms": pct(ms, 0.95),
        "max_ms": round(max(ms), 3),
    }


def bench_size(documents: int, requests: int) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="graphrag-hb-", dir="E:/graphrag-cache/tmp"))
    app = create_app(Settings(data_dir=tmp, environment="test", _env_file=None))
    Base.metadata.create_all(app.state.engine)
    counter = SqlCounter(app.state.engine)
    result: dict = {"documents": documents}
    with TestClient(app) as client:
        t = time.perf_counter()
        populate(app, documents)
        result["populate_s"] = round(time.perf_counter() - t, 2)
        orgs = max(documents // 10, 3)
        queries = [f"Who works at Org{i % orgs}?" for i in range(requests)]

        # first request pays for the index sync and the alias index; measure it separately
        t = time.perf_counter()
        r = client.post("/api/v1/search/hybrid", json={"query": queries[0]})
        assert r.status_code == 200, r.text
        result["first_request_ms"] = round((time.perf_counter() - t) * 1000, 1)
        result["vectors_indexed"] = client.get("/api/v1/vector/index").json()["vector_count"]
        overview = client.get("/api/v1/graph/overview").json()
        result["corpus_graph"] = {
            k: overview[k] for k in ("node_count", "edge_count") if k in overview
        } or list(overview)[:6]

        endpoints = {
            "vector": ("/api/v1/search/vector", lambda q: {"query": q, "top_k": 20}),
            "graph": ("/api/v1/search/graph", lambda q: {"query": q}),
            "hybrid": ("/api/v1/search/hybrid", lambda q: {"query": q}),
        }
        for name, (url, body) in endpoints.items():
            times, counts = [], []
            for q in queries:
                counter.arm()
                t = time.perf_counter()
                resp = client.post(url, json=body(q))
                times.append((time.perf_counter() - t) * 1000)
                counts.append(counter.disarm())
                assert resp.status_code == 200, resp.text
            result[name] = {
                **summarize(times),
                "sql_statements_per_request": statistics.median(counts),
            }
            if name == "hybrid":
                counter.arm()
                client.post(url, json=body(queries[1]))
                counter.disarm()
                result["hybrid_statements"] = list(counter.statements)

        # breakdown reported by the service itself
        breakdown: dict[str, list[float]] = {}
        for q in queries[:30]:
            timings = client.post("/api/v1/search/hybrid", json={"query": q}).json()["timings_ms"]
            for key, value in timings.items():
                breakdown.setdefault(key, []).append(value)
        result["hybrid_breakdown_median_ms"] = {
            k: round(statistics.median(v), 3) for k, v in breakdown.items()
        }

        # redundant work: how often were the derived structures rebuilt?
        result["alias_index_builds"] = app.state.graph_retriever.linker.builds
        embedder = app.state.vectors.embedder
        if hasattr(embedder, "texts_embedded"):
            result["texts_embedded_total"] = embedder.texts_embedded

        # concurrency: 4 workers, same endpoint
        def one(q: str) -> tuple[int, float]:
            t = time.perf_counter()
            resp = client.post("/api/v1/search/hybrid", json={"query": q})
            return resp.status_code, (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(one, queries))
        wall = time.perf_counter() - t
        result["concurrent_hybrid_4_workers"] = {
            "requests": len(outcomes),
            "all_200": all(code == 200 for code, _ in outcomes),
            "wall_s": round(wall, 3),
            "requests_per_s": round(len(outcomes) / wall, 1),
            **summarize([ms for _, ms in outcomes]),
        }
    app.state.engine.dispose()
    return result


def main() -> None:
    report = {
        "provider": args.provider,
        "note": "preliminary single run on one machine; latency and query counts only",
        "python": sys.version.split()[0],
        "sizes": [],
    }
    for size in [int(s) for s in args.sizes.split(",")]:
        print(f"-- corpus of {size} documents", flush=True)
        entry = bench_size(size, args.requests)
        report["sizes"].append(entry)
        for name in ("vector", "graph", "hybrid"):
            e = entry[name]
            print(
                f"  {name:<7} median {e['median_ms']:>8.2f} ms  p95 {e['p95_ms']:>8.2f} ms  "
                f"SQL/request {e['sql_statements_per_request']:g}"
            )
        print(
            f"  first request {entry['first_request_ms']} ms; breakdown {entry['hybrid_breakdown_median_ms']}"
        )
        c = entry["concurrent_hybrid_4_workers"]
        print(
            f"  4 workers: {c['requests_per_s']} req/s, median {c['median_ms']} ms, all_200={c['all_200']}"
        )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
