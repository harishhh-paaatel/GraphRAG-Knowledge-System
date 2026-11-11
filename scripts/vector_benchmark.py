r"""Measure embedding, indexing and search performance. Everything reported is measured.

Usage (from the repo root, ideally with nothing else running):
    backend\.venv\Scripts\python scripts\vector_benchmark.py
    backend\.venv\Scripts\python scripts\vector_benchmark.py --chunks 500 --sizes 10000   # quick

The corpus is SYNTHETIC and seeded (same seed -> same texts): it exists to measure speed and
memory with realistic chunk lengths, not retrieval quality. The "scaling" section uses random
unit vectors and says so: it measures FAISS latency/memory only. Results are written as JSON.
"""

import argparse
import ctypes
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.core.database import make_engine, make_session_factory  # noqa: E402
from app.models import Document, DocumentChunk  # noqa: E402
from app.services.embeddings.embedder import SentenceTransformerEmbedder  # noqa: E402
from app.services.embeddings.revision import bump_vector_revision  # noqa: E402
from app.services.embeddings.service import VectorService  # noqa: E402
from app.services.embeddings.vector_index import FaissFlatIPIndex, IndexEntry  # noqa: E402
from app.services.embeddings.vector_store import load_index, save_index  # noqa: E402
from app.services.retrieval.vector_search import search_vectors  # noqa: E402

SUBJECTS = [
    "The ingestion service",
    "A vector index",
    "The knowledge graph",
    "Each worker thread",
    "The embedding model",
    "A relational database",
    "The API layer",
    "Every document chunk",
    "The scheduler",
    "A cache layer",
    "The retrieval engine",
    "A background job",
    "The parser",
    "The entity resolver",
    "A search request",
    "The storage layer",
    "The chunking routine",
    "A migration script",
    "The configuration loader",
    "The health check",
    "A query planner",
    "The metrics collector",
    "An evaluation run",
    "The persistence layer",
    "A text normalizer",
]
VERBS = [
    "stores",
    "validates",
    "rebuilds",
    "tracks",
    "normalizes",
    "ranks",
    "persists",
    "retrieves",
    "indexes",
    "compresses",
    "filters",
    "merges",
    "records",
    "schedules",
    "serializes",
    "verifies",
    "replaces",
    "summarizes",
    "deduplicates",
    "synchronizes",
]
OBJECTS = [
    "entity relationships",
    "source documents",
    "similarity scores",
    "text embeddings",
    "page numbers",
    "configuration files",
    "query results",
    "processing jobs",
    "metadata records",
    "retrieval candidates",
    "chunk identifiers",
    "document checksums",
    "graph edges",
    "model fingerprints",
    "error messages",
    "section headings",
    "evidence snippets",
    "revision counters",
    "cache entries",
    "audit events",
]
QUALIFIERS = [
    "during ingestion",
    "after every update",
    "for each request",
    "when the model changes",
    "before the file is saved",
    "in a background thread",
    "using exact search",
    "across all documents",
    "without blocking readers",
    "inside one transaction",
    "whenever a document is deleted",
    "while the service is starting",
    "on a fixed schedule",
]


def make_corpus(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)

    def pick(items):
        return items[int(rng.integers(len(items)))]

    chunks = []
    while len(chunks) < n:
        sentences = [
            f"{pick(SUBJECTS)} {pick(VERBS)} {pick(OBJECTS)} {pick(QUALIFIERS)}."
            for _ in range(int(rng.integers(5, 10)))
        ]
        chunks.append(" ".join(sentences))
    return chunks


def make_queries(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed + 1)
    return [
        f"How does {SUBJECTS[int(rng.integers(len(SUBJECTS)))].lower()} handle "
        f"{OBJECTS[int(rng.integers(len(OBJECTS)))]}?"
        for _ in range(n)
    ]


def rss_mb() -> float:
    """Resident memory of this process in MB (Windows via psapi, POSIX via resource).

    Raises instead of returning 0 if the OS call fails: a silent zero is not a measurement.
    """
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        c = Counters()
        c.cb = ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        if c.WorkingSetSize == 0:
            raise OSError("process working set reported as 0 bytes")
        return c.WorkingSetSize / 1e6
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def summarize_ms(samples_s: list[float]) -> dict:
    ms = [s * 1000 for s in samples_s]
    return {
        "n": len(ms),
        "median_ms": round(statistics.median(ms), 4),
        "p95_ms": round(pct(ms, 95), 4),
        "p99_ms": round(pct(ms, 99), 4),
        "mean_ms": round(statistics.fmean(ms), 4),
    }


def time_each(fn, items) -> list[float]:
    out = []
    for item in items:
        t = time.perf_counter()
        fn(item)
        out.append(time.perf_counter() - t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunks", type=int, default=2000)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--sizes", type=int, nargs="*", default=[10_000, 50_000, 100_000])
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation" / "reports" / "phase5_vector_benchmark.json",
    )
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    import faiss
    import torch

    result: dict = {
        "note": "Synthetic seeded corpus: measures speed/memory, NOT retrieval quality.",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpus": __import__("os").cpu_count(),
            "torch": torch.__version__,
            "faiss": faiss.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "faiss_threads": faiss.omp_get_max_threads(),
        },
        "params": {"chunks": args.chunks, "queries": args.queries, "seed": args.seed},
    }
    env = result["environment"]
    print(f"Environment: {env}")
    rss0 = rss_mb()

    corpus = make_corpus(args.chunks, args.seed)
    lengths = [len(c) for c in corpus]
    result["corpus"] = {
        "chunks": len(corpus),
        "mean_chars": round(statistics.fmean(lengths), 1),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "mean_words": round(statistics.fmean(len(c.split()) for c in corpus), 1),
        "distinct_texts": len(set(corpus)),
    }
    print(f"Corpus: {result['corpus']}")

    with tempfile.TemporaryDirectory(prefix="graphrag-bench-", ignore_cleanup_errors=True) as tmp:
        settings = Settings(
            data_dir=Path(tmp),
            embedding_provider="sentence_transformers",
            embedding_cache_dir=ROOT / "data" / "models",
            _env_file=None,
        )
        url = settings.resolved_database_url
        cfg = Config(str(BACKEND / "alembic.ini"))
        cfg.set_main_option("script_location", str(BACKEND / "migrations"))
        cfg.attributes["url"] = url
        command.upgrade(cfg, "head")
        engine = make_engine(url)
        factory = make_session_factory(engine)

        # ---- model load
        embedder = SentenceTransformerEmbedder(
            settings.embedding_model,
            revision=settings.embedding_model_revision,
            cache_dir=settings.model_cache_dir,
            batch_size=32,
        )
        t = time.perf_counter()
        ident = embedder.identity()
        load_s = time.perf_counter() - t
        rss_model = rss_mb()
        result["model"] = {
            "id": ident.model_id,
            "revision": ident.revision,
            "dimension": ident.dimension,
            "device": "cpu",
            "load_seconds": round(load_s, 2),
            "rss_mb_before_load": round(rss0, 1),
            "rss_mb_after_load": round(rss_model, 1),
            "rss_mb_model_delta": round(rss_model - rss0, 1),
        }
        print(f"Model: {result['model']}")

        # ---- embedding throughput vs batch size (first 500 chunks, median of 3 runs, 1 warm-up)
        sample = corpus[: min(500, len(corpus))]
        model = embedder._model
        model.encode(sample[:64], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        batch_rows = []
        for bs in (1, 8, 32, 64, 128):
            runs = []
            for _ in range(3):
                t = time.perf_counter()
                model.encode(
                    sample, batch_size=bs, normalize_embeddings=True, show_progress_bar=False
                )
                runs.append(time.perf_counter() - t)
            med = statistics.median(runs)
            batch_rows.append(
                {
                    "batch_size": bs,
                    "texts": len(sample),
                    "median_seconds": round(med, 3),
                    "texts_per_second": round(len(sample) / med, 1),
                    "ms_per_text": round(med * 1000 / len(sample), 2),
                    "runs_seconds": [round(r, 3) for r in runs],
                }
            )
            print(
                f"  embed batch_size={bs:>3}: {len(sample) / med:7.1f} texts/s  ({med * 1000 / len(sample):.2f} ms/text)"
            )
        result["embedding_throughput_by_batch_size"] = batch_rows

        # ---- real system: insert chunks, full VectorService.sync (DB read + embed + index + persist)
        with factory() as session:
            per_doc = 10
            for d in range(0, len(corpus), per_doc):
                doc = Document(
                    id=f"b{d // per_doc:05d}",
                    filename=f"bench{d // per_doc}.txt",
                    stored_path="x",
                    file_type="txt",
                    size_bytes=1,
                    checksum=f"{d // per_doc + 1:064d}",
                    processing_status="completed",
                )
                session.add(doc)
                session.flush()
                for i, text in enumerate(corpus[d : d + per_doc]):
                    session.add(
                        DocumentChunk(
                            id=f"{doc.id}-c{i}",
                            document_id=doc.id,
                            chunk_index=i,
                            content=text,
                            chunk_metadata={},
                        )
                    )
            bump_vector_revision(session)
            session.commit()
        vectors = VectorService(
            factory, settings.vector_index_file, embedder, sync_on_startup=False
        )
        t = time.perf_counter()
        report = vectors.sync()
        total_s = time.perf_counter() - t
        rss_index = rss_mb()
        index_file = settings.vector_index_file
        result["indexing"] = {
            "chunks": report.embedded,
            "total_seconds": round(total_s, 3),
            "embedding_seconds": round(report.embed_ms / 1000, 3),
            "embedding_texts_per_second": round(report.embedded / (report.embed_ms / 1000), 1),
            "non_embedding_seconds": round(total_s - report.embed_ms / 1000, 3),
            "index_file_bytes": index_file.stat().st_size,
            "raw_vector_bytes": report.embedded * ident.dimension * 4,
            "rss_mb_after_index": round(rss_index, 1),
            "rss_mb_index_delta": round(rss_index - rss_model, 1),
        }
        print(f"Indexing: {result['indexing']}")

        # ---- persistence
        index, meta = load_index(index_file)
        save_runs, load_runs = [], []
        for _ in range(5):
            t = time.perf_counter()
            save_index(index, meta, index_file)
            save_runs.append(time.perf_counter() - t)
            t = time.perf_counter()
            load_index(index_file)
            load_runs.append(time.perf_counter() - t)
        result["persistence"] = {
            "save_median_ms": round(statistics.median(save_runs) * 1000, 2),
            "load_median_ms": round(statistics.median(load_runs) * 1000, 2),
            "file_bytes": index_file.stat().st_size,
            "runs": 5,
        }
        print(f"Persistence: {result['persistence']}")

        # ---- query embedding + FAISS search + end-to-end
        queries = make_queries(args.queries, args.seed)
        embed_q = time_each(lambda q: embedder.embed([q]), queries)
        result["query_embedding"] = summarize_ms(embed_q)
        qvecs = [embedder.embed([q])[0] for q in queries]
        faiss_rows, e2e_rows = [], []
        for k in (1, 5, 10, 50, 100):
            samples = []
            for _ in range(3):
                samples += time_each(lambda v, k=k: index.search(v, k), qvecs)
            faiss_rows.append({"top_k": k, **summarize_ms(samples)})
        for k in (1, 5, 10, 50):
            e2e = time_each(lambda q, k=k: search_vectors(vectors, factory, q, top_k=k), queries)
            e2e_rows.append({"top_k": k, **summarize_ms(e2e)})
        doc_filter = {f"b{i:05d}" for i in range(0, max(1, len(corpus) // per_doc // 10))}
        filt = []
        for _ in range(3):
            filt += time_each(lambda v: index.search(v, 10, document_ids=doc_filter), qvecs)
        result["faiss_search_by_top_k"] = faiss_rows
        result["faiss_search_with_document_filter"] = {
            "top_k": 10,
            "allowed_documents": len(doc_filter),
            **summarize_ms(filt),
        }
        result["end_to_end_search_by_top_k"] = e2e_rows
        for row in faiss_rows:
            print(
                f"  FAISS-only top_k={row['top_k']:>3}: median {row['median_ms']:.3f} ms  p95 {row['p95_ms']:.3f} ms"
            )
        for row in e2e_rows:
            print(
                f"  end-to-end top_k={row['top_k']:>3}: median {row['median_ms']:.2f} ms  p95 {row['p95_ms']:.2f} ms  (includes query embedding + SQLite lookup)"
            )
        print(f"  query embedding alone: {result['query_embedding']}")
        engine.dispose()

    # ---- scaling with synthetic random vectors (latency / memory only)
    rng = np.random.default_rng(args.seed)
    d = ident.dimension
    scaling = []
    for n in args.sizes:
        rss_before = rss_mb()
        vecs = rng.normal(size=(n, d)).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        idx = FaissFlatIPIndex(d)
        t = time.perf_counter()
        idx.add([IndexEntry(f"c{i}", f"d{i % 1000}", "h") for i in range(n)], vecs)
        add_s = time.perf_counter() - t
        t = time.perf_counter()
        payload = idx.serialize()
        ser_s = time.perf_counter() - t
        q = vecs[rng.integers(n, size=100)]
        row = {
            "vectors": n,
            "dimension": d,
            "add_seconds": round(add_s, 3),
            "serialize_seconds": round(ser_s, 3),
            "serialized_bytes": len(payload),
            "rss_mb_delta": round(rss_mb() - rss_before, 1),
            "kind": "random unit vectors",
        }
        for k in (10, 50):
            samples = time_each(lambda v, k=k, idx=idx: idx.search(v, k), list(q) * 3)
            row[f"search_top{k}"] = summarize_ms(samples)
        scaling.append(row)
        print(
            f"  scaling {n:>7} vectors: add {add_s:.2f}s  search top10 median {row['search_top10']['median_ms']:.2f} ms "
            f"p95 {row['search_top10']['p95_ms']:.2f} ms  serialized {len(payload) / 1e6:.0f} MB"
        )
        del vecs, idx, payload
    result["scaling_random_vectors"] = scaling

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
